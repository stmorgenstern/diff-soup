"""Nonlinear-solver contract and adjoint compatibility exports.

Defines the role-object surface over diff_soup's fused Newton corrector —
without rewriting the hot kernel. Wraps (does not rewrite) the fused primal
in [diff_soup.nonlinear.solver][] / [diff_soup.nonlinear.box_newton][].

Concrete solvers: [Newton][diff_soup.nonlinear._api.Newton]
(unconstrained), [BoxConstrainedActiveSet][diff_soup.nonlinear._api.BoxConstrainedActiveSet]
(reduced-space IFT — exact-zero gradient at active bounds is contractual,
verified by the box-golden tests). Canonical adjoints live in
[`diff_soup._adjoints`][diff_soup._adjoints]; this module re-exports
[ImplicitAdjoint][diff_soup.nonlinear._api.ImplicitAdjoint] (IFT reverse,
default), [ForwardAdjoint][diff_soup.nonlinear._api.ForwardAdjoint]. All norms
route through
``utils._safe_math``: the grad of WRMS at an exact-zero update is finite.
``NewtonInfo`` and ``NonlinearConfig`` are the real records (re-exported from
[diff_soup.nonlinear.solver][]).

There is no ``postprocess`` protocol method on
[NonlinearSolver][diff_soup.nonlinear._api.NonlinearSolver].
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from diff_soup._adjoints import AbstractAdjoint, ForwardAdjoint, ImplicitAdjoint
from diff_soup._solution import RESULT
from diff_soup.linear._api import LinearSolver
from diff_soup.nonlinear._helpers import benson_munson_bind
from diff_soup.nonlinear._kkt import kkt_solve
from diff_soup.nonlinear.box_bounds import BoxBounds, resolve_box_arrays
from diff_soup.nonlinear.box_newton import (
    BoxConstrainedUnifiedConfig,
    box_constrained_solve_jvp,
    box_constrained_solve_vjp,
)
from diff_soup.nonlinear.solver import (
    NewtonInfo,
    NonlinearConfig,
    _factor_state_block,
    _kkt_newton_step,
    _wrms_norm,
    _wrms_norm_per_block,
    nonlinear_solve_primal,
)

__all__ = [
    "NewtonInfo",
    "NonlinearConfig",
    "NonlinearState",
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "ForwardAdjoint",
    "NonlinearSolver",
    "Newton",
    "BoxConstrainedActiveSet",
]


# ---------------------------------------------------------------------------
# NonlinearState — array-only threaded Newton state (driveable view)
# ---------------------------------------------------------------------------
class NonlinearState(eqx.Module):
    """Array-only Newton state threaded ``init -> step``.

    A first-class pytree mirror of the closure ``Carry`` that
    [nonlinear_solve_primal][diff_soup.nonlinear.solver.nonlinear_solve_primal]
    carries
    internally. Surfacing it as an ``eqx.Module`` makes the fused corrector
    externally step-driveable and warm-restartable; the field set is a 1:1 image
    of the kernel ``Carry``.

    Attributes:
        it: Newton iteration counter (int32).
        u: Differential-block iterate.
        lam: Global-algebraic (Schur) iterate; shape ``(0,)`` when absent.
        err: WRMS norm of the last Newton update.
        first_nrm: WRMS norm at the first iteration (rate denominator).
        rate: Geometric-mean convergence-rate estimate.
        tooslow: Rate-guard latch (stall / projected failure).
        done: Convergence achieved.
        residual_norm: Max-abs ``|[R; g]|`` from the last step.
    """

    it: Array
    u: Array
    lam: Array
    err: Array
    first_nrm: Array
    rate: Array
    tooslow: Array
    done: Array
    residual_norm: Array


def _wrms_converged(
    nrm: Array, rate: Array, first_iter: Array, cfg: NonlinearConfig
) -> tuple[Array, Array]:
    """IDA-style WRMS convergence + rate-guard test (kernel-identical).

    Mirrors the convergence / ``tooslow`` predicates inside
    [nonlinear_solve_primal][diff_soup.nonlinear.solver.nonlinear_solve_primal]'s
    body so the decomposed ``step`` makes the same accept/diverge decision.

    Args:
        nrm: WRMS norm of the current Newton update.
        rate: Geometric-mean convergence-rate estimate.
        first_iter: Whether this is the first Newton iteration.
        cfg: The frozen
        [NonlinearConfig][diff_soup.nonlinear.solver.NonlinearConfig].

    Returns:
        Tuple ``(converged, tooslow)`` of boolean scalars.
    """
    minnrm = jnp.asarray(100.0, dtype=nrm.dtype) * jnp.finfo(nrm.dtype).eps
    safety = jnp.asarray(0.1, dtype=nrm.dtype)
    s_est = jnp.where(
        first_iter,
        jnp.asarray(20.0, dtype=nrm.dtype),
        rate / jnp.maximum(1.0 - rate, jnp.asarray(1e-12, dtype=nrm.dtype)),
    )
    converged = jnp.logical_or(nrm <= minnrm, s_est * nrm <= safety)
    tooslow = jnp.logical_and(
        cfg.newton_rate_guard,
        jnp.logical_and(jnp.logical_not(converged), rate > jnp.asarray(0.9, dtype=nrm.dtype)),
    )
    return converged, tooslow


# ---------------------------------------------------------------------------
# NonlinearSolver — the L3 nonlinear-layer solver
# ---------------------------------------------------------------------------
class NonlinearSolver(eqx.Module):
    """Base class of the Newton correctors for a bordered system `R(u, lam) = 0`, `g(u, lam) = 0`.

    `solve` runs the whole iteration, and `init`, `step`, and `terminate` drive it one iteration at a time. A solve never raises on non-convergence, so read the returned status.

    Attributes:
        config: Iteration settings, a `NonlinearConfig`.
        linear: Linear solver stored on the corrector and not read by its methods, which take `backend`.
        adjoint: Differentiation strategy, used only by `BoxConstrainedActiveSet`.
    """

    config: NonlinearConfig = eqx.field(static=True, default_factory=NonlinearConfig)
    linear: Optional[LinearSolver] = None
    adjoint: AbstractAdjoint = eqx.field(default_factory=ImplicitAdjoint)

    # ------------------------------------------------------------------
    # The public solve — wrap, don't rewrite (XLA-identical to the kernel)
    # ------------------------------------------------------------------
    def solve(
        self,
        operator: Any,
        backend: LinearSolver,
        analysis: Any,
        u0: Array,
        lam0: Array,
        theta: Optional[Array],
        ctx: Any,
        *,
        frozen_blocks: Optional[tuple] = None,
        factor_refit: bool = True,
        mass_refresh_fn: Optional[Any] = None,
        early_exit: bool = False,
    ) -> tuple[Array, Array, NewtonInfo]:
        """Run Newton's method from `(u0, lam0)` and return the last iterate with its diagnostics.

        Args:
            operator: Object with `residual(u, lam, theta, ctx)` returning `(R, g)` and `jacobian_blocks(u, lam, theta, ctx)`.
            backend: Linear solver for the state block.
            analysis: Analysis of the state block's sparsity, from `diff_soup.linear.analyze`.
            u0: Initial state.
            lam0: Initial global unknowns.
            theta: Parameters passed to `operator`.
            ctx: Context passed to `operator`.
            frozen_blocks: Jacobian blocks to factor in place of evaluating them at `u0`, used when `config.jacobian_cache` is `"reuse"`.
            factor_refit: Accepted and ignored.
            mass_refresh_fn: Function of `(u, lam)` returning `ctx.mass_diag` for each iterate.
            early_exit: If `True`, stop at convergence with a loop that is not reverse-differentiable.

        Returns:
            The final `u`, the final `lam`, and a `NewtonInfo`.

        Raises:
            TypeError: If `early_exit` is set while `u0` is traced and `config.safe_mode` is off.
        """
        return nonlinear_solve_primal(
            operator,
            backend,
            analysis,
            u0,
            lam0,
            theta,
            ctx,
            self.config,
            frozen_blocks=frozen_blocks,
            factor_refit=factor_refit,
            mass_refresh_fn=mass_refresh_fn,
            early_exit=early_exit,
        )

    # ------------------------------------------------------------------
    # Decomposed init / step / terminate (externally driveable view)
    # ------------------------------------------------------------------
    def init(
        self,
        operator: Any,
        u0: Array,
        lam0: Array,
        theta: Optional[Array],
        ctx: Any,
    ) -> NonlinearState:
        """Return the iteration state at `(u0, lam0)` before the first step.

        Args:
            operator: Unused by the base class.
            u0: Initial state.
            lam0: Initial global unknowns.
            theta: Unused by the base class.
            ctx: Unused by the base class.

        Returns:
            A `NonlinearState` with the iteration count at zero.
        """
        del operator, theta, ctx
        inf = jnp.asarray(jnp.inf, dtype=u0.dtype)
        return NonlinearState(
            it=jnp.asarray(0, dtype=jnp.int32),
            u=u0,
            lam=lam0,
            err=inf,
            first_nrm=inf,
            rate=jnp.asarray(0.0, dtype=u0.dtype),
            tooslow=jnp.asarray(False),
            done=jnp.asarray(False),
            residual_norm=inf,
        )

    def step(
        self,
        operator: Any,
        backend: LinearSolver,
        analysis: Any,
        state: NonlinearState,
        theta: Optional[Array],
        ctx: Any,
    ) -> tuple[NonlinearState, dict]:
        """Take one Newton iteration with a freshly evaluated Jacobian.

        The caller stops on `terminate` and bounds the number of iterations.

        Args:
            operator: Residual and Jacobian provider, as in `solve`.
            backend: Linear solver for the state block.
            analysis: Analysis of the state block's sparsity.
            state: Current `NonlinearState`.
            theta: Parameters passed to `operator`.
            ctx: Context passed to `operator`.

        Returns:
            The next `NonlinearState` and a dict with `"update_norm"` and `"iteration"`.
        """
        cfg = self.config
        du, dlam, _err_pre, _blocks = _kkt_newton_step(
            operator, backend, analysis, state.u, state.lam, theta, ctx, cfg
        )
        u_next = state.u + du
        lam_next = state.lam + dlam
        if cfg.lam_atol is None and cfg.lam_rtol is None:
            delta = jnp.concatenate((du, dlam), axis=0)
            yref = jnp.concatenate((u_next, lam_next), axis=0)
            nrm = _wrms_norm(delta, cfg.atol, cfg.rtol, yref)
        else:
            nrm = _wrms_norm_per_block(
                du, dlam, u_next, lam_next, cfg.atol, cfg.rtol, cfg.lam_atol, cfg.lam_rtol
            )
        first_iter = state.it == 0
        first_nrm = jnp.where(first_iter, nrm, state.first_nrm)
        order = jnp.maximum(state.it, jnp.asarray(1, dtype=state.it.dtype))
        nrm_ratio = nrm / jnp.maximum(first_nrm, jnp.asarray(1e-30, dtype=nrm.dtype))
        rate_raw = nrm_ratio ** (1.0 / order.astype(nrm.dtype))
        rate = jnp.clip(jnp.where(first_iter, 0.0, rate_raw), 0.0, 0.999999)
        done, tooslow = _wrms_converged(nrm, rate, first_iter, cfg)
        next_state = NonlinearState(
            it=state.it + 1,
            u=u_next,
            lam=lam_next,
            err=nrm,
            first_nrm=first_nrm,
            rate=rate,
            tooslow=tooslow,
            done=done,
            residual_norm=_err_pre,
        )
        return next_state, {"update_norm": nrm, "iteration": next_state.it}

    def terminate(self, operator: Any, state: NonlinearState) -> tuple[Array, RESULT]:
        """Report whether to stop, and the status of the iteration so far.

        Args:
            operator: Unused by the base class.
            state: Current `NonlinearState`.

        Returns:
            A boolean stop flag and a `RESULT`, which is `SUCCESS` when converged, `NONLINEAR_DIVERGED` when convergence stalled, and `UNKNOWN` otherwise.
        """
        del operator
        stop = jnp.logical_or(state.done, state.tooslow)
        result = RESULT.where(
            state.done,
            RESULT.SUCCESS,
            RESULT.where(state.tooslow, RESULT.NONLINEAR_DIVERGED, RESULT.UNKNOWN),
        )
        return stop, result


class Newton(NonlinearSolver):
    """Newton's method for a bordered system `R(u, lam) = 0`, `g(u, lam) = 0`.

    It converges on a weighted RMS norm of the update, scaled by the observed contraction rate. A contraction rate above 0.9 stops it as stalled, which `terminate` reports as `NONLINEAR_DIVERGED`.
    """


class BoxConstrainedActiveSet(NonlinearSolver):
    """Newton's method with box bounds on selected state entries, by an active-set iteration.

    Entries held at a bound are pinned, and iterates are clipped into the box. It stops when the max-norm of the pinned residual falls to `config.atol`. Its derivative uses the forward implicit rule under `adjoint=ForwardAdjoint()` and the reverse rule otherwise.

    Attributes:
        config: Iteration and bound settings, with the uniform `lower_scalar`, `upper_scalar`, and `constrained_indices`.
        bounds: Per-entry `BoxBounds`, used only when `lower` or `upper` is a tuple, otherwise the bounds come from `config`.
    """

    config: BoxConstrainedUnifiedConfig = eqx.field(
        static=True, default_factory=BoxConstrainedUnifiedConfig
    )
    bounds: Optional[BoxBounds] = eqx.field(static=True, default=None)

    def solve(
        self,
        operator: Any,
        backend: LinearSolver,
        analysis: Any,
        u0: Array,
        lam0: Array,
        theta: Optional[Array],
        ctx: Any,
        *,
        frozen_blocks: Optional[tuple] = None,
        factor_refit: bool = True,
        mass_refresh_fn: Optional[Any] = None,
        early_exit: bool = False,
    ) -> tuple[Array, Array, Any]:
        """Run the box-constrained corrector; returns ``(u, lam, info)``.

        Dispatches to the AD-aware box primal selected by
        [adjoint][diff_soup.nonlinear._api.BoxConstrainedActiveSet.adjoint]
        (``_jvp`` for
        [ForwardAdjoint][diff_soup.nonlinear._api.ForwardAdjoint], else the
        reduced-space ``_vjp``). Per-DOF
        [bounds][diff_soup.nonlinear._api.BoxConstrainedActiveSet.bounds] are
        resolved and injected into ``ctx.bounds``; uniform scalar bounds ride
        [config][diff_soup.nonlinear._api.BoxConstrainedActiveSet.config] directly.

        Args:
            operator: The KKT-form residual + Jacobian-blocks operator.
            backend: Linear solver for the ``A``-block factor + solve.
            analysis: The analysis of the ``A`` block's pattern: a contract
                [Analysis][diff_soup.linear.Analysis] or the state a backend's
                ``analyze`` returns.
            u0: Initial guess for the differential part.
            lam0: Initial guess for the global algebraic part.
            theta: Optional parameter pytree passed to the residual.
            ctx: Step context (mass diagonal + bounds / warm-start channels).
            frozen_blocks: Ignored (the box primal relinearises each iterate).
            factor_refit: Ignored (the box primal always refits the pinned ``A``).
            mass_refresh_fn: Optional state-dependent mass-refresh callback.
            early_exit: Ignored (the box primal owns its own ``while_loop``).

        Returns:
            Tuple ``(u, lam, info)`` — the converged iterate and box counters.
        """
        del frozen_blocks, factor_refit, early_exit
        ctx_eff = self._inject_bounds(ctx, u0)
        solve_impl = (
            box_constrained_solve_jvp
            if isinstance(self.adjoint, ForwardAdjoint)
            else box_constrained_solve_vjp
        )
        return solve_impl(
            operator, backend, analysis, u0, lam0, theta, ctx_eff, self.config, mass_refresh_fn
        )

    def _inject_bounds(self, ctx: Any, u0: Array) -> Any:
        """Resolve per-DOF [bounds][diff_soup.nonlinear._api.BoxConstrainedActiveSet.bounds] into ``ctx.bounds`` (uniform = no-op).

        Args:
            ctx: The incoming step context.
            u0: Initial iterate (only its shape / dtype are used).

        Returns:
            The context with ``ctx.bounds`` populated for per-DOF tuple bounds,
            or unchanged for the scalar uniform-bound fast path.
        """
        if self.bounds is None:
            return ctx
        lower_is_tuple = isinstance(self.bounds.lower, tuple)
        upper_is_tuple = isinstance(self.bounds.upper, tuple)
        if not lower_is_tuple and not upper_is_tuple:
            # Scalar uniform bounds: config owns the values; keep ctx untouched.
            return ctx
        resolved = self.bounds.resolve(n_state=u0.shape[0], dtype=u0.dtype)
        return ctx._replace(bounds=resolved)

    def init(
        self,
        operator: Any,
        u0: Array,
        lam0: Array,
        theta: Optional[Array],
        ctx: Any,
    ) -> NonlinearState:
        """Build the box [NonlinearState][diff_soup.nonlinear._api.NonlinearState] seed.

        The seed is clipped into the box, with an inf error.

        Args:
            operator: The KKT-form operator (unused; carried for symmetry).
            u0: Initial differential iterate (clipped into the box at
                constrained DOFs).
            lam0: Initial global-algebraic iterate.
            theta: Optional parameter pytree (unused at seed time).
            ctx: Step context (reads ``ctx.bounds`` for per-DOF bounds).

        Returns:
            The [NonlinearState][diff_soup.nonlinear._api.NonlinearState] seed
            clipped into the feasible box with an infinite KKT residual.
        """
        del operator, theta
        ctx_eff = self._inject_bounds(ctx, u0)
        lb, ub, mask = resolve_box_arrays(ctx_eff, self.config, u0.shape[0], u0.dtype)
        u_init = jnp.where(mask, jnp.clip(u0, lb, ub), u0)
        inf = jnp.asarray(jnp.inf, dtype=u0.dtype)
        return NonlinearState(
            it=jnp.asarray(0, dtype=jnp.int32),
            u=u_init,
            lam=lam0,
            err=inf,
            first_nrm=inf,
            rate=jnp.asarray(0.0, dtype=u0.dtype),
            tooslow=jnp.asarray(False),
            done=jnp.asarray(False),
            residual_norm=inf,
        )

    def terminate(self, operator: Any, state: NonlinearState) -> tuple[Array, RESULT]:
        """Max-abs KKT convergence test (overrides the base WRMS test).

        Reports convergence when the carried KKT residual norm
        (``state.residual_norm``) is at or below ``config.atol`` — the same
        predicate the RSLS-Bertsekas primal uses internally.

        Args:
            operator: The KKT-form operator (unused; the box ``step`` carries the
                KKT norm on the state).
            state: The current
            [NonlinearState][diff_soup.nonlinear._api.NonlinearState].

        Returns:
            Tuple ``(stop, result)`` — whether to stop and the traced
            [RESULT][diff_soup._solution.RESULT] (``SUCCESS`` once the KKT norm is
            below ``atol``, else ``UNKNOWN``).
        """
        del operator
        stop = state.done
        result = RESULT.where(state.done, RESULT.SUCCESS, RESULT.UNKNOWN)
        return stop, result

    def step(
        self,
        operator: Any,
        backend: LinearSolver,
        analysis: Any,
        state: NonlinearState,
        theta: Optional[Array],
        ctx: Any,
    ) -> tuple[NonlinearState, dict]:
        """Run one branchless RSLS-Bertsekas box-Newton body iteration.

        The decomposed image of the primal's ``while_loop`` body: re-derive the
        bind set (Benson-Munson), pin active rows, Schur-solve the inactive
        block, take a projected step, and report the max-abs KKT norm at the new
        iterate.

        Args:
            operator: The KKT-form operator.
            backend: Linear solver for the pinned ``A``-block.
            analysis: The analysis of the ``A`` block's pattern.
            state: The current
            [NonlinearState][diff_soup.nonlinear._api.NonlinearState].
            theta: Optional parameter pytree.
            ctx: Step context (reads ``ctx.bounds`` for per-DOF bounds).

        Returns:
            Tuple ``(next_state, info)`` — the advanced state and a small
            per-step info dict (current KKT norm and iteration index).
        """
        from diff_soup.nonlinear.active_set import _apply_active_set_to_blocks

        cfg = self.config
        ctx_eff = self._inject_bounds(ctx, state.u)
        dtype = state.u.dtype
        lb, ub, mask = resolve_box_arrays(ctx_eff, cfg, state.u.shape[0], dtype)

        R, g = operator.residual(state.u, state.lam, theta, ctx_eff)
        bind_l, bind_u = benson_munson_bind(
            state.u, R, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
        )
        active_int = jnp.where(bind_l, -1, jnp.where(bind_u, 1, 0))
        Avals, B, C, D, topo = operator.jacobian_blocks(state.u, state.lam, theta, ctx_eff)
        Avals_p, B_p = _apply_active_set_to_blocks(Avals, B, topo.rows, topo.cols, active_int)
        R_pin = jnp.where(bind_l, state.u - lb, jnp.where(bind_u, state.u - ub, R))
        held = _factor_state_block(backend, analysis, Avals_p)
        du, dlam = kkt_solve(held, B_p, C, D, -R_pin, -g, cfg.schur_eps)

        u_next = jnp.where(mask, jnp.clip(state.u + du, lb, ub), state.u + du)
        lam_next = state.lam + dlam

        R_n, g_n = operator.residual(u_next, lam_next, theta, ctx_eff)
        bind_l_n, bind_u_n = benson_munson_bind(
            u_next, R_n, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
        )
        R_n_pin = jnp.where(bind_l_n, u_next - lb, jnp.where(bind_u_n, u_next - ub, R_n))
        kkt = jnp.max(jnp.abs(jnp.concatenate((R_n_pin, g_n), axis=0)))
        it_next = state.it + 1
        done = kkt <= jnp.asarray(cfg.atol, dtype=kkt.dtype)
        next_state = NonlinearState(
            it=it_next,
            u=u_next,
            lam=lam_next,
            err=kkt,
            first_nrm=jnp.where(state.it == 0, kkt, state.first_nrm),
            rate=jnp.asarray(0.0, dtype=dtype),
            tooslow=jnp.asarray(False),
            done=done,
            residual_norm=kkt,
        )
        return next_state, {"kkt_norm": kkt, "iteration": it_next}
