# > 500 LOC: base Newton solver + IFT JVP/VJP rules + utility helpers; primal +
# tangent + cotangent paths share the state spec for the hot path, and splitting
# the custom-AD rules from the primal they wrap would scatter one contract.
"""Unconstrained Newton corrector + IFT-based custom JVP/VJP for DAE stages.

Provides [nonlinear_solve_primal][] (the KKT-style corrector with Schur
complement over the global algebraic block) and the two AD-boundary entry points
[nonlinear_solve_ift_jvp][] / [nonlinear_solve_ift_vjp][] that wrap the
primal with implicit-function-theorem (IFT) derivatives.

The primal supports both fresh-Jacobian (``jacobian_cache="none"``) and
sticky-Jacobian (``jacobian_cache="reuse"``) Newton iterations; the sticky path
optionally retries with a fresh Jacobian when the rate estimate signals a stall.
The IFT rules avoid differentiating through the Newton loop entirely — they
linearise around the converged point and solve a single KKT system there. Every
factorization of the state block runs on the factor contract of
[diff_soup.linear][diff_soup.linear] with the provided
[LinearSolver][diff_soup.linear._api.LinearSolver]: the adjoint is a transposed
solve with the forward factor, never a factorization of the transpose.

Every WRMS norm routes through ``utils._safe_math.safe_sqrt`` and floors its
weights, so the grad of WRMS at an exact-zero update is finite under
checkpointed / IFT adjoints while staying value-identical on healthy input.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.custom_derivatives import zero_from_primal

from diff_soup.linear._api import LinearSolver
from diff_soup.nonlinear._kkt import (
    RouteFactor,
    _cast_like,
    _stable_small_solve,
    kkt_solve,
    kkt_solve_transposed,
    linear_route,
)
from diff_soup.utils import safe_sqrt


@dataclass(frozen=True)
class NonlinearConfig:
    """Frozen configuration for the unconstrained Newton corrector.

    Attributes:
        maxiter: Maximum inner Newton iterations per stage.
        rtol: Relative tolerance used for the WRMS convergence target.
        atol: Absolute tolerance used for the WRMS convergence target.
        lam_rtol: Optional per-block relative tolerance for the global
            algebraic (lam) slice of the WRMS test. ``None`` reuses ``rtol``.
        lam_atol: Optional per-block absolute tolerance for the lam slice.
            ``None`` reuses ``atol``.
        schur_eps: Tikhonov regularisation for the small Schur-complement
            solve over the global algebraic block.
        relinearize_on_tooslow: If ``True`` and ``jacobian_cache == "reuse"``,
            retry with a fresh Jacobian from the current iterate when the
            simplified-Newton rate estimate stalls.
        max_newton_iters_fsa: Newton iteration cap used by downstream forward
            sensitivity code paths.
        newton_rate_guard: Whether to latch ``tooslow`` based on the rate
            estimate when the projected residual fails to converge.
        safe_mode: Diagnostic toggle preserved for downstream callers; forces
            the bounded ``fori_loop`` form (disables the early-exit
            ``while_loop``).
        jacobian_cache: ``"reuse"`` for chord/simplified Newton (default) or
            ``"none"`` to refresh the Jacobian every iteration.
        return_residual: When ``True``, populate ``NewtonInfo.residual`` with
            the per-DOF ``[R; g]`` vector at the converged iterate (default
            ``False`` keeps a zero-length sentinel).
        max_stale_retries: Number of outer "fresh-Jacobian, zeroed-iterate"
            retries to perform when a chord sweep with a stale supplied
            Jacobian fails recoverably: re-setup with a fresh Jacobian, restart
            the Newton unknown from the predictor, and retry. This is the
            standard chord-Newton fallback for a stale linearization that fails
            to converge. Default ``0`` disables outer retries.
    """

    maxiter: int = 15
    rtol: float = 1e-6
    atol: float = 1e-8
    lam_rtol: Optional[float] = None
    lam_atol: Optional[float] = None
    schur_eps: float = 1e-12
    relinearize_on_tooslow: bool = True
    max_newton_iters_fsa: int = 10
    newton_rate_guard: bool = True
    safe_mode: bool = False
    jacobian_cache: Literal["none", "reuse"] = "reuse"
    return_residual: bool = False
    # Outer fresh-Jacobian, zeroed-iterate retries an engine performs when a
    # chord sweep with a stale supplied Jacobian fails recoverably.
    max_stale_retries: int = 0


class NewtonInfo(NamedTuple):
    """Diagnostics of one Newton solve, returned by `NonlinearSolver.solve`.

    Attributes:
        converged: Whether the convergence test passed.
        iterations: Newton iterations taken.
        residual_norm: Max-norm of the residual `[R; g]` at the returned iterate.
        tooslow: Whether the contraction rate stalled before convergence.
        ndecomps: Factorizations, equal to `factorizations`.
        nsolves: Linear solves.
        nfevals: Residual evaluations, including the final one.
        npds: Jacobian evaluations, equal to `jacobian_evals`.
        jacobian_evals: Jacobian evaluations.
        factorizations: Factorizations of the state block.
        solve_only: Solves against an existing factorization.
        cached_retries: 1 when the solve retried with fresh Jacobians after a stall, else 0.
        rho: Last observed contraction rate.
        residual: Residual `[R; g]` at the returned iterate when `config.return_residual` is set, else an empty array.
    """

    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    tooslow: jax.Array
    ndecomps: jax.Array
    nsolves: jax.Array
    nfevals: jax.Array
    npds: jax.Array
    jacobian_evals: jax.Array
    factorizations: jax.Array
    solve_only: jax.Array
    cached_retries: jax.Array
    rho: jax.Array
    residual: jax.Array


def _zero_newton_info(dtype) -> NewtonInfo:
    """Build an all-zero [NewtonInfo][diff_soup.nonlinear.solver.NewtonInfo] pytree in the given dtype."""
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    zero = jnp.asarray(0.0, dtype=dtype)
    return NewtonInfo(
        converged=jnp.asarray(False),
        iterations=zero_i32,
        residual_norm=zero,
        tooslow=jnp.asarray(False),
        ndecomps=zero_i32,
        nsolves=zero_i32,
        nfevals=zero_i32,
        npds=zero_i32,
        jacobian_evals=zero_i32,
        factorizations=zero_i32,
        solve_only=zero_i32,
        cached_retries=zero_i32,
        rho=zero,
        residual=jnp.zeros((0,), dtype=dtype),
    )


def _zero_tangent_like(x):
    """Build a zero tangent for ``x``, derived from its abstract value.

    The tangent inherits ``x``'s shape, its tangent dtype (``float0`` for
    integer or boolean leaves), and any device-placement annotation ``x``
    carries. Deriving the zero from the primal rather than building a fresh
    zero array is what lets these statistics tangents satisfy the custom-JVP
    output check when the solve is partitioned across a device mesh — a
    fabricated zero loses the partition annotation and is rejected.
    """
    return zero_from_primal(x, symbolic_zeros=False)


def _zero_newton_info_tangent(info: NewtonInfo) -> NewtonInfo:
    """Build a zero-tangent pytree in the shape of ``info`` for JVP output."""
    return jax.tree_util.tree_map(_zero_tangent_like, info)


def _residual_concat(
    operator: Any,
    u: jax.Array,
    lam: jax.Array,
    theta: Any,
    ctx: Any,
) -> jax.Array:
    """Evaluate ``[R; g]`` and return the stacked KKT residual vector."""
    R, g = operator.residual(u, lam, theta, ctx)
    return jnp.concatenate((R, g), axis=0)


def _safe_neg_leaf(x):
    """Negate inexact leaves; return ``None`` for integer/float0 leaves."""
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact):
        return -x
    return None


def _jacobian_blocks(operator: Any, u, lam, theta, ctx):
    """Read ``(Avals, B, C, D)`` from either operator block API."""
    if hasattr(operator, "jacobian_blocks_no_topology"):
        return operator.jacobian_blocks_no_topology(u, lam, theta, ctx)
    Avals, B, C, D, _ = operator.jacobian_blocks(u, lam, theta, ctx)
    return Avals, B, C, D


def _kkt_newton_step(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u: jax.Array,
    lam: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """One fresh-Jacobian KKT Newton step with Schur complement on globals."""
    R, g = operator.residual(u, lam, theta, ctx)
    Avals, B, C, D = _jacobian_blocks(operator, u, lam, theta, ctx)
    held = _factor_state_block(backend, analysis, Avals)

    du, dlam = kkt_solve(held, B, C, D, -R, -g, cfg.schur_eps)
    return du, dlam, jnp.max(jnp.abs(jnp.concatenate((R, g), axis=0))), (Avals, B, C, D)


def _factor_state_block(backend: LinearSolver, analysis: Any, Avals: jax.Array) -> RouteFactor:
    """Factor the state block ``A`` from its values on the factor contract.

    ``analysis`` is the analysis the caller holds for ``A``'s pattern: a
    backend's analysis state or a contract
    [Analysis][diff_soup.linear.Analysis]; its precision policy selects the
    solve mode.
    """
    Avals = jnp.asarray(Avals)
    return RouteFactor(linear_route(backend, analysis, dtype=Avals.dtype), Avals)


def _build_frozen_linearization(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u: jax.Array,
    lam: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
    *,
    frozen_blocks: Optional[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
):
    """Factor ``A`` once and pre-build Schur data for chord Newton sweeps.

    Returns ``(held, du_B, C, schur)``: the
    [RouteFactor][diff_soup.nonlinear._kkt.RouteFactor] of ``A`` (one
    factorization, reused by every chord iteration), the border solve
    ``A^{-1} B``, the coupling block and the Schur complement.
    """
    if frozen_blocks is not None:
        Avals, B, C, D = frozen_blocks
    else:
        Avals, B, C, D = _jacobian_blocks(operator, u, lam, theta, ctx)
    m = D.shape[0]
    held = _factor_state_block(backend, analysis, Avals)
    if B.shape[1] == 0:
        du_B = jnp.zeros((u.shape[0], m), dtype=u.dtype)
    else:
        du_B = _cast_like(held.solve(B.T), B).T
    schur = D - C @ du_B
    return held, du_B, C, schur


def _kkt_step_with_frozen_linearization(
    operator: Any,
    held: RouteFactor,
    du_B: jax.Array,
    C: jax.Array,
    schur: jax.Array,
    u: jax.Array,
    lam: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Chord-Newton KKT step reusing the pre-built factor and Schur data.

    ``held`` is the frozen factor from
    [_build_frozen_linearization][diff_soup.nonlinear.solver._build_frozen_linearization];
    the step solves against it without refactoring.
    """
    R, g = operator.residual(u, lam, theta, ctx)
    du_R = _cast_like(held.solve(-R), R)
    rhs_lam = -g - C @ du_R
    dlam = _stable_small_solve(schur, rhs_lam, cfg.schur_eps)
    du = du_R - du_B @ dlam
    return du, dlam, jnp.max(jnp.abs(jnp.concatenate((R, g), axis=0)))


def _wrms_norm(v: jax.Array, atol: float, rtol: float, y_ref: jax.Array) -> jax.Array:
    """Weighted RMS norm ``sqrt(mean((v / (atol + rtol |y_ref|))^2))``.

    The weight is floored (``jnp.maximum(wt, tiny)``) and the root goes through
    [safe_sqrt][diff_soup.utils._safe_math.safe_sqrt], so the cotangent stays finite
    at an exact-zero update. Value-identical to the unguarded form on healthy
    input (positive ``atol`` keeps ``wt`` above ``tiny``; ``safe_sqrt`` equals
    ``sqrt`` for positive arguments).
    """
    tiny = jnp.finfo(v.dtype).tiny
    wt = jnp.asarray(atol, dtype=v.dtype) + jnp.asarray(rtol, dtype=v.dtype) * jnp.abs(y_ref)
    wt = jnp.maximum(wt, tiny)
    return safe_sqrt(jnp.mean((v / wt) ** 2))


def _wrms_norm_per_block(
    du: jax.Array,
    dlam: jax.Array,
    u_ref: jax.Array,
    lam_ref: jax.Array,
    atol: float,
    rtol: float,
    lam_atol: Optional[float],
    lam_rtol: Optional[float],
) -> jax.Array:
    """WRMS norm with separate u-block and lam-block tolerances.

    When a per-element atol vector is provided, each block contributes
    element-wise squared-error terms against its own (atol, rtol) pair, then
    averaged across the combined length. When
    ``lam_atol`` / ``lam_rtol`` are ``None`` they fall back to the state-block
    ``atol`` / ``rtol``, making the per-block norm exactly equal to
    [_wrms_norm][diff_soup.nonlinear.solver._wrms_norm] on the concatenated vector.

    Weights are floored and the root uses ``safe_sqrt`` (finite grad at zero,
    value-identical on healthy input).
    """
    dtype = du.dtype
    tiny = jnp.finfo(dtype).tiny
    la = atol if lam_atol is None else lam_atol
    lr = rtol if lam_rtol is None else lam_rtol
    wt_u = jnp.asarray(atol, dtype=dtype) + jnp.asarray(rtol, dtype=dtype) * jnp.abs(u_ref)
    wt_lam = jnp.asarray(la, dtype=dtype) + jnp.asarray(lr, dtype=dtype) * jnp.abs(lam_ref)
    wt_u = jnp.maximum(wt_u, tiny)
    wt_lam = jnp.maximum(wt_lam, tiny)
    sq_u = (du / wt_u) ** 2
    sq_lam = (dlam / wt_lam) ** 2
    n_total = jnp.asarray(du.size + dlam.size, dtype=dtype)
    mean_sq = (jnp.sum(sq_u) + jnp.sum(sq_lam)) / jnp.maximum(
        n_total, jnp.asarray(1.0, dtype=dtype)
    )
    return safe_sqrt(mean_sq)


def nonlinear_solve_primal(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
    frozen_blocks: Optional[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
    factor_refit: bool = True,
    mass_refresh_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    early_exit: bool = False,
    newton_step_scale: Any = 1.0,
    refresh: Optional[jax.Array] = None,
    slot_reuse: bool = True,
    _ift_internal: bool = False,
) -> tuple[jax.Array, jax.Array, NewtonInfo]:
    """Run the KKT-style Newton corrector and return ``(u, lam, info)``.

    Dispatches to either the fresh-Jacobian or the sticky-Jacobian Newton loop
    per ``cfg.jacobian_cache``. If the sticky path stalls and
    ``cfg.relinearize_on_tooslow`` is set, a single retry with a fresh Jacobian
    from the latched iterate is performed.

    Args:
        operator: The KKT-form residual + Jacobian-blocks operator.
        backend: Linear solver used for factor + solve of the ``A`` block.
        analysis: The analysis of the ``A`` block's pattern: a contract
            [Analysis][diff_soup.linear.Analysis] or the state a backend's
            ``analyze`` returns (its precision policy selects the solve mode).
        u0: Initial guess for the differential part.
        lam0: Initial guess for the global algebraic part.
        theta: Optional parameter pytree passed to the residual.
        ctx: Step context carrying mass diagonal and other side-channels.
        cfg: Frozen [NonlinearConfig][diff_soup.nonlinear.solver.NonlinearConfig].
        frozen_blocks: Optional pre-computed ``(A, B, C, D)`` to reuse.
        factor_refit: Accepted and has no effect: every frozen linearization
            factors ``A`` once.
        mass_refresh_fn: Optional callback to refresh ``ctx.mass_diag`` at each
            iterate for state-dependent mass matrices.
        early_exit: Use a dynamic ``while_loop`` that stops as soon as Newton
            converges. Disable for direct reverse-mode differentiation through
            the primal; the IFT wrappers keep it enabled because the Newton
            loop is behind a custom-derivative boundary.
        newton_step_scale: stale-coefficient Newton-step rescale applied
            to every accepted iterate's update ``[du; dlam]``. Default ``1.0``
            is a no-op (XLA folds it) so every existing caller — JNDF's IFT
            wrappers and the box corrector — is bit-identical.
            The cross-step factor-reuse path passes ``2/(1+cjratio)`` on the
            cached arm (``cjratio=1`` ⇒ ``1.0`` on the fresh arm) so a chord
            step taken under a drifted iteration-matrix coefficient has the
            correct length.
        refresh: Accepted and has no effect: the chord sweep factors its
            frozen linearization once per call.
        slot_reuse: Accepted and has no effect.

    Returns:
        Tuple ``(u, lam, info)`` with the converged iterate and a
        [NewtonInfo][diff_soup.nonlinear.solver.NewtonInfo] of iteration counters.

    Raises:
        TypeError: If ``early_exit`` is requested on a traced ``u0``: the
            dynamic ``while_loop`` is not reverse-differentiable. Use the IFT
            wrappers ([nonlinear_solve_ift_vjp][] /
            [nonlinear_solve_ift_jvp][]) or set ``early_exit=False``.
    """
    # The early-exit while_loop is not reverse-differentiable. Guard eagerly at
    # trace time so the failure is a clear configuration error, not a cryptic
    # XLA transpose failure deep in an adjoint. The IFT wrappers below call with
    # ``early_exit=True`` behind their custom-derivative boundary, where ``u0``
    # is a tracer but reverse mode is supplied by the custom_vjp (the primal
    # while_loop is never reverse-transposed by JAX). Those calls pass
    # ``_ift_internal=True`` to opt out of the user-facing guard — the corrector
    # reaches the primal only through the IFT wrappers, so a tracer ``u0`` here
    # is the legitimate in-loop case.
    if (
        bool(early_exit)
        and not bool(_ift_internal)
        and not bool(cfg.safe_mode)
        and isinstance(u0, jax.core.Tracer)
    ):
        raise TypeError(
            "nonlinear_solve_primal(early_exit=True) uses a dynamic while_loop "
            "that is not reverse-differentiable; call through the IFT wrappers "
            "(nonlinear_solve_ift_vjp / _jvp) or set early_exit=False."
        )

    class Carry(NamedTuple):
        it: jax.Array
        u: jax.Array
        lam: jax.Array
        err: jax.Array
        first_nrm: jax.Array
        rate: jax.Array
        tooslow: jax.Array
        done: jax.Array
        residual_norm: jax.Array  # max-abs |[R; g]| from last iteration

    init = Carry(
        it=jnp.asarray(0, dtype=jnp.int32),
        u=u0,
        lam=lam0,
        err=jnp.asarray(jnp.inf, dtype=u0.dtype),
        first_nrm=jnp.asarray(jnp.inf, dtype=u0.dtype),
        rate=jnp.asarray(0.0, dtype=u0.dtype),
        tooslow=jnp.asarray(False),
        done=jnp.asarray(False),
        residual_norm=jnp.asarray(jnp.inf, dtype=u0.dtype),
    )

    def _run(carry_init: Carry, cfg_run: NonlinearConfig, use_frozen: bool):
        if use_frozen:
            held, du_B, C, schur = _build_frozen_linearization(
                operator,
                backend,
                analysis,
                carry_init.u,
                carry_init.lam,
                theta,
                ctx,
                cfg_run,
                frozen_blocks=frozen_blocks,
            )
            # One factorisation per frozen setup.
            fact_inc = jnp.asarray(1, dtype=jnp.int32)
            ndecomp_base = fact_inc
            npd_base = jnp.asarray(0 if frozen_blocks is not None else 1, dtype=jnp.int32)
            jac_base = npd_base
            fact_base = fact_inc
            # Building frozen Schur data uses one factor-solve over B^T.
            solve_only_base = jnp.asarray(1, dtype=jnp.int32)
        else:
            held = du_B = C = schur = None
            ndecomp_base = jnp.asarray(0, dtype=jnp.int32)
            npd_base = jnp.asarray(0, dtype=jnp.int32)
            jac_base = jnp.asarray(0, dtype=jnp.int32)
            fact_base = jnp.asarray(0, dtype=jnp.int32)
            solve_only_base = jnp.asarray(0, dtype=jnp.int32)

        def body(x: Carry):
            # Refresh mass diagonal at the current Newton iterate when the mass
            # is state-dependent. Keeps ctx.mass_diag consistent with x.u.
            if mass_refresh_fn is not None:
                ctx_iter = ctx._replace(mass_diag=mass_refresh_fn(x.u, x.lam))
            else:
                ctx_iter = ctx
            if use_frozen:
                du, dlam, err = _kkt_step_with_frozen_linearization(
                    operator,
                    held,
                    du_B,
                    C,
                    schur,
                    x.u,
                    x.lam,
                    theta,
                    ctx_iter,
                    cfg_run,
                )
            else:
                du, dlam, err, _blocks = _kkt_newton_step(
                    operator,
                    backend,
                    analysis,
                    x.u,
                    x.lam,
                    theta,
                    ctx_iter,
                    cfg_run,
                )
            # Stale-coefficient Newton-step rescale
            # (``b *= 2/(1+cjratio)``). ``newton_step_scale`` is 1.0 (folded
            # no-op) on every non-reuse path, so the convergence-rate / WRMS
            # bookkeeping below is computed on the rescaled (actually-taken)
            # step — exactly as it would be with a fresh factor at the correct
            # cj. Scales both blocks (the full Newton update vector b).
            du = du * newton_step_scale
            dlam = dlam * newton_step_scale
            u_next = x.u + du
            lam_next = x.lam + dlam
            if cfg_run.lam_atol is None and cfg_run.lam_rtol is None:
                # Bit-exact single-block WRMS fallback. This is critical for
                # non-DAE problems (Burgers etc.) where tiny FP-order
                # differences vs the per-block helper alter the convergence path
                # on stiff discontinuous ICs at tight Newton tols.
                delta = jnp.concatenate((du, dlam), axis=0)
                yref = jnp.concatenate((u_next, lam_next), axis=0)
                nrm = _wrms_norm(delta, cfg_run.atol, cfg_run.rtol, yref)
            else:
                # Per-block state/global-algebraic tolerances.
                nrm = _wrms_norm_per_block(
                    du,
                    dlam,
                    u_next,
                    lam_next,
                    cfg_run.atol,
                    cfg_run.rtol,
                    cfg_run.lam_atol,
                    cfg_run.lam_rtol,
                )
            minnrm = jnp.asarray(100.0, dtype=nrm.dtype) * jnp.finfo(nrm.dtype).eps
            first_iter = x.it == 0
            first_nrm = jnp.where(first_iter, nrm, x.first_nrm)
            order = jnp.maximum(x.it, jnp.asarray(1, dtype=x.it.dtype))
            nrm_ratio = nrm / jnp.maximum(first_nrm, jnp.asarray(1e-30, dtype=nrm.dtype))
            rate_raw = nrm_ratio ** (1.0 / order.astype(nrm.dtype))
            rate = jnp.clip(jnp.where(first_iter, 0.0, rate_raw), 0.0, 0.999999)
            # Rate-ratio nonlinear convergence test: S * ||delta||_WRMS < safety,
            # with S = R / (1 - R) estimated from the observed per-iterate
            # contraction rate R (cf. Hindmarsh et al. 2005). The first
            # iteration has no rate estimate yet, so S starts at a conservative
            # 20. We use safety=0.1 to keep small-problem FD-vs-AD agreement
            # while still letting stiff hyperbolic problems converge.
            safety = jnp.asarray(0.1, dtype=nrm.dtype)
            s_est = jnp.where(
                first_iter,
                jnp.asarray(20.0, dtype=nrm.dtype),
                rate / jnp.maximum(1.0 - rate, jnp.asarray(1e-12, dtype=nrm.dtype)),
            )
            converged = jnp.logical_or(nrm <= minnrm, s_est * nrm <= safety)
            tooslow = jnp.logical_and(
                cfg_run.newton_rate_guard,
                jnp.logical_and(
                    jnp.logical_not(converged),
                    rate > jnp.asarray(0.9, dtype=nrm.dtype),
                ),
            )
            return Carry(
                it=x.it + 1,
                u=u_next,
                lam=lam_next,
                err=nrm,
                first_nrm=first_nrm,
                rate=rate,
                tooslow=tooslow,
                done=converged,
                residual_norm=err,
            )

        def cond(x: Carry):
            return jnp.logical_and(
                x.it < jnp.asarray(cfg_run.maxiter, dtype=x.it.dtype),
                jnp.logical_not(jnp.logical_or(x.done, x.tooslow)),
            )

        if bool(early_exit) and not bool(cfg_run.safe_mode):
            out = jax.lax.while_loop(cond, body, carry_init)
        else:

            def fori_body(_i, c: Carry):
                return jax.lax.cond(
                    jnp.logical_or(c.done, c.tooslow),
                    lambda x: x,
                    body,
                    c,
                )

            out = jax.lax.fori_loop(0, int(cfg_run.maxiter), fori_body, carry_init)
        iters = out.it
        nsolves = iters
        nfevals = iters
        ndecomps = ndecomp_base + jnp.where(use_frozen, jnp.asarray(0, dtype=jnp.int32), iters)
        npds = npd_base + jnp.where(use_frozen, jnp.asarray(0, dtype=jnp.int32), iters)
        jacobian_evals = jac_base + jnp.where(use_frozen, jnp.asarray(0, dtype=jnp.int32), iters)
        factorizations = fact_base + jnp.where(use_frozen, jnp.asarray(0, dtype=jnp.int32), iters)
        solve_only = solve_only_base + jnp.where(use_frozen, iters, jnp.asarray(0, dtype=jnp.int32))
        return out, ndecomps, nsolves, nfevals, npds, jacobian_evals, factorizations, solve_only

    use_frozen = cfg.jacobian_cache == "reuse"
    out, ndecomps, nsolves, nfevals, npds, jacobian_evals, factorizations, solve_only = _run(
        init, cfg, use_frozen
    )

    cached_retries = jnp.asarray(0, dtype=jnp.int32)

    # If simplified Newton is too slow, retry from the current iterate with
    # refreshed Jacobians.
    do_retry = use_frozen and bool(cfg.relinearize_on_tooslow)
    if do_retry:
        cfg_retry = NonlinearConfig(
            maxiter=cfg.maxiter,
            rtol=cfg.rtol,
            atol=cfg.atol,
            lam_rtol=cfg.lam_rtol,
            lam_atol=cfg.lam_atol,
            schur_eps=cfg.schur_eps,
            jacobian_cache="none",
            relinearize_on_tooslow=False,
            max_newton_iters_fsa=cfg.max_newton_iters_fsa,
            newton_rate_guard=cfg.newton_rate_guard,
            safe_mode=cfg.safe_mode,
        )
        retry_seed = Carry(
            it=jnp.asarray(0, dtype=jnp.int32),
            u=out.u,
            lam=out.lam,
            err=out.err,
            first_nrm=jnp.asarray(jnp.inf, dtype=out.err.dtype),
            rate=jnp.asarray(0.0, dtype=out.err.dtype),
            tooslow=jnp.asarray(False),
            done=jnp.asarray(False),
            residual_norm=jnp.asarray(jnp.inf, dtype=out.err.dtype),
        )
        need_retry = jnp.logical_and(out.tooslow, jnp.logical_not(out.done))

        def _run_retry(_):
            return _run(retry_seed, cfg_retry, False)

        def _skip_retry(_):
            zero_i32 = jnp.asarray(0, dtype=jnp.int32)
            return out, zero_i32, zero_i32, zero_i32, zero_i32, zero_i32, zero_i32, zero_i32

        out_retry, ndecomp2, nsolves2, nfevals2, npds2, jac2, fact2, solve_only2 = jax.lax.cond(
            need_retry,
            _run_retry,
            _skip_retry,
            operand=None,
        )
        out = out_retry
        ndecomps = ndecomps + ndecomp2
        nsolves = nsolves + nsolves2
        nfevals = nfevals + nfevals2
        npds = npds + npds2
        jacobian_evals = jacobian_evals + jac2
        factorizations = factorizations + fact2
        solve_only = solve_only + solve_only2
        cached_retries = jnp.where(
            need_retry, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)
        )

    # Compute the true max-abs residual norm at the converged solution.
    # out.residual_norm is |R(u_i)| (BEFORE the last Newton update); the NDF
    # core's nl_res_ok gate needs the actual post-update residual |R(u_{i+1})|.
    if mass_refresh_fn is not None:
        ctx_final = ctx._replace(mass_diag=mass_refresh_fn(out.u, out.lam))
    else:
        ctx_final = ctx
    R_final, g_final = operator.residual(out.u, out.lam, theta, ctx_final)
    # stop_gradient: this norm feeds only a boolean comparison (nl_res_ok), so
    # its derivatives are dead — stopping them avoids wasted tangent/cotangent
    # propagation through the residual under AD.
    final_res_norm = jax.lax.stop_gradient(
        jnp.max(jnp.abs(jnp.concatenate((R_final, g_final), axis=0)))
    )

    # Emit the per-DOF [R; g] residual when the caller asked for it (gate
    # predicate only; derivatives dead — same stop_gradient treatment).
    if cfg.return_residual:
        residual_full = jax.lax.stop_gradient(jnp.concatenate((R_final, g_final), axis=0))
    else:
        residual_full = jnp.zeros((0,), dtype=out.u.dtype)

    info = NewtonInfo(
        converged=out.done,
        iterations=out.it,
        residual_norm=final_res_norm,
        tooslow=out.tooslow,
        ndecomps=ndecomps,
        nsolves=nsolves,
        nfevals=nfevals + jnp.asarray(1, dtype=jnp.int32),
        npds=npds,
        jacobian_evals=jacobian_evals,
        factorizations=factorizations,
        solve_only=solve_only,
        cached_retries=cached_retries,
        rho=out.rate,
        residual=residual_full,
    )
    return out.u, out.lam, info


def linearize_at_root_jvp(operator, backend, analysis, cfg, u, lam, theta, ctx, dtheta, dctx):
    """Tangent of the constrained root w.r.t. ``(theta, ctx)`` at a supplied root.

    The implicit-function tangent: factor the state block at ``(u, lam)``, push
    the ``(dtheta, dctx)`` directional residual derivative through the KKT
    elimination. The tangent solves are linear in the directional residual,
    so JAX transposes this rule for reverse mode and differentiates it again.
    The caller owns producing the converged root.
    """
    n = u.shape[0]
    Avals, B, C, D = _jacobian_blocks(operator, u, lam, theta, ctx)
    held = _factor_state_block(backend, analysis, Avals)
    _, dF = jax.jvp(
        lambda th, cx: _residual_concat(operator, u, lam, th, cx), (theta, ctx), (dtheta, dctx)
    )
    du, dlam = kkt_solve(held, B, C, D, -dF[:n], -dF[n:], cfg.schur_eps, linear=True)
    return du, dlam


def linearize_at_root_vjp(operator, backend, analysis, cfg, u, lam, theta, ctx, ct_u, ct_lam):
    """Cotangent pullback onto ``(theta, ctx)`` at a supplied root (transposed system).

    The transposed system is solved with the forward factor of ``A`` at the
    root (a transposed solve), never with a factorization of ``A^T``.
    """
    Avals, B, C, D = _jacobian_blocks(operator, u, lam, theta, ctx)
    held = _factor_state_block(backend, analysis, Avals)
    p, q = kkt_solve_transposed(
        held, B, C, D, jnp.asarray(ct_u), jnp.asarray(ct_lam), cfg.schur_eps
    )
    adj = jnp.concatenate((p, q), axis=0)
    if theta is not None:
        _, vjp_both = jax.vjp(lambda th, cx: _residual_concat(operator, u, lam, th, cx), theta, ctx)
        g_th, g_cx = vjp_both(adj)
        return (
            jax.tree_util.tree_map(_safe_neg_leaf, g_th),
            jax.tree_util.tree_map(_safe_neg_leaf, g_cx),
        )
    _, vjp_ctx = jax.vjp(lambda cx: _residual_concat(operator, u, lam, theta, cx), ctx)
    return None, jax.tree_util.tree_map(_safe_neg_leaf, vjp_ctx(adj)[0])


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 7))
def nonlinear_solve_ift_vjp(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
    frozen_blocks=None,
    newton_step_scale: Any = 1.0,
    refresh=None,
):
    """Newton corrector wrapped in a custom VJP based on the IFT adjoint.

    ``frozen_blocks`` (a pre-evaluated ``(A, B, C, D)`` Jacobian-block tuple to
    reuse instead of re-evaluating), ``newton_step_scale`` (the cross-step
    stale-``cj`` chord correction ``2/(1+cjratio)``), and ``refresh`` (accepted
    and without effect) are **primal-only** controls: they speed the forward
    iterate but never change the converged root. They ride as ordinary
    arguments (so a per-step traced value is accepted) but are gradient-dead —
    the adjoint returns a zero cotangent for them and recomputes the Jacobian at
    the root regardless, factoring it once and solving the transposed system
    with that factor.
    """
    u, lam, info = nonlinear_solve_primal(
        operator,
        backend,
        analysis,
        u0,
        lam0,
        theta,
        ctx,
        cfg,
        frozen_blocks=frozen_blocks,
        newton_step_scale=newton_step_scale,
        refresh=refresh,
        early_exit=True,
        _ift_internal=True,
    )
    return u, lam, info


def _ift_vjp_fwd(
    operator,
    backend,
    analysis,
    u0,
    lam0,
    theta,
    ctx,
    cfg,
    frozen_blocks,
    newton_step_scale,
    refresh,
):
    """Forward pass: run the primal and save the root plus the primal-only controls."""
    u, lam, info = nonlinear_solve_primal(
        operator,
        backend,
        analysis,
        u0,
        lam0,
        theta,
        ctx,
        cfg,
        frozen_blocks=frozen_blocks,
        newton_step_scale=newton_step_scale,
        refresh=refresh,
        early_exit=True,
        _ift_internal=True,
    )
    # u0/lam0 are not saved: grad_u0 = grad_lam0 = 0 (IFT — the root is
    # independent of the initial guess), and dropping them shrinks the tape.
    # frozen_blocks/newton_step_scale/refresh are saved only to shape their zero
    # cotangents; the adjoint recomputes the Jacobian at the converged root and
    # never differentiates through these primal-only controls.
    return (u, lam, info), (u, lam, theta, ctx, frozen_blocks, newton_step_scale, refresh)


def _ift_vjp_bwd(operator, backend, analysis, cfg, res, ct):
    """Backward pass: solve one adjoint KKT system and VJP through the residual."""
    u, lam, theta, ctx, frozen_blocks, newton_step_scale, refresh = res
    ct_u, ct_lam, _ct_info = ct

    grad_theta, grad_ctx = linearize_at_root_vjp(
        operator, backend, analysis, cfg, u, lam, theta, ctx, ct_u, ct_lam
    )

    # Root output is independent of the initial guesses (IFT).
    grad_u0 = jnp.zeros_like(u)
    grad_lam0 = jnp.zeros_like(lam)

    # frozen_blocks / newton_step_scale / refresh are primal-only: their
    # cotangent is a structural zero (the converged root, hence the loss, does
    # not depend on them). ``zero_from_primal`` matches each leaf's aval and
    # device placement; ``refresh`` is bool/None, so its tangent is float0/None.
    grad_frozen = jax.tree_util.tree_map(_zero_tangent_like, frozen_blocks)
    grad_scale = _zero_tangent_like(jnp.asarray(newton_step_scale))
    grad_refresh = jax.tree_util.tree_map(_zero_tangent_like, refresh)

    return grad_u0, grad_lam0, grad_theta, grad_ctx, grad_frozen, grad_scale, grad_refresh


nonlinear_solve_ift_vjp.defvjp(_ift_vjp_fwd, _ift_vjp_bwd)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 7))
def nonlinear_solve_ift_jvp(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: NonlinearConfig,
    frozen_blocks=None,
    newton_step_scale: Any = 1.0,
    refresh=None,
):
    """Newton corrector wrapped in a custom JVP based on the IFT tangent.

    ``frozen_blocks`` / ``newton_step_scale`` / ``refresh`` are primal-only (see
    the VJP entry): gradient-dead controls that speed the forward iterate without
    changing the converged root, so they ride as ordinary arguments (a per-step
    traced value is accepted) and the IFT tangent below ignores their tangents
    and recomputes the Jacobian at the root regardless.
    """
    u, lam, info = nonlinear_solve_primal(
        operator,
        backend,
        analysis,
        u0,
        lam0,
        theta,
        ctx,
        cfg,
        frozen_blocks=frozen_blocks,
        newton_step_scale=newton_step_scale,
        refresh=refresh,
        early_exit=True,
        _ift_internal=True,
    )
    return u, lam, info


@nonlinear_solve_ift_jvp.defjvp
def _ift_jvp(operator, backend, analysis, cfg, primals, tangents):
    """JVP: linearise ``F(y, theta, ctx)=0`` at ``y*`` and solve for ``dy``.

    The converged root is recomputed with the bounded (freeze-on-done) Newton
    loop rather than the early-exit ``while_loop``: this rule's graph is what
    a second application of autodiff differentiates, and the bounded loop
    supports both directions, so reverse-over-forward and
    forward-over-forward compositions through the forward sensitivities
    compose. The converged value is identical (iterations past convergence
    are frozen), and an undifferentiated call to the atom still runs the
    early-exit primal.
    """
    u0, lam0, theta, ctx, frozen_blocks, newton_step_scale, refresh = primals
    du0, dlam0, dtheta, dctx, dfrozen, dscale, drefresh = tangents
    # u0/lam0 tangents are dead (IFT); frozen_blocks/newton_step_scale/refresh
    # are primal-only, so their tangents do not enter the converged-root tangent.
    del du0, dlam0, dfrozen, dscale, drefresh

    u, lam, info = nonlinear_solve_primal(
        operator,
        backend,
        analysis,
        u0,
        lam0,
        theta,
        ctx,
        cfg,
        frozen_blocks=frozen_blocks,
        newton_step_scale=newton_step_scale,
        refresh=refresh,
        early_exit=False,
        _ift_internal=True,
    )

    du, dlam = linearize_at_root_jvp(
        operator, backend, analysis, cfg, u, lam, theta, ctx, dtheta, dctx
    )
    info_tan = _zero_newton_info_tangent(info)
    return (u, lam, info), (du, dlam, info_tan)
