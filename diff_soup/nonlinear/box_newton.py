"""Unified box-constrained Newton: branchless RSLS-Bertsekas primal + reduced-space IFT.

A single-level [jax.lax.while_loop][] box-constrained Newton. The bind set
is updated every Newton iteration via the Benson-Munson 2006 residual-sign rule
(no hysteresis, no probe); active rows are pinned via
[_apply_active_set_to_blocks][]. The line search is a fixed-alpha projected
Armijo scan ([projected_armijo_scan][]) — no [jax.lax.cond][] fires in
the hot path.

The ``custom_vjp`` / ``custom_jvp`` rules linearise the *same* equation the
forward primal solves: a reduced-space IFT under strict complementarity, with the
converged-point bind set held fixed. Active rows of the cotangent are zeroed
analytically (no Fischer-Burmeister sigma scaling), so gradients through
bound-active solves are unbiased — the exact-zero-at-active-bounds property
verified by the box golden tests.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp

from diff_soup.linear._api import LinearSolver
from diff_soup.nonlinear._helpers import (
    benson_munson_bind,
    projected_armijo_scan,
)
from diff_soup.nonlinear._kkt import kkt_solve, kkt_solve_transposed
from diff_soup.nonlinear.active_set import _apply_active_set_to_blocks
from diff_soup.nonlinear.box_bounds import resolve_box_arrays
from diff_soup.nonlinear.solver import (
    NonlinearConfig,
    _factor_state_block,
    _zero_tangent_like,
)


@dataclass(frozen=True)
class BoxConstrainedUnifiedConfig(NonlinearConfig):
    """Config for the unified RSLS-Bertsekas Newton.

    Field parity with the legacy ``BoxConstrainedConfig`` for the two scalar
    bounds (``lower_scalar``, ``upper_scalar``) is required by the resolver
    [resolve_box_arrays][diff_soup.nonlinear.box_bounds.resolve_box_arrays]
    — it reads these
    when ``ctx.bounds`` is unset (uniform-bound fast path). Per-DOF bounds flow
    through ``ctx.bounds`` as a ``_ResolvedBounds``.

    Attributes:
        constrained_indices: DOF indices that are box-constrained.
        bound_tol: Bound-proximity tolerance for the bind rule.
        bound_F_tol: Residual-sign tolerance for the bind rule.
        armijo_c1: Armijo descent constant.
        armijo_alphas: Fixed projected-Armijo step-size schedule.
        lower_scalar: Uniform lower bound (uniform-bound fast path).
        upper_scalar: Uniform upper bound (uniform-bound fast path).
    """

    constrained_indices: tuple = ()
    bound_tol: float = 1e-6
    bound_F_tol: float = 1e-12
    armijo_c1: float = 1e-4
    armijo_alphas: tuple = (1.0, 0.5, 0.25, 0.125)
    lower_scalar: float = 0.0
    upper_scalar: float = 1.0


class _BoxSolveParams(NamedTuple):
    """Resolved scalar params for the box solve, normalised across configs.

    The single typed image of the fields the box corrector needs from a
    time-layer config — resolved at one typed point
    ([resolve_box_solve_params][diff_soup.nonlinear.box_newton.resolve_box_solve_params])
    instead of inline ``getattr`` probes.

    Attributes:
        schur_eps: Schur-complement regularisation.
        maxiter: Newton iteration cap.
        newton_rate_guard: Whether the rate guard is active.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        use_jvp: Whether the forward-mode (JVP) box primal is selected.
    """

    schur_eps: float
    maxiter: int
    newton_rate_guard: bool
    rtol: float
    atol: float
    use_jvp: bool


def _adjoint_is_forward_mode(adjoint_obj) -> bool:
    """True iff the declared adjoint selects the forward-mode (JVP) box primal.

    Reads the cross-layer ``AbstractAdjoint`` discriminant: a
    [ForwardAdjoint][diff_soup.nonlinear._api.ForwardAdjoint] carries
    ``ift_mode == "jvp"``. Anything else (``None``, ``ImplicitAdjoint``,
    ``CheckpointAdjoint``) is reverse-mode.

    Args:
        adjoint_obj: The config's declared adjoint instance, or ``None``.

    Returns:
        Whether the forward-mode (``_jvp``) box primal should be selected.
    """
    if adjoint_obj is None:
        return False
    return getattr(type(adjoint_obj), "ift_mode", None) == "jvp"


def resolve_box_solve_params(cfg, *, default_maxiter: int, eps_floor: float) -> "_BoxSolveParams":
    """Normalise a solver time-config into typed box-solve params.

    The single typed dispatch point that retires per-field ``getattr``
    duck-typing in the time-layer box-corrector role. The two config shapes are
    discriminated once by the presence of a ``nonlinear`` sub-config versus its
    absence; every field is then read from the config that owns it.

    Args:
        cfg: The time-layer solver config — a nested config (has a ``nonlinear``
            [NonlinearConfig][diff_soup.nonlinear.solver.NonlinearConfig]
            sub-config, ``max_newton_iters``,
            ``newton_rate_guard``, ``adjoint``) or a flat config
            (``newton_max_iter``; no ``nonlinear`` / ``newton_rate_guard`` /
            ``adjoint``).
        default_maxiter: Fallback Newton iteration cap.
        eps_floor: Numerical floor for the Schur regularisation.

    Returns:
        The resolved
        [_BoxSolveParams][diff_soup.nonlinear.box_newton._BoxSolveParams].
    """
    nl_obj = getattr(cfg, "nonlinear", None)
    if nl_obj is not None and hasattr(nl_obj, "schur_eps"):
        # Nested shape: schur_eps on the nonlinear sub-config; max-iter on
        # ``max_newton_iters``; rate guard + adjoint slot are on the outer config.
        schur_eps = max(float(nl_obj.schur_eps), eps_floor)
        maxiter = int(getattr(cfg, "max_newton_iters", default_maxiter))
        newton_rate_guard = bool(getattr(cfg, "newton_rate_guard", False))
    else:
        # Flat shape: flat schur_eps (defaults to eps_floor); max-iter on
        # ``newton_max_iter``; no rate guard.
        schur_eps = max(float(getattr(cfg, "schur_eps", eps_floor)), eps_floor)
        maxiter = int(getattr(cfg, "newton_max_iter", default_maxiter))
        newton_rate_guard = False

    rtol = float(getattr(cfg, "rtol", 1e-6))
    atol = float(getattr(cfg, "atol", 1e-9))

    use_jvp = _adjoint_is_forward_mode(getattr(cfg, "adjoint", None))

    return _BoxSolveParams(
        schur_eps=schur_eps,
        maxiter=maxiter,
        newton_rate_guard=newton_rate_guard,
        rtol=rtol,
        atol=atol,
        use_jvp=use_jvp,
    )


class BoxConstrainedUnifiedInfo(NamedTuple):
    """Diagnostic counters returned by the box-constrained Newton primal.

    The unified primal runs a single-level loop, so ``active_set_iters`` equals
    the total Newton iteration count. ``rho`` is a sentinel zero (the
    ``while_loop`` does not carry the previous KKT norm). ``active_set_final``
    carries the converged active set in ``bool[(n_constrained,)]`` form for
    warm-start handoff.

    Attributes:
        converged: KKT-convergence achieved (bool).
        iterations: Newton iterations taken (int32).
        residual_norm: Max-abs KKT residual at the converged iterate (float).
        tooslow: Always ``False`` for the unified primal (bool).
        ndecomps: Decompositions (int32).
        nsolves: Linear solves (int32).
        nfevals: Residual evaluations (int32).
        npds: Jacobian rebuilds (int32).
        jacobian_evals: Jacobian evaluations (int32).
        factorizations: Numeric factorisations (int32).
        solve_only: Frozen-factor solves (int32).
        cached_retries: Sticky-Jacobian retries (int32).
        active_set_iters: Total Newton iters (single-level identification).
        rho: Convergence-rate sentinel (float, 0).
        active_set_final: Converged active set, ``bool[(n_constrained,)]``.
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
    active_set_iters: jax.Array
    rho: jax.Array
    active_set_final: jax.Array


def box_constrained_solve_primal_unified(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: BoxConstrainedUnifiedConfig,
    mass_refresh_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
) -> tuple[jax.Array, jax.Array, BoxConstrainedUnifiedInfo]:
    """Branchless RSLS-Bertsekas box-constrained Newton primal.

    Internal raw primal — not a public symbol and not in ``__all__``. Production
    callers go through
    [box_constrained_solve_vjp][diff_soup.nonlinear.box_newton.box_constrained_solve_vjp]
    /
    [box_constrained_solve_jvp][diff_soup.nonlinear.box_newton.box_constrained_solve_jvp],
    which add the reduced-space IFT rules.

    Args:
        operator: The KKT-form residual + Jacobian-blocks operator.
        backend: Linear solver for the pinned ``A``-block factor + solve.
        analysis: The analysis of the ``A`` block's pattern: a contract
            [Analysis][diff_soup.linear.Analysis] or the state a backend's
            ``analyze`` returns.
        u0: Initial guess for the differential part.
        lam0: Initial guess for the global algebraic part.
        theta: Optional parameter pytree passed to the residual.
        ctx: Step context (reads ``ctx.bounds`` / ``ctx.prev_active_set``).
        cfg: The frozen
            [BoxConstrainedUnifiedConfig][diff_soup.nonlinear.box_newton.BoxConstrainedUnifiedConfig].
        mass_refresh_fn: Optional state-dependent mass-refresh callback.

    Returns:
        Tuple ``(u, lam, info)`` — the converged iterate and box counters.
    """
    n_u = u0.shape[0]
    dtype = u0.dtype

    # Read bounds + constrained mask through the public resolver. ``ctx.bounds``
    # carries a ``_ResolvedBounds``; ``ctx.bounds=None`` falls back to
    # ``cfg.lower_scalar`` / ``cfg.upper_scalar`` + ``cfg.constrained_indices``.
    lb, ub, mask = resolve_box_arrays(ctx, cfg, n_u, dtype)
    alphas = jnp.asarray(cfg.armijo_alphas, dtype=dtype)

    u_init = jnp.where(mask, jnp.clip(u0, lb, ub), u0)

    # Warm-start active-set seed. ``ctx.prev_active_set`` is
    # ``bool[(n_constrained,)]``; the full-state ``int8`` representation is
    # internal. The seed is purely a "where to look first" hint — the first
    # iteration's ``benson_munson_bind`` recomputes the bind set.
    prev_as_bool = getattr(ctx, "prev_active_set", None)
    if prev_as_bool is None or len(cfg.constrained_indices) == 0:
        active_init = jnp.zeros((n_u,), dtype=jnp.int8)
    else:
        idx = jnp.asarray(cfg.constrained_indices, dtype=jnp.int32)
        prev_seed = jnp.asarray(prev_as_bool, dtype=jnp.bool_)
        active_init = (
            jnp.zeros((n_u,), dtype=jnp.int8)
            .at[idx]
            .set(jnp.where(prev_seed, jnp.int8(-1), jnp.int8(0)))
        )

    class Carry(NamedTuple):
        k: jax.Array  # iteration counter (int32)
        u: jax.Array
        lam: jax.Array
        kkt: jax.Array  # KKT residual norm
        done: jax.Array  # bool
        active_set: jax.Array  # int8 [(n_u,)] in {-1, 0, +1}; -1=lower, +1=upper

    def _body(c: Carry):
        ctx_iter = ctx
        if mass_refresh_fn is not None:
            ctx_iter = ctx._replace(mass_diag=mass_refresh_fn(c.u, c.lam))

        R, g = operator.residual(c.u, c.lam, theta, ctx_iter)

        # Benson-Munson bind: per-iter, no hysteresis.
        bind_l, bind_u = benson_munson_bind(
            c.u, R, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
        )
        active_int = jnp.where(bind_l, -1, jnp.where(bind_u, 1, 0))

        Avals, B, C, D, topo = operator.jacobian_blocks(c.u, c.lam, theta, ctx_iter)
        Avals_p, B_p = _apply_active_set_to_blocks(Avals, B, topo.rows, topo.cols, active_int)
        R_pin = jnp.where(bind_l, c.u - lb, jnp.where(bind_u, c.u - ub, R))

        held = _factor_state_block(backend, analysis, Avals_p)

        # Schur-complement solve: J · [du; dlam] = -[R_pin; g]
        du, dlam = kkt_solve(held, B_p, C, D, -R_pin, -g, cfg.schur_eps)

        # Projected Armijo on the KKT-aware merit.
        def merit(y):
            R_y, g_y = operator.residual(y, c.lam + dlam, theta, ctx_iter)
            R_y_pin = jnp.where(bind_l, y - lb, jnp.where(bind_u, y - ub, R_y))
            return 0.5 * (jnp.sum(R_y_pin * R_y_pin) + jnp.sum(g_y * g_y))

        # Analytic descent estimate avoids a second Jacobian eval.
        descent_est = -(jnp.sum(R_pin * R_pin) + jnp.sum(g * g))
        # Distribute the scalar descent across u-components so g0^T d ≈
        # descent_est (projected_armijo_scan only uses g0^T d).
        scale = descent_est / jnp.maximum(jnp.sum(du * du), 1e-30)
        g0 = scale * du
        alpha, u_new = projected_armijo_scan(
            c.u, du, lb, ub, merit, g0, alphas=alphas, c1=cfg.armijo_c1
        )
        lam_new = c.lam + alpha * dlam

        # KKT norm at the new iterate, using the *new* bind set — otherwise a
        # row that just became active reads as a large residual but is in fact
        # "bound-satisfied".
        R_n, g_n = operator.residual(u_new, lam_new, theta, ctx_iter)
        bind_l_n, bind_u_n = benson_munson_bind(
            u_new, R_n, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
        )
        R_n_pin = jnp.where(bind_l_n, u_new - lb, jnp.where(bind_u_n, u_new - ub, R_n))
        kkt = jnp.max(jnp.abs(jnp.concatenate((R_n_pin, g_n), axis=0)))

        # Persist the converged-iter bind set as int8 {-1, 0, +1} so the carry
        # can hand it back as the next step's warm-start seed.
        active_int_n = jnp.where(
            bind_l_n, jnp.int8(-1), jnp.where(bind_u_n, jnp.int8(1), jnp.int8(0))
        )

        k_next = c.k + jnp.asarray(1, jnp.int32)
        done = (kkt <= cfg.atol) | (k_next >= jnp.asarray(cfg.maxiter, jnp.int32))
        return Carry(k=k_next, u=u_new, lam=lam_new, kkt=kkt, done=done, active_set=active_int_n)

    def _cond(c: Carry):
        return jnp.logical_not(c.done)

    init = Carry(
        k=jnp.asarray(0, jnp.int32),
        u=u_init,
        lam=lam0,
        kkt=jnp.asarray(jnp.inf, dtype),
        done=jnp.asarray(False),
        active_set=active_init,
    )
    out = jax.lax.while_loop(_cond, _body, init)

    # Convert internal int8 active set -> carry-boundary bool form for
    # downstream warm-start handoff.
    active_set_final = active_set_int8_to_bool(out.active_set, cfg.constrained_indices)

    info = BoxConstrainedUnifiedInfo(
        converged=out.kkt <= jnp.asarray(cfg.atol, dtype),
        iterations=out.k,
        residual_norm=out.kkt,
        tooslow=jnp.asarray(False),
        ndecomps=out.k,
        nsolves=out.k,
        nfevals=out.k,
        npds=out.k,
        jacobian_evals=out.k,
        factorizations=out.k,
        solve_only=jnp.asarray(0, jnp.int32),
        cached_retries=jnp.asarray(0, jnp.int32),
        active_set_iters=out.k,
        rho=jnp.asarray(0.0, dtype),
        active_set_final=active_set_final,
    )
    return out.u, out.lam, info


# ---------------------------------------------------------------------------
# Reduced-space IFT custom_vjp + custom_jvp
# ---------------------------------------------------------------------------
# At converged ``(u*, lam*)`` with bind set ``B*`` the IFT system is
#
#     [A_p   B_p ] [du  ]   [ rhs_u   ]
#     [C     D   ] [dlam] = [ rhs_lam ]
#
# where ``A_p = active_set_pin(A, bind_l*, bind_u*)`` (active rows replaced by
# identity, active columns of ``B`` zeroed). Active rows of ``du`` are zero by
# construction; the inactive block solves through the same Schur factor as the
# primal.
#
# Two top-level entries (custom_vjp precludes forward-mode AD, so an outer
# custom_vjp cannot delegate to an inner custom_jvp):
#
# * ``box_constrained_solve_vjp`` — custom_vjp, for jax.grad / jax.vjp.
# * ``box_constrained_solve_jvp`` — custom_jvp, for jax.jvp.
#
# Both delegate the primal forward to ``box_constrained_solve_primal_unified``
# so the no-AD path is bit-identical between them.


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 7, 8))
def box_constrained_solve_jvp(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: BoxConstrainedUnifiedConfig,
    mass_refresh_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
):
    """Forward-mode AD entry for the unified box-constrained Newton.

    Use under ``jax.jvp``. For ``jax.grad`` / ``jax.vjp`` use
    [box_constrained_solve_vjp][diff_soup.nonlinear.box_newton.box_constrained_solve_vjp]
    instead.
    """
    return box_constrained_solve_primal_unified(
        operator, backend, analysis, u0, lam0, theta, ctx, cfg, mass_refresh_fn
    )


@box_constrained_solve_jvp.defjvp
def _solve_jvp(operator, backend, analysis, cfg, mass_refresh_fn, primals, tangents):
    """Forward-mode tangent: solve ``H_p @ [u_dot; lam_dot] = -[R_dot; g_dot]``.

    ``R_dot``, ``g_dot`` come from ``jax.jvp(operator.residual, ...)`` so the
    chain rule through theta/ctx is the standard JAX-traced one. Active rows of
    the residual tangent are zeroed (active iterates have no theta/ctx
    dependence).
    """
    u0, lam0, theta, ctx = primals
    _u0_dot, _lam0_dot, theta_dot, ctx_dot = tangents

    primal_out = box_constrained_solve_primal_unified(
        operator, backend, analysis, u0, lam0, theta, ctx, cfg, mass_refresh_fn
    )
    u_star, lam_star, info = primal_out
    n_u = u_star.shape[0]
    dtype = u_star.dtype

    lb, ub, mask = resolve_box_arrays(ctx, cfg, n_u, dtype)
    R_star, _g_star = operator.residual(u_star, lam_star, theta, ctx)
    bind_l, bind_u = benson_munson_bind(
        u_star, R_star, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
    )
    active = bind_l | bind_u
    active_int = jnp.where(bind_l, -1, jnp.where(bind_u, 1, 0))

    Avals, B, C, D, topo = operator.jacobian_blocks(u_star, lam_star, theta, ctx)
    Avals_p, B_p = _apply_active_set_to_blocks(Avals, B, topo.rows, topo.cols, active_int)

    def _residual_fn(theta_, ctx_):
        return operator.residual(u_star, lam_star, theta_, ctx_)

    _, (R_dot, g_dot) = jax.jvp(_residual_fn, (theta, ctx), (theta_dot, ctx_dot))
    R_dot_I = jnp.where(active, jnp.zeros_like(R_dot), R_dot)

    held = _factor_state_block(backend, analysis, Avals_p)
    # The tangent solves are linear in the residual tangent, so JAX can
    # transpose this rule; the border block is solved apart from it.
    u_dot, lam_dot = kkt_solve(held, B_p, C, D, -R_dot_I, -g_dot, cfg.schur_eps, linear=True)

    info_dot = jax.tree_util.tree_map(_zero_tangent_like, info)
    return primal_out, (u_dot, lam_dot, info_dot)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 7, 8))
def box_constrained_solve_vjp(
    operator: Any,
    backend: LinearSolver,
    analysis: Any,
    u0: jax.Array,
    lam0: jax.Array,
    theta: Any,
    ctx: Any,
    cfg: BoxConstrainedUnifiedConfig,
    mass_refresh_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
):
    """Reverse-mode AD entry for the unified box-constrained Newton.

    Use under ``jax.grad`` / ``jax.vjp``. For ``jax.jvp`` use
    [box_constrained_solve_jvp][diff_soup.nonlinear.box_newton.box_constrained_solve_jvp]
    instead.
    """
    return box_constrained_solve_primal_unified(
        operator, backend, analysis, u0, lam0, theta, ctx, cfg, mass_refresh_fn
    )


def _solve_fwd(operator, backend, analysis, u0, lam0, theta, ctx, cfg, mass_refresh_fn):
    """custom_vjp forward: run primal + capture converged-point linearisation.

    Signature matches the primal argument order (JAX passes all args to
    ``defvjp.fwd`` positionally, even those flagged via ``nondiff_argnums``).
    """
    primal_out = box_constrained_solve_primal_unified(
        operator, backend, analysis, u0, lam0, theta, ctx, cfg, mass_refresh_fn
    )
    u_star, lam_star, _info = primal_out
    n_u = u_star.shape[0]
    dtype = u_star.dtype

    lb, ub, mask = resolve_box_arrays(ctx, cfg, n_u, dtype)
    R_star, _g_star = operator.residual(u_star, lam_star, theta, ctx)
    bind_l, bind_u = benson_munson_bind(
        u_star, R_star, lb, ub, mask, tol_b=cfg.bound_tol, tol_F=cfg.bound_F_tol
    )
    active = bind_l | bind_u
    active_int = jnp.where(bind_l, -1, jnp.where(bind_u, 1, 0))

    Avals, B, C, D, topo = operator.jacobian_blocks(u_star, lam_star, theta, ctx)
    Avals_p, B_p = _apply_active_set_to_blocks(Avals, B, topo.rows, topo.cols, active_int)

    residuals = (u_star, lam_star, theta, ctx, active, Avals_p, B_p, C, D)
    return primal_out, residuals


def _solve_bwd(operator, backend, analysis, cfg, mass_refresh_fn, residuals, cotangent):
    """custom_vjp backward: reduced-space IFT adjoint via transposed Schur.

    Active iterate cotangents are zeroed analytically — the pinned outputs
    ``u_A = bound_A`` have no theta/ctx dependence under the static BoxBounds
    contract (exact-zero-at-active-bounds gradient property).
    ``u0`` / ``lam0`` cotangents are zero (the primal converges to a fixed point
    independent of the initial guess).
    """
    (u_star, lam_star, theta, ctx, active, Avals_p, B_p, C, D) = residuals
    du_bar, dlam_bar, _info_bar = cotangent

    du_bar_I = jnp.where(active, jnp.zeros_like(du_bar), du_bar)

    # One factor of A_p; the adjoint solves are transposed solves with it.
    held = _factor_state_block(backend, analysis, Avals_p)

    # Transposed Schur:
    #   v = A_p^{-T}(du_bar_I - C^T w)
    #   (D^T - B_p^T A_p^{-T} C^T) w = dlam_bar - B_p^T A_p^{-T} du_bar_I
    v, w = kkt_solve_transposed(held, B_p, C, D, du_bar_I, dlam_bar, cfg.schur_eps)

    # Chain rule through theta/ctx via jax.vjp on the residual function. The
    # active-row contribution drops out: pinned active rows of v are zero, and
    # the bound itself is static-config so has no theta/ctx tangent.
    def _residual_fn(theta_, ctx_):
        return operator.residual(u_star, lam_star, theta_, ctx_)

    _, residual_vjp = jax.vjp(_residual_fn, theta, ctx)
    theta_bar, ctx_bar = residual_vjp((-v, -w))

    return (
        jnp.zeros_like(u_star),
        jnp.zeros_like(lam_star),
        theta_bar,
        ctx_bar,
    )


box_constrained_solve_vjp.defvjp(_solve_fwd, _solve_bwd)


# ---------------------------------------------------------------------------
# Active-set representation converters
# ---------------------------------------------------------------------------
# The unified primal carries the active set as full-state ``int8[(n_u,)]`` in
# ``{-1, 0, +1}`` (lower / interior / upper). At carry boundaries the
# representation collapses to ``bool[(n_constrained,)]`` — "is active" on the
# constrained subset only. Roundtrip int8 -> bool -> int8 is lossy on the
# lower/upper distinction; the bool form is purely a warm-start hint.


def active_set_int8_to_bool(active_int: jax.Array, constrained_indices: tuple) -> jax.Array:
    """Internal ``int8[(n_u,)]`` -> carry-boundary ``bool[(n_constrained,)]``.

    Args:
        active_int: Full-state ``int8`` active set in ``{-1, 0, +1}``.
        constrained_indices: Tuple of DOF indices that are box-constrained.

    Returns:
        Boolean array of length ``len(constrained_indices)``: ``True`` where the
        corresponding DOF is active (lower or upper), ``False`` otherwise.
    """
    if not constrained_indices:
        return jnp.zeros((0,), dtype=jnp.bool_)
    idx = jnp.asarray(constrained_indices, dtype=jnp.int32)
    return active_int[idx] != jnp.int8(0)


def active_set_bool_to_int8(
    active_bool: jax.Array, n_u: int, constrained_indices: tuple
) -> jax.Array:
    """Carry-boundary ``bool[(n_constrained,)]`` -> internal ``int8[(n_u,)]``.

    Roundtrip-safe in the "is active" sense; the lower/upper distinction is set
    to ``-1`` (lower) by default for warm-started actives — the primal's first
    iteration re-checks against ``ub`` and flips to ``+1`` if appropriate.

    Args:
        active_bool: ``bool`` array of length ``len(constrained_indices)``.
        n_u: Total number of state DOFs.
        constrained_indices: Tuple of DOF indices that are box-constrained.

    Returns:
        Full-state ``int8`` active set in ``{-1, 0}``.
    """
    out = jnp.zeros((n_u,), dtype=jnp.int8)
    if not constrained_indices:
        return out
    idx = jnp.asarray(constrained_indices, dtype=jnp.int32)
    return out.at[idx].set(jnp.where(active_bool, jnp.int8(-1), jnp.int8(0)))
