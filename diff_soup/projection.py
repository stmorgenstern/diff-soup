"""Full-state algebraic projection onto a DAE's constraint manifold.

``project_algebraic`` takes a full canonical state (fields followed by
globals), holds every differential field and differential global exactly at
its input value, and solves every algebraic field and algebraic global
together with Newton so the returned state satisfies the algebraic residual
rows at the requested tolerances. Typical inputs are interpolated states
between accepted solver steps, where the local interpolant satisfies the
constraints only to interpolation accuracy.

The solve is JAX-safe end to end: failures are reported through the traced
integer ``status`` (branch on it with ``jnp.where`` / ``lax.cond``; vmap
lanes may fail independently), while eager host code calls
``AlgebraicProjectionResult.require_success()`` for a typed exception.
Derivatives come from paired implicit-function rules linearized at the
converged root — never from differentiating the Newton iterations — with
``AlgebraicProjectionOptions.ad_mode`` selecting the forward or reverse
entry.

This root module is orchestration wiring (like ``pdesolve``): it may touch
the problem layer and the solver layers.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import zero_from_primal

from diff_soup._errors import ConfigurationError, DiffSoupError
from diff_soup.linear import Dense, analyze
from diff_soup.nonlinear._kkt import kkt_solve, kkt_solve_transposed
from diff_soup.nonlinear.solver import (
    NonlinearConfig,
    _factor_state_block,
    nonlinear_solve_primal,
)
from diff_soup.problem.implicit.consistent_init import resolve_backend_route
from diff_soup.problem.implicit.problems import MassProblem
from diff_soup.problem.implicit.solver_problem import get_or_build_solver_view
from diff_soup.timestepping.consistent_ic.pin_differential import (
    ContractOperator,
    PinnedOperator,
    _ic_topology,
)


class AlgebraicProjectionStatus(enum.IntEnum):
    """Outcome code of [`project_algebraic`][diff_soup.project_algebraic], one per state when batched with `vmap`.

    Attributes:
        SUCCESS: The Newton iteration converged.
        MAX_ITERATIONS: The iteration did not converge within `max_steps`.
        SINGULAR_JACOBIAN: The inputs were finite but the solve produced a non-finite value.
        NONFINITE: The time, the kept state entries, the algebraic seed, or `state_dot` was non-finite.
    """

    SUCCESS = 0
    MAX_ITERATIONS = 1
    SINGULAR_JACOBIAN = 2
    NONFINITE = 3


_SUCCESS = int(AlgebraicProjectionStatus.SUCCESS)
_MAX_ITERATIONS = int(AlgebraicProjectionStatus.MAX_ITERATIONS)
_SINGULAR_JACOBIAN = int(AlgebraicProjectionStatus.SINGULAR_JACOBIAN)
_NONFINITE = int(AlgebraicProjectionStatus.NONFINITE)

_LINEAR_SOLVERS = ("auto", "dense", "klu", "superlu", "cudss")
_AD_MODES = ("forward", "reverse")


@dataclasses.dataclass(frozen=True)
class AlgebraicProjectionOptions:
    """Settings of the Newton solve in [`project_algebraic`][diff_soup.project_algebraic].

    Every field is checked on construction, and an out-of-range value raises `ValueError`.

    Attributes:
        rtol: Relative tolerance of the convergence test, positive.
        atol: Absolute tolerance of the convergence test, positive.
        max_steps: Maximum number of Newton iterations, at least 1.
        linear_solver: Linear solver of each Newton step, one of `"auto"`, `"dense"`, `"klu"`, `"superlu"`, `"cudss"`.
        ad_mode: `"forward"` differentiates in forward and reverse mode, `"reverse"` in reverse mode only.
    """

    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 10
    linear_solver: str = "auto"
    ad_mode: str = "forward"

    def __post_init__(self) -> None:
        """Validate every field eagerly at construction."""
        if not (float(self.rtol) > 0.0):
            raise ValueError(f"rtol must be positive; got {self.rtol!r}.")
        if not (float(self.atol) > 0.0):
            raise ValueError(f"atol must be positive; got {self.atol!r}.")
        if int(self.max_steps) < 1:
            raise ValueError(f"max_steps must be at least 1; got {self.max_steps!r}.")
        if self.linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"linear_solver must be one of {_LINEAR_SOLVERS}; got {self.linear_solver!r}."
            )
        if self.ad_mode not in _AD_MODES:
            raise ValueError(f"ad_mode must be one of {_AD_MODES}; got {self.ad_mode!r}.")


class AlgebraicProjectionError(DiffSoupError):
    """Raised by `AlgebraicProjectionResult.require_success()` when a projection did not succeed.

    Attributes:
        status: The [`AlgebraicProjectionStatus`][diff_soup.AlgebraicProjectionStatus] codes, as a NumPy array.
        iterations: Newton iterations taken, as a NumPy array.
        residual_norm: Final residual norm, as a NumPy array.
    """

    def __init__(self, message: str, *, status, iterations, residual_norm) -> None:
        """Store the failure diagnostics alongside the message."""
        super().__init__(message)
        self.status = status
        self.iterations = iterations
        self.residual_norm = residual_norm


class AlgebraicProjectionResult(eqx.Module):
    """Result of [`project_algebraic`][diff_soup.project_algebraic].

    A failed projection does not raise. Branch on `status` in traced code, or call
    `require_success()` on a concrete result, which raises
    [`AlgebraicProjectionError`][diff_soup.AlgebraicProjectionError] on failure.

    Attributes:
        state: Full state with the differential entries kept and the algebraic entries solved.
        status: Integer [`AlgebraicProjectionStatus`][diff_soup.AlgebraicProjectionStatus] code.
        iterations: Number of Newton iterations taken.
        residual_norm: Norm of the final Newton residual.
    """

    state: jax.Array
    status: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array

    def require_success(self) -> "AlgebraicProjectionResult":
        """Raise :class:`AlgebraicProjectionError` unless every status is SUCCESS.

        Host-only: call it on concrete results. Inside ``jit``/``vmap``/AD
        tracing there is no concrete status to inspect — branch on ``status``
        with ``jnp.where`` / ``lax.cond`` there instead.
        """
        if isinstance(self.status, jax.core.Tracer):
            raise TypeError(
                "require_success() reads a concrete status and is host-only; this "
                "result is traced (inside jit/vmap/AD). Branch on the integer "
                "`status` leaf with jnp.where or lax.cond in traced code, and call "
                "require_success() only on concrete results."
            )
        status = np.asarray(self.status)
        if np.all(status == _SUCCESS):
            return self
        iterations = np.asarray(self.iterations)
        residual_norm = np.asarray(self.residual_norm)
        if status.ndim == 0:
            name = AlgebraicProjectionStatus(int(status)).name
            detail = (
                f"algebraic projection failed with status {name} after "
                f"{int(iterations)} iteration(s); final residual norm "
                f"{float(residual_norm):.6e}."
            )
        else:
            failing = status[status != _SUCCESS]
            names = sorted({AlgebraicProjectionStatus(int(s)).name for s in failing})
            detail = (
                f"algebraic projection failed in {failing.size} of {status.size} "
                f"lane(s) with status(es) {', '.join(names)}; worst residual norm "
                f"{float(np.nanmax(residual_norm)):.6e}."
            )
        raise AlgebraicProjectionError(
            detail + " Improve the algebraic seed, loosen the tolerances (rtol/atol), or "
            "raise max_steps in AlgebraicProjectionOptions.",
            status=status,
            iterations=iterations,
            residual_norm=residual_norm,
        )


class _ProjectionContext(NamedTuple):
    """Traced side-channel the projection residual reads.

    Duck-compatible with the consistent-initialization operator contract:
    ``t0``/``ydot`` feed the canonical residual, ``u_pin``/``lam_pin`` are the
    held-row targets of the pinned operator.
    """

    t0: Any
    ydot: Any
    u_pin: Any
    lam_pin: Any


class _ProjectionOperator(ContractOperator):
    """Canonical-problem adapter threading the explicit control channel.

    The corrector's parameter slot carries ``(theta, control)`` so runtime
    control values reach the residual and the exact Jacobian alongside the
    parameter pytree.
    """

    def residual(self, u, lam, theta, ctx):
        """Return ``(R, g)`` = the canonical residual split at ``n_u``."""
        runtime, control = theta
        y = jnp.concatenate([u, lam])
        resid = self.problem.residual(ctx.t0, y, ctx.ydot, runtime, control)
        return resid[: self.n_u], resid[self.n_u :]

    def jacobian_blocks(self, u, lam, theta, ctx):
        """Return ``(A_values, B, C, D, topology)`` at the projection's fixed derivative."""
        runtime, control = theta
        y = jnp.concatenate([u, lam])
        a, b, c, d = self.problem.state_jacobian_blocks(
            ctx.t0, y, ctx.ydot, runtime, self.cj, control=control
        )
        a_values = jnp.reshape(a, (-1,)) if jnp.ndim(a) == 2 else jnp.asarray(a)
        return a_values, b, c, d, self.topology


class _ProjectionPlan:
    """Static machinery for one (canonical view, options) projection family."""

    __slots__ = (
        "view",
        "options",
        "n_u",
        "n_lam",
        "n",
        "diff_u",
        "diff_lam",
        "mask_full",
        "needs_state_dot",
        "operator",
        "backend",
        "analysis",
        "ncfg",
    )

    def __init__(self, view, options: AlgebraicProjectionOptions) -> None:
        """Build the pinned operator, the linear route and the Newton settings.

        Args:
            view: The solver view of the problem.
            options: The projection options (tolerances, step budget, linear
                solver, derivative mode).
        """
        self.view = view
        self.options = options
        self.n_u = int(view.n_u)
        self.n_lam = int(view.n_lam)
        self.n = self.n_u + self.n_lam
        id_arr = np.asarray(view.id_mask).astype(bool)
        self.diff_u = id_arr[: self.n_u]
        self.diff_lam = id_arr[self.n_u :]
        self.mask_full = id_arr
        self.needs_state_dot = _state_dot_required(view)
        topology = _ic_topology(view, self.n_u)
        base = _ProjectionOperator(
            problem=view,
            n_u=self.n_u,
            n_lam=self.n_lam,
            topology=topology,
            cj=0.0,
        )
        lam_pin_mask = self.diff_lam if bool(self.diff_lam.any()) else None
        self.operator = PinnedOperator(base=base, pin_mask=self.diff_u, lam_pin_mask=lam_pin_mask)
        _route, instance = resolve_backend_route(
            options.linear_solver, plan_mode=view.plan.mode, n_dofs=self.n_u
        )
        self.backend = Dense() if instance is None else instance
        self.analysis = analyze(
            topology,
            backend=self.backend,
            precision="fp64" if jax.config.read("jax_enable_x64") else "fp32",
        )
        # Fresh Jacobian every iteration: a projection is a standalone root
        # solve from an arbitrary interpolated seed, so quadratic convergence
        # and robustness beat the engines' chord-factor reuse economics.
        self.ncfg = NonlinearConfig(
            maxiter=int(options.max_steps),
            rtol=float(options.rtol),
            atol=float(options.atol),
            jacobian_cache="none",
            newton_rate_guard=False,
        )


def _state_dot_required(view) -> bool:
    """Whether the declared derivative structure lets algebraic rows read ``ydot``.

    A mass-form canonical problem proves absence when its derivative operator
    has no structural entry in any algebraic residual row; a fully implicit
    canonical residual cannot prove absence and always requires the derivative.
    """
    canonical = view.problem
    if not isinstance(canonical, MassProblem):
        return True
    rows = np.asarray(canonical.mass.coefficient_sparsity.rows)
    if rows.size == 0:
        return False
    algebraic = ~np.asarray(view.id_mask).astype(bool)
    return bool(np.any(algebraic[rows]))


def _projection_plan(view, options: AlgebraicProjectionOptions) -> _ProjectionPlan:
    """One cached plan per (view, options); options are the structural key."""
    cache = view.__dict__.setdefault("_projection_plans", {})
    plan = cache.get(options)
    if plan is None:
        plan = _ProjectionPlan(view, options)
        cache[options] = plan
    return plan


def _project_primal(plan, t, state, state_dot, theta, control, seed):
    """Run the pinned Newton solve and classify the traced status."""
    n_u, n_lam = plan.n_u, plan.n_lam
    mask_full = jnp.asarray(plan.mask_full)
    diff_u = jnp.asarray(plan.diff_u)
    y0 = jnp.where(mask_full, state, seed)
    ctx = _ProjectionContext(t0=t, ydot=state_dot, u_pin=state[:n_u], lam_pin=state[n_u:])
    u_star, lam_star, info = nonlinear_solve_primal(
        plan.operator,
        plan.backend,
        plan.analysis,
        y0[:n_u],
        y0[n_u:],
        (theta, control),
        ctx,
        plan.ncfg,
        early_exit=False,
    )
    u_out = jnp.where(diff_u, state[:n_u], u_star)
    if n_lam:
        lam_out = jnp.where(jnp.asarray(plan.diff_lam), state[n_u:], lam_star)
    else:
        lam_out = lam_star
    y_out = jnp.concatenate([u_out, lam_out])

    # Status classification consumes exactly what the solve consumed: the
    # differential entries of ``state``, the algebraic entries of ``seed``,
    # the fixed derivative, and the time.
    pins = jnp.where(mask_full, state, 0.0)
    seed_alg = jnp.where(mask_full, 0.0, seed)
    inputs_finite = (
        jnp.all(jnp.isfinite(pins))
        & jnp.all(jnp.isfinite(seed_alg))
        & jnp.all(jnp.isfinite(state_dot))
        & jnp.isfinite(t)
    )
    result_finite = (
        jnp.all(jnp.isfinite(u_star))
        & jnp.all(jnp.isfinite(lam_star))
        & jnp.isfinite(info.residual_norm)
    )
    status = jnp.where(
        ~inputs_finite,
        _NONFINITE,
        jnp.where(
            ~result_finite,
            _SINGULAR_JACOBIAN,
            jnp.where(info.converged, _SUCCESS, _MAX_ITERATIONS),
        ),
    ).astype(jnp.int32)
    iterations = jnp.asarray(info.iterations, dtype=jnp.int32)
    residual_norm = jnp.asarray(info.residual_norm, dtype=state.dtype)
    return y_out, status, iterations, residual_norm


def _solved_rows_fn(plan, y_star):
    """The solved-row residual as a function of every differentiable input.

    Pinned rows are zeroed (their identity residual carries no driving term;
    their derivative is the pass-through handled by the rules); the algebraic
    unknowns are held at the converged root. The root is not cut from its own
    derivative: under a second differentiation it carries the tangent the
    rule gave it, so the linearization point moves with the inputs.
    """
    mask_full = jnp.asarray(plan.mask_full)
    diff_u = jnp.asarray(plan.diff_u)
    y_root = y_star

    def solved_rows(t, state_dot, theta, control, pins):
        y = jnp.where(mask_full, pins, y_root)
        resid = plan.view.residual(t, y, state_dot, theta, control)
        r = jnp.where(diff_u, 0.0, resid[: plan.n_u])
        g = resid[plan.n_u :]
        if plan.n_lam and bool(plan.diff_lam.any()):
            g = jnp.where(jnp.asarray(plan.diff_lam), 0.0, g)
        return r, g

    return solved_rows


def _pinned_blocks_at_root(plan, y_star, t, state, state_dot, theta, control):
    """The pinned system's exact Jacobian blocks at the converged root."""
    ctx = _ProjectionContext(
        t0=t, ydot=state_dot, u_pin=state[: plan.n_u], lam_pin=state[plan.n_u :]
    )
    a_values, b, c, d, _topo = plan.operator.jacobian_blocks(
        y_star[: plan.n_u], y_star[plan.n_u :], (theta, control), ctx
    )
    return a_values, b, c, d


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _project_forward(plan, t, state, state_dot, theta, control, seed):
    """Projection primal with the forward-mode implicit rule attached."""
    return _project_primal(plan, t, state, state_dot, theta, control, seed)


@_project_forward.defjvp
def _project_forward_jvp(plan, primals, tangents):
    """Implicit-function tangent at the returned root; the seed is tangent-dead."""
    t, state, state_dot, theta, control, seed = primals
    dt, dstate, dstate_dot, dtheta, dcontrol, _dseed = tangents
    # The primal through the entry itself: differentiating this rule again
    # differentiates the root by this rule, never through the Newton loop.
    out = _project_forward(plan, t, state, state_dot, theta, control, seed)
    y_star, status, iterations, residual_norm = out

    solved_rows = _solved_rows_fn(plan, y_star)
    _res, (r_dot, g_dot) = jax.jvp(
        solved_rows,
        (t, state_dot, theta, control, state),
        (dt, dstate_dot, dtheta, dcontrol, dstate),
    )
    a_values, b, c, d = _pinned_blocks_at_root(plan, y_star, t, state, state_dot, theta, control)
    held = _factor_state_block(plan.backend, plan.analysis, a_values)
    du, dlam = kkt_solve(held, b, c, d, -r_dot, -g_dot, plan.ncfg.schur_eps, linear=True)
    mask_full = jnp.asarray(plan.mask_full)
    dy = jnp.where(mask_full, dstate, jnp.concatenate([du, dlam]))
    return out, (
        dy,
        zero_from_primal(status, symbolic_zeros=False),
        zero_from_primal(iterations, symbolic_zeros=False),
        jnp.zeros_like(residual_norm),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _project_reverse(plan, t, state, state_dot, theta, control, seed):
    """Projection primal with the reverse-mode implicit rule attached."""
    return _project_primal(plan, t, state, state_dot, theta, control, seed)


def _project_reverse_fwd(plan, t, state, state_dot, theta, control, seed):
    """Run the primal once; save the root with the inputs it was solved from.

    The primal runs through the forward entry, so a forward differentiation
    of this reverse rule (``jax.hessian``) differentiates the root by the
    implicit-function rule.
    """
    out = _project_forward(plan, t, state, state_dot, theta, control, seed)
    return out, (out[0], t, state, state_dot, theta, control, seed)


def _project_reverse_bwd(plan, saved, cotangents):
    """Transposed implicit-function cotangent at the saved root."""
    y_star, t, state, state_dot, theta, control, seed = saved
    ct_y, _ct_status, _ct_iterations, _ct_residual = cotangents
    n_u, n_lam = plan.n_u, plan.n_lam
    mask_full = jnp.asarray(plan.mask_full)

    ct_solved = jnp.where(mask_full, 0.0, ct_y)
    a_values, b, c, d = _pinned_blocks_at_root(plan, y_star, t, state, state_dot, theta, control)
    held = _factor_state_block(plan.backend, plan.analysis, a_values)
    v, w = kkt_solve_transposed(
        held,
        b,
        c,
        d,
        ct_solved[:n_u],
        ct_solved[n_u:],
        plan.ncfg.schur_eps,
    )
    v = jnp.where(jnp.asarray(plan.diff_u), 0.0, v)
    if n_lam and bool(plan.diff_lam.any()):
        w = jnp.where(jnp.asarray(plan.diff_lam), 0.0, w)

    solved_rows = _solved_rows_fn(plan, y_star)
    _res, vjp_fn = jax.vjp(solved_rows, t, state_dot, theta, control, state)
    grad_t, grad_state_dot, grad_theta, grad_control, grad_pins = vjp_fn((-v, -w))
    grad_state = grad_pins + jnp.where(mask_full, ct_y, 0.0)
    # The seed is a guess: the converged root, hence the loss, never depends on it.
    grad_seed = jnp.zeros_like(seed)
    return (grad_t, grad_state, grad_state_dot, grad_theta, grad_control, grad_seed)


_project_reverse.defvjp(_project_reverse_fwd, _project_reverse_bwd)


def _pinned_kkt_linear_solve(plan, a_values, b, c, d, rhs_u, rhs_lam):
    """Transpose-safe solve of the pinned bordered system through the backend.

    Solves ``[[A, B], [C, D]] [du; dlam] = [rhs_u; rhs_lam]`` — the pinned
    Jacobian blocks at a converged root — with the plan's linear backend
    (one state-block factorization + Schur elimination). The solves are
    linear in ``(rhs_u, rhs_lam)``, so reverse-mode transposition turns
    them into transposed solves with the same factor; memory and work
    follow the declared state pattern plus the bordered blocks, never a
    dense ``(n, n)`` matrix.

    The returned solution passes through ``lax.optimization_barrier``, whose
    transpose materializes symbolic-zero cotangents before they reach the
    transposed solves: a caller whose cotangent for the projected state is
    identically zero (an event projection consumed only by diagnostics)
    transposes without special cases.
    """
    held = _factor_state_block(plan.backend, plan.analysis, a_values)
    du, dlam = kkt_solve(held, b, c, d, rhs_u, rhs_lam, plan.ncfg.schur_eps, linear=True)
    return jax.lax.optimization_barrier(jnp.concatenate([du, dlam]))


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _project_bidirectional(plan, t, state, state_dot, theta, control, seed):
    """Projection entry with a transpose-safe implicit tangent.

    Projected event roots need a time-direction JVP while locating the root
    and a VJP when the surrounding solve uses a checkpointed adjoint. The
    public forward and reverse entries intentionally specialize for one
    direction each; this internal entry solves the pinned system through the
    plan's backend with solves linear in the right-hand side, so one implicit
    rule serves both directions while respecting the declared sparsity.
    """
    return _project_primal(plan, t, state, state_dot, theta, control, seed)


@_project_bidirectional.defjvp
def _project_bidirectional_jvp(plan, primals, tangents):
    """Exact implicit tangent through a transpose-safe sparse KKT solve."""
    t, state, state_dot, theta, control, seed = primals
    dt, dstate, dstate_dot, dtheta, dcontrol, _dseed = tangents
    # The primal through the entry itself, as in the forward rule.
    out = _project_bidirectional(plan, t, state, state_dot, theta, control, seed)
    y_star, status, iterations, residual_norm = out

    solved_rows = _solved_rows_fn(plan, y_star)
    _res, (r_dot, g_dot) = jax.jvp(
        solved_rows,
        (t, state_dot, theta, control, state),
        (dt, dstate_dot, dtheta, dcontrol, dstate),
    )
    a_values, b, c, d = _pinned_blocks_at_root(plan, y_star, t, state, state_dot, theta, control)
    dy_solved = _pinned_kkt_linear_solve(plan, a_values, b, c, d, -r_dot, -g_dot)
    mask_full = jnp.asarray(plan.mask_full)
    dy = jnp.where(mask_full, dstate, dy_solved)
    return out, (
        dy,
        zero_from_primal(status, symbolic_zeros=False),
        zero_from_primal(iterations, symbolic_zeros=False),
        jnp.zeros_like(residual_norm),
    )


def _project_algebraic_bidirectional(
    problem,
    *,
    t,
    state,
    seed,
    state_dot,
    theta,
    control,
    options,
) -> AlgebraicProjectionResult:
    """Internal event-root projection supporting both AD directions."""
    view = get_or_build_solver_view(problem)
    plan = _projection_plan(view, options)
    state = jnp.asarray(state)
    dtype = state.dtype
    seed = jnp.asarray(seed, dtype=dtype)
    state_dot = jnp.asarray(state_dot, dtype=dtype)
    if theta is None:
        theta = view.default_theta()
    control = view.normalize_control(control, dtype=dtype)
    t = jnp.asarray(t, dtype=dtype)
    y_out, status, iterations, residual_norm = _project_bidirectional(
        plan, t, state, state_dot, theta, control, seed
    )
    return AlgebraicProjectionResult(
        state=y_out,
        status=status,
        iterations=iterations,
        residual_norm=residual_norm,
    )


def project_algebraic(
    problem,
    *,
    t,
    state,
    seed,
    state_dot=None,
    theta=None,
    control=None,
    options: Optional[AlgebraicProjectionOptions] = None,
) -> AlgebraicProjectionResult:
    """Solve the algebraic entries of a state at one time, keeping its differential entries fixed.

    Use it on a state assembled or edited outside a solve. The result is
    differentiable in `t`, `state`, `state_dot`, `theta` and `control`, and its
    derivative in `seed` is zero.

    Args:
        problem: A [`MassProblem`][diff_soup.problem.MassProblem], an [`ImplicitProblem`][diff_soup.problem.ImplicitProblem], or a problem with a `solver_problem()` method such as a [`LoweredProblem`][diff_soup.problem.LoweredProblem].
        t: Time at which the residual is evaluated.
        state: Full state vector, fields then globals, whose differential entries are kept.
        seed: Full vector whose algebraic entries start the Newton iteration, with its other entries ignored.
        state_dot: Full time derivative, required for an `ImplicitProblem` or when an algebraic row reads a derivative, zero otherwise.
        theta: Parameters, or `None` for the problem's `default_theta()`.
        control: Control values or an operating [`ControlState`][diff_soup.ControlState], as in a solve.
        options: Solver settings, or `None` for the defaults of [`AlgebraicProjectionOptions`][diff_soup.AlgebraicProjectionOptions].

    Returns:
        An [`AlgebraicProjectionResult`][diff_soup.AlgebraicProjectionResult].

    Raises:
        ConfigurationError: If a vector has the wrong length, or `state_dot` is missing where it is required.
    """
    view = get_or_build_solver_view(problem)
    if options is None:
        options = AlgebraicProjectionOptions()
    plan = _projection_plan(view, options)
    n = plan.n

    state = jnp.asarray(state)
    dtype = state.dtype
    if state.ndim != 1 or state.shape[0] != n:
        raise ConfigurationError(
            f"state must be the full canonical vector of length {n} "
            f"(fields then globals); got shape {tuple(state.shape)}."
        )
    seed = jnp.asarray(seed, dtype=dtype)
    if seed.ndim != 1 or seed.shape[0] != n:
        raise ConfigurationError(
            f"seed must be the full canonical vector of length {n} "
            f"(fields then globals); got shape {tuple(seed.shape)}."
        )
    if state_dot is None:
        if plan.needs_state_dot:
            raise ConfigurationError(
                "state_dot is required for this problem: its declared derivative "
                "structure lets algebraic residual rows read differential "
                "derivatives (or the canonical residual is fully implicit, where "
                "absence cannot be proven). Pass the full canonical state_dot "
                "vector, for example the solver interpolant's derivative."
            )
        state_dot = jnp.zeros((n,), dtype=dtype)
    else:
        state_dot = jnp.asarray(state_dot, dtype=dtype)
        if state_dot.ndim != 1 or state_dot.shape[0] != n:
            raise ConfigurationError(
                f"state_dot must be the full canonical vector of length {n} "
                f"(fields then globals); got shape {tuple(state_dot.shape)}."
            )
    if theta is None:
        theta = view.default_theta()
    control = view.normalize_control(control, dtype=dtype)
    t = jnp.asarray(t, dtype=dtype)

    entry = _project_forward if options.ad_mode == "forward" else _project_reverse
    y_out, status, iterations, residual_norm = entry(
        plan, t, state, state_dot, theta, control, seed
    )
    return AlgebraicProjectionResult(
        state=y_out,
        status=status,
        iterations=iterations,
        residual_norm=residual_norm,
    )


__all__ = [
    "AlgebraicProjectionError",
    "AlgebraicProjectionOptions",
    "AlgebraicProjectionResult",
    "AlgebraicProjectionStatus",
    "project_algebraic",
]
