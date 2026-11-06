"""A primal-dual interior point method with a filter line search, after Wächter and Biegler 2006."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from diff_soup.optimization.kkt import DenseKKT
from diff_soup.optimization.solvers import _filter
from diff_soup.optimization.solvers._barrier import (
    barrier_gradient,
    barrier_value,
    bound_multiplier_steps,
    bounds_from,
    dual_fraction_to_boundary,
    fraction_to_boundary,
    primal_sigma,
    push_into_bounds,
    safeguard_multipliers,
)
from diff_soup.optimization.solvers._closures import (
    Evaluation,
    bounds_of,
    closures_of,
    evaluate,
    trial,
)
from diff_soup.optimization.solvers._line_search import filter_line_search
from diff_soup.optimization.solvers._optimality import constraint_violation, optimality_error
from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
    SolveStatus,
)

KAPPA_EPS = 10.0
LAMBDA_MAX = 1e3
DIVERGED_X = 1e20
_DELTA_C = 1e-8
_KAPPA_C = 0.25


def _int(value: Any) -> Any:
    """`value` as an int32 array, the dtype of every counter in the state."""
    return jnp.asarray(value, dtype=jnp.int32)


class InteriorPointState(SolverState):
    """The carry of [InteriorPoint][diff_soup.optimization.InteriorPoint].

    Attributes:
        slacks: The inequality slacks `s >= 0`, with `c_I(x) + s = 0` at a solution.
        lam_e: The equality multipliers.
        lam_i: The inequality multipliers.
        z_lower: The lower-bound multipliers of `x`, 0 where the bound is absent.
        z_upper: The upper-bound multipliers of `x`, 0 where the bound is absent.
        v: The multipliers of `s >= 0`.
        mu: The barrier parameter.
        tau: The fraction to the boundary.
        evaluation: The objective, constraints and first derivatives at `x`.
        filt: The line-search filter.
        theta_min: The switching threshold on the constraint violation.
        theta_max: The largest constraint violation a trial may have.
        delta_w_last: The last nonzero primal regularization.
        consecutive_failed: The failed trial evaluations in a row.
        failed_evaluations: The failed trial evaluations in all.
        kkt_error: The scaled optimality error `E_0` at `x`.
    """

    slacks: Any
    lam_e: Any
    lam_i: Any
    z_lower: Any
    z_upper: Any
    v: Any
    mu: Any
    tau: Any
    evaluation: Evaluation
    filt: _filter.Filter
    theta_min: Any
    theta_max: Any
    delta_w_last: Any
    consecutive_failed: Any
    failed_evaluations: Any
    kkt_error: Any


def _error(state: InteriorPointState, mu: Any, bounds: Any) -> Any:
    """The scaled optimality error `E_mu` at the state's primal-dual point."""
    return optimality_error(
        state.evaluation,
        state.x,
        state.slacks,
        state.lam_e,
        state.lam_i,
        state.z_lower,
        state.z_upper,
        state.v,
        mu,
        bounds,
    )


def _theta(evaluation_eq: Any, evaluation_ineq: Any, s: Any) -> tuple[Any, Any]:
    """`(theta, c)`: the 1-norm of `c = [c_E; c_I + s]`, and `c`."""
    constraint = jnp.concatenate([evaluation_eq, evaluation_ineq + s])
    return jnp.sum(jnp.abs(constraint)), constraint


class InteriorPoint(CompiledSolver):
    """Interior point with slacks on the inequalities, the barrier on the slacks and the bounds, and Ipopt's filter line search.

    Args:
        tol: The scaled optimality error at which the run ends.
        max_steps: The step budget.
        mu_init: The first barrier parameter; a warm start takes a smaller one from the given point's complementarity.
        mu_min: The barrier parameter is never reduced below this.
        kappa_mu: The linear factor of the monotone barrier update.
        theta_mu: The superlinear exponent of the monotone barrier update.
        tau_min: The least fraction to the boundary.
        bound_push: How far inside its bounds the start point is pushed, absolute.
        bound_frac: How far inside its bounds the start point is pushed, relative to the bound width.
        hessian: `"auto"`, `"bfgs"`, `"exact"` or `"gauss-newton"`.
        scaling: `"gradient"` for Ipopt's gradient-based scaling at the start point, `"none"` for the problem's own.
        restoration: Whether the feasibility restoration phase runs when the line search fails.
        max_failed_evaluations: Consecutive failed trial evaluations before the run ends with `SOLVER_FAILED`.
        kkt: The [KKTSolve][diff_soup.optimization.KKTSolve] backend.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=300)
    mu_init: float = eqx.field(static=True, default=0.1)
    mu_min: float = eqx.field(static=True, default=1e-11)
    kappa_mu: float = eqx.field(static=True, default=0.2)
    theta_mu: float = eqx.field(static=True, default=1.5)
    tau_min: float = eqx.field(static=True, default=0.99)
    bound_push: float = eqx.field(static=True, default=0.01)
    bound_frac: float = eqx.field(static=True, default=0.01)
    hessian: str = eqx.field(static=True, default="auto")
    scaling: str = eqx.field(static=True, default="gradient")
    restoration: bool = eqx.field(static=True, default=True)
    max_failed_evaluations: int = eqx.field(static=True, default=10)
    kkt: Any = eqx.field(static=True, default_factory=DenseKKT)

    @property
    def _mu_floor(self) -> float:
        """The least barrier parameter: `mu_min`, or below `tol` when `tol` is tighter."""
        return min(self.mu_min, self.tol / (KAPPA_EPS + 1.0))

    def _least_squares_multipliers(self, evaluation, z_lower, z_upper, v):
        """Ipopt's least-squares constraint multipliers, 0 when larger than `LAMBDA_MAX`."""
        n_eq = evaluation.equality.shape[0]
        n_ineq = evaluation.inequality.shape[0]
        if n_eq + n_ineq == 0:
            return evaluation.equality, evaluation.inequality
        dtype = evaluation.gradient.dtype
        jacobian = jnp.concatenate([evaluation.jacobian_equality, evaluation.jacobian_inequality])
        identity = jnp.eye(evaluation.gradient.shape[0], dtype=dtype)
        dual = jnp.concatenate([jnp.zeros(n_eq, dtype), jnp.ones(n_ineq, dtype)])
        factor = self.kkt.factor(identity, jacobian, 0.0, dual)
        rhs_primal = -(evaluation.gradient - z_lower + z_upper)
        rhs_dual = jnp.concatenate([jnp.zeros(n_eq, dtype), -v])
        _, lam = self.kkt.solve_factored(factor, rhs_primal, rhs_dual)
        ok = factor.ok & jnp.all(jnp.isfinite(lam)) & (jnp.max(jnp.abs(lam)) <= LAMBDA_MAX)
        lam = jnp.where(ok, lam, 0.0)
        return lam[:n_eq], lam[n_eq:]

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """Push `x0` inside its bounds, set the slacks from the inequalities, estimate the multipliers."""
        if warm_start is not None:
            raise NotImplementedError("InteriorPoint takes warm_start from milestone A2 on.")
        closures = closures_of(problem)
        x0 = jnp.asarray(x0)
        bounds = bounds_from(*bounds_of(problem, x0))
        x = push_into_bounds(x0, bounds, self.bound_push, self.bound_frac)
        evaluation = evaluate(closures, x, args)
        dtype = x.dtype
        s = jnp.maximum(-evaluation.inequality, self.bound_push)
        z_lower = jnp.where(bounds.has_lower, 1.0, 0.0).astype(dtype)
        z_upper = jnp.where(bounds.has_upper, 1.0, 0.0).astype(dtype)
        v = jnp.ones_like(s)
        lam_e, lam_i = self._least_squares_multipliers(evaluation, z_lower, z_upper, v)
        theta, _ = _theta(evaluation.equality, evaluation.inequality, s)
        scale = jnp.maximum(1.0, theta)
        mu = jnp.asarray(self.mu_init, dtype)
        valid = (
            jnp.isfinite(evaluation.value)
            & jnp.all(jnp.isfinite(evaluation.gradient))
            & jnp.isfinite(theta)
        )
        state = InteriorPointState(
            x=x,
            step=_int(0),
            done=~valid,
            status=_int(jnp.where(valid, SolveStatus.MAX_STEPS, SolveStatus.SOLVER_FAILED)),
            slacks=s,
            lam_e=lam_e,
            lam_i=lam_i,
            z_lower=z_lower,
            z_upper=z_upper,
            v=v,
            mu=mu,
            tau=jnp.maximum(self.tau_min, 1.0 - mu),
            evaluation=evaluation,
            filt=_filter.empty_filter(dtype),
            theta_min=_filter.THETA_MIN_FACTOR * scale,
            theta_max=_filter.THETA_MAX_FACTOR * scale,
            delta_w_last=jnp.zeros((), dtype),
            consecutive_failed=_int(0),
            failed_evaluations=_int(0),
            kkt_error=optimality_error(
                evaluation, x, s, lam_e, lam_i, z_lower, z_upper, v, 0.0, bounds
            ),
        )
        return self._with_termination(problem, state, args)

    def _with_termination(self, problem, state, args):
        """`state` with `done` and `status` from `terminate`."""
        done, status = self.terminate(problem, state, args)
        return eqx.tree_at(lambda s: (s.done, s.status), state, (done, status))

    def _update_barrier(self, state: InteriorPointState, bounds) -> InteriorPointState:
        """Ipopt's monotone update: reduce `mu` while the barrier problem is solved to `kappa_eps mu`."""
        floor = self._mu_floor

        def cond(carry):
            mu, _ = carry
            return (_error(state, mu, bounds) <= KAPPA_EPS * mu) & (mu > floor)

        def body(carry):
            mu, _ = carry
            return jnp.maximum(floor, jnp.minimum(self.kappa_mu * mu, mu**self.theta_mu)), True

        mu, changed = lax.while_loop(cond, body, (state.mu, jnp.asarray(False)))
        filt = jax.tree.map(
            lambda empty, kept: jnp.where(changed, empty, kept),
            _filter.empty_filter(mu.dtype),
            state.filt,
        )
        tau = jnp.maximum(self.tau_min, 1.0 - mu)
        return eqx.tree_at(lambda s: (s.mu, s.tau, s.filt), state, (mu, tau, filt))

    def _hessian(self, closures, state: InteriorPointState, args: Any) -> Any:
        """The Lagrangian Hessian the step uses."""
        if self.hessian not in ("exact", "auto"):
            raise NotImplementedError(f"hessian={self.hessian!r} arrives at milestone A2.")
        return closures.lagrangian_hessian(state.x, state.lam_e, state.lam_i, args)

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One barrier step: the KKT solve, fraction to the boundary, the filter line search."""
        closures = closures_of(problem)
        bounds = bounds_from(*bounds_of(problem, state.x))
        state = self._update_barrier(state, bounds)
        x, s, evaluation = state.x, state.slacks, state.evaluation
        mu, tau = state.mu, state.tau
        n_eq = evaluation.equality.shape[0]
        dtype = x.dtype

        jacobian = jnp.concatenate([evaluation.jacobian_equality, evaluation.jacobian_inequality])
        lam = jnp.concatenate([state.lam_e, state.lam_i])
        barrier_x, barrier_s = barrier_gradient(x, s, mu, bounds)
        grad_phi_x = evaluation.gradient + barrier_x
        r_x = -(grad_phi_x + jacobian.T @ lam)
        r_s = -(barrier_s + state.lam_i)
        s_over_v = s / state.v
        theta, constraint = _theta(evaluation.equality, evaluation.inequality, s)
        phi = evaluation.value + barrier_value(x, s, mu, bounds)

        primal_block = self._hessian(closures, state, args) + jnp.diag(
            primal_sigma(x, state.z_lower, state.z_upper, bounds)
        )
        dual_diagonal = jnp.concatenate([jnp.zeros(n_eq, dtype), s_over_v])
        factor, delta_w = self.kkt.factor_regularized(
            primal_block, jacobian, dual_diagonal, _DELTA_C * mu**_KAPPA_C, state.delta_w_last
        )
        slack_rhs = jnp.concatenate([jnp.zeros(n_eq, dtype), r_s * s_over_v])

        def direction(constraint_rhs):
            dx, dlam = self.kkt.solve_factored(factor, r_x, -constraint_rhs - slack_rhs)
            return dx, (r_s - dlam[n_eq:]) * s_over_v, dlam

        def trial_point(x_trial, s_trial):
            value, equality, inequality, valid = trial(closures, x_trial, args)
            theta_trial, constraint_trial = _theta(equality, inequality, s_trial)
            phi_trial = value + barrier_value(x_trial, s_trial, mu, bounds)
            return theta_trial, phi_trial, constraint_trial, valid

        dx, ds, dlam = direction(constraint)
        alpha_max = fraction_to_boundary(x, dx, s, ds, tau, bounds)
        slope = grad_phi_x @ dx + barrier_s @ ds
        eps = jnp.finfo(dtype).eps
        tiny = (
            jnp.maximum(
                jnp.max(jnp.abs(dx) / (1.0 + jnp.abs(x)), initial=0.0),
                jnp.max(jnp.abs(ds) / (1.0 + jnp.abs(s)), initial=0.0),
            )
            < 10.0 * eps
        )
        search = filter_line_search(
            trial_point,
            direction,
            x,
            s,
            dx,
            ds,
            dlam,
            alpha_max=alpha_max,
            theta=theta,
            phi=phi,
            slope=slope,
            constraint=constraint,
            filt=state.filt,
            theta_min=state.theta_min,
            theta_max=state.theta_max,
            tau=tau,
            bounds=bounds,
            consecutive_failed=state.consecutive_failed,
            max_failed=self.max_failed_evaluations,
            accept_full=tiny & factor.ok,
        )

        alpha = search.alpha
        x_new = x + alpha * search.dx
        s_new = s + alpha * search.ds
        lam_new = lam + alpha * search.dlam
        dz_lower, dz_upper, dv = bound_multiplier_steps(
            x, search.dx, s, search.ds, state.z_lower, state.z_upper, state.v, mu, bounds
        )
        alpha_dual = dual_fraction_to_boundary(
            state.z_lower, dz_lower, state.z_upper, dz_upper, state.v, dv, tau, bounds
        )
        z_lower, z_upper, v = safeguard_multipliers(
            x_new,
            s_new,
            state.z_lower + alpha_dual * dz_lower,
            state.z_upper + alpha_dual * dz_upper,
            state.v + alpha_dual * dv,
            mu,
            bounds,
        )
        evaluation_new = evaluate(closures, x_new, args)
        filt = jax.tree.map(
            lambda kept, grown: jnp.where(search.f_type, kept, grown),
            state.filt,
            _filter.augment(
                state.filt,
                (1.0 - _filter.GAMMA_THETA) * theta,
                phi - _filter.GAMMA_PHI * theta,
            ),
        )
        accepted = search.accepted & factor.ok
        moved = eqx.tree_at(
            lambda t: (
                t.x,
                t.slacks,
                t.lam_e,
                t.lam_i,
                t.z_lower,
                t.z_upper,
                t.v,
                t.evaluation,
                t.filt,
            ),
            state,
            (
                x_new,
                s_new,
                lam_new[:n_eq],
                lam_new[n_eq:],
                z_lower,
                z_upper,
                v,
                evaluation_new,
                filt,
            ),
        )
        state = jax.tree.map(lambda new, old: jnp.where(accepted, new, old), moved, state)
        exhausted = search.consecutive_failed >= self.max_failed_evaluations
        failure = jnp.where(exhausted, SolveStatus.SOLVER_FAILED, SolveStatus.RESTORATION_FAILED)
        kkt_error = _error(state, 0.0, bounds)
        return eqx.tree_at(
            lambda t: (
                t.step,
                t.done,
                t.status,
                t.delta_w_last,
                t.consecutive_failed,
                t.failed_evaluations,
                t.kkt_error,
            ),
            state,
            (
                _int(state.step + 1),
                state.done | ~accepted,
                _int(jnp.where(accepted, state.status, failure)),
                jnp.where(delta_w > 0.0, delta_w, state.delta_w_last),
                _int(search.consecutive_failed),
                _int(state.failed_evaluations + search.failed),
                kkt_error,
            ),
        )

    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """End on the scaled optimality error, the budget, or a failure."""
        converged = state.kkt_error <= self.tol
        finite = (
            jnp.all(jnp.isfinite(state.x))
            & jnp.all(jnp.isfinite(state.lam_e))
            & jnp.all(jnp.isfinite(state.lam_i))
            & jnp.isfinite(state.kkt_error)
        )
        diverged = ~finite | (jnp.max(jnp.abs(state.x), initial=0.0) > DIVERGED_X)
        over = state.step >= self.max_steps
        status = jnp.where(
            state.done,
            state.status,
            jnp.where(
                converged,
                SolveStatus.KKT_TOL,
                jnp.where(diverged, SolveStatus.DIVERGED, SolveStatus.MAX_STEPS),
            ),
        )
        return state.done | converged | diverged | over, _int(status)

    def run(
        self,
        problem: Any,
        x0: Any,
        args: Any,
        *,
        max_steps: int | None = None,
        warm_start: Any = None,
    ) -> tuple[Any, SolveRecord]:
        """The compiled loop; see [CompiledSolver.run][diff_soup.optimization.CompiledSolver.run]."""
        solver = self if max_steps is None else dataclasses.replace(self, max_steps=max_steps)
        state = solver.init(problem, x0, args, warm_start=warm_start)

        def body(current):
            new = solver._with_termination(problem, solver.step(problem, current, args), args)
            return jax.tree.map(
                lambda kept, moved: jnp.where(current.done, kept, moved), current, new
            )

        state = lax.while_loop(lambda current: ~current.done, body, state)
        return state.x, _record(state)


def _record(state: InteriorPointState) -> SolveRecord:
    """The `SolveRecord` of a finished state."""
    return SolveRecord(
        multipliers_equality=state.lam_e,
        multipliers_inequality=state.lam_i,
        slacks=state.slacks,
        bound_multipliers=(state.z_lower, state.z_upper),
        mu=state.mu,
        steps=state.step,
        failed_evaluations=state.failed_evaluations,
        kkt_error=state.kkt_error,
        constraint_violation=constraint_violation(state.evaluation),
        status=state.status,
        degenerate=jnp.asarray(False),
    )
