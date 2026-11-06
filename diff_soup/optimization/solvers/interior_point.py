"""A primal-dual interior point method with a filter line search, after Wächter and Biegler 2006."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import DenseKKT
from diff_soup.optimization.solvers import _filter, _hessian, _restoration
from diff_soup.optimization.solvers._barrier import (
    barrier_gradient,
    barrier_value,
    bound_multiplier_steps,
    bounds_from,
    distances,
    dual_fraction_to_boundary,
    fraction_to_boundary,
    primal_sigma,
    push_into_bounds,
    safeguard_multipliers,
)
from diff_soup.optimization.solvers._closures import (
    Evaluation,
    Scales,
    bounds_of,
    closures_of,
    evaluate,
    problem_scales,
    scaled_closures,
    trial,
    unit_scales,
)
from diff_soup.optimization.solvers._implicit import active_set, implicit_run
from diff_soup.optimization.solvers._line_search import filter_line_search
from diff_soup.optimization.solvers._optimality import constraint_violation, optimality_error
from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
)
from diff_soup.utils.lanes import live_lane_cond

KAPPA_EPS = 10.0
LAMBDA_MAX = 1e3
DIVERGED_X = 1e20
WARM_PUSH = 1e-6
BOUND_MULTIPLIER_RESET = 1e3
_DELTA_C = 1e-8
_KAPPA_C = 0.25


def _int(value: Any) -> Any:
    """`value` as an int32 array, the dtype of every counter in the state."""
    return jnp.asarray(value, dtype=jnp.int32)


class InteriorPointState(SolverState):
    """The carry of [InteriorPoint][diff_soup.optimization.InteriorPoint], in the scaled problem.

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
        scales: The objective and constraint scales.
        quasi_newton: The BFGS matrix, `(n, n)`, or `(0, 0)` when not used.
        quasi_newton_updates: How many updates the BFGS matrix has had.
        restoration_steps: The steps taken in restoration phases.
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
    scales: Scales
    quasi_newton: Any
    quasi_newton_updates: Any
    restoration_steps: Any


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


def _theta(equality: Any, inequality: Any, s: Any) -> tuple[Any, Any]:
    """`(theta, c)`: the 1-norm of `c = [c_E; c_I + s]`, and `c`."""
    constraint = jnp.concatenate([equality, inequality + s])
    return jnp.sum(jnp.abs(constraint)), constraint


def _select(pred: Any, new: Any, old: Any) -> Any:
    """`new` where `pred`, else `old`, leaf by leaf."""
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), new, old)


class InteriorPoint(CompiledSolver):
    """A primal-dual interior point with Ipopt's filter line search, after Wächter and Biegler (2006).

    The inequalities take slacks, and the slacks and the bounds of `x` take the log barrier. A
    failed line search starts Ipopt's feasibility restoration phase. `hessian="auto"` is
    Gauss-Newton plus damped BFGS on a least-squares objective, and damped BFGS otherwise.

    Args:
        tol: The scaled optimality error at which the run ends, `1e-8` by default.
        max_steps: The step budget, `300` by default.
        mu_init: The first barrier parameter, `0.1` by default, or smaller from a warm start.
        mu_min: The least barrier parameter, `1e-11` by default.
        kappa_mu: The linear factor of the monotone barrier update, `0.2` by default.
        theta_mu: The superlinear exponent of the monotone barrier update, `1.5` by default.
        tau_min: The least fraction to the boundary, `0.99` by default.
        bound_push: How far inside its bounds the start point moves, absolute, `0.01` by default.
        bound_frac: How far inside its bounds the start point moves, relative to their width, `0.01` by default.
        hessian: `"auto"` (default), `"bfgs"`, `"exact"` or `"gauss-newton"`.
        scaling: `"gradient"` (default) uses the problem's own scales or else Ipopt's gradient-based ones, and `"none"` solves unscaled.
        restoration: Whether a failed line search starts the restoration phase, `True` by default.
        max_failed_evaluations: Failed trial evaluations in a row before the run ends with `SOLVER_FAILED`, `10` by default.
        kkt: The [KKTSolve][diff_soup.optimization.KKTSolve] backend, `DenseKKT()` by default.
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

    # -- set-up ---------------------------------------------------------------------------

    def _prepare(self, problem: Any, like: Any) -> tuple[Any, Any, Any]:
        """The raw closures, the scaled closures and the bounds of `problem`."""
        raw = closures_of(problem)
        _hessian.check_mode(self.hessian, raw)
        if self.scaling not in ("gradient", "none"):
            raise ConfigurationError(f"scaling must be 'gradient' or 'none'; got {self.scaling!r}.")
        return raw, scaled_closures(raw), bounds_from(*bounds_of(problem, like))

    def _scales(self, problem: Any, raw: Any, x: Any, args: Any) -> Scales:
        """The problem's own scales, else the gradient-based ones, else 1."""
        if self.scaling == "none":
            return unit_scales(raw, x.dtype)
        return problem_scales(problem, raw, x, args)

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

    def _start(
        self,
        closures,
        bounds,
        x,
        *,
        args,
        lam_e,
        lam_i,
        z_lower,
        z_upper,
        v,
        mu,
        scales,
        slack_push=None,
        warm=False,
    ) -> InteriorPointState:
        """The state at a point inside the bounds.

        `lam_e = None` takes the least-squares multipliers; `v = None` takes 1; `warm` takes
        `mu` from the complementarity of the point, between the floor and `mu`.
        """
        dtype = x.dtype
        n = x.shape[0]
        evaluation = evaluate(closures, x, args, _hessian.uses_gauss_newton(self.hessian, closures))
        push = self.bound_push if slack_push is None else slack_push
        s = jnp.maximum(-evaluation.inequality, push)
        v = jnp.ones_like(s) if v is None else v
        if lam_e is None:
            lam_e, lam_i = self._least_squares_multipliers(evaluation, z_lower, z_upper, v)
        if warm:
            below, above = distances(x, bounds)
            products = jnp.concatenate(
                [
                    jnp.where(bounds.has_lower, below * z_lower, 0.0),
                    jnp.where(bounds.has_upper, above * z_upper, 0.0),
                    s * v,
                ]
            )
            count = jnp.sum(bounds.has_lower) + jnp.sum(bounds.has_upper) + s.shape[0]
            mean = jnp.sum(products) / jnp.maximum(count, 1)
            mu = jnp.clip(mean, self._mu_floor, mu)
        mu = jnp.asarray(mu, dtype)
        theta, _ = _theta(evaluation.equality, evaluation.inequality, s)
        scale = jnp.maximum(1.0, theta)
        if scales is None:
            scales = Scales(
                jnp.ones((), dtype),
                jnp.ones((closures.n_equality,), dtype),
                jnp.ones((closures.n_inequality,), dtype),
            )
        valid = evaluation.finite
        state = InteriorPointState(
            x=x,
            step=_int(0),
            done=~valid,
            status=_int(
                jnp.where(valid, TerminationStatus.MAX_STEPS, TerminationStatus.SOLVER_FAILED)
            ),
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
            kkt_error=jnp.zeros((), dtype),
            scales=scales,
            quasi_newton=_hessian.initial_matrix(self.hessian, closures, n, dtype),
            quasi_newton_updates=_int(0),
            restoration_steps=_int(0),
        )
        state = eqx.tree_at(lambda t: t.kkt_error, state, _error(state, 0.0, bounds))
        return self._with_termination(state)

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """Push `x0` inside its bounds, set the slacks from the inequalities, estimate the multipliers.

        A `warm_start` record supplies the multipliers instead. The point is then pushed only
        `WARM_PUSH` inside its bounds, and `mu` starts from the point's complementarity.
        """
        x0 = jnp.asarray(x0)
        raw, closures, bounds = self._prepare(problem, x0)
        dtype = x0.dtype
        if warm_start is None:
            x = push_into_bounds(x0, bounds, self.bound_push, self.bound_frac)
            scales = self._scales(problem, raw, x, args)
            return self._start(
                closures,
                bounds,
                x,
                args=(args, scales),
                lam_e=None,
                lam_i=None,
                z_lower=jnp.where(bounds.has_lower, 1.0, 0.0).astype(dtype),
                z_upper=jnp.where(bounds.has_upper, 1.0, 0.0).astype(dtype),
                v=None,
                mu=self.mu_init,
                scales=scales,
            )
        record = warm_start
        x = push_into_bounds(x0, bounds, WARM_PUSH, WARM_PUSH)
        scales = self._scales(problem, raw, x, args)
        objective = scales.objective
        z_lower, z_upper = record.bound_multipliers
        lam_i = jnp.asarray(record.multipliers_inequality, dtype) * objective / scales.inequality
        return self._start(
            closures,
            bounds,
            x,
            args=(args, scales),
            lam_e=jnp.asarray(record.multipliers_equality, dtype) * objective / scales.equality,
            lam_i=lam_i,
            z_lower=jnp.where(bounds.has_lower, jnp.maximum(objective * z_lower, WARM_PUSH), 0.0),
            z_upper=jnp.where(bounds.has_upper, jnp.maximum(objective * z_upper, WARM_PUSH), 0.0),
            v=jnp.maximum(lam_i, WARM_PUSH),
            mu=self.mu_init,
            scales=scales,
            slack_push=WARM_PUSH,
            warm=True,
        )

    # -- one step -------------------------------------------------------------------------

    def _with_termination(self, state):
        """`state` with `done` and `status` from `terminate`."""
        done, status = self.terminate(None, state, None)
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
        filt = _select(changed, _filter.empty_filter(mu.dtype), state.filt)
        tau = jnp.maximum(self.tau_min, 1.0 - mu)
        return eqx.tree_at(lambda s: (s.mu, s.tau, s.filt), state, (mu, tau, filt))

    def _inner(self) -> InteriorPoint:
        """The interior point that solves the restoration problem."""
        return dataclasses.replace(
            self, max_steps=_restoration.MAX_STEPS, restoration=False, scaling="none"
        )

    def _iterate(self, closures, bounds, state: InteriorPointState, args) -> InteriorPointState:
        """One step on prepared closures, without the termination test."""
        state = self._update_barrier(state, bounds)
        x, s, evaluation = state.x, state.slacks, state.evaluation
        mu, tau = state.mu, state.tau
        n_eq = closures.n_equality
        has_rows = closures.n_equality + closures.n_inequality > 0
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

        hessian = _hessian.lagrangian_hessian(
            self.hessian,
            closures,
            x,
            state.lam_e,
            state.lam_i,
            args,
            evaluation,
            state.quasi_newton,
        )
        primal_block = hessian + jnp.diag(primal_sigma(x, state.z_lower, state.z_upper, bounds))
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
        searched = search.accepted & factor.ok
        exhausted = search.consecutive_failed >= self.max_failed_evaluations
        filt_current = _filter.augment(
            state.filt, (1.0 - _filter.GAMMA_THETA) * theta, phi - _filter.GAMMA_PHI * theta
        )

        # The regular step.
        alpha = search.alpha
        x_step = x + alpha * search.dx
        s_step = s + alpha * search.ds
        dz_lower, dz_upper, dv = bound_multiplier_steps(
            x, search.dx, s, search.ds, state.z_lower, state.z_upper, state.v, mu, bounds
        )
        alpha_dual = dual_fraction_to_boundary(
            state.z_lower, dz_lower, state.z_upper, dz_upper, state.v, dv, tau, bounds
        )
        regular = (
            x_step,
            s_step,
            lam + alpha * search.dlam,
            state.z_lower + alpha_dual * dz_lower,
            state.z_upper + alpha_dual * dz_upper,
            state.v + alpha_dual * dv,
        )

        # The restoration phase, where the line search failed.
        need = ~searched & ~exhausted & (theta > self.tol)
        if self.restoration and has_rows:
            outcome = live_lane_cond(
                need,
                lambda: self._restore(closures, bounds, state, args, need, theta, filt_current),
                lambda: _restoration.skipped(x, s),
                live=~state.done,
                false_is_free=True,
            )
        else:
            outcome = _restoration.skipped(x, s)
        restored = self._after_restoration(outcome, state, bounds, lam)
        moved = searched | outcome.ok
        x_new, s_new, lam_new, z_lower, z_upper, v = _select(searched, regular, restored)
        z_lower, z_upper, v = safeguard_multipliers(x_new, s_new, z_lower, z_upper, v, mu, bounds)
        evaluation_new = evaluate(
            closures, x_new, args, _hessian.uses_gauss_newton(self.hessian, closures)
        )
        accepted = moved & evaluation_new.finite

        filt = _select(searched & search.f_type, state.filt, filt_current)
        lam_e_new, lam_i_new = lam_new[:n_eq], lam_new[n_eq:]
        gradient_change = (
            evaluation_new.gradient
            - evaluation.gradient
            + (evaluation_new.jacobian_equality - evaluation.jacobian_equality).T @ lam_e_new
            + (evaluation_new.jacobian_inequality - evaluation.jacobian_inequality).T @ lam_i_new
        )
        quasi_newton, updates = _hessian.update(
            self.hessian,
            closures,
            state.quasi_newton,
            state.quasi_newton_updates,
            x_new - x,
            gradient_change,
            evaluation_new,
        )
        moved_state = eqx.tree_at(
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
                t.quasi_newton,
                t.quasi_newton_updates,
            ),
            state,
            (
                x_new,
                s_new,
                lam_e_new,
                lam_i_new,
                z_lower,
                z_upper,
                v,
                evaluation_new,
                filt,
                quasi_newton,
                updates,
            ),
        )
        state = _select(accepted, moved_state, state)

        consecutive = jnp.where(
            moved & ~evaluation_new.finite, search.consecutive_failed + 1, search.consecutive_failed
        )
        consecutive = jnp.where(accepted, 0, consecutive)
        failure = jnp.where(
            exhausted | (consecutive >= self.max_failed_evaluations),
            TerminationStatus.SOLVER_FAILED,
            jnp.where(
                outcome.infeasible,
                TerminationStatus.INFEASIBLE,
                TerminationStatus.RESTORATION_FAILED,
            ),
        )
        failed_now = (~moved) | (consecutive >= self.max_failed_evaluations)
        return eqx.tree_at(
            lambda t: (
                t.step,
                t.done,
                t.status,
                t.delta_w_last,
                t.consecutive_failed,
                t.failed_evaluations,
                t.kkt_error,
                t.restoration_steps,
            ),
            state,
            (
                _int(state.step + 1),
                state.done | failed_now,
                _int(jnp.where(failed_now, failure, state.status)),
                jnp.where(delta_w > 0.0, delta_w, state.delta_w_last),
                _int(consecutive),
                _int(state.failed_evaluations + search.failed + outcome.failed),
                _error(state, 0.0, bounds),
                _int(state.restoration_steps + outcome.steps),
            ),
        )

    def _restore(self, closures, bounds, state, args, need, theta, filt):
        """The restoration phase on the scaled problem."""
        return _restoration.restore(
            self._inner(), closures, bounds, state, args, run=need, theta=theta, filt=filt
        )

    def _after_restoration(self, outcome, state, bounds, lam):
        """The regular phase's point after restoration: `lam = 0` and Newton-updated bound multipliers.

        The bound multipliers take the complementarity Newton step for the primal change of
        the whole phase, cut at the boundary fraction, and are all reset to 1 when one passes
        `BOUND_MULTIPLIER_RESET`, as Ipopt does.
        """
        x, s, mu, tau = state.x, state.slacks, state.mu, state.tau
        dz_lower, dz_upper, dv = bound_multiplier_steps(
            x, outcome.x - x, s, outcome.s - s, state.z_lower, state.z_upper, state.v, mu, bounds
        )
        alpha = dual_fraction_to_boundary(
            state.z_lower, dz_lower, state.z_upper, dz_upper, state.v, dv, tau, bounds
        )
        z_lower = state.z_lower + alpha * dz_lower
        z_upper = state.z_upper + alpha * dz_upper
        v = state.v + alpha * dv
        largest = jnp.maximum(
            jnp.max(z_lower, initial=0.0),
            jnp.maximum(jnp.max(z_upper, initial=0.0), jnp.max(v, initial=0.0)),
        )
        reset = largest > BOUND_MULTIPLIER_RESET
        z_lower = jnp.where(reset, jnp.where(bounds.has_lower, 1.0, 0.0), z_lower)
        z_upper = jnp.where(reset, jnp.where(bounds.has_upper, 1.0, 0.0), z_upper)
        v = jnp.where(reset, 1.0, v)
        return outcome.x, outcome.s, jnp.zeros_like(lam), z_lower, z_upper, v

    def _advance(self, closures, bounds, state, args) -> InteriorPointState:
        """One step on prepared closures, with the termination test."""
        return self._with_termination(self._iterate(closures, bounds, state, args))

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One barrier step: the KKT solve, fraction to the boundary, the filter line search."""
        _, closures, bounds = self._prepare(problem, state.x)
        return self._iterate(closures, bounds, state, (args, state.scales))

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
                TerminationStatus.KKT_TOL,
                jnp.where(diverged, TerminationStatus.DIVERGED, TerminationStatus.MAX_STEPS),
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
        """The compiled loop, as [CompiledSolver.run][diff_soup.optimization.CompiledSolver.run] describes."""
        solver = self if max_steps is None else dataclasses.replace(self, max_steps=max_steps)
        return implicit_run(solver._run_primal, _raw, problem, x0, args, warm_start)

    def _run_primal(self, problem: Any, x0: Any, args: Any, warm_start: Any):
        """The loop itself, never differentiated."""
        state = self.init(problem, x0, args, warm_start=warm_start)
        _, closures, bounds = self._prepare(problem, state.x)

        def body(current):
            new = self._advance(closures, bounds, current, (args, current.scales))
            return _select(current.done, current, new)

        state = lax.while_loop(lambda current: ~current.done, body, state)
        return state.x, _record(state, bounds)


def _raw(problem: Any, x: Any) -> tuple[Any, Any]:
    """The raw closures and the bounds of `problem`, for the implicit derivative."""
    return closures_of(problem), bounds_from(*bounds_of(problem, x))


def _record(state: InteriorPointState, bounds: Any) -> SolveRecord:
    """The `SolveRecord` of a finished state, with the multipliers and slacks unscaled."""
    scales = state.scales
    objective = scales.objective
    record = SolveRecord(
        multipliers_equality=state.lam_e * scales.equality / objective,
        multipliers_inequality=state.lam_i * scales.inequality / objective,
        slacks=state.slacks / scales.inequality,
        bound_multipliers=(state.z_lower / objective, state.z_upper / objective),
        mu=state.mu,
        steps=state.step,
        failed_evaluations=state.failed_evaluations,
        kkt_error=state.kkt_error,
        constraint_violation=constraint_violation(state.evaluation, scales),
        status=state.status,
        degenerate=jnp.asarray(False),
    )
    ev = state.evaluation
    degenerate = active_set(
        state.x, record, bounds, ev.jacobian_equality, ev.jacobian_inequality
    ).degenerate
    return eqx.tree_at(lambda r: r.degenerate, record, degenerate)
