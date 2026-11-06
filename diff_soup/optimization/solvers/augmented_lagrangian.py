"""An augmented Lagrangian with a compiled projected Newton trust region inside.

The penalty is Powell-Hestenes-Rockafellar's on the slack-free form,

    L_A = f + lam_E c_E + rho |c_E|^2 / 2 + sum(max(0, lam_I + rho c_I)^2 - lam_I^2) / (2 rho),

minimized over the bounds of `x` by `_projected_newton.py`. Its gradient is that of the
Lagrangian at the first-order estimates `lam_E + rho c_E` and `max(0, lam_I + rho c_I)`, and
its model Hessian is the Lagrangian Hessian there (exact or quasi-Newton, as `hessian` says)
plus `rho J^T J` on the equalities and the inequalities whose estimate is positive.

The outer updates are LANCELOT's (Conn, Gould and Toint 1991, as Nocedal and Wright,
Framework 17.4, state them), with the violation `max(|c_E|, |min(-c_I, lam_I / rho)|)` of
Birgin and Martínez: when it is below `eta` the multipliers take the estimates, `eta` falls
by `rho^0.9` and the inner tolerance `omega` by `rho`; otherwise `rho` grows by `rho_factor`
and `eta = rho^-0.1`, `omega = 1 / rho`. The run ends on the scaled optimality error at the
estimates, with the bound multipliers read from the inner gradient.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.solvers import _hessian, _projected_newton
from diff_soup.optimization.solvers._barrier import bounds_from
from diff_soup.optimization.solvers._closures import (
    Evaluation,
    Scales,
    bounds_of,
    closures_of,
    evaluate,
    problem_scales,
    scaled_closures,
    trial,
)
from diff_soup.optimization.solvers._implicit import active_set, implicit_run
from diff_soup.optimization.solvers._optimality import constraint_violation, optimality_error
from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
)
from diff_soup.optimization.solvers.interior_point import DIVERGED_X, _int, _raw, _select

RHO_MAX = 1e12
RADIUS_INIT = 1.0
_ETA_POWER = 0.9
_ETA_RESET = 0.1


class AugmentedLagrangianState(SolverState):
    """The carry of [AugmentedLagrangian][diff_soup.optimization.AugmentedLagrangian], in the scaled problem.

    Attributes:
        lam_e: The equality multipliers of the next inner problem.
        lam_i: The inequality multipliers of the next inner problem.
        estimate_e: The first-order equality multipliers at `x`, `lam_e + rho c_E`.
        estimate_i: The first-order inequality multipliers at `x`, `max(0, lam_i + rho c_I)`.
        z_lower: The lower-bound multipliers at `x`.
        z_upper: The upper-bound multipliers at `x`.
        rho: The penalty.
        omega: The inner tolerance on the projected gradient.
        eta: The violation below which the multipliers are updated.
        evaluation: The objective, constraints and first derivatives at `x`.
        radius: The inner trust-region radius, carried from one inner problem to the next.
        quasi_newton: The BFGS matrix, or `(0, 0)`.
        quasi_newton_updates: How many updates it has had.
        consecutive_failed: The failed trial evaluations in a row.
        failed_evaluations: The failed trial evaluations in all.
        kkt_error: The scaled optimality error at `x` and the estimates.
        inner_steps: The projected Newton steps of every inner problem so far.
        scales: The objective and constraint scales.
    """

    lam_e: Any
    lam_i: Any
    estimate_e: Any
    estimate_i: Any
    z_lower: Any
    z_upper: Any
    rho: Any
    omega: Any
    eta: Any
    evaluation: Evaluation
    radius: Any
    quasi_newton: Any
    quasi_newton_updates: Any
    consecutive_failed: Any
    failed_evaluations: Any
    kkt_error: Any
    inner_steps: Any
    scales: Scales


def _estimates(equality: Any, inequality: Any, lam_e: Any, lam_i: Any, rho: Any):
    """The first-order multipliers `lam_e + rho c_E` and `max(0, lam_i + rho c_I)`."""
    return lam_e + rho * equality, jnp.maximum(lam_i + rho * inequality, 0.0)


def _penalty(value: Any, equality: Any, inequality: Any, lam_e: Any, lam_i: Any, rho: Any):
    """The PHR augmented Lagrangian from the objective and the constraint values."""
    shifted = jnp.maximum(lam_i + rho * inequality, 0.0)
    return (
        value
        + lam_e @ equality
        + 0.5 * rho * (equality @ equality)
        + (shifted @ shifted - lam_i @ lam_i) / (2.0 * rho)
    )


def _violation(equality: Any, inequality: Any, lam_i: Any, rho: Any) -> Any:
    """Birgin and Martínez's measure `max(|c_E|, |min(-c_I, lam_I / rho)|)`."""
    parts = jnp.concatenate([jnp.abs(equality), jnp.abs(jnp.minimum(-inequality, lam_i / rho))])
    return jnp.max(parts, initial=0.0)


def _error(evaluation, x, estimate_e, estimate_i, z_lower, z_upper, bounds) -> Any:
    """The scaled optimality error at the estimates, with slacks `max(-c_I, 0)`."""
    slacks = jnp.maximum(-evaluation.inequality, 0.0)
    return optimality_error(
        evaluation,
        x,
        slacks,
        estimate_e,
        estimate_i,
        z_lower,
        z_upper,
        estimate_i,
        0.0,
        bounds,
    )


class AugmentedLagrangian(CompiledSolver):
    """Powell-Hestenes-Rockafellar multipliers with the LANCELOT update rules.

    Each outer step minimizes the augmented Lagrangian under the bounds with a projected Newton
    trust region, then updates the multipliers or raises the penalty. It is the fallback for
    degenerate constraints.

    Args:
        tol: The scaled optimality error at which the run ends, `1e-8` by default.
        feasibility_tol: The constraint violation at which the run ends, `1e-8` by default.
        max_steps: The outer step budget, `50` by default.
        rho_init: The first penalty, `10.0` by default.
        rho_factor: The factor the penalty grows by when the violation did not fall enough, `10.0` by default.
        inner_max_steps: The step budget of each inner minimization, `200` by default.
        hessian: `"auto"` (default), `"bfgs"`, `"exact"` or `"gauss-newton"`, as for `InteriorPoint`.
        max_failed_evaluations: Failed trial evaluations in a row before the run ends with `SOLVER_FAILED`, `10` by default.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    feasibility_tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=50)
    rho_init: float = eqx.field(static=True, default=10.0)
    rho_factor: float = eqx.field(static=True, default=10.0)
    inner_max_steps: int = eqx.field(static=True, default=200)
    hessian: str = eqx.field(static=True, default="auto")
    max_failed_evaluations: int = eqx.field(static=True, default=10)

    def _prepare(self, problem: Any, like: Any) -> tuple[Any, Any, Any]:
        """The raw closures, the scaled closures and the bounds of `problem`."""
        raw = closures_of(problem)
        _hessian.check_mode(self.hessian, raw)
        if not (self.rho_init > 0.0 and self.rho_factor > 1.0):
            raise ConfigurationError("rho_init must be positive and rho_factor above 1.")
        return raw, scaled_closures(raw), bounds_from(*bounds_of(problem, like))

    @property
    def _omega_floor(self) -> float:
        """The least inner tolerance: a tenth of `tol`, so the outer test can pass."""
        return 0.1 * self.tol

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """The state at `x0` projected into its bounds, with zero multipliers or those of `warm_start`."""
        x0 = jnp.asarray(x0)
        raw, closures, bounds = self._prepare(problem, x0)
        dtype = x0.dtype
        x = _projected_newton.project(x0, bounds)
        scales = problem_scales(problem, raw, x, args)
        packed = (args, scales)
        evaluation = evaluate(
            closures, x, packed, _hessian.uses_gauss_newton(self.hessian, closures)
        )
        if warm_start is None:
            lam_e = jnp.zeros((closures.n_equality,), dtype)
            lam_i = jnp.zeros((closures.n_inequality,), dtype)
        else:
            f = scales.objective
            lam_e = jnp.asarray(warm_start.multipliers_equality, dtype) * f / scales.equality
            lam_i = jnp.asarray(warm_start.multipliers_inequality, dtype) * f / scales.inequality
        rho = jnp.asarray(self.rho_init, dtype)
        constrained = closures.n_equality + closures.n_inequality > 0
        omega = 1.0 / rho if constrained else jnp.asarray(self._omega_floor, dtype)
        valid = evaluation.finite
        state = AugmentedLagrangianState(
            x=x,
            step=_int(0),
            done=~valid,
            status=_int(
                jnp.where(valid, TerminationStatus.MAX_STEPS, TerminationStatus.SOLVER_FAILED)
            ),
            lam_e=lam_e,
            lam_i=lam_i,
            estimate_e=lam_e,
            estimate_i=lam_i,
            z_lower=jnp.zeros_like(x),
            z_upper=jnp.zeros_like(x),
            rho=rho,
            omega=omega,
            eta=rho**-_ETA_RESET,
            evaluation=evaluation,
            radius=jnp.asarray(RADIUS_INIT, dtype),
            quasi_newton=_hessian.initial_matrix(self.hessian, closures, x.shape[0], dtype),
            quasi_newton_updates=_int(0),
            consecutive_failed=_int(0),
            failed_evaluations=_int(0),
            kkt_error=jnp.zeros((), dtype),
            inner_steps=_int(0),
            scales=scales,
        )
        state = self._with_estimates(state, state.evaluation, None, bounds)
        return self._with_termination(state)

    def _with_estimates(self, state, evaluation, gradient, bounds):
        """`state` with the estimates, bound multipliers and optimality error at its `x`."""
        estimate_e, estimate_i = _estimates(
            evaluation.equality, evaluation.inequality, state.lam_e, state.lam_i, state.rho
        )
        if gradient is None:
            gradient = (
                evaluation.gradient
                + evaluation.jacobian_equality.T @ estimate_e
                + evaluation.jacobian_inequality.T @ estimate_i
            )
        z_lower, z_upper = _projected_newton.bound_multipliers(state.x, gradient, bounds)
        error = _error(evaluation, state.x, estimate_e, estimate_i, z_lower, z_upper, bounds)
        return eqx.tree_at(
            lambda s: (s.estimate_e, s.estimate_i, s.z_lower, s.z_upper, s.kkt_error),
            state,
            (estimate_e, estimate_i, z_lower, z_upper, error),
        )

    def _with_termination(self, state):
        """`state` with `done` and `status` from `terminate`."""
        done, status = self.terminate(None, state, None)
        return eqx.tree_at(lambda s: (s.done, s.status), state, (done, status))

    def _model(self, closures, x, evaluation, lam_e, lam_i, rho, args, matrix):
        """The augmented Lagrangian's value, gradient and model Hessian at `x`."""
        estimate_e, estimate_i = _estimates(
            evaluation.equality, evaluation.inequality, lam_e, lam_i, rho
        )
        jac_eq, jac_ineq = evaluation.jacobian_equality, evaluation.jacobian_inequality
        value = _penalty(
            evaluation.value, evaluation.equality, evaluation.inequality, lam_e, lam_i, rho
        )
        gradient = evaluation.gradient + jac_eq.T @ estimate_e + jac_ineq.T @ estimate_i
        curvature = _hessian.lagrangian_hessian(
            self.hessian, closures, x, estimate_e, estimate_i, args, evaluation, matrix
        )
        active = (lam_i + rho * evaluation.inequality > 0.0).astype(x.dtype)
        penalty = rho * (jac_eq.T @ jac_eq + jac_ineq.T @ (active[:, None] * jac_ineq))
        hessian = curvature + penalty
        return value, gradient, hessian

    def _iterate(self, closures, bounds, state: AugmentedLagrangianState, args):
        """One outer step on prepared closures."""
        lam_e, lam_i, rho = state.lam_e, state.lam_i, state.rho
        gauss_newton = _hessian.uses_gauss_newton(self.hessian, closures)

        def penalty_at(x):
            value, equality, inequality, valid = trial(closures, x, args)
            return _penalty(value, equality, inequality, lam_e, lam_i, rho), valid

        def derivatives(x_new, x_old, aux):
            ev_old, matrix, updates = aux
            ev = evaluate(closures, x_new, args, gauss_newton)
            estimate_e, estimate_i = _estimates(ev.equality, ev.inequality, lam_e, lam_i, rho)
            change = (
                ev.gradient
                - ev_old.gradient
                + (ev.jacobian_equality - ev_old.jacobian_equality).T @ estimate_e
                + (ev.jacobian_inequality - ev_old.jacobian_inequality).T @ estimate_i
            )
            matrix, updates = _hessian.update(
                self.hessian, closures, matrix, updates, x_new - x_old, change, ev
            )
            value, gradient, hessian = self._model(
                closures, x_new, ev, lam_e, lam_i, rho, args, matrix
            )
            finite = ev.finite & jnp.all(jnp.isfinite(hessian))
            return value, gradient, hessian, (ev, matrix, updates), finite

        value, gradient, hessian = self._model(
            closures, state.x, state.evaluation, lam_e, lam_i, rho, args, state.quasi_newton
        )
        start = _projected_newton.start_state(
            state.x,
            value,
            gradient,
            hessian,
            (state.evaluation, state.quasi_newton, state.quasi_newton_updates),
            jnp.maximum(state.radius, 1e-4 * (1.0 + jnp.max(jnp.abs(state.x), initial=0.0))),
            state.done,
        )
        end = _projected_newton.minimize(
            penalty_at,
            derivatives,
            start,
            bounds,
            tol=jnp.maximum(state.omega, self._omega_floor),
            max_steps=self.inner_max_steps,
            max_failed=self.max_failed_evaluations,
        )
        evaluation, matrix, updates = end.aux
        state = eqx.tree_at(
            lambda s: (s.x, s.evaluation, s.radius, s.quasi_newton, s.quasi_newton_updates),
            state,
            (end.x, evaluation, end.radius, matrix, updates),
        )
        state = self._with_estimates(state, evaluation, end.gradient, bounds)

        violation = _violation(evaluation.equality, evaluation.inequality, lam_i, rho)
        update = violation <= state.eta
        raised = jnp.minimum(rho * self.rho_factor, RHO_MAX)
        stuck = ~update & (rho >= RHO_MAX)
        omega_floor = self._omega_floor
        exhausted = end.consecutive_failed >= self.max_failed_evaluations
        failure = jnp.where(
            exhausted, TerminationStatus.SOLVER_FAILED, TerminationStatus.INFEASIBLE
        )
        ended = exhausted | stuck
        return eqx.tree_at(
            lambda s: (
                s.lam_e,
                s.lam_i,
                s.rho,
                s.omega,
                s.eta,
                s.step,
                s.done,
                s.status,
                s.consecutive_failed,
                s.failed_evaluations,
                s.inner_steps,
            ),
            state,
            (
                jnp.where(update, state.estimate_e, lam_e),
                jnp.where(update, state.estimate_i, lam_i),
                jnp.where(update, rho, raised),
                jnp.maximum(jnp.where(update, state.omega / rho, 1.0 / raised), omega_floor),
                jnp.maximum(
                    jnp.where(update, state.eta / rho**_ETA_POWER, raised**-_ETA_RESET),
                    0.1 * min(self.tol, self.feasibility_tol),
                ),
                _int(state.step + 1),
                state.done | ended,
                _int(jnp.where(ended, failure, state.status)),
                _int(end.consecutive_failed),
                _int(state.failed_evaluations + end.failed_evaluations),
                _int(state.inner_steps + end.steps),
            ),
        )

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One outer step: the inner minimization, then the multiplier or penalty update."""
        _, closures, bounds = self._prepare(problem, state.x)
        return self._iterate(closures, bounds, state, (args, state.scales))

    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """End on the optimality error and the violation, the budget, or a failure."""
        violation = constraint_violation(state.evaluation, state.scales)
        converged = (state.kkt_error <= self.tol) & (violation <= self.feasibility_tol)
        finite = (
            jnp.all(jnp.isfinite(state.x))
            & jnp.all(jnp.isfinite(state.estimate_e))
            & jnp.all(jnp.isfinite(state.estimate_i))
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
        """The compiled loop, as [CompiledSolver.run][diff_soup.optimization.solvers.CompiledSolver.run] describes."""
        if self.hessian not in _hessian.MODES:
            raise ConfigurationError(f"hessian must be one of {_hessian.MODES}.")
        solver = self if max_steps is None else dataclasses.replace(self, max_steps=max_steps)
        return implicit_run(solver._run_primal, _raw, problem, x0, args, warm_start)

    def _run_primal(self, problem: Any, x0: Any, args: Any, warm_start: Any):
        """The loop itself, never differentiated."""
        state = self.init(problem, x0, args, warm_start=warm_start)
        _, closures, bounds = self._prepare(problem, state.x)

        def body(current):
            new = self._with_termination(
                self._iterate(closures, bounds, current, (args, current.scales))
            )
            return _select(current.done, current, new)

        state = lax.while_loop(lambda current: ~current.done, body, state)
        return state.x, _record(state, bounds)


def _record(state: AugmentedLagrangianState, bounds: Any) -> SolveRecord:
    """The `SolveRecord` of a finished state, unscaled; `mu` is 0."""
    scales = state.scales
    f = scales.objective
    ev = state.evaluation
    record = SolveRecord(
        multipliers_equality=state.estimate_e * scales.equality / f,
        multipliers_inequality=state.estimate_i * scales.inequality / f,
        slacks=jnp.maximum(-ev.inequality, 0.0) / scales.inequality,
        bound_multipliers=(state.z_lower / f, state.z_upper / f),
        mu=jnp.zeros((), state.x.dtype),
        steps=state.step,
        failed_evaluations=state.failed_evaluations,
        kkt_error=state.kkt_error,
        constraint_violation=constraint_violation(ev, scales),
        status=state.status,
        degenerate=jnp.asarray(False),
    )
    degenerate = active_set(
        state.x, record, bounds, ev.jacobian_equality, ev.jacobian_inequality
    ).degenerate
    return eqx.tree_at(lambda r: r.degenerate, record, degenerate)
