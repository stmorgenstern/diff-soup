"""Sequential quadratic programming on the interior point's KKT step and filter line search."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import DenseKKT
from diff_soup.optimization.solvers import _filter, _hessian
from diff_soup.optimization.solvers._barrier import bounds_from
from diff_soup.optimization.solvers._closures import (
    Closures,
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
from diff_soup.optimization.solvers.interior_point import InteriorPoint, _int, _raw, _select

_MAX_TRIALS = 40


class SQPState(SolverState):
    """The carry of [SQP][diff_soup.optimization.SQP], in the scaled problem.

    Attributes:
        lam_e: The equality multipliers.
        lam_i: The inequality multipliers.
        z_lower: The lower-bound multipliers.
        z_upper: The upper-bound multipliers.
        evaluation: The objective, constraints and first derivatives at `x`.
        filt: The line-search filter on `(theta, f)`.
        theta_min: The switching threshold on the constraint violation.
        theta_max: The largest constraint violation a trial may have.
        consecutive_failed: The failed trial evaluations in a row.
        failed_evaluations: The failed trial evaluations in all.
        kkt_error: The scaled optimality error at `x`.
        scales: The objective and constraint scales.
        quasi_newton: The BFGS matrix, or `(0, 0)`.
        quasi_newton_updates: How many updates it has had.
        qp_steps: The interior point steps of every QP so far.
    """

    lam_e: Any
    lam_i: Any
    z_lower: Any
    z_upper: Any
    evaluation: Evaluation
    filt: _filter.Filter
    theta_min: Any
    theta_max: Any
    consecutive_failed: Any
    failed_evaluations: Any
    kkt_error: Any
    scales: Scales
    quasi_newton: Any
    quasi_newton_updates: Any
    qp_steps: Any


def _violation(equality: Any, inequality: Any) -> Any:
    """`theta = |c_E|_1 + |max(c_I, 0)|_1`."""
    return jnp.sum(jnp.abs(equality)) + jnp.sum(jnp.maximum(inequality, 0.0))


def _error(state: SQPState, bounds: Any) -> Any:
    """The scaled optimality error, with slacks `max(-c_I, 0)` and their multipliers `lam_I`."""
    ev = state.evaluation
    slacks = jnp.maximum(-ev.inequality, 0.0)
    return optimality_error(
        ev,
        state.x,
        slacks,
        state.lam_e,
        state.lam_i,
        state.z_lower,
        state.z_upper,
        jnp.maximum(state.lam_i, 0.0),
        0.0,
        bounds,
    )


def _qp_closures(n: int, n_eq: int, n_ineq: int) -> Closures:
    """The closures of the QP in the step `d`, whose `args` are `(g, B, c_E, J_E, c_I, J_I)`."""

    def value_and_gradient(d, qp):
        g, b = qp[0], qp[1]
        return g @ d + 0.5 * d @ (b @ d), g + b @ d

    def constraints_and_jacobians(d, qp):
        _, _, c_e, j_e, c_i, j_i = qp
        return c_e + j_e @ d, c_i + j_i @ d, j_e, j_i

    return Closures(
        value=lambda d, qp: value_and_gradient(d, qp)[0],
        value_and_gradient=value_and_gradient,
        constraints=lambda d, qp: constraints_and_jacobians(d, qp)[:2],
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=lambda d, le, li, qp: qp[1],
        gauss_newton=None,
        is_valid=None,
        n_equality=n_eq,
        n_inequality=n_ineq,
    )


class SQP(CompiledSolver):
    """Sequential quadratic programming with a filter line search.

    Each step solves the QP of the model Hessian and the linearized constraints by the interior
    point. A warm start takes an earlier record's multipliers, so it suits a sequence of nearby
    problems.

    Args:
        tol: The scaled optimality error at which the run ends, `1e-8` by default.
        max_steps: The step budget, `100` by default.
        hessian: `"auto"` (default), `"bfgs"`, `"exact"` or `"gauss-newton"`, as for `InteriorPoint`.
        qp_tol: The scaled optimality error at which each QP solve ends, `1e-10` by default.
        qp_max_steps: The step budget of each QP solve, `50` by default.
        max_failed_evaluations: Failed trial evaluations in a row before the run ends with `SOLVER_FAILED`, `10` by default.
        kkt: The [KKTSolve][diff_soup.optimization.solvers.KKTSolve] backend, `DenseKKT()` by default.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=100)
    hessian: str = eqx.field(static=True, default="auto")
    qp_tol: float = eqx.field(static=True, default=1e-10)
    qp_max_steps: int = eqx.field(static=True, default=50)
    max_failed_evaluations: int = eqx.field(static=True, default=10)
    kkt: Any = eqx.field(static=True, default_factory=DenseKKT)

    def _prepare(self, problem: Any, like: Any) -> tuple[Any, Any, Any]:
        """The raw closures, the scaled closures and the bounds of `problem`."""
        raw = closures_of(problem)
        _hessian.check_mode(self.hessian, raw)
        return raw, scaled_closures(raw), bounds_from(*bounds_of(problem, like))

    def _qp_solver(self) -> InteriorPoint:
        """The interior point that solves each QP."""
        return InteriorPoint(
            tol=self.qp_tol,
            max_steps=self.qp_max_steps,
            hessian="exact",
            scaling="none",
            restoration=True,
            kkt=self.kkt,
        )

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """The state at `x0` projected into its bounds, with the multipliers of `warm_start` when given."""
        x0 = jnp.asarray(x0)
        raw, closures, bounds = self._prepare(problem, x0)
        dtype = x0.dtype
        x = jnp.clip(
            x0,
            jnp.where(bounds.has_lower, bounds.lower, -jnp.inf),
            jnp.where(bounds.has_upper, bounds.upper, jnp.inf),
        )
        scales = problem_scales(problem, raw, x, args)
        packed = (args, scales)
        evaluation = evaluate(
            closures, x, packed, _hessian.uses_gauss_newton(self.hessian, closures)
        )
        n = x.shape[0]
        if warm_start is None:
            lam_e = jnp.zeros((closures.n_equality,), dtype)
            lam_i = jnp.zeros((closures.n_inequality,), dtype)
            z_lower = jnp.zeros((n,), dtype)
            z_upper = jnp.zeros((n,), dtype)
        else:
            f = scales.objective
            lam_e = jnp.asarray(warm_start.multipliers_equality, dtype) * f / scales.equality
            lam_i = jnp.asarray(warm_start.multipliers_inequality, dtype) * f / scales.inequality
            z_lower = jnp.asarray(warm_start.bound_multipliers[0], dtype) * f
            z_upper = jnp.asarray(warm_start.bound_multipliers[1], dtype) * f
        theta = _violation(evaluation.equality, evaluation.inequality)
        scale = jnp.maximum(1.0, theta)
        valid = evaluation.finite
        state = SQPState(
            x=x,
            step=_int(0),
            done=~valid,
            status=_int(
                jnp.where(valid, TerminationStatus.MAX_STEPS, TerminationStatus.SOLVER_FAILED)
            ),
            lam_e=lam_e,
            lam_i=lam_i,
            z_lower=z_lower,
            z_upper=z_upper,
            evaluation=evaluation,
            filt=_filter.empty_filter(dtype),
            theta_min=_filter.THETA_MIN_FACTOR * scale,
            theta_max=_filter.THETA_MAX_FACTOR * scale,
            consecutive_failed=_int(0),
            failed_evaluations=_int(0),
            kkt_error=jnp.zeros((), dtype),
            scales=scales,
            quasi_newton=_hessian.initial_matrix(self.hessian, closures, n, dtype),
            quasi_newton_updates=_int(0),
            qp_steps=_int(0),
        )
        state = eqx.tree_at(lambda t: t.kkt_error, state, _error(state, bounds))
        return self._with_termination(state)

    def _with_termination(self, state):
        """`state` with `done` and `status` from `terminate`."""
        done, status = self.terminate(None, state, None)
        return eqx.tree_at(lambda s: (s.done, s.status), state, (done, status))

    def _solve_qp(self, closures, bounds, state, hessian, args, bounded):
        """The QP step `d` and its multipliers `(lam_e, lam_i, z_lower, z_upper)`, and its steps."""
        ev = state.evaluation
        n = state.x.shape[0]
        n_eq, n_ineq = closures.n_equality, closures.n_inequality
        dtype = state.x.dtype
        if n_ineq == 0 and not bounded:
            factor, _ = self.kkt.factor_regularized(
                hessian,
                ev.jacobian_equality,
                jnp.zeros((n_eq,), dtype),
                0.0,
                jnp.zeros((), dtype),
            )
            d, lam = self.kkt.solve_factored(factor, -ev.gradient, -ev.equality)
            zeros = jnp.zeros((n,), dtype)
            return d, lam, jnp.zeros((0,), dtype), zeros, zeros, _int(1), factor.ok
        qp = _qp_closures(n, n_eq, n_ineq)
        qp_bounds = bounds._replace(
            lower=jnp.where(bounds.has_lower, bounds.lower - state.x, 0.0),
            upper=jnp.where(bounds.has_upper, bounds.upper - state.x, 0.0),
        )
        qp_args = (
            ev.gradient,
            hessian,
            ev.equality,
            ev.jacobian_equality,
            ev.inequality,
            ev.jacobian_inequality,
        )
        inner = self._qp_solver()
        d0 = jnp.zeros((n,), dtype)
        start = inner._start(
            qp,
            qp_bounds,
            jnp.clip(
                d0,
                jnp.where(qp_bounds.has_lower, qp_bounds.lower + 1e-8, -jnp.inf),
                jnp.where(qp_bounds.has_upper, qp_bounds.upper - 1e-8, jnp.inf),
            ),
            args=qp_args,
            lam_e=None,
            lam_i=None,
            z_lower=jnp.where(qp_bounds.has_lower, 1.0, 0.0).astype(dtype),
            z_upper=jnp.where(qp_bounds.has_upper, 1.0, 0.0).astype(dtype),
            v=None,
            mu=inner.mu_init,
            scales=None,
        )
        end = lax.while_loop(
            lambda s: ~s.done, lambda s: inner._advance(qp, qp_bounds, s, qp_args), start
        )
        ok = end.status == TerminationStatus.KKT_TOL
        return end.x, end.lam_e, end.lam_i, end.z_lower, end.z_upper, end.step, ok

    def _iterate(self, closures, bounds, state: SQPState, args, bounded=True) -> SQPState:
        """One SQP step on prepared closures."""
        ev = state.evaluation
        x = state.x
        n_eq = closures.n_equality
        hessian = _hessian.lagrangian_hessian(
            self.hessian, closures, x, state.lam_e, state.lam_i, args, ev, state.quasi_newton
        )
        d, qp_e, qp_i, qp_zl, qp_zu, qp_steps, qp_ok = self._solve_qp(
            closures, bounds, state, hessian, args, bounded
        )
        theta = _violation(ev.equality, ev.inequality)
        phi = ev.value
        slope = ev.gradient @ d
        alpha_min = _filter.minimum_step(theta, slope, state.theta_min)

        def cond(carry):
            alpha, accepted, stop, *_ = carry
            return ~accepted & ~stop

        def body(carry):
            alpha, _, _, f_type, failed, consecutive, trials = carry
            value, equality, inequality, valid = trial(closures, x + alpha * d, args)
            theta_t = _violation(equality, inequality)
            accepted, f_type = _filter.acceptable(
                state.filt,
                theta_t,
                value,
                alpha,
                theta,
                phi,
                slope,
                state.theta_min,
                state.theta_max,
            )
            accepted = accepted & valid
            consecutive = jnp.where(valid, 0, consecutive + 1)
            next_alpha = jnp.where(accepted, alpha, 0.5 * alpha)
            stop = ~accepted & (
                (next_alpha < alpha_min)
                | (consecutive >= self.max_failed_evaluations)
                | (trials + 1 >= _MAX_TRIALS)
            )
            return (
                next_alpha,
                accepted,
                stop,
                f_type & accepted,
                failed + ~valid,
                _int(consecutive),
                _int(trials + 1),
            )

        # A step too small to measure is taken whole, as the interior point does.
        tiny = jnp.max(jnp.abs(d) / (1.0 + jnp.abs(x)), initial=0.0) < 10.0 * jnp.finfo(x.dtype).eps
        start = (
            jnp.ones((), x.dtype),
            tiny,
            jnp.asarray(False),
            jnp.asarray(False),
            _int(0),
            _int(state.consecutive_failed),
            _int(0),
        )
        alpha, accepted, _, f_type, failed, consecutive, _ = lax.while_loop(cond, body, start)
        x_new = x + alpha * d
        lam_e = state.lam_e + alpha * (qp_e - state.lam_e)
        lam_i = state.lam_i + alpha * (qp_i - state.lam_i)
        z_lower = state.z_lower + alpha * (qp_zl - state.z_lower)
        z_upper = state.z_upper + alpha * (qp_zu - state.z_upper)
        ev_new = evaluate(closures, x_new, args, _hessian.uses_gauss_newton(self.hessian, closures))
        moved = accepted & ev_new.finite
        gradient_change = (
            ev_new.gradient
            - ev.gradient
            + (ev_new.jacobian_equality - ev.jacobian_equality).T @ lam_e
            + (ev_new.jacobian_inequality - ev.jacobian_inequality).T @ lam_i
        )
        quasi_newton, updates = _hessian.update(
            self.hessian,
            closures,
            state.quasi_newton,
            state.quasi_newton_updates,
            x_new - x,
            gradient_change,
            ev_new,
        )
        filt = _select(
            f_type,
            state.filt,
            _filter.augment(
                state.filt, (1.0 - _filter.GAMMA_THETA) * theta, phi - _filter.GAMMA_PHI * theta
            ),
        )
        new = eqx.tree_at(
            lambda t: (
                t.x,
                t.lam_e,
                t.lam_i,
                t.z_lower,
                t.z_upper,
                t.evaluation,
                t.filt,
                t.quasi_newton,
                t.quasi_newton_updates,
            ),
            state,
            (x_new, lam_e, lam_i, z_lower, z_upper, ev_new, filt, quasi_newton, updates),
        )
        state = _select(moved, new, state)
        consecutive = jnp.where(accepted & ~ev_new.finite, consecutive + 1, consecutive)
        exhausted = consecutive >= self.max_failed_evaluations
        failed_now = ~moved & (~accepted | exhausted)
        failure = jnp.where(
            exhausted, TerminationStatus.SOLVER_FAILED, TerminationStatus.RESTORATION_FAILED
        )
        del n_eq, qp_ok
        return eqx.tree_at(
            lambda t: (
                t.step,
                t.done,
                t.status,
                t.consecutive_failed,
                t.failed_evaluations,
                t.kkt_error,
                t.qp_steps,
            ),
            state,
            (
                _int(state.step + 1),
                state.done | failed_now,
                _int(jnp.where(failed_now, failure, state.status)),
                _int(jnp.where(moved, 0, consecutive)),
                _int(state.failed_evaluations + failed),
                _error(state, bounds),
                _int(state.qp_steps + qp_steps),
            ),
        )

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One SQP step: the QP, the line search, the quasi-Newton update."""
        _, closures, bounds = self._prepare(problem, state.x)
        bounded = getattr(problem, "bounds", None) is not None
        return self._iterate(closures, bounds, state, (args, state.scales), bounded)

    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """End on the scaled optimality error, the budget, or a failure."""
        return InteriorPoint.terminate(self, problem, state, args)

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
        bounded = getattr(problem, "bounds", None) is not None

        def body(current):
            new = self._with_termination(
                self._iterate(closures, bounds, current, (args, current.scales), bounded)
            )
            return _select(current.done, current, new)

        state = lax.while_loop(lambda current: ~current.done, body, state)
        return state.x, _record(state, bounds)


def _record(state: SQPState, bounds: Any) -> SolveRecord:
    """The `SolveRecord` of a finished state, unscaled; `mu` is 0."""
    scales = state.scales
    f = scales.objective
    record = SolveRecord(
        multipliers_equality=state.lam_e * scales.equality / f,
        multipliers_inequality=state.lam_i * scales.inequality / f,
        slacks=jnp.maximum(-state.evaluation.inequality, 0.0) / scales.inequality,
        bound_multipliers=(state.z_lower / f, state.z_upper / f),
        mu=jnp.zeros((), state.x.dtype),
        steps=state.step,
        failed_evaluations=state.failed_evaluations,
        kkt_error=state.kkt_error,
        constraint_violation=constraint_violation(state.evaluation, scales),
        status=state.status,
        degenerate=jnp.asarray(False),
    )
    return eqx.tree_at(
        lambda r: r.degenerate,
        record,
        active_set(
            state.x,
            record,
            bounds,
            state.evaluation.jacobian_equality,
            state.evaluation.jacobian_inequality,
        ).degenerate,
    )
