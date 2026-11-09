"""Sequential quadratic programming on the interior point's KKT step and filter line search."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, PyTree

from diff_soup._adjoints import AbstractAdjoint, ForwardAdjoint
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.solvers import _filter, _hessian
from diff_soup.optimization.solvers._barrier import Bounds
from diff_soup.optimization.solvers._closures import Closures, evaluate, trial
from diff_soup.optimization.solvers._core import (
    ConstrainedSolver,
    CoreState,
    Start,
    core_fields,
    lagrangian_secant,
    slack_free_error,
)
from diff_soup.optimization.solvers._line_search import filter_line_search
from diff_soup.optimization.solvers._protocol import _int, _select
from diff_soup.optimization.solvers.interior_point import InteriorPoint

#: Ours: how far inside the shifted bounds each QP starts.
QP_PUSH = 1e-8


class SQPState(CoreState):
    """The carry of [SQP][diff_soup.optimization.SQP], in the scaled problem.

    Attributes:
        filt: The line-search filter on `(theta, f)`.
        theta_min: The switching threshold on the constraint violation.
        theta_max: The largest constraint violation a trial may have.
    """

    filt: _filter.Filter
    theta_min: Float[Array, ""]
    theta_max: Float[Array, ""]


def _violation(equality: Float[Array, "n_eq"], inequality: Float[Array, "n_ineq"]):
    """`theta = |c_E|_1 + |max(c_I, 0)|_1`."""
    return jnp.sum(jnp.abs(equality)) + jnp.sum(jnp.maximum(inequality, 0.0))


def _qp_closures(n_eq: int, n_ineq: int) -> Closures:
    """The closures of the QP in the step `d`, whose `args` are `(g, B, c_E, J_E, c_I, J_I)`."""

    def value_and_gradient(d, qp):
        g, b = qp[0], qp[1]
        return g @ d + 0.5 * d @ (b @ d), g + b @ d

    def constraints_and_jacobians(d, qp):
        _, _, c_e, j_e, c_i, j_i = qp
        return c_e + j_e @ d, c_i + j_i @ d, j_e, j_i

    def evaluate(d, qp):
        return (value_and_gradient(d, qp)[0], *constraints_and_jacobians(d, qp)[:2])

    def evaluate_with_derivatives(d, qp):
        return (*value_and_gradient(d, qp), *constraints_and_jacobians(d, qp))

    return Closures(
        evaluate=evaluate,
        evaluate_with_derivatives=evaluate_with_derivatives,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=lambda d, le, li, qp: qp[1],
        constraint_hessian=None,
        gauss_newton=None,
        is_valid=lambda d, qp: jnp.asarray(True),
        n_equality=n_eq,
        n_inequality=n_ineq,
        bounded=True,
    )


class SQP(ConstrainedSolver):
    """Sequential quadratic programming with a filter line search.

    Each step solves the QP of the model Hessian and the linearized constraints by the interior
    point. A QP that does not solve ends the run with `SOLVER_FAILED`, and the record's
    `subproblem_status` says how it ended. A warm start takes an earlier record's multipliers,
    so it suits a sequence of nearby problems.

    Args:
        tol: The scaled optimality error at which the run ends, by default `max(1e-8, 100 * eps)` for the iterate's dtype: `1e-8` in float64 and `1.2e-5` in float32.
        max_steps: The step budget, `100` by default.
        hessian: `"auto"` (default), `"bfgs"`, `"exact"` or `"gauss-newton"`, as for `InteriorPoint`.
        qp_tol: The scaled optimality error at which each QP solve ends, by default `max(1e-10, 100 * eps)`: `1e-10` in float64 and `1.2e-5` in float32. A run's `tol` below `tol` lowers it by the same factor.
        qp_max_steps: The step budget of each QP solve, `50` by default.
        max_failed_evaluations: Failed trial evaluations in a row before the run ends with `SOLVER_FAILED`, `10` by default.
        kkt: The [KKTSolve][diff_soup.optimization.solvers.KKTSolve] backend, `DenseKKT()` by default.
        adjoint: How the solution is differentiated in `args`. [ForwardAdjoint][diff_soup.ForwardAdjoint] (default) is one forward rule: `jax.jvp` and `jax.jacfwd` cost one KKT solve per tangent, and `jax.grad` transposes it. [ImplicitAdjoint][diff_soup.nonlinear.ImplicitAdjoint] is one reverse rule: `jax.grad` and `jax.vjp` cost one transposed KKT solve whatever the size of `args`, and forward mode raises. Choose it for a large `args`, or for closures that run a solve under `CheckpointAdjoint(reverse_over_reverse=True)`, with `hessian="exact"` or `"bfgs"` there, since Gauss-Newton takes the residual's Jacobian in forward mode.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})
    _default_tolerances: ClassVar[dict[str, float]] = {"tol": 1e-8, "qp_tol": 1e-10}

    tol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=100)
    hessian: str = eqx.field(static=True, default="auto")
    qp_tol: float | None = eqx.field(static=True, default=None)
    qp_max_steps: int = eqx.field(static=True, default=50)
    max_failed_evaluations: int = eqx.field(static=True, default=10)
    kkt: KKTSolve = eqx.field(static=True, default_factory=DenseKKT)
    adjoint: AbstractAdjoint = eqx.field(static=True, default_factory=ForwardAdjoint)

    @property
    def _kkt(self) -> KKTSolve:
        """The backend of the implicit derivative's KKT system: `kkt`."""
        return self.kkt

    def _place(self, x0: Float[Array, "n"], bounds: Bounds, warm: bool) -> Float[Array, "n"]:
        """`x0` clipped into its bounds."""
        return jnp.clip(
            x0,
            jnp.where(bounds.has_lower, bounds.lower, -jnp.inf),
            jnp.where(bounds.has_upper, bounds.upper, jnp.inf),
        )

    def _begin(self, closures: Closures, bounds: Bounds, start: Start) -> SQPState:
        """The filter and its thresholds from the first point's violation."""
        scale = jnp.maximum(1.0, _violation(start.evaluation.equality, start.evaluation.inequality))
        state = SQPState(
            **core_fields(start, closures, self.hessian),
            filt=_filter.empty_filter(start.x.dtype),
            theta_min=_filter.THETA_MIN_FACTOR * scale,
            theta_max=_filter.THETA_MAX_FACTOR * scale,
        )
        return eqx.tree_at(lambda t: t.kkt_error, state, self._error(state, bounds))

    def _error(self, state: SQPState, bounds: Bounds) -> Float[Array, ""]:
        """The slack-free scaled optimality error at the state."""
        return slack_free_error(
            state.evaluation,
            state.x,
            state.lam_e,
            state.lam_i,
            state.z_lower,
            state.z_upper,
            bounds,
        )

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

    def _solve_qp(
        self, closures: Closures, bounds: Bounds, state: SQPState, hessian: Float[Array, "n n"]
    ) -> tuple:
        """The QP step `d`, its multipliers `(lam_e, lam_i, z_lower, z_upper)`, its steps and status."""
        ev = state.evaluation
        n = state.x.shape[0]
        n_eq, n_ineq = closures.n_equality, closures.n_inequality
        dtype = state.x.dtype
        if n_ineq == 0 and not closures.bounded:
            factor, _ = self.kkt.factor_regularized(
                hessian,
                ev.jacobian_equality,
                jnp.zeros((n_eq,), dtype),
                0.0,
                jnp.zeros((), dtype),
            )
            d, lam = self.kkt.solve_factored(factor, -ev.gradient, -ev.equality)
            zeros = jnp.zeros((n,), dtype)
            status = jnp.where(
                factor.ok, TerminationStatus.KKT_TOL, TerminationStatus.SOLVER_FAILED
            )
            return d, lam, jnp.zeros((0,), dtype), zeros, zeros, _int(1), _int(status)
        qp = _qp_closures(n_eq, n_ineq)
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
        d0 = jnp.clip(
            jnp.zeros((n,), dtype),
            jnp.where(qp_bounds.has_lower, qp_bounds.lower + QP_PUSH, -jnp.inf),
            jnp.where(qp_bounds.has_upper, qp_bounds.upper - QP_PUSH, jnp.inf),
        )
        start = inner._start(
            qp,
            qp_bounds,
            d0,
            qp_args,
            self.qp_tol * jnp.minimum(1.0, state.tol / self.tol),
            z_lower=jnp.where(qp_bounds.has_lower, 1.0, 0.0).astype(dtype),
            z_upper=jnp.where(qp_bounds.has_upper, 1.0, 0.0).astype(dtype),
        )
        end = lax.while_loop(
            lambda s: ~s.done, lambda s: inner._advance(qp, qp_bounds, s, qp_args), start
        )
        return end.x, end.lam_e, end.lam_i, end.z_lower, end.z_upper, end.step, end.status

    def _iterate(
        self, closures: Closures, bounds: Bounds, state: SQPState, args: PyTree
    ) -> SQPState:
        """One SQP step: the QP, the filter line search, the quasi-Newton update."""
        ev = state.evaluation
        x = state.x
        hessian = _hessian.lagrangian_hessian(
            self.hessian, closures, x, state.lam_e, state.lam_i, args, ev, state.quasi_newton
        )
        d, qp_e, qp_i, qp_zl, qp_zu, qp_steps, qp_status = self._solve_qp(
            closures, bounds, state, hessian
        )
        qp_ok = qp_status == TerminationStatus.KKT_TOL
        theta = _violation(ev.equality, ev.inequality)

        def trial_point(x_trial, s_trial):
            value, equality, inequality, valid = trial(closures, x_trial, args)
            return _violation(equality, inequality), value, equality, valid

        tiny = jnp.max(jnp.abs(d) / (1.0 + jnp.abs(x)), initial=0.0) < 10.0 * jnp.finfo(x.dtype).eps
        empty = jnp.zeros((0,), x.dtype)
        search = filter_line_search(
            trial_point,
            x,
            empty,
            d,
            empty,
            alpha_max=jnp.ones((), x.dtype),
            theta=theta,
            phi=ev.value,
            slope=ev.gradient @ d,
            filt=state.filt,
            theta_min=state.theta_min,
            theta_max=state.theta_max,
            consecutive_failed=state.consecutive_failed,
            max_failed=self.max_failed_evaluations,
            accept_full=tiny,
        )
        alpha = search.alpha
        accepted = search.accepted & qp_ok
        x_new = x + alpha * d
        lam_e = state.lam_e + alpha * (qp_e - state.lam_e)
        lam_i = state.lam_i + alpha * (qp_i - state.lam_i)
        z_lower = state.z_lower + alpha * (qp_zl - state.z_lower)
        z_upper = state.z_upper + alpha * (qp_zu - state.z_upper)
        ev_new = evaluate(closures, x_new, args, _hessian.uses_gauss_newton(self.hessian, closures))
        moved = accepted & ev_new.finite
        quasi_newton, updates = _hessian.update(
            self.hessian,
            closures,
            state.quasi_newton,
            state.quasi_newton_updates,
            x_new - x,
            lagrangian_secant(ev_new, ev, lam_e, lam_i),
            ev_new,
        )
        filt = _select(
            search.f_type,
            state.filt,
            _filter.augment(
                state.filt,
                (1.0 - _filter.GAMMA_THETA) * theta,
                ev.value - _filter.GAMMA_PHI * theta,
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
        consecutive = search.consecutive_failed + (accepted & ~ev_new.finite)
        exhausted = consecutive >= self.max_failed_evaluations
        failed_now = ~qp_ok | (~moved & (~search.accepted | exhausted))
        failure = jnp.where(
            ~qp_ok | exhausted,
            TerminationStatus.SOLVER_FAILED,
            TerminationStatus.RESTORATION_FAILED,
        )
        return eqx.tree_at(
            lambda t: (
                t.step,
                t.done,
                t.status,
                t.consecutive_failed,
                t.failed_evaluations,
                t.kkt_error,
                t.inner_steps,
                t.subproblem_status,
            ),
            state,
            (
                _int(state.step + 1),
                state.done | failed_now,
                _int(jnp.where(failed_now, failure, state.status)),
                _int(jnp.where(moved, 0, consecutive)),
                _int(state.failed_evaluations + search.failed),
                self._error(state, bounds),
                _int(state.inner_steps + qp_steps),
                _int(jnp.where(qp_ok, state.subproblem_status, qp_status)),
            ),
        )
