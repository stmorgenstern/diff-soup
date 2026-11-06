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

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.solvers import _hessian, _projected_newton
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
from diff_soup.optimization.solvers._optimality import constraint_violation
from diff_soup.optimization.solvers._protocol import _int

#: Conn, Gould and Toint's (LANCELOT) largest penalty; at it a failed violation test ends the run.
RHO_MAX = 1e12
#: Ours: the first inner trust-region radius.
RADIUS_INIT = 1.0
#: LANCELOT's (Nocedal and Wright, Framework 17.4) exponents: after a multiplier update
#: `eta <- eta / rho**ETA_POWER`, after a penalty increase `eta <- rho**-ETA_RESET`.
ETA_POWER = 0.9
ETA_RESET = 0.1
#: Ours: the least inner radius a new inner problem starts from, relative to `1 + |x|`.
RADIUS_FLOOR = 1e-4


class AugmentedLagrangianState(CoreState):
    """The carry of [AugmentedLagrangian][diff_soup.optimization.AugmentedLagrangian], in the scaled problem.

    The core's `lam_e` and `lam_i` are the first-order multipliers at `x`, `shift_e + rho c_E`
    and `max(0, shift_i + rho c_I)`, which the record reports.

    Attributes:
        shift_e: The equality multipliers the next inner problem's penalty is shifted by.
        shift_i: The inequality multipliers the next inner problem's penalty is shifted by.
        rho: The penalty.
        omega: The inner tolerance on the projected gradient.
        eta: The violation below which the multipliers are updated.
        radius: The inner trust-region radius, carried from one inner problem to the next.
    """

    shift_e: Float[Array, "n_eq"]
    shift_i: Float[Array, "n_ineq"]
    rho: Float[Array, ""]
    omega: Float[Array, ""]
    eta: Float[Array, ""]
    radius: Float[Array, ""]


def _estimates(equality, inequality, lam_e, lam_i, rho):
    """The first-order multipliers `lam_e + rho c_E` and `max(0, lam_i + rho c_I)`."""
    return lam_e + rho * equality, jnp.maximum(lam_i + rho * inequality, 0.0)


def _penalty(value, equality, inequality, lam_e, lam_i, rho):
    """The PHR augmented Lagrangian from the objective and the constraint values."""
    shifted = jnp.maximum(lam_i + rho * inequality, 0.0)
    return (
        value
        + lam_e @ equality
        + 0.5 * rho * (equality @ equality)
        + (shifted @ shifted - lam_i @ lam_i) / (2.0 * rho)
    )


def _violation(equality, inequality, lam_i, rho):
    """Birgin and Martínez's measure `max(|c_E|, |min(-c_I, lam_I / rho)|)`."""
    parts = jnp.concatenate([jnp.abs(equality), jnp.abs(jnp.minimum(-inequality, lam_i / rho))])
    return jnp.max(parts, initial=0.0)


class AugmentedLagrangian(ConstrainedSolver):
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

    def _check(self, raw: Closures) -> None:
        """Refuse a Hessian mode the problem cannot serve, or a penalty that cannot grow."""
        super()._check(raw)
        if not (self.rho_init > 0.0 and self.rho_factor > 1.0):
            raise ConfigurationError("rho_init must be positive and rho_factor above 1.")

    @property
    def _omega_floor(self) -> float:
        """The least inner tolerance: a tenth of `tol`, so the outer test can pass."""
        return 0.1 * self.tol

    def _place(self, x0: Float[Array, "n"], bounds: Bounds, warm: bool) -> Float[Array, "n"]:
        """`x0` projected into its bounds."""
        return _projected_newton.project(x0, bounds)

    def _begin(self, closures: Closures, bounds: Bounds, start: Start) -> AugmentedLagrangianState:
        """Zero shifts or the warm start's multipliers, the first penalty, and the estimates."""
        dtype = start.x.dtype
        fields = core_fields(start, closures, self.hessian)
        rho = jnp.asarray(self.rho_init, dtype)
        constrained = closures.n_equality + closures.n_inequality > 0
        state = AugmentedLagrangianState(
            **fields,
            shift_e=fields["lam_e"],
            shift_i=fields["lam_i"],
            rho=rho,
            omega=1.0 / rho if constrained else jnp.asarray(self._omega_floor, dtype),
            eta=rho**-ETA_RESET,
            radius=jnp.asarray(RADIUS_INIT, dtype),
        )
        return self._with_estimates(state, start.evaluation, None, bounds)

    def _with_estimates(self, state, evaluation, gradient, bounds):
        """`state` with the estimates, bound multipliers and optimality error at its `x`."""
        lam_e, lam_i = _estimates(
            evaluation.equality, evaluation.inequality, state.shift_e, state.shift_i, state.rho
        )
        if gradient is None:
            gradient = (
                evaluation.gradient
                + evaluation.jacobian_equality.T @ lam_e
                + evaluation.jacobian_inequality.T @ lam_i
            )
        z_lower, z_upper = _projected_newton.bound_multipliers(state.x, gradient, bounds)
        error = slack_free_error(evaluation, state.x, lam_e, lam_i, z_lower, z_upper, bounds)
        return eqx.tree_at(
            lambda s: (s.lam_e, s.lam_i, s.z_lower, s.z_upper, s.kkt_error),
            state,
            (lam_e, lam_i, z_lower, z_upper, error),
        )

    def _converged(self, state: AugmentedLagrangianState) -> Bool[Array, ""]:
        """The optimality error below `tol` and the unscaled violation below `feasibility_tol`."""
        violation = constraint_violation(state.evaluation, state.scales)
        return (state.kkt_error <= self.tol) & (violation <= self.feasibility_tol)

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
        return value, gradient, curvature + penalty

    def _iterate(
        self, closures: Closures, bounds: Bounds, state: AugmentedLagrangianState, args: PyTree
    ) -> AugmentedLagrangianState:
        """One outer step: the inner minimization, then the multiplier or penalty update."""
        lam_e, lam_i, rho = state.shift_e, state.shift_i, state.rho
        gauss_newton = _hessian.uses_gauss_newton(self.hessian, closures)

        def penalty_at(x):
            value, equality, inequality, valid = trial(closures, x, args)
            return _penalty(value, equality, inequality, lam_e, lam_i, rho), valid

        def derivatives(x_new, x_old, aux):
            ev_old, matrix, updates = aux
            ev = evaluate(closures, x_new, args, gauss_newton)
            estimate_e, estimate_i = _estimates(ev.equality, ev.inequality, lam_e, lam_i, rho)
            matrix, updates = _hessian.update(
                self.hessian,
                closures,
                matrix,
                updates,
                x_new - x_old,
                lagrangian_secant(ev, ev_old, estimate_e, estimate_i),
                ev,
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
            jnp.maximum(
                state.radius, RADIUS_FLOOR * (1.0 + jnp.max(jnp.abs(state.x), initial=0.0))
            ),
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
        exhausted = end.consecutive_failed >= self.max_failed_evaluations
        failure = jnp.where(
            exhausted, TerminationStatus.SOLVER_FAILED, TerminationStatus.INFEASIBLE
        )
        ended = exhausted | stuck
        return eqx.tree_at(
            lambda s: (
                s.shift_e,
                s.shift_i,
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
                jnp.where(update, state.lam_e, lam_e),
                jnp.where(update, state.lam_i, lam_i),
                jnp.where(update, rho, raised),
                jnp.maximum(jnp.where(update, state.omega / rho, 1.0 / raised), self._omega_floor),
                jnp.maximum(
                    jnp.where(update, state.eta / rho**ETA_POWER, raised**-ETA_RESET),
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
