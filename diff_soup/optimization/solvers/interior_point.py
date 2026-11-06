"""A primal-dual interior point method with a filter line search, after Wächter and Biegler 2006."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Float, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._abc import OptimizationProblem
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.solvers import _filter, _hessian, _restoration
from diff_soup.optimization.solvers._barrier import (
    Bounds,
    barrier_gradient,
    barrier_value,
    bound_multiplier_steps,
    distances,
    dual_fraction_to_boundary,
    fraction_to_boundary,
    primal_sigma,
    push_into_bounds,
    safeguard_multipliers,
)
from diff_soup.optimization.solvers._closures import (
    Closures,
    Evaluation,
    Scales,
    evaluate,
    trial,
    unit_scales,
)
from diff_soup.optimization.solvers._core import (
    ConstrainedSolver,
    CoreState,
    Start,
    core_fields,
    lagrangian_secant,
)
from diff_soup.optimization.solvers._line_search import Correction, filter_line_search
from diff_soup.optimization.solvers._optimality import optimality_error
from diff_soup.optimization.solvers._protocol import _int, _select
from diff_soup.utils.lanes import live_lane_cond

#: Ipopt's `barrier_tol_factor`: the barrier problem is solved to `KAPPA_EPS * mu`.
KAPPA_EPS = 10.0
#: Ipopt's `constr_mult_init_max`: larger least-squares multipliers start at 0.
LAMBDA_MAX = 1e3
#: Ours: how far inside its bounds a warm start's point, slacks and multipliers start.
WARM_PUSH = 1e-6
#: Ipopt's `bound_mult_reset_threshold`: after restoration, larger bound multipliers reset to 1.
BOUND_MULTIPLIER_RESET = 1e3
#: Ipopt's `jacobian_regularization_value` and `jacobian_regularization_exponent`:
#: `delta_c = DELTA_C * mu**KAPPA_C` for a rank-deficient constraint Jacobian.
DELTA_C = 1e-8
KAPPA_C = 0.25


class InteriorPointState(CoreState):
    """The carry of [InteriorPoint][diff_soup.optimization.InteriorPoint], in the scaled problem.

    Attributes:
        slacks: The inequality slacks `s >= 0`, with `c_I(x) + s = 0` at a solution.
        v: The multipliers of `s >= 0`.
        mu: The barrier parameter.
        tau: The fraction to the boundary.
        filt: The line-search filter.
        theta_min: The switching threshold on the constraint violation.
        theta_max: The largest constraint violation a trial may have.
        delta_w_last: The last nonzero primal regularization.
    """

    slacks: Float[Array, "n_ineq"]
    v: Float[Array, "n_ineq"]
    mu: Float[Array, ""]
    tau: Float[Array, ""]
    filt: _filter.Filter
    theta_min: Float[Array, ""]
    theta_max: Float[Array, ""]
    delta_w_last: Float[Array, ""]


def _error(state: InteriorPointState, mu: Float[ArrayLike, ""], bounds: Bounds) -> Float[Array, ""]:
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


def _theta(
    equality: Float[Array, "n_eq"], inequality: Float[Array, "n_ineq"], s: Float[Array, "n_ineq"]
) -> tuple[Float[Array, ""], Float[Array, "m"]]:
    """`(theta, c)`: the 1-norm of `c = [c_E; c_I + s]`, and `c`."""
    constraint = jnp.concatenate([equality, inequality + s])
    return jnp.sum(jnp.abs(constraint)), constraint


class InteriorPoint(ConstrainedSolver):
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
        kkt: The [KKTSolve][diff_soup.optimization.solvers.KKTSolve] backend, `DenseKKT()` by default.
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
    kkt: KKTSolve = eqx.field(static=True, default_factory=DenseKKT)

    @property
    def _mu_floor(self) -> float:
        """The least barrier parameter: `mu_min`, or below `tol` when `tol` is tighter."""
        return min(self.mu_min, self.tol / (KAPPA_EPS + 1.0))

    # -- the start -------------------------------------------------------------------------

    def _check(self, raw: Closures) -> None:
        """Refuse a Hessian mode the problem cannot serve, or an unknown scaling."""
        super()._check(raw)
        if self.scaling not in ("gradient", "none"):
            raise ConfigurationError(f"scaling must be 'gradient' or 'none'; got {self.scaling!r}.")

    def _scales(
        self, problem: OptimizationProblem, raw: Closures, evaluation: Evaluation
    ) -> Scales:
        """The problem's own scales, else the gradient-based ones, else 1."""
        if self.scaling == "none":
            return unit_scales(raw, evaluation.gradient.dtype)
        return super()._scales(problem, raw, evaluation)

    def _place(self, x0: Float[Array, "n"], bounds: Bounds, warm: bool) -> Float[Array, "n"]:
        """`x0` pushed inside its bounds: by `bound_push` and `bound_frac`, or `WARM_PUSH` warm."""
        if warm:
            return push_into_bounds(x0, bounds, WARM_PUSH, WARM_PUSH)
        return push_into_bounds(x0, bounds, self.bound_push, self.bound_frac)

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

    def _begin(self, closures: Closures, bounds: Bounds, start: Start) -> InteriorPointState:
        """Slacks from the inequalities, and the least-squares or the warm start's multipliers."""
        dtype = start.x.dtype
        if start.warm is None:
            return self._at(
                closures,
                bounds,
                start,
                z_lower=jnp.where(bounds.has_lower, 1.0, 0.0).astype(dtype),
                z_upper=jnp.where(bounds.has_upper, 1.0, 0.0).astype(dtype),
            )
        lam_e, lam_i, z_lower, z_upper = start.warm
        return self._at(
            closures,
            bounds,
            start,
            lam_e=lam_e,
            lam_i=lam_i,
            z_lower=jnp.where(bounds.has_lower, jnp.maximum(z_lower, WARM_PUSH), 0.0),
            z_upper=jnp.where(bounds.has_upper, jnp.maximum(z_upper, WARM_PUSH), 0.0),
            v=jnp.maximum(lam_i, WARM_PUSH),
            slack_push=WARM_PUSH,
            warm=True,
        )

    def _at(
        self,
        closures,
        bounds,
        start,
        *,
        z_lower,
        z_upper,
        lam_e=None,
        lam_i=None,
        v=None,
        mu=None,
        slack_push=None,
        warm=False,
    ) -> InteriorPointState:
        """The state at a point inside the bounds, from its evaluation.

        `lam_e = None` takes the least-squares multipliers, `v = None` takes 1 and `mu = None`
        takes `mu_init`. `warm` takes `mu` from the point's complementarity instead, between
        the floor and `mu`.
        """
        x, evaluation = start.x, start.evaluation
        dtype = x.dtype
        push = self.bound_push if slack_push is None else slack_push
        s = jnp.maximum(-evaluation.inequality, push)
        v = jnp.ones_like(s) if v is None else v
        if lam_e is None:
            lam_e, lam_i = self._least_squares_multipliers(evaluation, z_lower, z_upper, v)
        mu = self.mu_init if mu is None else mu
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
            mu = jnp.clip(jnp.sum(products) / jnp.maximum(count, 1), self._mu_floor, mu)
        mu = jnp.asarray(mu, dtype)
        theta, _ = _theta(evaluation.equality, evaluation.inequality, s)
        scale = jnp.maximum(1.0, theta)
        fields = core_fields(
            start._replace(warm=(lam_e, lam_i, z_lower, z_upper)), closures, self.hessian
        )
        state = InteriorPointState(
            **fields,
            slacks=s,
            v=v,
            mu=mu,
            tau=jnp.maximum(self.tau_min, 1.0 - mu),
            filt=_filter.empty_filter(dtype),
            theta_min=_filter.THETA_MIN_FACTOR * scale,
            theta_max=_filter.THETA_MAX_FACTOR * scale,
            delta_w_last=jnp.zeros((), dtype),
        )
        return eqx.tree_at(lambda t: t.kkt_error, state, _error(state, 0.0, bounds))

    def _start(self, closures, bounds, x, args, **multipliers) -> InteriorPointState:
        """The state at `x` on closures used as given (a QP, the restoration problem), unscaled."""
        evaluation = evaluate(closures, x, args, _hessian.uses_gauss_newton(self.hessian, closures))
        start = Start(x, evaluation, unit_scales(closures, x.dtype), None)
        return self._with_termination(self._at(closures, bounds, start, **multipliers))

    # -- one step --------------------------------------------------------------------------

    def _slacks(self, state: InteriorPointState) -> Float[Array, "n_ineq"]:
        """The record's slacks: the state's own."""
        return state.slacks

    def _mu(self, state: InteriorPointState) -> Float[Array, ""]:
        """The record's barrier parameter: the final `mu`."""
        return state.mu

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

    def _iterate(
        self, closures: Closures, bounds: Bounds, state: InteriorPointState, args: PyTree
    ) -> InteriorPointState:
        """One barrier step: the KKT solve, fraction to the boundary, the filter line search."""
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
            primal_block, jacobian, dual_diagonal, DELTA_C * mu**KAPPA_C, state.delta_w_last
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
            x,
            s,
            dx,
            ds,
            alpha_max=alpha_max,
            theta=theta,
            phi=phi,
            slope=slope,
            filt=state.filt,
            theta_min=state.theta_min,
            theta_max=state.theta_max,
            consecutive_failed=state.consecutive_failed,
            max_failed=self.max_failed_evaluations,
            accept_full=tiny & factor.ok,
            correction=Correction(direction, constraint, dlam, tau, bounds),
        )
        searched = search.accepted & factor.ok
        exhausted = search.consecutive_failed >= self.max_failed_evaluations
        filt_current = _filter.augment(
            state.filt, (1.0 - _filter.GAMMA_THETA) * theta, phi - _filter.GAMMA_PHI * theta
        )

        alpha = search.alpha
        dz_lower, dz_upper, dv = bound_multiplier_steps(
            x, search.dx, s, search.ds, state.z_lower, state.z_upper, state.v, mu, bounds
        )
        alpha_dual = dual_fraction_to_boundary(
            state.z_lower, dz_lower, state.z_upper, dz_upper, state.v, dv, tau, bounds
        )
        regular = (
            x + alpha * search.dx,
            s + alpha * search.ds,
            lam + alpha * search.dlam,
            state.z_lower + alpha_dual * dz_lower,
            state.z_upper + alpha_dual * dz_upper,
            state.v + alpha_dual * dv,
        )

        need = ~searched & ~exhausted & (theta > self.tol)
        if self.restoration and has_rows:
            outcome = live_lane_cond(
                need,
                lambda: _restoration.restore(
                    self._inner(),
                    closures,
                    bounds,
                    state,
                    args,
                    run=need,
                    theta=theta,
                    filt=filt_current,
                ),
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
        quasi_newton, updates = _hessian.update(
            self.hessian,
            closures,
            state.quasi_newton,
            state.quasi_newton_updates,
            x_new - x,
            lagrangian_secant(evaluation_new, evaluation, lam_e_new, lam_i_new),
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
                t.inner_steps,
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
                _int(state.inner_steps + outcome.steps),
            ),
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
