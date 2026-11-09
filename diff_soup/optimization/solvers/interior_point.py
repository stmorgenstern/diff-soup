"""A primal-dual interior point method with a filter line search, after Wächter and Biegler 2006."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, PyTree

from diff_soup._adjoints import AbstractAdjoint, ForwardAdjoint
from diff_soup._errors import ConfigurationError
from diff_soup.optimization._abc import OptimizationProblem
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.solvers import _filter, _hessian, _restoration
from diff_soup.optimization.solvers._barrier import Bounds, distances, push_into_bounds
from diff_soup.optimization.solvers._barrier_problem import (
    InteriorPointState,
    phase_error,
    regular_error,
    violation,
)
from diff_soup.optimization.solvers._barrier_step import barrier_step
from diff_soup.optimization.solvers._closures import (
    Closures,
    Evaluation,
    Scales,
    evaluate,
    unit_scales,
)
from diff_soup.optimization.solvers._core import ConstrainedSolver, Start, core_fields
from diff_soup.optimization.solvers._protocol import _select

#: Ipopt's `barrier_tol_factor`: the barrier problem is solved to `KAPPA_EPS * mu`.
KAPPA_EPS = 10.0
#: Ipopt's `constr_mult_init_max`: larger least-squares multipliers start at 0.
LAMBDA_MAX = 1e3
#: Ours: how far inside its bounds a warm start's point, slacks and multipliers start.
WARM_PUSH = 1e-6


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
        adjoint: How the solution is differentiated in `args`. [ForwardAdjoint][diff_soup.ForwardAdjoint] (default) is one forward rule: `jax.jvp` and `jax.jacfwd` cost one KKT solve per tangent, and `jax.grad` transposes it. [ImplicitAdjoint][diff_soup.nonlinear.ImplicitAdjoint] is one reverse rule: `jax.grad` and `jax.vjp` cost one transposed KKT solve whatever the size of `args`, and forward mode raises. Choose it for a large `args`, or for closures that run a solve under `CheckpointAdjoint(reverse_over_reverse=True)`, with `hessian="exact"` or `"bfgs"` there, since Gauss-Newton takes the residual's Jacobian in forward mode.
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
    adjoint: AbstractAdjoint = eqx.field(static=True, default_factory=ForwardAdjoint)

    @property
    def _kkt(self) -> KKTSolve:
        """The backend of the implicit derivative's KKT system: `kkt`."""
        return self.kkt

    @property
    def _mu_floor(self) -> float:
        """The least barrier parameter: `mu_min`, or below `tol` when `tol` is tighter."""
        return min(self.mu_min, self.tol / (KAPPA_EPS + 1.0))

    def _restores(self, closures: Closures) -> bool:
        """Whether a failed line search can start the restoration phase."""
        return self.restoration and closures.n_equality + closures.n_inequality > 0

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
        slack_push=None,
        warm=False,
    ) -> InteriorPointState:
        """The state at a point inside the bounds, from its evaluation.

        `lam_e = None` takes the least-squares multipliers and `v = None` takes 1. `warm` takes
        `mu` from the point's complementarity, between the floor and `mu_init`.
        """
        x, evaluation = start.x, start.evaluation
        dtype = x.dtype
        push = self.bound_push if slack_push is None else slack_push
        s = jnp.maximum(-evaluation.inequality, push)
        v = jnp.ones_like(s) if v is None else v
        if lam_e is None:
            lam_e, lam_i = self._least_squares_multipliers(evaluation, z_lower, z_upper, v)
        mu = self.mu_init
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
        theta, _ = violation(evaluation.equality, evaluation.inequality, s)
        scale = jnp.maximum(1.0, theta)
        fields = core_fields(
            start._replace(warm=(lam_e, lam_i, z_lower, z_upper)), closures, self.hessian
        )
        tau = jnp.maximum(self.tau_min, 1.0 - mu)
        filt = _filter.empty_filter(dtype)
        theta_min = _filter.THETA_MIN_FACTOR * scale
        theta_max = _filter.THETA_MAX_FACTOR * scale
        zero = jnp.zeros((), dtype)
        saved = _restoration.Saved(
            z_lower,
            z_upper,
            v,
            mu,
            tau,
            filt,
            theta_min,
            theta_max,
            zero,
            fields["quasi_newton"],
            fields["quasi_newton_updates"],
            evaluation,
        )
        m = closures.n_equality + closures.n_inequality
        state = InteriorPointState(
            **fields,
            slacks=s,
            v=v,
            mu=mu,
            tau=tau,
            filt=filt,
            theta_min=theta_min,
            theta_max=theta_max,
            delta_w_last=zero,
            restoration=_restoration.idle(x, s, m),
            saved=saved,
        )
        return dataclasses.replace(state, kkt_error=regular_error(state, 0.0, bounds))

    def _start(self, closures, bounds, x, args, **multipliers) -> InteriorPointState:
        """The state at `x` on closures used as given (a QP), unscaled."""
        evaluation = evaluate(closures, x, args, _hessian.uses_gauss_newton(self.hessian, closures))
        start = Start(
            x, evaluation, unit_scales(closures, x.dtype), None, (self.max_steps, self.tol)
        )
        return self._with_termination(self._at(closures, bounds, start, **multipliers))

    def _slacks(self, state: InteriorPointState) -> Float[Array, "n_ineq"]:
        """The record's slacks: the state's own."""
        return state.slacks

    def _mu(self, state: InteriorPointState) -> Float[Array, ""]:
        """The record's barrier parameter: the final `mu`."""
        return state.mu

    def _converged(self, state: InteriorPointState) -> Bool[Array, ""]:
        """Whether the original problem's scaled optimality error is below the run's `tol`."""
        return (state.kkt_error <= state.tol) & ~state.restoration.on

    def _update_barrier(
        self, state: InteriorPointState, bounds: Bounds, restores: bool
    ) -> InteriorPointState:
        """Ipopt's monotone update: reduce `mu` while the barrier problem is solved to `kappa_eps mu`."""
        floor = self._mu_floor

        def cond(carry):
            mu, _ = carry
            return (phase_error(state, mu, bounds, restores) <= KAPPA_EPS * mu) & (mu > floor)

        def body(carry):
            mu, _ = carry
            return jnp.maximum(floor, jnp.minimum(self.kappa_mu * mu, mu**self.theta_mu)), True

        mu, changed = lax.while_loop(cond, body, (state.mu, jnp.asarray(False)))
        filt = _select(changed, _filter.empty_filter(mu.dtype), state.filt)
        tau = jnp.maximum(self.tau_min, 1.0 - mu)
        return dataclasses.replace(state, mu=mu, tau=tau, filt=filt)

    def _iterate(
        self, closures: Closures, bounds: Bounds, state: InteriorPointState, args: PyTree
    ) -> InteriorPointState:
        """One barrier step of the phase the state is in, and the phase changes it makes."""
        restores = self._restores(closures)
        state = self._update_barrier(state, bounds, restores)
        return barrier_step(self, closures, bounds, state, args, restores)
