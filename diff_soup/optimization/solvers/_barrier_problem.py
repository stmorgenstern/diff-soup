"""The interior point's state, and the barrier problem of each of its phases.

The regular phase solves the original problem's barrier problem; the restoration phase that
of [diff_soup.optimization.solvers._restoration][]. Both share the state's iterate and the
problem's evaluation, and differ in the merit function, the optimality error and the terms
of the Newton system, which this module selects by the phase flag.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from diff_soup.optimization.solvers import _filter, _restoration
from diff_soup.optimization.solvers._barrier import Bounds, barrier_gradient, barrier_value
from diff_soup.optimization.solvers._core import CoreState
from diff_soup.optimization.solvers._optimality import optimality_error


class InteriorPointState(CoreState):
    """The carry of [InteriorPoint][diff_soup.optimization.InteriorPoint], in the scaled problem.

    In the restoration phase the iterate, its multipliers, `mu`, the filter and `kkt_error`
    are the restoration problem's, and `saved` holds the regular phase's.

    Attributes:
        slacks: The inequality slacks `s >= 0`, with `c_I(x) + s = 0` at a solution.
        v: The multipliers of `s >= 0`.
        mu: The barrier parameter.
        tau: The fraction to the boundary.
        filt: The line-search filter.
        theta_min: The switching threshold on the constraint violation.
        theta_max: The largest constraint violation a trial may have.
        delta_w_last: The last nonzero primal regularization.
        restoration: The restoration phase's own variables.
        saved: The regular phase's fields during the restoration phase.
    """

    slacks: Float[Array, "n_ineq"]
    v: Float[Array, "n_ineq"]
    mu: Float[Array, ""]
    tau: Float[Array, ""]
    filt: _filter.Filter
    theta_min: Float[Array, ""]
    theta_max: Float[Array, ""]
    delta_w_last: Float[Array, ""]
    restoration: _restoration.Phase
    saved: _restoration.Saved


def violation(
    equality: Float[Array, "n_eq"], inequality: Float[Array, "n_ineq"], s: Float[Array, "n_ineq"]
) -> tuple[Float[Array, ""], Float[Array, "m"]]:
    """`(theta, c)`: the 1-norm of `c = [c_E; c_I + s]`, and `c`."""
    constraint = jnp.concatenate([equality, inequality + s])
    return jnp.sum(jnp.abs(constraint)), constraint


def regular_error(
    state: InteriorPointState, mu: Float[ArrayLike, ""], bounds: Bounds
) -> Float[Array, ""]:
    """The scaled optimality error `E_mu` of the original problem at the state."""
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


def phase_error(
    state: InteriorPointState, mu: Float[ArrayLike, ""], bounds: Bounds, restores: bool
) -> Float[Array, ""]:
    """`E_mu` of the problem the state's phase solves."""
    regular = regular_error(state, mu, bounds)
    if not restores:
        return regular
    restoration = _restoration.error(
        state.restoration,
        state.evaluation,
        state.x,
        state.slacks,
        jnp.concatenate([state.lam_e, state.lam_i]),
        state.z_lower,
        state.z_upper,
        state.v,
        jnp.asarray(mu, state.x.dtype),
        bounds,
    )
    return jnp.where(state.restoration.on, restoration, regular)


class Terms(eqx.Module):
    """The phase's pieces of the Newton system and of its multiplier steps."""

    grad_x: Float[Array, "n"]
    grad_s: Float[Array, "n_ineq"]
    r_x: Float[Array, "n"]
    r_s: Float[Array, "n_ineq"]
    inv_sigma_s: Float[Array, "n_ineq"]
    primal_extra: Float[Array, "n"]
    dual_extra: Float[Array, "m"]
    rhs_extra: Float[Array, "m"]
    r_p: Float[Array, "m"]
    r_n: Float[Array, "m"]
    inv_p: Float[Array, "m"]
    inv_n: Float[Array, "m"]
    grad_p: Float[Array, "m"]
    grad_n: Float[Array, "m"]


def terms(state: InteriorPointState, bounds: Bounds, restores: bool) -> Terms:
    """The gradients, right-hand sides and diagonals of the step in the state's phase.

    The regular phase keeps the slacks eliminated with `s / v`; the restoration phase adds the
    proximity term in `x` and `s` and eliminates `p` and `n` with `p / z_p` and `n / z_n`.
    """
    x, s, ev, mu = state.x, state.slacks, state.evaluation, state.mu
    lam = jnp.concatenate([state.lam_e, state.lam_i])
    jacobian = jnp.concatenate([ev.jacobian_equality, ev.jacobian_inequality])
    barrier_x, barrier_s = barrier_gradient(x, s, mu, bounds)
    m = lam.shape[0]
    zero_m = jnp.zeros((m,), x.dtype)
    if restores:
        phase = state.restoration
        on = phase.on
        prox_x, prox_s, diagonal_x, diagonal_s = _restoration.proximity(phase, x, s)
        grad_p, grad_n = _restoration.gradients(phase, mu)
        grad_x = jnp.where(on, prox_x, ev.gradient) + barrier_x
        grad_s = barrier_s + jnp.where(on, prox_s, 0.0)
        inv_sigma_s = jnp.where(on, 1.0 / (diagonal_s + state.v / s), s / state.v)
        inv_p, inv_n = phase.p / phase.z_p, phase.n / phase.z_n
        r_p, r_n = -(grad_p - lam), -(grad_n + lam)
        primal_extra = jnp.where(on, diagonal_x, 0.0)
        dual_extra = jnp.where(on, inv_p + inv_n, 0.0)
        rhs_extra = jnp.where(on, r_n * inv_n - r_p * inv_p, 0.0)
    else:
        grad_x, grad_s, inv_sigma_s = ev.gradient + barrier_x, barrier_s, s / state.v
        primal_extra = jnp.zeros_like(x)
        dual_extra = rhs_extra = r_p = r_n = inv_p = inv_n = grad_p = grad_n = zero_m
    return Terms(
        grad_x=grad_x,
        grad_s=grad_s,
        r_x=-(grad_x + jacobian.T @ lam),
        r_s=-(grad_s + state.lam_i),
        inv_sigma_s=inv_sigma_s,
        primal_extra=primal_extra,
        dual_extra=dual_extra,
        rhs_extra=rhs_extra,
        r_p=r_p,
        r_n=r_n,
        inv_p=inv_p,
        inv_n=inv_n,
        grad_p=grad_p,
        grad_n=grad_n,
    )


def merit(state, bounds, restores, x, s, p, n, value, equality, inequality):
    """`(theta, phi, c)` of the state's phase at a point: the violation, the barrier objective, `c`."""
    mu = state.mu
    theta, constraint = violation(equality, inequality, s)
    if not restores:
        return theta, value + barrier_value(x, s, mu, bounds), constraint
    on = state.restoration.on
    constraint = constraint + jnp.where(on, n - p, 0.0)
    resto = _restoration.objective(state.restoration, x, s, p, n, mu)
    phi = jnp.where(on, resto, value) + barrier_value(x, s, mu, bounds)
    return jnp.sum(jnp.abs(constraint)), phi, constraint
