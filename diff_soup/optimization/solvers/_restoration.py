"""Ipopt's feasibility restoration phase, as a phase of the interior point's one loop.

When the line search fails, the interior point minimizes the constraint violation near the
current point `(x_R, s_R)`:

    min  rho * sum(p + n) + zeta / 2 * (|D_x (x - x_R)|^2 + |D_s (s - s_R)|^2)
    s.t. c(x, s) - p + n = 0,   p, n >= 0,   s >= 0,   the bounds of x,

where `c(x, s) = [c_E(x); c_I(x) + s]`, `zeta = sqrt(mu)` and `D = min(1, 1 / |x_R|)`
(Wächter and Biegler 2006, section 3.3). Its Newton step eliminates `p` and `n` as the
regular step eliminates the slacks, so it is a KKT system of the original problem's shape:
the primal block is `W_c + zeta D_x^2 + Sigma_x`, with `W_c` the curvature of the constraint
terms alone, and each row's dual diagonal gains `p / z_p + n / z_n`. The regular phase's
factorization, line search and evaluation serve it, and the problem's closures are traced
once. The phase ends as soon as the original filter accepts an iterate whose violation fell
by `KAPPA_RESTO`; the original problem is infeasible when the restoration problem converges
first.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from diff_soup.optimization.solvers import _filter, _hessian
from diff_soup.optimization.solvers._barrier import (
    KAPPA_D,
    Bounds,
    barrier_value,
    bound_multiplier_steps,
    distances,
    dual_fraction_to_boundary,
    safeguard_multipliers,
)
from diff_soup.optimization.solvers._closures import Closures, Evaluation
from diff_soup.optimization.solvers._core import lagrangian_secant
from diff_soup.optimization.solvers._optimality import scaled_error

#: Ipopt's `resto_penalty_parameter`, the weight `rho` of the violation.
RHO = 1000.0
#: Ipopt's `required_infeasibility_reduction`: the violation an accepted iterate must reach.
KAPPA_RESTO = 0.9
#: Ours: the steps of one restoration phase.
MAX_STEPS = 200
#: Ipopt's `bound_mult_reset_threshold`: after restoration, larger bound multipliers reset to 1.
BOUND_MULTIPLIER_RESET = 1e3


class Phase(eqx.Module):
    """The restoration phase's own variables and its reference point.

    Outside the phase `p`, `n` and their multipliers hold 1, which no formula reads.

    Attributes:
        on: Whether the restoration phase runs.
        p: The positive part of the violation of each row.
        n: The negative part of the violation of each row.
        z_p: The multipliers of `p >= 0`.
        z_n: The multipliers of `n >= 0`.
        x_ref: `x_R`.
        s_ref: `s_R`.
        zeta: The weight of the proximity term, `sqrt(mu)` at entry.
        weight_x: `D_x`.
        weight_s: `D_s`.
        theta: The original violation at entry.
        steps: The steps of this phase.
    """

    on: Bool[Array, ""]
    p: Float[Array, "m"]
    n: Float[Array, "m"]
    z_p: Float[Array, "m"]
    z_n: Float[Array, "m"]
    x_ref: Float[Array, "dim"]
    s_ref: Float[Array, "n_ineq"]
    zeta: Float[Array, ""]
    weight_x: Float[Array, "dim"]
    weight_s: Float[Array, "n_ineq"]
    theta: Float[Array, ""]
    steps: Int[Array, ""]


class Saved(eqx.Module):
    """The regular phase's fields, held while the restoration phase runs.

    Attributes:
        z_lower: The lower-bound multipliers at `x_R`.
        z_upper: The upper-bound multipliers at `x_R`.
        v: The slack multipliers at `x_R`.
        mu: The barrier parameter.
        tau: The fraction to the boundary.
        filt: The filter, augmented with `x_R`.
        theta_min: The switching threshold.
        theta_max: The largest violation a trial may have.
        delta_w_last: The last nonzero primal regularization.
        quasi_newton: The BFGS matrix.
        quasi_newton_updates: How many updates it has had.
        evaluation: The evaluation at `x_R`.
    """

    z_lower: Float[Array, "dim"]
    z_upper: Float[Array, "dim"]
    v: Float[Array, "n_ineq"]
    mu: Float[Array, ""]
    tau: Float[Array, ""]
    filt: _filter.Filter
    theta_min: Float[Array, ""]
    theta_max: Float[Array, ""]
    delta_w_last: Float[Array, ""]
    quasi_newton: Float[Array, "k k"]
    quasi_newton_updates: Int[Array, ""]
    evaluation: Evaluation


def idle(x: Float[Array, "dim"], s: Float[Array, "n_ineq"], m: int) -> Phase:
    """The phase outside restoration."""
    dtype = x.dtype
    one = jnp.ones((m,), dtype)
    return Phase(
        on=jnp.asarray(False),
        p=one,
        n=one,
        z_p=one,
        z_n=one,
        x_ref=x,
        s_ref=s,
        zeta=jnp.zeros((), dtype),
        weight_x=jnp.ones_like(x),
        weight_s=jnp.ones_like(s),
        theta=jnp.zeros((), dtype),
        steps=jnp.asarray(0, jnp.int32),
    )


def proximity(phase: Phase, x: Float[Array, "dim"], s: Float[Array, "n_ineq"]):
    """The gradient of the proximity term in `x` and in `s`, and its diagonals."""
    diagonal_x = phase.zeta * phase.weight_x**2
    diagonal_s = phase.zeta * phase.weight_s**2
    return diagonal_x * (x - phase.x_ref), diagonal_s * (s - phase.s_ref), diagonal_x, diagonal_s


def objective(
    phase: Phase,
    x: Float[Array, "dim"],
    s: Float[Array, "n_ineq"],
    p: Float[Array, "m"],
    n: Float[Array, "m"],
    mu: Float[Array, ""],
) -> Float[Array, ""]:
    """The restoration objective and the barrier on `p` and `n`, without the barrier on `x` and `s`."""
    offset_x, offset_s = x - phase.x_ref, s - phase.s_ref
    proximity_value = (
        0.5
        * phase.zeta
        * (jnp.sum(phase.weight_x**2 * offset_x**2) + jnp.sum(phase.weight_s**2 * offset_s**2))
    )
    barrier = -mu * (jnp.sum(jnp.log(p)) + jnp.sum(jnp.log(n))) + KAPPA_D * mu * jnp.sum(p + n)
    return RHO * jnp.sum(p + n) + proximity_value + barrier


def gradients(phase: Phase, mu: Float[Array, ""]):
    """The gradient of `objective` in `p` and in `n`."""
    return RHO - mu / phase.p + KAPPA_D * mu, RHO - mu / phase.n + KAPPA_D * mu


def error(
    phase: Phase,
    evaluation: Evaluation,
    x: Float[Array, "dim"],
    s: Float[Array, "n_ineq"],
    lam: Float[Array, "m"],
    z_lower: Float[Array, "dim"],
    z_upper: Float[Array, "dim"],
    v: Float[Array, "n_ineq"],
    mu: Float[Array, ""],
    bounds: Bounds,
) -> Float[Array, ""]:
    """The scaled optimality error `E_mu` of the restoration problem."""
    n_eq = evaluation.equality.shape[0]
    prox_x, prox_s, _, _ = proximity(phase, x, s)
    jacobian = jnp.concatenate([evaluation.jacobian_equality, evaluation.jacobian_inequality])
    dual = (
        prox_x + jacobian.T @ lam - z_lower + z_upper,
        prox_s + lam[n_eq:] - v,
        RHO - lam - phase.z_p,
        RHO + lam - phase.z_n,
    )
    primal = jnp.concatenate([evaluation.equality, evaluation.inequality + s]) - phase.p + phase.n
    below, above = distances(x, bounds)
    complementarity = (
        jnp.where(bounds.has_lower, below * z_lower - mu, 0.0),
        jnp.where(bounds.has_upper, above * z_upper - mu, 0.0),
        s * v - mu,
        phase.p * phase.z_p - mu,
        phase.n * phase.z_n - mu,
    )
    m = lam.shape[0]
    n_z = jnp.sum(bounds.has_lower) + jnp.sum(bounds.has_upper) + s.shape[0] + 2 * m
    z_norm = (
        jnp.sum(jnp.abs(z_lower))
        + jnp.sum(jnp.abs(z_upper))
        + jnp.sum(jnp.abs(v))
        + jnp.sum(jnp.abs(phase.z_p))
        + jnp.sum(jnp.abs(phase.z_n))
    )
    return scaled_error(dual, (primal,), complementarity, jnp.sum(jnp.abs(lam)), m, z_norm, n_z)


def start(
    x: Float[Array, "dim"],
    s: Float[Array, "n_ineq"],
    constraint: Float[Array, "m"],
    mu: Float[Array, ""],
    theta: Float[Array, ""],
) -> tuple[Phase, Float[Array, ""]]:
    """The phase at entry from `(x_R, s_R)` and its violation, and the phase's first `mu`.

    `p` and `n` take Ipopt's closed form, the minimizer of the restoration barrier problem in
    them at fixed `x` and `s`, and their multipliers `mu_R / p` and `mu_R / n`.
    """
    mu_r = jnp.maximum(mu, jnp.max(jnp.abs(constraint), initial=0.0))
    half = (mu_r - RHO * constraint) / (2.0 * RHO)
    n = half + jnp.sqrt(half**2 + mu_r * constraint / (2.0 * RHO))
    p = constraint + n
    phase = Phase(
        on=jnp.asarray(True),
        p=p,
        n=n,
        z_p=mu_r / p,
        z_n=mu_r / n,
        x_ref=x,
        s_ref=s,
        zeta=jnp.sqrt(mu),
        weight_x=jnp.minimum(1.0, 1.0 / jnp.abs(x)),
        weight_s=jnp.minimum(1.0, 1.0 / jnp.abs(s)),
        theta=theta,
        steps=jnp.asarray(0, jnp.int32),
    )
    return phase, mu_r


def accepts(
    saved: Saved,
    phase: Phase,
    evaluation: Evaluation,
    x: Float[Array, "dim"],
    s: Float[Array, "n_ineq"],
    bounds: Bounds,
) -> tuple[Bool[Array, ""], Float[Array, ""]]:
    """Whether the original filter accepts the iterate with its violation reduced, and the violation."""
    constraint = jnp.concatenate([evaluation.equality, evaluation.inequality + s])
    theta = jnp.sum(jnp.abs(constraint))
    phi = evaluation.value + barrier_value(x, s, saved.mu, bounds)
    ok = (
        jnp.isfinite(phi)
        & ~_filter.blocks(saved.filt, theta, phi)
        & (theta <= KAPPA_RESTO * phase.theta)
    )
    return ok, theta


def bound_multipliers_after(
    saved: Saved,
    phase: Phase,
    x: Float[Array, "dim"],
    s: Float[Array, "n_ineq"],
    bounds: Bounds,
):
    """The regular phase's bound multipliers after restoration.

    They take the complementarity Newton step for the primal change of the whole phase, cut
    at the boundary fraction, and are all reset to 1 when one passes
    `BOUND_MULTIPLIER_RESET`, as Ipopt does.
    """
    x_ref, s_ref = phase.x_ref, phase.s_ref
    dz_lower, dz_upper, dv = bound_multiplier_steps(
        x_ref, x - x_ref, s_ref, s - s_ref, saved.z_lower, saved.z_upper, saved.v, saved.mu, bounds
    )
    alpha = dual_fraction_to_boundary(
        saved.z_lower, dz_lower, saved.z_upper, dz_upper, saved.v, dv, saved.tau, bounds
    )
    z_lower = saved.z_lower + alpha * dz_lower
    z_upper = saved.z_upper + alpha * dz_upper
    v = saved.v + alpha * dv
    largest = jnp.maximum(
        jnp.max(z_lower, initial=0.0),
        jnp.maximum(jnp.max(z_upper, initial=0.0), jnp.max(v, initial=0.0)),
    )
    reset = largest > BOUND_MULTIPLIER_RESET
    z_lower = jnp.where(reset, jnp.where(bounds.has_lower, 1.0, 0.0), z_lower)
    z_upper = jnp.where(reset, jnp.where(bounds.has_upper, 1.0, 0.0), z_upper)
    v = jnp.where(reset, 1.0, v)
    return z_lower, z_upper, v


def enter(state, theta, constraint, filt, tau_min: float):
    """The interior point's state as the restoration phase starts at its point.

    The regular phase's fields go to `saved`, with `filt` (the filter augmented with the
    point); the restoration phase starts with zero constraint multipliers, the bound and
    slack multipliers capped at `RHO`, its own `mu`, an empty filter and a quasi-Newton matrix
    of zeros, as Ipopt's restoration phase does.
    """
    phase, mu = start(state.x, state.slacks, constraint, state.mu, theta)
    saved = Saved(
        z_lower=state.z_lower,
        z_upper=state.z_upper,
        v=state.v,
        mu=state.mu,
        tau=state.tau,
        filt=filt,
        theta_min=state.theta_min,
        theta_max=state.theta_max,
        delta_w_last=state.delta_w_last,
        quasi_newton=state.quasi_newton,
        quasi_newton_updates=state.quasi_newton_updates,
        evaluation=state.evaluation,
    )
    scale = jnp.maximum(1.0, jnp.sum(jnp.abs(constraint - phase.p + phase.n)))
    return dataclasses.replace(
        state,
        lam_e=jnp.zeros_like(state.lam_e),
        lam_i=jnp.zeros_like(state.lam_i),
        z_lower=jnp.minimum(RHO, state.z_lower),
        z_upper=jnp.minimum(RHO, state.z_upper),
        v=jnp.minimum(RHO, state.v),
        mu=mu,
        tau=jnp.maximum(tau_min, 1.0 - mu),
        filt=_filter.empty_filter(mu.dtype),
        theta_min=_filter.THETA_MIN_FACTOR * scale,
        theta_max=_filter.THETA_MAX_FACTOR * scale,
        delta_w_last=jnp.zeros_like(state.delta_w_last),
        quasi_newton=jnp.zeros_like(state.quasi_newton),
        quasi_newton_updates=jnp.zeros_like(state.quasi_newton_updates),
        restoration=phase,
        saved=saved,
    )


def leave(state, bounds: Bounds, hessian: str, closures: Closures):
    """The interior point's state as the regular phase resumes at the restoration iterate.

    The constraint multipliers are 0, the bound multipliers take [bound_multipliers_after][],
    and the regular quasi-Newton matrix learns from the gradient change over the whole phase.
    """
    saved, phase = state.saved, state.restoration
    x, s, evaluation = state.x, state.slacks, state.evaluation
    z_lower, z_upper, v = bound_multipliers_after(saved, phase, x, s, bounds)
    z_lower, z_upper, v = safeguard_multipliers(x, s, z_lower, z_upper, v, saved.mu, bounds)
    lam_e, lam_i = jnp.zeros_like(state.lam_e), jnp.zeros_like(state.lam_i)
    quasi_newton, updates = _hessian.update(
        hessian,
        closures,
        saved.quasi_newton,
        saved.quasi_newton_updates,
        x - phase.x_ref,
        lagrangian_secant(evaluation, saved.evaluation, lam_e, lam_i),
        evaluation,
    )
    return dataclasses.replace(
        state,
        lam_e=lam_e,
        lam_i=lam_i,
        z_lower=z_lower,
        z_upper=z_upper,
        v=v,
        mu=saved.mu,
        tau=saved.tau,
        filt=saved.filt,
        theta_min=saved.theta_min,
        theta_max=saved.theta_max,
        delta_w_last=saved.delta_w_last,
        quasi_newton=quasi_newton,
        quasi_newton_updates=updates,
        restoration=idle(x, s, phase.p.shape[0]),
    )
