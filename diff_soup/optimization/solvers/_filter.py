"""The filter of Wächter and Biegler's line search: which trial points are acceptable.

A trial point is judged by its constraint violation `theta` and its barrier objective `phi`.
The filter holds `(theta, phi)` pairs a trial must improve on in one of the two. A step whose
direction decreases `phi` enough relative to `theta` (the switching condition) must give
Armijo decrease in `phi` and does not augment the filter; any other step must give a
sufficient decrease in `theta` or `phi` and augments it. The filter lives in a fixed number
of slots, so its shape is static under `jit`.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

# Ipopt's defaults.
GAMMA_THETA = 1e-5
GAMMA_PHI = 1e-8
ETA_PHI = 1e-8
DELTA = 1.0
S_THETA = 1.1
S_PHI = 2.3
GAMMA_ALPHA = 0.05
THETA_MAX_FACTOR = 1e4
THETA_MIN_FACTOR = 1e-4
CAPACITY = 64


class Filter(eqx.Module):
    """The filter entries, `+inf` in the free slots.

    Attributes:
        theta: The constraint violation of each entry, `(CAPACITY,)`.
        phi: The barrier objective of each entry, `(CAPACITY,)`.
    """

    theta: Any
    phi: Any


def empty_filter(dtype: Any) -> Filter:
    """A filter with every slot free."""
    free = jnp.full((CAPACITY,), jnp.inf, dtype=dtype)
    return Filter(free, free)


def blocks(filt: Filter, theta: Any, phi: Any) -> Any:
    """Whether an entry of the filter dominates the trial `(theta, phi)`."""
    return jnp.any((theta >= filt.theta) & (phi >= filt.phi))


def augment(filt: Filter, theta: Any, phi: Any) -> Filter:
    """The filter with `(theta, phi)` added and the entries it dominates removed.

    A full filter overwrites the entry of largest `theta`, the one that blocks the least.
    """
    dominated = (theta <= filt.theta) & (phi <= filt.phi)
    theta_kept = jnp.where(dominated, jnp.inf, filt.theta)
    phi_kept = jnp.where(dominated, jnp.inf, filt.phi)
    free = jnp.isinf(theta_kept)
    slot = jnp.where(jnp.any(free), jnp.argmax(free), jnp.argmax(theta_kept))
    return Filter(theta_kept.at[slot].set(theta), phi_kept.at[slot].set(phi))


def _less_equal(lhs: Any, rhs: Any, reference: Any) -> Any:
    """`lhs <= rhs` up to ten machine epsilons of `reference`, as Ipopt compares."""
    eps = jnp.finfo(jnp.result_type(lhs)).eps
    return lhs - rhs <= 10.0 * eps * jnp.abs(reference)


def switching(alpha: Any, theta: Any, slope: Any) -> Any:
    """Whether the step decreases `phi` enough relative to `theta` to be held to Armijo."""
    descent = jnp.maximum(-slope, 0.0)
    return (slope < 0.0) & (alpha * descent**S_PHI > DELTA * theta**S_THETA)


def acceptable(
    filt: Filter,
    theta_trial: Any,
    phi_trial: Any,
    alpha: Any,
    theta: Any,
    phi: Any,
    slope: Any,
    theta_min: Any,
    theta_max: Any,
) -> tuple[Any, Any]:
    """Whether a trial is accepted, and whether it is an `f`-type step that leaves the filter alone.

    Args:
        filt: The current filter.
        theta_trial: The constraint violation at the trial point.
        phi_trial: The barrier objective at the trial point.
        alpha: The step length the switching condition and Armijo test use.
        theta: The constraint violation at the current point.
        phi: The barrier objective at the current point.
        slope: The directional derivative of `phi` along the step.
        theta_min: Below this violation the switching condition may apply.
        theta_max: The largest violation a trial may have.

    Returns:
        `(accepted, f_type)`.
    """
    f_type = (theta <= theta_min) & switching(alpha, theta, slope)
    armijo = _less_equal(phi_trial, phi + ETA_PHI * alpha * slope, phi)
    decrease = _less_equal(theta_trial, (1.0 - GAMMA_THETA) * theta, theta) | _less_equal(
        phi_trial, phi - GAMMA_PHI * theta, phi
    )
    finite = jnp.isfinite(theta_trial) & jnp.isfinite(phi_trial)
    accepted = (
        finite
        & ~blocks(filt, theta_trial, phi_trial)
        & (theta_trial <= theta_max)
        & jnp.where(f_type, armijo, decrease)
    )
    return accepted, f_type


def minimum_step(theta: Any, slope: Any, theta_min: Any) -> Any:
    """The step length below which the line search gives up (Wächter and Biegler, eq. 23)."""
    descent = jnp.maximum(-slope, jnp.finfo(jnp.result_type(slope)).tiny)
    by_phi = GAMMA_PHI * theta / descent
    by_switch = DELTA * theta**S_THETA / descent**S_PHI
    down = slope < 0.0
    bound = jnp.where(
        down & (theta <= theta_min),
        jnp.minimum(GAMMA_THETA, jnp.minimum(by_phi, by_switch)),
        jnp.where(down, jnp.minimum(GAMMA_THETA, by_phi), GAMMA_THETA),
    )
    return GAMMA_ALPHA * bound
