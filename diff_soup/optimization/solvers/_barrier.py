"""The barrier terms of the interior point: bounds, the log barrier, fraction to the boundary.

The decision vector `x` carries the barrier on its finite bounds directly; the inequality
slacks `s` carry it on `s >= 0`. A variable with one finite bound gets Ipopt's linear damping
term `kappa_d * mu * distance`, which keeps the barrier problem bounded.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

#: Ipopt's `kappa_d`: the linear damping of a variable with one finite bound.
KAPPA_D = 1e-5
#: Ipopt's `kappa_sigma`: how far a bound multiplier may stray from `mu / distance`.
KAPPA_SIGMA = 1e10


class Bounds(NamedTuple):
    """Finite bounds and their masks, with zeros where a bound is absent.

    Attributes:
        lower: The lower bounds, 0 where absent.
        upper: The upper bounds, 0 where absent.
        has_lower: Where a lower bound is finite.
        has_upper: Where an upper bound is finite.
    """

    lower: Float[Array, "n"]
    upper: Float[Array, "n"]
    has_lower: Bool[Array, "n"]
    has_upper: Bool[Array, "n"]


def bounds_from(lower: Any, upper: Any) -> Bounds:
    """The `Bounds` of `(lower, upper)` arrays, infinite where absent."""
    has_lower = jnp.isfinite(lower)
    has_upper = jnp.isfinite(upper)
    return Bounds(
        jnp.where(has_lower, lower, 0.0), jnp.where(has_upper, upper, 0.0), has_lower, has_upper
    )


def distances(x: Any, bounds: Bounds) -> tuple[Any, Any]:
    """`(x - lower, upper - x)`, 1 where the bound is absent."""
    below = jnp.where(bounds.has_lower, x - bounds.lower, 1.0)
    above = jnp.where(bounds.has_upper, bounds.upper - x, 1.0)
    return below, above


def push_into_bounds(x0: Any, bounds: Bounds, push: float, frac: float) -> Any:
    """Ipopt's start point: `x0` moved strictly inside its bounds by `bound_push` and `bound_frac`."""
    width = bounds.upper - bounds.lower
    both = bounds.has_lower & bounds.has_upper
    push_lower = push * jnp.maximum(1.0, jnp.abs(bounds.lower))
    push_upper = push * jnp.maximum(1.0, jnp.abs(bounds.upper))
    push_lower = jnp.where(both, jnp.minimum(push_lower, frac * width), push_lower)
    push_upper = jnp.where(both, jnp.minimum(push_upper, frac * width), push_upper)
    x = jnp.where(bounds.has_lower, jnp.maximum(x0, bounds.lower + push_lower), x0)
    return jnp.where(bounds.has_upper, jnp.minimum(x, bounds.upper - push_upper), x)


def barrier_value(x: Any, s: Any, mu: Any, bounds: Bounds) -> Any:
    """The barrier and damping terms of `phi_mu`, without the objective."""
    below, above = distances(x, bounds)
    only_lower = bounds.has_lower & ~bounds.has_upper
    only_upper = bounds.has_upper & ~bounds.has_lower
    logs = (
        jnp.sum(jnp.where(bounds.has_lower, jnp.log(below), 0.0))
        + jnp.sum(jnp.where(bounds.has_upper, jnp.log(above), 0.0))
        + jnp.sum(jnp.log(s))
    )
    damping = (
        jnp.sum(jnp.where(only_lower, below, 0.0))
        + jnp.sum(jnp.where(only_upper, above, 0.0))
        + jnp.sum(s)
    )
    return -mu * logs + KAPPA_D * mu * damping


def barrier_gradient(x: Any, s: Any, mu: Any, bounds: Bounds) -> tuple[Any, Any]:
    """The gradient of `barrier_value` in `x` and in `s`."""
    below, above = distances(x, bounds)
    only_lower = bounds.has_lower & ~bounds.has_upper
    only_upper = bounds.has_upper & ~bounds.has_lower
    grad_x = (
        jnp.where(bounds.has_upper, mu / above, 0.0)
        - jnp.where(bounds.has_lower, mu / below, 0.0)
        + KAPPA_D * mu * (only_lower.astype(x.dtype) - only_upper.astype(x.dtype))
    )
    grad_s = -mu / s + KAPPA_D * mu
    return grad_x, grad_s


def primal_sigma(x: Any, z_lower: Any, z_upper: Any, bounds: Bounds) -> Any:
    """The primal-dual barrier diagonal `Z_L / (x - lower) + Z_U / (upper - x)`."""
    below, above = distances(x, bounds)
    return jnp.where(bounds.has_lower, z_lower / below, 0.0) + jnp.where(
        bounds.has_upper, z_upper / above, 0.0
    )


def bound_multiplier_steps(
    x: Any, dx: Any, s: Any, ds: Any, z_lower: Any, z_upper: Any, v: Any, mu: Any, bounds: Bounds
) -> tuple[Any, Any, Any]:
    """The steps of the bound multipliers from the primal step: `dz = mu / d - z - Sigma dd`."""
    below, above = distances(x, bounds)
    dz_lower = jnp.where(bounds.has_lower, (mu - z_lower * below - z_lower * dx) / below, 0.0)
    dz_upper = jnp.where(bounds.has_upper, (mu - z_upper * above + z_upper * dx) / above, 0.0)
    dv = (mu - v * s - v * ds) / s
    return dz_lower, dz_upper, dv


def _largest_step(value: Any, step: Any, tau: Any, mask: Any) -> Any:
    """The largest `alpha` in (0, 1] keeping `value + alpha * step >= (1 - tau) * value` where `mask`."""
    shrinking = mask & (step < 0.0)
    safe = jnp.where(shrinking, step, -1.0)
    return jnp.min(jnp.where(shrinking, -tau * value / safe, 1.0), initial=1.0)


def fraction_to_boundary(x: Any, dx: Any, s: Any, ds: Any, tau: Any, bounds: Bounds) -> Any:
    """The largest primal step length keeping `x` and `s` a fraction `tau` inside their bounds."""
    below, above = distances(x, bounds)
    return jnp.minimum(
        jnp.minimum(
            _largest_step(below, dx, tau, bounds.has_lower),
            _largest_step(above, -dx, tau, bounds.has_upper),
        ),
        _largest_step(s, ds, tau, jnp.ones_like(s, dtype=bool)),
    )


def dual_fraction_to_boundary(
    z_lower: Any,
    dz_lower: Any,
    z_upper: Any,
    dz_upper: Any,
    v: Any,
    dv: Any,
    tau: Any,
    bounds: Bounds,
) -> Any:
    """The largest dual step length keeping the bound multipliers a fraction `tau` above 0."""
    return jnp.minimum(
        jnp.minimum(
            _largest_step(z_lower, dz_lower, tau, bounds.has_lower),
            _largest_step(z_upper, dz_upper, tau, bounds.has_upper),
        ),
        _largest_step(v, dv, tau, jnp.ones_like(v, dtype=bool)),
    )


def safeguard_multipliers(
    x: Any, s: Any, z_lower: Any, z_upper: Any, v: Any, mu: Any, bounds: Bounds
) -> tuple[Any, Any, Any]:
    """Ipopt's `kappa_sigma` safeguard: each bound multiplier within a factor of `mu / distance`."""
    below, above = distances(x, bounds)

    def clip(z, distance, mask):
        target = mu / distance
        clipped = jnp.clip(z, target / KAPPA_SIGMA, KAPPA_SIGMA * target)
        return jnp.where(mask, clipped, 0.0)

    return (
        clip(z_lower, below, bounds.has_lower),
        clip(z_upper, above, bounds.has_upper),
        clip(v, s, jnp.ones_like(s, dtype=bool)),
    )
