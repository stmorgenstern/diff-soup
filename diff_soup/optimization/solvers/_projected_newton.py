"""A compiled projected Newton trust region: minimize a smooth function over a box.

Each step splits the coordinates as Bertsekas (1982) does. A coordinate within `eps` of a
bound whose gradient pushes it out is binding: it moves onto its bound, by at most the
radius. The others take the trust-region step of the quadratic model restricted to them,
solved on the boundary by an eigendecomposition and Newton's method on the secular equation
(Moré and Sorensen 1983), the hard case included. The trial is projected into the box, and
the ratio of the actual to the predicted decrease accepts it and sets the next radius. `eps`
is the smaller of `EPSILON` and the projected gradient, so near a solution with strict
complementarity the binding set is the active set and the step is Newton's on the free
coordinates.

The loop is one `lax.while_loop` with masked lanes. A failed evaluation (NaN, a refused
point) rejects the trial and shrinks the radius; `max_failed` of them in a row end the loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from diff_soup.optimization.solvers._barrier import Bounds, distances

EPSILON = 1e-3
ETA = 1e-4
SHRINK = 0.25
GROW = 2.0
RADIUS_MAX = 1e10
RADIUS_MIN = 1e-14
_SECULAR_STEPS = 60
_SECULAR_TOL = 1e-12


class InnerState(eqx.Module):
    """The carry of the projected Newton loop.

    Attributes:
        x: The current point, inside the box.
        value: The function at `x`.
        gradient: Its gradient at `x`.
        hessian: The model Hessian at `x`.
        aux: Whatever the caller's derivative closure carries from point to point.
        radius: The trust-region radius.
        steps: The steps taken.
        done: Whether the loop has ended for this lane.
        converged: Whether it ended with the projected gradient below the tolerance.
        consecutive_failed: Failed evaluations in a row.
        failed_evaluations: Failed evaluations in all.
    """

    x: Any
    value: Any
    gradient: Any
    hessian: Any
    aux: Any
    radius: Any
    steps: Any
    done: Any
    converged: Any
    consecutive_failed: Any
    failed_evaluations: Any


def _limits(bounds: Bounds) -> tuple[Any, Any]:
    """The bounds with `-inf` and `inf` where absent."""
    return (
        jnp.where(bounds.has_lower, bounds.lower, -jnp.inf),
        jnp.where(bounds.has_upper, bounds.upper, jnp.inf),
    )


def project(x: Any, bounds: Bounds) -> Any:
    """`x` clipped into the box."""
    lower, upper = _limits(bounds)
    return jnp.clip(x, lower, upper)


def projected_gradient(x: Any, gradient: Any, bounds: Bounds) -> Any:
    """`x - P(x - g)`, zero exactly at a stationary point of the box-constrained problem."""
    return x - project(x - gradient, bounds)


def bound_multipliers(x: Any, gradient: Any, bounds: Bounds) -> tuple[Any, Any]:
    """The `(z_lower, z_upper)` that best account for the gradient at `x`.

    A lower bound at distance `d` takes `max(g, 0) / (1 + d)`, which balances the residual
    `g - z` against the complementarity `d z`; on the bound it is the whole positive
    gradient, far from it nearly zero. The upper bound takes the negative gradient alike.
    """
    below, above = distances(x, bounds)
    z_lower = jnp.where(bounds.has_lower, jnp.maximum(gradient, 0.0) / (1.0 + below), 0.0)
    z_upper = jnp.where(bounds.has_upper, jnp.maximum(-gradient, 0.0) / (1.0 + above), 0.0)
    return z_lower, z_upper


def trust_region_step(hessian: Any, gradient: Any, radius: Any) -> Any:
    """The minimizer of `g d + d H d / 2` over `|d| <= radius` (Moré and Sorensen 1983).

    With `H = Q diag(lam) Q^T`: the Newton step when `H` is positive definite and the step
    fits; otherwise the step `d(sigma) = -(H + sigma I)^-1 g` with `sigma > max(0, -lam_min)`
    and `|d(sigma)| = radius`, by Newton's method on `1 / |d(sigma)| = 1 / radius` from the
    left, where it rises monotonically. In the hard case (`g` orthogonal to the lowest
    eigenvectors and `d` short of the boundary at the least shift) the lowest eigenvector
    takes the step to the boundary.
    """
    lam, q = jnp.linalg.eigh(hessian)
    a = q.T @ gradient
    dtype = gradient.dtype
    n = gradient.shape[0]
    width = 10.0 * n * jnp.finfo(dtype).eps * jnp.maximum(jnp.max(jnp.abs(lam)), 1e-300)
    lowest = lam[0]

    definite = lowest > width
    newton = q @ jnp.where(definite, -a / jnp.where(definite, lam, 1.0), 0.0)
    interior = definite & (jnp.linalg.norm(newton) <= radius)

    start = jnp.maximum(0.0, -lowest) + 2.0 * width

    def coefficients(sigma):
        return -a / (lam + sigma)

    def norm_at(sigma):
        return jnp.linalg.norm(coefficients(sigma))

    hard = norm_at(start) <= radius

    def cond(carry):
        sigma, count = carry
        norm = norm_at(sigma)
        searching = ~(interior | hard) & (jnp.abs(norm / radius - 1.0) > _SECULAR_TOL)
        return (count < _SECULAR_STEPS) & searching

    def body(carry):
        sigma, count = carry
        shifted = lam + sigma
        norm = norm_at(sigma)
        slope = jnp.sum(a**2 / shifted**3)
        step = (norm / radius - 1.0) * norm**2 / jnp.where(slope > 0.0, slope, 1.0)
        candidate = sigma + jnp.where(slope > 0.0, step, 0.0)
        candidate = jnp.where(jnp.isfinite(candidate), candidate, sigma)
        return jnp.maximum(candidate, start), count + 1

    sigma, _ = lax.while_loop(cond, body, (start, jnp.asarray(0, jnp.int32)))
    boundary = q @ coefficients(sigma)

    short = q @ coefficients(start)
    along = q[:, 0]
    overlap = along @ short
    tau = -overlap + jnp.sqrt(jnp.maximum(overlap**2 + radius**2 - short @ short, 0.0))
    escape = short + tau * along

    return jnp.where(interior, newton, jnp.where(hard, escape, boundary)).astype(dtype)


def _binding(x: Any, gradient: Any, bounds: Bounds) -> tuple[Any, Any]:
    """Bertsekas' binding sets: within `eps` of a bound with the gradient pushing outward."""
    below, above = distances(x, bounds)
    eps = jnp.minimum(
        EPSILON, jnp.max(jnp.abs(projected_gradient(x, gradient, bounds)), initial=0.0)
    )
    lower = bounds.has_lower & (below <= eps) & (gradient > 0.0)
    upper = bounds.has_upper & (above <= eps) & (gradient < 0.0)
    return lower, upper


def _proposal(state: InnerState, bounds: Bounds) -> Any:
    """The projected trial point of one step."""
    x, gradient, hessian, radius = state.x, state.gradient, state.hessian, state.radius
    lower, upper = _binding(x, gradient, bounds)
    keep = (~(lower | upper)).astype(x.dtype)
    big = 1.0 + jnp.sqrt(jnp.sum(hessian**2))
    reduced = keep[:, None] * hessian * keep[None, :] + jnp.diag((1.0 - keep) * big)
    free = keep * trust_region_step(reduced, keep * gradient, radius)
    below, above = distances(x, bounds)
    onto = jnp.where(lower, -jnp.minimum(below, radius), 0.0) + jnp.where(
        upper, jnp.minimum(above, radius), 0.0
    )
    return project(x + free + onto, bounds)


def start_state(
    x: Any, value: Any, gradient: Any, hessian: Any, aux: Any, radius: Any, done: Any
) -> InnerState:
    """The loop's carry at `x`, inside the box, with its value and derivatives."""
    zero = jnp.asarray(0, jnp.int32)
    return InnerState(
        x=x,
        value=value,
        gradient=gradient,
        hessian=hessian,
        aux=aux,
        radius=jnp.asarray(radius, x.dtype),
        steps=zero,
        done=jnp.asarray(done),
        converged=jnp.asarray(False),
        consecutive_failed=zero,
        failed_evaluations=zero,
    )


def minimize(
    trial: Callable,
    derivatives: Callable,
    state: InnerState,
    bounds: Bounds,
    *,
    tol: Any,
    max_steps: int,
    max_failed: int,
) -> InnerState:
    """Run the loop from `state` until the projected gradient falls below `tol`.

    Args:
        trial: `x -> (value, valid)`, the function at a trial point without derivatives.
        derivatives: `(x_new, x_old, aux) -> (value, gradient, hessian, aux, finite)` at an
            accepted point; `aux` carries what the caller needs from the old point (a
            quasi-Newton matrix, the last evaluation).
        state: The carry from [start_state][].
        bounds: The box.
        tol: The projected gradient, in the infinity norm, at which the loop ends.
        max_steps: The step budget.
        max_failed: Consecutive failed evaluations before the loop ends.

    Returns:
        The final carry; `converged` says whether the tolerance was met.
    """

    def stationary(current):
        return jnp.max(
            jnp.abs(projected_gradient(current.x, current.gradient, bounds)), initial=0.0
        )

    def finish(current):
        converged = stationary(current) <= tol
        tiny = current.radius <= RADIUS_MIN * (1.0 + jnp.max(jnp.abs(current.x), initial=0.0))
        ended = (
            converged
            | tiny
            | (current.steps >= max_steps)
            | (current.consecutive_failed >= max_failed)
        )
        return eqx.tree_at(
            lambda s: (s.done, s.converged),
            current,
            (current.done | ended, current.converged | converged),
        )

    def body(current):
        x = current.x
        candidate = _proposal(current, bounds)
        step = candidate - x
        predicted = -(current.gradient @ step + 0.5 * step @ (current.hessian @ step))
        value, valid = trial(candidate)
        actual = current.value - value
        noise = 10.0 * jnp.finfo(x.dtype).eps * jnp.maximum(1.0, jnp.abs(current.value))
        small = predicted <= noise
        ratio = jnp.where(small, 1.0, actual / jnp.where(predicted > 0.0, predicted, 1.0))
        accepted = valid & jnp.where(small, actual >= -noise, ratio >= ETA)
        new_value, gradient, hessian, aux, finite = derivatives(candidate, x, current.aux)
        moved = accepted & finite
        failed = ~valid | (accepted & ~finite)
        length = jnp.linalg.norm(step)
        radius = jnp.where(
            moved,
            jnp.where(
                ratio < 0.25,
                SHRINK * length,
                jnp.where(ratio > 0.75, jnp.maximum(current.radius, GROW * length), current.radius),
            ),
            SHRINK * jnp.minimum(current.radius, length),
        )
        radius = jnp.minimum(radius, RADIUS_MAX)
        moved_state = eqx.tree_at(
            lambda s: (s.x, s.value, s.gradient, s.hessian, s.aux),
            current,
            (candidate, new_value, gradient, hessian, aux),
        )
        new = jax.tree.map(lambda a, b: jnp.where(moved, a, b), moved_state, current)
        new = eqx.tree_at(
            lambda s: (s.radius, s.steps, s.consecutive_failed, s.failed_evaluations),
            new,
            (
                radius,
                current.steps + 1,
                jnp.where(failed, current.consecutive_failed + 1, 0),
                current.failed_evaluations + failed,
            ),
        )
        new = finish(new)
        return jax.tree.map(lambda a, b: jnp.where(current.done, a, b), current, new)

    return lax.while_loop(lambda current: ~current.done, body, finish(state))
