"""The trust-region subproblem: More-Sorensen on a factored Hessian, Steihaug-CG, and the SR1 update."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import local_value
from diff_soup.optimization.solvers.factorization import (
    cholesky_factor,
    qr_normal_factor,
    solve_factorized_half,
    solve_factorized_normal,
)
from diff_soup.optimization.solvers.tangent import (
    value_and_jacfwd_tiled,
)


def _quadratic_reduction(B: jax.Array, g: jax.Array, step: jax.Array) -> jax.Array:
    """Return the reduction predicted by ``g.T p + 0.5 p.T B p``."""
    return -(g @ step + 0.5 * step @ (B @ step))


def _boundary_intersection(point: jax.Array, direction: jax.Array, radius: float) -> jax.Array:
    """Return the nonnegative distance along ``direction`` to a sphere."""
    a = direction @ direction
    b = 2.0 * (point @ direction)
    c = point @ point - radius**2
    discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    return (-b + jnp.sqrt(discriminant)) / (2.0 * a)


def _more_sorensen(
    B: jax.Array,
    g: jax.Array,
    scale: jax.Array,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve a dense scaled trust-region subproblem by More--Sorensen.

    The eigensystem exposes negative curvature and the hard case directly.
    Away from the hard case, safeguarded bisection of the secular equation
    supplies the unique multiplier; this has the same safeguarded convergence
    target as Newton iteration without depending on a well-scaled derivative.

    Returns:
        ``(step, predicted_reduction, multiplier)``.
    """
    B = 0.5 * (jnp.asarray(B) + jnp.asarray(B).T)
    g = jnp.asarray(g)
    scale = jnp.asarray(scale)
    radius = float(radius)
    inv_scale = 1.0 / scale
    scaled_B = inv_scale[:, None] * B * inv_scale[None, :]
    scaled_g = inv_scale * g
    eigenvalues, eigenvectors = jnp.linalg.eigh(scaled_B)
    coefficients = eigenvectors.T @ scaled_g

    dtype = B.dtype
    eps = float(jnp.finfo(dtype).eps)
    spectral_scale = max(1.0, float(local_value(jnp.max(jnp.abs(eigenvalues)))))
    singular_tol = 64.0 * eps * spectral_scale
    minimum = float(local_value(eigenvalues[0]))
    singular = jnp.abs(eigenvalues - eigenvalues[0]) <= singular_tol

    # A positive-definite interior Newton step is the exact solution.
    if minimum > singular_tol:
        newton_coefficients = -coefficients / eigenvalues
        if float(local_value(jnp.linalg.norm(newton_coefficients))) <= radius:
            scaled_step = eigenvectors @ newton_coefficients
            step = inv_scale * scaled_step
            return step, _quadratic_reduction(B, g, step), jnp.asarray(0.0, dtype=dtype)

    lower = max(0.0, -minimum)
    denominators = eigenvalues + lower
    base_coefficients = jnp.where(singular, 0.0, -coefficients / denominators)
    base_norm = float(local_value(jnp.linalg.norm(base_coefficients)))
    singular_gradient = float(local_value(jnp.linalg.norm(jnp.where(singular, coefficients, 0.0))))
    gradient_scale = max(1.0, float(local_value(jnp.linalg.norm(coefficients))))

    # Positive-semidefinite singular models can have an interior pseudoinverse
    # solution. Negative-curvature hard cases add the missing minimum-eigenvector
    # component to reach the boundary.
    if minimum >= -singular_tol and singular_gradient <= singular_tol * gradient_scale:
        if base_norm <= radius:
            scaled_step = eigenvectors @ base_coefficients
            step = inv_scale * scaled_step
            return step, _quadratic_reduction(B, g, step), jnp.asarray(lower, dtype=dtype)
    if (
        minimum < -singular_tol
        and singular_gradient <= singular_tol * gradient_scale
        and base_norm <= radius
    ):
        hard_coefficients = base_coefficients.at[0].set(
            jnp.sqrt(jnp.maximum(radius**2 - base_norm**2, 0.0))
        )
        scaled_step = eigenvectors @ hard_coefficients
        step = inv_scale * scaled_step
        return step, _quadratic_reduction(B, g, step), jnp.asarray(lower, dtype=dtype)

    def secular_norm(multiplier):
        safe = jnp.maximum(eigenvalues + multiplier, jnp.asarray(singular_tol, dtype=dtype))
        return jnp.linalg.norm(coefficients / safe)

    # Find an upper bracket whose shifted solution is inside the region.
    lo = jnp.asarray(lower, dtype=dtype)
    hi = jnp.asarray(max(1.0, lower + 1.0), dtype=dtype)

    def bracket_cond(state):
        _lo, upper, count = state
        return (secular_norm(upper) > radius) & (count < 80)

    def bracket_body(state):
        lower_bound, upper, count = state
        return lower_bound, 2.0 * upper + 1.0, count + 1

    lo, hi, _ = jax.lax.while_loop(
        bracket_cond,
        bracket_body,
        (lo, hi, jnp.asarray(0, dtype=jnp.int32)),
    )

    def secular_value_and_derivative(multiplier):
        shifted = jnp.maximum(
            eigenvalues + multiplier,
            jnp.asarray(singular_tol, dtype=dtype),
        )
        norm = jnp.linalg.norm(coefficients / shifted)
        value = 1.0 / norm - 1.0 / radius
        derivative = jnp.sum(coefficients**2 / shifted**3) / norm**3
        return value, derivative

    def newton_body(_index, state):
        lower_bound, upper_bound, current = state
        value, derivative = secular_value_and_derivative(current)
        newton = current - value / derivative
        midpoint = 0.5 * (lower_bound + upper_bound)
        safeguarded = jnp.where(
            jnp.isfinite(newton) & (newton > lower_bound) & (newton < upper_bound),
            newton,
            midpoint,
        )
        outside = secular_norm(safeguarded) > radius
        next_lower = jnp.where(outside, safeguarded, lower_bound)
        next_upper = jnp.where(outside, upper_bound, safeguarded)
        return next_lower, next_upper, safeguarded

    lo, hi, multiplier = jax.lax.fori_loop(0, 50, newton_body, (lo, hi, hi))
    shifted = jnp.maximum(eigenvalues + multiplier, jnp.asarray(singular_tol, dtype=dtype))
    scaled_step = eigenvectors @ (-coefficients / shifted)
    step = inv_scale * scaled_step
    return step, _quadratic_reduction(B, g, step), multiplier


def _factorized_psd_trust_region(
    B: jax.Array,
    g: jax.Array,
    scale: jax.Array,
    radius: float,
    linear_solver: str,
    jacobian: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, bool, int]:
    """Solve a convex dense trust region with Cholesky or QR factors.

    This path is for Gauss--Newton matrices. Cholesky factors the shifted
    normal matrix; QR factors the augmented scaled Jacobian and uses its
    triangular factor in the secular iteration. The sparse state-space
    systems inside each PDE evaluation remain the responsibility of the
    configured time integrator.
    """
    B = 0.5 * (jnp.asarray(B) + jnp.asarray(B).T)
    g = jnp.asarray(g)
    scale = jnp.asarray(scale)
    radius = float(radius)
    inv_scale = 1.0 / scale
    scaled_B = inv_scale[:, None] * B * inv_scale[None, :]
    scaled_g = inv_scale * g
    identity = jnp.eye(B.shape[0], dtype=B.dtype)
    scaled_jacobian = None if jacobian is None else jnp.asarray(jacobian) * inv_scale[None, :]
    factorizations = 0

    def eigen_fallback():
        step, predicted, multiplier = _more_sorensen(B, g, scale, radius)
        return step, predicted, multiplier, True, factorizations

    def shifted_factor(multiplier):
        nonlocal factorizations
        factorizations += 1
        if linear_solver == "cholesky":
            return cholesky_factor(scaled_B + multiplier * identity)
        if scaled_jacobian is None:
            raise ValueError(
                "NewtonTrustRegion linear_solver='qr' requires the current "
                "Gauss-Newton residual Jacobian."
            )
        augmented = jnp.concatenate(
            (scaled_jacobian, jnp.sqrt(multiplier) * identity),
            axis=0,
        )
        return qr_normal_factor(augmented)

    def factorized_point(multiplier):
        factor = shifted_factor(multiplier)
        return factor, -solve_factorized_normal(factor, scaled_g)

    zero = jnp.asarray(0.0, dtype=B.dtype)
    _zero_factor, interior = factorized_point(zero)
    interior_finite = bool(local_value(jnp.all(jnp.isfinite(interior))))
    if interior_finite and float(local_value(jnp.linalg.norm(interior))) <= radius:
        step = inv_scale * interior
        return step, _quadratic_reduction(B, g, step), zero, False, factorizations

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        hi_factor, hi_point = factorized_point(jnp.asarray(hi, dtype=B.dtype))
        hi_norm = float(local_value(jnp.linalg.norm(hi_point)))
        if np.isfinite(local_value(hi_norm)) and hi_norm <= radius:
            break
        lo = hi
        hi = 2.0 * hi + 1.0
    else:
        return eigen_fallback()
    current = hi
    factor = hi_factor
    point = hi_point
    tolerance = 64.0 * np.finfo(np.dtype(B.dtype)).eps * max(1.0, radius)
    for iteration in range(30):
        norm = float(local_value(jnp.linalg.norm(point)))
        if not np.isfinite(local_value(norm)):
            lo = current
            current = 0.5 * (lo + hi)
            if iteration < 29:
                factor, point = factorized_point(jnp.asarray(current, dtype=B.dtype))
            continue
        elif norm <= radius:
            hi = current
            hi_factor, hi_point = factor, point
            if radius - norm <= tolerance:
                break
        else:
            lo = current
        q = solve_factorized_half(factor, point)
        denominator = float(local_value(q @ q))
        newton = current + (norm**2 / max(denominator, np.finfo(float).tiny)) * (
            (norm - radius) / radius
        )
        current = (
            newton if np.isfinite(local_value(newton)) and lo < newton < hi else 0.5 * (lo + hi)
        )
        if iteration < 29:
            factor, point = factorized_point(jnp.asarray(current, dtype=B.dtype))

    multiplier = jnp.asarray(hi, dtype=B.dtype)
    scaled_step = hi_point
    if not bool(local_value(jnp.all(jnp.isfinite(scaled_step)))):
        return eigen_fallback()
    step = inv_scale * scaled_step
    return step, _quadratic_reduction(B, g, step), multiplier, False, factorizations


def _steihaug_cg(
    hessian_vector_product: Callable[[jax.Array], jax.Array],
    g: jax.Array,
    scale: jax.Array,
    radius: float,
    max_iter: int,
) -> tuple[jax.Array, jax.Array, int, bool, bool]:
    """Solve a scaled trust-region subproblem by Steihaug conjugate gradients.

    Returns:
        ``(step, predicted_reduction, products, boundary, negative_curvature)``.
    """
    g = jnp.asarray(g)
    scale = jnp.asarray(scale)
    inv_scale = 1.0 / scale
    scaled_gradient = inv_scale * g
    point = jnp.zeros_like(g)
    scaled_B_point = jnp.zeros_like(g)
    residual = scaled_gradient
    direction = -residual
    residual_sq = residual @ residual
    scaled_gradient_norm = float(local_value(jnp.linalg.norm(scaled_gradient)))
    forcing = min(0.5, np.sqrt(scaled_gradient_norm)) * scaled_gradient_norm
    products = 0
    boundary = False
    negative_curvature = False

    for _ in range(int(local_value(max_iter))):
        unscaled_direction = inv_scale * direction
        B_direction = hessian_vector_product(unscaled_direction)
        products += 1
        scaled_B_direction = inv_scale * B_direction
        curvature = float(local_value(direction @ scaled_B_direction))
        if curvature <= 0.0:
            tau = _boundary_intersection(point, direction, radius)
            point = point + tau * direction
            scaled_B_point = scaled_B_point + tau * scaled_B_direction
            boundary = True
            negative_curvature = True
            break

        alpha = residual_sq / curvature
        candidate = point + alpha * direction
        if float(local_value(jnp.linalg.norm(candidate))) >= radius:
            tau = _boundary_intersection(point, direction, radius)
            point = point + tau * direction
            scaled_B_point = scaled_B_point + tau * scaled_B_direction
            boundary = True
            break

        point = candidate
        scaled_B_point = scaled_B_point + alpha * scaled_B_direction
        next_residual = residual + alpha * scaled_B_direction
        next_residual_sq = next_residual @ next_residual
        if float(local_value(jnp.sqrt(next_residual_sq))) <= forcing:
            residual = next_residual
            break
        beta = next_residual_sq / residual_sq
        direction = -next_residual + beta * direction
        residual = next_residual
        residual_sq = next_residual_sq

    step = inv_scale * point
    predicted = -(scaled_gradient @ point + 0.5 * point @ scaled_B_point)
    return step, predicted, products, boundary, negative_curvature


def _sr1_update(
    B: jax.Array,
    step: jax.Array,
    gradient_change: jax.Array,
) -> tuple[jax.Array, bool]:
    """Apply a safeguarded symmetric rank-one secant update."""
    B = jnp.asarray(B)
    step = jnp.asarray(step)
    correction = jnp.asarray(gradient_change) - B @ step
    denominator = float(local_value(correction @ step))
    threshold = (
        1e-8
        * float(local_value(jnp.linalg.norm(step)))
        * float(local_value(jnp.linalg.norm(correction)))
    )
    if abs(denominator) < threshold or denominator == 0.0:
        return B, False
    updated = B + jnp.outer(correction, correction) / denominator
    return 0.5 * (updated + updated.T), True


#: The tiles ``hessian_tile="auto"`` tries above ``dense_cutoff`` parameters, widest first.
_AUTO_HESSIAN_TILES = (32, 16, 8)


def _tiled_hessian(value_and_gradient: Callable, tile: int) -> Callable:
    """``x -> (loss, gradient, Hessian)``, the gradient pushed forward ``tile`` directions at a time."""

    def loss_and_gradient(point):
        loss_value, gradient_value = value_and_gradient(point)
        return jnp.concatenate((jnp.reshape(loss_value, (1,)), gradient_value))

    @jax.jit
    def evaluate(point):
        value, derivative = value_and_jacfwd_tiled(loss_and_gradient, point, tile=tile)
        return value[0], value[1:], derivative[1:, :]

    return evaluate
