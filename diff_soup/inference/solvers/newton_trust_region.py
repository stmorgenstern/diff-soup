"""Trust-region Newton optimization with exact or Gauss--Newton curvature."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import any_process, assert_decision_value, local_value
from diff_soup._errors import ConfigurationError
from diff_soup.inference._abc import OptSolver
from diff_soup.inference.fitting._forward_seam import _FAILED_SENTINEL
from diff_soup.inference.fitting.bounds import projected_gradient_norm
from diff_soup.inference.fitting.callbacks import StopFit
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import (
    EvalCounts,
    OptResult,
    StepInfo,
    TerminationStatus,
    TrustRegionReport,
)
from diff_soup.inference.fitting.scales import HessianScale, JacScale, XScale
from diff_soup.inference.solvers._loop import cost_threshold, trial_valid
from diff_soup.inference.solvers._termination import (
    DEFAULT_COST_ATOL,
    DEFAULT_COST_TOL,
    DEFAULT_STEP_TOL,
    absolute_tolerance,
    gradient_tolerance,
    model_held_the_step,
    relative_tolerance,
    stall_reason_of,
    step_budget,
)
from diff_soup.inference.solvers.factorization import (
    cholesky_factor,
    qr_normal_factor,
    solve_factorized_half,
    solve_factorized_normal,
)
from diff_soup.inference.solvers.tangent import (
    deferred_value_and_grad,
    jacfwd_from_pushforward,
    linearize_value,
    normalize_tangent_tile,
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


class NewtonTrustRegion(OptSolver):
    """The Newton trust-region method, steps on the exact Hessian or its Gauss-Newton approximation.

    It needs the gradient and the exact Hessian of the objective, or the Gauss-Newton matrix
    under `hessian="gauss-newton"`. Box bounds are kept. It is the default optimizer of
    `ds.inference.optsolve`.

    Args:
        max_steps: The step budget.
        hessian: `"true"` steps on the exact Hessian, `"gauss-newton"` on `J^T J` from the residual Jacobian.
        subproblem: `"exact"`, `"cg"`, or `"auto"`, which picks CG only for an exact Hessian above `dense_cutoff` parameters without `HessianScale`.
        linear_solver: `"eigh"`, `"cholesky"` or `"qr"` for the exact subproblem, or `"auto"` for `"eigh"` on the exact Hessian and Cholesky or QR otherwise.
        dense_cutoff: The parameter count above which `subproblem="auto"` may pick CG and `hessian_tile="auto"` tiles the Hessian.
        radius_init: The initial trust radius, in scaled parameter units.
        radius_max: The largest trust radius.
        eta: The ratio of achieved to predicted decrease a step must reach to be accepted.
        grad_tol: The gradient-norm tolerance, or `None` for the dtype's default.
        step_tol: The step-norm tolerance relative to the parameter norm, or `None` for the dtype's default.
        cost_tol: The tolerance on the change in the objective relative to the objective, or `None` for the dtype's default.
        cost_atol: An absolute floor under the cost-change test, zero when `None`.
        x_scale: The parameter scaling, `HessianScale()` by default, with `JacScale` refused under `hessian="true"` and `HessianScale` under CG.
        composition: `"forward-over-reverse"` or `"forward-over-forward"` for the exact Hessian, or `"auto"` to pick one the time-stepping adjoint supports.
        hessian_refresh: How often the exact Hessian is recomputed, with SR1 updates between: `"adaptive"`, `"every"`, or every `k` steps for an integer `k`.
        hessian_tile: How many exact-Hessian directions are formed at once, which sets the memory: 8 to 128, `"full"`, or `"auto"` (whole up to `dense_cutoff`, then 32).
        cg_max_iter: The conjugate-gradient iteration budget, the parameter count when `None`.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.
    """

    max_steps: int = eqx.field(static=True, default=50)
    hessian: str = eqx.field(static=True, default="true")
    subproblem: str = eqx.field(static=True, default="auto")
    linear_solver: str = eqx.field(static=True, default="auto")
    dense_cutoff: int = eqx.field(static=True, default=64)
    radius_init: float = eqx.field(static=True, default=1.0)
    radius_max: float = eqx.field(static=True, default=1e3)
    eta: float = eqx.field(static=True, default=1e-4)
    grad_tol: float | None = eqx.field(static=True, default=None)
    step_tol: float | None = eqx.field(static=True, default=None)
    cost_tol: float | None = eqx.field(static=True, default=None)
    cost_atol: float | None = eqx.field(static=True, default=None)
    x_scale: XScale = eqx.field(default_factory=HessianScale)
    composition: str = eqx.field(static=True, default="auto")
    hessian_refresh: Any = eqx.field(static=True, default="adaptive")
    hessian_tile: Any = eqx.field(static=True, default="auto")
    cg_max_iter: int | None = eqx.field(static=True, default=None)
    callbacks: tuple = eqx.field(static=True, default=())

    def _resolved_hessian_tile(self, n_params: int, lanes: bool) -> Any:
        """The tangent tile the solve starts with: an int, or ``"full"``."""
        if self.hessian_tile != "auto":
            return self.hessian_tile
        if lanes or n_params <= self.dense_cutoff:
            return "full"
        return _AUTO_HESSIAN_TILES[0]

    @property
    def requires(self) -> frozenset:
        """The structure the solve needs: the exact Hessian, or the Gauss-Newton statistics."""
        if self.hessian == "gauss-newton":
            return frozenset({"value_gradient", "gauss_newton"})
        return frozenset({"value_gradient", "hessian"})

    def _resolve_options(self, problem: Any) -> tuple[str, str | None, str]:
        """Return the concrete subproblem, composition, and linear solver."""
        if self.hessian not in ("true", "gauss-newton"):
            raise ConfigurationError(
                f"NewtonTrustRegion hessian must be 'true' or 'gauss-newton'; got {self.hessian!r}."
            )
        if self.subproblem not in ("auto", "exact", "cg"):
            raise ConfigurationError(
                "NewtonTrustRegion subproblem must be 'auto', 'exact', or 'cg'; "
                f"got {self.subproblem!r}."
            )
        if self.linear_solver not in ("auto", "eigh", "cholesky", "qr"):
            raise ConfigurationError(
                "NewtonTrustRegion linear_solver must be 'auto', 'eigh', "
                f"'cholesky', or 'qr'; got {self.linear_solver!r}."
            )
        if not isinstance(self.x_scale, XScale):
            raise ConfigurationError(
                "NewtonTrustRegion x_scale must be a ds.inference XScale object."
            )
        if self.radius_init <= 0.0 or self.radius_max < self.radius_init:
            raise ConfigurationError("NewtonTrustRegion requires 0 < radius_init <= radius_max.")
        if not 0.0 <= self.eta < 1.0:
            raise ConfigurationError("NewtonTrustRegion eta must satisfy 0 <= eta < 1.")
        if self.dense_cutoff < 0:
            raise ConfigurationError("NewtonTrustRegion dense_cutoff must be nonnegative.")
        if self.cg_max_iter is not None and self.cg_max_iter <= 0:
            raise ConfigurationError("NewtonTrustRegion cg_max_iter must be positive or None.")
        step_budget(self.max_steps, label="NewtonTrustRegion")

        if self.hessian == "gauss-newton":
            if self.composition != "auto":
                raise ConfigurationError(
                    "NewtonTrustRegion composition is not used with "
                    "hessian='gauss-newton'; leave composition='auto'."
                )
            if self.hessian_refresh != "adaptive":
                raise ConfigurationError(
                    "NewtonTrustRegion hessian_refresh is not used with "
                    "hessian='gauss-newton'; leave hessian_refresh='adaptive'."
                )
            composition = None
        else:
            composition = problem.hessian_composition

        n_params = int(local_value(np.asarray(problem.x0).size))
        resolved_subproblem = self.subproblem
        if resolved_subproblem == "auto":
            # CG needs a scale that does not read the Hessian diagonal, so the
            # default HessianScale keeps the dense exact solve at every size.
            cg_capable = (
                self.hessian == "true"
                and n_params > self.dense_cutoff
                and not isinstance(self.x_scale, HessianScale)
            )
            resolved_subproblem = "cg" if cg_capable else "exact"
        if self.hessian == "true" and isinstance(self.x_scale, JacScale):
            raise ConfigurationError(
                "NewtonTrustRegion cannot use JacScale with hessian='true' because "
                "that pairing forms a full forward residual Jacobian only for "
                "scaling. Use ds.inference.HessianScale() for an exact subproblem "
                "or ds.inference.ConstantScale(...) for an exact or CG subproblem."
            )
        if resolved_subproblem == "cg" and isinstance(self.x_scale, HessianScale):
            raise ConfigurationError(
                "NewtonTrustRegion cannot use HessianScale with subproblem='cg' "
                "because a matrix-free Hessian has no available diagonal. Use "
                "ds.inference.ConstantScale(...) for a CG solve."
            )
        if resolved_subproblem == "cg" and self.linear_solver != "auto":
            raise ConfigurationError(
                "NewtonTrustRegion linear_solver is not used with subproblem='cg'; "
                "leave linear_solver='auto'."
            )
        resolved_linear_solver = self.linear_solver
        if resolved_subproblem == "cg":
            resolved_linear_solver = "cg"
        elif resolved_linear_solver == "auto":
            if self.hessian == "gauss-newton":
                single = np.dtype(np.asarray(problem.x0).dtype).itemsize <= 4
                reduced = bool(problem.reduced_gauss_newton) or (
                    "residual_jacobian" not in problem.structure
                )
                resolved_linear_solver = "qr" if single and not reduced else "cholesky"
            else:
                resolved_linear_solver = "eigh"
        if self.hessian == "true" and resolved_linear_solver in ("cholesky", "qr"):
            raise ConfigurationError(
                "NewtonTrustRegion true Hessians can be indefinite and require "
                "linear_solver='eigh'; use hessian='gauss-newton' for Cholesky or QR."
            )
        if self.hessian == "true" and resolved_subproblem == "exact":
            refresh = self.hessian_refresh
            valid_refresh = refresh in ("adaptive", "every") or (
                isinstance(refresh, int) and not isinstance(refresh, bool) and refresh > 0
            )
            if not valid_refresh:
                raise ConfigurationError(
                    "NewtonTrustRegion hessian_refresh must be 'adaptive', 'every', "
                    "or a positive integer."
                )
        if self.hessian_tile != "auto":
            normalize_tangent_tile(self.hessian_tile, max(n_params, 1))
        return resolved_subproblem, composition, resolved_linear_solver

    def _checked_hessian(self, problem: Any, x: jax.Array, tile: Any, lanes: bool) -> Callable:
        """The exact-Hessian evaluation, memory-checked against the device before any step.

        ``"full"`` uses the problem's own Hessian. A tile pushes the gradient
        forward that many directions at a time. Either is compiled and checked
        here, so the problem's own check inside it is skipped and the solve
        reuses the compilation; under ``hessian_tile="auto"`` a tile that does
        not fit is narrowed down to 8 directions before the error is raised.
        """
        from diff_soup.inference.fitting.inference_problem import (
            _HESSIAN_GUARD_MIN_WIDTH,
            _HessianMemoryError,
            _MemoryCheckedHessian,
        )

        if lanes or (tile in (None, "full") and x.shape[0] <= _HESSIAN_GUARD_MIN_WIDTH):
            return problem.value_gradient_hessian
        if tile in (None, "full"):
            checked = _MemoryCheckedHessian(problem.value_gradient_hessian, tile="full")
            checked.check(x)
            return checked
        candidates = [tile]
        if self.hessian_tile == "auto":
            candidates += [t for t in _AUTO_HESSIAN_TILES if t < tile and t < x.shape[0]]
        error = None
        for candidate in candidates:
            checked = _MemoryCheckedHessian(
                _tiled_hessian(problem.value_and_gradient, candidate), tile=candidate
            )
            try:
                checked.check(x)
            except _HessianMemoryError as exc:
                error = exc
                continue
            return checked
        raise error

    def _radius_held_the_step(
        self, attempt: tuple, threshold: float, bounds: Any, lb: Any, ub: Any, cg_iters: int
    ) -> bool:
        """Whether the radius held the stopping step short of the model's own step.

        ``attempt`` is the model at the step's start: the point, its gradient, the
        model matrix (``None`` for the matrix-free Hessian), the scale, the
        Hessian-vector product, and the decrease predicted for the step taken.
        The model's own step is its step within ``radius_max``.
        """
        point, gradient, matrix, scale, model_hvp, predicted = attempt
        if matrix is not None:
            step, decrease, _multiplier = _more_sorensen(matrix, gradient, scale, self.radius_max)
        else:
            step, decrease, *_ = _steihaug_cg(model_hvp, gradient, scale, self.radius_max, cg_iters)
        if bounds is not None:
            step = jnp.clip(point + step, lb, ub) - point
            product = matrix @ step if matrix is not None else model_hvp(step)
            decrease = -(gradient @ step + 0.5 * step @ product)
        return model_held_the_step(float(local_value(decrease)), predicted, threshold)

    def solve(self, problem: Any, *, callbacks: tuple = ()) -> OptResult:
        """Optimize an inference problem with a scaled Newton trust region."""
        resolved_subproblem, resolved_composition, resolved_linear_solver = self._resolve_options(
            problem
        )
        cbs = tuple(self.callbacks) + tuple(callbacks)
        x0 = jnp.asarray(problem.x0)
        x = x0
        grad_tolerance = gradient_tolerance(self.grad_tol, x0.dtype, label="NewtonTrustRegion")
        grad_tol = None
        step_tol = relative_tolerance(
            self.step_tol, DEFAULT_STEP_TOL, x0.dtype, name="step_tol", label="NewtonTrustRegion"
        )
        cost_tol = relative_tolerance(
            self.cost_tol, DEFAULT_COST_TOL, x0.dtype, name="cost_tol", label="NewtonTrustRegion"
        )
        cost_atol = absolute_tolerance(
            self.cost_atol, DEFAULT_COST_ATOL, name="cost_atol", label="NewtonTrustRegion"
        )
        n_params = int(local_value(x.shape[0]))
        n_segments = int(problem.n_segments)
        reg = NoReg() if problem.regularization is None else problem.regularization
        bounds = problem.bounds
        lb = ub = None
        if bounds is not None:
            lb, ub = (jnp.asarray(side) for side in bounds)
            x = jnp.clip(x, lb, ub)
        if not trial_valid(problem, x):
            raise ValueError("NewtonTrustRegion initial point failed the trial predicate.")

        # The residual, when the problem has one, gives the Gauss-Newton linearization
        # and a forward-mode trial value; a problem evaluated on the lanes of a lowered
        # study has host oracles: every trial takes from one evaluation what an accepted
        # trial needs (the Gauss-Newton statistics, or the value and the gradient), and
        # the exact Hessian is the lanes' own. Gauss-Newton statistics reduced without a
        # residual Jacobian (the lanes, or a sharded joint) are used as they come, as
        # they are for a problem with no residual at all.
        forward_residual = problem.residual if "residual_jacobian" in problem.structure else None
        lanes = "batched_lanes" in problem.structure
        gauss_newton_lanes = self.hessian == "gauss-newton" and (
            bool(problem.reduced_gauss_newton) or forward_residual is None
        )
        gauss_newton = problem.gauss_newton if gauss_newton_lanes else None
        if lanes and self.hessian == "true" and resolved_subproblem != "exact":
            raise ConfigurationError(
                "NewtonTrustRegion with hessian='true' on a lowered study takes the exact "
                "Hessian of its lanes, which needs subproblem='exact'; a CG subproblem "
                "differentiates the gradient matrix-free, which the lanes do not offer."
            )
        if lanes and self.hessian == "true" and self.hessian_tile not in (None, "full", "auto"):
            raise ConfigurationError(
                "NewtonTrustRegion with hessian='true' on a lowered study takes the exact "
                "Hessian of its lanes whole; leave hessian_tile='full'."
            )
        if gauss_newton_lanes and resolved_linear_solver == "qr":
            raise ConfigurationError(
                "NewtonTrustRegion linear_solver='qr' factors the Gauss-Newton residual "
                "Jacobian, and a lowered study's lanes evaluate only J.T @ J and J.T @ r; use "
                "linear_solver='cholesky'."
            )
        data_loss = problem.value
        data_gradient = None
        data_hessian = None
        gradient_ad = None
        trial_data_loss = data_loss
        if self.hessian == "true":
            gradient_ad = "reverse" if resolved_composition == "forward-over-reverse" else "forward"
            data_gradient = problem.value_and_gradient
            if gradient_ad == "forward" and forward_residual is not None:

                @jax.jit
                def trial_data_loss(point):
                    residual = forward_residual(point)
                    return 0.5 * jnp.sum(residual * residual)

        # The Jacobian of the Gauss-Newton path is pushed forward without a reverse
        # tape, so "auto" forms it whole; only the exact Hessian is tiled by size.
        jacobian_tile = "full" if self.hessian_tile == "auto" else self.hessian_tile
        hessian_tile = self._resolved_hessian_tile(n_params, lanes)
        if self.hessian == "true" and resolved_subproblem == "exact":
            data_hessian = self._checked_hessian(problem, x, hessian_tile, lanes)
            hessian_tile = data_hessian.tile if hasattr(data_hessian, "tile") else hessian_tile

        counts = {
            "residual": 0,
            "jacobian": 0,
            "gradient": 0,
            "hessian": 0,
            "hvp": 0,
            "trial_predicate": 0,
            "invalid": 0,
            "factorization_fallback": 0,
            "factorizations": 0,
        }

        def eval_fg(point):
            loss, gradient = data_gradient(point)
            reg_loss, reg_gradient = jax.value_and_grad(reg.penalty)(point)
            counts["gradient"] += 1
            return loss + reg_loss, gradient + reg_gradient

        def eval_dense(point):
            loss, gradient, hessian = data_hessian(point)
            reg_loss, reg_gradient = jax.value_and_grad(reg.penalty)(point)
            reg_hessian = jax.hessian(reg.penalty)(point)
            counts["hessian"] += 1
            return (
                loss + reg_loss,
                gradient + reg_gradient,
                0.5 * (hessian + hessian.T) + reg_hessian,
            )

        def linearize_gauss_newton(point):
            data_residual, pushforward = linearize_value(forward_residual, point)
            reg_residual = reg.residual(point)
            residual = (
                jnp.concatenate((data_residual, reg_residual))
                if reg_residual.shape[0]
                else data_residual
            )
            counts["residual"] += 1
            loss = 0.5 * jnp.sum(residual * residual)
            failed = jnp.any(jnp.abs(data_residual) >= _FAILED_SENTINEL)
            return (
                loss,
                data_residual,
                reg_residual,
                pushforward,
                failed | ~jnp.isfinite(loss),
            )

        def finish_gauss_newton(point, data_residual, reg_residual, pushforward):
            jacobian = jacfwd_from_pushforward(
                pushforward,
                point,
                data_residual,
                tile=jacobian_tile,
            )
            residual = data_residual
            if reg_residual.shape[0]:
                residual = jnp.concatenate((data_residual, reg_residual))
                jacobian = jnp.concatenate(
                    (jacobian, jax.jacfwd(reg.residual)(point)),
                    axis=0,
                )
            counts["jacobian"] += 1
            gradient = jacobian.T @ residual
            hessian = jacobian.T @ jacobian
            return gradient, 0.5 * (hessian + hessian.T), jacobian, residual

        def eval_gauss_newton(point):
            linearized = linearize_gauss_newton(point)
            loss, data_residual, reg_residual, pushforward, failed = linearized
            gradient, hessian, jacobian, residual = finish_gauss_newton(
                point, data_residual, reg_residual, pushforward
            )
            return loss, gradient, hessian, jacobian, residual, failed

        def lanes_gauss_newton(point):
            loss, gradient, hessian, failed = gauss_newton(point)
            reg_residual = reg.residual(point)
            if reg_residual.shape[0]:
                reg_jacobian = jax.jacfwd(reg.residual)(point)
                loss = loss + 0.5 * jnp.sum(reg_residual * reg_residual)
                gradient = gradient + reg_jacobian.T @ reg_residual
                hessian = hessian + reg_jacobian.T @ reg_jacobian
            counts["residual"] += 1
            failed = jnp.asarray(failed > 0) | ~jnp.isfinite(loss)
            return loss, gradient, 0.5 * (hessian + hessian.T), failed

        def linearize_true_trial(point):
            if lanes:
                data_value, data_slope = data_gradient(point)

                def data_grad():
                    return data_slope

            else:
                data_value, data_grad = deferred_value_and_grad(
                    trial_data_loss,
                    point,
                    ad=gradient_ad,
                )
            total = data_value + reg.penalty(point)
            counts["residual"] += 1
            failed = (
                ~jnp.isfinite(total)
                | ~jnp.isfinite(data_value)
                | (data_value >= 0.5 * _FAILED_SENTINEL**2)
            )
            return total, data_grad, failed

        def finish_true_trial(point, value, data_grad):
            _reg_loss, reg_gradient = jax.value_and_grad(reg.penalty)(point)
            counts["gradient"] += 1
            return value, data_grad() + reg_gradient

        def jacobian_for_scale(point):
            _residual, jacobian = value_and_jacfwd_tiled(
                forward_residual,
                point,
                tile=jacobian_tile,
            )
            reg_rows = reg.residual(point)
            if reg_rows.shape[0]:
                reg_jacobian = jax.jacfwd(reg.residual)(point)
                jacobian = jnp.concatenate((jacobian, reg_jacobian), axis=0)
            counts["residual"] += 1
            counts["jacobian"] += 1
            return jacobian

        # The Gauss-Newton path holds the residual and Jacobian of its current point.
        current_residual = None
        if gauss_newton_lanes:
            f, g, B, residual_failed = lanes_gauss_newton(x)
            counts["jacobian"] += 1
            current_jacobian = None
        elif self.hessian == "gauss-newton":
            f, g, B, current_jacobian, current_residual, residual_failed = eval_gauss_newton(x)
        elif resolved_subproblem == "exact":
            f, g, B = eval_dense(x)
            current_jacobian = None
            residual_failed = jnp.asarray(False)
        else:
            f, g = eval_fg(x)
            B = None
            current_jacobian = None
            residual_failed = jnp.asarray(False)
        initial_failed = residual_failed | ~jnp.isfinite(f) | (f >= 0.5 * _FAILED_SENTINEL**2)
        if bool(local_value(initial_failed)):
            raise ValueError("NewtonTrustRegion initial PDE solve failed.")
        if not bool(local_value(jnp.all(jnp.isfinite(g)))):
            raise ValueError("NewtonTrustRegion initial loss and gradient must be finite.")

        n_residuals = int(getattr(problem, "n_residuals", None) or 0)
        for callback in cbs:
            callback.on_start(n_params, n_residuals, self.max_steps)

        t0_wall = time.perf_counter()
        radius = float(self.radius_init)
        scale = None
        cost_history = []
        grad_history = []
        radius_history = [radius]
        n_accepted = 0
        failure_growth_lock = 0
        steps_since_hessian = 0
        refresh_at_start = False
        hessian_exact_at_x = self.hessian == "true" and resolved_subproblem == "exact"
        status = TerminationStatus.MAX_STEPS
        stall_reason = None
        cg_iters = n_params if self.cg_max_iter is None else self.cg_max_iter

        for step_index in range(self.max_steps):
            if refresh_at_start:
                f, g, B = eval_dense(x)
                refresh_at_start = False
                steps_since_hessian = 0
                hessian_exact_at_x = True

            if isinstance(self.x_scale, HessianScale):
                candidate_scale = jnp.sqrt(self.x_scale.damping_diagonal(B, x))
                scale = candidate_scale if scale is None else jnp.maximum(scale, candidate_scale)
            elif isinstance(self.x_scale, JacScale) and gauss_newton_lanes:
                scale = jnp.sqrt(self.x_scale.damping_from_gram(B, x))
            elif isinstance(self.x_scale, JacScale):
                scale_jacobian = (
                    current_jacobian if current_jacobian is not None else jacobian_for_scale(x)
                )
                scale = jnp.sqrt(self.x_scale.damping_diagonal(scale_jacobian, x))
            else:
                matrix = B if B is not None else jnp.eye(n_params, dtype=x.dtype)
                scale = jnp.sqrt(self.x_scale.damping_diagonal(matrix, x))

            grad_norm = float(local_value(jnp.linalg.norm(g)))
            stop_gradient = (
                grad_norm
                if bounds is None
                else float(local_value(projected_gradient_norm(x, g, lb, ub)))
            )
            cost_history.append(float(local_value(f)))
            grad_history.append(grad_norm)
            if grad_tol is None:
                grad_tol = grad_tolerance.threshold(stop_gradient)
            if assert_decision_value(stop_gradient <= grad_tol, label="Newton trust region"):
                status = TerminationStatus.GRAD_TOL
                break

            model_hvp = None
            if resolved_subproblem == "exact":
                if resolved_linear_solver == "eigh":
                    proposed_step, predicted, multiplier = _more_sorensen(B, g, scale, radius)
                else:
                    proposed_step, predicted, multiplier, used_fallback, factorizations = (
                        _factorized_psd_trust_region(
                            B,
                            g,
                            scale,
                            radius,
                            resolved_linear_solver,
                            current_jacobian,
                        )
                    )
                    counts["factorization_fallback"] += int(local_value(used_fallback))
                    counts["factorizations"] += factorizations
            else:
                if gauss_newton_lanes:

                    def model_hvp(vector):
                        return B @ vector

                elif self.hessian == "gauss-newton":
                    reg_hessian = B - current_jacobian.T @ current_jacobian

                    def model_hvp(vector):
                        return (
                            current_jacobian.T @ (current_jacobian @ vector) + reg_hessian @ vector
                        )

                else:
                    data_hessian_vector = problem.hessian_vector

                    def model_hvp(vector, x=x):
                        data_product = data_hessian_vector(x, vector)
                        reg_product = jax.jvp(jax.grad(reg.penalty), (x,), (vector,))[1]
                        return data_product + reg_product

                proposed_step, predicted, products, _boundary, _negative = _steihaug_cg(
                    model_hvp,
                    g,
                    scale,
                    radius,
                    cg_iters,
                )
                counts["hvp"] += products
                multiplier = None

            trial = x + proposed_step
            if bounds is not None:
                trial = jnp.clip(trial, lb, ub)
            actual_step = trial - x
            scaled_step_norm = float(local_value(jnp.linalg.norm(scale * actual_step)))
            step_norm = float(local_value(jnp.linalg.norm(actual_step)))
            if bounds is not None:
                if B is not None:
                    predicted = _quadratic_reduction(B, g, actual_step)
                else:
                    product = model_hvp(actual_step)
                    counts["hvp"] += 1
                    predicted = -(g @ actual_step + 0.5 * actual_step @ product)

            try:
                trial_ok = trial_valid(problem, trial)
            except Exception:
                trial_ok = False
            if not trial_ok:
                counts["trial_predicate"] += 1
                trial_cost = jnp.asarray(jnp.inf, dtype=f.dtype)
                invalid_pde = False
            else:
                try:
                    if gauss_newton_lanes:
                        trial_linearized = lanes_gauss_newton(trial)
                        trial_cost, *_unused, failed = trial_linearized
                    elif self.hessian == "gauss-newton":
                        trial_linearized = linearize_gauss_newton(trial)
                        trial_cost, *_unused, failed = trial_linearized
                    else:
                        trial_cost, trial_data_grad, failed = linearize_true_trial(trial)
                    invalid_pde = bool(local_value(failed))
                except Exception:
                    trial_cost = jnp.asarray(jnp.inf, dtype=f.dtype)
                    invalid_pde = True
                if invalid_pde:
                    counts["invalid"] += 1

            predicted_value = float(local_value(predicted))
            if (
                invalid_pde
                or not np.isfinite(local_value(float(local_value(trial_cost))))
                or predicted_value <= 0.0
            ):
                rho = float("-inf")
            else:
                rho = (float(local_value(f)) - float(local_value(trial_cost))) / predicted_value
            accepted = assert_decision_value(
                trial_ok and not invalid_pde and rho >= self.eta, label="Newton trust region accept"
            )
            # The model at the point this step was taken from, for the stall test.
            attempt = (x, g, B, scale, model_hvp, predicted_value)

            if rho < 0.25:
                radius = max(radius / 4.0, np.finfo(float).tiny)
            elif rho > 0.75 and scaled_step_norm >= 0.99 * radius and failure_growth_lock == 0:
                radius = min(2.0 * radius, self.radius_max)

            previous_f = f
            previous_g = g
            if accepted:
                x = trial
                n_accepted += 1
                if failure_growth_lock:
                    failure_growth_lock = max(0, failure_growth_lock - 1)
                if gauss_newton_lanes:
                    f, g, B, _failed = trial_linearized
                    counts["jacobian"] += 1
                elif self.hessian == "gauss-newton":
                    f, data_residual, reg_residual, pushforward, _failed = trial_linearized
                    g, B, current_jacobian, current_residual = finish_gauss_newton(
                        x, data_residual, reg_residual, pushforward
                    )
                else:
                    f, g = finish_true_trial(x, trial_cost, trial_data_grad)
                    if resolved_subproblem == "exact":
                        hessian_exact_at_x = False
                        refresh = self.hessian_refresh
                        steps_since_hessian += 1
                        refresh_at_start = (
                            refresh == "every"
                            or (refresh == "adaptive" and rho < 0.25)
                            or (refresh == "adaptive" and steps_since_hessian >= 5)
                            or (
                                isinstance(refresh, int)
                                and not isinstance(refresh, bool)
                                and steps_since_hessian >= refresh
                            )
                        )
                        if not refresh_at_start:
                            B, _updated = _sr1_update(B, actual_step, g - previous_g)
            else:
                if invalid_pde:
                    failure_growth_lock = 2
                if self.hessian == "true" and resolved_subproblem == "exact":
                    steps_since_hessian += 1
                    refresh = self.hessian_refresh
                    refresh_at_start = (
                        refresh == "every"
                        or (refresh == "adaptive" and not hessian_exact_at_x)
                        or (
                            isinstance(refresh, int)
                            and not isinstance(refresh, bool)
                            and steps_since_hessian >= refresh
                        )
                    )

            radius_history.append(radius)
            step_info = StepInfo(
                step=step_index,
                cost=float(local_value(f)),
                grad_norm=float(local_value(jnp.linalg.norm(g))),
                params=local_value(x),
                accepted=accepted,
                delta_norm=step_norm,
                wall_time=time.perf_counter() - t0_wall,
                rho=rho,
                lam=None if multiplier is None else float(multiplier),
                radius=float(radius),
                accepted_count=n_accepted,
                residual_evals=counts["residual"],
                jacobian_evals=counts["jacobian"],
                trial_predicate_rejections=counts["trial_predicate"],
                invalid_pde_trials=counts["invalid"],
            )
            stop_requested = False
            for callback in cbs:
                try:
                    callback.on_step(step_info)
                except StopFit:
                    stop_requested = True
            if any_process(stop_requested):
                status = TerminationStatus.STOPPED
                break

            actual_reduction = float(local_value(previous_f)) - float(local_value(trial_cost))
            # Relative to the cost itself, so where a fit stops does not depend
            # on the units of its objective; ``cost_atol`` is the absolute floor.
            threshold = cost_threshold(float(local_value(previous_f)), cost_tol, cost_atol)
            stop = None
            if assert_decision_value(
                trial_ok
                and not invalid_pde
                and step_norm <= step_tol * (1.0 + float(local_value(jnp.linalg.norm(x)))),
                label="Newton trust region",
            ):
                stop = TerminationStatus.STEP_TOL
            # A cost stop needs the model to agree with the decrease (SciPy's ratio).
            elif assert_decision_value(
                accepted and rho > 0.25 and abs(actual_reduction) <= threshold,
                label="Newton trust region",
            ):
                stop = TerminationStatus.COST_TOL
            if stop is not None:
                status = stop
                if assert_decision_value(
                    self._radius_held_the_step(attempt, threshold, bounds, lb, ub, cg_iters),
                    label="Newton trust region stall",
                ):
                    status = TerminationStatus.STALLED
                    stall_reason = stall_reason_of(
                        invalid=failure_growth_lock > 0,
                        noise=not accepted and abs(actual_reduction) <= threshold,
                    )
                break
        else:
            step_index = self.max_steps - 1

        wall_time = time.perf_counter() - t0_wall
        final_grad_norm = float(local_value(jnp.linalg.norm(g)))
        final_projected_grad_norm = (
            final_grad_norm
            if bounds is None
            else float(local_value(projected_gradient_norm(x, g, lb, ub)))
        )
        fitted = problem.fitted(x)

        # The Gauss-Newton path holds the residual and Jacobian of the returned
        # point: an accepted trial is linearized at its point and a rejected one
        # keeps the point with its linearization. Rows are the problem's residual
        # followed by its regularization rows. They are recorded only when their
        # cost reproduces the returned cost, as LevenbergMarquardt records them.
        final_arrays = {}
        if current_residual is not None and current_jacobian is not None:
            final_cost = 0.5 * float(local_value(jnp.sum(current_residual * current_residual)))
            if final_cost == float(local_value(f)):
                final_arrays = {
                    "final_residual": np.asarray(local_value(current_residual)),
                    "final_jacobian": np.asarray(local_value(current_jacobian)),
                }

        total_pde_evaluations = (
            counts["residual"]
            + counts["jacobian"]
            + counts["gradient"]
            + n_params * counts["hessian"]
            + counts["hvp"]
        )
        result = OptResult(
            opt_params=fitted,
            params=local_value(x),
            initial_params=local_value(x0),
            cost=float(local_value(f)),
            grad_norm=final_grad_norm,
            projected_grad_norm=final_projected_grad_norm,
            status=status,
            message=status.value,
            n_steps=step_index + 1,
            max_steps=self.max_steps,
            n_accepted=n_accepted,
            eval_counts=EvalCounts(
                residual_evals=counts["residual"],
                jacobian_evals=counts["jacobian"],
                gradient_evals=counts["gradient"],
                hessian_evals=counts["hessian"],
                hessian_vector_products=counts["hvp"],
                pde_solves=total_pde_evaluations * n_segments,
                trial_predicate_rejections=counts["trial_predicate"],
                invalid_pde_trials=counts["invalid"],
            ),
            cost_history=local_value(cost_history),
            grad_history=local_value(grad_history),
            wall_time=wall_time,
            info={
                "hessian": self.hessian,
                "hessian_composition": resolved_composition,
                "subproblem": resolved_subproblem,
                "hessian_tile": (
                    hessian_tile
                    if self.hessian == "true" and resolved_subproblem == "exact"
                    else None
                ),
                "linear_solver": resolved_linear_solver,
                "linear_solver_fallbacks": counts["factorization_fallback"],
                "linear_factorizations": counts["factorizations"],
                **({} if stall_reason is None else {"stall_reason": stall_reason}),
                **final_arrays,
            },
            trust_region=TrustRegionReport(radius_history=radius_history),
        )
        for callback in cbs:
            callback.on_end(result)
        return result


__all__ = ["NewtonTrustRegion"]
