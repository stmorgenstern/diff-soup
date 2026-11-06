"""The Lagrangian Hessian a step uses: exact, Gauss-Newton, damped BFGS, or the two combined.

- `"exact"`: the problem's `lagrangian_hessian`.
- `"gauss-newton"`: `J_r^T J_r` of a least-squares objective; the constraint curvature is left out.
- `"bfgs"`: Powell-damped BFGS on the gradient of the Lagrangian, from the identity scaled at
  the first update.
- `"auto"`: on a least-squares objective, Gauss-Newton plus a damped BFGS matrix that learns
  the rest of the curvature (the structured secant `B s = y - J_r^T J_r s`, from zero); on any
  other objective, `"bfgs"`.

Powell's damping keeps every BFGS matrix positive semidefinite (Nocedal and Wright, Procedure
18.2), so a quasi-Newton step never needs the inertia correction for its own sake.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from diff_soup._errors import ConfigurationError

MODES = ("auto", "bfgs", "exact", "gauss-newton")
_DAMPING = 0.2


def check_mode(mode: str, closures: Any) -> None:
    """Refuse a mode the problem cannot serve, before anything is traced."""
    if mode not in MODES:
        raise ConfigurationError(f"hessian must be one of {MODES}; got {mode!r}.")
    if mode == "exact" and closures.lagrangian_hessian is None:
        raise ConfigurationError(
            "hessian='exact' needs the problem's lagrangian_hessian closure; the problem has "
            "none, so use hessian='auto' or 'bfgs'."
        )
    if mode == "gauss-newton" and closures.gauss_newton is None:
        raise ConfigurationError(
            "hessian='gauss-newton' needs a least-squares objective (the 'gauss_newton' "
            "structure); use hessian='auto' or 'bfgs'."
        )


def uses_gauss_newton(mode: str, closures: Any) -> bool:
    """Whether the step adds the Gauss-Newton matrix of the objective."""
    return mode == "gauss-newton" or (mode == "auto" and closures.gauss_newton is not None)


def uses_quasi_newton(mode: str) -> bool:
    """Whether the step carries a BFGS matrix."""
    return mode in ("auto", "bfgs")


def initial_matrix(mode: str, closures: Any, n: int, dtype: Any) -> Any:
    """The BFGS matrix before the first update: the identity, zero when it adds to Gauss-Newton."""
    if not uses_quasi_newton(mode):
        return jnp.zeros((0, 0), dtype)
    if uses_gauss_newton(mode, closures):
        return jnp.zeros((n, n), dtype)
    return jnp.eye(n, dtype=dtype)


def lagrangian_hessian(
    mode: str,
    closures: Any,
    x: Any,
    lam_e: Any,
    lam_i: Any,
    args: Any,
    evaluation: Any,
    matrix: Any,
) -> Any:
    """The Lagrangian Hessian, or its approximation, at the current point."""
    if mode == "exact":
        return closures.lagrangian_hessian(x, lam_e, lam_i, args)
    hessian = evaluation.gauss_newton if uses_gauss_newton(mode, closures) else 0.0
    if uses_quasi_newton(mode):
        hessian = hessian + matrix
    return hessian


def update(
    mode: str,
    closures: Any,
    matrix: Any,
    updates: Any,
    step: Any,
    gradient_change: Any,
    evaluation: Any,
) -> tuple[Any, Any]:
    """The damped BFGS update after a step.

    Args:
        mode: The Hessian mode.
        closures: The problem's closures.
        matrix: The BFGS matrix before the step.
        updates: How many updates the matrix has had.
        step: `x_new - x`.
        gradient_change: `grad L(x_new, lam_new) - grad L(x, lam_new)`.
        evaluation: The evaluation at `x_new`, whose Gauss-Newton matrix the structured secant
            subtracts.

    Returns:
        `(matrix, updates)`.
    """
    if not uses_quasi_newton(mode):
        return matrix, updates
    tiny = jnp.finfo(matrix.dtype).tiny
    secant = gradient_change
    if uses_gauss_newton(mode, closures):
        secant = secant - evaluation.gauss_newton @ step
    else:
        curvature = step @ secant
        initial = (secant @ secant) / jnp.where(curvature > 0.0, curvature, 1.0)
        first = (updates == 0) & (curvature > 0.0)
        matrix = jnp.where(first, initial * jnp.eye(matrix.shape[0], dtype=matrix.dtype), matrix)
    matrix_step = matrix @ step
    quadratic = step @ matrix_step
    curvature = step @ secant
    theta = jnp.where(
        curvature >= _DAMPING * quadratic,
        1.0,
        (1.0 - _DAMPING)
        * quadratic
        / jnp.where(quadratic - curvature > tiny, quadratic - curvature, 1.0),
    )
    damped = theta * secant + (1.0 - theta) * matrix_step
    damped_curvature = step @ damped
    removed = jnp.where(
        quadratic > tiny,
        jnp.outer(matrix_step, matrix_step) / jnp.where(quadratic > tiny, quadratic, 1.0),
        0.0,
    )
    scale = jnp.sqrt(step @ step) * jnp.sqrt(damped @ damped)
    added_ok = damped_curvature > 1e-12 * scale
    added = jnp.where(
        added_ok,
        jnp.outer(damped, damped) / jnp.where(added_ok, damped_curvature, 1.0),
        0.0,
    )
    moved = jnp.any(step != 0.0) & jnp.all(jnp.isfinite(damped))
    new = matrix - removed + added
    new = 0.5 * (new + new.T)
    return jnp.where(moved, new, matrix), updates + moved.astype(updates.dtype)
