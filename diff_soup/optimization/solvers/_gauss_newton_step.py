"""The Gauss-Newton step of Levenberg-Marquardt, through the Jacobian or the normal equations, and its predicted decrease."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup._distributed import local_value
from diff_soup.optimization.solvers.factorization import (
    cholesky_factor,
    solve_augmented_qr,
    solve_factorized_normal,
)

# ---------------------------------------------------------------------------
# Gauss-Newton step (pure JAX, runs on GPU for small n_params)
# ---------------------------------------------------------------------------


def _solve_spd(matrix: jax.Array, rhs: jax.Array, linear_solver: str) -> jax.Array:
    """Solve a dense damped normal equation with the requested factorization."""
    if linear_solver == "solve":
        return jnp.linalg.solve(matrix, rhs)
    return solve_factorized_normal(cholesky_factor(matrix), rhs)


def _gn_step(
    J: jax.Array,
    r: jax.Array,
    lam: float,
    max_step_norm: float,
    d: jax.Array,
    linear_solver: str = "solve",
):
    """Compute damped Gauss-Newton step with optional clipping.

    Args:
        J: Jacobian of shape ``(m, n)``.
        r: Residual of shape ``(m,)``.
        lam: Levenberg-Marquardt damping parameter.
        max_step_norm: If positive, clip the step to this norm.
        d: Positive damping diagonal of length ``n`` (from the scale object).
            The damped normal equations use ``D = diag(d)``.
        linear_solver: ``"cholesky"`` or ``"solve"`` for the damped normal
            equations, or ``"qr"`` for the augmented least-squares system.

    Returns ``(delta, dnorm, pred_red, Jtr, grad_norm, JtJ)``.
    """
    Jtr = J.T @ r
    grad_norm = jnp.linalg.norm(Jtr)
    JtJ = J.T @ J

    D = jnp.diag(d)

    if linear_solver == "qr":
        damping_rows = jnp.sqrt(lam * d)[:, None] * jnp.eye(J.shape[1], dtype=J.dtype)
        augmented_jacobian = jnp.concatenate((J, damping_rows), axis=0)
        augmented_rhs = jnp.concatenate((-r, jnp.zeros(J.shape[1], dtype=r.dtype)))
        delta = solve_augmented_qr(augmented_jacobian, augmented_rhs)
    else:
        delta = _solve_spd(JtJ + lam * D, -Jtr, linear_solver)
    dnorm = jnp.linalg.norm(delta)

    # Step-norm clipping
    clip = jnp.where(
        (max_step_norm > 0) & (dnorm > max_step_norm),
        max_step_norm / dnorm,
        1.0,
    )
    delta = delta * clip
    dnorm = dnorm * clip

    pred_red = -(Jtr @ delta + 0.5 * delta @ (JtJ @ delta))

    return delta, dnorm, pred_red, Jtr, grad_norm, JtJ


def _gn_step_normal(
    JtJ: jax.Array,
    Jtr: jax.Array,
    lam: float,
    max_step_norm: float,
    d: jax.Array,
    linear_solver: str = "solve",
):
    """Damped Gauss-Newton step from the normal equations directly.

    The distributed path reduces ``J^T J`` and ``J^T r`` across the device mesh and
    never forms the full Jacobian, so the step is computed from those replicated
    matrices.  Mirrors
    [_gn_step][diff_soup.optimization.solvers._gauss_newton_step._gn_step]
    exactly given the same ``(J^T J, J^T r)``.

    Returns ``(delta, dnorm, pred_red, Jtr, grad_norm)``.
    """
    grad_norm = jnp.linalg.norm(Jtr)
    D = jnp.diag(d)

    delta = _solve_spd(JtJ + lam * D, -Jtr, linear_solver)
    dnorm = jnp.linalg.norm(delta)

    clip = jnp.where(
        (max_step_norm > 0) & (dnorm > max_step_norm),
        max_step_norm / dnorm,
        1.0,
    )
    delta = delta * clip
    dnorm = dnorm * clip

    pred_red = -(Jtr @ delta + 0.5 * delta @ (JtJ @ delta))

    return delta, dnorm, pred_red, Jtr, grad_norm


def _gauss_newton_decrease(gram, gradient, point, lb=None, ub=None) -> float:
    """The decrease the Gauss-Newton model predicts for its undamped step at ``point``.

    The step is the minimum-norm minimizer of ``g.T p + 0.5 p.T G p`` over the
    range of ``G`` (eigenvalues above ``n * eps`` of the largest), projected onto
    the box when bounds are given.
    """
    gram = 0.5 * (gram + gram.T)
    values, vectors = jnp.linalg.eigh(gram)
    cutoff = values.shape[0] * jnp.finfo(values.dtype).eps * jnp.maximum(values[-1], 0.0)
    coefficients = vectors.T @ gradient
    keep = values > cutoff
    safe = jnp.where(keep, values, 1.0)
    step = -vectors @ jnp.where(keep, coefficients / safe, 0.0)
    if lb is not None:
        step = jnp.clip(point + step, lb, ub) - point
    return float(local_value(-(gradient @ step + 0.5 * step @ (gram @ step))))
