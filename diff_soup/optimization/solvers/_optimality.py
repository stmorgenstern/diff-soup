"""The scaled optimality error of the barrier problem (Wächter and Biegler 2006, eq. 5).

`E_mu` is the largest of the dual infeasibility, the constraint violation and the
complementarity error `x z - mu`, with the dual and complementarity parts divided by `s_d`
and `s_c`, which grow with the multipliers past `s_max`. `E_0` is the error of the original
problem; the run ends when it falls below `tol`, and the barrier parameter is reduced when
`E_mu <= kappa_eps * mu`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from diff_soup.optimization.solvers._barrier import Bounds, distances

S_MAX = 100.0


def _max_abs(*parts: Any) -> Any:
    """The infinity norm over several vectors, 0 when all are empty."""
    return jnp.max(jnp.concatenate([jnp.abs(p).ravel() for p in parts]), initial=0.0)


def dual_residuals(evaluation: Any, lam_e: Any, lam_i: Any, z_lower: Any, z_upper: Any, v: Any):
    """`(r_x, r_s)`: the gradient of the Lagrangian in `x` and in the slacks."""
    r_x = (
        evaluation.gradient
        + evaluation.jacobian_equality.T @ lam_e
        + evaluation.jacobian_inequality.T @ lam_i
        - z_lower
        + z_upper
    )
    return r_x, lam_i - v


def optimality_error(
    evaluation: Any,
    x: Any,
    s: Any,
    lam_e: Any,
    lam_i: Any,
    z_lower: Any,
    z_upper: Any,
    v: Any,
    mu: Any,
    bounds: Bounds,
) -> Any:
    """The scaled optimality error `E_mu` at one primal-dual point."""
    r_x, r_s = dual_residuals(evaluation, lam_e, lam_i, z_lower, z_upper, v)
    below, above = distances(x, bounds)
    complementarity = (
        jnp.where(bounds.has_lower, below * z_lower - mu, 0.0),
        jnp.where(bounds.has_upper, above * z_upper - mu, 0.0),
        s * v - mu,
    )
    n_z = jnp.sum(bounds.has_lower) + jnp.sum(bounds.has_upper) + s.shape[0]
    n_lam = lam_e.shape[0] + lam_i.shape[0]
    z_norm = jnp.sum(jnp.abs(z_lower)) + jnp.sum(jnp.abs(z_upper)) + jnp.sum(jnp.abs(v))
    lam_norm = jnp.sum(jnp.abs(lam_e)) + jnp.sum(jnp.abs(lam_i))
    s_d = jnp.maximum(S_MAX, (lam_norm + z_norm) / jnp.maximum(n_lam + n_z, 1)) / S_MAX
    s_c = jnp.maximum(S_MAX, z_norm / jnp.maximum(n_z, 1)) / S_MAX
    dual = _max_abs(r_x, r_s) / s_d
    primal = _max_abs(evaluation.equality, evaluation.inequality + s)
    return jnp.maximum(jnp.maximum(dual, primal), _max_abs(*complementarity) / s_c)


def constraint_violation(evaluation: Any) -> Any:
    """The largest violation of the original constraints, without slacks."""
    return jnp.maximum(_max_abs(evaluation.equality), jnp.max(evaluation.inequality, initial=0.0))
