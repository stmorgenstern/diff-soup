"""The scaled optimality error of the barrier problem (Wächter and Biegler 2006, eq. 5).

`E_mu` is the largest of the dual infeasibility, the constraint violation and the
complementarity error `x z - mu`, with the dual and complementarity parts divided by `s_d`
and `s_c`, which grow with the multipliers past `s_max`. `E_0` is the error of the original
problem; the run ends when it falls below `tol`, and the barrier parameter is reduced when
`E_mu <= kappa_eps * mu`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from diff_soup.optimization.solvers._barrier import Bounds, distances
from diff_soup.optimization.solvers._closures import Evaluation, Scales

#: Ipopt's `s_max`: the multiplier size past which the dual and complementarity errors shrink.
S_MAX = 100.0


def _max_abs(*parts: Float[Array, "..."]) -> Float[Array, ""]:
    """The infinity norm over several vectors, 0 when all are empty."""
    return jnp.max(jnp.concatenate([jnp.abs(p).ravel() for p in parts]), initial=0.0)


def dual_residuals(
    evaluation: Evaluation,
    lam_e: Float[Array, " n_eq"],
    lam_i: Float[Array, " n_ineq"],
    z_lower: Float[Array, " n"],
    z_upper: Float[Array, " n"],
    v: Float[Array, " n_ineq"],
) -> tuple[Float[Array, " n"], Float[Array, " n_ineq"]]:
    """`(r_x, r_s)`: the gradient of the Lagrangian in `x` and in the slacks."""
    r_x = (
        evaluation.gradient
        + evaluation.jacobian_equality.T @ lam_e
        + evaluation.jacobian_inequality.T @ lam_i
        - z_lower
        + z_upper
    )
    return r_x, lam_i - v


def scaled_error(
    dual: tuple[Float[Array, "..."], ...],
    primal: tuple[Float[Array, "..."], ...],
    complementarity: tuple[Float[Array, "..."], ...],
    lam_norm: Float[Array, ""],
    n_lam: int,
    z_norm: Float[Array, ""],
    n_z: Int[Array, ""],
) -> Float[Array, ""]:
    """`E_mu` from its parts: the dual and complementarity residuals scaled by `s_d` and `s_c`."""
    s_d = jnp.maximum(S_MAX, (lam_norm + z_norm) / jnp.maximum(n_lam + n_z, 1)) / S_MAX
    s_c = jnp.maximum(S_MAX, z_norm / jnp.maximum(n_z, 1)) / S_MAX
    error = jnp.maximum(_max_abs(*dual) / s_d, _max_abs(*primal))
    return jnp.maximum(error, _max_abs(*complementarity) / s_c)


def optimality_error(
    evaluation: Evaluation,
    x: Float[Array, " n"],
    s: Float[Array, " n_ineq"],
    lam_e: Float[Array, " n_eq"],
    lam_i: Float[Array, " n_ineq"],
    z_lower: Float[Array, " n"],
    z_upper: Float[Array, " n"],
    v: Float[Array, " n_ineq"],
    mu: Float[ArrayLike, ""],
    bounds: Bounds,
) -> Float[Array, ""]:
    """The scaled optimality error `E_mu` at one primal-dual point."""
    r_x, r_s = dual_residuals(evaluation, lam_e, lam_i, z_lower, z_upper, v)
    below, above = distances(x, bounds)
    complementarity = (
        jnp.where(bounds.has_lower, below * z_lower - mu, 0.0),
        jnp.where(bounds.has_upper, above * z_upper - mu, 0.0),
        s * v - mu,
    )
    n_z = jnp.sum(bounds.has_lower) + jnp.sum(bounds.has_upper) + s.shape[0]
    z_norm = jnp.sum(jnp.abs(z_lower)) + jnp.sum(jnp.abs(z_upper)) + jnp.sum(jnp.abs(v))
    lam_norm = jnp.sum(jnp.abs(lam_e)) + jnp.sum(jnp.abs(lam_i))
    return scaled_error(
        (r_x, r_s),
        (evaluation.equality, evaluation.inequality + s),
        complementarity,
        lam_norm,
        lam_e.shape[0] + lam_i.shape[0],
        z_norm,
        n_z,
    )


def constraint_violation(evaluation: Evaluation, scales: Scales) -> Float[Array, ""]:
    """The largest violation of the original, unscaled constraints, without slacks."""
    equality = evaluation.equality / scales.equality
    inequality = evaluation.inequality / scales.inequality
    return jnp.maximum(_max_abs(equality), jnp.max(inequality, initial=0.0))
