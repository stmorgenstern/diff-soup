"""Reusable dense factorizations for parameter-space optimizer systems."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TriangularFactor(NamedTuple):
    """A triangular square-root factor of a positive-semidefinite matrix."""

    triangular: jax.Array
    lower: bool


def cholesky_factor(matrix: jax.Array) -> TriangularFactor:
    """Factor a symmetric positive-definite matrix as ``L @ L.T``."""
    return TriangularFactor(jnp.linalg.cholesky(matrix), True)


def qr_normal_factor(matrix: jax.Array) -> TriangularFactor:
    """Factor the normal matrix of a rectangular matrix through its QR factor."""
    _orthogonal, triangular = jnp.linalg.qr(matrix, mode="reduced")
    return TriangularFactor(triangular, False)


def solve_factorized_normal_with_half(
    factor: TriangularFactor, rhs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Solve a normal system and retain its intermediate half solve."""
    triangular, lower = factor
    first = jax.scipy.linalg.solve_triangular(
        triangular if lower else triangular.T,
        rhs,
        lower=True,
    )
    solution = jax.scipy.linalg.solve_triangular(
        triangular.T if lower else triangular,
        first,
        lower=False,
    )
    return solution, first


def solve_factorized_normal(factor: TriangularFactor, rhs: jax.Array) -> jax.Array:
    """Solve the normal system represented by one triangular factor."""
    return solve_factorized_normal_with_half(factor, rhs)[0]


def solve_factorized_half(factor: TriangularFactor, rhs: jax.Array) -> jax.Array:
    """Apply the inverse factor used by a trust-region secular derivative."""
    triangular, lower = factor
    return jax.scipy.linalg.solve_triangular(
        triangular if lower else triangular.T,
        rhs,
        lower=True,
    )


def solve_augmented_qr(matrix: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve a rectangular least-squares problem with one reduced QR factorization."""
    orthogonal, triangular = jnp.linalg.qr(matrix, mode="reduced")
    return jax.scipy.linalg.solve_triangular(
        triangular,
        orthogonal.T @ rhs,
        lower=False,
    )


__all__ = [
    "TriangularFactor",
    "cholesky_factor",
    "qr_normal_factor",
    "solve_augmented_qr",
    "solve_factorized_half",
    "solve_factorized_normal",
    "solve_factorized_normal_with_half",
]
