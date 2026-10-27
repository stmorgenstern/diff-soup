"""Small COO-pattern kernels shared by the implicit problem model.

Entry order follows the owning pattern's ``rows``/``cols``; duplicate
positions accumulate. All kernels are linear in the value vector and in the
multiplied vector, so JAX differentiates and transposes them exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["coo_matvec", "coo_rmatvec"]


def coo_matvec(values, rows: np.ndarray, cols: np.ndarray, n: int, x) -> jax.Array:
    """Return ``A @ x`` for the matrix the ``(nnz,)`` values represent."""
    x = jnp.asarray(x)
    out = jnp.zeros((n,), dtype=jnp.result_type(values, x))
    return out.at[rows].add(jnp.asarray(values) * x[cols])


def coo_rmatvec(values, rows: np.ndarray, cols: np.ndarray, n: int, w) -> jax.Array:
    """Return ``A.T @ w`` for the matrix the ``(nnz,)`` values represent."""
    w = jnp.asarray(w)
    out = jnp.zeros((n,), dtype=jnp.result_type(values, w))
    return out.at[cols].add(jnp.asarray(values) * w[rows])
