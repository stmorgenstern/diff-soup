"""Pivoted LU of batches of small dense blocks, with forward and transposed solves.

The structured backends factor many small ``b x b`` blocks at once (the
diagonal blocks of a block-tridiagonal chain, the capacitance matrices of the
low-rank terms). Every function here takes arrays of shape ``(..., b, b)``
with any leading batch axes and is a fixed-shape ``jax.numpy`` program,
unrolled over the static block size ``b``, so it lowers to elementwise
kernels and batches under ``vmap``.

The factorization is ``P a = L U`` with scaled row partial pivoting: the pivot
of column ``k`` is the remaining row with the largest ``|a[i, k]| / max_j
|a[i, j]|``, so the choice does not depend on how the rows are scaled. ``L``
is unit lower triangular and stored below the diagonal of the packed ``lu``;
``U`` is stored on and above it. ``perm[..., i]`` is the source row of row
``i`` of ``P a``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

#: A pivot at most this many unit roundoffs of the largest magnitude in its
#: original row marks the block singular (the Dense backend uses the same
#: margin on its pivots).
SINGULAR_PIVOT_ROUNDOFFS = 16.0

__all__ = [
    "SINGULAR_PIVOT_ROUNDOFFS",
    "bmm",
    "small_lu_factor",
    "small_lu_solve",
    "small_lu_solve_transpose",
]


def bmm(a: Array, b: Array) -> Array:
    """Batched ``a @ b`` over the two trailing axes, written as a fusible contraction."""
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


def small_lu_factor(a: ArrayLike) -> tuple[Array, Array, Array]:
    """Pivoted LU of a batch of small dense matrices.

    Args:
        a: Array of shape ``(..., b, b)``.

    Returns:
        ``(lu, perm, singular)``: the packed factors ``(..., b, b)``, the row
        permutation ``(..., b)`` (``int32``) and a boolean ``(...)`` that holds
        where a pivot is zero, not finite, or at most
        ``SINGULAR_PIVOT_ROUNDOFFS`` unit roundoffs of the largest magnitude
        of its row before elimination.
    """
    a = jnp.asarray(a)
    b = int(a.shape[-1])
    rows = jnp.arange(b, dtype=jnp.int32)
    perm = jnp.broadcast_to(rows, a.shape[:-1]).astype(jnp.int32)
    row_scale = jnp.max(jnp.abs(a), axis=-1)
    row_scale = jnp.where(row_scale > 0, row_scale, jnp.ones((), row_scale.dtype))
    roundoff = SINGULAR_PIVOT_ROUNDOFFS * jnp.finfo(a.dtype).eps / 2
    singular = jnp.zeros(a.shape[:-2], dtype=bool)
    for k in range(b):
        if k < b - 1:
            candidates = jnp.abs(a[..., k:, k]) / row_scale[..., k:]
            p = (jnp.argmax(candidates, axis=-1) + k).astype(jnp.int32)
            onehot_k = (rows == k).astype(a.dtype)
            onehot_p = (rows == p[..., None]).astype(a.dtype)
            row_k = a[..., k, :]
            row_p = jnp.take_along_axis(a, p[..., None, None], axis=-2)[..., 0, :]
            a = a + (onehot_k - onehot_p)[..., :, None] * (row_p - row_k)[..., None, :]
            perm_k = perm[..., k]
            perm_p = jnp.take_along_axis(perm, p[..., None], axis=-1)[..., 0]
            perm = jnp.where(rows == k, perm_p[..., None], perm)
            perm = jnp.where(rows == p[..., None], perm_k[..., None], perm)
            scale_k = row_scale[..., k]
            scale_p = jnp.take_along_axis(row_scale, p[..., None], axis=-1)[..., 0]
            row_scale = jnp.where(rows == k, scale_p[..., None], row_scale)
            row_scale = jnp.where(rows == p[..., None], scale_k[..., None], row_scale)
        pivot = a[..., k, k]
        singular = (
            singular | ~jnp.isfinite(pivot) | (jnp.abs(pivot) <= roundoff * row_scale[..., k])
        )
        below = rows > k
        factor = jnp.where(below, a[..., :, k] / pivot[..., None], 0.0)
        update = factor[..., :, None] * a[..., k, None, :]
        a = a - jnp.where(below[:, None] & below[None, :], update, 0.0)
        a = jnp.where(below[:, None] & (rows == k)[None, :], factor[..., :, None], a)
    return a, perm, singular


def small_lu_solve(lu: Array, perm: Array, rhs: Array) -> Array:
    """Solve ``a x = rhs`` from a [small_lu_factor][] result.

    Args:
        lu: Packed factors, shape ``(..., b, b)``.
        perm: Row permutation, shape ``(..., b)``.
        rhs: Right-hand sides, shape ``(..., b, m)``.

    Returns:
        ``x`` with the shape of ``rhs``.
    """
    b = int(lu.shape[-1])
    y = jnp.take_along_axis(rhs, perm[..., :, None], axis=-2)
    rows = [y[..., i, :] for i in range(b)]
    for i in range(b):
        acc = rows[i]
        for j in range(i):
            acc = acc - lu[..., i, j, None] * rows[j]
        rows[i] = acc
    for i in range(b - 1, -1, -1):
        acc = rows[i]
        for j in range(i + 1, b):
            acc = acc - lu[..., i, j, None] * rows[j]
        rows[i] = acc / lu[..., i, i, None]
    return jnp.stack(rows, axis=-2)


def small_lu_solve_transpose(lu: Array, perm: Array, rhs: Array) -> Array:
    """Solve ``a^T x = rhs`` from the same [small_lu_factor][] result.

    With ``P a = L U``, ``a^T = U^T L^T P``: a forward substitution with
    ``U^T``, a back substitution with the unit ``L^T`` and the inverse row
    permutation.

    Args:
        lu: Packed factors, shape ``(..., b, b)``.
        perm: Row permutation, shape ``(..., b)``.
        rhs: Right-hand sides, shape ``(..., b, m)``.

    Returns:
        ``x`` with the shape of ``rhs``.
    """
    b = int(lu.shape[-1])
    rows = [rhs[..., i, :] for i in range(b)]
    for i in range(b):
        acc = rows[i]
        for j in range(i):
            acc = acc - lu[..., j, i, None] * rows[j]
        rows[i] = acc / lu[..., i, i, None]
    for i in range(b - 1, -1, -1):
        acc = rows[i]
        for j in range(i + 1, b):
            acc = acc - lu[..., j, i, None] * rows[j]
        rows[i] = acc
    z = jnp.stack(rows, axis=-2)
    # x[perm[i]] = z[i]: scatter through the one-hot permutation (b is small).
    onehot = (perm[..., :, None] == jnp.arange(b, dtype=perm.dtype)).astype(z.dtype)
    return jnp.sum(onehot[..., :, :, None] * z[..., :, None, :], axis=-3)
