"""Pointwise tensor primitives.

Domain-agnostic ``jnp`` passthroughs registered under ``q.ops``.
Vector/tensor inputs use the trailing dim as the geometric (component) dim.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def trace(T: Float[Array, "... d d"]) -> Float[Array, "..."]:
    """Trace (sum of diagonals) of a square tensor along trailing axes."""
    return jnp.trace(T, axis1=-2, axis2=-1)


def det(T: Float[Array, "... d d"]) -> Float[Array, "..."]:
    """Determinant of a square tensor along trailing axes."""
    return jnp.linalg.det(T)


def inv(T: Float[Array, "... d d"]) -> Float[Array, "... d d"]:
    """Matrix inverse along trailing axes."""
    return jnp.linalg.inv(T)


def transpose(T: Float[Array, "... a b"]) -> Float[Array, "... b a"]:
    """Swap last two axes (matrix transpose along trailing axes)."""
    return jnp.swapaxes(T, -1, -2)


def sym(T: Float[Array, "... d d"]) -> Float[Array, "... d d"]:
    """Symmetric part: 0.5 * (T + Tᵀ)."""
    return 0.5 * (T + transpose(T))


def skew(T: Float[Array, "... d d"]) -> Float[Array, "... d d"]:
    """Skew-symmetric part: 0.5 * (T - Tᵀ)."""
    return 0.5 * (T - transpose(T))


def outer(u: Float[Array, "... d"], v: Float[Array, "... d"]) -> Float[Array, "... d d"]:
    """Outer product u ⊗ v along trailing axes."""
    return u[..., :, None] * v[..., None, :]


def tensor_contract(A, B, axes: int = 1):
    """Tensor contraction (``jnp.tensordot`` passthrough)."""
    return jnp.tensordot(A, B, axes=axes)


def double_dot(A, B):
    """Batched tensor double contraction ``A : B`` over the last two axes.

    Contracts the last two axes of ``A`` with the last two axes of ``B``,
    summing them out; leading (batch) axes broadcast.  Covers the two
    constitutive cases:

    * rank-2 ``:`` rank-2 -> scalar   ``A[...,i,j] B[...,i,j] -> [...]``
      (e.g. ``sigma : epsilon``, ``sigma0 : epsilon``).
    * rank-4 ``:`` rank-2 -> rank-2   ``C[...,a,b,i,j] B[...,i,j] -> [...,a,b]``
      (e.g. the stiffness contraction ``C : epsilon``).

    Dispatch is by the trailing shape of ``A``: a ``(d,d,d,d)`` tail is the
    rank-4 operator case, anything else is the scalar case.  Either operand
    may be an un-batched constant (its empty batch broadcasts).
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    d = B.shape[-1]
    if A.ndim >= 4 and A.shape[-4:] == (d, d, d, d):
        return jnp.einsum("...abij,...ij->...ab", A, B)
    return jnp.einsum("...ij,...ij->...", A, B)


def scale(scalar_field, tensor):
    """Per-point product of a scalar field and a tensor, aligning batch axes.

    ``scalar_field`` has shape ``(*batch)``; ``tensor`` has shape
    ``(*batch_or_empty, *component)`` where the component axes are the
    trailing axes beyond the scalar's batch rank.  The scalar broadcasts
    over the component axes — no manual ``[..., None, None]`` padding.

    Backs ``QuadField.__mul__`` / ``__rmul__`` against a tensor operand.
    """
    s = jnp.asarray(scalar_field)
    T = jnp.asarray(tensor)
    n_component = T.ndim - s.ndim if T.ndim >= s.ndim else 0
    # If T carries no batch (e.g. a (d,d) constant) treat ALL of T as component.
    if T.ndim < s.ndim or T.shape[: s.ndim] != s.shape:
        n_component = T.ndim
    s_b = s.reshape(s.shape + (1,) * n_component)
    return s_b * T


__all__ = [
    "trace",
    "det",
    "inv",
    "transpose",
    "sym",
    "skew",
    "outer",
    "tensor_contract",
    "double_dot",
    "scale",
]
