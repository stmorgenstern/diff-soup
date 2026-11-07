"""Gates on the tangents of an implicit derivative whose root or sensitivity solve failed.

An implicit derivative is linear in its tangents. When its root or its sensitivity solve is
invalid, its output tangents are poisoned with NaN, so a wrong number is never returned, and
its input tangents are zeroed by selection rather than by product, so a leaf that is not kept
becomes exactly zero even when it is infinite or NaN, and so does its cotangent under
transposition. Integer (``float0``) leaves and ``None`` pass unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import PyTree


def poison(tree: PyTree, valid: ArrayLike) -> PyTree:
    """Return ``tree`` with NaN in every inexact leaf where ``valid`` is false.

    Args:
        tree: The tangents to gate.
        valid: Whether the solve the tangents come from is valid.
    """
    valid = jnp.asarray(valid)

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(valid, leaf, jnp.full_like(leaf, jnp.nan))

    return jax.tree.map(gate, tree)


def zero_where_not(keep: ArrayLike, tree: PyTree) -> PyTree:
    """Return ``tree`` with exact zeros in every inexact leaf where ``keep`` is false.

    Args:
        keep: Whether to keep the leaves.
        tree: The tangents to gate.
    """
    keep = jnp.asarray(keep)

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(keep, leaf, jnp.zeros_like(leaf))

    return jax.tree.map(gate, tree)
