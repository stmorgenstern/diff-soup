"""Static pytree/flat-vector codec for native solver parameter boundaries.

A solver problem keeps ``theta`` as its authored pytree. Native libraries need
one contiguous real-valued buffer instead. [ThetaCodec][diff_soup.problem.implicit.theta_codec.ThetaCodec] records the
tree, leaf shapes, and leaf dtypes once, then performs a differentiable
ravel/unravel without changing the public parameter contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["ThetaCodec", "ThetaLeafSpec"]


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _path_entry_signature(entry: Any) -> tuple[str, Any]:
    """Return a stable, hashable description of one JAX key-path entry."""
    for attribute in ("name", "key", "idx"):
        if hasattr(entry, attribute):
            value = getattr(entry, attribute)
            try:
                hash(value)
            except TypeError:
                value = repr(value)
            return (_type_name(entry), value)
    return (_type_name(entry), repr(entry))


def pytree_structure_signature(tree: Any) -> tuple:
    """Value-independent leaf paths, shapes, and dtypes of ``tree``.

    JAX ``PyTreeDef`` auxiliary data can contain static callable identities for
    Equinox modules. Those identities are correct for codec validation but too
    strict for cross-problem structural equivalence. This signature records
    only the public leaf paths and array structure.
    """
    leaves = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        path_signature = tuple(_path_entry_signature(entry) for entry in path)
        try:
            array = jnp.asarray(leaf)
        except (TypeError, ValueError):
            leaves.append((path_signature, "opaque", _type_name(leaf)))
            continue
        leaves.append(
            (
                path_signature,
                "array",
                tuple(int(size) for size in array.shape),
                np.dtype(array.dtype).str,
            )
        )
    return (_type_name(tree), tuple(leaves))


@dataclass(frozen=True)
class ThetaLeafSpec:
    """One flat-buffer segment and its authored array representation."""

    shape: tuple[int, ...]
    dtype: np.dtype
    size: int


@dataclass(frozen=True)
class ThetaCodec:
    """A differentiable codec between an authored theta pytree and one vector.

    The codec accepts real floating leaves only. Integer, boolean, complex,
    and opaque leaves cannot ride a real-valued parameter vector.
    """

    _treedef: Any
    leaf_specs: tuple[ThetaLeafSpec, ...]
    flat_dtype: np.dtype
    size: int
    tree_signature: tuple

    @classmethod
    def from_template(
        cls,
        template: Any,
        *,
        flat_dtype: Optional[Any] = None,
    ) -> "ThetaCodec":
        """Build a codec from one representative public theta pytree."""
        leaves, treedef = jax.tree_util.tree_flatten(template)
        specs = []
        for index, leaf in enumerate(leaves):
            try:
                array = jnp.asarray(leaf)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"theta leaf {index} has type {_type_name(leaf)}; native theta "
                    "leaves must be real floating arrays."
                ) from exc
            dtype = np.dtype(array.dtype)
            if not np.issubdtype(dtype, np.floating):
                raise TypeError(
                    f"theta leaf {index} has dtype {dtype}; native theta leaves "
                    "must be real floating arrays."
                )
            shape = tuple(int(size) for size in array.shape)
            specs.append(ThetaLeafSpec(shape=shape, dtype=dtype, size=int(array.size)))

        if flat_dtype is None:
            resolved_dtype = (
                np.dtype(jnp.result_type(*(spec.dtype for spec in specs)))
                if specs
                else np.dtype(jnp.float64)
            )
        else:
            resolved_dtype = np.dtype(flat_dtype)
        if not np.issubdtype(resolved_dtype, np.floating):
            raise TypeError(
                f"theta flat_dtype must be a real floating dtype; got {resolved_dtype}."
            )
        return cls(
            _treedef=treedef,
            leaf_specs=tuple(specs),
            flat_dtype=resolved_dtype,
            size=sum(spec.size for spec in specs),
            tree_signature=pytree_structure_signature(template),
        )

    def ravel(self, theta: Any) -> jax.Array:
        """Validate and concatenate ``theta`` without changing its public tree."""
        leaves, treedef = jax.tree_util.tree_flatten(theta)
        if treedef != self._treedef:
            raise ValueError(
                "theta pytree structure does not match the template used to build "
                "this native parameter codec."
            )
        pieces = []
        for index, (leaf, spec) in enumerate(zip(leaves, self.leaf_specs)):
            array = jnp.asarray(leaf)
            shape = tuple(int(size) for size in array.shape)
            if shape != spec.shape:
                raise ValueError(f"theta leaf {index} has shape {shape}, expected {spec.shape}.")
            dtype = np.dtype(array.dtype)
            if dtype != spec.dtype:
                raise TypeError(f"theta leaf {index} has dtype {dtype}, expected {spec.dtype}.")
            pieces.append(jnp.reshape(array, (-1,)).astype(self.flat_dtype))
        if not pieces:
            return jnp.zeros((0,), dtype=self.flat_dtype)
        return jnp.concatenate(pieces)

    def unravel(self, flat: Any) -> Any:
        """Rebuild the authored theta pytree from one native parameter vector."""
        vector = jnp.asarray(flat)
        if vector.ndim != 1:
            raise ValueError(
                f"native theta buffer must be one-dimensional; got shape {vector.shape}."
            )
        if int(vector.size) != self.size:
            raise ValueError(
                f"native theta buffer has length {int(vector.size)}, expected {self.size}."
            )
        leaves = []
        offset = 0
        for spec in self.leaf_specs:
            segment = vector[offset : offset + spec.size]
            leaves.append(jnp.reshape(segment, spec.shape).astype(spec.dtype))
            offset += spec.size
        return jax.tree_util.tree_unflatten(self._treedef, leaves)
