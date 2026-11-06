"""sparse_jac.py — Sparse Jacobian via graph coloring + batched JVP.

Implements the Curtis-Powell-Reid (CPR) approach: color the columns of the
Jacobian sparsity pattern so that no two same-colored columns share a nonzero
row.  All colors are evaluated in parallel via ``vmap(jvp)``.

For FEM problems, element-level ``jacfwd + scatter`` (see
``diff_soup.domains.mesh.fem.assembly``) is typically faster because
n_local_dofs < n_colors.  This CPR approach is a general-purpose fallback
for problems where element-level decomposition is not available.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.problem._cache import BoundedContentCache
from diff_soup.utils.sparse import (
    build_padded_color_arrays,
    build_sparse_jvp_groups,
    sparse_jvp_jacobian_values,
)

# Colorings keyed on their pattern's content ``(cls, n_dofs, nnz, rows, cols)``:
# one per requested entry order, and one row-major core per COO multiset that
# every order of that multiset shares.
_COLORING_CACHE_CAPACITY = 256
_COLORING_CACHE = BoundedContentCache(capacity=_COLORING_CACHE_CAPACITY)
_COLORING_CORE_CACHE = BoundedContentCache(capacity=_COLORING_CACHE_CAPACITY)


@dataclass(frozen=True, eq=False)
class SparseJacobian:
    """Precomputed coloring for efficient sparse Jacobian evaluation.

    Create via [from_coo][] or [from_topology][], then call
    [evaluate][] with a residual function and state vector.

    [evaluate][] uses ``vmap(jvp)`` to evaluate all colors in parallel.

    Attributes:
        rows, cols: COO indices of the sparsity pattern.
        n_dofs: Number of columns (= rows for square Jacobians).
        n_colors: Number of JVP evaluations needed.
        nnz: Number of nonzero entries.
    """

    rows: np.ndarray
    cols: np.ndarray
    n_dofs: int
    nnz: int
    n_colors: int
    # Padded color arrays for vmap
    _col_idx: jnp.ndarray = field(repr=False)
    _col_mask: jnp.ndarray = field(repr=False)
    _entry_idx: jnp.ndarray = field(repr=False)
    _row_idx: jnp.ndarray = field(repr=False)
    _entry_mask: jnp.ndarray = field(repr=False)

    def __hash__(self) -> int:
        """Hash by shape, color count, pattern and seeded columns."""
        return hash(
            (
                self.n_dofs,
                self.nnz,
                self.n_colors,
                self.rows.tobytes(),
                self.cols.tobytes(),
                np.asarray(self._col_mask).tobytes(),
            )
        )

    def __eq__(self, other) -> bool:
        """Equal shape, color count, pattern and seeded columns."""
        if not isinstance(other, SparseJacobian):
            return NotImplemented
        return (
            self.n_dofs == other.n_dofs
            and self.nnz == other.nnz
            and self.n_colors == other.n_colors
            and np.array_equal(self.rows, other.rows)
            and np.array_equal(self.cols, other.cols)
            and np.array_equal(np.asarray(self._col_mask), np.asarray(other._col_mask))
        )

    def column_colors(self) -> np.ndarray:
        """Return each column's color, or -1 for a column no color seeds."""
        colors = np.full(self.n_dofs, -1, dtype=np.int64)
        col_idx, col_mask = np.asarray(self._col_idx), np.asarray(self._col_mask)
        for color in range(col_idx.shape[0]):
            colors[col_idx[color][col_mask[color]]] = color
        return colors

    def seeding_only(self, cols) -> SparseJacobian:
        """Return this coloring with only the columns in ``cols`` seeded.

        A column outside ``cols`` then adds nothing to the tangent of its color.
        """
        col_idx, col_mask = np.asarray(self._col_idx), np.asarray(self._col_mask)
        seeded = col_mask & np.isin(col_idx, np.asarray(cols))
        return dataclasses.replace(self, _col_mask=jnp.asarray(seeded))

    @classmethod
    def from_coo(
        cls,
        rows: np.ndarray,
        cols: np.ndarray,
        n_dofs: int,
    ) -> SparseJacobian:
        """Build from COO sparsity pattern.

        Args:
            rows: Row indices of nonzero entries (``int32`` array-like).
            cols: Column indices of nonzero entries (``int32`` array-like).
            n_dofs: Number of DOFs (columns).
        """
        rows = np.ascontiguousarray(rows, dtype=np.int32).reshape(-1)
        cols = np.ascontiguousarray(cols, dtype=np.int32).reshape(-1)
        n_dofs = int(n_dofs)
        if rows.size != cols.size:
            raise ValueError(
                f"rows and cols must have equal length; got {rows.size} and {cols.size}."
            )
        key = (cls, n_dofs, int(rows.size), rows.tobytes(), cols.tobytes())
        return _COLORING_CACHE.get_or_build(
            key, None, lambda: cls._build_ordered(rows, cols, n_dofs)
        )

    @classmethod
    def _build_ordered(cls, rows: np.ndarray, cols: np.ndarray, n_dofs: int) -> SparseJacobian:
        """Build the coloring for one entry order from its multiset's shared core.

        Coloring depends on the COO multiset, not its presentation order. One
        row-major core is cached per multiset, and only its padded
        output-entry indices are remapped for callers whose value vectors use
        a different order. This shares the graph-coloring computation between
        the IR provider's stencil order and the linearization plan's sorted
        order without changing either caller's contract.
        """
        order = np.lexsort((cols, rows))
        sorted_rows = np.ascontiguousarray(rows[order], dtype=np.int32)
        sorted_cols = np.ascontiguousarray(cols[order], dtype=np.int32)
        core_key = (cls, n_dofs, int(rows.size), sorted_rows.tobytes(), sorted_cols.tobytes())
        core = _COLORING_CORE_CACHE.get_or_build(
            core_key, None, lambda: cls._build_core(sorted_rows, sorted_cols, n_dofs)
        )
        if rows.size == 0 or np.array_equal(order, np.arange(rows.size)):
            return core
        rows = rows.copy()
        cols = cols.copy()
        rows.flags.writeable = False
        cols.flags.writeable = False
        sorted_to_requested = jnp.asarray(order, dtype=jnp.int32)
        return cls(
            rows=rows,
            cols=cols,
            n_dofs=n_dofs,
            nnz=len(rows),
            n_colors=core.n_colors,
            _col_idx=core._col_idx,
            _col_mask=core._col_mask,
            _entry_idx=sorted_to_requested[core._entry_idx],
            _row_idx=core._row_idx,
            _entry_mask=core._entry_mask,
        )

    @classmethod
    def _build_core(
        cls, sorted_rows: np.ndarray, sorted_cols: np.ndarray, n_dofs: int
    ) -> SparseJacobian:
        """Color one row-major COO pattern."""
        sorted_rows.flags.writeable = False
        sorted_cols.flags.writeable = False
        col_groups, entry_groups, row_groups = build_sparse_jvp_groups(
            sorted_rows, sorted_cols, n_dofs
        )
        col_idx, col_mask, entry_idx, row_idx, entry_mask = build_padded_color_arrays(
            col_groups, entry_groups, row_groups
        )
        return cls(
            rows=sorted_rows,
            cols=sorted_cols,
            n_dofs=n_dofs,
            nnz=len(sorted_rows),
            n_colors=len(col_groups),
            _col_idx=col_idx,
            _col_mask=col_mask,
            _entry_idx=entry_idx,
            _row_idx=row_idx,
            _entry_mask=entry_mask,
        )

    @classmethod
    def from_topology(cls, topology) -> SparseJacobian:
        """Build from a ``SparseTopology`` object.

        Args:
            topology: Has ``.rows``, ``.cols``, ``.n_dofs`` attributes.
        """
        return cls.from_coo(
            np.asarray(topology.rows, dtype=np.int32),
            np.asarray(topology.cols, dtype=np.int32),
            int(topology.n_dofs),
        )

    def evaluate(
        self,
        residual_fn: Callable[[jax.Array], jax.Array],
        u: jax.Array,
    ) -> jax.Array:
        """Compute sparse Jacobian values via ``vmap(jvp)`` over all colors.

        All colors are evaluated simultaneously; uses O(n_colors * n_dofs)
        memory.

        Args:
            residual_fn: callable
                ``residual_fn(u) -> r`` where ``r`` has the same shape as ``u``
                (or compatible with the sparsity pattern rows).
            u: jax.Array
                State vector at which to evaluate the Jacobian.

        Returns:
            jax.Array, shape ``(nnz,)``
                Nonzero values of the Jacobian in the same order as ``rows, cols``.
        """
        return sparse_jvp_jacobian_values(
            residual_fn,
            u,
            self._col_idx,
            self._col_mask,
            self._entry_idx,
            self._row_idx,
            self._entry_mask,
            self.nnz,
        )
