"""Coordinate patterns: the distinct ``(row, col)`` entries of a sparse matrix, and their colorings."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
from jaxtyping import Int

from diff_soup._jax_compat import eager_constant_folding, internal_core
from diff_soup.linear import SparseTopology
from diff_soup.problem._cache import BoundedContentCache


class UniqueCOO(NamedTuple):
    """The distinct entries of a coordinate pattern.

    Attributes:
        rows: Row of each distinct entry.
        cols: Column of each distinct entry.
        inverse: For each input entry, the position of its pair in ``rows`` and ``cols``.
    """

    rows: Int[np.ndarray, " unique"]
    cols: Int[np.ndarray, " unique"]
    inverse: Int[np.ndarray, " entries"]


def dedup_coo(rows, cols, n_cols: int, *, first_seen: bool = False) -> UniqueCOO:
    """Return the distinct ``(row, col)`` pairs of a pattern.

    Args:
        rows: Row indices.
        cols: Column indices, each below ``n_cols``.
        n_cols: The number of columns.
        first_seen: Keep each pair where it first appears instead of in row-major order.

    Returns:
        A [UniqueCOO][diff_soup.problem._ad.pattern.UniqueCOO] with ``int32`` rows and
        columns.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    keys = rows * np.int64(n_cols) + cols
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if first_seen:
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        unique, inverse = unique[order], rank[inverse]
    return UniqueCOO(
        (unique // np.int64(n_cols)).astype(np.int32),
        (unique % np.int64(n_cols)).astype(np.int32),
        inverse.reshape(-1).astype(np.int32),
    )


# Colorings keyed on their pattern's content and the kind and order asked: the plan
# colors three patterns per problem, each in up to a few kinds, and a build session
# holds a few dozen problems; ours.
_COLORING_CAPACITY = 256
_COLORINGS = BoundedContentCache(capacity=_COLORING_CAPACITY)


class Pattern:
    """The distinct entries of a sparse matrix, in row-major order.

    Equal content gives equal patterns and one cached coloring per kind and order.
    Build with [of][diff_soup.problem._ad.pattern.Pattern.of].

    Attributes:
        rows: Row of each entry, ``int32``, read-only.
        cols: Column of each entry, ``int32``, read-only.
        shape: ``(n_rows, n_cols)``.
    """

    __slots__ = ("rows", "cols", "shape", "_key", "_hash")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]):
        """Wrap entries that are already distinct and row-major; see ``of``."""
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        rows.flags.writeable = False
        cols.flags.writeable = False
        self.rows, self.cols = rows, cols
        self.shape = (int(shape[0]), int(shape[1]))
        self._key = (self.shape, rows.tobytes(), cols.tobytes())
        self._hash = hash(self._key)

    @classmethod
    def of(cls, rows, cols, shape) -> Pattern:
        """Return the pattern of the entries ``(rows[k], cols[k])``, repeats merged.

        Args:
            rows: Row indices.
            cols: Column indices.
            shape: ``(n_rows, n_cols)``, or one ``n`` for a square matrix.
        """
        shape = (int(shape), int(shape)) if np.ndim(shape) == 0 else tuple(shape)
        unique = dedup_coo(rows, cols, shape[1])
        return cls(unique.rows, unique.cols, shape)

    @classmethod
    def of_topology(cls, topology: SparseTopology) -> Pattern:
        """Return the pattern of a square [SparseTopology][diff_soup.linear.SparseTopology]."""
        return cls.of(topology.rows, topology.cols, int(topology.n_dofs))

    def topology(self) -> SparseTopology:
        """Return this square pattern as a [SparseTopology][diff_soup.linear.SparseTopology]."""
        return SparseTopology(rows=self.rows, cols=self.cols, n_dofs=self.shape[0])

    @property
    def nnz(self) -> int:
        """The number of entries."""
        return int(self.rows.size)

    def __hash__(self) -> int:
        """Hash by content."""
        return self._hash

    def __eq__(self, other) -> bool:
        """Equal shape and entries."""
        return isinstance(other, Pattern) and self._key == other._key

    def __repr__(self) -> str:
        """Shape and entry count."""
        return f"Pattern(shape={self.shape}, nnz={self.nnz})"

    def row_degrees(self) -> Int[np.ndarray, " n_rows"]:
        """Entries per row."""
        return np.bincount(self.rows, minlength=self.shape[0]).astype(np.int64)

    def union(self, *others: Pattern) -> Pattern:
        """Return the entries of this pattern or any of ``others``, which share its shape."""
        if any(other.shape != self.shape for other in others):
            raise ValueError(
                f"union of patterns of shapes {[self.shape, *(o.shape for o in others)]}"
            )
        rows = np.concatenate([self.rows, *(other.rows for other in others)])
        cols = np.concatenate([self.cols, *(other.cols for other in others)])
        return Pattern.of(rows, cols, self.shape)

    def stamp(self, offsets, shape) -> Pattern:
        """Return copies of this block placed at each ``(row, col)`` offset of a larger matrix.

        Args:
            offsets: ``(copies, 2)`` integer offsets, the block's top-left corner per copy.
            shape: The larger matrix's ``(n_rows, n_cols)``.
        """
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        rows = (offsets[:, :1] + self.rows[None, :]).reshape(-1)
        cols = (offsets[:, 1:] + self.cols[None, :]).reshape(-1)
        return Pattern.of(rows, cols, shape)

    def restrict(self, rows) -> Pattern:
        """Return the entries in ``rows``, at their positions in the same shape."""
        keep = np.isin(self.rows, np.asarray(rows, dtype=np.int64))
        return Pattern(self.rows[keep], self.cols[keep], self.shape)

    def positions(self, other: Pattern) -> Int[np.ndarray, " other_nnz"]:
        """Return where each entry of ``other``, whose entries this pattern holds, sits in it."""
        keys = self.rows.astype(np.int64) * self.shape[1] + self.cols
        wanted = other.rows.astype(np.int64) * self.shape[1] + other.cols
        where = np.searchsorted(keys, wanted)
        if wanted.size and (where.max() >= keys.size or np.any(keys[where] != wanted)):
            raise ValueError("positions asked of entries this pattern does not hold")
        return where.astype(np.int32)

    def coloring(self, kind: str = "column", order: str = "largest_first"):
        """Return this pattern's [Coloring][diff_soup.problem._ad.coloring.Coloring], cached by content.

        Args:
            kind: One of [KINDS][diff_soup.problem._ad.coloring.KINDS].
            order: One of [ORDERS][diff_soup.problem._ad.coloring.ORDERS].
        """
        from diff_soup.problem._ad.coloring import color

        return _COLORINGS.get_or_build(
            (kind, order, self._key),
            None,
            lambda: color(self.rows, self.cols, self.shape, kind=kind, order=order),
        )


_T = TypeVar("_T")


def outside_compile_time_eval(fn: Callable[[], _T]) -> _T:
    """Return ``fn()`` run outside the caller's trace and its compile-time evaluation.

    A residual traced for its pattern then stages its constant subexpressions instead of
    compiling each one, although the solver view builds its pattern inside
    ``jax.ensure_compile_time_eval``, and the known part it compiles runs eagerly even when
    the caller is being traced. Switching compile-time evaluation off alone would stage that
    compilation into the caller's trace.
    """
    with internal_core.eval_context(), eager_constant_folding(False):
        return fn()
