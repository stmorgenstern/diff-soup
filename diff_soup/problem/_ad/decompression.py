"""Decompression of a coloring's compressed products: direct reads and substitution by levels.

Every table indexes the stacked products ``concat(forward.ravel(), reverse.ravel(), [0])``.
An entry recovered directly is one gather (SparseMatrixColorings.jl, SMC, ``decompress!``;
SMC paper, Appendix A.3.1). An acyclic coloring recovers its off-diagonal entries by
substitution over two-colored trees (SMC paper, Appendix A.3.2). SMC solves one tree at a
time, leaf to root; here all trees are solved together, one level of leaves per step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Inexact, Int

if TYPE_CHECKING:
    from diff_soup.problem._ad.coloring import Coloring


class Sweep(NamedTuple):
    """Consecutive substitution levels padded to one width, run as one ``lax.scan``.

    Attributes:
        src: Per edge, where the stacked products hold its leaf's row in the column of its
            parent's color.
        leaf: Per edge, the tree vertex it leaves.
        parent: Per edge, the tree vertex it reaches.
        entry: Per edge, the positions of ``(leaf, parent)`` and ``(parent, leaf)`` among the
            entries.
    """

    src: Int[np.ndarray, "levels width"]
    leaf: Int[np.ndarray, "levels width"]
    parent: Int[np.ndarray, "levels width"]
    entry: Int[np.ndarray, "2 levels width"]


class Table(NamedTuple):
    """How a coloring reads the pattern's entries from its compressed products.

    Attributes:
        direct: Per entry, its index in the stacked products; a substituted entry reads the
            trailing zero.
        sweeps: The substitution, level by level; empty unless the coloring is acyclic.
        n_nodes: The tree vertices the substitution holds, one per tree a vertex is in.
    """

    direct: Int[np.ndarray, " nnz"]
    sweeps: tuple[Sweep, ...]
    n_nodes: int


def decompress(
    coloring: Coloring,
    forward: Inexact[ArrayLike, "n_forward n_rows"],
    reverse: Inexact[ArrayLike, "n_reverse n_cols"],
) -> Inexact[Array, " nnz"]:
    """Return the values of the pattern's entries, in the order the coloring was given them.

    Args:
        coloring: A [Coloring][diff_soup.problem._ad.coloring.Coloring] of the pattern.
        forward: ``forward[c, i]``, the sum of ``J[i, j]`` over the columns ``j`` of color ``c``.
        reverse: ``reverse[c, j]``, the sum of ``J[i, j]`` over the rows ``i`` of color ``c``.

    Raises:
        ValueError: If a product's shape is not the coloring's.
    """
    n_rows, n_cols = coloring.shape
    if np.shape(forward) != (coloring.n_forward, n_rows) or np.shape(reverse) != (
        coloring.n_reverse,
        n_cols,
    ):
        raise ValueError(
            f"products of shapes {np.shape(forward)} and {np.shape(reverse)} for a coloring "
            f"with {coloring.n_forward} forward and {coloring.n_reverse} reverse seeds "
            f"of a {coloring.shape} pattern"
        )
    dtype = jnp.result_type(forward, reverse)
    stacked = jnp.concatenate(
        [jnp.ravel(forward).astype(dtype), jnp.ravel(reverse).astype(dtype), jnp.zeros(1, dtype)]
    )
    table = coloring.table
    values = stacked[table.direct]
    if not table.sweeps:
        return values

    def level(buffer, edges):
        src, leaf, parent = edges
        value = stacked[src] - buffer[leaf]
        return buffer.at[parent].add(value), value

    buffer = jnp.zeros(table.n_nodes + 1, dtype)
    for sweep in table.sweeps:
        buffer, solved = jax.lax.scan(level, buffer, (sweep.src, sweep.leaf, sweep.parent))
        updates = jnp.broadcast_to(solved, sweep.entry.shape)
        values = values.at[sweep.entry].set(updates, mode="drop")
    return values


def colored_values(coloring: Coloring, fn, primal) -> Array:
    """Return a pattern's values from the tangents and pullbacks one coloring seeds.

    Args:
        coloring: The pattern's coloring.
        fn: The map whose Jacobian has the pattern, of one array.
        primal: Where to take the Jacobian.

    Returns:
        The values of the pattern's entries, in the coloring's entry order.
    """
    n_rows, n_cols = coloring.shape
    forward = jnp.zeros((0, n_rows), dtype=primal.dtype)
    reverse = jnp.zeros((0, n_cols), dtype=primal.dtype)
    if coloring.n_forward:
        colors = jnp.asarray(coloring.column_colors)
        tangents = (jnp.arange(coloring.n_forward)[:, None] == colors[None, :]).astype(primal.dtype)
        forward = jax.vmap(lambda v: jax.jvp(fn, (primal,), (v,))[1])(tangents)
    if coloring.n_reverse:
        residual, pull = jax.vjp(fn, primal)
        colors = jnp.asarray(coloring.row_colors)
        # A cotangent must match the residual's dtype, which may differ from the state's.
        dtype = jnp.asarray(residual).dtype
        seeds = (jnp.arange(coloring.n_reverse)[:, None] == colors[None, :]).astype(dtype)
        reverse = jax.vmap(lambda w: pull(w)[0])(seeds)
    return decompress(coloring, forward, reverse)


def substitution_sweeps(
    a: Int[np.ndarray, " edges"],
    b: Int[np.ndarray, " edges"],
    node_a: Int[np.ndarray, " edges"],
    node_b: Int[np.ndarray, " edges"],
    n_nodes: int,
    leaf_a: Bool[np.ndarray, " edges"],
    color: Int[np.ndarray, " n"],
    entry: Int[np.ndarray, "2 edges"],
    n_entries: int,
    zero: int,
) -> tuple[Sweep, ...]:
    """Return the substitution over a forest of two-colored trees, one level of leaves per step.

    An edge is solved once every other edge at its leaf is: its value is the leaf's row in
    the column of its parent's color, less the edges solved into the leaf. This is SMC
    ``decompress!`` for a ``TreeSetColoringResult`` (SMC paper, Appendix A.3.2), with the
    reverse breadth-first order of each tree (SMC's ``TreeSet``) replaced by rounds of leaf
    pruning over all trees at once. Consecutive levels padded to the same power of two form one sweep, so padding at
    most doubles a sweep.

    Args:
        a: Each edge's first vertex.
        b: Each edge's second vertex.
        node_a: The tree vertex of ``a`` in the edge's tree.
        node_b: The tree vertex of ``b`` in the edge's tree.
        n_nodes: The number of tree vertices.
        leaf_a: Whether ``a`` is the leaf when both ends of an edge are leaves.
        color: Each vertex's color.
        entry: Per edge, the positions of ``(a, b)`` and ``(b, a)`` among the entries.
        n_entries: The number of entries.
        zero: The index of the trailing zero in the stacked products.
    """
    n = color.size
    ends = np.concatenate((node_a, node_b))
    edge_ids = np.arange(a.size, dtype=np.int64)
    degree = np.bincount(ends, minlength=n_nodes)
    # A leaf's one remaining edge is the sum of the edges still at it.
    remaining = np.bincount(ends, weights=np.tile(edge_ids, 2), minlength=n_nodes).astype(np.int64)
    leaves = np.flatnonzero(degree == 1)
    levels = []
    while leaves.size:
        edges = np.unique(remaining[leaves])
        at_a, at_b = degree[node_a[edges]] == 1, degree[node_b[edges]] == 1
        is_a = np.where(at_a & at_b, leaf_a[edges], at_a)
        leaf, parent = np.where(is_a, a[edges], b[edges]), np.where(is_a, b[edges], a[edges])
        levels.append(
            (
                color[parent] * n + leaf,
                np.where(is_a, node_a[edges], node_b[edges]),
                np.where(is_a, node_b[edges], node_a[edges]),
                np.where(is_a, entry[:, edges], entry[::-1, edges]),
            )
        )
        touched = np.concatenate((node_a[edges], node_b[edges]))
        np.subtract.at(degree, touched, 1)
        np.subtract.at(remaining, touched, np.tile(edges, 2))
        touched = np.unique(touched)
        leaves = touched[degree[touched] == 1]
    widths = [1 << (len(level[0]) - 1).bit_length() for level in levels]
    sweeps = []
    start = 0
    for stop in range(1, len(levels) + 1):
        if stop < len(levels) and widths[stop] == widths[start]:
            continue
        shape = (stop - start, widths[start])
        src, leaf, parent = (
            np.full(shape, fill, dtype=np.int64) for fill in (zero, n_nodes, n_nodes)
        )
        pos = np.full((2, *shape), n_entries, dtype=np.int64)
        for row, level in enumerate(levels[start:stop]):
            k = level[0].size
            src[row, :k], leaf[row, :k], parent[row, :k], pos[:, row, :k] = level
        sweeps.append(Sweep(src, leaf, parent, pos))
        start = stop
    return tuple(sweeps)
