"""Sparse-matrix colorings: which columns share a forward seed and which rows a reverse seed.

A coloring groups the columns of a sparse matrix ``J`` into forward seeds, its rows into
reverse seeds, or both, so that every entry of ``J`` can be read back from the compressed
products ``forward[c, i]`` (the sum of ``J[i, j]`` over the columns ``j`` of color ``c``)
and ``reverse[c, j]`` (the sum over the rows ``i`` of color ``c``). Colorings are built in
NumPy and SciPy once per pattern; [decompress][diff_soup.problem._ad.coloring.decompress]
is a JAX function of the products.

The greedy colorings, the vertex orders and the postprocessing are ports of
SparseMatrixColorings.jl 0.4.22 (SMC; Montoison, Dalle and Gebremedhin, "Revisiting
sparse matrix coloring and bicoloring", 2025, Appendix A), with SMC's iteration orders
and tie-breaking, so a kind and an order give SMC's colors. Each vertex's step is one set
of NumPy operations over its neighbors' neighbors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Int

from diff_soup.problem._ad.decompression import (
    Table,
    colored_values,
    decompress,
    substitution_sweeps,
)

__all__ = ["KINDS", "MAX_PAIRS", "ORDERS", "Coloring", "color", "colored_values", "decompress"]

ORDERS = ("natural", "largest_first", "smallest_last", "incidence_degree", "dynamic_largest_first")
KINDS = ("column", "row", "bidirectional", "star", "acyclic")
# The largest neighbor-of-neighbor product a column or row coloring forms, in pairs of
# entries sharing a row (column), repeats counted: a row of d entries adds d**2. The
# product holds at most one int32 index and one bool per pair, 1.3 GB at the bound, a sixth
# of an 8 GB build job; a row of 16,385 entries passes it alone; ours.
MAX_PAIRS = 2**28
# An adjacency graph in CSR: pointers, each vertex's neighbors ascending, each one's edge id.
Graph = tuple[Int[np.ndarray, " vertices_1"], Int[np.ndarray, " slots"], Int[np.ndarray, " slots"]]


@dataclass(frozen=True, eq=False)
class Coloring:
    """A coloring of a sparse pattern and the table that decompresses its products.

    Attributes:
        kind: One of [KINDS][diff_soup.problem._ad.coloring.KINDS].
        order: One of [ORDERS][diff_soup.problem._ad.coloring.ORDERS].
        shape: ``(n_rows, n_cols)``.
        column_colors: Each column's forward seed, ``-1`` when no forward seed needs it.
        row_colors: Each row's reverse seed, ``-1`` when no reverse seed needs it.
        n_forward: The number of forward seeds (JVPs).
        n_reverse: The number of reverse seeds (VJPs).
        table: How [decompress][diff_soup.problem._ad.coloring.decompress] reads each entry.
    """

    kind: str
    order: str
    shape: tuple[int, int]
    column_colors: Int[np.ndarray, " n_cols"]
    row_colors: Int[np.ndarray, " n_rows"]
    n_forward: int
    n_reverse: int
    table: Table


def color(
    rows: Int[np.ndarray, " nnz"],
    cols: Int[np.ndarray, " nnz"],
    shape: tuple[int, int],
    kind: str = "column",
    order: str = "largest_first",
) -> Coloring:
    """Color a sparse pattern for compressed evaluation.

    - ``column``: a partial distance-2 coloring of the columns, so no two columns of one
      color share a row (SMC ``ColoringProblem(:nonsymmetric, :column)``).
    - ``row``: the same for the rows (SMC ``:row``).
    - ``bidirectional``: a star bicoloring, each entry read from the forward or the reverse
      product (SMC ``:bidirectional`` with ``:direct`` decompression and postprocessing).
    - ``star``: a star coloring of a structurally symmetric pattern, ``J[i, j]`` read from
      ``forward[c(j), i]`` or ``forward[c(i), j]`` (SMC ``:symmetric`` with ``:direct``
      decompression and postprocessing). It reads ``J[j, i]`` for ``J[i, j]``, so it holds
      only for numerically symmetric matrices such as Hessians.
    - ``acyclic``: an acyclic coloring of a structurally symmetric pattern, decompressed by
      substitution (SMC ``:symmetric`` with ``:substitution`` decompression and
      postprocessing); it too holds only for numerically symmetric matrices.

    A column or row with no entry gets no seed. Colors are ``0, 1, ...`` with no gaps.

    Args:
        rows: Row of each distinct entry.
        cols: Column of each distinct entry, in any order; decompression follows it.
        shape: ``(n_rows, n_cols)``.
        kind: One of [KINDS][diff_soup.problem._ad.coloring.KINDS].
        order: One of [ORDERS][diff_soup.problem._ad.coloring.ORDERS], the order in which the
            greedy colors the vertices.

    Raises:
        ValueError: On an unknown kind or order, an entry outside ``shape`` or repeated, a
            ``star`` or ``acyclic`` pattern that is not structurally symmetric, or a
            ``column`` (``row``) pattern whose rows (columns) pair more than
            [MAX_PAIRS][diff_soup.problem._ad.coloring.MAX_PAIRS] entries.
    """
    if kind not in KINDS or order not in ORDERS:
        raise ValueError(f"kind {kind!r} or order {order!r} not among {KINDS} and {ORDERS}")
    m, n = int(shape[0]), int(shape[1])
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows.size and (rows.min() < 0 or rows.max() >= m or cols.min() < 0 or cols.max() >= n):
        raise ValueError(f"an entry lies outside the shape {(m, n)}")
    keys = rows * n + cols
    key_order = np.argsort(keys, kind="stable")
    if np.any(np.diff(keys[key_order]) == 0):
        raise ValueError("the pattern repeats an entry")
    has_col = np.bincount(cols, minlength=n) > 0
    has_row = np.bincount(rows, minlength=m) > 0
    col_colors, row_colors = np.full(n, -1, np.int64), np.full(m, -1, np.int64)
    sweeps, n_nodes = (), 0
    if kind == "column":
        col_colors = _partial_distance2(rows, cols, (m, n), order)
        direct = col_colors[cols] * m + rows
    elif kind == "row":
        row_colors = _partial_distance2(cols, rows, (n, m), order)
        direct = row_colors[rows] * n + cols
    elif kind == "bidirectional":
        csc = np.lexsort((rows, cols))
        a, b = cols[csc], n + rows[csc]
        graph = _adjacency(a, b, n + m)
        symmetric, hub = _star(graph, a, b, _order(order, graph), np.zeros(n + m, dtype=bool))
        col_colors = _first_seen(np.where(has_col, symmetric[:n], -1))
        row_colors = _first_seen(np.where(has_row, symmetric[n:], -1))
        from_column = np.empty(rows.size, dtype=bool)
        from_column[csc] = hub < n
        forward_at = col_colors[cols] * m + rows
        reverse_at = (col_colors.max(initial=-1) + 1) * m + row_colors[rows] * n + cols
        direct = np.where(from_column, forward_at, reverse_at)
    else:
        a, b, diagonal = _symmetric(rows, cols, (m, n))
        graph = _adjacency(a, b, n)
        vertices = _order(order, graph)
        on_diagonal = rows == cols
        if kind == "star":
            symmetric, hub = _star(graph, a, b, vertices, diagonal)
            col_colors = np.where(has_col, symmetric, -1)
            edge = np.searchsorted(a * n + b, np.minimum(rows, cols) * n + np.maximum(rows, cols))
            centre = np.where(on_diagonal, rows, np.append(hub, 0)[edge])
            direct = col_colors[centre] * n + rows + cols - centre
        else:
            symmetric, root = _acyclic(graph, a.size, vertices)
            symmetric, node_a, node_b, n_nodes, leaf_a = _trees(symmetric, root, a, b, diagonal)
            col_colors = np.where(has_col, symmetric, -1)
            zero = (int(col_colors.max(initial=-1)) + 1) * m
            direct = np.where(on_diagonal, col_colors[rows] * n + rows, zero)
            ordered_keys = keys[key_order]
            entry = key_order[np.searchsorted(ordered_keys, np.stack((a * n + b, b * n + a)))]
            sweeps = substitution_sweeps(
                a, b, node_a, node_b, n_nodes, leaf_a, col_colors, entry, rows.size, zero
            )
    n_forward = int(col_colors.max(initial=-1)) + 1
    n_reverse = int(row_colors.max(initial=-1)) + 1
    table = Table(direct.astype(np.int64), sweeps, n_nodes)
    return Coloring(kind, order, (m, n), col_colors, row_colors, n_forward, n_reverse, table)


def _grouped(
    major: Int[np.ndarray, " k"], minor: Int[np.ndarray, " k"], n_major: int
) -> tuple[Int[np.ndarray, " major_1"], Int[np.ndarray, " k"], Int[np.ndarray, " k"]]:
    """Group ``minor`` by ``major``, both ascending: CSR pointers, members, input positions."""
    order = np.lexsort((minor, major))
    ptr = np.zeros(n_major + 1, dtype=np.int64)
    np.cumsum(np.bincount(major, minlength=n_major), out=ptr[1:])
    return ptr, minor[order], order


def _slots(
    ptr: Int[np.ndarray, " rows_1"], items: Int[np.ndarray, " items"]
) -> tuple[Int[np.ndarray, " slots"], Int[np.ndarray, " slots"]]:
    """Return the CSR positions of rows ``items`` concatenated, and the item of each."""
    starts = ptr[items]
    lengths = ptr[items + 1] - starts
    item = np.repeat(np.arange(items.size), lengths)
    ends = np.cumsum(lengths)
    return np.arange(item.size) + (starts - ends + lengths)[item], item


def _adjacency(
    a: Int[np.ndarray, " edges"], b: Int[np.ndarray, " edges"], n_vertices: int
) -> Graph:
    """Return the CSR of an undirected graph: pointers, neighbors and each one's edge id.

    Neighbors ascend, as SMC's ``AdjacencyGraph`` stores them; edge ``e`` joins ``a[e]`` and
    ``b[e]``.
    """
    ids = np.arange(a.size, dtype=np.int64)
    ptr, neighbors, slot = _grouped(np.concatenate((a, b)), np.concatenate((b, a)), n_vertices)
    return ptr, neighbors, np.concatenate((ids, ids))[slot]


def _symmetric(
    rows: Int[np.ndarray, " nnz"], cols: Int[np.ndarray, " nnz"], shape: tuple[int, int]
) -> tuple[Int[np.ndarray, " edges"], Int[np.ndarray, " edges"], Bool[np.ndarray, " n"]]:
    """Return a structurally symmetric pattern's edges and which vertices have a diagonal.

    The edges ``a < b`` ascend as SMC's ``build_edge_to_index`` numbers them (lower
    triangle, column by column). SMC needs no diagonal entry.

    Raises:
        ValueError: If the pattern is not square with ``(i, j)`` present iff ``(j, i)`` is.
    """
    n = shape[1]
    off = rows != cols
    if shape[0] != n or not np.array_equal(
        np.sort(rows[off] * n + cols[off]), np.sort(cols[off] * n + rows[off])
    ):
        raise ValueError(f"a pattern of shape {shape} that is not structurally symmetric")
    lower = rows > cols
    a, b = cols[lower], rows[lower]
    edges = np.lexsort((b, a))
    diagonal = np.zeros(n, dtype=bool)
    diagonal[rows[~off]] = True
    return a[edges], b[edges], diagonal


def _order(
    order: str,
    graph: Graph | None,
    degree: Int[np.ndarray, " n"] | None = None,
    updates: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> Int[np.ndarray, " n"]:
    """Return the vertices in the order the greedy colors them.

    Ports SMC ``vertices(g, order)`` (``order.jl``; SMC paper, Appendix A.4): ``natural`` is
    ``0, 1, ...``;
    ``largest_first`` sorts by degree, largest first and ties by index (Julia's stable
    sort); the dynamic orders are SMC ``DynamicDegreeBasedOrder`` with its default
    buckets (``DegreeBucketsSMC``), after ColPack (Gebremedhin et al. 2013, Section 5):
    ``smallest_last`` takes the least back degree and fills from the end,
    ``incidence_degree`` the largest back degree, ``dynamic_largest_first`` the largest
    forward degree.

    Args:
        order: One of [ORDERS][diff_soup.problem._ad.coloring.ORDERS].
        graph: An adjacency CSR ``(ptr, neighbors, edge_ids)``, or ``None`` with ``degree``
            and ``updates`` given.
        degree: Each vertex's degree, when ``graph`` is ``None``.
        updates: ``updates(u, ordered)``, the unordered vertices whose dynamic degree moves
            when ``u`` is ordered, in SMC's visiting order.
    """
    if graph is not None:
        ptr, neighbors, _ = graph
        degree = np.diff(ptr)

        def updates(u, ordered):
            around = neighbors[ptr[u] : ptr[u + 1]]
            return around[~ordered[around]]

    n = degree.size
    if order == "natural":
        return np.arange(n)
    if order == "largest_first":
        return np.argsort(-degree, kind="stable")
    rising = order == "incidence_degree"
    low_to_high = order != "smallest_last"
    top = int(degree.max(initial=0))
    start = np.zeros(n, dtype=np.int64) if rising else degree.astype(np.int64)
    count = np.bincount(start, minlength=top + 1)
    high = (np.cumsum(count) - 1).tolist()
    low = (np.cumsum(count) - count).tolist()
    store = np.argsort(start, kind="stable")
    where = np.empty(n, dtype=np.int64)
    where[store] = np.arange(n)
    store, where, deg = store.tolist(), where.tolist(), start.tolist()
    ordered = np.zeros(n, dtype=bool)
    result = np.empty(n, dtype=np.int64)
    d = top if low_to_high else 0
    for index in range(n) if low_to_high else range(n - 1, -1, -1):
        if low_to_high:
            d = min(d + 1, top)
            while high[d] < low[d]:
                d -= 1
        else:
            d = max(d - 1, 0)
            while high[d] < low[d]:
                d += 1
        u = store[high[d]]
        high[d] -= 1
        deg[u] = -1
        ordered[u] = True
        result[index] = u
        for v in updates(u, ordered).tolist():
            dv, p = deg[v], where[v]
            if rising:
                w = store[high[dv]]
                store[p], where[w] = w, p
                high[dv] -= 1
                low[dv + 1] -= 1
                store[low[dv + 1]], where[v], deg[v] = v, low[dv + 1], dv + 1
            else:
                w = store[low[dv]]
                store[p], where[w] = w, p
                low[dv] += 1
                high[dv - 1] += 1
                store[high[dv - 1]], where[v], deg[v] = v, high[dv - 1], dv - 1
    return result


def _partial_distance2(
    rows: Int[np.ndarray, " nnz"], cols: Int[np.ndarray, " nnz"], shape: tuple[int, int], order: str
) -> Int[np.ndarray, " n"]:
    """Color the columns so that no two columns of one color share a row.

    Ports SMC ``partial_distance2_coloring`` (SMC paper, Appendix A.2.1; Gebremedhin, Manne
    and Pothen 2005, "What color is your Jacobian?", Algorithm 3.2): each column in turn takes the least color
    no column sharing a row with it holds. The columns sharing a row are one SciPy product
    ``AᵀA``; SMC's dynamic orders visit them row by row, first seen first.

    Raises:
        ValueError: If the rows pair more than [MAX_PAIRS][diff_soup.problem._ad.coloring.MAX_PAIRS]
            entries.
    """
    m, n = shape
    pairs = int(np.sum(np.bincount(rows, minlength=m).astype(np.int64) ** 2))
    if pairs > MAX_PAIRS:
        raise ValueError(
            f"coloring by {pairs} pairs of entries sharing a line, past MAX_PAIRS = {MAX_PAIRS}; "
            "hold the dense lines out of the coloring"
        )
    ones = np.ones(rows.size, dtype=bool)
    product = sp.csr_matrix((ones, (cols, rows)), shape=(n, m)) @ sp.csr_matrix(
        (ones, (rows, cols)), shape=(m, n)
    )
    ptr, near = product.indptr.astype(np.int64), product.indices
    has_col = np.bincount(cols, minlength=n) > 0
    col_ptr, col_rows, _ = _grouped(cols, rows, n)
    row_ptr, row_cols, _ = _grouped(rows, cols, m)

    def updates(u, ordered):
        seen = row_cols[_slots(row_ptr, col_rows[col_ptr[u] : col_ptr[u + 1]])[0]]
        seen = seen[np.sort(np.unique(seen, return_index=True)[1])]
        return seen[(seen != u) & ~ordered[seen]]

    colors = np.full(n, -1, dtype=np.int64)
    stamp = np.full(n + 1, -1, dtype=np.int64)
    bounds = ptr.tolist()
    for v in _order(order, None, np.diff(ptr) - has_col, updates).tolist():
        taken = colors[near[bounds[v] : bounds[v + 1]]]
        taken = taken[taken >= 0]
        stamp[taken] = v
        colors[v] = np.argmax(stamp[: taken.size + 1] != v)
    colors[~has_col] = -1
    used = np.zeros(n + 1, dtype=bool)
    used[colors[has_col]] = True
    return _relabel(colors, used)


def _star(
    graph: Graph,
    a: Int[np.ndarray, " edges"],
    b: Int[np.ndarray, " edges"],
    vertices: Int[np.ndarray, " n"],
    diagonal: Bool[np.ndarray, " n"],
) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " edges"]]:
    """Star-color an adjacency graph; return the colors and each edge's hub.

    Ports SMC ``star_coloring`` with ``postprocessing=true`` (SMC paper, Appendices A.2.2
    and A.2.5; Gebremedhin, Tarafdar, Manne and Pothen 2007, "New acyclic and star coloring
    algorithms with application to computing Hessians", Algorithm 4.1, which keeps the
    two-colored stars and their hubs). A neighbor ``w`` of ``v`` whose color repeats around
    ``v`` forbids every color around ``w`` (case 1); one whose color does not forbids the
    color of a hub next to it (case 2). The postprocessing makes neutral every color that
    no hub and no diagonal entry reads (SMC paper, Section 3). On ``[[0, Aᵀ], [A, 0]]``
    this is a star bicoloring (Coleman and Verma 1998; SMC paper, Section 2 and Appendix
    A.2.4): each entry is read from the side of its hub.

    Args:
        graph: Adjacency CSR ``(ptr, neighbors, edge_ids)``.
        a: Each edge's lower vertex, edges in SMC's order.
        b: Each edge's higher vertex.
        vertices: The coloring order.
        diagonal: Which vertices have a diagonal entry, whose color is always kept.
    """
    ptr, neighbors, edge_ids = graph
    colors = np.full(ptr.size - 1, -1, dtype=np.int64)
    star = np.full(a.size, -1, dtype=np.int64)
    hub = np.empty(a.size, dtype=np.int64)
    n_stars = 0
    stamp = np.full(ptr.size, -1, dtype=np.int64)
    bounds = ptr.tolist()
    for v in vertices.tolist():
        lo, hi = bounds[v], bounds[v + 1]
        w = neighbors[lo:hi]
        cw = colors[w]
        known = cw >= 0
        if not known.any():
            colors[v] = 0
            continue
        w, cw, e = w[known], cw[known], edge_ids[lo:hi][known]
        slot, of = _slots(ptr, w)
        x = neighbors[slot]
        cx = colors[x]
        known = cx >= 0
        x, cx, of, ex = x[known], cx[known], of[known], edge_ids[slot[known]]
        repeated = np.bincount(cw)[cw] > 1
        forbidden = np.concatenate((cw, cx[repeated[of] | (hub[star[ex]] == x)]))
        stamp[forbidden] = v
        cv = int(np.argmax(stamp[: forbidden.size + 1] != v))
        colors[v] = cv
        # SMC _update_stars!: v joins the star of a neighbor's neighbor of its color, centred
        # on that neighbor; repeated colors around v form stars centred on v.
        hit = np.flatnonzero(cx == cv)
        joins, first = np.unique(of[hit], return_index=True)
        stars = np.empty(w.size, dtype=np.int64)
        stars[joins] = star[ex[hit[first]]]
        hub[stars[joins]] = w[joins]
        fresh = ~repeated
        fresh[joins] = False
        k = int(np.count_nonzero(fresh))
        stars[fresh] = np.arange(n_stars, n_stars + k)
        hub[n_stars : n_stars + k] = -1
        n_stars += k
        if repeated.any():
            centred, group = np.unique(cw[repeated], return_inverse=True)
            stars[repeated] = n_stars + group
            hub[n_stars : n_stars + centred.size] = v
            n_stars += centred.size
        star[e] = stars
    # SMC postprocess! for a StarSet: hubs keep their colors; a single-edge star is centred
    # on its higher vertex unless its lower one's color is already kept.
    used = np.zeros(ptr.size, dtype=bool)
    used[colors[diagonal]] = True
    hubs = hub[:n_stars]
    used[colors[hubs[hubs >= 0]]] = True
    single = np.flatnonzero(hubs[star] < 0)
    switch = _switch_single(used, colors[a[single]], colors[b[single]])
    hubs[star[single]] = np.where(switch, a[single], b[single])
    return _relabel(colors, used), hubs[star]


def _acyclic(
    graph: Graph, n_edges: int, vertices: Int[np.ndarray, " n"]
) -> tuple[Int[np.ndarray, " n"], Int[np.ndarray, " edges"]]:
    """Acyclic-color an adjacency graph; return the colors and each edge's tree root.

    Ports SMC ``acyclic_coloring`` (SMC paper, Appendix A.2.3; Gebremedhin, Tarafdar, Manne
    and Pothen 2007, Algorithm 3.1): the edges between two colors form trees kept by union-find, and a
    color that would close a two-colored cycle is forbidden. A tree reached from ``v``
    through two distinct neighbors forbids its other color (SMC ``_prevent_cycle!``); then
    ``v``'s edges to one color join one tree (``_grow_star!``), and each joins the tree
    beyond its neighbor in ``v``'s color (``_merge_trees!``). Union by size and path
    compression replace SMC's union by rank; the trees are the same.
    """
    ptr, neighbors, edge_ids = graph
    colors = np.full(ptr.size - 1, -1, dtype=np.int64)
    parent = np.arange(n_edges, dtype=np.int64)
    size = np.ones(n_edges, dtype=np.int64)
    stamp = np.full(ptr.size, -1, dtype=np.int64)
    bounds = ptr.tolist()
    for v in vertices.tolist():
        lo, hi = bounds[v], bounds[v + 1]
        w = neighbors[lo:hi]
        cw = colors[w]
        known = cw >= 0
        if not known.any():
            colors[v] = 0
            continue
        w, cw, e = w[known], cw[known], edge_ids[lo:hi][known]
        stamp[cw] = v
        slot, of = _slots(ptr, w)
        cx = colors[neighbors[slot]]
        known = (cx >= 0) & (stamp[cx] != v)
        cx, of = cx[known], of[known]
        root = _find(parent, edge_ids[slot[known]])
        forbidden = cw
        if root.size:
            by_tree = np.lexsort((of, root))
            starts = np.flatnonzero(np.diff(root[by_tree], prepend=-1))
            lasts = np.append(starts[1:], root.size) - 1
            twice = of[by_tree[starts]] != of[by_tree[lasts]]
            forbidden = np.concatenate((cw, cx[by_tree[starts[twice]]]))
        stamp[forbidden] = v
        cv = int(np.argmax(stamp[: forbidden.size + 1] != v))
        colors[v] = cv
        _, first, group = np.unique(cw, return_index=True, return_inverse=True)
        head = e[first]
        total = np.bincount(group)
        hit = np.flatnonzero(cx == cv)
        if hit.size:
            joins, at = np.unique(of[hit], return_index=True)
            trees, tree_group = root[hit[at]], group[joins]
            largest = np.lexsort((-size[trees], tree_group))
            lead = np.flatnonzero(np.diff(tree_group[largest], prepend=-1))
            head[tree_group[largest[lead]]] = trees[largest[lead]]
            parent[trees] = head[tree_group]
            total = total + np.bincount(tree_group, size[trees], minlength=total.size).astype(int)
        parent[e] = head[group]
        size[head] = total
    return colors, _find(parent, np.arange(n_edges))


def _find(parent: Int[np.ndarray, " all"], edges: Int[np.ndarray, " k"]) -> Int[np.ndarray, " k"]:
    """Return the root of each edge's tree, pointing the edges at their roots."""
    root = parent[edges]
    while True:
        up = parent[root]
        if np.array_equal(up, root):
            break
        root = up
    parent[edges] = root
    return root


def _trees(
    colors: Int[np.ndarray, " n"],
    root: Int[np.ndarray, " edges"],
    a: Int[np.ndarray, " edges"],
    b: Int[np.ndarray, " edges"],
    diagonal: Bool[np.ndarray, " n"],
) -> tuple[
    Int[np.ndarray, " n"],
    Int[np.ndarray, " edges"],
    Int[np.ndarray, " edges"],
    int,
    Bool[np.ndarray, " edges"],
]:
    """Split an acyclic coloring's edges into tree vertices, and drop the colors no edge needs.

    Ports SMC ``TreeSet`` and its ``postprocess!`` (SMC paper, Appendix A.2.5): a vertex is one tree vertex per tree it is
    in; a star keeps its hub's color, any other tree both its colors, and a single-edge tree
    is solved into its higher vertex unless its lower one's color is already kept.

    Returns:
        The colors, each edge's two tree vertices, the number of tree vertices, and whether
        a single edge's leaf is its lower vertex.
    """
    n = colors.size
    tree = np.unique(root, return_inverse=True)[1]
    n_trees = int(tree.max(initial=-1)) + 1
    nodes, node = np.unique(np.concatenate((tree * n + a, tree * n + b)), return_inverse=True)
    inner = np.bincount(node, minlength=nodes.size) > 1
    is_star = np.bincount(nodes[inner] // n, minlength=n_trees) <= 1
    used = np.zeros(n + 1, dtype=bool)
    used[colors[diagonal]] = True
    used[colors[nodes[inner & is_star[nodes // n]] % n]] = True
    full = ~is_star[tree]
    used[colors[a[full]]] = True
    used[colors[b[full]]] = True
    single = np.flatnonzero(np.bincount(tree, minlength=n_trees)[tree] == 1)
    leaf_a = np.ones(a.size, dtype=bool)
    leaf_a[single] = ~_switch_single(used, colors[a[single]], colors[b[single]])
    return _relabel(colors, used), node[: a.size], node[a.size :], nodes.size, leaf_a


def _switch_single(
    used: Bool[np.ndarray, " colors"], lower: Int[np.ndarray, " k"], higher: Int[np.ndarray, " k"]
) -> Bool[np.ndarray, " k"]:
    """Return which single edges SMC's ``postprocess!`` centres on their lower vertex.

    In edge order, an edge whose lower vertex's color is kept is centred there; otherwise on
    its higher vertex, whose color is then kept. The kept set only grows, so each pass runs
    to the next edge that grows it.
    """
    switch = np.zeros(lower.size, dtype=bool)
    start = 0
    while start < lower.size:
        at_lower = used[lower[start:]]
        grows = ~at_lower & ~used[higher[start:]]
        k = int(np.argmax(grows)) if grows.any() else grows.size
        switch[start : start + k] = at_lower[:k]
        if k < grows.size:
            used[higher[start + k]] = True
        start += k + 1
    return switch


def _relabel(colors: Int[np.ndarray, " n"], used: Bool[np.ndarray, " c"]) -> Int[np.ndarray, " n"]:
    """Renumber the ``used`` colors ``0, 1, ...`` in order; every other vertex gets ``-1``."""
    new = np.cumsum(used) - 1
    safe = np.maximum(colors, 0)
    return np.where((colors >= 0) & used[safe], new[safe], -1)


def _first_seen(colors: Int[np.ndarray, " n"]) -> Int[np.ndarray, " n"]:
    """Renumber the colors ``0, 1, ...`` by first appearance, as SMC ``remap_colors`` does."""
    out = np.full(colors.size, -1, dtype=np.int64)
    known = np.flatnonzero(colors >= 0)
    _, first, inverse = np.unique(colors[known], return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(first.size)
    out[known] = rank[inverse]
    return out
