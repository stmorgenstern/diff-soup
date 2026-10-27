"""The Gauss-Newton system of a parameter hierarchy, solved along its tree.

The system of a hierarchy is stored per node of every level, as the level
reducer sums it ([LevelBlocks][diff_soup.inference.engine.LevelBlocks]): per
node its diagonal block and its coupling to each of its ancestors, and at the
root the shared block. Joint columns are laid out level by level: the shared
coordinates, then every node of level 1, and so on down to the experiments.

A damped solve eliminates the levels from the experiments up. Every node of a
level is factored at once, and the Schur complement of its block is folded into
the blocks of its ancestors, so the stored set of blocks never grows. The
shared block is solved last, and the steps follow top-down. Equality rows are
imposed by one small projection, and the factors give the inverse-norm
``s^T (H + lambda I)^-1 s`` a secular search on ``lambda`` needs.

Every array computation is compiled once per shape; the damping, the scale,
the diagonal added to the scaled matrix, the point, and which coordinates are
fixed and which nodes are held enter as arguments. The secular search runs on
the host.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------


class LevelSystem(eqx.Module):
    """The Gauss-Newton matrix and gradient of a hierarchy at one point, per node.

    Fields indexed by level carry ``None`` at level 0. Level ``l`` has ``N_l``
    nodes of width ``v_l``; the root is the shared block of width ``g``.

    Attributes:
        root: ``(g, g)``, the shared block.
        root_gradient: ``(g,)``.
        diagonal: ``diagonal[l]``, ``(N_l, v_l, v_l)``.
        cross: ``cross[l][a]``, ``(N_l, v_l, v_a)`` for ``a = 0 .. l - 1``:
            each node's coupling to its level-``a`` ancestor (``a = 0``, the
            shared block).
        gradient: ``gradient[l]``, ``(N_l, v_l)``.
        point: ``(n,)``, the point of the system.
        constraint_matrix: ``(c, n)``, equality rows ``A``; a step ``s``
            satisfies ``A (point + s) = constraint_rhs``.
        constraint_rhs: ``(c,)``.
        cost: The objective at the point: the weighted lane costs and the
            regularizer rows, each block times its weight.
        ancestors: ``ancestors[l][a]``, ``(N_l,)``, the level-``a`` ancestor of
            each node of level ``l``, for ``1 <= a < l``.
        descendants: ``descendants[l][a]``, ``(N_a, K)``, the nodes of level
            ``l`` below each node of level ``a``, in node order, padded.
        present: Like ``descendants``: whether an entry names a node.
        fixed_mask: ``(n,)``, boolean: whether each joint coordinate is fixed.
        held_mask: ``held_mask[l]``, ``(N_l,)``, boolean: whether each node of
            level ``l`` is held at its value, a node without data and without
            a regularizer row of nonzero weight.
        counts: ``(N_1, ..., N_D)``.
        widths: ``(g, v_1, ..., v_D)``.

    The fixed coordinates and the held nodes are arrays of the system, not
    part of its structure: systems of one hierarchy that fix or hold different
    sets share every compiled program. ``fixed`` and ``held`` list them on the
    host.
    """

    root: Any
    root_gradient: Any
    diagonal: tuple
    cross: tuple
    gradient: tuple
    point: Any
    constraint_matrix: Any
    constraint_rhs: Any
    cost: Any
    ancestors: tuple
    descendants: tuple
    present: tuple
    fixed_mask: Any
    held_mask: tuple
    counts: tuple = eqx.field(static=True)
    widths: tuple = eqx.field(static=True)

    @property
    def depth(self) -> int:
        """The number of levels below the root."""
        return len(self.counts)

    @property
    def size(self) -> int:
        """The number of joint coordinates the system holds."""
        return _offsets(self.counts, self.widths)[-1]

    @property
    def fixed(self) -> tuple:
        """The fixed joint coordinates, in increasing order."""
        return tuple(int(index) for index in np.flatnonzero(np.asarray(self.fixed_mask)))

    @property
    def held(self) -> tuple:
        """``(level, node)`` of every node held at its value, level by level in node order."""
        return tuple(
            (level, int(node))
            for level in range(1, self.depth + 1)
            for node in np.flatnonzero(np.asarray(self.held_mask[level]))
        )


def _offsets(counts: tuple, widths: tuple) -> tuple:
    """The first joint column of every level 0 .. D, then the number of columns."""
    offsets = [0, int(widths[0])]
    for level in range(1, len(counts) + 1):
        offsets.append(offsets[-1] + int(counts[level - 1]) * int(widths[level]))
    return tuple(offsets)


def _locate(column: int, counts: tuple, widths: tuple, offsets: tuple) -> tuple | None:
    """``(level, node, position)`` of a joint column, or ``None`` outside the levels."""
    if column < widths[0]:
        return (0, 0, column)
    for level in range(1, len(counts) + 1):
        if offsets[level] <= column < offsets[level + 1]:
            node, position = divmod(column - offsets[level], widths[level])
            return (level, int(node), int(position))
    return None


def _tree_indices(counts: tuple, parents: tuple) -> tuple:
    """Host tables of ancestors and descendants between every pair of levels."""
    depth = len(counts)
    ancestors: list = [None] + [None] * depth
    descendants: list = [None] + [None] * depth
    present: list = [None] + [None] * depth
    for level in range(1, depth + 1):
        above_nodes: list = [None]
        for above in range(1, level):
            parent = np.asarray(parents[level], dtype=np.int64)
            if above == level - 1:
                above_nodes.append(parent)
            else:
                above_nodes.append(ancestors[level - 1][above][parent])
        ancestors[level] = tuple(above_nodes)
        tables: list = [None]
        masks: list = [None]
        for above in range(1, level):
            owner = ancestors[level][above]
            count = int(counts[above - 1])
            per_node = np.bincount(owner, minlength=count)
            most = max(int(per_node.max(initial=0)), 1)
            order = np.argsort(owner, kind="stable")
            starts = np.concatenate(([0], np.cumsum(per_node)[:-1]))
            slots = np.arange(owner.shape[0]) - starts[owner[order]]
            table = np.zeros((count, most), dtype=np.int32)
            mask = np.zeros((count, most), dtype=bool)
            table[owner[order], slots] = order
            mask[owner[order], slots] = True
            tables.append(table)
            masks.append(mask)
        descendants[level] = tuple(tables)
        present[level] = tuple(masks)
    return tuple(ancestors), tuple(descendants), tuple(present)


def _device(tables: tuple, device: Any = None) -> tuple:
    """Nested host tables as arrays on ``device`` (the default device for ``None``), ``None`` kept."""
    return tuple(
        None
        if row is None
        else tuple(None if entry is None else jax.device_put(entry, device) for entry in row)
        for row in tables
    )


class LevelLayout:
    """What every level system of one set of level tables shares, built once.

    The index tables between the levels depend on the tables only, never on
    the point: ``ancestors``, ``descendants`` and ``present`` are built on the
    host once and placed on the device once. Arrays every system of a fit
    places on the device unchanged (the fixed mask, the held masks, the
    equality rows when no coordinate is frozen) are placed once per distinct
    value and reused; only the point and the cost move with every system.

    Attributes:
        counts: ``(N_1, ..., N_D)``.
        widths: ``(g, v_1, ..., v_D)``.
        offsets: The first joint column of every level, then the number of
            columns.
        ancestors: The host tables of ``_tree_indices``.
        device: The JAX device the system is placed on; ``None`` for the
            default device.
        device_ancestors: ``ancestors`` on ``device``.
        device_descendants: The descendant tables on ``device``.
        device_present: Their masks on ``device``.
    """

    # The distinct arrays kept on the device; beyond it the oldest is dropped.
    KEPT = 64

    def __init__(self, counts: tuple, widths: tuple, parents: tuple, device: Any = None):
        """Build the index tables of a hierarchy of ``counts`` nodes and ``widths`` coordinates."""
        self.counts = tuple(int(count) for count in counts)
        self.widths = tuple(int(width) for width in widths)
        self.offsets = _offsets(self.counts, self.widths)
        self.device = device
        ancestors, descendants, present = _tree_indices(self.counts, parents)
        self.ancestors = ancestors
        self.device_ancestors = _device(ancestors, device)
        self.device_descendants = _device(descendants, device)
        self.device_present = _device(present, device)
        self._kept: dict = {}

    def constant(self, name: str, value: np.ndarray) -> Any:
        """``value`` on the device, placed once per distinct value of ``name``."""
        value = np.ascontiguousarray(value)
        key = (name, value.dtype.str, value.shape, value.tobytes())
        placed = self._kept.get(key)
        if placed is None:
            if len(self._kept) >= self.KEPT:
                self._kept.pop(next(iter(self._kept)))
            placed = self._kept[key] = jax.device_put(value, self.device)
        return placed


def level_layout(tables: Any, device: Any = None) -> LevelLayout:
    """The [LevelLayout][diff_soup.inference.hierarchical._level_elimination.LevelLayout] of ``tables``.

    Args:
        tables: The [LevelTables][diff_soup.inference.engine.LevelTables].
        device: The JAX device the systems are placed on; ``None`` for the
            default device.
    """
    parents = tuple(None if entry is None else np.asarray(entry) for entry in tables.parents)
    return LevelLayout(tables.counts, tables.widths, parents, device)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _listed(values: Any, limit: int = 8) -> str:
    """``values`` separated by commas, cut after ``limit`` with the number left."""
    values = tuple(values)
    text = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        text += f" and {len(values) - limit} more"
    return text


def _regularizer_blocks(
    regularizer: Any,
    point: np.ndarray,
    counts: tuple,
    widths: tuple,
    offsets: tuple,
    ancestors: tuple,
    dtype: Any,
    names: Any,
    members: Any,
    block_weights: Any = None,
    device: Any = None,
) -> tuple:
    """The regularizer's Gauss-Newton blocks per node, their cost, and the nodes they touch.

    ``block_weights`` holds one weight per block of ``block_rows``, in its
    order, or is ``None`` for weight one everywhere. A block contributes its
    weight times its curvature, slope and cost; a block of weight zero
    contributes nothing and touches no node. The regularizer is evaluated at
    ``point`` placed on ``device``.
    """
    rows = regularizer.residual(jax.device_put(point, device))
    total_rows = int(np.shape(rows)[0])
    if total_rows == 0:
        return None, 0.0, frozenset()
    provider = getattr(regularizer, "block_rows", None)
    if not callable(provider):
        raise ValueError(
            f"the regularizer has {total_rows} residual rows and no block_rows(x) naming the "
            "columns each row addresses; the levels of a hierarchy take a regularizer whose "
            "rows declare their columns, as a regularizer declared on the study does."
        )
    declared = tuple(provider(jax.device_put(point, device)))
    covered = sum(int(np.size(item[1])) for item in declared)
    if covered != total_rows:
        raise ValueError(
            f"the regularizer's block_rows(x) cover {covered} rows and its residual(x) has "
            f"{total_rows}; name the columns of every row exactly once."
        )
    if block_weights is not None and len(block_weights) != len(declared):
        raise ValueError(
            f"{len(block_weights)} regularizer block weights were given and the regularizer's "
            f"block_rows(x) declares {len(declared)} blocks; give one weight per block, in the "
            "order of block_rows(x)."
        )
    depth = len(counts)
    g = int(widths[0])
    root = np.zeros((g, g), dtype=dtype)
    root_gradient = np.zeros((g,), dtype=dtype)
    diagonal = [None] + [
        np.zeros((counts[level - 1], widths[level], widths[level]), dtype=dtype)
        for level in range(1, depth + 1)
    ]
    cross = [None] + [
        tuple(
            np.zeros((counts[level - 1], widths[level], widths[above]), dtype=dtype)
            for above in range(level)
        )
        for level in range(1, depth + 1)
    ]
    gradient = [None] + [
        np.zeros((counts[level - 1], widths[level]), dtype=dtype) for level in range(1, depth + 1)
    ]

    def level_name(level: int) -> Any:
        return names[level - 1] if names is not None else level

    def member(level: int, node: int) -> Any:
        return members[level - 1][node] if members is not None else node

    cost = 0.0
    touched: set = set()
    for index, (columns, residual, jacobian) in enumerate(declared):
        columns = tuple(int(column) for column in columns)
        residual = np.ravel(np.asarray(residual, dtype=dtype))
        jacobian = np.asarray(jacobian, dtype=dtype)
        if jacobian.shape != (residual.size, len(columns)):
            raise ValueError(
                f"regularizer block {index} has a Jacobian of shape {jacobian.shape}; its "
                f"{residual.size} rows and {len(columns)} columns need "
                f"{(residual.size, len(columns))}."
            )
        where = []
        for column in columns:
            located = _locate(column, counts, widths, offsets)
            if located is None:
                raise ValueError(
                    f"regularizer block {index} addresses column {column}, outside the "
                    f"{offsets[-1]} coordinates of the levels; a regularizer on the levels of "
                    "a hierarchy addresses their coordinates only."
                )
            where.append(located)
        nodes = sorted({(level, node) for level, node, _ in where if level > 0})
        if nodes:
            deepest_level, deepest_node = max(nodes)
            for level, node in nodes:
                expected = (
                    deepest_node
                    if level == deepest_level
                    else int(ancestors[deepest_level][level][deepest_node])
                )
                if node != expected:
                    first, second = sorted((node, expected))
                    raise ValueError(
                        f"regularizer block {index} addresses members {member(level, first)} "
                        f"and {member(level, second)} of level {level_name(level)!r}; a block "
                        "may couple a member with its ancestors only."
                    )
        weight = None if block_weights is None else float(block_weights[index])
        if weight == 0.0:
            continue
        touched.update(nodes)
        curvature = jacobian.T @ jacobian
        slope = jacobian.T @ residual
        if weight is None:
            cost += 0.5 * float(residual @ residual)
        else:
            curvature = weight * curvature
            slope = weight * slope
            cost += weight * (0.5 * float(residual @ residual))
        for i, (level_i, node_i, position_i) in enumerate(where):
            if level_i == 0:
                root_gradient[position_i] += slope[i]
            else:
                gradient[level_i][node_i, position_i] += slope[i]
            for j, (level_j, _, position_j) in enumerate(where):
                if level_i == level_j:
                    if level_i == 0:
                        root[position_i, position_j] += curvature[i, j]
                    else:
                        diagonal[level_i][node_i, position_i, position_j] += curvature[i, j]
                elif level_i > level_j:
                    cross[level_i][level_j][node_i, position_i, position_j] += curvature[i, j]
    extra = (root, root_gradient, tuple(diagonal), tuple(cross), tuple(gradient))
    return extra, cost, frozenset(touched)


def regularizer_blocks_at(
    regularizer: Any,
    point: np.ndarray,
    layout: LevelLayout,
    dtype: Any,
    names: Any = None,
    members: Any = None,
    block_weights: Any = None,
) -> tuple:
    """The regularizer's blocks per node at ``point``, their cost, and the nodes they touch.

    What ``level_system`` adds for ``regularizer`` at ``point`` (its argument
    ``regularizer_blocks``): ``(extra, cost, touched)``, ``extra`` ``None``
    for a regularizer without rows. The cost is every block's ``0.5 |r_b|^2``
    times its weight in ``block_weights``. The regularizer is evaluated once,
    at ``point`` placed on the layout's device.

    Raises:
        ValueError: As ``level_system`` for the regularizer's blocks.
    """
    return _regularizer_blocks(
        regularizer,
        np.asarray(point, dtype=dtype),
        layout.counts,
        layout.widths,
        layout.offsets,
        layout.ancestors,
        np.dtype(dtype),
        names,
        members,
        block_weights,
        layout.device,
    )


def _held_nodes(blocks: Any, tables: Any, lane_weights: Any, touched: frozenset) -> tuple:
    """The nodes without data: no residual rows or no weight below them, and no regularizer row.

    ``touched`` holds the nodes a regularizer block of nonzero weight addresses;
    a block of weight zero does not count, so a member of weight zero with every
    node below it is held at its value.
    """
    n_lanes = int(np.shape(tables.nodes)[0])
    weights = np.ones((n_lanes,)) if lane_weights is None else np.asarray(lane_weights, dtype=float)
    held = []
    for level in range(1, len(tables.counts) + 1):
        rows = np.asarray(blocks.rows[level])
        lanes = np.asarray(tables.lanes[level])
        present = np.asarray(tables.present[level])
        total = np.where(present, weights[lanes], 0.0).sum(axis=1)
        for node in np.flatnonzero((rows == 0) | (total == 0.0)):
            if (level, int(node)) not in touched:
                held.append((level, int(node)))
    return tuple(held)


def regularizer_block_members(
    regularizer: Any, point: Any, tables: Any, level: int, device: Any = None
) -> Any:
    """The node of ``level`` that each regularizer block is attributed to, or -1.

    A block of ``regularizer.block_rows(point)`` belongs to the node of
    ``level`` (1 .. D) that is its deepest node or that node's ancestor, so a
    member's prior rows go with the member. A block whose deepest node lies
    above ``level``, or that addresses the shared coordinates only, belongs to
    no node of ``level`` (-1). A block may couple a node with its ancestors
    only, so the attribution is unique; a block that breaks that rule, or
    addresses a column outside the levels, gets -1 here and is refused by
    ``level_system``.

    Args:
        regularizer: The regularizer, with ``residual(x)`` and
            ``block_rows(x)``, or ``None``.
        point: ``(n,)``, a joint point; the blocks' columns do not depend on it.
        tables: The level tables of the hierarchy.
        level: The number of the level, 1 .. D.
        device: The JAX device the regularizer is evaluated on; ``None`` for
            the default device.

    Returns:
        ``(B,)`` node indices of ``level`` (-1 for none), in the order of
        ``block_rows``; ``None`` when the regularizer has no rows or no
        ``block_rows``.
    """
    if regularizer is None:
        return None
    rows = regularizer.residual(jax.device_put(point, device))
    if int(np.shape(rows)[0]) == 0:
        return None
    provider = getattr(regularizer, "block_rows", None)
    if not callable(provider):
        return None
    counts = tuple(int(count) for count in tables.counts)
    widths = tuple(int(width) for width in tables.widths)
    offsets = _offsets(counts, widths)
    parents = tuple(None if entry is None else np.asarray(entry) for entry in tables.parents)
    ancestors, _, _ = _tree_indices(counts, parents)
    attributed = []
    for item in provider(jax.device_put(point, device)):
        located = (_locate(int(column), counts, widths, offsets) for column in item[0])
        nodes = [(where[0], where[1]) for where in located if where is not None and where[0] > 0]
        if not nodes:
            attributed.append(-1)
            continue
        deepest_level, deepest_node = max(nodes)
        if deepest_level < level:
            attributed.append(-1)
        elif deepest_level == level:
            attributed.append(int(deepest_node))
        else:
            attributed.append(int(ancestors[deepest_level][level][deepest_node]))
    return np.asarray(attributed, dtype=np.int64)


@jax.jit
def _assemble(blocks, keep, extra, ancestors):
    depth = len(blocks.diagonal) - 1
    root = jnp.sum(blocks.shared, axis=0)
    root_gradient = jnp.sum(blocks.gradient_shared, axis=0)
    diagonal = [None] + list(blocks.diagonal[1:])
    cross = [None] + [list(row) for row in blocks.cross[1:]]
    gradient = [None] + list(blocks.gradient[1:])
    if extra is not None:
        extra_root, extra_root_gradient, extra_diagonal, extra_cross, extra_gradient = extra
        root = root + extra_root
        root_gradient = root_gradient + extra_root_gradient
        for level in range(1, depth + 1):
            diagonal[level] = diagonal[level] + extra_diagonal[level]
            gradient[level] = gradient[level] + extra_gradient[level]
            for above in range(level):
                cross[level][above] = cross[level][above] + extra_cross[level][above]
    shared_keep = keep[0]
    root = root * shared_keep[:, None] * shared_keep[None, :] + jnp.diag(1.0 - shared_keep)
    root_gradient = root_gradient * shared_keep
    for level in range(1, depth + 1):
        own = keep[level]
        diagonal[level] = diagonal[level] * own[:, :, None] * own[:, None, :] + jax.vmap(jnp.diag)(
            1.0 - own
        )
        gradient[level] = gradient[level] * own
        for above in range(level):
            if above == 0:
                other = shared_keep[None, None, :]
            else:
                other = keep[above][ancestors[level][above]][:, None, :]
            cross[level][above] = cross[level][above] * own[:, :, None] * other
    return (
        root,
        root_gradient,
        tuple(diagonal),
        (None,) + tuple(tuple(row) for row in cross[1:]),
        tuple(gradient),
    )


def level_system(
    blocks: Any,
    tables: Any,
    point: Any,
    *,
    constraint_matrix: Any = None,
    constraint_rhs: Any = None,
    fixed_indices: Any = (),
    regularizer: Any = None,
    lane_weights: Any = None,
    members: Any = None,
    block_weights: Any = None,
    device: Any = None,
    layout: LevelLayout | None = None,
    regularizer_blocks: Any = None,
) -> LevelSystem:
    """The system of the whole objective at ``point`` from the level blocks.

    Regularizer rows are added once to the blocks their columns address, each
    block of ``regularizer.block_rows`` times its weight in ``block_weights``
    (curvature, slope and cost alike). A fixed coordinate and every coordinate
    of a held node (no data and no regularizer row of nonzero weight) get a
    unit diagonal, a zero gradient and zero couplings, so a solve leaves them
    at their value; equality rows absorb their values.

    Args:
        blocks: The [LevelBlocks][diff_soup.inference.engine.LevelBlocks] at
            ``point``.
        tables: The [LevelTables][diff_soup.inference.engine.LevelTables] the
            blocks were summed with.
        point: ``(n,)``, the joint point, level by level.
        constraint_matrix: Optional ``(c, n)`` equality rows.
        constraint_rhs: Optional ``(c,)`` right-hand side.
        fixed_indices: Joint coordinates held at their value.
        regularizer: Optional regularizer with ``residual(x)`` and
            ``block_rows(x)``.
        lane_weights: Optional ``(L,)`` weight of every lane; a node whose lanes
            all weigh zero has no data.
        members: Optional member labels per level 1 .. D, for messages.
        block_weights: Optional weight of every block of
            ``regularizer.block_rows(point)``, in its order
            (``regularizer_block_members`` attributes each block to a member
            of the weighted level); ``None`` weighs every block one. A block of
            weight zero adds nothing and does not keep its nodes from being
            held.
        device: The JAX device the system is placed on, that of the blocks
            and tables; ``None`` for the default device.
        layout: The [LevelLayout][diff_soup.inference.hierarchical._level_elimination.LevelLayout]
            of ``tables`` on ``device``, to reuse its index tables and kept
            arrays; ``None`` builds one for this system alone.
        regularizer_blocks: What ``regularizer_blocks`` returned for this
            regularizer, point, layout, dtype and ``block_weights``, to reuse
            instead of evaluating the regularizer again; ``None`` evaluates it.

    Returns:
        The [LevelSystem][diff_soup.inference.hierarchical._level_elimination.LevelSystem].

    Raises:
        ValueError: If the point does not have the coordinates of the levels,
            a fixed index is out of range, a regularizer block couples two
            members of one level or addresses a column outside the levels, or
            ``block_weights`` does not give one weight per block.
    """
    if layout is None:
        layout = level_layout(tables, device)
    counts, widths, offsets = layout.counts, layout.widths, layout.offsets
    if counts != tuple(int(count) for count in tables.counts) or widths != tuple(
        int(width) for width in tables.widths
    ):
        raise ValueError(
            f"the layout holds the levels {counts} of widths {widths} and the tables "
            f"{tuple(tables.counts)} of widths {tuple(tables.widths)}; give the layout of "
            "these tables."
        )
    depth = len(counts)
    size = offsets[-1]
    dtype = np.dtype(blocks.shared.dtype)
    point = np.asarray(point, dtype=dtype)
    if point.shape != (size,):
        raise ValueError(
            f"the point has shape {point.shape} and the levels of the tables hold {size} "
            "coordinates; give the joint point level by level."
        )
    fixed = tuple(sorted({int(index) for index in fixed_indices}))
    if any(index < 0 or index >= size for index in fixed):
        raise ValueError(f"fixed_indices name coordinates 0 .. {size - 1} only; got {fixed}.")
    extra, regularizer_cost, touched = None, 0.0, frozenset()
    if regularizer_blocks is not None:
        extra, regularizer_cost, touched = regularizer_blocks
    elif regularizer is not None:
        extra, regularizer_cost, touched = regularizer_blocks_at(
            regularizer,
            point,
            layout,
            dtype,
            getattr(tables, "names", None),
            members,
            block_weights,
        )
    held = _held_nodes(blocks, tables, lane_weights, touched)
    fixed_mask = np.zeros((size,), dtype=bool)
    fixed_mask[list(fixed)] = True
    held_mask = [None] + [np.zeros((count,), dtype=bool) for count in counts]
    keep = ~fixed_mask
    for level, node in held:
        held_mask[level][node] = True
        start = offsets[level] + node * widths[level]
        keep[start : start + widths[level]] = False
    if constraint_matrix is None:
        matrix = np.zeros((0, size), dtype=dtype)
        rhs = np.zeros((0,), dtype=dtype)
    else:
        matrix = np.array(constraint_matrix, dtype=dtype, copy=True).reshape(-1, size)
        rhs = np.array(constraint_rhs, dtype=dtype, copy=True).reshape(-1)
    frozen = ~keep
    if matrix.shape[0] and frozen.any():
        rhs = rhs - matrix[:, frozen] @ point[frozen]
        matrix[:, frozen] = 0.0
    keep_levels = [keep[: widths[0]].astype(dtype)] + [
        keep[offsets[level] : offsets[level + 1]]
        .astype(dtype)
        .reshape(counts[level - 1], widths[level])
        for level in range(1, depth + 1)
    ]
    keep_levels = tuple(
        layout.constant(f"keep.{level}", entry) for level, entry in enumerate(keep_levels)
    )
    root, root_gradient, diagonal, cross, gradient = _assemble(
        blocks, keep_levels, extra, layout.device_ancestors
    )
    cost = float(np.sum(np.asarray(blocks.cost[depth], dtype=np.float64))) + regularizer_cost
    matrix = layout.constant("constraint_matrix", matrix)
    if rhs.shape[0] and frozen.any():
        # The right-hand side absorbs the values of the frozen columns at this point.
        rhs = jax.device_put(rhs, layout.device)
    else:
        rhs = layout.constant("constraint_rhs", rhs)
    point, cost = jax.device_put((point, np.asarray(cost, dtype=dtype)), layout.device)
    return LevelSystem(
        root=root,
        root_gradient=root_gradient,
        diagonal=diagonal,
        cross=cross,
        gradient=gradient,
        point=point,
        constraint_matrix=matrix,
        constraint_rhs=rhs,
        cost=cost,
        ancestors=layout.device_ancestors,
        descendants=layout.device_descendants,
        present=layout.device_present,
        fixed_mask=layout.constant("fixed_mask", fixed_mask),
        held_mask=(None,)
        + tuple(
            layout.constant(f"held_mask.{level}", held_mask[level]) for level in range(1, depth + 1)
        ),
        counts=counts,
        widths=widths,
    )


# ---------------------------------------------------------------------------
# The elimination (inside compiled programs)
# ---------------------------------------------------------------------------


def _split(system: LevelSystem, vector: Any) -> list:
    """A joint vector as the root part and one ``(N_l, v_l)`` part per level."""
    offsets = _offsets(system.counts, system.widths)
    parts = [vector[: system.widths[0]]]
    for level in range(1, system.depth + 1):
        parts.append(
            vector[offsets[level] : offsets[level + 1]].reshape(
                system.counts[level - 1], system.widths[level]
            )
        )
    return parts


def _node_sum(system: LevelSystem, values: Any, level: int, above: int) -> Any:
    """Sum per-node values of ``level`` over the nodes below each node of ``above``."""
    if above == 0:
        return jnp.sum(values, axis=0)
    table = system.descendants[level][above]
    mask = system.present[level][above]
    gathered = values[table]
    keep = jnp.reshape(mask, mask.shape + (1,) * (values.ndim - 1))
    return jnp.sum(jnp.where(keep, gathered, jnp.zeros((), values.dtype)), axis=1)


def _ancestor_values(system: LevelSystem, values: Any, level: int, above: int) -> Any:
    """The values of each node's level-``above`` ancestor, one row per node of ``level``."""
    return values[system.ancestors[level][above]]


def _lower_solve(factor: Any, right: Any) -> Any:
    """``L^-1 B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True))(factor, right)


def _upper_solve(factor: Any, right: Any) -> Any:
    """``L^-T B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True, trans="T"))(
        factor, right
    )


def _fold(system: LevelSystem, damping: Any, rhs: Any, lowest: int) -> tuple:
    """Eliminate the levels ``D .. lowest`` from the experiments up.

    Every node of a level is factored at once, ``L_n = chol(diag~_n + damping I)``;
    its couplings ``Y_n^a = L_n^-1 cross~_n^a`` and right-hand sides
    ``y_n = L_n^-1 R~_n`` are formed, and ``(Y_n^a)^T Y_n^b`` and
    ``(Y_n^a)^T y_n`` are folded into the node's ancestors, summed over the
    nodes below each ancestor in node order.
    """
    widths = system.widths
    depth = system.depth
    dtype = system.root.dtype
    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    root = system.root
    root_rhs = None if rhs is None else rhs[0]
    level_rhs = None if rhs is None else list(rhs)
    factors: list = [None] * (depth + 1)
    couplings: list = [None] * (depth + 1)
    solved: list = [None] * (depth + 1)
    for level in range(depth, lowest - 1, -1):
        width = widths[level]
        if width == 0:
            continue
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping * jnp.eye(width, dtype=dtype)
        )
        factors[level] = factor
        row: list = [None] * level
        for above in range(level):
            if widths[above]:
                row[above] = _lower_solve(factor, cross[level][above])
        couplings[level] = row
        if rhs is not None:
            solved[level] = _lower_solve(factor, level_rhs[level])
        for above in range(level):
            if not widths[above]:
                continue
            for other in range(above + 1):
                if not widths[other]:
                    continue
                product = jnp.einsum("nia,nib->nab", row[above], row[other])
                total = _node_sum(system, product, level, above)
                if above == 0:
                    root = root - total
                elif other == above:
                    diagonal[above] = diagonal[above] - total
                else:
                    cross[above][other] = cross[above][other] - total
            if rhs is not None:
                folded = jnp.einsum("nia,nik->nak", row[above], solved[level])
                total = _node_sum(system, folded, level, above)
                if above == 0:
                    root_rhs = root_rhs - total
                else:
                    level_rhs[above] = level_rhs[above] - total
    return root, root_rhs, diagonal, cross, level_rhs, factors, couplings, solved


@jax.jit
def _solve(system, damping):
    counts, widths = system.counts, system.widths
    depth = system.depth
    g = widths[0]
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)
    matrix = system.constraint_matrix
    n_rows = matrix.shape[0]
    columns = 1 + n_rows
    rhs = [jnp.concatenate([-system.root_gradient[:, None], matrix[:, :g].T], axis=1)]
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        rows = matrix[:, offsets[level] : offsets[level + 1]].T.reshape(count, width, n_rows)
        rhs.append(jnp.concatenate([-system.gradient[level][..., None], rows], axis=2))
    root, root_rhs, _, _, _, factors, couplings, solved = _fold(system, damping, rhs, 1)
    if g:
        root_factor = jnp.linalg.cholesky(root + damping * jnp.eye(g, dtype=dtype))
        half = jax.scipy.linalg.solve_triangular(root_factor, root_rhs, lower=True)
        top = jax.scipy.linalg.solve_triangular(root_factor, half, lower=True, trans="T")
    else:
        root_factor = None
        top = jnp.zeros((0, columns), dtype=dtype)
    steps = [top]
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        if width == 0:
            steps.append(jnp.zeros((count, 0, columns), dtype=dtype))
            continue
        right = solved[level]
        for above in range(level):
            if not widths[above]:
                continue
            if above == 0:
                right = right - jnp.einsum("nia,ak->nik", couplings[level][0], steps[0])
            else:
                above_steps = _ancestor_values(system, steps[above], level, above)
                right = right - jnp.einsum("nia,nak->nik", couplings[level][above], above_steps)
        steps.append(_upper_solve(factors[level], right))
    stacked = jnp.concatenate(
        [steps[0]] + [steps[level].reshape(-1, columns) for level in range(1, depth + 1)], axis=0
    )
    step = stacked[:, 0]
    if n_rows:
        spread = stacked[:, 1:]
        target = system.constraint_rhs - matrix @ system.point
        multipliers = jnp.linalg.solve(matrix @ spread, matrix @ step - target)
        step = step - spread @ multipliers
    parts = _split(system, step)
    halves: list = [None] * (depth + 1)
    q_squared = jnp.zeros((), dtype=dtype)
    for level in range(depth, 0, -1):
        if not widths[level]:
            continue
        right = parts[level]
        for deeper in range(level + 1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][level], halves[deeper])
            right = right - _node_sum(system, carried, deeper, level)
        halves[level] = _lower_solve(factors[level], right[..., None])[..., 0]
        q_squared = q_squared + jnp.sum(halves[level] * halves[level])
    if g:
        right = parts[0]
        for deeper in range(1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][0], halves[deeper])
            right = right - _node_sum(system, carried, deeper, 0)
        top_half = jax.scipy.linalg.solve_triangular(root_factor, right, lower=True)
        q_squared = q_squared + top_half @ top_half
    return step, q_squared


@jax.jit
def _scaled(system, scale, addition):
    # The system in the coordinates z = scale * s, with ``addition`` added to the
    # diagonal of the scaled matrix.
    inverse = 1.0 / scale
    parts = _split(system, inverse)
    added = _split(system, addition)
    shared = parts[0]
    diagonal: list = [None]
    cross: list = [None]
    gradient: list = [None]
    for level in range(1, system.depth + 1):
        own = parts[level]
        diagonal.append(
            system.diagonal[level] * own[:, :, None] * own[:, None, :]
            + jax.vmap(jnp.diag)(added[level])
        )
        gradient.append(system.gradient[level] * own)
        row = []
        for above in range(level):
            if above == 0:
                other = shared[None, None, :]
            else:
                other = _ancestor_values(system, parts[above], level, above)[:, None, :]
            row.append(system.cross[level][above] * own[:, :, None] * other)
        cross.append(tuple(row))
    return dataclasses.replace(
        system,
        root=system.root * shared[:, None] * shared[None, :] + jnp.diag(added[0]),
        root_gradient=system.root_gradient * shared,
        diagonal=tuple(diagonal),
        cross=tuple(cross),
        gradient=tuple(gradient),
        point=scale * system.point,
        constraint_matrix=system.constraint_matrix * inverse[None, :],
    )


@jax.jit
def _quadratic(system, delta):
    parts = _split(system, delta)
    total = parts[0] @ system.root @ parts[0]
    for level in range(1, system.depth + 1):
        own = parts[level]
        total = total + jnp.einsum("ni,nij,nj->", own, system.diagonal[level], own)
        for above in range(level):
            if not system.widths[above]:
                continue
            if above == 0:
                coupled = jnp.einsum("ni,nij,j->", own, system.cross[level][0], parts[0])
            else:
                other = _ancestor_values(system, parts[above], level, above)
                coupled = jnp.einsum("ni,nij,nj->", own, system.cross[level][above], other)
            total = total + 2.0 * coupled
    return total


@jax.jit
def _vectors(system):
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    diagonal = jnp.concatenate(
        [jnp.diagonal(system.root)]
        + [jnp.diagonal(entry, axis1=1, axis2=2).reshape(-1) for entry in system.diagonal[1:]]
    )
    return gradient, diagonal


# The phases of the multiplier search on the device, as ``_secular_search`` runs it.
_AT_ZERO, _PROBE, _BRACKET, _NEWTON, _FINAL, _NEARLY, _FOUND = range(7)
# The search's iteration limits: the bracket's doublings and the Newton iterations.
_BRACKETS = 80
_NEWTONS = 30


def _norm(vector: Any) -> Any:
    return jnp.sqrt(jnp.dot(vector, vector))


def _all_finite(vector: Any) -> Any:
    return jnp.all(jnp.isfinite(vector))


def _search(solve: Any, zero: tuple, radius: Any, eps: float, tiny: float) -> tuple:
    """``_secular_search`` on the device: every branch and limit of the host generator.

    ``solve(damping)`` gives ``(z, q_squared)``; ``zero`` is its answer at
    zero damping. Returns ``(multiplier, z, solves, failed)``: ``solves``
    counts ``zero`` and every solve made, ``failed`` is true when the bracket
    found no finite step inside the radius.
    """
    z0, q0 = zero
    dtype = z0.dtype
    scalar = functools.partial(jnp.asarray, dtype=dtype)
    radius = scalar(radius)
    tolerance = scalar(np.sqrt(eps)) * jnp.maximum(scalar(1.0), radius)
    probe = scalar(np.sqrt(eps))
    one = scalar(1.0)
    tiny = scalar(tiny)
    machine = scalar(float(np.finfo(float).eps))

    def at_zero(state):
        n = _norm(state["z"])
        found = _all_finite(state["z"]) & jnp.isfinite(state["q"]) & (n <= radius)
        factor_bound = ~jnp.isfinite(state["q"])
        return {
            **state,
            "phase": jnp.where(found, _FOUND, jnp.where(factor_bound, _PROBE, _BRACKET)),
            "multiplier": jnp.zeros((), dtype),
            "out": state["z"],
            "lo": jnp.zeros((), dtype),
            "lower_is_factor_bound": factor_bound,
            "hi": one,
            "need": ~found,
            "damping": jnp.where(factor_bound, probe, one),
        }

    def at_probe(state):
        n = _norm(state["z"])
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        found = valid & (n <= radius)
        return {
            **state,
            "phase": jnp.where(found, _FOUND, _BRACKET),
            "multiplier": probe,
            "out": state["z"],
            "lo": probe,
            "lower_is_factor_bound": ~valid,
            "hi": one,
            "need": ~found,
            "damping": one,
        }

    def at_bracket(state):
        n = _norm(state["z"])
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        inside = valid & (n <= radius)
        hi = jnp.where(inside, state["hi"], 2.0 * state["hi"] + 1.0)
        count = state["brackets"] + 1
        failed = ~inside & (count >= _BRACKETS)
        return {
            **state,
            "phase": jnp.where(inside, _NEWTON, jnp.where(failed, _FOUND, _BRACKET)),
            "failed": failed,
            "hi_step": jnp.where(inside, state["z"], state["hi_step"]),
            "current": jnp.where(inside, state["hi"], state["current"]),
            "lo": jnp.where(inside, state["lo"], state["hi"]),
            "lower_is_factor_bound": jnp.where(inside, state["lower_is_factor_bound"], ~valid),
            "hi": hi,
            "brackets": count,
            "need": ~inside & ~failed,
            "damping": hi,
        }

    def at_newton(state):
        n = _norm(state["z"])
        q = state["q"]
        current, lo, hi = state["current"], state["lo"], state["hi"]
        broken = ~jnp.isfinite(n) | ~jnp.isfinite(q)
        inside = ~broken & (n <= radius)
        close = inside & (radius - n <= tolerance)
        lo = jnp.where(broken | ~inside, current, lo)
        hi = jnp.where(inside, current, hi)
        bound = jnp.where(broken, True, jnp.where(inside, state["lower_is_factor_bound"], False))
        newton = current + (n * n / jnp.maximum(q, tiny)) * ((n - radius) / radius)
        stepped = jnp.where(
            jnp.isfinite(newton) & (lo < newton) & (newton < hi), newton, 0.5 * (lo + hi)
        )
        current = jnp.where(broken, 0.5 * (lo + hi), stepped)
        more = ~close & (state["newtons"] < _NEWTONS - 1)
        return {
            **state,
            "phase": jnp.where(more, _NEWTON, _FINAL),
            "hi_step": jnp.where(inside, state["z"], state["hi_step"]),
            "current": current,
            "lo": lo,
            "hi": hi,
            "lower_is_factor_bound": bound,
            "newtons": state["newtons"] + 1,
            "need": more,
            "damping": current,
        }

    def at_final(state):
        lo, hi = state["lo"], state["hi"]
        nearly = (
            state["lower_is_factor_bound"]
            & (lo > 0.0)
            & ((hi - lo) <= 1e-7 * jnp.maximum(one, jnp.abs(lo)))
        )
        return {
            **state,
            "phase": jnp.where(nearly, _NEARLY, _FOUND),
            "multiplier": hi,
            "out": state["hi_step"],
            "need": nearly,
            "damping": lo + jnp.maximum(1e-6 * jnp.abs(lo), machine),
        }

    def at_nearly(state):
        good = _all_finite(state["z"]) & jnp.isfinite(state["q"])
        return {
            **state,
            "phase": jnp.asarray(_FOUND),
            "multiplier": jnp.where(good, state["damping"], state["multiplier"]),
            "out": jnp.where(good, state["z"], state["out"]),
            "need": jnp.asarray(False),
        }

    branches = (at_zero, at_probe, at_bracket, at_newton, at_final, at_nearly)

    def body(state):
        state = jax.lax.switch(state["phase"], branches, state)
        z, q = jax.lax.cond(
            state["need"],
            lambda: solve(state["damping"]),
            lambda: (state["z"], state["q"]),
        )
        return {**state, "z": z, "q": q, "solves": state["solves"] + state["need"]}

    zeros = jnp.zeros_like(z0)
    state = {
        "phase": jnp.asarray(_AT_ZERO),
        "z": z0,
        "q": q0,
        "damping": jnp.zeros((), dtype),
        "need": jnp.asarray(False),
        "lo": jnp.zeros((), dtype),
        "hi": one,
        "current": one,
        "lower_is_factor_bound": jnp.asarray(False),
        "hi_step": zeros,
        "out": zeros,
        "multiplier": jnp.zeros((), dtype),
        "brackets": jnp.asarray(0),
        "newtons": jnp.asarray(0),
        "solves": jnp.asarray(1),
        "failed": jnp.asarray(False),
    }
    state = jax.lax.while_loop(lambda state: state["phase"] != _FOUND, body, state)
    return state["multiplier"], state["out"], state["solves"], state["failed"]


@functools.partial(jax.jit, static_argnames=("eps", "tiny"))
def _attempt(system, scale, addition, radius, *, eps, tiny):
    # One attempt's linear algebra: the scaled system, its solve at zero damping and the
    # decrease the model predicts for that full step, then the multiplier search of
    # radius ``radius`` and the decrease predicted for its step.
    scaled = _scaled(system, scale, addition)
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    inverse = 1.0 / scale

    def solve(damping):
        return _solve(scaled, damping)

    def decrease(z):
        step = inverse * z
        quadratic = _quadratic(system, step) + jnp.sum(
            jnp.where(addition != 0.0, addition * (z * z), 0.0)
        )
        return -(gradient @ step + 0.5 * quadratic)

    zero = solve(jnp.zeros((), scale.dtype))
    finite = _all_finite(zero[0]) & jnp.isfinite(zero[1])
    full = jnp.where(finite, decrease(jnp.where(finite, zero[0], 0.0)), jnp.nan)
    multiplier, scaled_step, solves, failed = _search(solve, zero, radius, eps, tiny)
    return (
        zero[0],
        zero[1],
        full,
        inverse * scaled_step,
        decrease(scaled_step),
        multiplier,
        solves,
        failed,
    )


@functools.partial(jax.jit, static_argnames=("level",))
def _inner_solve(system, damping, *, level):
    """Every member of ``level`` solves its own subtree with its own damping.

    The member's subtree (its block and every level below it) is eliminated
    from the experiments up, as ``_fold`` does, but nothing is folded above
    ``level``: the coordinates there are held. Member ``n`` solves
    ``(H_n + damping_n I) s_n = -g_n``, ``H_n`` and ``g_n`` its subtree's
    blocks, and ``q_n^2 = s_n^T (H_n + damping_n I)^-1 s_n`` comes from the same
    factors. Returns the steps in joint columns (zero above ``level``) and
    ``q^2`` per member.
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)

    def own_damping(at: int) -> Any:
        return damping if at == level else damping[system.ancestors[at][level]]

    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    rhs: list = [None] * (depth + 1)
    for at in range(level, depth + 1):
        rhs[at] = -system.gradient[at]
    factors: list = [None] * (depth + 1)
    couplings: list = [None] * (depth + 1)
    solved: list = [None] * (depth + 1)
    for at in range(depth, level, -1):
        width = widths[at]
        if width == 0:
            continue
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[at] + own_damping(at)[:, None, None] * jnp.eye(width, dtype=dtype)
        )
        factors[at] = factor
        row: list = [None] * at
        for above in range(level, at):
            if widths[above]:
                row[above] = _lower_solve(factor, cross[at][above])
        couplings[at] = row
        solved[at] = _lower_solve(factor, rhs[at][..., None])[..., 0]
        for above in range(level, at):
            if not widths[above]:
                continue
            for other in range(level, above + 1):
                if not widths[other]:
                    continue
                product = jnp.einsum("nia,nib->nab", row[above], row[other])
                total = _node_sum(system, product, at, above)
                if other == above:
                    diagonal[above] = diagonal[above] - total
                else:
                    cross[above][other] = cross[above][other] - total
            folded = jnp.einsum("nia,ni->na", row[above], solved[at])
            rhs[above] = rhs[above] - _node_sum(system, folded, at, above)
    steps: list = [None] * (depth + 1)
    if widths[level]:
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping[:, None, None] * jnp.eye(widths[level], dtype=dtype)
        )
        factors[level] = factor
        solved[level] = _lower_solve(factor, rhs[level][..., None])[..., 0]
        steps[level] = _upper_solve(factor, solved[level][..., None])[..., 0]
    else:
        steps[level] = jnp.zeros((counts[level - 1], 0), dtype=dtype)
    for at in range(level + 1, depth + 1):
        if not widths[at]:
            steps[at] = jnp.zeros((counts[at - 1], 0), dtype=dtype)
            continue
        right = solved[at]
        for above in range(level, at):
            if not widths[above]:
                continue
            above_steps = _ancestor_values(system, steps[above], at, above)
            right = right - jnp.einsum("nia,na->ni", couplings[at][above], above_steps)
        steps[at] = _upper_solve(factors[at], right[..., None])[..., 0]
    halves: list = [None] * (depth + 1)
    q_squared = jnp.zeros((counts[level - 1],), dtype=dtype)
    for at in range(depth, level - 1, -1):
        if not widths[at]:
            continue
        right = steps[at]
        for deeper in range(at + 1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][at], halves[deeper])
            right = right - _node_sum(system, carried, deeper, at)
        halves[at] = _lower_solve(factors[at], right[..., None])[..., 0]
        squares = jnp.sum(halves[at] * halves[at], axis=1)
        q_squared = q_squared + (squares if at == level else _node_sum(system, squares, at, level))
    step = jnp.concatenate(
        [jnp.zeros((offsets[level],), dtype=dtype)]
        + [steps[at].reshape(-1) for at in range(level, depth + 1)]
    )
    return step, q_squared


@functools.partial(jax.jit, static_argnames=("level",))
def _node_model(system, delta, *, level):
    """Per member of ``level``: ``g_n . delta_n`` and ``delta_n^T H_n delta_n`` over its subtree."""
    parts = _split(system, delta)
    count = system.counts[level - 1]
    dtype = system.root.dtype
    linear = jnp.zeros((count,), dtype=dtype)
    quadratic = jnp.zeros((count,), dtype=dtype)
    for at in range(level, system.depth + 1):
        if not system.widths[at]:
            continue
        own = parts[at]
        slope = jnp.einsum("ni,ni->n", own, system.gradient[at])
        curvature = jnp.einsum("ni,nij,nj->n", own, system.diagonal[at], own)
        for above in range(level, at):
            if not system.widths[above]:
                continue
            other = _ancestor_values(system, parts[above], at, above)
            curvature = curvature + 2.0 * jnp.einsum(
                "ni,nij,nj->n", own, system.cross[at][above], other
            )
        if at == level:
            linear = linear + slope
            quadratic = quadratic + curvature
        else:
            linear = linear + _node_sum(system, slope, at, level)
            quadratic = quadratic + _node_sum(system, curvature, at, level)
    return linear, quadratic


@functools.partial(jax.jit, static_argnames=("level",))
def _reduce(system, *, level):
    # The gradient is folded as a right-hand side column with the matrix, so the
    # reduced gradient is the outer gradient minus the inner gradients carried up
    # through the inner factors. The equality rows are not folded; the nested fit
    # refuses a row that addresses the eliminated coordinates.
    rhs = [-system.root_gradient[:, None]] + [-entry[..., None] for entry in system.gradient[1:]]
    root, root_rhs, diagonal, cross, level_rhs, _, _, _ = _fold(
        system, jnp.zeros((), dtype=system.root.dtype), rhs, level
    )
    prefix = _offsets(system.counts, system.widths)[level]
    return LevelSystem(
        root=root,
        root_gradient=-root_rhs[:, 0],
        diagonal=tuple(diagonal[:level]),
        cross=(None,) + tuple(tuple(row) for row in cross[1:level]),
        gradient=(None,) + tuple(-entry[..., 0] for entry in level_rhs[1:level]),
        point=system.point[:prefix],
        constraint_matrix=system.constraint_matrix[:, :prefix],
        constraint_rhs=system.constraint_rhs,
        cost=system.cost,
        ancestors=tuple(system.ancestors[:level]),
        descendants=tuple(system.descendants[:level]),
        present=tuple(system.present[:level]),
        fixed_mask=system.fixed_mask[:prefix],
        held_mask=tuple(system.held_mask[:level]),
        counts=tuple(system.counts[: level - 1]),
        widths=tuple(system.widths[:level]),
    )


@functools.partial(jax.jit, static_argnames=("level",))
def _response(system, scale, damping, outer_step, addition, *, level):
    # In the coordinates z = scale * s, with ``addition`` on the scaled diagonal: the
    # levels level .. D are eliminated at the damping with the gradient as right-hand
    # side, then solved from ``level`` down with the outer steps in place of the
    # ancestors' solved steps (no root solve). Also the inner gradient of the scaled
    # model with the outer step held, g_z,i + W_z^T z_o, in joint columns and per
    # inner level.
    scaled = _scaled(system, scale, addition)
    counts, widths = scaled.counts, scaled.widths
    depth = scaled.depth
    dtype = scaled.root.dtype
    offsets = _offsets(counts, widths)
    prefix = offsets[level]
    rhs = [-scaled.root_gradient[:, None]] + [-entry[..., None] for entry in scaled.gradient[1:]]
    _, _, _, _, _, factors, couplings, solved = _fold(scaled, damping, rhs, level)
    outer_z = scale[:prefix] * outer_step
    parts = _split(scaled, jnp.concatenate([outer_z, jnp.zeros((offsets[-1] - prefix,), dtype)]))
    steps: list = [part[..., None] for part in parts[:level]] + [None] * (depth + 1 - level)
    for at in range(level, depth + 1):
        count, width = counts[at - 1], widths[at]
        if width == 0:
            steps[at] = jnp.zeros((count, 0, 1), dtype=dtype)
            continue
        right = solved[at]
        for above in range(at):
            if not widths[above]:
                continue
            if above == 0:
                right = right - jnp.einsum("nia,ak->nik", couplings[at][0], steps[0])
            else:
                above_steps = _ancestor_values(scaled, steps[above], at, above)
                right = right - jnp.einsum("nia,nak->nik", couplings[at][above], above_steps)
        steps[at] = _upper_solve(factors[at], right)
    inner = jnp.concatenate([steps[at].reshape(-1) for at in range(level, depth + 1)])
    shifted = []
    levels = []
    for at in range(level, depth + 1):
        right = scaled.gradient[at]
        for above in range(level):
            if not widths[above] or not widths[at]:
                continue
            if above == 0:
                right = right + jnp.einsum("nia,a->ni", scaled.cross[at][0], parts[0])
            else:
                above_parts = _ancestor_values(scaled, parts[above], at, above)
                right = right + jnp.einsum("nia,na->ni", scaled.cross[at][above], above_parts)
        shifted.append(right.reshape(-1))
        levels.append(right)
    outer_zeros = jnp.zeros((prefix,), dtype)
    return (
        jnp.concatenate([outer_step, inner / scale[prefix:]]),
        jnp.concatenate([outer_zeros, inner]),
        jnp.concatenate([outer_zeros] + shifted),
        tuple(levels),
    )


# ---------------------------------------------------------------------------
# Host entry points
# ---------------------------------------------------------------------------


def _dtype(system: LevelSystem) -> np.dtype:
    return np.dtype(system.root.dtype)


def _addition(system: LevelSystem, addition: Any) -> np.ndarray:
    """The diagonal addition as an ``(n,)`` host array of the system's dtype; zeros for ``None``."""
    if addition is None:
        return np.zeros((system.size,), dtype=_dtype(system))
    return np.asarray(addition, dtype=_dtype(system))


def _added_quadratic(addition: Any, scaled_step: np.ndarray) -> float:
    """``z^T diag(addition) z`` for a scaled step ``z``; 0 without an addition."""
    if addition is None:
        return 0.0
    with np.errstate(all="ignore"):
        return float(np.asarray(addition, dtype=np.float64) @ (scaled_step * scaled_step))


def scaled_system(system: LevelSystem, scale: Any, addition: Any = None) -> LevelSystem:
    """The system in the coordinates ``z = scale * s``, ``addition`` added to its diagonal.

    Its matrix is ``D H D + diag(addition)`` and its gradient ``D g``, with
    ``D = diag(1 / scale)``; with a unit scale it is the system with the
    addition on the diagonal of its matrix.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        addition: Optional ``(n,)``, nonnegative; ``None`` adds nothing.

    Returns:
        A [LevelSystem][diff_soup.inference.hierarchical._level_elimination.LevelSystem]
        of the same shape.
    """
    return _scaled(system, np.asarray(scale, dtype=_dtype(system)), _addition(system, addition))


def solve_damped(system: LevelSystem, damping: Any) -> tuple:
    """Solve ``(H + damping I) s = -g`` along the tree, under the system's equality rows.

    Args:
        system: The system.
        damping: The nonnegative damping ``lambda``.

    Returns:
        ``(step, q_squared)``: the step ``(n,)`` in joint columns and
        ``s^T (H + damping I)^-1 s``. A matrix that is not positive definite at
        this damping gives non-finite values.
    """
    return _solve(system, np.asarray(damping, dtype=_dtype(system)))


def solve_scaled(system: LevelSystem, scale: Any, damping: Any, addition: Any = None) -> tuple:
    """One damped solve in the coordinates ``z = scale * s``, on the host.

    The matrix is the scaled one with ``addition`` on its diagonal
    (``scaled_system``).

    Returns:
        ``(z, q_squared)`` as a host array and a float.
    """
    scaled = scaled_system(system, scale, addition)
    step, q_squared = _solve(scaled, np.asarray(damping, dtype=_dtype(system)))
    return np.asarray(step), float(q_squared)


def level_quadratic(system: LevelSystem, delta: Any) -> float:
    """``delta^T H delta`` from the blocks, without a dense matrix.

    Args:
        system: The system.
        delta: ``(n,)``, a step in joint columns.

    Returns:
        The quadratic form as a float.
    """
    return float(_quadratic(system, np.asarray(delta, dtype=_dtype(system))))


def level_vectors(system: LevelSystem) -> tuple:
    """The gradient and the diagonal of the matrix in joint columns, as host arrays."""
    gradient, diagonal = _vectors(system)
    return np.asarray(gradient), np.asarray(diagonal)


def level_secular_step(
    system: LevelSystem, scale: Any, radius: Any, *, first: Any = None, addition: Any = None
) -> tuple:
    """The trust-region step of radius ``radius`` in the scaled norm ``||scale * s||``.

    The multiplier ``lambda`` is zero when the Gauss-Newton step fits;
    otherwise it is found by a safeguarded Newton iteration on the secular
    equation ``||z(lambda)|| = radius`` above the Cholesky bound, each trial a
    damped solve along the tree in the scaled coordinates.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        first: Optional ``(z, q_squared)`` of the solve at ``lambda = 0`` in
            these coordinates, already made; it counts as this step's first
            solve.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``); the model whose step is solved
            and whose decrease is predicted carries it.

    Returns:
        ``(step, predicted, multiplier, solves)``: the step in joint columns,
        the decrease the model predicts, the multiplier, and the number of
        damped solves made.

    Raises:
        ValueError: If no finite step inside the radius is found.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    inverse_scale = 1.0 / scale
    eps = float(np.finfo(dtype).eps)
    search = _secular_search(float(radius), eps, float(np.finfo(float).tiny))
    gradient, _ = level_vectors(system)
    # Every damping the search asks for is one damped solve; the solve at zero damping
    # given as ``first`` answers its first request.
    damping = next(search)
    answer = None if first is None else (np.asarray(first[0]), float(first[1]))
    solves = 0
    try:
        while True:
            if answer is None:
                z, q_squared = _solve(scaled, np.asarray(damping, dtype=dtype))
                answer = np.asarray(z), float(q_squared)
            solves += 1
            damping, answer = search.send(answer), None
    except StopIteration as found:
        multiplier, scaled_step = found.value
    with np.errstate(all="ignore"):
        step = inverse_scale * scaled_step
    quadratic = level_quadratic(system, step) + _added_quadratic(addition, scaled_step)
    predicted = -(float(gradient @ step) + 0.5 * quadratic)
    return step, predicted, float(multiplier), solves


class AttemptSolve(NamedTuple):
    """The linear algebra of one attempt of the joint fit, from one program on the device.

    Indexed as a pair it is the solve at zero damping ``(z, q_squared)`` that
    ``solve_scaled`` gives, so it stands wherever that pair is taken.

    Attributes:
        z: The full step at zero damping in the coordinates ``z = scale * s``.
        q_squared: Its inverse-norm ``z^T (H_z)^-1 z``.
        decrease: The decrease the model predicts for the full step ``z /
            scale``, in cost units, not divided by anything; ``nan`` when the
            solve at zero damping is not finite.
        step: The step of radius ``radius`` in joint columns, as
            ``level_secular_step`` gives it.
        predicted: The decrease the model predicts for ``step``.
        multiplier: The multiplier of ``step``.
        solves: The damped solves the search made, the one at zero damping
            included.
        radius: The radius ``step`` was solved for.
        scale: The scale of the coordinates.
        addition: The diagonal added to the scaled matrix, or ``None``.
        failed: Whether the search found no finite step inside the radius
            (a point off its equality rows whose restoring step exceeds the
            radius, or a scaling no damping cures). ``z``, ``q_squared`` and
            ``decrease`` stand either way; ``step``, ``predicted`` and
            ``multiplier`` are then zero, and the loop that takes the step
            raises there, as ``level_secular_step`` does, so a fit that ends
            before it takes a step is not ended by the search.
    """

    z: np.ndarray
    q_squared: float
    decrease: float
    step: np.ndarray
    predicted: float
    multiplier: float
    solves: int
    radius: float
    scale: np.ndarray
    addition: Any
    failed: bool = False

    def answers(self, scale: Any, radius: Any, addition: Any = None) -> bool:
        """Whether this attempt was solved for ``scale``, ``radius`` and ``addition``."""
        if float(radius) != self.radius or not np.array_equal(np.asarray(scale), self.scale):
            return False
        if addition is None or self.addition is None:
            return addition is None and self.addition is None
        return np.array_equal(np.asarray(addition), self.addition)


def attempt_solve(
    system: LevelSystem, scale: Any, radius: Any, addition: Any = None
) -> AttemptSolve:
    """One attempt's solve at zero damping, its predicted decrease and its trust-region step.

    The same numbers as ``solve_scaled(system, scale, 0.0, addition)``, the
    full step's predicted decrease, and ``level_secular_step(system, scale,
    radius, first=..., addition=addition)``, from one compiled program per
    system shape and one transfer to the host: the multiplier search runs on
    the device, branch for branch as on the host. The numbers agree with that
    chain to rounding.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix.

    Returns:
        The [AttemptSolve][diff_soup.inference.hierarchical._level_elimination.AttemptSolve].
        A search that finds no finite step inside the radius sets its
        ``failed``; nothing is raised here, so the solve at zero damping and
        its predicted decrease serve the tests a loop makes before it takes a
        step.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    added = _addition(system, addition)
    outputs = _attempt(
        system,
        scale,
        added,
        np.asarray(radius, dtype=dtype),
        eps=float(np.finfo(dtype).eps),
        tiny=float(np.finfo(float).tiny),
    )
    z, q_squared, decrease, step, predicted, multiplier, solves, failed = jax.device_get(outputs)
    return AttemptSolve(
        z=np.asarray(z),
        q_squared=float(q_squared),
        decrease=float(decrease),
        step=np.array(step),
        predicted=float(predicted),
        multiplier=float(multiplier),
        solves=int(solves),
        radius=float(radius),
        scale=scale,
        addition=None if addition is None else added,
        failed=bool(failed),
    )


def node_of_columns(system: LevelSystem, level: int) -> np.ndarray:
    """``(n,)``: the member of ``level`` whose subtree holds each joint column, ``-1`` above it."""
    counts, widths = system.counts, system.widths
    offsets = _offsets(counts, widths)
    owners = np.full((offsets[-1],), -1, dtype=np.int64)
    for at in range(level, system.depth + 1):
        if at == level:
            member = np.arange(counts[level - 1], dtype=np.int64)
        else:
            member = np.asarray(system.ancestors[at][level], dtype=np.int64)
        owners[offsets[at] : offsets[at + 1]] = np.repeat(member, widths[at])
    return owners


def _secular_search(radius: float, eps: float, tiny: float):
    """The multiplier search of one trust-region problem of radius ``radius``, as a generator.

    It yields the damping of the next solve and receives ``(z, q_squared)``,
    the problem's scaled step at that damping and its inverse-norm; it returns
    ``(multiplier, z)``. ``level_secular_step`` runs one for the whole system,
    ``_lockstep_search`` one per member of a level.
    """

    def norm(vector: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return float(np.linalg.norm(vector))

    def finite(vector: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(vector)))

    scaled_step, zero_q_squared = yield 0.0
    if finite(scaled_step) and np.isfinite(zero_q_squared) and norm(scaled_step) <= radius:
        return 0.0, scaled_step

    lo = 0.0
    lower_is_factor_bound = not np.isfinite(zero_q_squared)
    if lower_is_factor_bound:
        probe = float(np.sqrt(eps))
        probe_step, probe_q_squared = yield probe
        probe_norm = norm(probe_step)
        probe_valid = np.isfinite(probe_norm) and np.isfinite(probe_q_squared)
        if probe_valid and probe_norm <= radius:
            return probe, probe_step
        lo = probe
        lower_is_factor_bound = not probe_valid
    hi = 1.0
    for _ in range(80):
        candidate, q_squared = yield hi
        candidate_norm = norm(candidate)
        valid_factor = np.isfinite(candidate_norm) and np.isfinite(q_squared)
        if valid_factor and candidate_norm <= radius:
            hi_step = candidate
            break
        lo = hi
        lower_is_factor_bound = not valid_factor
        hi = 2.0 * hi + 1.0
    else:
        raise ValueError(
            "TrustRegion could not bracket a finite Gauss-Newton trust-region step; improve "
            "parameter scaling."
        )

    current = hi
    tolerance = float(np.sqrt(eps)) * max(1.0, radius)
    for iteration in range(30):
        candidate_norm = norm(candidate)
        if not np.isfinite(candidate_norm) or not np.isfinite(q_squared):
            lo = current
            lower_is_factor_bound = True
            current = 0.5 * (lo + hi)
            if iteration < 29:
                candidate, q_squared = yield current
            continue
        elif candidate_norm <= radius:
            hi = current
            hi_step = candidate
            if radius - candidate_norm <= tolerance:
                break
        else:
            lo = current
            lower_is_factor_bound = False
        newton = current + (candidate_norm * candidate_norm / max(q_squared, tiny)) * (
            (candidate_norm - radius) / radius
        )
        current = newton if np.isfinite(newton) and lo < newton < hi else 0.5 * (lo + hi)
        if iteration < 29:
            candidate, q_squared = yield current

    multiplier = hi
    scaled_step = hi_step
    if lower_is_factor_bound and lo > 0.0 and (hi - lo) <= 1e-7 * max(1.0, abs(lo)):
        nearly_exact = lo + max(1e-6 * abs(lo), float(np.finfo(float).eps))
        nearly_exact_step, nearly_exact_q_squared = yield nearly_exact
        if finite(nearly_exact_step) and np.isfinite(nearly_exact_q_squared):
            multiplier = nearly_exact
            scaled_step = nearly_exact_step
    return multiplier, scaled_step


def inner_solve_scaled(
    system: LevelSystem, scale: Any, level: int, *, addition: Any = None
) -> tuple:
    """Every member of ``level`` solves its subtree at zero damping, in ``z = scale * s``.

    The coordinates above ``level`` are held. This is the solve the members'
    multiplier searches of ``inner_secular_step`` make first, for every
    member at once.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        level: The level of the inner problems, ``1 .. D``.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).

    Returns:
        ``(z, q_squared)`` as host arrays: the scaled steps ``(n,)`` in joint
        columns (zero above ``level``) and ``z_n^T H_n^-1 z_n`` per member
        ``(N_level,)``, ``H_n`` the member's scaled subtree matrix. A member
        whose matrix has no finite factor gets non-finite values.
    """
    dtype = _dtype(system)
    scaled = scaled_system(system, scale, addition)
    count = system.counts[int(level) - 1]
    step, q_squared = _inner_solve(scaled, np.zeros((count,), dtype=dtype), level=int(level))
    return np.asarray(step), np.asarray(q_squared)


def _lockstep_search(
    scaled: LevelSystem, level: int, radii: Any, active: Any, first: Any, *, hold: bool = False
) -> tuple:
    """The multiplier searches of the active members of ``level`` on a scaled system.

    Every active member runs ``_secular_search`` with its own radius on its
    subtree of ``scaled``, the coordinates above ``level`` held. The members
    search in lockstep: every round is one damped solve of all subtrees at
    once, each with its own damping. ``first`` is an optional
    ``(z, q_squared)`` of the solve at zero damping, already made; it answers
    every search's first request and counts as one solve. With ``hold``, a
    member whose search finds no finite step inside its radius takes a zero
    step and a ``nan`` multiplier instead of raising.

    Returns:
        ``(multipliers, scaled_step, solves)``: the multipliers
        ``(N_level,)``, the scaled steps ``(n,)`` in joint columns (zero above
        ``level`` and on inactive members), and the number of solves made.
    """
    dtype = _dtype(scaled)
    owners = node_of_columns(scaled, level)
    count = scaled.counts[level - 1]
    columns = [np.flatnonzero(owners == member) for member in range(count)]
    eps = float(np.finfo(dtype).eps)
    tiny = float(np.finfo(float).tiny)
    searches = {
        int(member): _secular_search(float(radii[member]), eps, tiny)
        for member in np.flatnonzero(active)
    }
    requests = {member: next(search) for member, search in searches.items()}
    multipliers = np.zeros((count,), dtype=np.float64)
    scaled_step = np.zeros((owners.shape[0],), dtype=dtype)
    solves = 0

    def answer(requests: dict, step: np.ndarray, q_squared: np.ndarray) -> dict:
        """Send every requesting member its solve; the requests of the next round."""
        following = {}
        for member in requests:
            try:
                following[member] = searches[member].send(
                    (step[columns[member]], float(q_squared[member]))
                )
            except StopIteration as found:
                multiplier, member_step = found.value
                multipliers[member] = multiplier
                scaled_step[columns[member]] = member_step
            except ValueError:
                if not hold:
                    raise
                multipliers[member] = np.nan
        return following

    if first is not None and requests:
        # Every search asks for zero damping first; the solve already made answers it.
        solves += 1
        requests = answer(requests, np.asarray(first[0]), np.asarray(first[1]))
    while requests:
        damping = np.ones((count,), dtype=dtype)
        for member, value in requests.items():
            damping[member] = value
        step, q_squared = _inner_solve(scaled, damping, level=level)
        solves += 1
        requests = answer(requests, np.asarray(step), np.asarray(q_squared))
    return multipliers, scaled_step, solves


def inner_secular_step(
    system: LevelSystem,
    scale: Any,
    radii: Any,
    level: int,
    active: Any,
    *,
    first: Any = None,
    addition: Any = None,
    hold: bool = False,
) -> tuple:
    """The trust-region step of every active member of ``level`` over its own subtree.

    Each active member runs the multiplier search of ``level_secular_step``
    with its own radius in the scaled norm ``||scale * s_n||``, the
    coordinates above ``level`` held. The members search in lockstep: every
    round is one damped solve of all subtrees at once, each with its own
    multiplier.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radii: ``(N_level,)``, the radius of every member.
        level: The level of the inner problems, ``1 .. D``.
        active: ``(N_level,)`` bool; an inactive member takes a zero step.
        first: Optional ``(z, q_squared)`` of ``inner_solve_scaled`` with
            this scale and level, already made: every active member's search
            takes its solve at zero damping from it, and it counts as this
            step's first solve.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``); every member's model carries it.
        hold: A member for which no finite step inside its radius is found
            takes a zero step, a zero predicted decrease and a ``nan``
            multiplier instead of raising; the other members are unchanged.

    Returns:
        ``(step, predicted, multipliers, solves)``: the step ``(n,)`` in joint
        columns (zero above ``level`` and on inactive members), the decrease
        each member's model predicts ``(N_level,)`` (zero when inactive), the
        multipliers ``(N_level,)``, and the number of solves made.

    Raises:
        ValueError: If no finite step inside its radius is found for a member
            and ``hold`` is false.
    """
    dtype = _dtype(system)
    level = int(level)
    scale = np.asarray(scale, dtype=dtype)
    radii = np.asarray(radii, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    scaled = scaled_system(system, scale, addition)
    multipliers, scaled_step, solves = _lockstep_search(
        scaled, level, radii, active, first, hold=hold
    )
    with np.errstate(all="ignore"):
        delta = scaled_step / scale
    predicted = inner_predicted(system, delta, level)
    if addition is not None:
        predicted -= 0.5 * inner_added(system, addition, scaled_step, level)
    predicted[~active | np.isnan(multipliers)] = 0.0
    return delta, predicted, multipliers, solves


def inner_predicted(system: LevelSystem, delta: Any, level: int) -> np.ndarray:
    """``-(g_n . delta_n + delta_n^T H_n delta_n / 2)`` for every member ``n`` of ``level``.

    ``H_n`` and ``g_n`` are the blocks and the gradient of the member's subtree;
    the columns of ``delta`` above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    linear, quadratic = _node_model(system, delta, level=int(level))
    return -(np.asarray(linear, dtype=np.float64) + 0.5 * np.asarray(quadratic, dtype=np.float64))


def model_split(system: LevelSystem, delta: Any, level: int) -> tuple:
    """The decrease the Gauss-Newton model predicts for ``delta``, split by member of ``level``.

    Returns ``(members, rest)``: ``members`` ``(N_level,)``, for every member ``n``
    its subtree's terms, ``-(g_n . delta_n + delta_n^T H_n delta_n / 2 +
    delta_n^T C_n delta_a)`` with ``C_n`` the coupling of the subtree to the
    coordinates above ``level`` and ``delta_a`` their step; ``rest`` the terms
    of the coordinates above ``level`` alone. ``members.sum() + rest`` is the
    model's decrease ``-(g . delta + delta^T H delta / 2)``.
    """
    level = int(level)
    delta = np.asarray(delta, dtype=_dtype(system))
    parts = _split(system, delta)
    count = system.counts[level - 1]
    coupling = np.zeros((count,), dtype=np.float64)
    full = float(parts[0] @ np.asarray(system.root_gradient)) + 0.5 * float(
        parts[0] @ np.asarray(system.root) @ parts[0]
    )
    for at in range(1, system.depth + 1):
        if not system.widths[at]:
            continue
        own = np.asarray(parts[at])
        terms = np.einsum("ni,ni->n", own, np.asarray(system.gradient[at])) + 0.5 * np.einsum(
            "ni,nij,nj->n", own, np.asarray(system.diagonal[at]), own
        )
        for above in range(0, at):
            if not system.widths[above]:
                continue
            if above == 0:
                other = np.broadcast_to(np.asarray(parts[0]), (own.shape[0], system.widths[0]))
            else:
                other = np.asarray(_ancestor_values(system, parts[above], at, above))
            pair = np.einsum("ni,nij,nj->n", own, np.asarray(system.cross[at][above]), other)
            terms = terms + pair
            if at >= level and above < level:
                coupling += np.asarray(
                    pair if at == level else _node_sum(system, pair, at, level), dtype=np.float64
                )
        full += float(np.sum(terms))
    members = inner_predicted(system, delta, level) - coupling
    return members, float(-full - np.sum(members))


def nonfinite_members(system: LevelSystem, level: int) -> np.ndarray:
    """``(N_level,)`` bool: the members of ``level`` whose subtree's blocks or gradient hold a non-finite entry."""
    level = int(level)
    count = system.counts[level - 1]
    broken = np.zeros((count,), dtype=bool)
    for at in range(level, system.depth + 1):
        if not system.widths[at]:
            continue
        rows = ~np.isfinite(np.asarray(system.gradient[at])).all(axis=1)
        rows |= ~np.isfinite(np.asarray(system.diagonal[at])).all(axis=(1, 2))
        for above in range(0, at):
            if system.widths[above] and system.cross[at][above] is not None:
                rows |= ~np.isfinite(np.asarray(system.cross[at][above])).all(axis=(1, 2))
        member = np.arange(count) if at == level else np.asarray(system.ancestors[at][level])
        np.logical_or.at(broken, member, rows)
    return broken


def inner_quadratic(system: LevelSystem, delta: Any, level: int) -> np.ndarray:
    """``delta_n^T H_n delta_n`` for every member ``n`` of ``level``, ``(N_level,)``.

    ``H_n`` is the matrix of the member's subtree; the columns of ``delta``
    above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    _, quadratic = _node_model(system, delta, level=int(level))
    return np.asarray(quadratic, dtype=np.float64)


def inner_added(system: LevelSystem, addition: Any, scaled_step: Any, level: int) -> np.ndarray:
    """``z_n^T diag(addition_n) z_n`` for every member ``n`` of ``level``, ``(N_level,)``.

    ``z`` is a scaled step in joint columns; the columns above ``level`` do
    not enter.
    """
    level = int(level)
    owners = node_of_columns(system, level)
    below = owners >= 0
    with np.errstate(all="ignore"):
        terms = np.asarray(addition, dtype=np.float64) * np.square(
            np.asarray(scaled_step, dtype=np.float64)
        )
    return np.bincount(owners[below], weights=terms[below], minlength=system.counts[level - 1])


def inner_singular(system: LevelSystem, level: int) -> np.ndarray:
    """``(N_level,)``: the members whose subtree's Gauss-Newton matrix has no finite factor."""
    level = int(level)
    count = system.counts[level - 1]
    step, q_squared = _inner_solve(system, np.zeros((count,), dtype=_dtype(system)), level=level)
    owners = node_of_columns(system, level)
    step = np.asarray(step)
    singular = ~np.isfinite(np.asarray(q_squared))
    for member in range(count):
        if not np.all(np.isfinite(step[owners == member])):
            singular[member] = True
    return singular


def inner_programs(system: LevelSystem, level: int) -> None:
    """Compile the programs of the inner problems of ``level`` before a fit's loop.

    These are the members' damped solves, their models, and the solve of the
    members' bounded response (``inner_bounded_response``).
    """
    level = int(level)
    count = system.counts[level - 1]
    dtype = _dtype(system)
    ones = np.ones((system.size,), dtype=dtype)
    scaled = scaled_system(system, ones)
    _inner_solve(scaled, np.ones((count,), dtype=dtype), level=level)
    _node_model(system, np.zeros((system.size,), dtype=dtype), level=level)
    prefix = _offsets(system.counts, system.widths)[level]
    _, _, _, shifted = _response(
        system,
        ones,
        np.asarray(0.0, dtype=dtype),
        np.zeros((prefix,), dtype=dtype),
        np.zeros((system.size,), dtype=dtype),
        level=level,
    )
    _inner_solve(
        _shifted_system(scaled, shifted, level), np.ones((count,), dtype=dtype), level=level
    )


def _response_level(system: LevelSystem, outer_step: np.ndarray) -> int:
    """The cut whose outer columns ``outer_step`` covers; refused otherwise."""
    offsets = _offsets(system.counts, system.widths)
    levels = [
        level
        for level in range(1, system.depth + 1)
        if outer_step.ndim == 1 and offsets[level] == outer_step.shape[0]
    ]
    if not levels:
        raise ValueError(
            "inner_response takes an outer step over the columns above a level of the system, "
            f"of length {', '.join(str(offsets[level]) for level in range(1, system.depth + 1))}; "
            f"got shape {outer_step.shape}."
        )
    return levels[0]


def inner_response_parts(
    system: LevelSystem, scale: Any, damping: Any, outer_step: Any, *, addition: Any = None
) -> tuple:
    """``inner_response`` with the scaled inner step and the inner gradient it answers.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).

    Returns:
        ``(step, z, shifted)`` as host arrays ``(n,)``: the step of
        ``inner_response``; its inner part in the coordinates
        ``z = scale * s`` (zero on the outer columns); and the inner gradient
        of the scaled model with the outer step held, ``g_{z,i} + W_z^T z_o``
        (zero on the outer columns).

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system.
    """
    dtype = _dtype(system)
    outer_step = np.asarray(outer_step, dtype=dtype)
    step, z, shifted, _ = _response(
        system,
        np.asarray(scale, dtype=dtype),
        np.asarray(damping, dtype=dtype),
        outer_step,
        _addition(system, addition),
        level=_response_level(system, outer_step),
    )
    return np.asarray(step), np.asarray(z), np.asarray(shifted)


def _shifted_system(scaled: LevelSystem, shifted: tuple, level: int) -> LevelSystem:
    """``scaled`` with the gradients of the levels ``level .. D`` replaced by ``shifted``."""
    return dataclasses.replace(scaled, gradient=tuple(scaled.gradient[:level]) + tuple(shifted))


def inner_bounded_response(
    system: LevelSystem,
    scale: Any,
    damping: Any,
    outer_step: Any,
    radii: Any,
    *,
    addition: Any = None,
) -> tuple:
    """``inner_response_parts`` with every member's response bounded by its radius.

    The members are the nodes of the level whose columns follow those of
    ``outer_step``. A member whose response has a scaled norm ``||z_n||``
    above its radius ``r_n`` responds instead with its own trust-region step
    of radius ``r_n`` at the gradient the response answers: ``z_n`` solves
    ``(V_{z,n} + A_n + lambda_n I) z_n = -(g_{z,n} + W_{z,n}^T z_o)`` over the
    member's subtree, with the multiplier ``lambda_n`` at least ``damping``,
    found by the multiplier search of ``inner_secular_step`` so that
    ``||z_n||`` is ``r_n`` to within ``sqrt(eps) max(1, r_n)`` below it. A
    member within its radius responds as without radii, and a member of
    radius 0 with a nonzero response responds with zero. Fixed inner
    coordinates and held nodes respond with zero, the outer part is
    ``outer_step``.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns, ``p`` the number
            of columns above a level.
        radii: ``(N_level,)``, every member's radius in the scaled norm, at
            least 0.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).

    Returns:
        ``(step, z, shifted, norms, capped, solves)``: ``step``, ``z`` and
        ``shifted`` as ``inner_response_parts`` returns them, with the parts of
        the bounded members recomputed; ``norms`` ``(N_level,)``, every
        member's scaled response norm before the bound; ``capped``
        ``(N_level,)`` bool, the members whose response was above its radius
        (a response that is not finite counts as above); and the number of
        damped solves the recomputation made.

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system, if ``radii`` does not hold one finite
            radius of at least 0 per member of that level, or if no finite step
            within its radius is found for a member.
    """
    dtype = _dtype(system)
    outer_step = np.asarray(outer_step, dtype=dtype)
    level = _response_level(system, outer_step)
    count = system.counts[level - 1]
    radii = np.asarray(radii, dtype=np.float64)
    if radii.shape != (count,) or not np.all(np.isfinite(radii) & (radii >= 0.0)):
        raise ValueError(
            "inner_bounded_response takes one finite radius of at least 0 per member of level "
            f"{level}, {count} in all; got {radii!r}."
        )
    scale = np.asarray(scale, dtype=dtype)
    added = _addition(system, addition)
    multiplier = np.asarray(damping, dtype=dtype)
    step, z, shifted, levels = _response(system, scale, multiplier, outer_step, added, level=level)
    step, z, shifted = np.array(step), np.array(z), np.asarray(shifted)
    owners = node_of_columns(system, level)
    below = owners >= 0
    with np.errstate(all="ignore"):
        norms = np.sqrt(np.bincount(owners[below], weights=z[below] ** 2, minlength=count))
    capped = ~(norms <= radii)
    solves = 0
    if capped.any():
        # The bounded members' searches on their subtrees, the outer step's damping on
        # the inner diagonal and the gradient the response answers in place of theirs.
        prefix = _offsets(system.counts, system.widths)[level]
        inner_added = np.array(added, copy=True)
        inner_added[prefix:] += multiplier
        scaled = _shifted_system(_scaled(system, scale, inner_added), levels, level)
        searched = capped & (radii > 0.0)
        scaled_step = np.zeros_like(z)
        if searched.any():
            _, scaled_step, solves = _lockstep_search(scaled, level, radii, searched, None)
        columns = below & capped[np.maximum(owners, 0)]
        z[columns] = scaled_step[columns]
        with np.errstate(all="ignore"):
            step[columns] = scaled_step[columns] / scale[columns]
    return step, z, shifted, norms, capped, solves


def inner_response(
    system: LevelSystem,
    scale: Any,
    damping: Any,
    outer_step: Any,
    *,
    addition: Any = None,
    radii: Any = None,
) -> np.ndarray:
    """The step whose outer part is ``outer_step`` and whose inner part responds to it.

    The outer step covers the columns above a level of the system; that level
    and the levels below it are the inner ones. In the coordinates
    ``z = scale * s`` the inner part minimizes the damped quadratic model with
    the outer part held:
    ``z_i = -(V_z + A + damping I)^-1 (g_{z,i} + W_z^T z_o)``, with ``V_z`` the
    inner blocks, ``A`` the diagonal ``addition`` (zero without one), ``W_z``
    their couplings to the outer blocks and ``g_{z,i}`` the inner gradient of
    the scaled system. It is computed along the tree: the inner levels are
    eliminated from the experiments up at that damping, then solved from the
    cut down with the outer step in place of the solved ancestors. Without an
    addition and at ``damping = 0`` it is the inner part of the joint model's
    minimizer at the fixed outer step, ``-V^-1 (g_i + W^T s_o)``, whatever the
    scale. Fixed inner coordinates and held nodes respond with zero.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns, ``p`` the number
            of columns above a level.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).
        radii: Optional ``(N_level,)``, one radius per member of the inner
            level, each at least 0: a member whose response has a scaled norm
            above its radius responds with its own step of that radius
            instead (``inner_bounded_response``). ``None`` bounds no member.

    Returns:
        ``(n,)`` as a host array: ``outer_step`` followed by the inner response.

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system, or ``radii`` does not hold one finite
            radius of at least 0 per member of the inner level.
    """
    if radii is not None:
        bounded = inner_bounded_response(
            system, scale, damping, outer_step, radii, addition=addition
        )
        return bounded[0]
    step, _, _ = inner_response_parts(system, scale, damping, outer_step, addition=addition)
    return step


def reduce_to(system: LevelSystem, level: int) -> LevelSystem:
    """The system over the levels above ``level``, the levels ``level .. D`` eliminated.

    Its blocks are the Schur complement of the levels ``level .. D`` into the
    blocks above them (no damping). Its gradient is the outer part of the
    gradient with the inner gradients folded in through the same elimination,
    ``g_o - W V^-1 g_i`` (``V`` the eliminated blocks, ``W`` their couplings to
    the outer blocks): the gradient, at a zero outer step, of the quadratic
    model minimized over the eliminated coordinates. Where the inner gradients
    are zero it is the outer part of the gradient. It keeps the outer part of
    the point, the fixed outer coordinates and the outer columns of the equality
    rows, which is exact when no row addresses the eliminated coordinates.

    Args:
        system: The system.
        level: The cut, ``1 .. D``.

    Returns:
        A [LevelSystem][diff_soup.inference.hierarchical._level_elimination.LevelSystem]
        of depth ``level - 1`` (the root alone for ``level = 1``).

    Raises:
        ValueError: If ``level`` is not a level of the system.
    """
    level = int(level)
    if not 1 <= level <= system.depth:
        raise ValueError(f"reduce_to cuts at a level 1 .. {system.depth}; got {level}.")
    return _reduce(system, level=level)


def level_dense(system: LevelSystem) -> tuple:
    """The dense matrix and gradient of a system, for small checks and reports.

    Of a system that ``reduce_to`` returned, these are the Schur complement and
    the outer gradient with the inner gradients folded in.

    Returns:
        ``(H, g)`` as host arrays over the system's joint columns.
    """
    counts, widths = system.counts, system.widths
    offsets = _offsets(counts, widths)
    size = offsets[-1]
    dtype = _dtype(system)
    matrix = np.zeros((size, size), dtype=dtype)
    gradient = np.zeros((size,), dtype=dtype)
    g = widths[0]
    matrix[:g, :g] = np.asarray(system.root)
    gradient[:g] = np.asarray(system.root_gradient)

    def columns(level: int, node: int) -> np.ndarray:
        if level == 0:
            return np.arange(g)
        start = offsets[level] + node * widths[level]
        return np.arange(start, start + widths[level])

    for level in range(1, system.depth + 1):
        diagonal = np.asarray(system.diagonal[level])
        own = np.asarray(system.gradient[level])
        couplings = {above: np.asarray(system.cross[level][above]) for above in range(level)}
        above_nodes = {
            above: np.asarray(system.ancestors[level][above]) for above in range(1, level)
        }
        for node in range(counts[level - 1]):
            here = columns(level, node)
            matrix[np.ix_(here, here)] = diagonal[node]
            gradient[here] = own[node]
            for above, coupling in couplings.items():
                other = columns(above, 0 if above == 0 else int(above_nodes[above][node]))
                matrix[np.ix_(here, other)] = coupling[node]
                matrix[np.ix_(other, here)] = coupling[node].T
    return matrix, gradient


__all__ = [
    "AttemptSolve",
    "LevelLayout",
    "LevelSystem",
    "attempt_solve",
    "inner_added",
    "inner_bounded_response",
    "inner_predicted",
    "inner_programs",
    "inner_quadratic",
    "inner_response",
    "inner_response_parts",
    "inner_secular_step",
    "inner_singular",
    "inner_solve_scaled",
    "level_dense",
    "level_layout",
    "level_quadratic",
    "level_secular_step",
    "level_system",
    "level_vectors",
    "node_of_columns",
    "reduce_to",
    "regularizer_blocks_at",
    "scaled_system",
    "solve_damped",
    "solve_scaled",
]
