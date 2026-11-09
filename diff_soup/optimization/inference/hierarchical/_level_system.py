"""The level system of a study's tree: its layout, its regularizer blocks and its assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from numpy.typing import DTypeLike

from diff_soup.optimization.inference.engine._levels import LevelBlocks, LevelTables

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
        constraint_active: Optional ``(c,)``, boolean: which equality rows a step
            keeps; ``None`` keeps every row. The inequality rows of a fit ride
            here at a fixed count, so one program solves every active set.

    The fixed coordinates and the held nodes are arrays of the system, not
    part of its structure: systems of one hierarchy that fix or hold different
    sets share every compiled program. ``fixed`` and ``held`` list them on the
    host.
    """

    root: jax.Array
    root_gradient: jax.Array
    diagonal: tuple
    cross: tuple
    gradient: tuple
    point: jax.Array
    constraint_matrix: jax.Array
    constraint_rhs: jax.Array
    cost: jax.Array
    ancestors: tuple
    descendants: tuple
    present: tuple
    fixed_mask: jax.Array
    held_mask: tuple
    counts: tuple = eqx.field(static=True)
    widths: tuple = eqx.field(static=True)
    constraint_active: jax.Array | None = None

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


def _device(tables: tuple, device: jax.Device | jax.sharding.Sharding | None = None) -> tuple:
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

    def __init__(
        self,
        counts: tuple,
        widths: tuple,
        parents: tuple,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ):
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

    def constant(self, name: str, value: np.ndarray) -> jax.Array:
        """``value`` on the device, placed once per distinct value of ``name``."""
        value = np.ascontiguousarray(value)
        key = (name, value.dtype.str, value.shape, value.tobytes())
        placed = self._kept.get(key)
        if placed is None:
            if len(self._kept) >= self.KEPT:
                self._kept.pop(next(iter(self._kept)))
            placed = self._kept[key] = jax.device_put(value, self.device)
        return placed


def level_layout(
    tables: LevelTables, device: jax.Device | jax.sharding.Sharding | None = None
) -> LevelLayout:
    """The [LevelLayout][diff_soup.optimization.inference.hierarchical._level_system.LevelLayout] of ``tables``.

    Args:
        tables: The [LevelTables][diff_soup.optimization.inference.engine.LevelTables].
        device: The JAX device the systems are placed on; ``None`` for the
            default device.
    """
    parents = tuple(None if entry is None else np.asarray(entry) for entry in tables.parents)
    return LevelLayout(tables.counts, tables.widths, parents, device)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _listed(values: Iterable, limit: int = 8) -> str:
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
    dtype: DTypeLike,
    names: Any,
    members: Any,
    block_weights: Any = None,
    device: jax.Device | jax.sharding.Sharding | None = None,
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
    dtype: DTypeLike,
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


def _held_nodes(
    blocks: LevelBlocks,
    tables: LevelTables,
    lane_weights: ArrayLike | Sequence | None,
    touched: frozenset,
) -> tuple:
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
    regularizer: Any,
    point: ArrayLike,
    tables: LevelTables,
    level: int,
    device: jax.Device | jax.sharding.Sharding | None = None,
) -> np.ndarray | None:
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
    blocks: LevelBlocks,
    tables: LevelTables,
    point: Any,
    *,
    constraint_matrix: ArrayLike | Sequence | None = None,
    constraint_rhs: ArrayLike | Sequence | None = None,
    fixed_indices: Any = (),
    regularizer: Any = None,
    lane_weights: ArrayLike | Sequence | None = None,
    members: Any = None,
    block_weights: Any = None,
    device: jax.Device | jax.sharding.Sharding | None = None,
    layout: LevelLayout | None = None,
    regularizer_blocks: tuple | None = None,
) -> LevelSystem:
    """The system of the whole objective at ``point`` from the level blocks.

    Regularizer rows are added once to the blocks their columns address, each
    block of ``regularizer.block_rows`` times its weight in ``block_weights``
    (curvature, slope and cost alike). A fixed coordinate and every coordinate
    of a held node (no data and no regularizer row of nonzero weight) get a
    unit diagonal, a zero gradient and zero couplings, so a solve leaves them
    at their value; equality rows absorb their values.

    Args:
        blocks: The [LevelBlocks][diff_soup.optimization.inference.engine.LevelBlocks] at
            ``point``.
        tables: The [LevelTables][diff_soup.optimization.inference.engine.LevelTables] the
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
        layout: The [LevelLayout][diff_soup.optimization.inference.hierarchical._level_system.LevelLayout]
            of ``tables`` on ``device``, to reuse its index tables and kept
            arrays; ``None`` builds one for this system alone.
        regularizer_blocks: What ``regularizer_blocks`` returned for this
            regularizer, point, layout, dtype and ``block_weights``, to reuse
            instead of evaluating the regularizer again; ``None`` evaluates it.

    Returns:
        The [LevelSystem][diff_soup.optimization.inference.hierarchical._level_system.LevelSystem].

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
# Host entry points
# ---------------------------------------------------------------------------


def _dtype(system: LevelSystem) -> np.dtype:
    return np.dtype(system.root.dtype)


def _addition(system: LevelSystem, addition: ArrayLike | Sequence | None) -> np.ndarray:
    """The diagonal addition as an ``(n,)`` host array of the system's dtype; zeros for ``None``."""
    if addition is None:
        return np.zeros((system.size,), dtype=_dtype(system))
    return np.asarray(addition, dtype=_dtype(system))


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
    "LevelLayout",
    "LevelSystem",
    "level_dense",
    "level_layout",
    "level_system",
    "node_of_columns",
    "regularizer_blocks_at",
]
