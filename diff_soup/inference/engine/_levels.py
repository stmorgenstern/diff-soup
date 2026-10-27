"""Tables and reductions over a parameter hierarchy of any depth.

A study declared with a [Hierarchy][diff_soup.inference.Hierarchy] has levels
numbered from the study (level 0, the shared coordinates) through its
intermediate levels to the experiments (level D). Every lane carries, level by
level, the coordinates of its node at every level, so its narrow point is
``[level 0 | level 1 | ... | level D]`` with the widths ``(g, v_1, ..., v_D)``.

The reduction makes one pass per level: every node of every level gets its own
diagonal block and one coupling block to each of its ancestor levels, summed
over the lanes below it. The structural zeros between siblings are never
stored, and every nonzero block of the Gauss-Newton matrix of the fitted
vector appears exactly once::

    H[S, S]           = sum_n shared[n]
    H[n, n]           = diagonal[l][n]
    H[n, ancestor_a]  = cross[l][a][n]      (and its transpose)

Every sum runs over a table that lists the lanes of a node in a fixed order,
so its value does not depend on the order of the lane results or on which
device produced them. A reduction refuses a set of lanes that contains a
failed one of nonzero weight; a failed lane of weight zero adds nothing, its
cost and derivatives being zero. The reduction of many population members at
once, [reduce_member_gradients][diff_soup.inference.engine.reduce_member_gradients],
refuses nothing: a member with such a failed lane gets a NaN cost and gradient.
"""

from __future__ import annotations

import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.engine._reducers import refuse_failed
from diff_soup.inference.engine._status import ABSENT

Array = jax.Array


def _listed(values: Any, limit: int = 8) -> str:
    """Representations of ``values``, cut after ``limit`` with the number left."""
    values = tuple(values)
    text = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        text += f" and {len(values) - limit} more"
    return text


def _integers(value: Any) -> np.ndarray | None:
    """``value`` as a host integer array, or ``None`` when it does not hold integers."""
    array = np.asarray(value)
    if array.dtype == bool or not np.issubdtype(array.dtype, np.integer):
        return None
    return array.astype(np.int64)


def _validated(nodes, counts, widths, parents, lanes, present, names, lane_ids) -> tuple:
    """Check the structure of level tables on host copies; return the host arrays.

    Raises:
        ValueError: If a field has the wrong shape or count, an index is out of
            range, a lane's node is not the child of its node one level up, a
            node above the experiment level has no child, or a lane is not
            listed exactly once per level under its own node.
    """
    depth = len(counts)
    raw = np.asarray(nodes)
    n_lanes = int(raw.shape[0]) if raw.ndim >= 1 else 0

    def need(field: str, found: str, expected: str):
        return ValueError(
            f"LevelTables {field} has {found}, and counts={counts}, widths={widths} with "
            f"{n_lanes} lanes need {expected}."
        )

    if depth < 1 or any(count < 1 for count in counts):
        raise need("counts", f"the entries {counts}", "at least one level of at least one node")
    if len(widths) != depth + 1 or any(width < 0 for width in widths):
        raise need("widths", f"the entries {widths}", f"{depth + 1} widths of 0 or more")
    host_nodes = _integers(raw)
    if host_nodes is None or host_nodes.ndim != 2 or host_nodes.shape[1] != depth:
        raise need(
            "nodes", f"shape {raw.shape} and dtype {raw.dtype}", f"integers of shape (L, {depth})"
        )
    if n_lanes == 0:
        raise ValueError("the level tables are built for at least one lane; got none.")
    parents, lanes, present = tuple(parents), tuple(lanes), tuple(present)
    for field, entries in (("parents", parents), ("lanes", lanes), ("present", present)):
        if len(entries) != depth + 1:
            raise need(field, f"{len(entries)} entries", f"{depth + 1}, one per level 0 .. {depth}")
    if parents[0] is not None or parents[1] is not None:
        raise need("parents", "an entry at level 0 or 1", "None at levels 0 and 1")
    if lanes[0] is not None or present[0] is not None:
        raise need("lanes and present", "an entry at level 0", "None at level 0")
    host_parents: list = [None, None]
    for level in range(2, depth + 1):
        entry = _integers(parents[level])
        if entry is None or entry.shape != (counts[level - 1],):
            shape = np.shape(parents[level])
            raise need(
                f"parents[{level}]", f"shape {shape}", f"integers of shape ({counts[level - 1]},)"
            )
        host_parents.append(entry)
    host_lanes: list = [None]
    host_present: list = [None]
    for level in range(1, depth + 1):
        table = _integers(lanes[level])
        mask = np.asarray(present[level])
        if table is None or table.ndim != 2 or table.shape[0] != counts[level - 1]:
            shape = np.shape(lanes[level])
            raise need(
                f"lanes[{level}]",
                f"shape {shape}",
                f"integers of shape ({counts[level - 1]}, M), M the most lanes under one node",
            )
        if mask.shape != table.shape or mask.dtype != bool:
            raise need(
                f"present[{level}]",
                f"shape {mask.shape} and dtype {mask.dtype}",
                f"booleans of shape {table.shape}",
            )
        host_lanes.append(table)
        host_present.append(mask)
    if names is not None and len(names) != depth:
        raise need("names", f"{len(names)} entries", f"{depth}, one per level 1 .. {depth}")
    if lane_ids is not None and len(lane_ids) != n_lanes:
        raise need("lane_ids", f"{len(lane_ids)} entries", f"{n_lanes}, one per lane")

    # Indices in range: a lane's node, a node's parent, a listed lane.
    for level in range(1, depth + 1):
        column, count = host_nodes[:, level - 1], counts[level - 1]
        bad = np.flatnonzero((column < 0) | (column >= count))
        if bad.size:
            lane = int(bad[0])
            raise ValueError(
                f"lane {lane} names node {int(column[lane])} of level {level}, which has "
                f"{count} nodes (0 .. {count - 1}); give node indices within counts."
            )
    for level in range(2, depth + 1):
        entry, count = host_parents[level], counts[level - 2]
        bad = np.flatnonzero((entry < 0) | (entry >= count))
        if bad.size:
            node = int(bad[0])
            raise ValueError(
                f"parents[{level}] gives node {node} of level {level} the parent "
                f"{int(entry[node])}, and level {level - 1} has {count} nodes (0 .. {count - 1})."
            )
    for level in range(1, depth + 1):
        table, mask = host_lanes[level], host_present[level]
        bad = np.argwhere(mask & ((table < 0) | (table >= n_lanes)))
        if bad.size:
            node, slot = (int(value) for value in bad[0])
            raise ValueError(
                f"lanes[{level}] lists lane {int(table[node, slot])} under node {node}, and the "
                f"tables hold {n_lanes} lanes (0 .. {n_lanes - 1})."
            )

    # A lane's node at every level is the parent of its node at the next level.
    for level in range(2, depth + 1):
        above = host_parents[level][host_nodes[:, level - 1]]
        bad = np.flatnonzero(above != host_nodes[:, level - 2])
        if bad.size:
            lane = int(bad[0])
            raise ValueError(
                f"lane {lane} sits under node {int(host_nodes[lane, level - 1])} of level "
                f"{level}, whose parent in parents[{level}] is node {int(above[lane])} of level "
                f"{level - 1}, while the lane's node at level {level - 1} is "
                f"{int(host_nodes[lane, level - 2])}; a lane's node at each level is the parent "
                "of its node at the next level."
            )

    # Every node above the experiment level has a child.
    for level in range(1, depth):
        named = np.zeros(counts[level - 1], dtype=bool)
        named[host_parents[level + 1]] = True
        childless = np.flatnonzero(~named)
        if childless.size:
            node = int(childless[0])
            raise ValueError(
                f"node {node} of level {level} has no node of level {level + 1} below it in "
                f"parents[{level + 1}]; every node above the experiment level has a child, so "
                "leave it out of counts or name it as a parent."
            )

    # Every lane is listed exactly once per level, under its own node.
    for level in range(1, depth + 1):
        table, mask = host_lanes[level], host_present[level]
        listed_nodes, listed_slots = np.nonzero(mask)
        listed = table[listed_nodes, listed_slots]
        own = host_nodes[listed, level - 1]
        wrong = np.flatnonzero(own != listed_nodes)
        if wrong.size:
            index = int(wrong[0])
            raise ValueError(
                f"lanes[{level}] lists lane {int(listed[index])} under node "
                f"{int(listed_nodes[index])}, and nodes gives it node {int(own[index])} at level "
                f"{level}; list each lane exactly once, under its own node."
            )
        times = np.bincount(listed, minlength=n_lanes)
        wrong = np.flatnonzero(times != 1)
        if wrong.size:
            lane = int(wrong[0])
            raise ValueError(
                f"lanes[{level}] lists lane {lane} {int(times[lane])} times; list each lane "
                "exactly once, under its own node."
            )
    return host_nodes, host_parents, host_lanes, host_present


class LevelTables(eqx.Module):
    """Where every lane sits in the tree of a parameter hierarchy, level by level.

    Levels are numbered 1 .. D below the study, the experiments at D. The
    fields indexed by level number carry ``None`` at the levels they do not
    describe.

    Attributes:
        nodes: ``(L, D)`` int32, lane ``i``'s node at the levels 1 .. D.
        parents: ``(None, None, parents_2, ..., parents_D)``: per node of level
            ``l``, its node at level ``l - 1``.
        lanes: ``(None, lanes_1, ..., lanes_D)``: per node of level ``l``, the
            positions of its lanes in the lane results, ``(N_l, M_l)`` int32,
            padded to the most lanes under one node of that level.
        present: Like ``lanes``: whether an entry names a lane.
        counts: ``(N_1, ..., N_D)``, the nodes of each level.
        widths: ``(g, v_1, ..., v_D)``, the shared width and each level's width.
        names: The D level names, ``"experiment"`` last, or ``None``.
        lane_ids: The identifier of each lane, or ``None``.
    """

    nodes: Array
    parents: tuple
    lanes: tuple
    present: tuple
    counts: tuple = eqx.field(static=True)
    widths: tuple = eqx.field(static=True)
    names: Any = eqx.field(static=True)
    lane_ids: Any = eqx.field(static=True)

    def __init__(
        self,
        *,
        nodes: Any,
        counts: Any,
        widths: Any,
        parents: Any,
        lanes: Any,
        present: Any,
        names: Any = None,
        lane_ids: Any = None,
        device: Any = None,
    ):
        """Check that the tables describe a tree, then store them on ``device``.

        Args:
            nodes: ``(L, D)`` integers, each lane's node at the levels 1 .. D.
            counts: The number of nodes of each level 1 .. D.
            widths: ``(g, v_1, ..., v_D)``.
            parents: ``D + 1`` entries: ``None`` at levels 0 and 1, then per
                level the ``(N_l,)`` integer parent of each node.
            lanes: ``D + 1`` entries: ``None`` at level 0, then per level the
                ``(N_l, M_l)`` integer positions of each node's lanes.
            present: Like ``lanes``, booleans: whether an entry names a lane.
            names: Optional level names, one per level 1 .. D.
            lane_ids: Optional lane identifiers, one per lane.
            device: The JAX device of the tables; ``None`` for the default
                device.

        Raises:
            ValueError: If a field has the wrong shape, an index is out of
                range, a lane's node is not the child of its node one level up,
                a node above the experiment level has no child, or a lane is not
                listed exactly once per level under its own node.
        """
        counts = tuple(int(count) for count in counts)
        widths = tuple(int(width) for width in widths)
        names = None if names is None else tuple(names)
        lane_ids = None if lane_ids is None else tuple(lane_ids)
        host_nodes, host_parents, host_lanes, host_present = _validated(
            nodes, counts, widths, parents, lanes, present, names, lane_ids
        )
        self.nodes = jax.device_put(host_nodes.astype(np.int32), device)
        self.parents = (None, None) + tuple(
            jax.device_put(entry.astype(np.int32), device) for entry in host_parents[2:]
        )
        self.lanes = (None,) + tuple(
            jax.device_put(table.astype(np.int32), device) for table in host_lanes[1:]
        )
        self.present = (None,) + tuple(jax.device_put(mask, device) for mask in host_present[1:])
        self.counts = counts
        self.widths = widths
        self.names = names
        self.lane_ids = lane_ids

    @property
    def depth(self) -> int:
        """The number of levels below the study, the experiment level included."""
        return len(self.counts)


class LevelBlocks(eqx.Module):
    """The blocks of the Gauss-Newton matrix of a hierarchy, one pass per level.

    The fields indexed by level number carry ``None`` at level 0.

    Attributes:
        diagonal: ``diagonal[l]``, ``(N_l, v_l, v_l)``: per node of level ``l``,
            the weighted sum over its lanes of their level-``l`` block.
        cross: ``cross[l][a]``, ``(N_l, v_l, v_a)`` for ``a = 0 .. l - 1``: per
            node, the weighted sum of its lanes' coupling of level ``l`` with
            level ``a`` (``a = 0``, the shared coordinates).
        gradient: ``gradient[l]``, ``(N_l, v_l)``.
        cost: ``cost[l]``, ``(N_l,)``, weighted.
        rows: ``rows[l]``, ``(N_l,)``, the residual rows, unweighted.
        shared: ``(N_1, g, g)``: per node of level 1, its share of the shared
            block; the shared block is their sum.
        gradient_shared: ``(N_1, g)``: per node of level 1, its share of the
            shared gradient.
    """

    diagonal: tuple
    cross: tuple
    gradient: tuple
    cost: tuple
    rows: tuple
    shared: Array
    gradient_shared: Array


class LevelGradients(eqx.Module):
    """The gradient of a hierarchy level by level, with the cost and rows per node.

    Attributes:
        gradient: ``gradient[l]``, ``(N_l, v_l)``; ``None`` at level 0.
        gradient_shared: ``(N_1, g)``: per node of level 1, its share of the
            shared gradient.
        cost: ``cost[l]``, ``(N_l,)``, weighted; ``None`` at level 0.
        rows: ``rows[l]``, ``(N_l,)``, unweighted; ``None`` at level 0.
    """

    gradient: tuple
    gradient_shared: Array
    cost: tuple
    rows: tuple


def level_tables(
    hierarchy: Any, experiments: Any, *, lanes: Any = None, device: Any = None
) -> LevelTables:
    """Build the level tables of a hierarchy study on the host and place them on ``device``.

    Node indices are those of the study's tree (first appearance over the
    experiments in study order), whatever ``lanes`` holds: a node whose lanes
    are all absent from ``lanes`` stays in the tables and sums to zero.

    Args:
        hierarchy: The study's [Hierarchy][diff_soup.inference.Hierarchy].
        experiments: Every experiment of the study, in study order.
        lanes: The ids of the lane results, in their order; ``None`` for every
            experiment in study order.
        device: The JAX device of the tables; ``None`` for the default device.

    Returns:
        The [LevelTables][diff_soup.inference.engine.LevelTables], with the
        level names and the lane ids.

    Raises:
        ValueError: If the experiments do not form a tree under the hierarchy,
            ``lanes`` is empty, names an experiment twice, or names an id that
            is not an experiment of the study.
    """
    from diff_soup.inference.statistics.hierarchy import _tree

    tree = _tree(hierarchy, experiments)
    ids = tree.ids if lanes is None else tuple(lanes)
    if not ids:
        raise ValueError("the level tables are built for at least one lane; got none.")
    index_of = {identifier: index for index, identifier in enumerate(tree.ids)}
    seen: set = set()
    experiment_of_lane = []
    for lane in ids:
        if lane not in index_of:
            raise ValueError(
                f"lane {lane!r} is not an experiment of the study; name lanes by experiment "
                "ids, as Results.lane_ids gives them."
            )
        if lane in seen:
            raise ValueError(
                f"the lanes name experiment {lane!r} more than once; name each lane once."
            )
        seen.add(lane)
        experiment_of_lane.append(index_of[lane])
    experiment_of_lane = np.asarray(experiment_of_lane, dtype=np.int64)
    nodes = np.asarray(tree.nodes, dtype=np.int64)[experiment_of_lane]
    tables: list = [None]
    present: list = [None]
    for level in range(1, tree.depth + 1):
        column, count = nodes[:, level - 1], tree.counts[level - 1]
        per_node = np.bincount(column, minlength=count)
        most = int(per_node.max())
        # The lanes of a node, sorted by experiment index.
        order = np.lexsort((experiment_of_lane, column))
        sorted_nodes = column[order]
        starts = np.concatenate(([0], np.cumsum(per_node)[:-1]))
        slots = np.arange(len(ids)) - starts[sorted_nodes]
        table = np.zeros((count, most), dtype=np.int32)
        mask = np.zeros((count, most), dtype=bool)
        table[sorted_nodes, slots] = order
        mask[sorted_nodes, slots] = True
        tables.append(table)
        present.append(mask)
    return LevelTables(
        nodes=nodes,
        counts=tree.counts,
        widths=tree.widths,
        parents=tree.parents,
        lanes=tuple(tables),
        present=tuple(present),
        names=tree.names,
        lane_ids=ids,
        device=device,
    )


def _node_sum(lanes: Array, present: Array, per_lane: Array) -> Array:
    """Sum a per-lane array over the lanes of every node, in table order."""
    gathered = per_lane[lanes]
    keep = jnp.reshape(present, present.shape + (1,) * (per_lane.ndim - 1))
    return jnp.sum(jnp.where(keep, gathered, jnp.zeros((), per_lane.dtype)), axis=1)


def _slices(widths: tuple) -> tuple:
    """The slice of each level 0 .. D in a narrow point."""
    starts = np.concatenate(([0], np.cumsum(widths)[:-1])).astype(int)
    return tuple(slice(int(start), int(start) + int(width)) for start, width in zip(starts, widths))


@functools.partial(jax.jit, static_argnames=("widths",))
def _reduce_levels(lanes, present, normal, gradient, cost, rows, weight, *, widths):
    at = _slices(widths)
    normal = weight[:, None, None] * normal
    gradient = weight[:, None] * gradient
    cost = weight * cost
    diagonal, cross, own, node_cost, node_rows = [None], [None], [None], [None], [None]
    for level in range(1, len(widths)):
        here = at[level]

        def total(per_lane, level=level):
            return _node_sum(lanes[level], present[level], per_lane)

        diagonal.append(total(normal[:, here, here]))
        cross.append(tuple(total(normal[:, here, at[above]]) for above in range(level)))
        own.append(total(gradient[:, here]))
        node_cost.append(total(cost))
        node_rows.append(total(rows))
    return LevelBlocks(
        diagonal=tuple(diagonal),
        cross=tuple(cross),
        gradient=tuple(own),
        cost=tuple(node_cost),
        rows=tuple(node_rows),
        shared=_node_sum(lanes[1], present[1], normal[:, at[0], at[0]]),
        gradient_shared=_node_sum(lanes[1], present[1], gradient[:, at[0]]),
    )


@functools.partial(jax.jit, static_argnames=("widths",))
def _reduce_level_gradients(lanes, present, gradient, cost, rows, weight, *, widths):
    at = _slices(widths)
    gradient = weight[:, None] * gradient
    cost = weight * cost
    own, node_cost, node_rows = [None], [None], [None]
    for level in range(1, len(widths)):

        def total(per_lane, level=level):
            return _node_sum(lanes[level], present[level], per_lane)

        own.append(total(gradient[:, at[level]]))
        node_cost.append(total(cost))
        node_rows.append(total(rows))
    return LevelGradients(
        gradient=tuple(own),
        gradient_shared=_node_sum(lanes[1], present[1], gradient[:, at[0]]),
        cost=tuple(node_cost),
        rows=tuple(node_rows),
    )


@functools.partial(jax.jit, static_argnames=("widths",))
def _reduce_member_gradients(lanes, present, gradient, cost, rows, weight, *, widths):
    def one(gradient, cost, rows, weight):
        levels = _reduce_level_gradients(
            lanes, present, gradient, cost, rows, weight, widths=widths
        )
        joint = [jnp.sum(levels.gradient_shared, axis=0)]
        joint.extend(levels.gradient[level].reshape(-1) for level in range(1, len(widths)))
        return jnp.sum(levels.cost[-1]), jnp.concatenate(joint)

    return jax.vmap(one)(gradient, cost, rows, weight)


def _lane_weights(tables: LevelTables, weights: Any, n_lanes: int, dtype: Any) -> np.ndarray:
    """Each lane's weight on the host: one, or that of its node at the weighted level."""
    if weights is None:
        return np.ones((n_lanes,), dtype=dtype)
    if not isinstance(weights, (tuple, list)) or len(weights) != 2:
        raise ValueError(
            "weights= is None or (level, values) with one value per node of that level; got "
            f"{weights!r}."
        )
    level, values = weights
    depth = tables.depth
    if isinstance(level, str):
        if tables.names is None:
            raise ValueError(
                f"weights= names the level {level!r}, and these tables carry no level names; "
                f"give its number, 1 .. {depth}."
            )
        if level not in tables.names:
            raise ValueError(
                f"weights= names the level {level!r}; the levels of the tables are "
                f"{_listed(tables.names)}, or their numbers 1 .. {depth}."
            )
        number = tables.names.index(level) + 1
    elif (
        isinstance(level, (int, np.integer))
        and not isinstance(level, bool)
        and (1 <= int(level) <= depth)
    ):
        number = int(level)
    elif tables.names is not None:
        raise ValueError(
            f"weights= names the level {level!r}; the levels of the tables are "
            f"{_listed(tables.names)}, or their numbers 1 .. {depth}."
        )
    else:
        raise ValueError(
            f"weights= names the level {level!r}, and these tables carry no level names; "
            f"give its number, 1 .. {depth}."
        )
    values = np.asarray(values, dtype=dtype)
    count = tables.counts[number - 1]
    if values.ndim != 1 or values.shape[0] != count:
        name = tables.names[number - 1] if tables.names is not None else number
        raise ValueError(
            f"weights= gives {int(values.size)} values for level {name!r}, which has {count} "
            "nodes; give one value per node."
        )
    return values[np.asarray(tables.nodes)[:, number - 1]]


def _fits(tables: LevelTables, n_lanes: int, lane_ids: tuple, width: int) -> None:
    """Refuse lane results that do not fit the tables."""
    expected = int(tables.nodes.shape[0])
    if n_lanes != expected:
        raise ValueError(
            f"the results carry {n_lanes} lanes and the tables {expected}; build the tables for "
            "the lanes of these results."
        )
    if tables.lane_ids is not None and tuple(lane_ids) != tuple(tables.lane_ids):
        raise ValueError(
            f"the results are for the lanes {_listed(lane_ids)} and the tables for "
            f"{_listed(tables.lane_ids)}; build the tables for the lanes of these results, "
            "LoweredStudy.tables(results.lane_ids)."
        )
    total = int(sum(tables.widths))
    if width != total:
        raise ValueError(
            f"the lanes have narrow width {width} and the levels of the tables add up to "
            f"{total} (widths {tables.widths}); build the tables from the study these results "
            "come from."
        )


def _checked(tables: LevelTables, results: Any, lane_ids: tuple, width: int, weights: Any, dtype):
    """Refuse results that do not fit the tables; return the lane weights (host)."""
    n_lanes = int(np.shape(results.cost)[0])
    _fits(tables, n_lanes, lane_ids, width)
    lane_weights = _lane_weights(tables, weights, n_lanes, dtype)
    # A failed lane of weight zero is outside the sum (its cost and derivatives are zero).
    refuse_failed(results.status, tuple(lane_ids), None if weights is None else lane_weights)
    return lane_weights


def reduce_levels(
    tables: LevelTables, results: Any, lane_ids: Any, weights: Any = None
) -> LevelBlocks:
    """Sum the blocks of a set of lanes into the blocks of every node of every level.

    Args:
        tables: The level tables of the lanes, in the order of the results.
        results: The lane results of the ``blocks`` mode.
        lane_ids: The identifier of each lane, in the order of the results.
        weights: ``None`` for one, or ``(level, values)``: one value per node
            of one level, the level given by its number (1 .. D) or, when the
            tables carry level names, by its name. A lane takes the weight of
            its node at that level, applied before the sum to every field but
            the rows. A failed lane of weight zero adds nothing and is not
            refused.

    Returns:
        The [LevelBlocks][diff_soup.inference.engine.LevelBlocks].

    Raises:
        LaneFailure: If a lane of the set failed whose weight is not zero.
        ValueError: If the results carry no blocks or do not fit the tables,
            or ``weights`` is malformed.
    """
    if results.normal is None or results.gradient is None:
        raise ValueError(
            "the blocks of the levels are summed from lane results of the blocks mode; these "
            "results carry no normal matrix."
        )
    dtype = np.dtype(results.normal.dtype)
    width = int(results.normal.shape[-1])
    weight = _checked(tables, results, tuple(lane_ids), width, weights, dtype)
    return _reduce_levels(
        tables.lanes,
        tables.present,
        results.normal,
        results.gradient,
        results.cost,
        results.rows,
        weight,
        widths=tables.widths,
    )


def reduce_level_gradients(
    tables: LevelTables, results: Any, lane_ids: Any, weights: Any = None
) -> LevelGradients:
    """Sum the gradients of a set of lanes into the gradient of every node of every level.

    Args:
        tables: The level tables of the lanes, in the order of the results.
        results: The lane results of the ``gradient`` or the ``blocks`` mode.
        lane_ids: The identifier of each lane, in the order of the results.
        weights: As for [reduce_levels][diff_soup.inference.engine.reduce_levels].

    Returns:
        The [LevelGradients][diff_soup.inference.engine.LevelGradients].

    Raises:
        LaneFailure: If a lane of the set failed whose weight is not zero.
        ValueError: If the results carry no gradient or do not fit the tables,
            or ``weights`` is malformed.
    """
    if results.gradient is None:
        raise ValueError(
            "the gradient of the levels is summed from lane results of the gradient or the "
            "blocks mode; these results carry no gradient."
        )
    dtype = np.dtype(results.gradient.dtype)
    width = int(results.gradient.shape[-1])
    weight = _checked(tables, results, tuple(lane_ids), width, weights, dtype)
    return _reduce_level_gradients(
        tables.lanes,
        tables.present,
        results.gradient,
        results.cost,
        results.rows,
        weight,
        widths=tables.widths,
    )


def reduce_member_gradients(
    tables: LevelTables, results: Any, lane_ids: Any, lane_weights: Any = None
) -> tuple:
    """The cost and the joint gradient of every population member of a result, in one reduction.

    Each member's lanes are summed as
    [reduce_level_gradients][diff_soup.inference.engine.reduce_level_gradients]
    sums them, under that member's own weight of every lane, and its gradient
    is laid out as one joint vector: the shared coordinates, then every node of
    level 1 in node order, and so on down to the experiments. A member none of
    whose lanes failed gets the weighted sum of its lane costs and the gradient
    of that sum. A member with a failed lane of nonzero weight gets NaN in its
    cost and in every entry of its gradient, and the other members are summed
    as usual; ``results.failures()`` names the failed lane. A failed lane of
    weight zero adds nothing.

    Args:
        tables: The level tables of the lanes, in the order of the results.
        results: The [Results][diff_soup.inference.engine.Results] of the
            ``gradient`` or the ``blocks`` mode, of ``M`` members and ``L``
            lanes.
        lane_ids: The identifier of each lane, in the order of the results.
        lane_weights: ``None`` weighs every lane one; else ``(M, L)``: per
            member, the weight of every lane.

    Returns:
        ``(cost, gradient)`` as host arrays: ``cost`` of shape ``(M,)``, the
        weighted cost of every member, and ``gradient`` of shape ``(M, n)``,
        its joint gradient.

    Raises:
        ValueError: If the results carry no gradient or do not fit the tables,
            or ``lane_weights`` does not give one weight per member and lane.
    """
    if results.gradient is None:
        raise ValueError(
            "the gradients of the members are summed from lane results of the gradient or the "
            "blocks mode; these results carry no gradient."
        )
    gradient = np.asarray(results.gradient)
    members, n_lanes, width = gradient.shape
    _fits(tables, n_lanes, tuple(lane_ids), width)
    if lane_weights is None:
        weight = np.ones((members, n_lanes), dtype=gradient.dtype)
    else:
        weight = np.asarray(lane_weights, dtype=gradient.dtype)
        if weight.shape != (members, n_lanes):
            raise ValueError(
                f"lane_weights= gives one weight per member and lane, shape ({members}, "
                f"{n_lanes}); got {weight.shape}."
            )
    cost, joint = _reduce_member_gradients(
        tables.lanes,
        tables.present,
        gradient,
        np.asarray(results.cost),
        np.asarray(results.rows),
        weight,
        widths=tables.widths,
    )
    cost, joint = np.array(cost), np.array(joint)
    kinds = np.asarray(results.status.kind)
    absent = (kinds == ABSENT) & (weight != 0.0)
    if np.any(absent):
        row, column = (int(index) for index in np.argwhere(absent)[0])
        raise ValueError(
            f"lane {tuple(lane_ids)[column]!r} of member row {row} was not evaluated (status "
            "absent: the request's units= left it out) and weighs more than zero; evaluate it, "
            "or give it the weight zero."
        )
    failed = np.any((kinds != 0) & (kinds != ABSENT) & (weight != 0.0), axis=1)
    cost[failed] = np.nan
    joint[failed] = np.nan
    return cost, joint


# ---------------------------------------------------------------------------------------
# Sums over lanes held by several processes
# ---------------------------------------------------------------------------------------


def partial_levels(tables: LevelTables, results: Any, device: Any = None) -> Any:
    """Every member's sums of these lanes over the level tables, every lane weighing one.

    The sums of a set of lanes that is part of a study's lanes: ``tables`` are
    the study's level tables built for these lanes (every node of the study,
    a node with none of these lanes summing to zero), so the partial sums of
    disjoint sets of lanes add up to the sums over their union. Nothing is
    refused: a failed lane and a unit that was not evaluated carry zeros and
    add nothing; the caller refuses them with the weights it applies.

    Args:
        tables: The level tables of the lanes of ``results``, in their order.
        results: The [Results][diff_soup.inference.engine.Results] of the
            ``blocks``, ``hessian`` or ``gradient`` mode.
        device: The JAX device the sums run on; ``None`` for the default.

    Returns:
        A [LevelBlocks][diff_soup.inference.engine.LevelBlocks] (``blocks``,
        ``hessian``) or [LevelGradients][diff_soup.inference.engine.LevelGradients]
        (``gradient``) of host arrays whose leaves lead with the member axis
        of ``results``.

    Raises:
        ValueError: If the results carry no gradient, or do not fit the tables.
    """
    if results.gradient is None:
        raise ValueError(
            "the sums of the levels are taken from lane results of the gradient, blocks or "
            "hessian mode; these results carry no gradient."
        )
    gradient = np.asarray(results.gradient)
    members, n_lanes, width = gradient.shape
    _fits(tables, n_lanes, tuple(results.lane_ids), width)
    dtype = gradient.dtype
    ones = jax.device_put(np.ones((n_lanes,), dtype=dtype), device)
    lanes, present = tables.lanes, tables.present
    rows_of = []
    for row in range(members):
        cost = jax.device_put(np.asarray(results.cost)[row], device)
        rows = jax.device_put(np.asarray(results.rows)[row], device)
        if results.normal is not None:
            normal = jax.device_put(np.asarray(results.normal)[row], device)
            sums = _reduce_levels(
                lanes,
                present,
                normal,
                jax.device_put(gradient[row], device),
                cost,
                rows,
                ones,
                widths=tables.widths,
            )
        else:
            sums = _reduce_level_gradients(
                lanes,
                present,
                jax.device_put(gradient[row], device),
                cost,
                rows,
                ones,
                widths=tables.widths,
            )
        rows_of.append(jax.device_get(sums))
    return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *rows_of)


def _level_one_of_nodes(tables: LevelTables) -> tuple:
    """Per level 1 .. D, the level-1 node above each of its nodes (host arrays); ``None`` at 0."""
    above: list = [None, np.arange(tables.counts[0], dtype=np.int64)]
    for level in range(2, tables.depth + 1):
        above.append(above[level - 1][np.asarray(tables.parents[level], dtype=np.int64)])
    return tuple(above)


def weigh_levels(tables: LevelTables, sums: Any, weights: Any) -> Any:
    """One member's sums over the levels, weighed by one weight per member of level 1.

    The sums of [partial_levels][diff_soup.inference.engine._levels.partial_levels]
    weigh every lane one. A lane's weight that is the weight of its node at
    level 1 multiplies every sum below that node, so weighing the sums of each
    node by the weight of its level-1 node, after the sum, gives the weighted
    sums (to rounding). Rows stay unweighted.

    Args:
        tables: The level tables of the study.
        sums: One member's [LevelBlocks][diff_soup.inference.engine.LevelBlocks]
            or [LevelGradients][diff_soup.inference.engine.LevelGradients],
            host arrays.
        weights: ``None`` for one, or ``(level, values)`` with the level 1 (by
            number or by name) and one value per node of level 1.

    Returns:
        The weighed sums, of the type of ``sums``.

    Raises:
        ValueError: If ``weights`` weighs another level than level 1: the sums
            of the nodes above that level mix lanes of different weights and
            cannot be weighed after the sum.
    """
    if weights is None:
        return sums
    level, values = weights
    number = (
        tables.names.index(level) + 1
        if isinstance(level, str) and tables.names is not None and level in tables.names
        else level
    )
    if isinstance(number, str) or int(number) != 1:
        first = tables.names[0] if tables.names is not None else 1
        raise ValueError(
            f"weights= weighs the level {level!r}, and the lanes of this study are summed in "
            "several processes, where weights apply after the sum and so only to the members "
            f"of level 1 ({first!r}); weigh the members of level 1, or lower the study in one "
            "process."
        )
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    above = _level_one_of_nodes(tables)
    blocks = isinstance(sums, LevelBlocks)

    def scaled(array: Any, level: int) -> np.ndarray:
        array = np.asarray(array)
        factor = values[above[level]].astype(array.dtype)
        return factor.reshape(factor.shape + (1,) * (array.ndim - 1)) * array

    depth = tables.depth
    gradient = (None,) + tuple(scaled(sums.gradient[level], level) for level in range(1, depth + 1))
    cost = (None,) + tuple(scaled(sums.cost[level], level) for level in range(1, depth + 1))
    rows = tuple(sums.rows)
    gradient_shared = scaled(sums.gradient_shared, 1)
    if not blocks:
        return LevelGradients(
            gradient=gradient, gradient_shared=gradient_shared, cost=cost, rows=rows
        )
    return LevelBlocks(
        diagonal=(None,)
        + tuple(scaled(sums.diagonal[level], level) for level in range(1, depth + 1)),
        cross=(None,)
        + tuple(
            tuple(scaled(entry, level) for entry in sums.cross[level])
            for level in range(1, depth + 1)
        ),
        gradient=gradient,
        cost=cost,
        rows=rows,
        shared=scaled(sums.shared, 1),
        gradient_shared=gradient_shared,
    )


def member_row(sums: Any, row: int) -> Any:
    """The sums of row ``row`` of sums whose leaves lead with the member axis."""
    return jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[row], sums)


def _joint_offsets(tables: LevelTables) -> tuple:
    """The first joint column of every level 0 .. D: the shared block, then each level's nodes."""
    offsets = [0, int(tables.widths[0])]
    for level in range(1, tables.depth):
        offsets.append(offsets[-1] + int(tables.counts[level - 1]) * int(tables.widths[level]))
    return tuple(offsets)


def joint_gradient(tables: LevelTables, sums: Any) -> tuple:
    """``(cost, gradient)`` of one member's sums, the gradient laid out as one joint vector.

    The layout is that of
    [reduce_member_gradients][diff_soup.inference.engine.reduce_member_gradients]:
    the shared coordinates, then every node of level 1 in node order, and so
    on down to the experiments.
    """
    parts = [np.sum(np.asarray(sums.gradient_shared), axis=0)]
    parts.extend(
        np.asarray(sums.gradient[level]).reshape(-1) for level in range(1, tables.depth + 1)
    )
    return float(np.sum(np.asarray(sums.cost[-1]))), np.concatenate(parts)


def joint_matrix(tables: LevelTables, sums: LevelBlocks) -> np.ndarray:
    """One member's level blocks as the dense matrix of the joint vector of ``joint_gradient``.

    Every nonzero block appears exactly once in the level blocks: the shared
    block is the sum of the level-1 shares, each node's diagonal block sits at
    its own columns, and each coupling of a node with an ancestor level at the
    columns of its ancestor there (and transposed).
    """
    widths, counts, depth = tables.widths, tables.counts, tables.depth
    offsets = _joint_offsets(tables)
    size = offsets[-1] + int(counts[depth - 1]) * int(widths[depth])
    dtype = np.asarray(sums.shared).dtype
    matrix = np.zeros((size, size), dtype=dtype)
    g = int(widths[0])
    matrix[:g, :g] = np.sum(np.asarray(sums.shared), axis=0)
    # The node of every level above each node, level by level.
    ancestors: list = [None] * (depth + 1)
    for level in range(1, depth + 1):
        chain = {level: np.arange(counts[level - 1], dtype=np.int64)}
        for above in range(level, 1, -1):
            chain[above - 1] = np.asarray(tables.parents[above], dtype=np.int64)[chain[above]]
        ancestors[level] = chain
    for level in range(1, depth + 1):
        width = int(widths[level])
        if width == 0:
            continue
        diagonal = np.asarray(sums.diagonal[level])
        for node in range(int(counts[level - 1])):
            start = offsets[level] + node * width
            here = slice(start, start + width)
            matrix[here, here] = diagonal[node]
            for above, block in enumerate(sums.cross[level]):
                other = int(widths[above])
                if other == 0:
                    continue
                if above == 0:
                    there = slice(0, g)
                else:
                    first = offsets[above] + int(ancestors[level][above][node]) * other
                    there = slice(first, first + other)
                entry = np.asarray(block)[node]
                matrix[here, there] = entry
                matrix[there, here] = entry.T
    return matrix


__all__ = [
    "LevelBlocks",
    "LevelGradients",
    "LevelTables",
    "level_tables",
    "reduce_level_gradients",
    "reduce_levels",
    "reduce_member_gradients",
]
