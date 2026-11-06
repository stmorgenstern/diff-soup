"""How tangents cross each primitive of a tangent map: the rules of the pattern walk.

Each rule maps the flows of an operation's operands to the flows of its results (see
[flows][diff_soup.problem._ad.flows]). A tangent map is linear in its tangents, so an
operation either moves elements, combines them, or scales them by a primal value; only
a value that holds whatever the inputs are, a known value, can rule an entry out.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from diff_soup.problem._ad import flows as F
from diff_soup.problem._ad.pattern_walk import register_pattern_rule


def _elementwise(walk, op):
    """Each output element is reached by the matching element of each input."""
    shape, total = walk.shape(op.outvars[0]), None
    for atom in op.invars:
        flow = walk.flow(atom)
        if flow is not None:
            total = F.union(total, F.broadcast_flow(flow, walk.shape(atom), shape))
    return [total]


def _mul(walk, op):
    """A known zero coefficient passes no tangent."""
    shape, total = walk.shape(op.outvars[0]), None
    for atom, other in (op.invars, op.invars[::-1]):
        flow = walk.flow(atom)
        if flow is None:
            continue
        flow = F.broadcast_flow(flow, walk.shape(atom), shape)
        coefficient = walk.value(other)
        if coefficient is not None:
            keep = np.broadcast_to(coefficient != 0, shape).ravel()
            if not keep.all():
                flow = F.select(flow, np.where(keep, np.arange(keep.size), -1))
        total = F.union(total, flow)
    return [total]


def _select_n(walk, op):
    """A known condition picks one case per element; otherwise any case may be taken."""
    which = walk.value(op.invars[0])
    if which is None:
        return _elementwise(walk, op)
    shape = walk.shape(op.outvars[0])
    which = np.broadcast_to(which, shape).astype(np.int64).ravel()
    total = None
    for case, atom in enumerate(op.invars[1:]):
        flow = walk.flow(atom)
        if flow is not None:
            flow = F.broadcast_flow(flow, walk.shape(atom), shape)
            total = F.union(
                total, F.select(flow, np.where(which == case, np.arange(which.size), -1))
            )
    return [total]


def _broadcast(op, table):
    """Broadcast an index table as ``broadcast_in_dim`` broadcasts its operand."""
    shape, dims = op.params["shape"], op.params["broadcast_dimensions"]
    expanded = [1] * len(shape)
    for axis, dim in enumerate(dims):
        expanded[dim] = table.shape[axis]
    return np.broadcast_to(table.reshape(expanded), shape)


def _slice(op, table):
    """Slice an index table as ``slice`` slices its operand."""
    starts, limits = op.params["start_indices"], op.params["limit_indices"]
    strides = op.params["strides"] or (1,) * table.ndim
    return table[tuple(slice(*spec) for spec in zip(starts, limits, strides))]


# Operations that move elements, as functions of the operands' index tables.
_MOVES = {
    "reshape": lambda op, table: table.reshape(op.params["new_sizes"]),
    "squeeze": lambda op, table: table.squeeze(tuple(op.params["dimensions"])),
    "transpose": lambda op, table: table.transpose(op.params["permutation"]),
    "rev": lambda op, table: np.flip(table, tuple(op.params["dimensions"])),
    "broadcast_in_dim": _broadcast,
    "slice": _slice,
    "concatenate": lambda op, *tables: np.concatenate(tables, op.params["dimension"]),
    "stack": lambda op, *tables: np.stack(tables, op.params["axis"]),
    "tile": lambda op, table: np.tile(table, op.params["reps"]),
}


def _moving(walk, op):
    """Outputs are inputs moved around: run the operation on the row numbers."""
    tables, stacked = walk.tables(op.invars)
    return [F.select(stacked, _MOVES[op.primitive.name](op, *tables))]


def _unstack(walk, op):
    """Each output is one slice of the operand along ``axis``."""
    tables, stacked = walk.tables(op.invars[:1])
    parts = np.moveaxis(tables[0], op.params["axis"], 0)
    return [F.select(stacked, part) for part in parts]


def _split(walk, op):
    """Each output is one piece of the operand along ``axis``."""
    tables, stacked = walk.tables(op.invars[:1])
    pieces = F.split_tables(tables[0], op.params["sizes"], op.params["axis"])
    return [F.select(stacked, piece) for piece in pieces]


def _known_operands(walk, op, data):
    """The values of the operands outside ``data``, or ``None`` if one is not known."""
    values = {}
    for position, atom in enumerate(op.invars):
        if position in data:
            continue
        if walk.flow(atom) is not None or walk.value(atom) is None:
            return None
        values[position] = walk.value(atom)
    return values


def _gather(walk, op):
    """A gather at known indices moves elements."""
    values = _known_operands(walk, op, (0,))
    if values is None:
        return walk.fallback(op, "reads an index not known at build time")
    tables, stacked = walk.tables(op.invars[:1])
    table = F.gather_table(
        tables[0],
        values[1],
        op.params["dimension_numbers"],
        op.params["slice_sizes"],
        op.params["mode"],
    )
    return [F.select(stacked, table)]


def _dynamic_slice(walk, op):
    """A slice at known starts moves elements."""
    values = _known_operands(walk, op, (0,))
    if values is None:
        return walk.fallback(op, "reads an index not known at build time")
    tables, stacked = walk.tables(op.invars[:1])
    starts = [values[position] for position in range(1, len(op.invars))]
    return [F.select(stacked, F.dynamic_slice_table(tables[0], starts, op.params["slice_sizes"]))]


def _dynamic_update_slice(walk, op):
    """An update at known starts moves elements of both operands."""
    values = _known_operands(walk, op, (0, 1))
    if values is None:
        return walk.fallback(op, "reads an index not known at build time")
    tables, stacked = walk.tables(op.invars[:2])
    starts = [values[position] for position in range(2, len(op.invars))]
    return [F.select(stacked, F.dynamic_update_slice_table(tables[0], tables[1], starts))]


def _pad(walk, op):
    """Padding moves elements of the operand and copies the padding value."""
    tables, stacked = walk.tables(op.invars[:2])
    return [F.select(stacked, F.pad_table(tables[0], tables[1], op.params["padding_config"]))]


def _scatter(walk, op):
    """A write at known indices sends each update element to the element it lands on."""
    operand, indices, updates = op.invars
    where = walk.value(indices)
    if where is None or walk.flow(indices) is not None:
        return walk.fallback(op, "reads an index not known at build time")
    target = F.scatter_targets(
        walk.shape(operand),
        where,
        op.params["dimension_numbers"],
        walk.shape(updates),
        op.params["mode"],
    )
    kept = walk.flow(operand)
    written = np.unique(target[target >= 0])
    if kept is not None and op.primitive.name == "scatter" and written.size == np.sum(target >= 0):
        table = np.arange(kept.shape[0])
        table[written] = -1
        kept = F.select(kept, table)
    inflow = walk.flow(updates)
    if inflow is not None:
        inflow = (F.gathering(walk.size(operand), target, inflow.shape[0]) @ inflow).tocsr()
    return [F.union(kept, inflow)]


def _flat(coordinates, shape, length):
    """Flat indices of ``length`` elements, also for the single element of a scalar."""
    if coordinates:
        return np.ravel_multi_index(tuple(coordinates), shape)
    return np.zeros(length, dtype=np.int64)


def _reduce_sum(walk, op):
    """Each output element reads every input element it sums."""
    atom = op.invars[0]
    shape, flow = walk.shape(atom), walk.flow(atom)
    axes = tuple(int(axis) for axis in op.params["axes"])
    kept = [axis for axis in range(len(shape)) if axis not in axes]
    grid = np.indices(shape)
    coordinates = [grid[axis].ravel() for axis in kept]
    target = _flat(coordinates, tuple(shape[axis] for axis in kept), flow.shape[0])
    return [(F.gathering(walk.size(op.outvars[0]), target, flow.shape[0]) @ flow).tocsr()]


def _cumulative(walk, op):
    """Element ``i`` along ``axis`` reads the elements up to ``i``, or from it when reversed."""
    atom = op.invars[0]
    shape, flow = walk.shape(atom), walk.flow(atom)
    axis, length = int(op.params["axis"]), int(walk.shape(atom)[op.params["axis"]])
    late, early = np.tril_indices(length)
    if op.params.get("reverse", False):
        late, early = early, late
    grid = np.moveaxis(np.arange(flow.shape[0]).reshape(shape), axis, 0).reshape(length, -1)
    rows, cols = grid[late].ravel(), grid[early].ravel()
    reach = sparse.csr_matrix(
        (np.ones(rows.size, dtype=bool), (rows, cols)), shape=(flow.shape[0], flow.shape[0])
    )
    return [(reach @ flow).tocsr()]


def _dot(walk, op):
    """Output ``(batch, left free, right free)`` reads each side where the other can be nonzero."""
    (left_c, right_c), (left_b, right_b) = op.params["dimension_numbers"]
    left, right = op.invars
    out_shape = walk.shape(op.outvars[0])
    total = None
    sides = (
        (left, right, left_c, right_c, left_b, right_b),
        (right, left, right_c, left_c, right_b, left_b),
    )
    for side, (atom, other, contract, other_c, batch, other_b) in enumerate(sides):
        flow = walk.flow(atom)
        if flow is None:
            continue
        shape = walk.shape(atom)
        coefficient = walk.value(other)
        spots = np.argwhere(
            np.ones(walk.shape(other), bool) if coefficient is None else coefficient
        )
        own_free = [axis for axis in range(len(shape)) if axis not in (*contract, *batch)]
        other_free = [axis for axis in range(spots.shape[1]) if axis not in (*other_c, *other_b)]
        sizes = tuple(shape[axis] for axis in own_free)
        count = int(np.prod(sizes, dtype=np.int64))
        grid = np.tile(np.indices(sizes).reshape(len(own_free), count), spots.shape[0])
        spots = np.repeat(spots, count, axis=0)
        own = dict(zip(own_free, grid))
        paired = zip((*batch, *contract), (*other_b, *other_c))
        own.update((axis, spots[:, position]) for axis, position in paired)
        element = _flat([own[axis] for axis in range(len(shape))], shape, spots.shape[0])
        theirs = [spots[:, axis] for axis in other_free]
        free = (*grid, *theirs) if side == 0 else (*theirs, *grid)
        target = _flat([*(spots[:, o] for o in other_b), *free], out_shape, spots.shape[0])
        matrix = sparse.csr_matrix(
            (np.ones(target.size, dtype=bool), (target, element)),
            shape=(walk.size(op.outvars[0]), flow.shape[0]),
        )
        total = F.union(total, (matrix @ flow).tocsr())
    return [total]


def _lane_where(walk, op):
    """``diffsoup_lane_where`` unbatched is its ``taken`` operand."""
    return [walk.flow(op.invars[1])]


def _no_tangent(walk, op):
    """A boolean result carries no tangent."""
    return [None] * len(op.outvars)


_ELEMENTWISE = (
    "add",
    "add_any",
    "sub",
    "neg",
    "div",
    "convert_element_type",
    "copy",
    "copy_p",
    "real",
    "imag",
    "complex",
    "conj",
    "reduce_precision",
)
_CUMULATIVE = ("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp")
_LANE_REDUCTIONS = (
    "diffsoup_any_lane",
    "diffsoup_all_lanes",
    "diffsoup_any_live_lane",
    "diffsoup_all_live_lanes",
)
_RULES = {
    **dict.fromkeys(_ELEMENTWISE, _elementwise),
    **dict.fromkeys(_MOVES, _moving),
    **dict.fromkeys(_CUMULATIVE, _cumulative),
    **dict.fromkeys(_LANE_REDUCTIONS, _no_tangent),
    **dict.fromkeys(("scatter", "scatter-add", "scatter_add"), _scatter),
    "mul": _mul,
    "select_n": _select_n,
    "unstack": _unstack,
    "split": _split,
    "gather": _gather,
    "dynamic_slice": _dynamic_slice,
    "dynamic_update_slice": _dynamic_update_slice,
    "pad": _pad,
    "reduce_sum": _reduce_sum,
    "dot_general": _dot,
    "diffsoup_lane_where": _lane_where,
}
for _name, _rule in _RULES.items():
    register_pattern_rule(_name, _rule)
