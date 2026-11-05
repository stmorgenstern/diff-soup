"""The exact pattern of a Jacobian, read from a function's tangent map.

An entry ``df_i/dy_j`` can be nonzero only if a tangent seeded at ``y_j`` reaches
``f_i``. [tangent_pattern][diff_soup.problem._ad.tangent_pattern.tangent_pattern]
walks the jaxpr of ``jax.jvp`` once, carrying for each array one boolean row per
element: which seeds reach it. No state is evaluated, so the answer does not depend
on values the residual may not be finite at, and an abstract value, such as a state
or a parameter, stands for anything: both sides of a state-dependent branch count, and
so does a coefficient that a parameter could make nonzero. Only values that hold
whatever the state and parameters are, literals and integer indices, may rule an entry
out: a concrete zero coefficient, a concrete branch condition, a concrete index.

Each tangent operation is linear, which is why so few rules are needed. A primitive
with no rule takes the sound answer, every output on every input, with a warning.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
from jax.extend.core import Literal
from scipy import sparse

from diff_soup._warnings import warn_user


def _select(flow, table):
    """The rows of ``flow`` listed in ``table``; a negative entry is an empty row."""
    table = np.asarray(table).ravel()
    valid = table >= 0
    lengths = np.zeros(table.size, dtype=np.int64)
    lengths[valid] = np.diff(flow.indptr)[table[valid]]
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    within = np.arange(indptr[-1]) - np.repeat(indptr[:-1], lengths)
    take = within + np.repeat(np.where(valid, flow.indptr[np.maximum(table, 0)], 0), lengths)
    indices = flow.indices[take]
    data = np.ones(indices.size, dtype=bool)
    return sparse.csr_matrix((data, indices, indptr), shape=(table.size, flow.shape[1]))


def _union(first, second):
    """Row-wise union of two flows, either of which may be ``None``."""
    if first is None or first is second:
        return second
    return first if second is None else first + second


def _gathering(rows, table, columns):
    """Boolean ``rows`` by ``columns`` matrix sending column ``k`` to row ``table[k]``."""
    table = np.asarray(table).ravel()
    keep = table >= 0
    data = np.ones(int(keep.sum()), dtype=bool)
    positions = (table[keep], np.arange(table.size)[keep])
    return sparse.csr_matrix((data, positions), shape=(rows, columns))


def _size(atom):
    """The number of elements ``atom`` holds."""
    return int(np.prod(atom.aval.shape, dtype=np.int64))


class _Scope:
    """One jaxpr being walked: what each variable is reached by, and what it is known to hold."""

    def __init__(self, closed, flows, known, fallbacks):
        """Walk ``closed`` from the flows and known values of its inputs."""
        jaxpr = getattr(closed, "jaxpr", closed)
        self.flows = {var: flow for var, flow in zip(jaxpr.invars, flows) if flow is not None}
        self.known = {var: value for var, value in zip(jaxpr.invars, known) if value is not None}
        for var, value in zip(jaxpr.constvars, getattr(closed, "consts", ())):
            self.known[var] = functools.partial(np.asarray, value)
        self.fallbacks = fallbacks
        for eqn in jaxpr.eqns:
            if any(self.flow(atom) is not None for atom in eqn.invars):
                outputs = _RULES.get(eqn.primitive.name, _conservative)(self, eqn)
                self.flows.update(
                    (var, flow) for var, flow in zip(eqn.outvars, outputs) if flow is not None
                )
            elif all(self.is_known(atom) for atom in eqn.invars):
                self._defer(eqn)
        self.outputs = [self.flow(atom) for atom in jaxpr.outvars]

    def flow(self, atom):
        """The tangent flow that reaches ``atom``, or ``None`` when none does."""
        return None if isinstance(atom, Literal) else self.flows.get(atom)

    def is_known(self, atom):
        """Whether ``atom`` holds one value whatever the state and parameters are."""
        return isinstance(atom, Literal) or atom in self.known

    def value(self, atom):
        """The array ``atom`` holds whatever the state and parameters are, or ``None``."""
        if isinstance(atom, Literal):
            return np.asarray(atom.val)
        thunk = self.known.get(atom)
        return None if thunk is None else thunk()

    def thunk(self, atom):
        """A deferred [value][diff_soup.problem._ad.tangent_pattern._Scope.value] for a callee."""
        return (lambda: self.value(atom)) if self.is_known(atom) else None

    def _defer(self, eqn):
        """Record how to compute ``eqn`` if a rule ever asks: each op compiled costs time."""

        @functools.cache
        def evaluate():
            values = [self.value(atom) for atom in eqn.invars]
            result = eqn.primitive.bind(*values, **eqn.primitive.get_bind_params(eqn.params))
            return result if eqn.primitive.multiple_results else (result,)

        for index, var in enumerate(eqn.outvars):
            self.known[var] = functools.partial(lambda index: np.asarray(evaluate()[index]), index)

    def tables(self, atoms):
        """Row numbers of ``atoms`` in one stacked matrix; a tangent-free array holds -1."""
        tables, blocks, start = [], [], 0
        for atom in atoms:
            flow = self.flow(atom)
            if flow is None:
                tables.append(np.full(atom.aval.shape, -1, dtype=np.int64))
            else:
                tables.append(np.arange(start, start + flow.shape[0]).reshape(atom.aval.shape))
                blocks.append(flow)
                start += flow.shape[0]
        return tables, blocks[0] if len(blocks) == 1 else sparse.vstack(blocks, format="csr")


def _anywhere(scope, eqn):
    """Every output on every input, which is exact for an index that may take any value."""
    flows = [scope.flow(atom) for atom in eqn.invars if scope.flow(atom) is not None]
    columns = np.unique(np.concatenate([flow.indices for flow in flows]))
    outputs = []
    for var in eqn.outvars:
        rows = _size(var)
        indptr = np.arange(rows + 1) * columns.size
        data = np.ones(rows * columns.size, dtype=bool)
        shape = (rows, flows[0].shape[1])
        outputs.append(sparse.csr_matrix((data, np.tile(columns, rows), indptr), shape=shape))
    return outputs


def _conservative(scope, eqn):
    """A primitive with no rule: sound, but it declares more than it may need, so it warns."""
    scope.fallbacks.add(eqn.primitive.name)
    return _anywhere(scope, eqn)


def _broadcast_flow(flow, atom, shape):
    """``flow`` of ``atom``, broadcast to an output ``shape`` as a lax operation does."""
    if atom.aval.shape == tuple(shape):
        return flow
    return _select(flow, np.broadcast_to(np.arange(flow.shape[0]).reshape(atom.aval.shape), shape))


def _elementwise(scope, eqn):
    """Each output element is reached by the matching element of each input."""
    shape, total = eqn.outvars[0].aval.shape, None
    for atom in eqn.invars:
        flow = scope.flow(atom)
        if flow is not None:
            total = _union(total, _broadcast_flow(flow, atom, shape))
    return [total]


def _mul(scope, eqn):
    """A concrete zero coefficient passes no tangent."""
    shape, total = eqn.outvars[0].aval.shape, None
    for atom, other in (eqn.invars, eqn.invars[::-1]):
        flow = scope.flow(atom)
        if flow is None:
            continue
        flow = _broadcast_flow(flow, atom, shape)
        coefficient = scope.value(other)
        if coefficient is not None:
            keep = np.broadcast_to(coefficient != 0, shape).ravel()
            if not keep.all():
                flow = _select(flow, np.where(keep, np.arange(keep.size), -1))
        total = _union(total, flow)
    return [total]


def _select_n(scope, eqn):
    """A concrete condition picks one case per element; otherwise any case may be taken."""
    which = scope.value(eqn.invars[0])
    if which is None:
        return _elementwise(scope, eqn)
    shape = eqn.outvars[0].aval.shape
    which = np.broadcast_to(which, shape).astype(np.int64).ravel()
    total = None
    for case, atom in enumerate(eqn.invars[1:]):
        flow = scope.flow(atom)
        if flow is not None:
            table = np.where(which == case, np.arange(which.size), -1)
            total = _union(total, _select(flow, table))
    return [total]


def _broadcast(eqn, table):
    """Broadcast an index table as ``broadcast_in_dim`` broadcasts its operand."""
    shape, dims = eqn.params["shape"], eqn.params["broadcast_dimensions"]
    expanded = [1] * len(shape)
    for axis, dim in enumerate(dims):
        expanded[dim] = table.shape[axis]
    return np.broadcast_to(table.reshape(expanded), shape)


def _slice(eqn, table):
    """Slice an index table as ``slice`` slices its operand."""
    starts, limits = eqn.params["start_indices"], eqn.params["limit_indices"]
    strides = eqn.params["strides"] or (1,) * table.ndim
    return table[tuple(slice(*spec) for spec in zip(starts, limits, strides))]


_RESHAPING = {
    "reshape": lambda eqn, table: table.reshape(eqn.params["new_sizes"]),
    "squeeze": lambda eqn, table: table.squeeze(tuple(eqn.params["dimensions"])),
    "transpose": lambda eqn, table: table.transpose(eqn.params["permutation"]),
    "rev": lambda eqn, table: np.flip(table, tuple(eqn.params["dimensions"])),
    "broadcast_in_dim": _broadcast,
    "slice": _slice,
    "concatenate": lambda eqn, *tables: np.concatenate(tables, eqn.params["dimension"]),
    "stack": lambda eqn, *tables: np.stack(tables, eqn.params["axis"]),
}


def _reshaping(scope, eqn):
    """Outputs are inputs moved around: run the operation on the row numbers."""
    tables, stacked = scope.tables(eqn.invars)
    return [_select(stacked, _RESHAPING[eqn.primitive.name](eqn, *tables))]


# Operations on row numbers that JAX evaluates, and which of their inputs carry data.
_INDEXED = {
    "gather": (0,),
    "dynamic_slice": (0,),
    "dynamic_update_slice": (0, 1),
    "pad": (0, 1),
    "split": (0,),
}


def _indexed(scope, eqn):
    """As [_reshaping][], when the operation is left to JAX and may need concrete indices."""
    data = _INDEXED[eqn.primitive.name]
    atoms = [eqn.invars[position] for position in data]
    tables, stacked = scope.tables(atoms)
    inputs = dict(zip(data, tables))
    for position, atom in enumerate(eqn.invars):
        if position not in inputs:
            inputs[position] = scope.value(atom)
            if inputs[position] is None or scope.flow(atom) is not None:
                return _anywhere(scope, eqn)
    params = dict(eqn.params)
    if "fill_value" in params:
        params["fill_value"] = -1
    result = eqn.primitive.bind(*(inputs[position] for position in range(len(inputs))), **params)
    results = result if eqn.primitive.multiple_results else (result,)
    return [_select(stacked, np.asarray(table)) for table in results]


def _scatter(scope, eqn):
    """A write sends each update element to the operand element it lands on."""
    operand, indices, updates = eqn.invars
    where = scope.value(indices)
    if where is None or scope.flow(indices) is not None:
        return _anywhere(scope, eqn)
    dims, mode = eqn.params["dimension_numbers"], eqn.params["mode"]
    shape = operand.aval.shape
    window = iter(updates.aval.shape[axis] for axis in dims.update_window_dims)
    collapsed = (*dims.inserted_window_dims, *dims.operand_batching_dims)
    target = jax.lax.gather(
        np.arange(_size(operand)).reshape(shape),
        where,
        jax.lax.GatherDimensionNumbers(
            offset_dims=dims.update_window_dims,
            collapsed_slice_dims=dims.inserted_window_dims,
            start_index_map=dims.scatter_dims_to_operand_dims,
            operand_batching_dims=dims.operand_batching_dims,
            start_indices_batching_dims=dims.scatter_indices_batching_dims,
        ),
        tuple(1 if axis in collapsed else next(window) for axis in range(len(shape))),
        mode=mode,
        fill_value=-1 if mode == jax.lax.GatherScatterMode.FILL_OR_DROP else None,
    )
    target = np.asarray(target).ravel()
    kept = scope.flow(operand)
    written = np.unique(target[target >= 0])
    if kept is not None and eqn.primitive.name == "scatter" and written.size == np.sum(target >= 0):
        table = np.arange(kept.shape[0])
        table[written] = -1
        kept = _select(kept, table)
    inflow = scope.flow(updates)
    if inflow is not None:
        inflow = (_gathering(_size(operand), target, inflow.shape[0]) @ inflow).tocsr()
    return [_union(kept, inflow)]


def _flat(coordinates, shape, length):
    """Flat indices of ``length`` elements, also for the single element of a scalar."""
    if coordinates:
        return np.ravel_multi_index(tuple(coordinates), shape)
    return np.zeros(length, dtype=np.int64)


def _reduce(scope, eqn):
    """Each output element reads every input element it sums."""
    atom = eqn.invars[0]
    shape, flow = atom.aval.shape, scope.flow(atom)
    axes = tuple(int(axis) for axis in eqn.params["axes"])
    kept = [axis for axis in range(len(shape)) if axis not in axes]
    grid = np.indices(shape)
    coordinates = [grid[axis].ravel() for axis in kept]
    target = _flat(coordinates, tuple(shape[axis] for axis in kept), flow.shape[0])
    return [(_gathering(_size(eqn.outvars[0]), target, flow.shape[0]) @ flow).tocsr()]


def _dot(scope, eqn):
    """Output ``(batch, left free, right free)`` reads each side where the other can be nonzero."""
    (left_c, right_c), (left_b, right_b) = eqn.params["dimension_numbers"]
    left, right = eqn.invars
    out_shape = eqn.outvars[0].aval.shape
    total = None
    sides = (
        (left, right, left_c, right_c, left_b, right_b),
        (right, left, right_c, left_c, right_b, left_b),
    )
    for side, (atom, other, contract, other_c, batch, other_b) in enumerate(sides):
        flow = scope.flow(atom)
        if flow is None:
            continue
        shape = atom.aval.shape
        coefficient = scope.value(other)
        spots = np.argwhere(np.ones(other.aval.shape, bool) if coefficient is None else coefficient)
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
            shape=(_size(eqn.outvars[0]), flow.shape[0]),
        )
        total = _union(total, (matrix @ flow).tocsr())
    return [total]


def _walk(closed, scope, atoms):
    """The tangent reach of a called jaxpr's outputs, given its caller's ``atoms``."""
    flows = [scope.flow(atom) for atom in atoms]
    known = [scope.thunk(atom) for atom in atoms]
    return _Scope(closed, flows, known, scope.fallbacks).outputs


def _call(scope, eqn):
    """Walk the jaxpr a call primitive runs."""
    closed = eqn.params["call_jaxpr"] if "call_jaxpr" in eqn.params else eqn.params["jaxpr"]
    return _walk(closed, scope, eqn.invars)


def _cond(scope, eqn):
    """A concrete index picks a branch; otherwise any branch may be taken."""
    index, *operands = eqn.invars
    branches = eqn.params["branches"]
    picked = scope.value(index)
    if picked is not None:
        branches = (branches[int(np.clip(picked, 0, len(branches) - 1))],)
    outputs = [None] * len(eqn.outvars)
    for branch in branches:
        results = _walk(branch, scope, operands)
        outputs = [_union(a, b) for a, b in zip(outputs, results)]
    return outputs


_RULES = {
    **dict.fromkeys(
        (
            "add",
            "add_any",
            "sub",
            "neg",
            "div",
            "convert_element_type",
            "copy",
            "real",
            "imag",
            "complex",
            "conj",
            "reduce_precision",
        ),
        _elementwise,
    ),
    **dict.fromkeys(_RESHAPING, _reshaping),
    **dict.fromkeys(_INDEXED, _indexed),
    **dict.fromkeys(("scatter", "scatter-add"), _scatter),
    **dict.fromkeys(("jit", "closed_call", "core_call", "checkpoint"), _call),
    "mul": _mul,
    "select_n": _select_n,
    "reduce_sum": _reduce,
    "dot_general": _dot,
    "cond": _cond,
}


def _host_device():
    """The CPU, where each constant a rule asks for compiles fastest, if JAX can reach it."""
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


def tangent_pattern(fn, y, *args):
    """Return the entries of ``d fn(y, *args) / dy`` that can ever be nonzero.

    Args:
        fn: Function of ``y`` and ``args`` returning a one-dimensional array. A value it
            closes over counts as fixed; anything that may vary goes in ``args``.
        y: One-dimensional array, read only for its shape and dtype.
        *args: Pytrees standing for any value of their shapes and dtypes.

    Returns:
        ``(rows, cols)``, sorted row-major.
    """
    size = int(y.size)

    def tangent(dy, value, rest):
        return jax.jvp(lambda current: fn(current, *rest), (value,), (dy,))[1]

    with jax.default_device(_host_device()):
        closed = jax.make_jaxpr(tangent)(y, y, args)
        count = len(closed.jaxpr.invars)
        flows = [sparse.eye(size, dtype=bool, format="csr")] + [None] * (count - 1)
        fallbacks = set()
        output = _Scope(closed, flows, [None] * count, fallbacks).outputs[0]
    if fallbacks:
        warn_user(
            f"tangent_pattern: primitive(s) {sorted(fallbacks)} have no pattern rule, so each "
            "of their outputs is declared to read each of their inputs."
        )
    if output is None:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    coo = output.tocoo()
    keys = np.unique(coo.row.astype(np.int64) * coo.shape[1] + coo.col)
    return (keys // coo.shape[1]).astype(np.int32), (keys % coo.shape[1]).astype(np.int32)
