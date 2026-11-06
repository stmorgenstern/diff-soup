"""Pattern rules for primitives that run a jaxpr or a window: branches, loops, solves, convolutions.

A loop either is walked once per iteration, when its trip count times its body is
small, or its carry is walked to a fixed point: the union of every iteration's reach,
reached in at most as many passes as the loop has iterations.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from diff_soup.problem._ad import flows as F
from diff_soup.problem._ad.pattern_walk import Walk, register_pattern_rule, scan_counts

# Ours: body operations a scan may walk in total before its carry goes to a fixed point.
_UNROLLED_OPERATIONS = 20_000
# Ours: passes a carry may take to settle before every result reads every input.
_FIXED_POINT_PASSES = 256


def _same(first, second) -> bool:
    """Whether two flows, either of which may be ``None``, reach the same entries."""
    if first is None or second is None:
        return first is second
    return first.nnz == second.nnz and (first != second).nnz == 0


def _empty(rows: int, seeds: int):
    """A flow of ``rows`` elements that no tangent reaches."""
    return sparse.csr_matrix((rows, seeds), dtype=bool)


def _cond(walk, op):
    """A known index picks a branch; otherwise any branch may be taken."""
    index, *operands = op.invars
    blocks = op.blocks
    picked = walk.value(index)
    if picked is not None:
        blocks = (blocks[int(np.clip(picked, 0, len(blocks) - 1))],)
    inputs = [walk.flow(atom) for atom in operands]
    outputs = [None] * len(op.outvars)
    for block in blocks:
        outputs = [F.union(a, b) for a, b in zip(outputs, walk.block(block, inputs))]
    return outputs


def _settle(step, state, passes: int):
    """Union ``state`` with ``step(state)`` for at most ``passes``; also whether it settled."""
    for _ in range(passes):
        grown = [F.union(old, new) for old, new in zip(state, step(state))]
        if all(_same(a, b) for a, b in zip(grown, state)):
            return grown, True
        state = grown
    return state, False


def _scan(walk, op):
    """Walk each iteration when the loop is small, else the carry to a fixed point."""
    params, body = op.params, op.blocks[0]
    consts, carries = scan_counts(params)
    if len(op.outvars) != len(body.outvars):
        return walk.fallback(op, "forwards an input to its outputs, which no rule covers")
    length = int(params["length"])
    const_flows = [walk.flow(atom) for atom in op.invars[:consts]]
    carry = [walk.flow(atom) for atom in op.invars[consts : consts + carries]]
    xs = [walk.flow(atom) for atom in op.invars[consts + carries :]]
    y_rows = [walk.size(var) // max(length, 1) for var in op.outvars[carries:]]
    if length * max(len(body.ops), 1) <= _UNROLLED_OPERATIONS:
        ys = [[None] * length for _ in y_rows]
        for i in reversed(range(length)) if params["reverse"] else range(length):
            sliced = [
                None if f is None else F.select(f, np.arange(f.shape[0]).reshape(length, -1)[i])
                for f in xs
            ]
            out = walk.block(body, const_flows + carry + sliced)
            carry = out[:carries]
            for column, flow in zip(ys, out[carries:]):
                column[i] = flow
        stacked = [
            None
            if all(flow is None for flow in column)
            else F.stack([_empty(rows, walk.seeds) if f is None else f for f in column], walk.seeds)
            for column, rows in zip(ys, y_rows)
        ]
        return [*carry, *stacked]
    merged = [
        None
        if f is None
        else (
            F.gathering(
                f.shape[0] // length, np.tile(np.arange(f.shape[0] // length), length), f.shape[0]
            )
            @ f
        ).tocsr()
        for f in xs
    ]
    passes = min(length, _FIXED_POINT_PASSES)
    step = lambda c: walk.block(body, const_flows + c + merged)[:carries]  # noqa: E731
    settled, done = _settle(step, carry, passes)
    if not done and length > passes:
        return walk.fallback(op, "carries a reach that did not settle")
    ys = walk.block(body, const_flows + settled + merged)[carries:]
    return [
        *settled,
        *(
            None if f is None else F.select(f, np.tile(np.arange(rows), length))
            for f, rows in zip(ys, y_rows)
        ),
    ]


def _while(walk, op):
    """The carry walked to a fixed point: a while loop may run any number of iterations."""
    params, body = op.params, op.blocks[1]
    cond_n, body_n = params["cond_nconsts"], params["body_nconsts"]
    const_flows = [walk.flow(atom) for atom in op.invars[cond_n : cond_n + body_n]]
    carry = [walk.flow(atom) for atom in op.invars[cond_n + body_n :]]
    settled, done = _settle(lambda c: walk.block(body, const_flows + c), carry, _FIXED_POINT_PASSES)
    if not done:
        return walk.fallback(op, "carries a reach that did not settle")
    return settled


def _linear_solve(walk, op):
    """``x = A^-1 b`` reads ``b`` where ``A``'s graph reaches: the closure of its pattern."""
    count = sum(op.params["const_lengths"])
    consts, rhs = op.invars[:count], op.invars[count:]
    if any(walk.flow(atom) is not None for atom in consts):
        return walk.fallback(op, "solves with a matrix a tangent reaches")
    sizes = [walk.size(atom) for atom in rhs]
    edges = np.cumsum([0, *sizes])
    total = int(edges[-1])
    identity = sparse.identity(total, dtype=bool, format="csr")
    inner = Walk(walk.program, total)
    inner.fallbacks = walk.fallbacks
    matvec = op.blocks[0]
    seeds = [identity[edges[i] : edges[i + 1]] for i in range(len(rhs))]
    leading = [None] * (len(matvec.invars) - len(rhs))
    pattern = F.stack(
        [
            _empty(size, total) if f is None else f
            for f, size in zip(inner.block(matvec, leading + seeds), sizes)
        ],
        total,
    )
    reach = identity
    for _ in range(total):
        grown = (reach + (pattern @ reach)).tocsr()
        if grown.nnz == reach.nnz:
            break
        reach = grown
    flows = [walk.flow(atom) for atom in rhs]
    stacked = F.stack(
        [_empty(size, walk.seeds) if f is None else f for f, size in zip(flows, sizes)], walk.seeds
    )
    solved = (reach @ stacked).tocsr()
    return [solved[edges[i] : edges[i + 1]] for i in range(len(rhs))]


def _conv_taps(params, lhs_shape, rhs_shape, out_shape, numbers):
    """Per spatial axis, the ``(output, kernel tap, input)`` triples a convolution pairs."""
    taps = []
    for axis in range(len(numbers.lhs_spec) - 2):
        n = lhs_shape[numbers.lhs_spec[2 + axis]]
        width = rhs_shape[numbers.rhs_spec[2 + axis]]
        m = out_shape[numbers.out_spec[2 + axis]]
        out, tap = np.meshgrid(np.arange(m), np.arange(width), indexing="ij")
        dilated = (
            out * params["window_strides"][axis]
            + tap * params["rhs_dilation"][axis]
            - params["padding"][axis][0]
        )
        stride = params["lhs_dilation"][axis]
        ok = (dilated >= 0) & (dilated % stride == 0) & (dilated // stride < n)
        taps.append((out[ok], tap[ok], dilated[ok] // stride))
    return taps


def _conv(walk, op):
    """An output element reads its window of the input through every kernel tap that may be nonzero."""
    params = op.params
    if params["batch_group_count"] != 1:
        return walk.fallback(op, "has batch groups, which no rule covers")
    lhs, rhs = op.invars
    numbers = params["dimension_numbers"]
    shapes = walk.shape(lhs), walk.shape(rhs), walk.shape(op.outvars[0])
    taps = _conv_taps(params, *shapes, numbers)
    picks = [g.ravel() for g in np.meshgrid(*(np.arange(t[0].size) for t in taps), indexing="ij")]
    spatial = [[t[part][pick] for t, pick in zip(taps, picks)] for part in range(3)]
    groups = params["feature_group_count"]
    out_features = shapes[2][numbers.out_spec[1]]
    per_in = shapes[0][numbers.lhs_spec[1]] // groups
    feature, inner = (
        g.ravel() for g in np.meshgrid(np.arange(out_features), np.arange(per_in), indexing="ij")
    )
    channel = feature // (out_features // groups) * per_in + inner
    full = (shapes[0][numbers.lhs_spec[0]], feature.size, picks[0].size if picks else 1)

    def index(spec, shape, first, second, positions):
        coordinates = [None] * len(shape)
        coordinates[spec[0]], coordinates[spec[1]] = first, second
        for axis, position in enumerate(positions):
            coordinates[spec[2 + axis]] = position[None, None, :]
        broadcast = (np.broadcast_to(c, full) for c in coordinates)
        return np.ravel_multi_index(tuple(broadcast), shape).ravel()

    batch = np.arange(full[0])[:, None, None]
    out_index = index(numbers.out_spec, shapes[2], batch, feature[None, :, None], spatial[0])
    lhs_index = index(numbers.lhs_spec, shapes[0], batch, channel[None, :, None], spatial[2])
    rhs_index = index(
        numbers.rhs_spec,
        shapes[1],
        feature[None, :, None] + 0 * batch,
        inner[None, :, None],
        spatial[1],
    )
    total = None
    for atom, own, other, other_index in (
        (lhs, lhs_index, rhs, rhs_index),
        (rhs, rhs_index, lhs, lhs_index),
    ):
        flow = walk.flow(atom)
        if flow is None:
            continue
        value = walk.value(other)
        keep = (
            np.ones(own.size, bool)
            if value is None
            else np.asarray(value).ravel()[other_index] != 0
        )
        matrix = sparse.csr_matrix(
            (np.ones(int(keep.sum()), bool), (out_index[keep], own[keep])),
            shape=(walk.size(op.outvars[0]), flow.shape[0]),
        )
        total = F.union(total, (matrix @ flow).tocsr())
    return [total]


register_pattern_rule("cond", _cond)
register_pattern_rule("scan", _scan)
register_pattern_rule("while", _while)
register_pattern_rule("custom_linear_solve", _linear_solve)
register_pattern_rule("conv_general_dilated", _conv)
