"""The exact pattern of a Jacobian, read from a function's tangent map.

An entry ``df_i/dy_j`` can be nonzero only if a tangent seeded at ``y_j`` reaches
``f_i``. [tangent_pattern][diff_soup.problem._ad.tangent_pattern.tangent_pattern]
walks the jaxpr of ``jax.jvp`` once, carrying for each array one boolean row per
element: which seeds reach it. No state is evaluated, so the answer does not depend
on values the residual may not be finite at, and an abstract value, such as a state
or a parameter, stands for anything: both sides of a state-dependent branch count, and
so does a coefficient that a parameter could make nonzero. Only values that hold
whatever the state and parameters are, literals, closed-over arrays and what is
computed from them alone, may rule an entry out: a zero coefficient, a branch
condition, an index. Those values are computed by one compiled program.

Each tangent operation is linear, which is why so few rules are needed (see
[pattern_rules][diff_soup.problem._ad.pattern_rules] and
[pattern_structured][diff_soup.problem._ad.pattern_structured]); another module adds
one with [register_pattern_rule][diff_soup.problem._ad.pattern_walk.register_pattern_rule].
A gather at an index not known at build time reads its window at every start. A
primitive with no rule, a slice or a write at such an index and a loop whose reach does
not settle take the sound answer, every output on every input, with a warning at the
first such operation.
"""

from __future__ import annotations

import jax
import numpy as np
from scipy import sparse

from diff_soup.problem._ad import pattern_rules, pattern_structured  # noqa: F401
from diff_soup.problem._ad.pattern import dedup_coo, outside_compile_time_eval
from diff_soup.problem._ad.pattern_walk import Program, Walk


def _host_device():
    """The CPU, where the known values compile fastest, if JAX can reach it."""
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


def tangent_flow(fn, y, *args, taint: bool = False):
    """Return the flow of ``d fn(y, *args) / dy``: a boolean CSR matrix of its reachable entries.

    Args:
        fn: Function of ``y`` and ``args`` returning one array.
        y: Array, read only for its shape and dtype.
        *args: Pytrees standing for any value of their shapes and dtypes.
        taint: Add a last column, set on each row a fallback decided.

    Returns:
        A boolean ``scipy.sparse.csr_matrix`` of shape ``(fn(y).size, y.size)``, one column
        more with ``taint``.

    The function is traced, and its known part compiled, outside any trace or
    ``jax.ensure_compile_time_eval`` of the caller's (see
    [outside_compile_time_eval][diff_soup.problem._ad.pattern.outside_compile_time_eval]).
    """
    size = int(np.size(y))
    seeds = size + 1 if taint else size

    def tangent(dy, value, rest):
        return jax.jvp(lambda current: fn(current, *rest), (value,), (dy,))[1]

    def lowered():
        with jax.default_device(_host_device()):
            closed = jax.make_jaxpr(tangent)(y, y, args)
            return closed, Program(closed)

    closed, program = outside_compile_time_eval(lowered)
    walk = Walk(program, seeds, size if taint else None)
    identity = sparse.eye(size, seeds, dtype=bool, format="csr")
    inputs = [identity] + [None] * (len(program.top.invars) - 1)
    output = walk.block(program.top, inputs)[0]
    rows = int(np.prod(closed.out_avals[0].shape, dtype=np.int64))
    return sparse.csr_matrix((rows, seeds), dtype=bool) if output is None else output


def jacobian_pattern(fn, y, *args, within=None):
    """Return the entries of `d fn(y, *args) / dy` that can be nonzero.

    The pattern is read from the tangent map of `fn` (`jax.jvp`), with no evaluation of
    `fn`: a state or a parameter stands for any value, so both sides of a branch on the
    state count, and only a value that holds whatever the inputs are, such as a literal, a
    closed-over array or an index computed from them, rules an entry out. A gather at an
    index the inputs set reads its window at every start. An operation no rule covers, a
    slice or a write at such an index, and a loop whose reach does not settle make each
    output they reach read every input that reaches them, and warn.

    ```python
    rows, cols = ds.jacobian_pattern(residual, y, theta)
    ```

    Args:
        fn: Function of `y` and `args` returning one array, read as flat.
        y: One-dimensional array, read only for its shape and dtype.
        *args: Pytrees standing for any value of their shapes and dtypes. A value `fn`
            closes over counts as fixed.
        within: `(rows, cols)` holding every entry that can be nonzero. Each row a
            fallback reaches keeps only its entries in `within`.

    Returns:
        `(rows, cols)`, two `int32` arrays sorted row-major.
    """
    size = int(np.size(y))
    flow = tangent_flow(fn, y, *args, taint=within is not None).tocoo()
    rows, cols = flow.row, flow.col
    if within is not None:
        tainted = np.zeros(flow.shape[0], dtype=bool)
        tainted[rows[cols == size]] = True
        bound = dedup_coo(*within, size)
        allowed = np.isin(
            rows.astype(np.int64) * size + cols,
            bound.rows.astype(np.int64) * size + bound.cols,
        )
        keep = (cols < size) & (~tainted[rows] | allowed)
        rows, cols = rows[keep], cols[keep]
    return dedup_coo(rows, cols, size)[:2]


def tangent_pattern(fn, y, *args):
    """Return [jacobian_pattern][diff_soup.problem.jacobian_pattern] with no bound."""
    return jacobian_pattern(fn, y, *args)
