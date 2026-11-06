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
A primitive with no rule, an index not known at build time and a loop whose reach does
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


def tangent_flow(fn, y, *args):
    """Return the flow of ``d fn(y, *args) / dy``: a boolean CSR matrix of its reachable entries.

    Args:
        fn: Function of ``y`` and ``args`` returning one array.
        y: Array, read only for its shape and dtype.
        *args: Pytrees standing for any value of their shapes and dtypes.

    Returns:
        A boolean ``scipy.sparse.csr_matrix`` of shape ``(fn(y).size, y.size)``.

    The function is traced, and its known part compiled, outside any trace or
    ``jax.ensure_compile_time_eval`` of the caller's (see
    [outside_compile_time_eval][diff_soup.problem._ad.pattern.outside_compile_time_eval]).
    """
    size = int(np.size(y))

    def tangent(dy, value, rest):
        return jax.jvp(lambda current: fn(current, *rest), (value,), (dy,))[1]

    def lowered():
        with jax.default_device(_host_device()):
            closed = jax.make_jaxpr(tangent)(y, y, args)
            return closed, Program(closed)

    closed, program = outside_compile_time_eval(lowered)
    walk = Walk(program, size)
    seeds = sparse.identity(size, dtype=bool, format="csr")
    inputs = [seeds] + [None] * (len(program.top.invars) - 1)
    output = walk.block(program.top, inputs)[0]
    rows = int(np.prod(closed.out_avals[0].shape, dtype=np.int64))
    return sparse.csr_matrix((rows, size), dtype=bool) if output is None else output


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
    coo = tangent_flow(fn, y, *args).tocoo()
    return dedup_coo(coo.row, coo.col, coo.shape[1])[:2]
