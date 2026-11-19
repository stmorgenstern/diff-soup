"""Throwaway: time jax.ops.segment_sum on the card, float64 against complex128 and a split complex sum.

The shape is that of MemberSchur's reduce_rhs on the cell: P members each add one row into one shared row
(the global row that reads every particle), plus a few distinct rows.
"""
import json
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def timeit(f, *a, reps=20):
    jax.block_until_ready(f(*a))
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(*a)); t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3


out = []
for P in (240, 2004, 10008):
    rng = np.random.default_rng(0)
    nseg = P + 27
    index = np.concatenate([np.full(P, 23), 27 + np.arange(P)])  # every member hits row 23, and its own row
    seg = jax.jit(lambda r: jax.ops.segment_sum(r, index, num_segments=nseg))
    split = jax.jit(lambda r: jax.ops.segment_sum(r.real, index, num_segments=nseg)
                    + 1j * jax.ops.segment_sum(r.imag, index, num_segments=nseg))
    at = jax.jit(lambda r: jnp.zeros((nseg, 1), r.dtype).at[index].add(r))
    xr = jnp.asarray(rng.standard_normal((index.size, 1)))
    xc = xr + 1j * jnp.asarray(rng.standard_normal((index.size, 1)))
    row = dict(P=P, device=str(jax.devices()[0]),
               seg_f64_ms=timeit(seg, xr), seg_c128_ms=timeit(seg, xc), split_c128_ms=timeit(split, xc),
               at_f64_ms=timeit(at, xr), at_c128_ms=timeit(at, xc),
               split_equal=bool(np.allclose(np.asarray(seg(xc)), np.asarray(split(xc)))))
    print(json.dumps(row), flush=True)
    out.append(row)
