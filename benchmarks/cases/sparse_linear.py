"""Sparse coloring and native cuDSS batch cases."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._core import Case, Workload

_ALL = ("pr-cpu", "nightly-cpu", "nightly-gpu", "release")
_GPU = ("nightly-gpu", "release")


def _sparse_coloring() -> Workload:
    from diff_soup.utils import (
        build_padded_color_arrays,
        build_sparse_jvp_groups,
        sparse_jvp_jacobian_values,
    )

    size = 32
    pattern = [
        (row, column)
        for row in range(size)
        for column in (row - 1, row, row + 1)
        if 0 <= column < size
    ]
    rows = np.asarray([row for row, _ in pattern], dtype=np.int32)
    cols = np.asarray([column for _, column in pattern], dtype=np.int32)
    colors = build_padded_color_arrays(*build_sparse_jvp_groups(rows, cols, size))
    state = jnp.linspace(0.2, 1.0, size)

    def residual(value):
        result = value**2
        result = result.at[1:].add(0.1 * value[:-1])
        return result.at[:-1].add(-0.2 * value[1:])

    def function(value):
        return sparse_jvp_jacobian_values(residual, value, *colors, rows.size)

    def check(value):
        expected = jax.jacfwd(residual)(state)[rows, cols]
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 1e-12, error, 1e-12

    return Workload(function, (state,), check)


def _cudss_native_batch() -> Workload:
    from diff_soup.linear import CuDSS, SolveContext, SparseTopology, analyze, factor, solve

    if CuDSS is None:
        raise RuntimeError("cuDSS case requires the CUDA extra and native extension")
    rows = np.arange(32, dtype=np.int32)
    analysis = analyze(SparseTopology(rows=rows, cols=rows, n_dofs=32), backend=CuDSS())
    # The factors live in this context for the life of the benchmark process.
    context = SolveContext(name="cudss-independent-matrix-vmap-16")
    base = jnp.linspace(2.0, 4.0, 32)
    rhs = jnp.linspace(1.0, 2.0, 32)
    scales = jnp.linspace(1.0, 2.0, 16)

    def run(scale):
        return solve(factor(analysis, scale * base, ctx=context), rhs)[0]

    function = jax.vmap(run)

    def check(value):
        expected = rhs[None, :] / (scales[:, None] * base[None, :])
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 1e-11, error, 1e-11

    return Workload(function, (scales,), check, CuDSS.contract_stats)


def cases() -> tuple[Case, ...]:
    """Return sparse differentiation and direct native batch cases."""
    return (
        Case(
            "sparse-coloring-vmap",
            "1",
            _ALL,
            _sparse_coloring,
            "none",
            "xla",
            "vmap-jvp",
            "representative-sparse-pde",
            {"states": 32, "parameters": 32},
            {"nnz": 94, "density": 0.091796875},
            3,
            3,
        ),
        Case(
            "cudss-independent-matrix-vmap-16",
            "1",
            _GPU,
            _cudss_native_batch,
            "none",
            "cudss-1.0",
            "vmap",
            "diagonal-sparse-linear",
            {"states": 32, "parameters": 16},
            {"nnz": 32, "density": 0.03125},
            16,
            0,
            expected_targets=(
                "diff_soup_cudss_contract_factor_coo_f64",
                "diff_soup_cudss_contract_solve_coo_f64",
            ),
            required_device="gpu",
        ),
    )
