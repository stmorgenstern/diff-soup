"""Linear contract, cuDSS: the cuDSS version check, device-resident storage, statuses, both
orientations of single and batched factors, batched refactors, symmetric declarations, the
settings of the backend object over the layout, replay, handler dispatch, and shared storage per
device. Every row but the symmetric-declaration refusal, the settings rule on a stand-in backend
and the host backends' per-device storage rows is ``requires_cudss``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    KLU,
    CuDSS,
    Dense,
    LinearSolver,
    SolveContext,
    SolveStatus,
    SparseTopology,
    SuperLU,
    analyze,
    empty_factor,
    factor,
    invalidate,
    refactor,
    solve,
    solve_checked,
)

N = 8
_ROOT = Path(__file__).resolve().parents[2]
cudss_only = pytest.mark.requires_cudss
# measured 2.1e-16 on the dense reference of these systems (CPU VM 2026-10-22,
# tests/contracts/test_linear_dense.py); the cuDSS measurement is owed with these rows
EXACT = {"rtol": 1e-10, "atol": 1e-12}


def _system(n=N, seed=7, dirichlet=False):
    pairs = [(i, j) for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n]
    pairs += [(0, 5), (2, 7), (6, 1), (4, 0), (n // 2, n // 2)]
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
    values = np.random.default_rng(seed).uniform(-1, 1, rows.size) + np.where(
        rows == cols, 4.0, 0.0
    )
    bc = np.asarray((1, 6), np.int32) if dirichlet else None
    return SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc), jnp.asarray(values)


def _symmetric(dirichlet=False, seed=13):
    rng = np.random.default_rng(seed)
    entries = {(i, i): 4.0 + rng.uniform() for i in range(N)}
    for i, j in sorted({(i, i + 1) for i in range(N - 1)} | {(0, 5), (2, 7), (1, 6), (0, 4)}):
        entries[(i, j)] = entries[(j, i)] = rng.uniform(-1.0, 1.0)
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*sorted(entries)))
    bc = np.asarray((1, 6), np.int32) if dirichlet else None
    return SparseTopology(rows=rows, cols=cols, n_dofs=N, bc_indices=bc), jnp.asarray(
        [entries[k] for k in sorted(entries)]
    )


def _saddle():
    """An 8 x 8 symmetric saddle-point matrix: a 6 x 6 positive definite block, two constraint
    rows and a zero diagonal block."""
    entries = {}
    for i in range(N - 2):
        entries[(i, i)] = 4.0
        if i + 1 < N - 2:
            entries[(i, i + 1)] = entries[(i + 1, i)] = -1.0
    for row, columns in ((N - 2, (0, 2, 5)), (N - 1, (1, 3, 4))):
        for column in columns:
            entries[(row, column)] = entries[(column, row)] = 1.0
    keys = sorted(entries)
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*keys))
    return SparseTopology(rows=rows, cols=cols, n_dofs=N), jnp.asarray([entries[k] for k in keys])


def _matrix(topology, values):
    a = np.zeros((topology.n_dofs, topology.n_dofs))
    np.add.at(a, (topology.rows, topology.cols), np.asarray(values))
    if topology.bc_indices is not None:
        a[topology.bc_indices] = 0.0
        a[topology.bc_indices, topology.bc_indices] = 1.0
    return a


_B = jnp.linspace(-1.0, 1.0, N)


def _steady(backend, program, *args):
    """``(output, counter increments)`` of a second run of ``program`` (``live_*`` excluded)."""
    jax.block_until_ready(program(*args))
    before = backend.contract_stats()
    out = jax.block_until_ready(program(*args))
    after = backend.contract_stats()
    return out, {k: v - before.get(k, 0) for k, v in after.items() if not k.startswith("live_")}


@cudss_only
def test_the_extension_refuses_a_cudss_library_of_another_major_version():
    """The extension reports cuDSS 1.x for its build and for the library this process loaded; an
    extension whose two majors differ, or that reports no version (a cuDSS 0.x build), is refused
    with a ConfigurationError naming the versions."""
    from diff_soup.linear._cudss._ffi_loader import check_cudss_version, native_solver

    built, loaded = native_solver.cudss_built_version(), native_solver.cudss_version()
    assert built[0] == loaded[0] == 1
    check_cudss_version(native_solver, 12)
    other = SimpleNamespace(cudss_built_version=lambda: (1, 0, 0), cudss_version=lambda: (2, 1, 0))
    with pytest.raises(ds.ConfigurationError, match=r"cuDSS 1\.0\.0, but .* is 2\.1\.0"):
        check_cudss_version(other, 12)
    with pytest.raises(ds.ConfigurationError, match="reports no cuDSS version"):
        check_cudss_version(SimpleNamespace(), 12)


@cudss_only
def test_statuses_and_precision_rungs_follow_the_contract():
    """A zero row reports SINGULAR and a NaN NONFINITE with no finite entry; the fp32 rung
    solves in float32 to single-precision accuracy and the fp64 rung exactly."""
    backend = CuDSS()
    topology, values = _system()
    analysis = analyze(topology, backend=backend)
    with SolveContext() as ctx:
        for broken, status in (
            (jnp.where(jnp.asarray(topology.rows == 5), 0.0, values), SolveStatus.SINGULAR),
            (values.at[0].set(jnp.nan), SolveStatus.NONFINITE),
        ):
            x, _, report = solve_checked(factor(analysis, broken, ctx=ctx), _B)
            assert int(report.status) == status and not np.isfinite(np.asarray(x)).any()
        exact = np.linalg.solve(_matrix(topology, values), _B)
        # measured 3.7e-8 for the dense fp32 rung (CPU VM 2026-10-22); the cuDSS value is owed
        for rung, dtype, bound in (("fp64", jnp.float64, 1e-10), ("fp32", jnp.float32, 1e-4)):
            x = solve(
                factor(analyze(topology, backend=backend, precision=rung), values, ctx=ctx), _B
            )[0]
            assert x.dtype == dtype
            np.testing.assert_allclose(np.asarray(x, np.float64), exact, rtol=bound, atol=bound)


@cudss_only
@pytest.mark.parametrize("declaration", ["general", "symmetric", "spd"])
def test_a_factor_solves_both_orientations_from_its_own_factorization(declaration):
    """``solve(f, b, transpose=True)`` and a factor of orientation ``"AT"`` each solve with
    ``A^T`` from one factorization of ``A``: the transposed solve factors nothing and the "AT"
    factor once, both in cuDSS's transpose solve mode on a general declaration
    (``transposed_solves`` grows by two) and in normal mode on a symmetric or spd one."""
    backend = CuDSS()
    topology, values = _system() if declaration == "general" else _symmetric()
    a = _matrix(topology, values)
    analysis = analyze(
        topology, backend=backend, **({} if declaration == "general" else {declaration: True})
    )
    with SolveContext(f"orientations-{declaration}") as ctx:
        f = factor(analysis, values, ctx=ctx)
        jax.block_until_ready(solve(f, _B)[0])
        before = backend.contract_stats()
        outputs = [
            solve(f, _B, transpose=True)[0],
            solve(factor(analysis, values, ctx=ctx, orientation="AT"), _B)[0],
        ]
        jax.block_until_ready(outputs)
        after = backend.contract_stats()
    delta = {key: after[key] - before[key] for key in ("factorizations", "transposed_solves")}
    assert delta == {"factorizations": 1, "transposed_solves": 2 if declaration == "general" else 0}
    for x in outputs:
        np.testing.assert_allclose(x, np.linalg.solve(a.T, _B), **EXACT)


def _lanes(values, count=4):
    """``count`` lanes of ``values``, lane ``k`` scaled by ``1 + k / 10``."""
    return jnp.stack([values * (1.0 + 0.1 * k) for k in range(count)])


@cudss_only
@pytest.mark.parametrize("layout", ["stacked", "uniform"])
def test_a_native_batch_solves_both_orientations_from_its_own_factorization(layout):
    """Under ``vmap`` a batched factor of either ``batch_layout`` solves with ``A^T`` (one and two
    right-hand sides per lane) in cuDSS's transpose solve mode on its own factorization: the
    transposed solves factor nothing, and every lane matches NumPy."""
    backend = CuDSS()
    topology, values = _system()
    lanes, rhs = _lanes(values), jnp.stack([_B, 2.0 * _B + 1.0])
    analysis = analyze(topology, backend=backend, options={"batch_layout": layout})
    with SolveContext(f"batch-orientations-{layout}") as ctx:
        f = jax.jit(jax.vmap(lambda v: factor(analysis, v, ctx=ctx)))(lanes)
        transposed = jax.jit(
            jax.vmap(lambda h: (solve(h, _B, transpose=True)[0], solve(h, rhs, transpose=True)[0]))
        )
        (one, two), counts = _steady(backend, transposed, f)
    assert (counts["batched_factorizations"], counts["transposed_solves"]) == (0, 2)
    for lane, x1, x2 in zip(lanes, one, two):
        at = _matrix(topology, lane).T
        # measured 2.2e-16 (both layouts, largest absolute error) against NumPy, RTX 4070 Ti,
        # cuDSS 1.0.0.35, 2026-10-23, pytest tests/contracts/test_linear_cudss.py::
        # test_a_native_batch_solves_both_orientations_from_its_own_factorization
        np.testing.assert_allclose(x1, np.linalg.solve(at, _B), **EXACT)
        np.testing.assert_allclose(x2, np.linalg.solve(at, np.asarray(rhs).T).T, **EXACT)


@cudss_only
@pytest.mark.parametrize("layout", ["stacked", "uniform"])
def test_a_batched_refactor_keeps_the_factors_of_the_lanes_that_do_not_ask(layout):
    """A batched refactor in place factors the lanes whose predicate holds and leaves the others
    with the factors they had: four lanes factor, two of them ask. The uniform batch factors only
    the asking lanes (six lanes in all); the stacked system factors all four again, the two that
    do not ask from the values they hold (eight in all, two of them unasked). Each lane solves
    its own matrix in both orientations."""
    backend = CuDSS()
    topology, values = _system()
    lanes = _lanes(values)
    asks = jnp.asarray([True, False, False, True])
    analysis = analyze(topology, backend=backend, options={"batch_layout": layout})
    with SolveContext(f"batch-asks-{layout}") as ctx:

        def run(v, ask):
            held = refactor(factor(analysis, v, ctx=ctx), 1.5 * v, ask, adopt=True)
            return solve(held, _B)[0], solve(held, _B, transpose=True)[0]

        (x, xt), counts = _steady(backend, jax.jit(jax.vmap(run)), lanes, asks)
    factored = {"stacked": (8, 2), "uniform": (6, 0)}[layout]
    assert (counts["lanes_factored"], counts["lanes_factored_unasked"]) == factored
    for lane, ask, x1, x2 in zip(lanes, asks, x, xt):
        a = _matrix(topology, 1.5 * lane if ask else lane)
        # measured 5.6e-17 (both layouts) against NumPy, RTX 4070 Ti, cuDSS 1.0.0.35, 2026-10-23,
        # pytest tests/contracts/test_linear_cudss.py::
        # test_a_batched_refactor_keeps_the_factors_of_the_lanes_that_do_not_ask
        np.testing.assert_allclose(x1, np.linalg.solve(a, _B), **EXACT)
        np.testing.assert_allclose(x2, np.linalg.solve(a.T, _B), **EXACT)


@cudss_only
@pytest.mark.parametrize("layout", ["stacked", "uniform"])
def test_a_native_batch_settles_each_lanes_status_by_its_layout(layout):
    """A batch of a regular, a singular (zero row), a non-finite and a regular lane reports
    SUCCESS, SINGULAR, NONFINITE and SUCCESS, and the regular lanes match NumPy: the stacked
    system takes each lane's status from its rows of the factor diagonal (no status solve), the
    uniform batch from one status solve. Both layouts keep float64 factors by default. The
    analysis reports its layout; an unknown layout is refused, and so are downcast factors
    (either mode) on the uniform batch."""
    backend = CuDSS()
    topology, values = _system()
    lanes = _lanes(values)
    lanes = lanes.at[1].set(jnp.where(jnp.asarray(topology.rows == 5), 0.0, lanes[1]))
    lanes = lanes.at[2, 0].set(jnp.nan)
    analysis = analyze(topology, backend=backend, options={"batch_layout": layout})
    downcast = backend.contract_stats()["downcast_configurations"]
    with SolveContext(f"batch-status-{layout}") as ctx:
        program = jax.jit(jax.vmap(lambda v: solve_checked(factor(analysis, v, ctx=ctx), _B)[::2]))
        (x, report), counts = _steady(backend, program, lanes)
    assert backend.contract_stats()["downcast_configurations"] == downcast
    assert backend.contract_analysis_info(analysis)["batch_layout"] == layout
    assert counts["status_solves"] == {"stacked": 0, "uniform": 1}[layout]
    assert np.asarray(report.status).tolist() == [
        SolveStatus.SUCCESS,
        SolveStatus.SINGULAR,
        SolveStatus.NONFINITE,
        SolveStatus.SUCCESS,
    ]
    for k in (0, 3):
        # measured 5.6e-17 (both layouts) against NumPy, RTX 4070 Ti, cuDSS 1.0.0.35, 2026-10-23,
        # pytest tests/contracts/test_linear_cudss.py::
        # test_a_native_batch_settles_each_lanes_status_by_its_layout
        np.testing.assert_allclose(x[k], np.linalg.solve(_matrix(topology, lanes[k]), _B), **EXACT)
    assert not np.isfinite(np.asarray(x[1:3])).any()
    for value, mode in (("pointer", None), ("uniform", True), ("uniform", "strict_diagonal")):
        with pytest.raises(ds.ConfigurationError, match="batch_layout must|the stacked batch_l"):
            options = {"batch_layout": value, "downcast_factors": mode}
            analyze(topology, backend=backend, options=options)


@cudss_only
@pytest.mark.parametrize("option", ["deterministic", "hybrid_memory"])
def test_a_native_batch_runs_lane_by_lane_where_the_uniform_batch_refuses_the_mode(option):
    """cuDSS's uniform batch refuses deterministic and hybrid memory modes, so a batch of such an
    analysis factors each lane on a factorization of its own (``lanes_factored_alone``) and
    solves both orientations as the default stacked batch does."""
    backend = CuDSS()
    topology, values = _system()
    lanes = _lanes(values)

    def solutions(options):
        analysis = analyze(topology, backend=backend, options=options)
        with SolveContext(f"batch-{option}") as ctx:
            program = jax.jit(
                jax.vmap(
                    lambda v: (lambda f: (solve(f, _B)[0], solve(f, _B, transpose=True)[0]))(
                        factor(analysis, v, ctx=ctx)
                    )
                )
            )
            return _steady(backend, program, lanes)

    (x, xt), counts = solutions({option: True})
    assert (counts["lanes_factored_alone"], counts["lanes_factored"]) == (4, 0)
    (ux, uxt), _ = solutions(None)
    # measured 5.6e-17 (deterministic) and 2.8e-17 (hybrid memory) between the two routes, RTX
    # 4070 Ti, cuDSS 1.0.0.35, 2026-10-23, pytest tests/contracts/test_linear_cudss.py::
    # test_a_native_batch_runs_lane_by_lane_where_the_uniform_batch_refuses_the_mode
    np.testing.assert_allclose(x, ux, **EXACT)
    np.testing.assert_allclose(xt, uxt, **EXACT)


@cudss_only
def test_a_symmetric_declaration_factors_with_the_fill_of_a_general_lu():
    """A ``symmetric`` declaration is passed to cuDSS as a general matrix, so on a symmetric
    matrix its factor has the fill (``lu_nnz_forward``) of the general declaration's."""
    backend = CuDSS()
    topology, values = _symmetric()
    fills = {}
    for declaration in ("general", "symmetric"):
        analysis = analyze(topology, backend=backend, symmetric=declaration == "symmetric")
        with SolveContext(f"fill-{declaration}") as ctx:
            jax.block_until_ready(solve(factor(analysis, values, ctx=ctx), _B)[0])
            fills[declaration] = backend.contract_analysis_info(analysis)["lu_nnz_forward"]
    assert fills["general"] is not None and fills["symmetric"] == fills["general"]


@cudss_only
def test_the_symmetric_factorization_option_factors_a_saddle_point_matrix_with_its_own_fill():
    """With ``symmetric_factorization=True`` a ``symmetric`` declaration factors with cuDSS's
    symmetric factorization: on a saddle-point matrix its fill is at most the general LU's and its
    solution is NumPy's. On a 4,500-row saddle-point matrix it took 106,479 entries against the
    general LU's 197,602, at errors of 3e-15 and 2e-15 (measured on an RTX 4070 Ti with cuDSS
    1.0.0.35)."""
    backend = CuDSS()
    topology, values = _saddle()
    fills, solutions = {}, {}
    for name, options in (("general", None), ("symmetric", {"symmetric_factorization": True})):
        analysis = analyze(topology, backend=backend, symmetric=True, options=options)
        with SolveContext(f"saddle-{name}") as ctx:
            solutions[name] = jax.block_until_ready(solve(factor(analysis, values, ctx=ctx), _B)[0])
            fills[name] = backend.contract_analysis_info(analysis)["lu_nnz_forward"]
    assert fills["symmetric"] <= fills["general"]
    np.testing.assert_allclose(
        solutions["symmetric"], np.linalg.solve(_matrix(topology, values), _B), **EXACT
    )


def _pivot_blocks(singular=False):
    """An 8 x 8 symmetric matrix of four coupled 2 x 2 blocks ``[[0, a], [a, 0]]``: regular, with
    a zero diagonal; ``singular`` makes the last two rows and columns equal."""
    entries = {}
    for k in range(N // 2):
        i = 2 * k
        entries[(i, i)] = entries[(i + 1, i + 1)] = 0.0
        entries[(i, i + 1)] = entries[(i + 1, i)] = 1.0 + 0.1 * k
        if i + 2 < N:
            entries[(i, i + 2)] = entries[(i + 2, i)] = 0.1
    if singular:
        entries[(N - 1, N - 1)] = entries[(N - 2, N - 2)] = entries[(N - 2, N - 1)]
        entries[(N - 3, N - 1)] = entries[(N - 1, N - 3)] = entries[(N - 3, N - 2)] = 0.0
    keys = sorted(entries)
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*keys))
    return SparseTopology(rows=rows, cols=cols, n_dofs=N), jnp.asarray([entries[k] for k in keys])


@cudss_only
def test_the_symmetric_factorization_takes_its_status_from_a_status_solve_not_its_diagonal():
    """Under ``symmetric_factorization`` cuDSS pivots on 2 x 2 blocks and reports only the
    diagonal of ``D``, which is zero for a regular matrix of blocks ``[[0, a], [a, 0]]``: its
    factor is SUCCESS, settled by one status solve, and solves like NumPy, while the singular
    variant of the same pattern is SINGULAR."""
    backend = CuDSS()
    found = {}
    for singular in (False, True):
        topology, values = _pivot_blocks(singular)
        analysis = analyze(
            topology, backend=backend, symmetric=True, options={"symmetric_factorization": True}
        )
        with SolveContext(f"pivot-blocks-{singular}") as ctx:
            before = backend.contract_stats()["status_solves"]
            x, _, report = solve_checked(factor(analysis, values, ctx=ctx), _B)
            found[singular] = int(report.status)
            if not singular:
                assert backend.contract_stats()["status_solves"] - before == 1
                np.testing.assert_allclose(
                    x, np.linalg.solve(_matrix(topology, values), _B), **EXACT
                )
    assert found == {False: SolveStatus.SUCCESS, True: SolveStatus.SINGULAR}


@cudss_only
def test_a_deterministic_factor_solves_any_number_of_right_hand_sides():
    """cuDSS's deterministic mode solves one right-hand side per call, so a deterministic factor
    solves several column by column: one, two, then one right-hand side each match NumPy."""
    backend = CuDSS()
    topology, values = _system()
    a = _matrix(topology, values)
    rhs = jnp.stack([_B, 2.0 * _B + 1.0])
    analysis = analyze(topology, backend=backend, options={"deterministic": True})
    with SolveContext("deterministic-widths") as ctx:
        f = factor(analysis, values, ctx=ctx)
        one, two, again = solve(f, _B)[0], solve(f, rhs)[0], solve(f, _B, transpose=True)[0]
    np.testing.assert_allclose(one, np.linalg.solve(a, _B), **EXACT)
    np.testing.assert_allclose(two, np.linalg.solve(a, np.asarray(rhs).T).T, **EXACT)
    np.testing.assert_allclose(again, np.linalg.solve(a.T, _B), **EXACT)


@cudss_only
@pytest.mark.parametrize("ir_nsteps", [0, 2])
def test_downcast_factors_solve_both_orientations_to_their_refinement_accuracy(ir_nsteps):
    """``downcast_factors=True`` stores the fp64 factor in float32: without refinement a solve
    is accurate to about single precision, with two refinement steps to double precision, in
    both orientations."""
    backend = CuDSS()
    topology, values = _system()
    a = _matrix(topology, values)
    options = {"downcast_factors": True, "ir_nsteps": ir_nsteps}
    analysis = analyze(topology, backend=backend, options=options)
    with SolveContext(f"downcast-{ir_nsteps}") as ctx:
        f = factor(analysis, values, ctx=ctx)
        x, xt = solve(f, _B)[0], solve(f, _B, transpose=True)[0]
    # measured 3.8e-9 (ir_nsteps=0) and 5.6e-17 (ir_nsteps=2) in both orientations against
    # NumPy, RTX 4070 Ti, cuDSS 1.0.0.35, 2026-10-23, pytest tests/contracts/test_linear_cudss.py::
    # test_downcast_factors_solve_both_orientations_to_their_refinement_accuracy
    bound = {"rtol": 1e-7, "atol": 1e-7} if ir_nsteps == 0 else EXACT
    np.testing.assert_allclose(x, np.linalg.solve(a, _B), **bound)
    np.testing.assert_allclose(xt, np.linalg.solve(a.T, _B), **bound)


@cudss_only
def test_the_zero_fill_follows_the_option_or_else_the_factor():
    """``zero_fill=True`` zeroes the solution buffer once before a solve and ``False`` skips it;
    left out, a float64 factor, downcast or not, skips it and a float32 factor zeroes it. Every
    solve matches NumPy."""
    backend = CuDSS()
    topology, values = _system()
    exact = np.linalg.solve(_matrix(topology, values), _B)
    cases = {
        "on": ({"zero_fill": True}, "fp64"),
        "off": ({"zero_fill": False}, "fp64"),
        "float64": ({}, "fp64"),
        "float32": ({}, "fp32"),
        "downcast": ({"downcast_factors": True}, "fp64"),
        "plain": ({"downcast_factors": False}, "fp64"),
    }
    fills, errors = {}, {}
    for name, (options, rung) in cases.items():
        analysis = analyze(topology, backend=backend, options=options, precision=rung)
        with SolveContext(f"zero-fill-{name}") as ctx:
            f = factor(analysis, values, ctx=ctx)
            x, counts = _steady(backend, jax.jit(lambda h: solve(h, _B)[0]), f)
        fills[name] = counts["zero_fills"]
        errors[name] = float(np.max(np.abs(np.asarray(x, np.float64) - exact)))
    assert fills == {"on": 1, "off": 0, "float64": 0, "float32": 1, "downcast": 0, "plain": 0}
    # measured 5.6e-17 (the float64 cases), 2.3e-8 (float32) and 4.9e-17 (downcast_factors=True,
    # one refinement step) absolute, RTX 4070 Ti, cuDSS 1.0.0.35, 2026-10-23, this row
    assert max(errors[name] for name in ("on", "off", "float64", "plain")) < 1e-12
    assert max(errors["float32"], errors["downcast"]) < 1e-4


class _CuDSSSettings(LinearSolver):
    """A stand-in with cuDSS's ordering and hybrid-memory settings whose analysis is cuDSS's own
    parse of the options it receives."""

    reorder: Optional[str] = eqx.field(static=True, default=None)
    hybrid_memory: Optional[bool] = eqx.field(static=True, default=None)
    hybrid_device_memory_limit_mb: Optional[int] = eqx.field(static=True, default=None)
    option_fields = ("reorder", "hybrid_memory", "hybrid_device_memory_limit_mb")
    supports_reordering = supports_hybrid_memory = True

    def _analyze(self, topology, options, layout, precision):
        from diff_soup.linear._cudss._contract import _parse_options

        parsed = _parse_options(options, layout)
        return parsed["reorder"], parsed["hybrid_mode"], parsed["hybrid_device_memory_limit_mb"]


_HYBRID = {"hybrid_memory": True, "hybrid_device_memory_limit_mb": 64}


@pytest.mark.parametrize(
    "settings,layout,options,parsed",
    [
        ({"reorder": "metis"}, {}, None, ("metis", 0, -1)),
        ({"reorder": "amd"}, {"reorder": "metis"}, None, ("amd", 0, -1)),
        ({}, {"reorder": "metis"}, None, ("metis", 0, -1)),
        ({}, {}, None, ("default", 0, -1)),
        ({"reorder": "metis"}, {}, {"reorder": "amd"}, ("amd", 0, -1)),
        ({"hybrid_memory": False}, _HYBRID, None, ("default", 0, -1)),
        ({"hybrid_device_memory_limit_mb": 32}, _HYBRID, None, ("default", 1, 32)),
        ({}, _HYBRID, None, ("default", 1, 64)),
    ],
)
def test_a_cudss_setting_wins_over_the_layout_and_an_explicit_option_over_the_setting(
    settings, layout, options, parsed
):
    """cuDSS's option parse sees a setting of the backend object over the layout's ``reorder``,
    ``hybrid_memory`` and device-memory limit, the layout's where the setting is left at
    ``None``, and a key of an explicit ``options=`` over the setting."""
    backend, layout = _CuDSSSettings(**settings), ds.Layout(**layout)
    analysis = analyze(_system()[0], backend=backend, layout=layout, options=options)
    assert analysis.backend_analysis == parsed


def _grid(side=8):
    """The 7-point Laplacian of a ``side``-cubed grid, which nested dissection and minimum
    degree order with different fill."""
    index = np.arange(side**3).reshape(side, side, side)
    rows, cols = [index.ravel()], [index.ravel()]
    for axis in range(3):
        low = np.take(index, range(side - 1), axis=axis).ravel()
        high = np.take(index, range(1, side), axis=axis).ravel()
        rows, cols = rows + [low, high], cols + [high, low]
    rows, cols = (np.concatenate(axis).astype(np.int32) for axis in (rows, cols))
    values = jnp.asarray(np.where(rows == cols, 6.0, -1.0))
    return SparseTopology(rows=rows, cols=cols, n_dofs=side**3), values


@cudss_only
def test_both_orderings_reach_cudss_from_the_backend_object():
    """``CuDSS(reorder="metis")`` and ``CuDSS(reorder="amd")`` each reach cuDSS's reordering:
    on a 3-D grid their factors fill differently, ``reorder="amd"`` keeps its fill under a
    layout that asks for METIS, every solve is NumPy's, and an ordering cuDSS does not offer is
    refused when the backend is built. On the 512-row grid METIS filled 20,008 entries and AMD
    22,150, at relative errors of 6.8e-15 and 7.8e-15 (measured on an L40S with cuDSS 1.0.0.35,
    2026-10-26)."""
    topology, values = _grid()
    b = jnp.linspace(-1.0, 1.0, topology.n_dofs)
    fills = {}
    for label, reorder, layout in (
        ("metis", "metis", ds.Layout()),
        ("amd", "amd", ds.Layout()),
        ("amd over metis", "amd", ds.Layout(reorder="metis")),
    ):
        backend = CuDSS(reorder=reorder)
        analysis = analyze(topology, backend=backend, layout=layout)
        with SolveContext(f"ordering-{label}") as ctx:
            x = jax.block_until_ready(solve(factor(analysis, values, ctx=ctx), b)[0])
            fills[label] = backend.contract_analysis_info(analysis)["lu_nnz_forward"]
        np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, values), b), **EXACT)
    assert fills["metis"] != fills["amd"] == fills["amd over metis"]
    with pytest.raises(ds.ConfigurationError, match="cuDSS option reorder"):
        CuDSS(reorder="colamd")


@pytest.mark.parametrize(
    "name",
    [
        "dense",
        pytest.param("klu", marks=pytest.mark.requires_klu),
        pytest.param("superlu", marks=pytest.mark.requires_superlu),
    ],
)
def test_symmetric_declarations_are_refused_on_a_topology_with_dirichlet_rows(name):
    """Identity rows make the factored operator unsymmetric: ``symmetric=True`` and
    ``spd=True`` are refused on a topology with Dirichlet rows, taken without them."""
    backend = {"dense": Dense, "klu": KLU, "superlu": SuperLU}[name]()
    with_rows, without_rows = _symmetric(dirichlet=True)[0], _symmetric()[0]
    for declaration in ("symmetric", "spd"):
        with pytest.raises(ds.ConfigurationError, match="(?i)symmetric|spd"):
            analyze(with_rows, backend=backend, **{declaration: True})
        assert getattr(analyze(without_rows, backend=backend, **{declaration: True}), declaration)


@cudss_only
@pytest.mark.requires_two_processes
def test_a_model_group_across_processes_takes_its_first_process_bytes_while_traced():
    """The processes of a model group agree on the id of their device communicator over MPI on
    the host, not with a JAX collective (which is refused while a program is traced): called
    while ``jax.jit`` traces, the group's broadcast hands every process the bytes of the group's
    first process, for the processes in order and reversed. The processes are the ranks of one
    MPI world."""
    from diff_soup.linear._cudss._model_axis._mgmn import broadcast_from_root

    ranks = tuple(range(jax.process_count()))
    sent = bytes([jax.process_index() + 1]) * 128
    received = []

    def traced(x):
        received.append(broadcast_from_root(1, ranks, sent))
        received.append(broadcast_from_root(2, ranks[::-1], sent))
        return x

    jax.jit(traced)(jnp.zeros(()))
    assert received == [bytes([ranks[0] + 1]) * 128, bytes([ranks[-1] + 1]) * 128]


@cudss_only
def test_an_invalidated_factor_is_refactored_from_its_next_solve_and_a_replay_stays_exact():
    """After an invalidation the next solve refactors the storage from the values it carries
    (one rebuild) and every solve matches NumPy; replaying a checkpointed segment from an
    invalidated entry factor rebuilds once and equals the original pass."""
    backend = CuDSS()
    topology, values = _system()
    analysis = analyze(topology, backend=backend)
    a = _matrix(topology, values)
    schedule = jnp.stack([values * (1.0 + 0.05 * (k + 1)) for k in range(4)])

    def step(held, v):
        x, held = solve(held, _B)
        return refactor(held, v, True), x

    with SolveContext("invalidate") as ctx:
        f = factor(analysis, values, ctx=ctx)
        jax.block_until_ready(solve(f, _B)[0])
        backend.reset_contract_stats()
        x = solve(invalidate(f), _B)[0]
        stats = backend.contract_stats()
        assert (stats["invalidations"], stats["rebuilds"]) == (1, 1)
        np.testing.assert_allclose(x, np.linalg.solve(a, _B), **EXACT)
        segment = jax.checkpoint(lambda held, vs: jax.lax.scan(step, held, vs))
        entry, first = jax.jit(lambda v, vs: segment(factor(analysis, v, ctx=ctx), vs))(
            values, schedule[:2]
        )
        original = jax.jit(lambda e, vs: segment(e, vs)[1])(entry, schedule[2:])
        backend.reset_contract_stats()
        replayed = jax.jit(lambda e, vs: segment(invalidate(e), vs)[1])(entry, schedule[2:])
        assert backend.contract_stats()["rebuilds"] == 1
    np.testing.assert_allclose(replayed, original, **EXACT)


@cudss_only
def test_storage_holds_fixed_slots_and_releases_them_and_programs_run_exactly():
    """An empty factor solves to NaN with EMPTY until a refactor fills it; adopting storage adds no
    slot; a scan factors its two sites once each, refactors in place at every later step and
    rebuilds nothing; ahead-of-time and scan-sliced programs stay exact; resident bytes grow with
    each factor, not with a transposed solve, and return to zero on release, after which a solve
    is refused."""
    backend = CuDSS()
    topology, values = _system()
    analysis = analyze(topology, backend=backend)
    with SolveContext("slots") as ctx:
        x, _, report = solve_checked(empty_factor(analysis, ctx=ctx), _B)
        assert int(report.status) == SolveStatus.EMPTY and not np.isfinite(np.asarray(x)).any()
        f = jax.jit(lambda v: refactor(factor(analysis, v, ctx=ctx), 1.1 * v, True))(values)
        live = backend.contract_stats()["live_slots"]
        adopted = jax.jit(lambda held, v: refactor(held, v, True, adopt=True))(f, 1.2 * values)
        np.testing.assert_allclose(
            solve(adopted, _B)[0], np.linalg.solve(_matrix(topology, 1.2 * values), _B), **EXACT
        )
        assert backend.contract_stats()["live_slots"] == live
        steps = jnp.stack([values * (1.0 + 0.1 * k) for k in range(4)])

        def scanned(v, vs):
            return jax.lax.scan(
                lambda held, w: (lambda x, h: (refactor(h, w, True), x))(*solve(held, _B)),
                factor(analysis, v, ctx=ctx),
                vs,
            )[1]

        xs, counts = _steady(backend, jax.jit(scanned), values, steps)
        # The first refactor factors the scan's held factor into its own site's slot; the three
        # later ones refactor that slot in place.
        assert [counts[k] for k in ("factorizations", "refactorizations", "rebuilds")] == [2, 3, 0]
        np.testing.assert_allclose(xs[1], np.linalg.solve(_matrix(topology, steps[0]), _B), **EXACT)
    with SolveContext("aot") as ctx:
        run = jax.jit(lambda v: solve(factor(analysis, v, ctx=ctx), _B)[0])
        compiled = run.lower(values).compile()
        for scale in (1.0, 1.5, 0.7):
            expected = np.linalg.solve(_matrix(topology, scale * values), _B)
            np.testing.assert_allclose(compiled(scale * values), expected, **EXACT)
        rhs = jnp.stack([_B, 2.0 * _B, -_B])
        scan = jax.jit(lambda f, rs: jax.lax.scan(lambda c, r: (c, solve(f, r)[0]), 0, rs)[1])
        expected = np.linalg.solve(_matrix(topology, values), np.asarray(rhs).T).T
        np.testing.assert_allclose(scan(factor(analysis, values, ctx=ctx), rhs), expected, **EXACT)
    b = _B
    with SolveContext("resident") as ctx:
        jax.block_until_ready(solve(factor(analysis, values, ctx=ctx), b)[0])
        one = ctx.resident_bytes()
        g = factor(analysis, 2.0 * values, ctx=ctx)
        jax.block_until_ready(solve(g, b)[0])
        two = ctx.resident_bytes()
        jax.block_until_ready(solve(g, b, transpose=True)[0])
        assert 0 < one < two == ctx.resident_bytes()
    assert ctx.resident_bytes() == 0
    with pytest.raises((jax.errors.JaxRuntimeError, ValueError), match="released"):
        jax.block_until_ready(solve(g, b)[0])


@cudss_only
def test_the_handlers_dispatch_statically_where_a_program_states_its_factor():
    """A straight-line factor and solves reads nothing from the device and dispatches every
    solve statically; a loop-carried handle reads once per solve; a native batch refactor runs
    only when a lane asks; a loop from shared storage rebuilds nothing."""
    backend = CuDSS()
    topology, values = _system(n=16)
    analysis = analyze(topology, backend=backend)
    b = jnp.linspace(-1.0, 1.0, 16)
    with SolveContext("dispatch") as ctx:

        def held(v):
            return factor(analysis, v, ctx=ctx)

        def straight(v):
            f = held(v)
            return jnp.stack([solve(f, k * b)[0] for k in (1.0, 2.0, 3.0)])

        def looped(v):
            step = lambda k, c: (lambda x, h: (h, c[1] + x))(*solve(c[0], b))  # noqa: E731
            return jax.lax.fori_loop(0, 3, step, (held(v), jnp.zeros(16)))[1]

        def batched(v, p):
            return refactor(held(v), 1.2 * v, p).generation

        def shared(v):
            empty = empty_factor(analysis, ctx=ctx, storage="shared")
            return jax.lax.fori_loop(0, 3, lambda k, f: refactor(f, v * (1.0 + k), True), empty)

        alone = _steady(backend, jax.jit(lambda v: held(v).handle), values)[1]
        line = _steady(backend, jax.jit(straight), values)[1]
        assert line["host_reads"] == 0
        assert line["static_dispatches"] - alone["static_dispatches"] == 3
        loop = _steady(backend, jax.jit(looped), values)[1]
        assert (loop["host_reads"], loop["rebuilds"]) == (3, 0)
        lanes, off = jnp.stack([values, 1.1 * values]), jnp.asarray([False, False])
        assert _steady(backend, jax.jit(jax.vmap(batched)), lanes, off)[1]["refactorizations"] == 0
        assert _steady(backend, jax.jit(lambda v: shared(v).generation), values)[1]["rebuilds"] == 0


@cudss_only
def test_the_csr_assembly_sums_duplicates_and_replaces_dirichlet_rows():
    """The values handed to the extension are the CSR values of ``A_bc``: shuffled coordinates
    with duplicates and Dirichlet rows assemble entry by entry in row-major order, eagerly and
    under jit."""
    from diff_soup.linear._cudss._contract import assemble_csr, csr_pattern

    base, base_values = _system(dirichlet=True)
    order = np.random.default_rng(5).permutation(base.rows.size + 4)
    rows = np.concatenate([base.rows, [1, 6, 0, 1]]).astype(np.int32)[order]
    cols = np.concatenate([base.cols, [1, 5, 1, 2]]).astype(np.int32)[order]
    values = jnp.concatenate(
        [base_values, jnp.asarray(np.random.default_rng(6).uniform(-1, 1, 4))]
    )[order]
    topology = SparseTopology(rows=rows, cols=cols, n_dofs=N, bc_indices=base.bc_indices)
    record, a = csr_pattern(topology), _matrix(topology, values)
    keys = np.unique(rows * N + cols)
    for got in (assemble_csr(record, values), jax.jit(lambda v: assemble_csr(record, v))(values)):
        np.testing.assert_allclose(got, a[keys // N, keys % N], **EXACT)


# One program on two devices fills one shared storage name with a different matrix on each
# device, five executions in a row, and reports every device's solution with NumPy's.
_PER_DEVICE_WORKER = r"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from diff_soup import linear

backend = {"cudss": linear.CuDSS, "klu": linear.KLU, "superlu": linear.SuperLU}[sys.argv[1]]()
devices = jax.devices()[:2]
assert len(devices) == 2, jax.devices()
n = 8
pairs = [(i, j) for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n] + [(0, 5), (6, 1)]
rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
base = np.random.default_rng(7).uniform(-1, 1, rows.size) + np.where(rows == cols, 4.0, 0.0)
values = np.stack([base, 1.5 * base + np.where(rows == cols, 3.0, 0.0)])
analysis = linear.analyze(linear.SparseTopology(rows=rows, cols=cols, n_dofs=n), backend=backend)
b = np.linspace(-1.0, 1.0, n)
mesh = Mesh(np.asarray(devices), ("d",))
expected = []
for lane in values:
    a = np.zeros((n, n))
    np.add.at(a, (rows, cols), lane)
    expected.append(np.linalg.solve(a, b).tolist())
with linear.SolveContext("per-device") as ctx:

    def per_device(v):
        held = linear.empty_factor(analysis, ctx=ctx, storage="per-device")
        return linear.solve(linear.refactor(held, v[0], True), jnp.asarray(b))[0][None]

    run = jax.jit(
        jax.shard_map(per_device, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)
    )
    stacked = jax.device_put(jnp.asarray(values), NamedSharding(mesh, P("d")))
    solutions = [np.asarray(run(stacked)).tolist() for _ in range(5)]
print("REPORT " + json.dumps({"solutions": solutions, "expected": expected}))
"""


def _per_device_report(name):
    """The worker's report for backend ``name``: two GPUs for cuDSS, two forced host devices for
    the host backends."""
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(JAX_ENABLE_X64="1", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="1", OMP_NUM_THREADS="1")
    if name != "cudss":
        env.update(
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
            CUDA_VISIBLE_DEVICES="",
        )
    run = subprocess.run(
        [sys.executable, "-c", _PER_DEVICE_WORKER, name],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("cudss", marks=(pytest.mark.requires_cudss, pytest.mark.requires_two_devices)),
        pytest.param("klu", marks=pytest.mark.requires_klu),
        pytest.param("superlu", marks=pytest.mark.requires_superlu),
    ],
)
def test_shared_storage_keeps_one_factor_per_device(name):
    """One shared storage name filled on two devices of one program with a different matrix on
    each holds a factor per device: every device solves its own system in every execution."""
    report = _per_device_report(name)
    for solution in report["solutions"]:
        np.testing.assert_allclose(solution, report["expected"], **EXACT)
