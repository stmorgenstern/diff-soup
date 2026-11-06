"""Optimization solvers through `ds.optimization`: each compiled solver through `optsolve`, a warm
start from a record or from a record of zeros, and `lanes` over a batch of problems on one device."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from tests.support.optimization import hs7, hs35

# The solvers' stopping tolerances; measured: every run ends KKT_TOL, workstation CPU 2026-11-06
SOLVERS = {
    "interior_point": ds.optimization.InteriorPoint(hessian="exact", tol=1e-10),
    "sqp": ds.optimization.SQP(hessian="exact", tol=1e-10),
    # measured: ends KKT_TOL at these tolerances, workstation CPU 2026-11-06
    "augmented_lagrangian": ds.optimization.AugmentedLagrangian(
        hessian="exact", tol=1e-10, feasibility_tol=1e-10
    ),
}
TOY = hs35()
WARM_TOY = hs7()

# measured 2.2e-11 (worst over solvers, optsolve and the record of zeros), workstation CPU 2026-11-06,
# pytest tests/public_api/test_optimization_solvers.py
_SOLUTION = 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_optsolve_returns_the_solution_and_its_record(solver):
    """`optsolve` returns the minimizer as `params` and the run's `SolveRecord` in `info`."""
    result = ds.optimization.optsolve(TOY.problem, SOLVERS[solver])
    exact = np.asarray(TOY.solution(TOY.problem.args)[0])
    record = result.info["record"]
    assert np.max(np.abs(result.params - exact)) <= _SOLUTION and (
        int(record.status) == ds.optimization.TerminationStatus.KKT_TOL
    )


def _work(record):
    """The run's steps and the inner steps of its subproblems."""
    return int(record.steps) + int(record.inner_steps)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_warm_start_from_a_record_takes_fewer_steps(solver):
    """Started from the last solution and its record, a solve with a moved right-hand side is shorter."""
    problem, run = WARM_TOY.problem, SOLVERS[solver].run
    warm = jax.jit(lambda x0, a, record: run(problem, x0, a, warm_start=record))
    cold = jax.jit(lambda x0, a: run(problem, x0, a))
    x, record = cold(problem.x0, 4.0)
    _, moved_cold = cold(problem.x0, 4.04)
    _, moved_warm = warm(x, 4.04, record)
    assert _work(moved_warm) < _work(moved_cold)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_record_of_zeros_warm_starts_a_run_to_the_solution(solver):
    """`SolveRecord.zeros` as the warm start: the run ends with `KKT_TOL` at the minimizer."""
    zeros = ds.optimization.SolveRecord.zeros(3, 0, 1)
    run = jax.jit(lambda a, w: SOLVERS[solver].run(TOY.problem, TOY.problem.x0, a, warm_start=w))
    x, record = run(TOY.problem.args, zeros)
    exact = np.asarray(TOY.solution(TOY.problem.args)[0])
    assert np.max(np.abs(np.asarray(x) - exact)) <= _SOLUTION and (
        int(record.status) == ds.optimization.TerminationStatus.KKT_TOL
    )


def test_lanes_on_one_device_is_vmap_of_run():
    """`lanes` without a layout returns exactly `vmap` of `run` over the batch."""
    solver = SOLVERS["interior_point"]
    n = TOY.args_batch.shape[0]
    x0 = jnp.broadcast_to(TOY.problem.x0, (n, 3))
    xs, _ = ds.optimization.lanes(solver, TOY.problem, TOY.args_batch, x0)
    expected, _ = jax.vmap(lambda x, a: solver.run(TOY.problem, x, a))(x0, TOY.args_batch)
    np.testing.assert_array_equal(np.asarray(xs), np.asarray(expected))
