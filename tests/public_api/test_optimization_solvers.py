"""Optimization solvers through `ds.optimization`: each compiled solver's `solve`, a warm start
from a record, and `lanes` over a batch of problems on one device."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from tests.support.optimization import hs35

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

# measured 2.2e-11 (interior point and SQP), workstation CPU 2026-11-06, pytest tests/public_api/test_optimization_solvers.py
_SOLUTION = 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_solve_returns_the_solution_and_its_record(solver):
    """`solve` returns the minimizer as `params` and the run's `SolveRecord` in `info`."""
    result = SOLVERS[solver].solve(TOY.problem)
    exact = np.asarray(TOY.solution(TOY.problem.args)[0])
    record = result.info["record"]
    assert np.max(np.abs(result.params - exact)) <= _SOLUTION and (
        int(record.status) == ds.optimization.TerminationStatus.KKT_TOL
    )


def test_a_warm_start_from_a_record_takes_fewer_steps():
    """Started from the last solution and its record, a solve with a moved limit is shorter."""
    solver = SOLVERS["interior_point"]
    warm = jax.jit(lambda x0, a, record: solver.run(TOY.problem, x0, a, warm_start=record))
    cold = jax.jit(lambda x0, a: solver.run(TOY.problem, x0, a))
    x, record = cold(TOY.problem.x0, 3.0)
    _, moved_cold = cold(TOY.problem.x0, 3.03)
    _, moved_warm = warm(x, 3.03, record)
    assert int(moved_warm.steps) < int(moved_cold.steps)


def test_lanes_on_one_device_is_vmap_of_run():
    """`lanes` without a layout returns exactly `vmap` of `run` over the batch."""
    solver = SOLVERS["interior_point"]
    n = TOY.args_batch.shape[0]
    x0 = jnp.broadcast_to(TOY.problem.x0, (n, 3))
    xs, _ = ds.optimization.lanes(solver, TOY.problem, TOY.args_batch, x0)
    expected, _ = jax.vmap(lambda x, a: solver.run(TOY.problem, x, a))(x0, TOY.args_batch)
    np.testing.assert_array_equal(np.asarray(xs), np.asarray(expected))
