"""Compiled constrained solvers: solutions in jit and vmap lanes, the implicit derivative in args,
the record's dtypes, degenerate and infeasible problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from diff_soup.optimization import SQP, AugmentedLagrangian, InteriorPoint, TerminationStatus
from tests.support.optimization import duplicate_rows, hs7, hs21, hs35, infeasible, nan_region

# The solvers' stopping tolerances; measured: every lane ends KKT_TOL, workstation CPU 2026-11-06
SOLVERS = {
    "interior_point": InteriorPoint(hessian="exact", tol=1e-12),
    "sqp": SQP(hessian="exact", tol=1e-12, qp_tol=1e-13),
    # measured: ends KKT_TOL at these tolerances, workstation CPU 2026-11-06
    "augmented_lagrangian": AugmentedLagrangian(hessian="exact", tol=1e-12, feasibility_tol=1e-12),
}
TOYS = {"equality": hs7(), "inequality": hs35(), "bound": hs21()}

# measured 3.1e-12 (worst over solvers, toys and lanes), workstation CPU 2026-11-06,
# pytest tests/contracts/test_compiled_solvers.py
_SOLUTION = 1e-10
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_LANE = 1e-15
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_TANGENT = 1e-9
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_LANE_GRADIENT = 1e-12


def _flat(x, record):
    """`x` and every multiplier, concatenated."""
    lower, upper = record.bound_multipliers
    return jnp.concatenate(
        [x, record.multipliers_equality, record.multipliers_inequality, lower, upper]
    )


def _lane_runs(solver, toy):
    """`jit(vmap(run))` over the toy's batch of `args`."""
    n = toy.args_batch.shape[0]
    x0 = jnp.broadcast_to(toy.problem.x0, (n,) + toy.problem.x0.shape)
    return jax.jit(jax.vmap(lambda x, a: solver.run(toy.problem, x, a)))(x0, toy.args_batch)


@pytest.fixture(scope="module")
def lanes():
    """One lane run per solver and toy, shared by the tests that read it."""
    return {(s, t): _lane_runs(SOLVERS[s], TOYS[t]) for s in SOLVERS for t in TOYS}


@pytest.mark.parametrize("toy", TOYS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_each_solver_finds_each_active_set_in_vmap_lanes(lanes, solver, toy):
    """Equality, inequality and bound active: every lane's point and multipliers are exact."""
    xs, records = lanes[solver, toy]
    exact = jax.vmap(lambda a: jnp.concatenate(TOYS[toy].solution(a)))(TOYS[toy].args_batch)
    assert float(jnp.max(jnp.abs(jax.vmap(_flat)(xs, records) - exact))) <= _SOLUTION


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_jitted_run_equals_its_vmap_lane(lanes, solver):
    """One problem run alone under jit ends where its lane in the batch ends."""
    toy = TOYS["inequality"]
    x, _ = jax.jit(lambda a: SOLVERS[solver].run(toy.problem, toy.problem.x0, a))(toy.args_batch[1])
    assert float(jnp.max(jnp.abs(x - lanes[solver, "inequality"][0][1]))) <= _LANE


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_implicit_tangent_is_the_derivative_of_the_solution(solver):
    """`jax.jvp` of the point and multipliers in a constraint limit equals the exact derivative."""
    toy = TOYS["inequality"]
    run = SOLVERS[solver].run
    a = toy.args_batch[0]
    _, tangent = jax.jvp(lambda b: _flat(*run(toy.problem, toy.problem.x0, b)), (a,), (1.0,))
    exact = jax.jacfwd(lambda b: jnp.concatenate(toy.solution(b)))(a)
    assert float(jnp.max(jnp.abs(tangent - exact)) / jnp.max(jnp.abs(exact))) <= _TANGENT


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_of_a_lane_sum_equals_the_lane_gradients(solver):
    """Reverse mode through `vmap(run)` gives each lane the gradient of its own solve."""
    toy = TOYS["bound"]
    run = SOLVERS[solver].run

    def lane(a):
        x, record = run(toy.problem, toy.problem.x0, a)
        return jnp.sum(x**2) + jnp.sum(record.bound_multipliers[0])

    batched = jax.jit(jax.grad(lambda b: jnp.sum(jax.vmap(lane)(b))))(toy.args_batch)
    single = jnp.stack([jax.jit(jax.grad(lane))(a) for a in toy.args_batch])
    assert float(jnp.max(jnp.abs(batched - single)) / jnp.max(jnp.abs(single))) <= _LANE_GRADIENT


def test_the_record_keeps_the_point_dtype_and_int32_counters(lanes):
    """The point and multipliers are float64, as `x0` is; the counters and the status are int32."""
    xs, records = lanes["interior_point", "inequality"]
    dtypes = {
        jnp.result_type(leaf)
        for leaf in (xs, records.multipliers_inequality, records.slacks, records.kkt_error)
    }
    assert dtypes == {jnp.dtype("float64")} and {
        jnp.result_type(leaf) for leaf in (records.steps, records.status)
    } == {jnp.dtype("int32")}


def test_two_identical_active_rows_are_degenerate_to_the_interior_point():
    """The interior point marks an active set whose rows repeat as degenerate."""
    problem = duplicate_rows()
    _, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(1.0)
    assert bool(record.degenerate)


def test_the_augmented_lagrangian_solves_two_identical_active_rows():
    """On repeated active rows the augmented Lagrangian ends at `x* = (a, a) / 2`."""
    problem = duplicate_rows()
    x, _ = jax.jit(lambda a: SOLVERS["augmented_lagrangian"].run(problem, problem.x0, a))(1.0)
    assert float(jnp.max(jnp.abs(x - 0.5))) <= _SOLUTION


@pytest.mark.parametrize("solver", ["interior_point", "augmented_lagrangian"])
def test_an_infeasible_problem_ends_infeasible(solver):
    """Constraints no point satisfies end the run with `INFEASIBLE`."""
    problem = infeasible()
    _, record = jax.jit(lambda a: SOLVERS[solver].run(problem, problem.x0, a))(1.0)
    assert int(record.status) == TerminationStatus.INFEASIBLE


def test_a_nan_region_ends_with_solver_failed_at_a_finite_point():
    """Repeated NaN trials end the run with `SOLVER_FAILED`, the last accepted point kept."""
    problem = nan_region()
    x, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(1.0)
    assert int(record.status) == TerminationStatus.SOLVER_FAILED and bool(jnp.all(jnp.isfinite(x)))
