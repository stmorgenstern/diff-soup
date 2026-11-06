"""Compiled constrained solvers: solutions in jit and vmap lanes, each Hessian mode and the unscaled
interior point, the implicit derivative in args, a forward-only residual under forward-over-forward,
the record's dtypes, and how a run ends: degenerate, infeasible, out of steps, diverged, a failed
subproblem, restoration, failed evaluations."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from diff_soup.optimization import (
    SQP,
    AugmentedLagrangian,
    Constraints,
    InteriorPoint,
    LeastSquaresProblem,
    TerminationStatus,
)
from tests.support.optimization import (
    bound_active,
    duplicate_rows,
    hs7,
    hs35,
    infeasible,
    least_squares,
    nan_region,
    unbounded,
    wachter_biegler,
)

# The solvers' stopping tolerances; measured: every lane ends KKT_TOL, workstation CPU 2026-11-06
SOLVERS = {
    "interior_point": InteriorPoint(hessian="exact", tol=1e-12),
    "sqp": SQP(hessian="exact", tol=1e-12, qp_tol=1e-13),
    # measured: ends KKT_TOL at these tolerances, workstation CPU 2026-11-06
    "augmented_lagrangian": AugmentedLagrangian(hessian="exact", tol=1e-12, feasibility_tol=1e-12),
}
TOYS = {"equality": hs7(), "inequality": hs35(), "bound": bound_active()}
LEAST_SQUARES = least_squares()

# measured 1.3e-12 (worst over solvers, toys, Hessian modes and lanes), workstation CPU 2026-11-06,
# pytest tests/contracts/test_compiled_solvers.py
_SOLUTION = 1e-10
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_LANE = 1e-15
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_TANGENT = 1e-9
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_LANE_GRADIENT = 1e-12
# measured 0 (point and jacobian), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_COMPOSITION = 1e-10


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


def _lane_error(solver, toy):
    """The largest error of any lane's point and multipliers."""
    xs, records = _lane_runs(solver, toy)
    exact = jax.vmap(lambda a: jnp.concatenate(toy.solution(a)))(toy.args_batch)
    return float(jnp.max(jnp.abs(jax.vmap(_flat)(xs, records) - exact)))


@pytest.mark.parametrize("mode", ["bfgs", "auto", "gauss-newton"])
@pytest.mark.parametrize("solver", SOLVERS)
def test_each_hessian_approximation_finds_the_solution(solver, mode):
    """`bfgs` on HS7, `auto` and `gauss-newton` on a least-squares objective: every lane is exact."""
    toy = TOYS["equality"] if mode == "bfgs" else LEAST_SQUARES
    assert _lane_error(dataclasses.replace(SOLVERS[solver], hessian=mode), toy) <= _SOLUTION


def test_an_unscaled_interior_point_finds_the_solution():
    """With `scaling="none"` every lane of the inequality toy is exact."""
    solver = dataclasses.replace(SOLVERS["interior_point"], scaling="none")
    assert _lane_error(solver, TOYS["inequality"]) <= _SOLUTION


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


def _residual(x, a):
    """A nonlinear residual whose first row reads `a`."""
    return jnp.stack([x[0] - a, x[1] - 1.0, x[0] * x[1] - 0.5])


def _forward_only(x, a):
    """`_residual` through one pass of a `lax.while_loop`, which JAX differentiates forward only."""
    r = _residual(x, a)
    return jax.lax.while_loop(lambda c: c[0] < 1, lambda c: (c[0] + 1, c[1] + r), (0, 0.0 * r))[1]


def _point_and_jacobian(residual, composition):
    """The interior point's solution under `x0 + x1 <= a` (active) at `a = 0.8`, and its jacfwd in `a`."""
    problem = LeastSquaresProblem.from_residual(
        jnp.zeros(2), residual, args=jnp.asarray(0.8), hessian_composition=composition
    ).with_constraints(Constraints(inequality=lambda x, a: x[:1] + x[1:] - a, n_inequality=1))

    def solve(a):
        x, _ = SOLVERS["interior_point"].run(problem, problem.x0, a)
        return x, x

    jacobian, x = jax.jit(jax.jacfwd(solve, has_aux=True))(jnp.asarray(0.8))
    return jnp.concatenate([x, jacobian])


def test_forward_over_forward_solves_and_differentiates_a_forward_only_residual():
    """Under `"forward-over-forward"` the point and its implicit jacfwd are the default's on `_residual`."""
    forward = _point_and_jacobian(_forward_only, "forward-over-forward")
    default = _point_and_jacobian(_residual, "forward-over-reverse")
    assert float(jnp.max(jnp.abs(forward - default))) <= _COMPOSITION


def test_the_default_composition_differentiates_the_residual_in_reverse_mode():
    """Under the default `"forward-over-reverse"` the forward-only residual raises in the exact Hessian."""
    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        _point_and_jacobian(_forward_only, "forward-over-reverse")


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_run_out_of_steps_ends_max_steps(solver):
    """Allowed one step on HS7, which needs more, a run ends with `MAX_STEPS`."""
    toy = TOYS["equality"]
    one = dataclasses.replace(SOLVERS[solver], max_steps=1)
    _, record = jax.jit(lambda a: one.run(toy.problem, toy.problem.x0, a))(4.0)
    assert int(record.status) == TerminationStatus.MAX_STEPS


def test_an_unbounded_objective_ends_the_interior_point_diverged():
    """Iterates that pass `1e20` on an objective without a minimizer end the run with `DIVERGED`."""
    problem = unbounded()
    _, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(1.0)
    assert int(record.status) == TerminationStatus.DIVERGED


def test_a_failed_subproblem_ends_sqp_with_its_status_recorded():
    """A QP allowed one step ends the run with `SOLVER_FAILED`, the QP's `MAX_STEPS` in the record."""
    toy = TOYS["inequality"]
    sqp = dataclasses.replace(SOLVERS["sqp"], qp_max_steps=1)
    _, record = jax.jit(lambda a: sqp.run(toy.problem, toy.problem.x0, a))(3.0)
    assert int(record.status) == TerminationStatus.SOLVER_FAILED and (
        int(record.subproblem_status) == TerminationStatus.MAX_STEPS
    )


def test_the_interior_point_solves_through_its_restoration_phase():
    """Wächter and Biegler's example takes restoration steps and ends with `KKT_TOL`."""
    problem = wachter_biegler()
    _, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(-1.0)
    assert int(record.inner_steps) > 0 and int(record.status) == TerminationStatus.KKT_TOL
