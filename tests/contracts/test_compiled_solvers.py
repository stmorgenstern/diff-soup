"""Compiled constrained solvers: solutions in vmap lanes, each Hessian mode, the implicit
derivative in args, its reverse form and the adjoint that selects it, its Gauss-Newton form, a
forward-only residual under forward-over-forward, a runtime budget and tolerance, and how a run
ends: degenerate, infeasible, out of steps, diverged, a failed subproblem, restoration, failed
evaluations, float32."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import diff_soup as ds
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
    constrained,
    hs7,
    hs35,
    least_squares,
    objective,
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
_TANGENT = 1e-9
# measured 0 (every solver), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_LANE_GRADIENT = 1e-12
# measured 0 (point and jacobian), workstation CPU 2026-11-06, pytest tests/contracts/test_compiled_solvers.py
_COMPOSITION = 1e-10
# measured 4.1e-16 (worst over solvers and lanes), workstation CPU 2026-11-06,
# pytest tests/contracts/test_compiled_solvers.py::test_the_reverse_form_gradient_equals_the_forward_form_jacobian
_REVERSE_FORM = 1e-10
# measured 8.4e-13 (worst of jit and lanes, tangents and cotangents), workstation CPU 2026-11-08,
# pytest tests/contracts/test_compiled_solvers.py::test_the_gauss_newton_form_equals_the_exact_form_at_a_zero_residual_solution
_RULE_EQUALITY = 1e-10


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


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_reverse_form_gradient_equals_the_forward_form_jacobian(solver):
    """`jax.grad` through `adjoint=ImplicitAdjoint()`, in jitted vmap lanes, equals `jax.jacfwd`
    through the default forward rule, lane by lane."""
    toy = TOYS["inequality"]

    def lane(adjoint, a):
        x, record = dataclasses.replace(SOLVERS[solver], adjoint=adjoint).run(
            toy.problem, toy.problem.x0, a
        )
        return jnp.sum(jnp.sin(x)) + jnp.sum(record.multipliers_inequality) + record.value

    reverse = jax.jit(jax.vmap(jax.grad(lambda a: lane(ds.nonlinear.ImplicitAdjoint(), a))))
    forward = jax.jit(jax.vmap(jax.jacfwd(lambda a: lane(ds.ForwardAdjoint(), a))))
    by_reverse, by_forward = reverse(toy.args_batch), forward(toy.args_batch)
    assert float(jnp.max(jnp.abs(by_reverse - by_forward)) / jnp.max(jnp.abs(by_forward))) <= (
        _REVERSE_FORM
    )


def test_a_solver_refuses_an_adjoint_without_an_implicit_rule():
    """A compiled solver's adjoint selects the forward or the reverse implicit rule; another raises."""
    toy = TOYS["inequality"]
    solver = dataclasses.replace(SOLVERS["interior_point"], adjoint=ds.CheckpointAdjoint())
    with pytest.raises(ds.ConfigurationError):
        solver.run(toy.problem, toy.problem.x0, toy.args_batch[0])


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


def _duplicate_rows():
    """`min |x - 2|^2` with `x1 + x2 <= a` twice: both rows active, `x* = (a, a) / 2`."""
    row = lambda x, a: x[0] + x[1] - a  # noqa: E731
    return constrained(
        objective([0.0, 0.0], lambda x: jnp.sum((x - 2.0) ** 2)),
        1.0,
        inequality=lambda x, a: jnp.stack([row(x, a), row(x, a)]),
        n_ineq=2,
    )


def test_two_identical_active_rows_are_degenerate_to_the_interior_point():
    """The interior point marks an active set whose rows repeat as degenerate."""
    problem = _duplicate_rows()
    _, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(1.0)
    assert bool(record.degenerate)


def test_the_augmented_lagrangian_solves_two_identical_active_rows():
    """On repeated active rows the augmented Lagrangian ends at `x* = (a, a) / 2`."""
    problem = _duplicate_rows()
    x, _ = jax.jit(lambda a: SOLVERS["augmented_lagrangian"].run(problem, problem.x0, a))(1.0)
    assert float(jnp.max(jnp.abs(x - 0.5))) <= _SOLUTION


def _infeasible():
    """`min x1 + x2` with `x1^2 + x2^2 + a <= 0`, which no point satisfies for `a > 0`."""
    return constrained(
        objective([1.0, 1.0], lambda x: x[0] + x[1]),
        1.0,
        inequality=lambda x, a: jnp.stack([x @ x + a]),
        n_ineq=1,
    )


def _nan_region():
    """`(x - 2)^2` below `x = 1` and NaN from there on, so the iterates crowd the edge."""
    value = lambda x: jnp.where(x[0] < 1.0, (x[0] - 2.0) ** 2, jnp.nan)  # noqa: E731
    return constrained(objective([0.0], value), 1.0)


def _unbounded():
    """`min -x0 - x1` without constraints, which has no minimizer."""
    return constrained(objective([0.0, 0.0], lambda x: -x[0] - x[1]), 1.0)


_STATUS = TerminationStatus
# How a run ends: the solver and its options, the problem, the value of `args`, the status.
ENDS = {
    "infeasible-interior_point": ("interior_point", {}, _infeasible, 1.0, _STATUS.INFEASIBLE),
    "infeasible-augmented_lagrangian": (
        "augmented_lagrangian",
        {},
        _infeasible,
        1.0,
        _STATUS.INFEASIBLE,
    ),
    "nan_region": ("interior_point", {}, _nan_region, 1.0, _STATUS.SOLVER_FAILED),
    "unbounded": ("interior_point", {}, _unbounded, 1.0, _STATUS.DIVERGED),
    "failed_subproblem": (
        "sqp",
        {"qp_max_steps": 1},
        lambda: hs35().problem,
        3.0,
        _STATUS.SOLVER_FAILED,
    ),
    **{
        f"max_steps-{name}": (name, {"max_steps": 1}, lambda: hs7().problem, 4.0, _STATUS.MAX_STEPS)
        for name in SOLVERS
    },
    # Each solver on the tolerances float32 resolves; measured: KKT error 2.1e-6, |x - x*| 5.2e-6,
    # multiplier error 4.3e-6 (interior point and SQP), 4.8e-7, 9.2e-7, 5.8e-7 (augmented
    # Lagrangian), workstation CPU 2026-11-08
    **{
        f"float32-{name}": (
            name,
            dict.fromkeys(tolerances, None),
            lambda: hs35(jnp.float32).problem,
            jnp.float32(3.0),
            _STATUS.KKT_TOL,
        )
        for name, tolerances in {
            "interior_point": ("tol",),
            "sqp": ("tol", "qp_tol"),
            "augmented_lagrangian": ("tol", "feasibility_tol"),
        }.items()
    },
}


@pytest.mark.parametrize("end", ENDS)
def test_each_way_a_run_ends_sets_its_status(end):
    """Infeasible rows, a NaN region, an unbounded objective, a QP allowed one step, a run
    allowed one step and each solver in float32 on its default tolerances each end with their own
    status, with every floating record in the iterate's dtype."""
    name, options, make, a, status = ENDS[end]
    problem = make()
    solver = dataclasses.replace(SOLVERS[name], **options)
    x, record = jax.jit(lambda a: solver.run(problem, problem.x0, a))(a)
    assert int(record.status) == status
    floating = [v for v in jax.tree.leaves(record) if jnp.issubdtype(v.dtype, jnp.floating)]
    assert x.dtype == problem.x0.dtype and {v.dtype for v in floating} == {x.dtype}


def _wachter_biegler():
    """Wächter and Biegler's example, on which an interior point started infeasible needs restoration.

    `min x0` with `x0^2 - x1 + a = 0`, `x0 - x2 - 1/2 = 0` and `x1, x2 >= 0`, from
    `(-0.5, 0.1, 0.1)`; for `a = -1`, `x* = (1, 0, 1/2)`.
    """
    lower = jnp.array([-jnp.inf, 0.0, 0.0])
    return constrained(
        objective([-0.5, 0.1, 0.1], lambda x: x[0], (lower, None)),
        -1.0,
        equality=lambda x, a: jnp.stack([x[0] ** 2 - x[1] + a, x[0] - x[2] - 0.5]),
        n_eq=2,
    )


def test_the_interior_point_solves_through_its_restoration_phase():
    """Wächter and Biegler's example takes restoration steps and ends with `KKT_TOL`."""
    problem = _wachter_biegler()
    _, record = jax.jit(lambda a: SOLVERS["interior_point"].run(problem, problem.x0, a))(-1.0)
    assert int(record.inner_steps) > 0 and int(record.status) == TerminationStatus.KKT_TOL


def _zero_residual_rows(z, a):
    """Rows that vanish at `z* = (1/2, 7/10 + w1/10, 2/5)` for the data `a["y"]` made there, and
    two linear rows that hold `z0` on its lower bound and `z1` on its inequality."""
    w = a["w"]
    model = jnp.stack(
        [
            jnp.exp(w[0] * z[2]) * z[0] + w[1] * z[1],
            z[1] * z[2] + w[1] * z[2] ** 2 + w[0] * z[0],
            jnp.sin(z[2]) * (1.0 + w[0]) + z[0] * z[1],
        ]
    )
    return jnp.concatenate([model - a["y"], jnp.stack([z[0] - 0.2, z[1] - 0.3])])


def _zero_residual_args(w):
    """`args` whose data the model meets exactly at `z*`."""
    z = jnp.array([0.5, 0.7 + 0.1 * w[1], 0.4])
    return {"w": w, "y": _zero_residual_rows(z, {"w": w, "y": jnp.zeros(3)})[:3]}


def test_the_gauss_newton_form_equals_the_exact_form_at_a_zero_residual_solution():
    """Under `hessian="gauss-newton"` the implicit derivative takes the Gauss-Newton form, which at
    a zero-residual solution with an active bound and an active inequality gives the exact form's
    tangents and cotangents, jitted and in lanes."""
    problem = LeastSquaresProblem.from_residual(
        jnp.array([1.0, 1.0, 0.0]),
        _zero_residual_rows,
        args=_zero_residual_args(jnp.array([0.3, 0.6])),
        bounds=(jnp.array([0.5, -jnp.inf, -jnp.inf]), None),
    ).with_constraints(
        Constraints(inequality=lambda z, a: 0.7 + 0.1 * a["w"][1:] - z[1:2], n_inequality=1)
    )
    ws = jnp.array([[0.3, 0.6], [0.4, 0.4], [0.15, 0.9]])
    x0 = jnp.broadcast_to(problem.x0, (3, 3))

    def lane(solver, w, data):
        x, record = solver.run(problem, problem.x0, {"w": w, "y": data})
        return _flat(x, record), jnp.sum(jnp.sin(x)) + jnp.sum(_flat(x, record)[3:])

    def derivatives(hessian):
        # measured: both modes end KKT_TOL at 1.3e-13, workstation CPU 2026-11-08
        solver = InteriorPoint(hessian=hessian, tol=1e-12)
        reverse = dataclasses.replace(solver, adjoint=ds.nonlinear.ImplicitAdjoint())
        data = jax.vmap(lambda w: _zero_residual_args(w)["y"])(ws)
        jitted = jax.jit(jax.jacfwd(lambda w: lane(solver, w, data[0])[0]))(ws[0])
        grad = jax.jit(jax.grad(lambda w: lane(reverse, w, data[0])[1]))(ws[0])

        def in_lanes(s, w):
            x, record = ds.optimization.lanes(s, problem, {"w": w, "y": data}, x0)
            return jax.vmap(_flat)(x, record)

        tangent = jax.jacfwd(lambda w: in_lanes(solver, w))(ws)
        cotangent = jax.grad(lambda w: jnp.sum(jnp.sin(in_lanes(reverse, w))))(ws)
        return jitted, grad, tangent, cotangent

    gauss_newton, exact = derivatives("gauss-newton"), derivatives("exact")
    errors = [
        float(jnp.max(jnp.abs(g - e)) / jnp.max(jnp.abs(e))) for g, e in zip(gauss_newton, exact)
    ]
    assert max(errors) <= _RULE_EQUALITY


def _decay_rows(x, a):
    """A decay fitted to data it does not quite meet: a nonzero residual."""
    t = jnp.linspace(0.0, 3.0, 40)
    return x[0] * jnp.exp(-x[1] * t) + x[2] - a * (jnp.exp(-0.7 * t) + 0.05 * jnp.sin(5.0 * t))


def test_a_runtime_budget_and_tolerance_run_without_a_new_compile():
    """`run(max_steps=, tol=)` takes both as runtime values: three budgets trace once, and a loose
    tolerance stops the run sooner, within that tolerance and above the tight one."""
    problem = LeastSquaresProblem.from_residual(jnp.array([1.0, 0.5, 0.0]), _decay_rows, args=2.0)
    # measured: KKT error 6.8e-14 after 9 steps, 3.4e-4 after 4 at 1e-3, workstation CPU 2026-11-08
    solver = InteriorPoint(hessian="gauss-newton", tol=1e-12, max_steps=200)
    traces = []

    def run(max_steps, tol):
        traces.append(max_steps)
        return solver.run(problem, problem.x0, 2.0, max_steps=max_steps, tol=tol)[1]

    run = jax.jit(run)
    budgets = [run(jnp.int32(m), 1e-12) for m in (2, 5, 200)]
    loose = run(jnp.int32(200), 1e-3)
    tight = budgets[-1]
    assert len(traces) == 1 and [int(r.steps) for r in budgets[:2]] == [2, 5]
    assert int(loose.status) == int(tight.status) == TerminationStatus.KKT_TOL
    assert 1e-12 < float(loose.kkt_error) <= 1e-3 and int(loose.steps) < int(tight.steps)
