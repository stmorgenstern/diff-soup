"""Time stepping: the engine and linear-backend matrix, the transformation safety matrix,
solver statistics and the precision a solve runs in."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from tests.support.problems import (
    ConstrainedDAE,
    DecayODE,
    SparseChainDAE,
    endpoint,
    reaction_diffusion_recipe,
)
from tests.support.programs import equations

_T1 = 0.05
#: Solver settings (requested tolerances, not asserted bounds).
_TIGHT = {"rtol": 1e-6, "atol": 1e-8}
_LOOSE = {"rtol": 1e-4, "atol": 1e-6}
_THETA = jnp.asarray(0.4)
#: The problem each engine is checked on: the ODE for JBDF, the index-one DAE for the others.
_CASES = {
    "jbdf": (ds.JBDF, DecayODE),
    "jndf": (ds.JNDF, ConstrainedDAE),
    "jida": (ds.JIDA, ConstrainedDAE),
}
_ENGINES = tuple(_CASES)
# measured worst closed-form state error 1.6e-6 (vmap lanes, JNDF; 1.0e-6 JBDF on the ODE,
# 4.3e-9 JNDF on the DAE), CPU VM 2026-10-22, pytest tests/public_api/test_time_stepping.py
_STATE_BOUND = 1e-5
# measured worst parameter-derivative error 2.4e-6 (JBDF), CPU VM 2026-10-22,
# pytest tests/public_api/test_time_stepping.py::test_forward_mode_derivatives_match_the_closed_form
_DERIVATIVE_BOUND = 5e-5
# measured worst float32 error 1.2e-4 (JIDA on the DAE at rtol 1e-4), CPU VM 2026-10-22
_SINGLE_BOUND = 1e-3


def _solver(engine, linear="dense", **keywords):
    return engine.stock(linear=linear, **_TIGHT, **keywords)


def _traced(fn, *args):
    """``(equations of fn's program, fn(*args))`` from one trace of ``fn``."""
    traced = jax.jit(fn).trace(*args)
    return equations(traced.jaxpr), traced.lower().compile()(*args)


def _wide_arrays(found, size):
    """Primitives producing a float64 array with a state-sized axis among ``found``."""
    wide = set()
    for eqn in found:
        for var in eqn.outvars:
            aval = getattr(var, "aval", None)
            if getattr(aval, "dtype", None) == jnp.float64 and size in getattr(aval, "shape", ()):
                wide.add(eqn.primitive.name)
    return sorted(wide)


@pytest.fixture(scope="module")
def jitted():
    """Per engine: one jitted program of the end state, its parameter tangent and the solver
    statistics, and the list of its traces."""
    programs = {}
    for name, (engine, kind) in _CASES.items():
        case, traces = kind(), []

        def run(theta, case=case, engine=engine, traces=traces):
            traces.append(None)

            def state(th):
                solution = ds.pdesolve(
                    case.problem, _solver(engine), (0.0, _T1), case.initial_state(), theta=th
                )
                state = solution.ys[-1]
                if solution.lambdas is not None:
                    state = jnp.concatenate([state, solution.lambdas[-1]])
                return state, solution.stats

            return jax.jvp(state, (theta,), (jnp.ones_like(theta),), has_aux=True)

        programs[name] = (case, jax.jit(run), traces)
    return programs


@pytest.mark.parametrize(
    "name,linear",
    [(name, "dense") for name in _ENGINES]
    + [pytest.param(name, "klu", marks=pytest.mark.requires_klu) for name in _ENGINES],
)
def test_every_engine_and_backend_reaches_the_closed_form(jitted, name, linear):
    """JBDF on the ODE and JNDF and JIDA on the DAE reach the exact state with either backend."""
    case, program, _ = jitted[name]
    if linear == "dense":
        state, _, _ = program(_THETA)
    else:
        state = endpoint(
            case.problem, _solver(_CASES[name][0], linear), _THETA, case.initial_state()
        )
    np.testing.assert_allclose(state, case.analytical(_T1, _THETA), rtol=0.0, atol=_STATE_BOUND)


@pytest.mark.parametrize("name", _ENGINES)
def test_a_new_parameter_value_reuses_the_compiled_solve(jitted, name):
    """A jitted solve called with another parameter value is not traced again, and two equal
    solver configurations are equal and hash alike: one compile-cache key."""
    case, program, traces = jitted[name]
    first, second = (_solver(_CASES[name][0], adjoint=ds.CheckpointAdjoint()) for _ in range(2))
    assert first == second and hash(first) == hash(second) and first != _solver(_CASES[name][0])
    program(_THETA)
    state, _, _ = program(jnp.asarray(0.5))
    assert len(traces) == 1
    np.testing.assert_allclose(state, case.analytical(_T1, 0.5), rtol=0.0, atol=_STATE_BOUND)


@pytest.mark.parametrize("name", _ENGINES)
def test_forward_mode_derivatives_match_the_closed_form(jitted, name):
    """The forward-mode tangent of the end state in the parameter is the exact derivative."""
    case, program, _ = jitted[name]
    exact = jax.jacfwd(lambda th: case.analytical(_T1, th))(_THETA)
    np.testing.assert_allclose(program(_THETA)[1], exact, rtol=0.0, atol=_DERIVATIVE_BOUND)


def _nested_problem():
    """A two-state ODE driven by a seven-coordinate nested parameter pytree."""
    projection = jnp.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])

    def rates(theta):
        kinetics = theta["kinetics"]
        return kinetics["base"] + kinetics["offsets"] + projection @ theta["weights"]

    problem = ds.MassProblem(
        rhs=lambda t, y, theta, control: -rates(theta) * y,
        mass=ds.MassOperator.identity(2),
        structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),
    )
    return problem, lambda theta: jnp.exp(-rates(theta) * _T1)


@pytest.mark.parametrize("name", _ENGINES)
def test_a_vmap_over_parameters_is_one_batched_loop_matching_every_lane(name):
    """A vmap over seven parameter lanes carries the batch through one adaptive loop (no
    sequential map or scan over the lanes) and every lane reaches its closed form."""
    problem, analytical = _nested_problem()
    lane = jnp.arange(7.0)
    thetas = {
        "kinetics": {
            "base": 0.3 + 0.02 * lane,
            "offsets": jnp.stack((0.1 + 0.01 * lane, 0.2 + 0.01 * lane), 1),
        },
        "weights": 0.05 * jnp.arange(1.0, 5.0) + 0.001 * lane[:, None],
    }
    batched = jax.vmap(lambda th: endpoint(problem, _solver(_CASES[name][0]), th, jnp.ones(2)))
    found, states = _traced(batched, thetas)
    loops = [e for e in found if e.primitive.name == "while"]
    assert any(v.aval.shape[:1] == (7,) for e in loops for v in e.invars if v.aval.shape)
    assert not any(e.primitive.name == "map" for e in found)
    assert not any(e.primitive.name == "scan" and int(e.params["length"]) == 7 for e in found)
    np.testing.assert_allclose(states, jax.vmap(analytical)(thetas), rtol=0.0, atol=_STATE_BOUND)


@pytest.mark.parametrize("name", _ENGINES)
def test_a_float32_problem_solves_in_float32_from_float64_inputs(name):
    """The problem owns the precision: a float32 problem integrates in float32 although its
    initial state and parameter arrive in float64."""
    engine, kind = _CASES[name]
    single = kind(ds.Precision(compute_dtype="float32"))
    solver, y0 = engine.stock(linear="dense", **_LOOSE), single.initial_state()
    found, state = _traced(partial(endpoint, single.problem, solver, y0=y0), _THETA)
    assert state.dtype == jnp.float32
    assert _wide_arrays(found, y0.shape[0]) == []
    np.testing.assert_allclose(state, single.analytical(_T1, 0.4), rtol=0.0, atol=_SINGLE_BOUND)


def test_solver_statistics_name_the_backend_adjoint_and_linear_work(jitted):
    """The stats record names the realized backend and adjoint, counts one linear solve per
    Newton iteration, multi-right-hand-side solves no more than factorizations, and one
    consistent initialization."""
    case, program, _ = jitted["jida"]
    stats = program(_THETA)[2]
    assert (stats.linear_backend, stats.adjoint, stats.adjoint_checkpoints) == (
        "dense",
        "ForwardAdjoint",
        None,
    )
    assert int(stats.linear_solves) == int(stats.newton_iters) > 0
    assert 0 < int(stats.multirhs_solves) <= int(stats.factorizations)
    assert int(stats.multirhs_columns) == int(stats.multirhs_solves) * case.n_lam
    assert int(stats.trial_steps) >= int(stats.accepts) + int(stats.rejects)
    assert (int(stats.ic_solves), int(stats.event_solves)) == (1, 0)


def _routes(precision, n=9):
    """One reaction-diffusion problem through the recipe and implicit routes."""
    spec = reaction_diffusion_recipe(f"precision_routes_{n}", n=n)
    structure = ds.DAEStructure(differential_vars=np.ones((n,), dtype=bool))

    def rhs(t, y, theta, control):
        return -theta * y

    implicit = ds.ImplicitProblem(
        residual=lambda t, y, ydot, theta, control: ydot - rhs(t, y, theta, control),
        structure=structure,
        linearization=ds.ResidualLinearization(
            sparsity=ds.ResidualSparsity(dy="dense", dydot="dense")
        ),
        precision=precision,
    )
    recipe = spec.build(precision=precision)
    routes = {
        "recipe": (recipe, ds.default_theta(recipe)),
        "implicit": (implicit, _THETA),
    }
    return routes, jnp.asarray(spec.assemble_state({}), jnp.float64)


@pytest.mark.parametrize("route", ("recipe", "implicit"))
def test_a_float32_problem_solves_and_differentiates_in_float32_on_every_route(route):
    """A float32 recipe or implicit problem integrates and takes forward derivatives in
    float32 with no float64 state-sized array in its program, from float64 inputs (the mass
    route is the float32 row of every engine)."""
    routes, y0 = _routes(ds.Precision(compute_dtype="float32"))
    problem, theta = routes[route]
    solver = ds.JIDA(linear="dense", **_LOOSE)
    theta32 = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf, jnp.float32), theta)

    def states(th):
        return ds.pdesolve(problem, solver, (0.0, 0.5), y0, theta=th, saveat=ds.SaveAt(t1=True)).ys

    def tangent(th):
        return jax.jvp(states, (th,), (th,))

    found, (primal, derivative) = _traced(tangent, theta32)
    assert (primal.dtype, derivative.dtype) == (jnp.float32, jnp.float32)
    assert bool(jnp.all(jnp.isfinite(primal)))
    assert _wide_arrays(found, y0.shape[0]) == []


def test_a_solver_inherits_the_problem_precision_and_a_disagreeing_policy_is_refused():
    """A solver inherits a float32 problem's precision, a disagreeing policy or double-only
    backend is refused, and an initialization float32 cannot reach names the float64 policies."""
    single, y0 = DecayODE(ds.Precision.fp32()).problem, DecayODE.initial_state(jnp.float32)

    def solve(problem, **solver):
        solver.setdefault("linear", "dense")
        return ds.pdesolve(
            problem, ds.JNDF(**solver), (0.0, 0.5), y0.astype(jnp.float32), theta=_THETA
        )

    inherited = solve(single)
    assert (inherited.ys.dtype, inherited.stats.linear_solve_mode) == (jnp.float32, "fp32")
    with pytest.raises(ds.ConfigurationError, match="belongs on the problem"):
        solve(single, precision=ds.Precision.fp32_ir())
    with pytest.raises(ds.ConfigurationError, match="double precision only"):
        solve(single, linear="klu")
    with pytest.raises(ds.ConfigurationError, match="double-precision residual"):
        ds.Precision(compute_dtype="float32", solve_mode="fp32_ir")

    below = ds.ToleranceSpec(rtol=1e-4, atol=1e-6, ic_atol=1e-10)  # measured float32 floor 4.9e-8
    chain = [SparseChainDAE(precision=p) for p in (ds.Precision.fp32(), None)]
    ic32, ic64 = (ds.pdesolve(c.problem, ds.JIDA(linear="dense"), (0.0, 0.5), c.initial_state(), theta=_THETA, tolerances=below).result for c in chain)  # fmt: skip
    assert ic32 == ds.RESULT.INCONSISTENT_IC and "Precision.fp32_ir()" in ds.RESULT[ic32]
    assert ic64 == ds.RESULT.SUCCESS


@pytest.mark.requires_cudss
def test_the_stats_tell_the_mixed_float32_policy_from_end_to_end_float32():
    """``stats.linear_solve_mode`` is ``"fp32_factor"`` for float32 factors under float64
    compute and ``"fp32"`` for end-to-end float32, on one problem and backend."""

    def mode(precision, dtype):
        problem, y0 = DecayODE(precision).problem, DecayODE.initial_state(dtype)
        solver = ds.JNDF(linear="cudss", **_LOOSE)
        return ds.pdesolve(problem, solver, (0.0, 0.5), y0, theta=_THETA).stats.linear_solve_mode

    mixed = mode(ds.Precision.fp32_factor(), jnp.float64)
    assert (mixed, mode(ds.Precision.fp32(), jnp.float32)) == ("fp32_factor", "fp32")


# measured 6.5e-7 (residual) and 2.8e-8 (Jacobian) relative to the float64 build in the max
# norm, workstation 2026-10-23,
# pytest tests/public_api/test_time_stepping.py::test_a_float32_recipe_assembles_its_residual_and_jacobian_in_float32
_ASSEMBLY_BOUND = 1e-5


def test_a_float32_recipe_assembles_its_residual_and_jacobian_in_float32():
    """A recipe built with a float32 precision assembles its residual and state Jacobian in
    float32, with no float64 state-sized array in the residual program (the grid stencil's
    float64 geometry is cast), and both agree with the float64 build to single precision."""
    spec = reaction_diffusion_recipe("float32_assembly", n=8)
    start = jnp.asarray(spec.assemble_state({}), jnp.float64)
    assembled = {}
    for dtype, precision in ((jnp.float32, ds.Precision.fp32()), (jnp.float64, None)):
        problem = spec.build(precision=precision)
        theta = jax.tree_util.tree_map(
            lambda leaf, dtype=dtype: jnp.asarray(leaf, dtype), problem.default_theta()
        )
        residual = partial(
            problem.solver_problem().residual,
            jnp.asarray(0.1, dtype),
            ydot=jnp.zeros(start.shape, dtype),
            theta=theta,
            control=None,
        )
        state = start.astype(dtype)
        found = equations(jax.make_jaxpr(residual)(state))
        assembled[dtype] = (residual(state), jax.jacfwd(residual)(state), found)
    rows, jacobian, found = assembled[jnp.float32]
    assert (rows.dtype, jacobian.dtype) == (jnp.float32, jnp.float32)
    assert _wide_arrays(found, start.shape[0]) == []
    for single, double in zip((rows, jacobian), assembled[jnp.float64][:2]):
        error = np.max(np.abs(np.asarray(single, np.float64) - double)) / np.max(np.abs(double))
        assert error <= _ASSEMBLY_BOUND


def _periodic_heat(precision):
    """``T' = 0.5 T_xx`` on a periodic 16-cell ``[0, 2 pi]`` from ``0.3 + sin x``: its start and
    the state at ``t = 0.4``."""
    spec = ds.problem("float32_periodic_heat").domain(
        "line", ds.cartesian_grid_1d(16, 0.0, 2.0 * np.pi, method="fvm")
    )
    spec.field("T", mass="differential", scale=1.0, init=lambda x: 0.3 + jnp.sin(x))
    spec.equation(writes="T", reads=("T",))(lambda q: 0.5 * q.ops.laplacian(q["T"]))
    spec.bc.periodic("T")
    start = spec.assemble_state({})
    solution = ds.pdesolve(
        spec.build(precision=precision),
        ds.JNDF.stock(linear="dense", **{"rtol": 1e-5, "atol": 1e-7}),
        (0.0, 0.4),
        start,
        saveat=ds.SaveAt(t1=True),
    )
    return np.asarray(start), solution.ys[-1]


# measured 1.8e-7 (CPU) and 1.7e-5 (RTX 4070 Ti) from the float64 solve at rtol 1e-5, and
# 1.2e-8 drift of the mean, workstation 2026-10-23,
# pytest tests/public_api/test_time_stepping.py::test_a_float32_periodic_problem_solves_in_float32_and_conserves_its_mean
_PERIODIC_BOUND = 1e-4


@pytest.mark.filterwarnings("ignore:a float32 problem is being solved on a CPU device:UserWarning")
def test_a_float32_periodic_problem_solves_in_float32_and_conserves_its_mean():
    """A periodic heat recipe built in float32 integrates in float32, agrees with its float64
    build to the integrator's tolerance, and keeps its mean temperature."""
    start, single = _periodic_heat(ds.Precision.fp32())
    _, double = _periodic_heat(None)
    assert single.dtype == jnp.float32
    np.testing.assert_allclose(single, double, rtol=0.0, atol=_PERIODIC_BOUND)
    np.testing.assert_allclose(np.mean(single), np.mean(start), rtol=0.0, atol=_PERIODIC_BOUND)
