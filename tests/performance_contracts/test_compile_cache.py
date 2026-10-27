"""Compile cache: value changes never retrace, the program is flat in the mesh, inexact-solve
arguments leave an exact solve's program unchanged, batched solves keep their corrector
branches, a checkpoint stores the iteration matrix once, a reverse sweep assembles a root's
matrix inside the solve that factors it, and an undifferentiated checkpoint solve keeps its
compiled temporaries at their measured size."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import rectangle_mesh
from diff_soup.linear import (
    SolveContext,
    SparseTopology,
    Tolerance,
    analyze,
    factor,
    solve,
    solve_checked,
)
from tests.support.problems import DecayODE, observed_recipe, operational_cell
from tests.support.programs import equations

#: Solver settings (requested tolerances, not asserted bounds).
_FINE = {"rtol": 1e-8, "atol": 1e-10}
# measured 0.0 (current rows) and 3.3e-16 (voltage row) from the operating target, CPU VM
# 2026-10-22, pytest tests/performance_contracts/test_compile_cache.py::test_control_and_parameter_values_reuse_one_compiled_solve
_TARGET = 1e-12
_SAVE = ds.SaveAt(ts=jnp.asarray([0.1, 0.3]))


def test_control_and_parameter_values_reuse_one_compiled_solve():
    """A jitted solve given another operating mode, target and parameter value traces once;
    each call holds its own mode's quantity at its own target."""
    problem, traces = operational_cell(decay=0.731).build(), []
    y0 = jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))])

    def run(control, theta):
        traces.append(None)
        solution = ds.pdesolve(
            problem,
            ds.JNDF.stock(linear="dense", **_FINE),
            (0.0, 0.3),
            y0,
            theta=theta,
            control_state0=control,
            saveat=_SAVE,
            consistent_ic=True,
            max_steps=400,
        )
        return solution.ys, solution.lambdas[:, 0]

    compiled, theta = jax.jit(run), ds.default_theta(problem)
    other = jax.tree_util.tree_map(lambda leaf: jnp.where(leaf == 0.731, 0.813, leaf), theta)
    calls = ((0, 0.25, theta), (0, 0.25, other), (1, 0.35, other))
    outputs = [compiled(ds.ControlState.operating(mode=m, target=t), th) for m, t, th in calls]
    assert len(traces) == 1
    assert not np.allclose(outputs[0][0], outputs[1][0])
    for (ys, currents), (mode, target, _) in zip(outputs, calls):
        held = (currents, jnp.mean(ys, axis=1) - 2.0 * currents)[mode]
        np.testing.assert_allclose(held, target, rtol=0.0, atol=_TARGET)


def _diffusion(nx, ny, name="flat"):
    mesh = rectangle_mesh(nx, ny, 1.0, 1.0, ele_type="TRI3")
    spec = ds.problem(f"{name}_{nx}x{ny}").domain("part", mesh)
    spec.param("kappa", value=0.7, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x, y: x * y)

    @spec.equation(writes="u", reads=("u",))
    def diffusion(q):
        return q.p.kappa * q.ops.laplacian(q.u) - 0.3 * q.u

    problem = spec.build()
    return problem, spec.assemble_state({}), ds.default_theta(problem)


@pytest.mark.parametrize(
    "linear",
    [
        pytest.param("klu", marks=pytest.mark.requires_klu),
        pytest.param("cudss", marks=pytest.mark.requires_cudss),
    ],
)
def test_the_program_is_flat_in_the_mesh(linear):
    """Doubling a mesh in each direction leaves the lowered solve with the same operations and
    (the mesh arrays reaching it as arguments, not constants) a text of the same length."""
    texts = []
    for nx, ny in ((12, 9), (24, 18)):
        problem, state, theta = _diffusion(nx, ny)
        solver = ds.JIDA.stock(linear=linear, **{"rtol": 1e-6, "atol": 1e-8})

        def run(th, problem=problem, state=state, solver=solver):
            return ds.pdesolve(problem, solver, (0.0, 0.1), state, theta=th, max_steps=500).ys

        texts.append(jax.jit(run).lower(theta).as_text())
    assert len(texts[0].splitlines()) == len(texts[1].splitlines())
    # measured 498586 / 498100 = 1.0010 on KLU, CPU VM 2026-10-22,
    # pytest tests/performance_contracts/test_compile_cache.py::test_the_program_is_flat_in_the_mesh[klu]
    assert len(texts[1]) <= 1.01 * len(texts[0])


@pytest.mark.parametrize(
    "linear", ["dense", pytest.param("cudss", marks=pytest.mark.requires_cudss)]
)
def test_inexact_solve_arguments_leave_an_exact_solve_program_unchanged(linear):
    """A tolerance, error weights and an initial guess lower an exact factor's solve (and its
    checked solve without weights) to the plain solve's text."""
    n = 6
    rows = np.repeat(np.arange(n), 2).astype(np.int32)
    cols = np.stack([np.arange(n), (np.arange(n) + 1) % n], 1).reshape(-1).astype(np.int32)
    backend = getattr(ds.linear, {"dense": "Dense", "cudss": "CuDSS"}[linear])()
    analysis = analyze(SparseTopology(rows=rows, cols=cols, n_dofs=n), backend=backend)
    values, b = jnp.asarray(np.where(rows == cols, 4.0, -1.0)), jnp.linspace(-1.0, 1.0, n)
    tol = Tolerance(**{"rtol": 1e-8, "atol": 0.0}, max_iter=20)
    weights, guess = jnp.linspace(0.5, 1.5, n), jnp.ones(n)
    with SolveContext() as ctx:
        context = None if type(backend).factor_residency == "arrays" else ctx

        def text(call, **extra):
            program = lambda v, r: call(factor(analysis, v, ctx=context), r, **extra)  # noqa: E731
            return jax.jit(program).lower(values, b).as_text()

        assert text(solve, tol=tol, weights=weights, x0=guess) == text(solve)
        assert text(solve_checked, tol=tol, x0=guess) == text(solve_checked)


# measured 6.0e-8 (JIDA), 1.8e-8 (JBDF) and 9.0e-8 (JNDF) relative between each batched lane
# and the exact decay at rtol 1e-8, CPU VM 2026-10-22,
# pytest tests/performance_contracts/test_compile_cache.py::test_batched_solves_keep_the_corrector_refresh_branches
_LANES = 1e-6
_DECAY, _SCALES, _TRACES = DecayODE(), jnp.asarray([0.5, 1.0, 2.0]), []


@functools.cache
def _final(method):
    """The final state of one decay solve from a scaled start (traces are counted)."""
    solver = getattr(ds, method).stock(linear="dense", **_FINE)

    def final(scale):
        _TRACES.append(method)
        y0 = _DECAY.initial_state() * scale
        return ds.pdesolve(_DECAY.problem, solver, (0.0, 1.0), y0, theta=jnp.asarray(0.7)).ys[-1]

    return final, jax.jit(jax.vmap(final))


@pytest.mark.parametrize("method", ["JIDA", "JBDF", "JNDF"])
def test_batched_solves_keep_the_corrector_refresh_branches(method):
    """Under vmap each lane-reduced predicate (all lanes, any lane) still gates a real branch,
    so a refresh runs only when a lane needs it, and every lane meets its own exact decay."""
    final, batched = _final(method)
    names = [e.primitive.name for e in equations(jax.make_jaxpr(jax.vmap(final))(_SCALES))]
    reduced = names.count("diffsoup_all_lanes") + names.count("diffsoup_any_lane")
    assert names.count("diffsoup_all_lanes") > 0 and names.count("cond") >= reduced
    exact = _SCALES[:, None] * _DECAY.analytical(1.0, 0.7)[None, :]
    np.testing.assert_allclose(batched(_SCALES), exact, rtol=_LANES)


def test_a_batch_width_specialises_once_and_then_stays_cached():
    """A jitted batch of solves retraces for a new lane count, never for new values at it."""
    batched = _final("JNDF")[1]
    batched(_SCALES).block_until_ready()
    before = len(_TRACES)
    for scales in (_SCALES[:2], _SCALES[:2] + 0.01):
        batched(scales).block_until_ready()
    assert len(_TRACES) == before + 1


def test_a_jida_checkpoint_stores_the_iteration_matrix_once():
    """Every bounded loop carrying the checkpoint store holds one leaf of the iteration
    matrix's values per saved state, and the rest of a saved state is the dense factor of that
    matrix and a bounded number of state-length vectors (the state Jacobian is never saved
    beside the matrix, and the save grid is not stored)."""
    n_save, checkpoints = 4, 5
    problem, state, theta = _diffusion(12, 9, name="carry")
    n, nnz = int(state.size), ds.topology_report(problem).nnz
    assert nnz > 6 * n
    solver = ds.JIDA.stock(
        linear="dense",
        adjoint=ds.CheckpointAdjoint(checkpoints=checkpoints),
        **{"rtol": 1e-6, "atol": 1e-8},
    )
    saveat = ds.SaveAt(ts=jnp.linspace(0.0, 0.2, n_save + 1)[1:])

    def loss(th):
        solution = ds.pdesolve(
            problem, solver, (0.0, 0.2), state, theta=th, saveat=saveat, max_steps=400
        )
        return jnp.sum(solution.ys**2)

    stores = []
    for eqn in equations(jax.make_jaxpr(jax.grad(loss))(theta)):
        if eqn.primitive.name == "while":
            carry = eqn.params["body_jaxpr"].jaxpr.invars[eqn.params["body_nconsts"] :]
            stored = [v.aval for v in carry if v.aval.ndim >= 2 and v.aval.shape[0] == checkpoints]
            stores += [stored] if stored else []
    assert stores
    # measured 160,328 bytes per saved state (160,968 with 64 save times) against this bound of
    # 170,280: the dense LU factor and pivots held across corrector sweeps, 8 n^2 + 4 n =
    # 135,720 (stored 135,200 + 520); the matrix and its derivative partial at nnz each; 20
    # state vectors; 32 bytes of per-slot metadata per save time (the saved outputs are loop
    # buffers outside the store), CPU VM 2026-10-22,
    # pytest tests/performance_contracts/test_compile_cache.py::test_a_jida_checkpoint_stores_the_iteration_matrix_once
    bound = 8 * n * n + 4 * n + 8 * (2 * nnz + 20 * n) + 32 * n_save + 512
    for stored in stores:
        assert sum(aval.shape[1:] == (nnz,) for aval in stored) == 1
        assert sum(aval.size * aval.dtype.itemsize for aval in stored) // checkpoints <= bound


def _factored_inside(program):
    """Whether every factorization in ``program`` factors values ``program`` computes itself."""
    made = {var: eqn for eqn in program.eqns for var in eqn.outvars}
    passed = ("stop_gradient", "convert_element_type", "copy", "copy_p")
    factored = []
    for eqn in program.eqns:
        if eqn.primitive.name == "phased_factor":
            values = eqn.invars[0]
            while values in made and made[values].primitive.name in passed:
                values = made[values].invars[0]
            factored.append(values in made)
    return bool(factored) and all(factored)


def test_a_reverse_sweep_assembles_each_root_matrix_inside_its_transposed_solve():
    """In the backward loop of a checkpointed JIDA gradient, the transposed solve of a step
    root's derivative rule assembles the matrix it factors: the matrix is not computed in the
    forward half of the step's vjp, which the loop also runs for every recomputed step."""
    problem, state, theta = _diffusion(12, 9, name="sweep")
    solver = ds.JIDA.stock(
        linear="dense", adjoint=ds.CheckpointAdjoint(checkpoints=5), **{"rtol": 1e-6, "atol": 1e-8}
    )

    def loss(th):
        solution = ds.pdesolve(problem, solver, (0.0, 0.2), state, theta=th, max_steps=400)
        return jnp.sum(solution.ys**2)

    solves = [
        inner.params["jaxprs"].solve.jaxpr
        for eqn in equations(jax.make_jaxpr(jax.grad(loss))(theta))
        if eqn.primitive.name == "while"
        for inner in equations(eqn.params["body_jaxpr"])
        if inner.primitive.name == "custom_linear_solve"
    ]
    assert solves
    assert all(map(_factored_inside, solves))


# measured 780,232 bytes of compiled temporaries, identical over three compiles (argument 2,531
# and output 1,944 bytes; the ForwardAdjoint solve of the same problem 24,776), CPU VM
# 2026-10-22, pytest tests/performance_contracts/test_compile_cache.py::test_an_undifferentiated_checkpoint_solve_keeps_its_compiled_temporaries
_PRIMAL_CHECKPOINT_TEMP = 1.05 * 780_232


def test_an_undifferentiated_checkpoint_solve_keeps_its_compiled_temporaries():
    """A jitted CheckpointAdjoint solve that is never differentiated (JNDF, dense, 24 save
    times, the default max_steps) compiles to temporaries of the measured size, the step
    decision tape a backward replay reads included."""
    problem, y0, rate = observed_recipe()
    solver = ds.JNDF.stock(
        linear="dense", adjoint=ds.CheckpointAdjoint(), **{"rtol": 1e-9, "atol": 1e-11}
    )
    saveat = ds.SaveAt(ts=jnp.linspace(0.05, 1.0, 24))

    def run(value):
        theta = ds.default_theta(problem, {rate: value})
        solution = ds.pdesolve(problem, solver, (0.0, 1.0), y0, theta=theta, saveat=saveat)
        return solution.ys, solution.lambdas, solution.ts

    memory = jax.jit(run).lower(jnp.asarray(1.3)).compile().memory_analysis()
    assert memory.temp_size_in_bytes <= _PRIMAL_CHECKPOINT_TEMP
