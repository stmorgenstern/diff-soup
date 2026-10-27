"""Model-axis cuDSS factors: one factorization spread over the devices of a "model" mesh axis
solves and differentiates, through the analyze / factor / solve contract and through an
integrator's corrector, as the one-device factor does. Every row but the refusal on the
backends that cannot serve a model axis needs cuDSS and two devices (one process, cuDSS MG; three
or four for the rows on a model group that leaves out the first device) or two processes (cuDSS MGMN)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import diff_soup as ds
from diff_soup.linear import (
    CuDSS,
    SolveContext,
    SolveStatus,
    SparseTopology,
    analyze,
    empty_factor,
    factor,
    invalidate,
    refactor,
    solve,
    solve_checked,
)
from tests.support.problems import endpoint, operational_cell

_ROOT = Path(__file__).resolve().parents[2]
N = 6
# measured 6.5e-16 on the dense reference, CPU VM 2026-10-22, the Dense contract on this system
# (solves, transposed and "AT" solves, refactors, the scan, gradients, jacfwd); the cuDSS
# measurement is owed to the GPU gate
EXACT = {"rtol": 1e-10, "atol": 1e-12}


def _system(seed=3):
    """A 6 x 6 unsymmetric tridiagonal pattern with four long-range entries."""
    pairs = [(i, j) for i in range(N) for j in (i - 1, i, i + 1) if 0 <= j < N]
    pairs += [(0, 4), (5, 1), (2, 5), (4, 0)]
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
    values = np.random.default_rng(seed).uniform(-1, 1, rows.size) + np.where(
        rows == cols, 4.0, 0.0
    )
    return SparseTopology(rows=rows, cols=cols, n_dofs=N), jnp.asarray(values)


def _matrix(topology, values):
    a = np.zeros((N, N))
    np.add.at(a, (topology.rows, topology.cols), np.asarray(values))
    return a


def _model_axis(topology):
    """The analysis of ``topology`` over every device on one "model" axis."""
    layout = ds.Layout(mesh=ds.global_mesh(("model",)))
    return analyze(topology, backend=CuDSS(), layout=layout)


_B = jnp.linspace(-1.0, 1.0, N)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_model_axis_factor_solves_as_the_one_device_factor():
    """One factorization over two local GPUs (cuDSS MG) solves the system as the one-device
    cuDSS factor and NumPy do."""
    topology, values = _system()
    with SolveContext() as ctx:
        x = solve(factor(_model_axis(topology), values, ctx=ctx), _B)[0]
        local = solve(factor(analyze(topology, backend=CuDSS()), values, ctx=ctx), _B)[0]
    np.testing.assert_allclose(x, local, **EXACT)
    np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, values), _B), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_model_axis_factor_keeps_the_values_its_predicate_chose():
    """A false predicate keeps solving with the held values, a true one with the new values; a
    scan that solves with its carried factor and refactors it each step solves each step with
    the previous step's values; an empty factor reports EMPTY until a refactor fills it."""
    topology, values = _system()
    other = 1.5 * values + 0.1
    analysis = _model_axis(topology)
    steps = jnp.stack([values * (1.0 + 0.1 * k) for k in range(4)])

    def step(held, v):
        x, held = solve(held, _B)
        return refactor(held, v, True), x

    with SolveContext() as ctx:
        f = factor(analysis, values, ctx=ctx)
        kept = solve(refactor(f, other, False), _B)[0]
        taken = solve(refactor(f, other, True), _B)[0]
        xs = jax.jit(lambda v, vs: jax.lax.scan(step, factor(analysis, v, ctx=ctx), vs)[1])(
            values, steps
        )
        empty, _, before = solve_checked(empty_factor(analysis, ctx=ctx), _B)
        filled, _, after = solve_checked(
            refactor(empty_factor(analysis, ctx=ctx), values, True), _B
        )
    np.testing.assert_allclose(kept, np.linalg.solve(_matrix(topology, values), _B), **EXACT)
    np.testing.assert_allclose(taken, np.linalg.solve(_matrix(topology, other), _B), **EXACT)
    for x, v in zip(xs, [values, *steps[:-1]]):
        np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, v), _B), **EXACT)
    assert int(before.status) == SolveStatus.EMPTY and not np.isfinite(np.asarray(empty)).any()
    assert int(after.status) == SolveStatus.SUCCESS
    np.testing.assert_allclose(filled, np.linalg.solve(_matrix(topology, values), _B), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_model_axis_factor_refactors_in_place_and_solves_as_the_one_device_factor():
    """A model-axis factor (cuDSS MG) refactored in place with new values twice counts one
    factorization and then one refactorization per new set of values, and each solve matches
    the one-device cuDSS factor of the same values."""
    topology, values = _system()
    news = (1.5 * values + 0.1, 1.2 * values - 0.05)
    with SolveContext() as ctx:
        CuDSS.reset_contract_stats()
        held, xs = factor(_model_axis(topology), values, ctx=ctx), []
        for v in news:
            held = refactor(held, v, True, adopt=True)
            xs.append(jax.block_until_ready(solve(held, _B)[0]))
        stats = CuDSS.contract_stats()
        local = analyze(topology, backend=CuDSS())
        expected = [solve(factor(local, v, ctx=ctx), _B)[0] for v in news]
    assert (stats["factorizations"], stats["refactorizations"], stats["solves"]) == (1, 2, 2)
    for x, want in zip(xs, expected):
        np.testing.assert_allclose(x, want, **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_model_axis_factor_solves_after_invalidation_and_in_both_orientations():
    """An invalidated factor solves as before; a transposed solve and a factor of the "AT"
    orientation both solve the transposed system."""
    topology, values = _system()
    a = _matrix(topology, values)
    analysis = _model_axis(topology)
    with SolveContext() as ctx:
        f = factor(analysis, values, ctx=ctx)
        plain = solve(f, _B)[0]
        again = solve(invalidate(f), _B)[0]
        transposed = solve(f, _B, transpose=True)[0]
        flipped = solve(factor(analysis, values, ctx=ctx, orientation="AT"), _B)[0]
    np.testing.assert_allclose(again, plain, **EXACT)
    np.testing.assert_allclose(plain, np.linalg.solve(a, _B), **EXACT)
    for x in (transposed, flipped):
        np.testing.assert_allclose(x, np.linalg.solve(a.T, _B), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_model_axis_derivatives_match_the_closed_forms():
    """The gradient of sum(x) is -adjoint[rows] * x[cols] in the values and the adjoint in the
    right-hand side; the forward Jacobian in the right-hand side is the inverse."""
    topology, values = _system()
    a = _matrix(topology, values)
    analysis = _model_axis(topology)
    with SolveContext() as ctx:

        def run(v, b):
            return solve(factor(analysis, v, ctx=ctx), b)[0]

        by_values, by_rhs = jax.grad(lambda v, b: jnp.sum(run(v, b)), argnums=(0, 1))(values, _B)
        jacobian = jax.jacfwd(lambda b: run(values, b))(_B)
    x, adjoint = np.linalg.solve(a, _B), np.linalg.solve(a.T, np.ones(N))
    np.testing.assert_allclose(by_values, -adjoint[topology.rows] * x[topology.cols], **EXACT)
    np.testing.assert_allclose(by_rhs, adjoint, **EXACT)
    np.testing.assert_allclose(jacobian, np.linalg.inv(a), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_model_axis_factor_solves_one_then_four_then_one_right_hand_sides():
    """One model-axis factor (cuDSS MG) solves one right-hand side, then a block of four, then
    one again, each as the one-device cuDSS factor does."""
    topology, values = _system()
    block = jnp.stack([(k + 1.0) * _B + k for k in range(4)])
    with SolveContext() as ctx:
        spread = factor(_model_axis(topology), values, ctx=ctx)
        local = factor(analyze(topology, backend=CuDSS()), values, ctx=ctx)
        pairs = []
        for b in (_B, block, _B):
            if b.ndim == 1:
                pairs.append((solve(spread, b)[0], solve(local, b)[0]))
            else:
                pairs.append(
                    tuple(jax.vmap(lambda r, f=f: solve(f, r)[0])(b) for f in (spread, local))
                )
    for x, expected in pairs:
        np.testing.assert_allclose(x, expected, **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_three_devices
def test_a_model_axis_factors_over_the_devices_of_its_model_group():
    """A model axis over the second and third devices (not the lowest ordinals) factors on that
    group: with its arrays on the group's first device it solves as the one-device factor does,
    and a solve whose arrays are on the first device, outside the group, is refused naming the
    process's model groups."""
    topology, values = _system()
    group = jax.devices()[1:3]
    layout = ds.Layout(mesh=Mesh(np.asarray(group), ("model",), axis_types=(AxisType.Auto,)))
    analysis = analyze(topology, backend=CuDSS(), layout=layout)
    named = ", ".join(str(device.local_hardware_id) for device in group)
    with SolveContext() as ctx:
        local = solve(factor(analyze(topology, backend=CuDSS()), values, ctx=ctx), _B)[0]
        on_group, b_on_group = jax.device_put((values, _B), group[0])
        x = solve(factor(analysis, on_group, ctx=ctx), b_on_group)[0]
        outside, b_outside = jax.device_put((values, _B), jax.devices()[0])
        with pytest.raises(
            jax.errors.JaxRuntimeError, match=re.escape(f"in none of its model groups [{named}]")
        ):
            jax.block_until_ready(solve(factor(analysis, outside, ctx=ctx), b_outside)[0])
    np.testing.assert_allclose(x, local, **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_four_devices
def test_each_model_group_of_a_data_and_model_mesh_solves_on_its_own_devices():
    """On a ("data", "model") mesh of 2 x 2 devices in one process (cuDSS MG), a solve whose
    arrays are on the first device of either model group factors on that group and solves as
    the one-device factor does."""
    topology, values = _system()
    mesh = ds.global_mesh(("data", "model"), shape=(2, 2))
    analysis = analyze(topology, backend=CuDSS(), layout=ds.Layout(mesh=mesh))
    with SolveContext() as ctx:
        local = solve(factor(analyze(topology, backend=CuDSS()), values, ctx=ctx), _B)[0]
        xs = []
        for group in mesh.devices:
            on_group, b_on_group = jax.device_put((values, _B), group[0])
            xs.append(solve(factor(analysis, on_group, ctx=ctx), b_on_group)[0])
    for x in xs:
        np.testing.assert_allclose(x, local, **EXACT)


# The bound admits a step sequence that a rounding difference between two factorizations moves
# (the solve's rtol is 1e-8); the cuDSS measurement is owed to the GPU gate. Measured 0.0 between
# a sparse direct backend's JIDA solve and the dense one at rtol 1e-8, CPU VM 2026-10-22,
# pytest tests/contracts/test_linear_host.py::test_a_batched_solve_above_ten_unknowns_runs_on_klu_lane_by_lane[klu];
# the cross-process rows' end state and decay gradient: measured 0.0 and 0.0 between KLU and
# dense on the decaying cell, arm64 Mac 2026-10-23 (the MGMN measurement is owed to the GPU gate)
ENGINE = {"rtol": 1e-6, "atol": 1e-8}


def _end_state(layout, linear):
    """The end state of JIDA on a small DAE recipe (four cells and an algebraic current held
    at its target) from a start off the algebraic manifold."""
    return endpoint(
        operational_cell().build(layout=layout),
        ds.JIDA.stock(linear=linear, **{"rtol": 1e-8, "atol": 1e-10}),
        None,
        jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))]),
        t1=0.3,
        control_state0=ds.ControlState.operating(mode=0, target=0.25),
        consistent_ic=True,
        max_steps=400,
    )


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_an_integrator_on_a_model_axis_layout_solves_as_on_one_device():
    """A recipe built with a "model" layout over the devices of one process (cuDSS MG) and
    solved by JIDA on cuDSS with a consistent initialization ends where the one-device cuDSS and
    the dense solves end. One process takes its arrays as they are: each factor spreads over
    every device of the model axis."""
    on_model_axis = _end_state(ds.Layout(mesh=ds.global_mesh(("model",))), "cudss")
    np.testing.assert_allclose(on_model_axis, _end_state(None, "cudss"), **ENGINE)
    np.testing.assert_allclose(on_model_axis, _end_state(None, "dense"), **ENGINE)


def _decaying_cell(layout, linear):
    """``(solve, arguments)``: the end state of JIDA (checkpointed reverse mode) on the cell
    whose differential rows decay at a learnable rate, built with ``layout``, as a function of
    ``(theta, y0, control)``, and those three arguments."""
    problem = operational_cell(decay=0.8).build(layout=layout)
    solver = ds.JIDA.stock(
        linear=linear, adjoint=ds.CheckpointAdjoint(), **{"rtol": 1e-8, "atol": 1e-10}
    )

    def solve(theta, y0, control):
        return endpoint(
            problem,
            solver,
            theta,
            y0,
            t1=0.3,
            control_state0=control,
            consistent_ic=True,
            max_steps=400,
        )

    arguments = (
        ds.default_theta(problem),
        jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))]),
        ds.ControlState.operating(mode=0, target=0.25),
    )
    return solve, arguments


def _across_processes(transform):
    """``transform(solve)`` of the decaying cell on a model axis over every process's GPU
    (cuDSS MGMN), its arguments placed on the layout's mesh, read on this process; and the same
    transform of the one-device dense solve."""
    layout = ds.Layout(mesh=ds.global_mesh(("model",)))
    solve, arguments = _decaying_cell(layout, "cudss")
    placed = jax.device_put(arguments, NamedSharding(layout.mesh, P()))
    result = jax.tree.map(ds.distributed.local_value, jax.jit(transform(solve))(*placed))
    reference, local = _decaying_cell(None, "dense")
    return result, jax.jit(transform(reference))(*local)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_processes
def test_an_integrator_on_a_model_axis_across_processes_ends_where_one_device_ends():
    """On a world of at least two processes with one GPU each (cuDSS MGMN), JIDA on cuDSS with a
    consistent initialization solves a recipe built with a "model" layout, its parameters, start
    and control state placed on the layout's mesh, to the end state of the one-device dense
    solve."""
    end, expected = _across_processes(lambda solve: solve)
    np.testing.assert_allclose(end, expected, **ENGINE)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_processes
def test_a_gradient_through_a_model_axis_across_processes_equals_the_one_device_gradient():
    """On a world of at least two processes with one GPU each (cuDSS MGMN), the reverse-mode
    gradient (CheckpointAdjoint) of the summed end state in the parameters, whose reverse sweep
    solves transposed systems with the factorization that spans the processes, equals the
    gradient of the one-device dense solve."""
    gradient, expected = _across_processes(
        lambda solve: jax.grad(lambda theta, *rest: jnp.sum(solve(theta, *rest)))
    )
    for got, want in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, want, **ENGINE)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_processes
def test_a_model_axis_across_processes_solves_on_the_mesh_without_a_reduction():
    """On a world of at least two processes with one GPU each (cuDSS MGMN), a factor over the
    model axis, in a program whose arrays are placed on the layout's mesh, solves as NumPy does
    and its compiled program holds no all-reduce: cuDSS writes the whole solution on every
    process."""
    topology, values = _system()
    layout = ds.Layout(mesh=ds.global_mesh(("model",)))
    analysis = analyze(topology, backend=CuDSS(), layout=layout)
    placed = NamedSharding(layout.mesh, P())
    values, b = (jax.device_put(array, placed) for array in (values, _B))
    with SolveContext() as ctx:
        run = jax.jit(lambda v, b: solve(factor(analysis, v, ctx=ctx), b)[0])
        compiled = run.lower(values, b).compile()
        x = ds.distributed.local_value(compiled(values, b))
    assert "all-reduce" not in compiled.as_text().lower()
    np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, values), _B), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_processes
def test_a_model_axis_across_processes_gives_every_process_one_solution_without_placement():
    """On a world of at least two processes with one GPU each (cuDSS MGMN), a factor over the
    model axis in a program that takes its arrays unplaced (each process compiles for its own
    device) solves as NumPy does, and every process holds the same solution."""
    topology, values = _system()
    analysis = analyze(topology, backend=CuDSS(), layout=ds.Layout(mesh=ds.global_mesh(("model",))))
    with SolveContext() as ctx:
        x = jax.jit(lambda v, b: solve(factor(analysis, v, ctx=ctx), b)[0])(values, _B)
    every = multihost_utils.process_allgather(x)
    for other in every[1:]:
        np.testing.assert_array_equal(other, every[0])
    np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, values), _B), **EXACT)


@pytest.mark.requires_cudss
@pytest.mark.requires_two_processes
def test_distributed_input_across_processes_solves_as_central_input_on_every_process():
    """On a world of at least two processes with one GPU each (cuDSS MGMN), a factor analyzed
    with ``mgmn_input="distributed"`` (each process hands cuDSS its own block of rows) solves one
    right-hand side, a block of three and the transposed system as the factor with central input
    does, on every process."""
    topology, values = _system()
    layout = ds.Layout(mesh=ds.global_mesh(("model",)))
    block = jnp.stack([(k + 1.0) * _B + k for k in range(3)])
    results = {}
    for mode in ("central", "distributed"):
        analysis = analyze(topology, backend=CuDSS(), layout=layout, options={"mgmn_input": mode})
        with SolveContext() as ctx:
            f = factor(analysis, values, ctx=ctx)
            results[mode] = (
                solve(f, _B)[0],
                jax.vmap(lambda r, f=f: solve(f, r)[0])(block),
                solve(f, _B, transpose=True)[0],
            )
    for central, distributed in zip(results["central"], results["distributed"]):
        np.testing.assert_allclose(distributed, central, **EXACT)


#: Run on two forced host CPU devices: every backend but cuDSS refuses a two-device model axis
#: at analysis (before any native library is loaded), and the dense backend accepts a two-device
#: data axis.
_WORKER = r"""
import json

import jax
import numpy as np

import diff_soup as ds
from diff_soup.linear import KLU, CuDSS, Dense, SparseTopology, SuperLU, analyze

assert jax.device_count() == 2, jax.devices()
topology = SparseTopology(rows=np.arange(2), cols=np.arange(2), n_dofs=2)
report = {"refusals": {}}
for name, backend in (("Dense", Dense()), ("KLU", KLU()), ("SuperLU", SuperLU())):
    try:
        analyze(topology, backend=backend, layout=ds.Layout(mesh=ds.global_mesh(("model",))))
        report["refusals"][name] = None
    except ds.ConfigurationError as error:
        report["refusals"][name] = str(error)
data = analyze(topology, backend=Dense(), layout=ds.Layout(mesh=ds.global_mesh(("data",))))
report["data_axis_backend"] = type(data.backend).__name__
print("REPORT " + json.dumps(report))
"""


@pytest.fixture(scope="module")
def two_host_devices():
    """The worker's report from one subprocess on two forced host devices."""
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        JAX_ENABLE_X64="1",
        JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", _WORKER],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


def test_backends_without_model_axis_factors_refuse_a_model_axis(two_host_devices):
    """Dense, KLU and SuperLU refuse a two-device model axis at analysis, naming themselves as
    diff_soup.linear exports them, the model axis and CuDSS; a two-device data axis is not a
    model axis and is accepted."""
    refusals = two_host_devices["refusals"]
    assert set(refusals) == {"Dense", "KLU", "SuperLU"}
    for name, message in refusals.items():
        assert message.startswith(f"{name} cannot factor across local devices on the model axis")
        assert "CuDSS" in message
    assert two_host_devices["data_axis_backend"] == "Dense"


@pytest.mark.requires_cudss
@pytest.mark.requires_two_devices
def test_a_cudss_model_axis_keeps_float64_factors_and_refuses_only_an_asked_downcast():
    """A model axis keeps its factors in float64: ``downcast_factors`` at ``None`` or ``False``
    is accepted without a refusal, and ``True`` is refused naming the option and the axis. On
    the visible CUDA devices: a model axis over host devices is refused at analysis."""
    topology, _ = _system()
    layout = ds.Layout(mesh=ds.global_mesh(("model",)))
    for value in (None, False):
        analyze(topology, backend=CuDSS(), layout=layout, options={"downcast_factors": value})
    with pytest.raises(ds.ConfigurationError, match="downcast_factors") as refused:
        analyze(topology, backend=CuDSS(), layout=layout, options={"downcast_factors": True})
    assert "'model' mesh axis" in str(refused.value)
