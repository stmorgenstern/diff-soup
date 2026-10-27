"""Sharded joints: reverse and forward fits equal to the mesh-free joint (on forced host devices,
and reverse mode on the visible devices), step balancing, heterogeneous groups, meshes, robust
reductions."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference import design
from tests.support.inference import decay_recipe, decay_study

_ROOT = Path(__file__).resolve().parents[2]

#: Run on two forced host CPU devices: a four-experiment joint over a two-device mesh whose
#: measurements load from counting callables, and the same joint without a mesh; four
#: experiments sharing one observation function (one kernel) fitted over the mesh with and
#: without step balancing, then reordered; a joint of a windowed and a single-shoot experiment
#: over the mesh and without it.
_WORKER = r"""
import json
import logging

import jax
import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup import inference as inf
from tests.support.inference import (
    TIMES,
    decay_experiment,
    decay_recipe,
    decay_values,
    mean_measurement,
    solver,
)

assert jax.device_count() == 2, jax.devices()
decay = decay_recipe("sharded_joint_decay", source=True)
calls = []


def experiment(index, source):
    def load():
        calls.append(index)
        return jnp.asarray(decay_values(rate=0.6, source=source))

    measurement = mean_measurement(load, likelihood=inf.Gaussian(scale=0.02))
    return inf.Experiment(
        decay.model, solver(), (0.0, 1.0), decay.initial_state, (measurement,), id=f"e{index}"
    )


def problem(layout):
    experiments = tuple(experiment(i, s) for i, s in enumerate((0.0, 0.0, 0.1, 0.1)))
    return inf.InferenceProblem(
        experiments,
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": ("A", "A", "B", "B")},
        layout=layout,
    )


sharded = problem(ds.Layout(mesh=ds.global_mesh(("data",))))
loads = sorted(calls)
local = problem(None)
report = {"is_sharded": [sharded.is_sharded, local.is_sharded], "loads": loads}
reverse = [inf.optsolve(p, inf.BFGS(max_steps=40), diagnostics=False) for p in (sharded, local)]
report["reverse"] = [[float(v) for v in ds.distributed.local_value(fit.params)] for fit in reverse]
joint, local_joint = sharded.diagnostic_problem, local.diagnostic_problem
report["auto"] = [
    joint.resolved_hessian_composition("auto"),
    local_joint.resolved_hessian_composition("auto"),
]
optimizer = inf.LevenbergMarquardt(max_steps=30)
fits = [inf.optsolve(p, optimizer, diagnostics=False) for p in (sharded, local)]
report["lm"] = [
    [float(v) for v in ds.distributed.local_value(fit.params)] for fit in fits
]
report["lm_status"] = [fit.status.value for fit in fits]
bfgs = inf.optsolve(sharded, inf.BFGS(ad="forward", max_steps=40), diagnostics=False)
report["bfgs"] = [float(v) for v in ds.distributed.local_value(bfgs.params)]
report["bfgs_status"] = bfgs.status.value


def statistics(evaluate, point):
    cost, gradient, matrix, rows, failed = (ds.distributed.local_value(v) for v in evaluate(point))
    return [np.ravel(cost), np.ravel(gradient), np.ravel(matrix)], [int(rows), int(failed)]


def largest_relative(got, expected):
    return [float(np.max(np.abs(g - e)) / np.max(np.abs(e))) for g, e in zip(got, expected)]


class Compiles(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        self.count += "Compiling" in record.getMessage()


def mean_of(state):
    return jnp.mean(state)


def one_kernel(balance_steps):
    experiments = tuple(
        inf.Experiment(
            decay.model,
            solver(),
            (0.0, 1.0),
            decay.initial_state,
            (
                inf.Measurement(
                    "mean",
                    times=jnp.asarray(TIMES),
                    values=jnp.asarray(decay_values(rate=0.6, source=source)),
                    observe=mean_of,
                    likelihood=inf.Gaussian(scale=0.02),
                ),
            ),
            id=f"k{index}",
        )
        for index, source in enumerate((0.0, 0.0, 0.1, 0.1))
    )
    return inf.InferenceProblem(
        experiments,
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": ("A", "A", "B", "B")},
        layout=ds.Layout(mesh=ds.global_mesh(("data",)), balance_steps=balance_steps),
    )


unbalanced, balanced = one_kernel(False), one_kernel(True)
fits = [inf.optsolve(p, optimizer, diagnostics=False) for p in (unbalanced, balanced)]
joint = balanced.diagnostic_problem
balance, point = joint.step_balance, jnp.asarray(joint.x0)
before = statistics(joint.evaluate_lsq_fn, point)[0]
fitted = [balance.reassignments, [int(k) for k in balance.last_steps]]
balance.reorder([0, 3, 2, 1])
compiles = Compiles()
logging.getLogger("jax").addHandler(compiles)
with jax.log_compiles(True):
    after = statistics(joint.evaluate_lsq_fn, point)[0]
logging.getLogger("jax").removeHandler(compiles)
report["balanced"] = {
    "fits": [[float(v) for v in ds.distributed.local_value(fit.params)] for fit in fits],
    "status": [fit.status.value for fit in fits],
    "planner": [joint.is_homogeneous, balance.enabled, balance.capacity, balance.evaluations],
    "reassignments": [fitted[0], balance.reassignments],
    "steps": [fitted[1], [int(k) for k in balance.last_steps]],
    "compiles": compiles.count,
    "reordered": largest_relative(after, before),
}

mixed = decay_recipe("sharded_joint_mixed", source=True)


def heterogeneous(layout):
    windowed = inf.Multishoot(inf.Windows.uniform(2))
    return inf.InferenceProblem(
        (
            decay_experiment(mixed, "windowed", source=0.0, trajectory=windowed),
            decay_experiment(mixed, "single", source=0.1),
        ),
        parameters=ds.ParameterSet(shared=(mixed.rate,), varying={"cell": (mixed.source,)}),
        clusters={"cell": ("A", "B")},
        layout=layout,
    )


placed, unplaced = (
    heterogeneous(layout).diagnostic_problem
    for layout in (ds.Layout(mesh=ds.global_mesh(("data",))), None)
)
point = jnp.asarray(unplaced.x0)
residual, jacobian = (np.asarray(v) for v in unplaced.stacked_rj_fn(point))
got, counts = statistics(placed.evaluate_lsq_fn, point)
expected = [
    np.ravel(0.5 * residual @ residual),
    np.ravel(jacobian.T @ residual),
    np.ravel(jacobian.T @ jacobian),
]
report["heterogeneous"] = {
    "placement": [placed.is_sharded, placed.is_homogeneous],
    "counts": [counts, [int(residual.size), 0]],
    "statistics": largest_relative(got, expected),
}
print("REPORT " + json.dumps(report))
"""


@pytest.fixture(scope="module")
def two_devices():
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
        timeout=420,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


def test_reverse_mode_on_a_sharded_joint_equals_the_mesh_free_joint(two_devices):
    """Reverse-mode BFGS sharded ends where the mesh-free joint ends; auto resolves
    forward-over-reverse on both."""
    assert two_devices["is_sharded"] == [True, False]
    sharded, local = two_devices["reverse"]
    assert sharded == local
    assert two_devices["auto"] == ["forward-over-reverse", "forward-over-reverse"]


def test_forward_fits_of_a_sharded_joint_equal_the_mesh_free_joint(two_devices):
    """LevenbergMarquardt sharded ends where the mesh-free joint ends; forward BFGS joins it."""
    sharded, local = two_devices["lm"]
    assert two_devices["lm_status"][0] == two_devices["lm_status"][1] == "gradient_tolerance"
    # measured 0.0 between the LevenbergMarquardt fits; BFGS against them 2.0e-9 relative on the
    # rate, 1.1e-8 on source B and 1.0e-9 absolute on the near-zero source A, CPU VM 2026-10-22
    np.testing.assert_allclose(sharded, local, rtol=1e-8)
    np.testing.assert_allclose(two_devices["bfgs"], local, rtol=1e-6, atol=1e-8)


def test_deferred_measurements_of_a_sharded_joint_load_once_each(two_devices):
    """On one process every experiment's callable values load exactly once."""
    assert two_devices["loads"] == [0, 1, 2, 3]


# measured 1.5e-16, 0.0 and 0.0 relative on the cost, J^T r and J^T J after the reorder, two
# forced host devices, arm64 Mac 2026-10-23, pytest tests/multi_device/test_sharded_joint.py
_REORDERED = 1e-12


def test_step_balancing_fits_as_the_unbalanced_joint_and_reorders_without_a_recompile(
    two_devices,
):
    """LevenbergMarquardt on a four-experiment joint of one kernel with balance_steps=True ends
    where the same joint without step balancing ends: the planner runs on two lanes of two
    experiments, reads every experiment's step count after each evaluation and reassigns none.
    A reorder that moves experiments between the lanes counts one reassignment, compiles nothing
    and leaves the step counts and the statistics as they were."""
    balanced = two_devices["balanced"]
    unbalanced_fit, balanced_fit = balanced["fits"]
    assert balanced_fit == unbalanced_fit
    assert balanced["status"][0] == balanced["status"][1]
    homogeneous, enabled, capacity, evaluations = balanced["planner"]
    assert (homogeneous, enabled, capacity) == (True, True, 2) and evaluations > 1
    assert balanced["reassignments"] == [0, 1] and balanced["compiles"] == 0
    steps = balanced["steps"]
    assert steps[0] == steps[1] and len(steps[0]) == 4 and min(steps[0]) >= 1
    assert max(balanced["reordered"]) <= _REORDERED


# measured 3.0e-16, 8.2e-17 and 3.2e-17 relative on the cost, J^T r and J^T J, two forced host
# devices, arm64 Mac 2026-10-23, pytest tests/multi_device/test_sharded_joint.py
_GROUPED = 1e-10


def test_a_heterogeneous_sharded_joint_reduces_the_statistics_of_the_mesh_free_joint(two_devices):
    """A joint of a windowed and a single-shoot experiment over the mesh shards as two
    structural groups whose summed cost, J^T r and J^T J equal those of the same joint without
    a mesh, over every residual row, with no failed solve."""
    grouped = two_devices["heterogeneous"]
    assert grouped["placement"] == [True, False]
    assert grouped["counts"][0] == grouped["counts"][1]
    assert max(grouped["statistics"]) <= _GROUPED


# The bound admits a reordered cross-device sum moving the end within BFGS's step tolerance
# (1e-8 relative); the GPU measurement is owed to the GPU gate. Measured 0.0 (equal end points)
# on two forced host devices, CPU VM 2026-10-22, XLA_FLAGS=--xla_force_host_platform_device_count=2
# pytest tests/multi_device/test_sharded_joint.py::test_reverse_mode_on_the_visible_devices_ends_where_the_mesh_free_joint_ends
_VISIBLE = {"rtol": 1e-6, "atol": 1e-8}


@pytest.mark.requires_two_devices
def test_reverse_mode_on_the_visible_devices_ends_where_the_mesh_free_joint_ends():
    """Reverse-mode BFGS on a joint of two decay experiments sharded over the visible devices of
    this process, on their own platform, ends where the mesh-free joint ends, with the same
    verdict."""
    study = decay_study(decay_recipe("sharded_joint_visible", source=True))
    owners = {"cell": study.clusters["cell"][:2]}
    sharded, local = (
        inf.InferenceProblem(
            study.experiments[:2], parameters=study.parameters, clusters=owners, layout=layout
        )
        for layout in (ds.Layout(mesh=ds.global_mesh(("data",))), None)
    )
    fits = [inf.optsolve(p, inf.BFGS(max_steps=40), diagnostics=False) for p in (sharded, local)]
    assert sharded.is_sharded and fits[0].status == fits[1].status
    ends = [ds.distributed.local_value(fit.params) for fit in fits]
    np.testing.assert_allclose(ends[0], ends[1], **_VISIBLE)


def test_a_global_mesh_reduces_process_local_data_to_a_replicated_value():
    """from_local_data, a psum under shard_map and local_value give the global sum."""
    mesh = ds.global_mesh(("data",))
    assert mesh.axis_types == (AxisType.Auto,)
    local = np.full((jax.local_device_count(),), jax.process_index() + 1.0)
    batch = ds.distributed.from_local_data(local, mesh, P("data"))
    reduced = jax.shard_map(
        lambda x: jax.lax.psum(jnp.sum(x), "data"),
        mesh=mesh,
        in_specs=P("data"),
        out_specs=P(),
    )(batch)
    expected = jax.local_device_count() * sum(range(1, jax.process_count() + 1))
    assert float(ds.distributed.local_value(reduced)) == expected
    np.testing.assert_array_equal(
        ds.distributed.local_value(jax.random.key_data(ds.distributed.key(17))), [0, 17]
    )


@pytest.mark.requires_two_processes
def test_local_value_refuses_an_array_partitioned_across_processes():
    """local_value of an array sharded over two processes names the partitioning."""
    mesh = ds.global_mesh(("data",))
    local = np.ones((jax.local_device_count(),))
    batch = ds.distributed.from_local_data(local, mesh, P("data"))
    with pytest.raises(ValueError, match="partitioned"):
        ds.distributed.local_value(batch)


@pytest.mark.requires_two_devices
def test_submeshes_partition_the_devices_and_refuse_bad_groupings():
    """mesh_groups gives one owned submesh per device; sizes must sum to the device count."""
    mesh = ds.global_mesh(("data",))
    groups = ds.mesh_groups(mesh, (1,) * jax.device_count())
    assert {device for group in groups for device in group.devices.flat} == set(jax.devices())
    assert sum(ds.distributed.owns(group) for group in groups) == jax.local_device_count()
    with pytest.raises(ValueError, match="summing"):
        ds.mesh_groups(mesh, (jax.device_count() + 1,))


@pytest.mark.requires_two_devices
def test_a_robust_gradient_over_two_devices_reduces_with_an_all_reduce():
    """Over two devices the robust gradient sums with one all-reduce and matches the mean."""
    samples = jnp.asarray([[0.2, 0.5], [0.4, 0.7], [0.8, 1.1], [1.3, 1.6]])
    point = jnp.asarray([0.9, 1.2])

    def criterion(sample, at):
        return jnp.sum((sample * at - 1.0) ** 2)

    devices = jax.devices()[:2]
    value, gradient = design.sharded_robust_gradient(criterion, point, samples, devices=devices)
    expected = jax.grad(lambda at: design.robust_criterion(criterion, samples, at))(point)
    hlo = (
        jax.jit(lambda at: design.sharded_robust_gradient(criterion, at, samples, devices=devices))
        .lower(point)
        .compile()
        .as_text()
    )
    assert "all-reduce" in hlo and "all-gather" not in hlo
    # measured 3.9e-16 relative on two forced host devices, CPU VM 2026-10-22,
    # XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest tests/multi_device/test_sharded_joint.py::test_a_robust_gradient_over_two_devices_reduces_with_an_all_reduce
    np.testing.assert_allclose(ds.distributed.local_value(gradient), expected, rtol=1e-12)


@pytest.mark.requires_two_processes
def test_campaign_submeshes_write_the_ledger_on_the_primary_process_only(tmp_path):
    """checkpointed_map over per-device submeshes runs every task and writes on rank 0 only."""
    mesh = ds.global_mesh(("data",))
    groups = ds.mesh_groups(mesh, (1,) * jax.device_count())
    path = tmp_path / f"rank-{jax.process_index()}" / "ledger.json"
    tasks = tuple({"task_id": str(index)} for index in range(4))
    result = design.checkpointed_map(
        lambda task: {"index": int(task["task_id"])},
        tasks,
        ledger_path=path,
        execution="processes",
        device_groups=groups,
    )
    assert result.ok and path.exists() == ds.distributed.is_primary()


@pytest.mark.requires_two_processes
def test_a_two_rank_fit_takes_the_same_path_on_every_rank():
    """LevenbergMarquardt on a sharded joint and TrustRegion on a lowered study end alike on
    both ranks."""
    from jax.experimental import multihost_utils

    decay = decay_recipe("sharded_joint_ranks", source=True)
    study = decay_study(decay)
    sharded = inf.InferenceProblem(
        study.experiments,
        parameters=study.parameters,
        clusters=study.clusters,
        layout=ds.Layout(mesh=ds.global_mesh(("data",))),
    )
    fits = (
        inf.optsolve(sharded, inf.LevenbergMarquardt(max_steps=30), diagnostics=False),
        inf.optsolve(ds.lower(study), inf.TrustRegion(max_steps=30), diagnostics=False),
    )
    for fit in fits:
        point = jnp.asarray(ds.distributed.local_value(fit.params), dtype=jnp.float64)
        everyone = np.asarray(multihost_utils.process_allgather(point, tiled=False))
        assert fit.converged
        np.testing.assert_array_equal(everyone, np.broadcast_to(everyone[0], everyone.shape))


def _free_port(port):
    """Whether nothing listens on a local TCP port, so a coordinator may serve it."""
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("inside_slurm", [True, False], ids=["inside-slurm", "outside-slurm"])
def test_every_rank_of_an_mpirun_launch_joins_one_world(inside_slurm):
    """Inside a Slurm job, mpirun's ranks on other nodes see a one-task step and still join the
    world that JAX's launcher detection names; outside Slurm, DIFFSOUP_COORDINATOR names it."""
    offset = next(k for k in range(4096) if _free_port(61440 + k))  # a job id names this port
    reachable, unreachable = f"127.0.0.1:{61440 + offset}", "192.0.2.1:62213"
    detected, given = (reachable, unreachable) if inside_slurm else (unreachable, reachable)
    launcher = ("SLURM_", "OMPI_", "PMI", "DIFFSOUP_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher)}
    env = {k: v for k, v in env.items() if "proxy" not in k.lower()}
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", DIFFSOUP_COORDINATOR=given)
    env.update(OMPI_COMM_WORLD_SIZE="2", OMPI_MCA_orte_hnp_uri=f"{offset << 12}.0;tcp://{detected}")
    script = (
        "import jax, diff_soup as ds; ds.distributed.initialize(initialization_timeout=60); "
        "print('WORLD', jax.process_index(), jax.process_count())"
    )
    ranks = []
    for rank in range(2):  # rank 1 runs under the remote daemon's one-task Slurm step
        slurm = {"SLURM_JOB_ID": "1", "SLURM_NTASKS": "1" if rank else "2"} if inside_slurm else {}
        ids = {"OMPI_COMM_WORLD_RANK": str(rank), "OMPI_COMM_WORLD_LOCAL_RANK": str(rank)}
        ranks.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=_ROOT,
                env={**env, **ids, **slurm},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    outputs = [process.communicate(timeout=240) for process in ranks]
    errors = "\n".join(err[-2000:] for _, err in outputs)
    assert [process.returncode for process in ranks] == [0, 0], errors
    worlds = [[line for line in out.splitlines() if line.startswith("WORLD")] for out, _ in outputs]
    assert worlds == [["WORLD 0 2"], ["WORLD 1 2"]]
