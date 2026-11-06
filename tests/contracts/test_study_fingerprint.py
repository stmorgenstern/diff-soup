"""Study fingerprints and lanes: device-free identity, memo safety, separability, effect-free
lanes, lanes placed over the devices of a mesh, the backends such a placement refuses, and the
platform, backend and factor sites a lane's default device selects."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.optimization import inference as inf
from diff_soup.utils import array_platform
from tests.support.inference import (
    TIMES,
    decay_experiment,
    decay_recipe,
    decay_study,
    decay_values,
    mean_measurement,
    solver,
)
from tests.support.problems import SparseChainDAE

_ROOT = Path(__file__).resolve().parents[2]

#: Run on two forced host CPU devices: the decay study lowered with its lanes placed over a
#: two-device mesh and without a mesh, fitted alike; then the study with a KLU solver, placed
#: on that mesh by ds.lower and by the sharded joint.
_PLACED_WORKER = r"""
import json

import jax
import numpy as np

import diff_soup as ds
from diff_soup.optimization import inference as inf
from tests.support.inference import decay_recipe, decay_study

assert jax.device_count() == 2, jax.devices()
mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
study = decay_study(decay_recipe("fingerprint_two_devices", source=True))
placed = ds.lower(study, layout=ds.Layout(mesh=mesh))
report = {
    "devices": [placed.plan().devices, ds.lower(study).plan().devices],
    "rounded": [
        bucket.chunk["value"].width
        for bucket in ds.lower(study, layout=ds.Layout(mesh=mesh), chunk=3).plan().buckets
    ],
    "programs": [[r.bucket, r.mode, r.chunk] for r in placed.prepare().records],
}
fits = [
    inf.optsolve(lowered, inf.TrustRegion(max_steps=30), diagnostics=False)
    for lowered in (placed, ds.lower(study))
]
report["points"] = [np.asarray(fit.params, dtype=np.float64).tolist() for fit in fits]
report["again"] = sorted({r.cache for r in placed.prepare().records})
try:
    ds.Layout(mesh=mesh, processes=2)
    report["refusal"] = None
except ds.ConfigurationError as error:
    report["refusal"] = str(error)


def refusal(build):
    try:
        build()
    except ds.ConfigurationError as error:
        return str(error)
    return None


klu = decay_study(decay_recipe("fingerprint_two_devices_klu", source=True), linear="klu")
report["native"] = [
    refusal(lambda: ds.lower(klu, layout=ds.Layout(mesh=mesh))),
    refusal(
        lambda: inf.InferenceProblem(
            klu.experiments,
            parameters=klu.parameters,
            clusters=klu.clusters,
            layout=ds.Layout(mesh=mesh),
        )
    ),
]
print("REPORT " + json.dumps(report))
"""


def _two_device_report(worker):
    """The report ``worker`` prints from one subprocess on two forced host devices."""
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
        [sys.executable, "-c", worker],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


@pytest.fixture(scope="module")
def two_devices():
    """The placed-study worker's report from one subprocess on two forced host devices."""
    return _two_device_report(_PLACED_WORKER)


def test_a_study_placed_over_two_devices_fits_to_the_point_of_one_device(two_devices):
    """TrustRegion on lanes split over a two-device mesh ends bitwise where one device ends."""
    placed, local = two_devices["points"]
    assert two_devices["devices"] == [2, 1]
    assert placed == local


def test_a_placed_study_compiles_one_program_per_mode_and_reuses_it_after_a_fit(two_devices):
    """One program per bucket and mode serves both devices; chunks round up to a multiple of
    two; prepare() after a fit, whose steps reorder the units, compiles nothing."""
    assert two_devices["programs"] == [[0, "value", 4], [0, "blocks", 4]]
    assert two_devices["rounded"] == [4]
    assert two_devices["again"] == ["reused"]


def test_a_layout_takes_one_data_placement_at_a_time(two_devices):
    """A two-device data axis with processes=2 is refused naming both placements."""
    assert "2 devices" in two_devices["refusal"] and "2 processes" in two_devices["refusal"]


def test_a_two_device_placement_refuses_a_backend_with_native_factor_storage(two_devices):
    """A KLU solver on a two-device data axis is refused by ds.lower and by the sharded joint,
    naming the backend as diff_soup.linear exports it, Dense and a one-device layout. The joint
    refuses while it validates its layout, before any solve is traced or native library loaded,
    so the row needs no KLU build."""
    names = ("KLU keeps", "2 devices", "Dense", "a layout whose data axis has one device")
    texts = two_devices["native"]
    assert [all(name in (text or "") for name in names) for text in texts] == [True, True]


#: Run on two forced host CPU devices: one jitted KLU factor and solve, run under each device as
#: the default device; the slots each run takes and the two solutions.
_SITES_WORKER = r"""
import json

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import KLU, SolveContext, SparseTopology, analyze, factor, solve

index = np.arange(12)
pattern = SparseTopology(
    rows=np.concatenate([index, index[1:], index[:-1]]),
    cols=np.concatenate([index, index[:-1], index[1:]]),
    n_dofs=12,
)
values = jnp.asarray(np.concatenate([np.full(12, 4.0), np.full(22, -1.0)]))
backend, slots, solutions = KLU(), [], []
with SolveContext() as ctx:
    analysis = analyze(pattern, backend=backend)
    program = jax.jit(lambda v: solve(factor(analysis, v, ctx=ctx), jnp.linspace(-1.0, 2.0, 12))[0])
    for device in jax.devices():
        with jax.default_device(device):
            before = backend.native_stats()["live_slots"]
            solutions.append(np.asarray(program(values)).tolist())
            slots.append(backend.native_stats()["live_slots"] - before)
print("REPORT " + json.dumps({"slots": slots, "same": solutions[0] == solutions[1]}))
"""


@pytest.mark.requires_klu
def test_one_program_under_two_default_devices_factors_at_one_site_per_device():
    """A jitted KLU factor and solve run under ``jax.default_device`` set to each of two host
    devices takes a factor slot in each run, one site per device rather than one shared, and
    both runs return the same solution."""
    assert _two_device_report(_SITES_WORKER) == {"slots": [1, 1], "same": True}


@pytest.mark.requires_cuda
def test_a_program_staged_under_a_cpu_default_device_reads_the_cpu_platform():
    """Under ``jax.default_device`` set to a CPU device, ``array_platform`` names ``"cpu"`` for an
    array ``jax.jit`` stages and for no array at all, whatever backend the process defaults to."""
    staged = []
    with jax.default_device(jax.devices("cpu")[0]):
        jax.jit(lambda x: staged.append(array_platform(x)) or x)(jnp.ones(2))
        staged.append(array_platform())
    assert staged == ["cpu", "cpu"]


@pytest.mark.requires_cuda
@pytest.mark.requires_klu
def test_an_auto_solve_under_a_cpu_default_device_runs_klu_on_that_device():
    """In a process whose default backend is a GPU, a jitted ``linear="auto"`` JIDA solve of a
    25-unknown DAE that initializes consistently, run under ``jax.default_device`` set to a CPU
    device, runs KLU on that device and stores bitwise what the same solve with ``linear="klu"``
    stores."""
    chain, cpu = SparseChainDAE(n=12), jax.devices("cpu")[0]

    def solve(linear):
        solver = ds.JIDA.stock(linear=linear, **{"rtol": 1e-8, "atol": 1e-10})

        def run(rate):
            solution = ds.pdesolve(
                chain.problem,
                solver,
                (0.0, 0.5),
                chain.initial_state(),
                theta=rate,
                saveat=ds.SaveAt(t1=True),
                consistent_ic=True,
            )
            return solution.ys, solution.lambdas, solution.stats

        with jax.default_device(cpu):
            return jax.jit(run)(jnp.asarray(0.7))

    auto, klu = solve("auto"), solve("klu")
    assert (auto[2].linear_backend, auto[0].devices()) == ("klu", {cpu})
    for got, expected in zip(auto[:2], klu[:2]):
        np.testing.assert_array_equal(got, expected)


def _placed_study(devices):
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    decay = decay_recipe("fingerprint_placed", layout=ds.Layout(mesh=mesh))
    return inf.Study(decay_experiment(decay), parameters=ds.ParameterSet(shared=(decay.rate,)))


def test_a_study_fingerprint_records_a_mesh_by_its_axes():
    """Two studies built on the same one-device mesh share a fingerprint; no mesh differs."""
    first = _placed_study(jax.devices()[:1])
    second = _placed_study(jax.devices()[:1])
    decay = decay_recipe("fingerprint_placed")
    unplaced = inf.Study(decay_experiment(decay), parameters=ds.ParameterSet(shared=(decay.rate,)))
    assert first.fingerprint == second.fingerprint != unplaced.fingerprint


@pytest.mark.requires_two_devices
def test_a_study_fingerprint_does_not_record_which_device_holds_the_mesh():
    """A one-device mesh on the second device fingerprints as one on the first."""
    devices = jax.devices()
    assert _placed_study(devices[1:2]).fingerprint == _placed_study(devices[:1]).fingerprint


def test_study_fingerprints_are_not_confused_by_freed_objects():
    """Forty studies built and freed in turn keep the fingerprints a fresh build gives them."""
    decay = decay_recipe("fingerprint_memo")

    def study(index):
        experiment = decay_experiment(decay, f"e{index}", rate=0.3 + 0.01 * index)
        return inf.Study(experiment, parameters=ds.ParameterSet(shared=(decay.rate,)))

    seen = []
    for index in range(40):
        built = study(index)
        seen.append(built.fingerprint)
        del built
    assert len(set(seen)) == 40
    assert seen == [study(index).fingerprint for index in range(40)]


def _windowed(likelihood):
    decay = decay_recipe("fingerprint_windows")
    experiment = inf.Experiment(
        decay.model,
        solver(),
        (0.0, 1.0),
        decay.initial_state,
        (
            inf.Measurement("u", TIMES, decay_values()[:, None], likelihood=inf.Gaussian(0.02)),
            mean_measurement(decay_values(), likelihood=likelihood, name="m"),
        ),
        id="windows",
    )
    return inf.InferenceProblem(
        experiment,
        parameters=decay.rate,
        trajectory=inf.Multishoot(inf.Windows.from_data(source="u")),
    )


def test_observed_windows_refuse_a_likelihood_that_couples_frames():
    """A frame-coupling CorrelatedGaussian is refused on observed-state windows; a factorized one is not."""
    coupling = np.eye(5) + 0.2 * (np.eye(5, k=1) + np.eye(5, k=-1))
    coupled = inf.CorrelatedGaussian(cholesky=jnp.asarray(0.02 * np.linalg.cholesky(coupling)))
    with pytest.raises(NotImplementedError, match="couples different frames"):
        _windowed(coupled)
    factorized = inf.CorrelatedGaussian(cholesky=jnp.diag(jnp.asarray([1.0, 2.0, 0.5, 1.5, 1.0])))
    assert _windowed(factorized).n_params == 1


@pytest.fixture(scope="module")
def lanes():
    """A three-owner MAP study lowered onto lanes."""
    decay = decay_recipe(
        "fingerprint_lanes",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    return ds.lower(decay_study(decay), objective=inf.MAP())


def test_the_lane_log_likelihood_carries_no_effect_batches_and_differentiates(lanes):
    """The lowered log likelihood traces without effects, maps over points and has a jacfwd."""
    x0 = jnp.asarray(lanes.x0)
    assert not jax.make_jaxpr(lanes.log_likelihood)(x0).effects
    points = jnp.stack([x0, 1.1 * x0])
    batched = jax.jit(jax.vmap(lanes.log_likelihood))(points)
    single = jnp.stack([lanes.log_likelihood(point) for point in points])
    # measured 0.0 relative difference between batched and single points, CPU VM 2026-10-22,
    # pytest tests/contracts/test_study_fingerprint.py
    np.testing.assert_allclose(batched, single, rtol=1e-9)
    gradient = jax.jit(jax.jacfwd(lanes.log_likelihood))(x0)
    assert gradient.shape == x0.shape and bool(jnp.all(jnp.isfinite(gradient)))
