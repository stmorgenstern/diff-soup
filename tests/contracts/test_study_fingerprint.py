"""Study fingerprints and lanes: device-free identity, memo safety, separability, effect-free
lanes, lanes placed over the devices of a mesh, and the backends such a placement refuses."""

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
from diff_soup import inference as inf
from tests.support.inference import (
    TIMES,
    decay_experiment,
    decay_recipe,
    decay_study,
    decay_values,
    mean_measurement,
    solver,
)

_ROOT = Path(__file__).resolve().parents[2]

#: Run on two forced host CPU devices: the decay study lowered with its lanes placed over a
#: two-device mesh and without a mesh, fitted alike; then the study with a KLU solver, placed
#: on that mesh by ds.lower and by the sharded joint.
_PLACED_WORKER = r"""
import json

import jax
import numpy as np

import diff_soup as ds
from diff_soup import inference as inf
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


@pytest.fixture(scope="module")
def two_devices():
    """The placed-study worker's report from one subprocess on two forced host devices."""
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
        [sys.executable, "-c", _PLACED_WORKER],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


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
