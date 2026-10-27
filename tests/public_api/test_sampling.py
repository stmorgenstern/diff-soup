"""Sampling: sample in one call, chain placement and the backends it refuses, one derivative
pass, predictions."""

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

from diff_soup import inference as inf
from tests.support.inference import NOISE, decay_experiment, decay_recipe, small_fit
from tests.support.programs import primitive_names

#: The advanced posterior target ``inf.sample`` documents beside a fitted problem.
LogDensity = inf.statistics.LogDensity
_ROOT = Path(__file__).resolve().parents[2]


def _fitted(linear):
    decay = decay_recipe("sampling_decay", rate_prior=inf.NormalPrior(mean=0.5, scale=0.3))
    problem = inf.InferenceProblem(
        decay_experiment(decay, noise=NOISE, linear=linear),
        parameters=decay.rate,
        objective=inf.MAP(),
    )
    return decay, problem, small_fit(problem, inf.LevenbergMarquardt(max_steps=30))


@pytest.fixture(scope="module")
def sampled():
    decay, problem, fit = _fitted("dense")
    draws = inf.sample(
        problem, fit, kernel="rmh", key=jax.random.key(23), chains=2, draws=100, warmup=20
    )
    return decay, problem, fit, draws


def test_sample_in_one_call_draws_laplace_scaled_chains_around_the_fit(sampled):
    """sample(problem, fit) returns Draws matching the Laplace posterior, read by mcmc_report."""
    decay, problem, fit, draws = sampled
    assert isinstance(draws, inf.Draws) and draws[decay.rate].shape == (2, 100)
    assert draws.chain_map == "vmap"
    np.testing.assert_array_equal(draws[decay.rate], draws.values[..., 0])
    laplace_sd = float(inf.laplace(problem, fit)[decay.rate].sd)
    # measured spread 0.0296 against the Laplace 0.0236 (ratio 1.26) and the mean 0.0058 off
    # the fit, CPU VM 2026-10-22, pytest tests/public_api/test_sampling.py
    np.testing.assert_allclose(np.std(draws[decay.rate]), laplace_sd, rtol=0.35)
    np.testing.assert_allclose(np.mean(draws[decay.rate]), float(fit[decay.rate]), atol=0.015)
    report = inf.mcmc_report(draws, rhat_max=1.1, ess_floor=1.0)
    # measured acceptance 0.40 and R-hat 1.015, CPU VM 2026-10-22,
    # pytest tests/public_api/test_sampling.py
    assert 0.15 < report.acceptance_rate < 0.8
    assert float(np.max(report.rhat)) < 1.1


def test_predict_returns_bands_from_a_fit_and_from_draws(sampled):
    """predict gives a Prediction of PredictionBands; the predictive band adds the noise."""
    _, problem, fit, draws = sampled
    latent = inf.predict(problem, fit)
    noisy = inf.predict(problem, fit, kind="predictive")
    from_draws = inf.predict(problem, draws)
    assert isinstance(latent, inf.Prediction) and isinstance(latent["mean"], inf.PredictionBand)
    assert latent.names == ("mean",) and from_draws["mean"].mean.shape == (5,)
    # measured 1.4e-16 relative to 0.02**2, CPU VM 2026-10-22,
    # pytest tests/public_api/test_sampling.py
    np.testing.assert_allclose(noisy["mean"].sd ** 2 - latent["mean"].sd ** 2, 0.02**2, rtol=1e-9)


@pytest.mark.requires_cudss
def test_sample_runs_on_the_cudss_backend():
    """A problem solved with cuDSS samples in one call like the dense one."""
    decay, problem, fit = _fitted("cudss")
    draws = inf.sample(problem, fit, kernel="rmh", key=jax.random.key(23), chains=2, draws=20)
    assert draws[decay.rate].shape == (2, 20) and bool(jnp.all(jnp.isfinite(draws.values)))


def test_chain_map_modes_draw_the_same_chains():
    """Sequential and per-device chains reproduce the batched chains."""
    target = LogDensity.from_callable(
        lambda x: -0.5 * jnp.sum((x - 2.0) ** 2), init=jnp.array([2.0, -1.0])
    )
    shared = dict(key=jax.random.key(11), kernel="rmh", chains=2, draws=40, warmup=10)
    batched = inf.sample(target, **shared)
    sequential = inf.sample(target, chain_map="sequential", **shared)
    placed = inf.sample(target, chain_map="devices", **shared)
    assert (batched.chain_map, sequential.chain_map, placed.chain_map) == (
        "vmap",
        "sequential",
        "devices",
    )
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/public_api/test_sampling.py
    np.testing.assert_allclose(sequential.values, batched.values, rtol=1e-12)
    np.testing.assert_allclose(placed.values, batched.values, rtol=1e-12)


#: Run on two forced host CPU devices: per-device chains and a sharded robust gradient over a
#: KLU solve, both of which would share one native factor slot between the two devices.
_SHARED_STORAGE_WORKER = r"""
import json

import jax
import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference import design
from diff_soup.linear import KLU, SolveContext, SparseTopology, analyze, factor, solve
from tests.support.inference import NOISE, decay_experiment, decay_recipe, small_fit

assert jax.device_count() == 2, jax.devices()


def refusal(run):
    try:
        run()
    except ds.ConfigurationError as error:
        return str(error)
    return None


decay = decay_recipe("sampling_two_devices", rate_prior=inf.NormalPrior(mean=0.5, scale=0.3))
problem = inf.InferenceProblem(
    decay_experiment(decay, noise=NOISE, linear="klu"), parameters=decay.rate, objective=inf.MAP()
)
fit = small_fit(problem, inf.LevenbergMarquardt(max_steps=30))
options = dict(kernel="rmh", key=jax.random.key(0), chains=2, draws=4, warmup=2)
report = {"sample": refusal(lambda: inf.sample(problem, fit, chain_map="devices", **options))}
topology = SparseTopology(rows=np.arange(2), cols=np.arange(2), n_dofs=2)
analysis = analyze(topology, backend=KLU())
with SolveContext() as ctx:

    def criterion(sample, at):
        return jnp.sum(solve(factor(analysis, sample * at, ctx=ctx), jnp.ones(2))[0])

    samples = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    report["gradient"] = refusal(
        lambda: design.sharded_robust_gradient(criterion, jnp.ones(2), samples)
    )
print("REPORT " + json.dumps(report))
"""


@pytest.mark.requires_klu
def test_chains_and_robust_gradients_over_two_devices_refuse_native_factor_storage():
    """On two local devices, sample(chain_map="devices") and design.sharded_robust_gradient
    refuse a KLU solve, whose factors live in storage the devices would share, naming KLU and
    the remedies: a backend whose factors are arrays, or the sequential chain maps."""
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
        [sys.executable, "-c", _SHARED_STORAGE_WORKER],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    report = json.loads(line.removeprefix("REPORT "))
    assert "KLU keeps its factors in host storage" in report["sample"]
    assert "chain_map='vmap' or 'sequential'" in report["sample"]
    assert "2 devices of this process" in report["gradient"]
    assert "host storage" in report["gradient"] and "Dense" in report["gradient"]


@pytest.mark.parametrize("gradient", ("forward", "reverse"))
def test_value_and_gradient_come_from_one_derivative_pass(gradient):
    """A target's value and gradient trace the density once, not once each."""

    def log_density(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2) + jnp.sin(x[0] * x[1])

    point = jnp.array([0.4, -1.3, 2.2])
    target = LogDensity.from_callable(log_density, init=point, gradient=gradient)
    separate = primitive_names(lambda x: (log_density(x), jax.grad(log_density)(x)), point)
    assert primitive_names(target.value_and_grad, point).count("sin") == 1
    assert separate.count("sin") == 2
