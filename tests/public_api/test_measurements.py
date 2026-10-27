"""Measurements: Measurement and Experiment, likelihood families, operators, windows, float32."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from tests.support.inference import (
    TIMES,
    decay_experiment,
    decay_recipe,
    decay_values,
    mean_measurement,
    small_fit,
)


def test_a_measurement_validates_its_declaration_and_loads_callable_values_once():
    """weight= and likelihood= are exclusive, a weight is positive and broadcasts, and
    callable values are read once, at the first lowering, against their declared shape."""
    plain = inf.Measurement("u", [0.0, 1.0], jnp.array([1.0, 2.0]))
    assert plain.likelihood is None and plain.weight is None and plain.shape == (2,)
    with pytest.raises(TypeError, match="weight= or likelihood="):
        inf.Measurement("u", [0.0], jnp.array([1.0]), weight=2.0, likelihood=inf.Gaussian(1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        inf.Measurement("u", [0.0], jnp.array([1.0]), weight=0.0)
    with pytest.raises(ValueError, match="does not broadcast"):
        inf.Measurement("u", [0.0, 1.0], jnp.zeros((2, 3)), weight=jnp.ones(2))
    calls = []

    def values():
        calls.append(None)
        return jnp.asarray(decay_values())

    decay = decay_recipe("measurements_deferred")
    deferred = mean_measurement(values, likelihood=inf.Gaussian(scale=0.02))
    experiment = decay_experiment(decay)
    experiment = inf.Experiment(
        experiment.model,
        experiment.solver,
        experiment.t_span,
        experiment.initial_state,
        (deferred,),
        id="decay",
    )
    assert deferred.deferred and calls == []
    inf.InferenceProblem(experiment, parameters=decay.rate)
    inf.InferenceProblem(experiment, parameters=decay.rate)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="5 leading frames"):
        inf.Measurement("mean", TIMES, lambda: jnp.ones(5), shape=(4,))


_OBSERVED = jnp.array([0.5, 1.2, 2.0])
_PREDICTIONS = (jnp.array([0.7, 1.0, 2.2]), jnp.array([0.4, 1.5, 1.7]))


@pytest.mark.parametrize(
    ("likelihood", "observed"),
    [
        (inf.Gaussian(scale=0.1), _OBSERVED),
        (inf.StudentT(df=4.0, scale=0.1), _OBSERVED),
        (inf.LogNormal(scale=0.1), _OBSERVED),
        (inf.Poisson(exposure=3.0), jnp.array([1.0, 3.0, 6.0])),
        (
            inf.CorrelatedGaussian(
                cholesky=jnp.array([[0.1, 0.0, 0.0], [0.05, 0.2, 0.0], [0.0, 0.02, 0.3]])
            ),
            _OBSERVED,
        ),
    ],
    ids=("gaussian", "student-t", "log-normal", "poisson", "correlated-gaussian"),
)
def test_least_squares_rows_carry_the_negative_log_likelihood(likelihood, observed):
    """-log_prob minus half the squared rows does not depend on the prediction."""
    gaps = []
    for prediction in _PREDICTIONS:
        rows = likelihood.least_squares_rows(observed, prediction)
        gaps.append(float(-likelihood.log_prob(observed, prediction) - 0.5 * jnp.sum(rows**2)))
    # measured 2.2e-16 relative (Student t), 2.1e-16 (Poisson), 0.0 for the rest, CPU VM 2026-10-22,
    # pytest tests/public_api/test_measurements.py
    np.testing.assert_allclose(gaps[0], gaps[1], rtol=1e-12)


def test_multiplicative_gaussian_rows_are_relative_errors():
    """MultiplicativeGaussian rows are (observed - predicted) / (scale * predicted)."""
    predicted = _PREDICTIONS[0]
    rows = inf.MultiplicativeGaussian(scale=0.1).least_squares_rows(_OBSERVED, predicted)
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/public_api/test_measurements.py
    np.testing.assert_allclose(rows, (_OBSERVED - predicted) / (0.1 * predicted), rtol=1e-12)


def test_observation_operators_gather_scatter_and_average_into_pixels():
    """Nodal extract/inject gather and scatter by index; the pixel operator averages an
    affine mesh field to its pixel-centre values."""
    source = jnp.array([10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(ds.NodalExtractOp((0, 2)).apply(source, 2), [10.0, 12.0])
    np.testing.assert_array_equal(
        ds.NodalInjectOp((1, 3)).apply(jnp.array([5.0, 6.0]), 4), [0.0, 5.0, 0.0, 6.0]
    )
    rows, cols = ds.NodalInjectOp((1, 3)).sparsity(2, 4)
    assert list(zip(rows, cols)) == [(1, 0), (3, 1)]
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    operator = inf.data.PixelAveragingOperator.from_mesh_and_image(
        points, cells, image_shape=(2, 2), pixel_origin=(0.25, 0.25), pixel_size=0.5
    )
    pixels = operator.predict(jnp.asarray(points[:, 0] + 2.0 * points[:, 1]), None, None)
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/public_api/test_measurements.py
    np.testing.assert_allclose(pixels, [0.75, 1.25, 1.75, 2.25], rtol=1e-12)


@pytest.fixture(scope="module")
def windowed():
    decay = decay_recipe("measurements_windows", nodes=2)
    experiment = decay_experiment(decay, likelihood=inf.Gaussian(scale=1e-3))
    trajectory = inf.Multishoot(inf.Windows.at(TIMES), inf.Continuity(weight=1.0e6))
    return decay, experiment, trajectory


def test_multishoot_continuity_rows_and_prepared_window_sweeps(windowed):
    """Continuity(weight=1e6) adds one finite row per interior boundary and state entry;
    prepare_windows keys a decision-variable sweep and returns None for a single shoot."""
    decay, experiment, trajectory = windowed
    problem = inf.InferenceProblem(experiment, parameters=decay.rate, trajectory=trajectory)
    rows = problem.stacked_r_fn(jnp.asarray(problem.x0))
    assert rows.ndim == 1 and rows.size == 5 + 3 * 2 and bool(jnp.all(jnp.isfinite(rows)))
    key = inf.prepare_windows(
        experiment, inf.Multishoot(inf.Windows.uniform(2)), parameters=(decay.rate,)
    )
    assert isinstance(key, str) and key
    assert inf.prepare_windows(experiment, parameters=(decay.rate,)) is None


def test_sensitivity_runs_label_their_coordinates_by_handle_name():
    """A sensitivity run names its analyzed and design coordinates after their handles."""
    decay = decay_recipe("measurements_labels", source=True)
    run = inf.sensitivity_solve(
        decay.model,
        (0.0, 1.0),
        decay.initial_state,
        wrt=(decay.rate,),
        design=(decay.source,),
        solver=ds.JNDF.stock(linear="dense"),
        saveat=ds.SaveAt(ts=jnp.array([0.0, 1.0])),
    )
    assert run.names == ("rate",) and run.design_names == ("source",)
    assert run.leaf_paths[0].endswith("rate") and run.design_leaf_paths[0].endswith("source")


def test_the_facade_experiment_is_the_only_public_experiment_class():
    """Every public name ending in Experiment is inf.Experiment itself."""
    namespaces = (ds, inf, inf.design, inf.approx)
    found = {
        getattr(namespace, name)
        for namespace in namespaces
        for name in namespace.__all__
        if name.endswith("Experiment")
    }
    assert found == {inf.Experiment}


@pytest.fixture(scope="module")
def precisions():
    """The same MAP fit and its Laplace posterior in float64 and float32."""
    results = {}
    for name, precision in (("float64", None), ("float32", ds.Precision(compute_dtype="float32"))):
        decay = decay_recipe(
            "measurements_precision",
            precision=precision,
            rate_prior=inf.NormalPrior(mean=0.6, scale=0.25),
        )
        # solver tolerances float32 resolves (measured: both fits converge, CPU VM 2026-10-22)
        experiment = decay_experiment(decay, rtol=1e-5, atol=1e-6)
        problem = inf.InferenceProblem(experiment, parameters=decay.rate, objective=inf.MAP())
        fit = small_fit(problem, inf.LevenbergMarquardt(max_steps=40))
        results[name] = (problem, fit, inf.laplace(problem, fit, curvature="expected"))
    return results


def test_a_float32_problem_evaluates_in_float32_and_agrees_with_float64(precisions):
    """The float32 problem's start, likelihood, rows and Jacobian are float32, and its fit and
    posterior agree with the float64 ones."""
    problem, _, _ = precisions["float32"]
    x0 = jnp.asarray(problem.x0)
    rows, jacobian = problem.data_residual_jacobian(x0)
    assert x0.dtype == np.float32 and problem.log_likelihood(x0).dtype == np.float32
    assert rows.dtype == jacobian.dtype == np.float32
    _, fit64, post64 = precisions["float64"]
    _, fit32, post32 = precisions["float32"]
    # measured 7.3e-6 on the point and 5.2e-6 on the standard deviation, CPU VM 2026-10-22,
    # pytest tests/public_api/test_measurements.py
    np.testing.assert_allclose(fit32.params, fit64.params, rtol=1e-4)
    np.testing.assert_allclose(post32.marginal_sd(), post64.marginal_sd(), rtol=1e-3)
