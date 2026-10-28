"""InferenceProblem: equivalent declarations, layouts, parameter order, windows and resume."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from tests.support.inference import (
    TIMES,
    TRUE_RATE,
    decay_experiment,
    decay_recipe,
    decay_values,
    small_fit,
)

# measured 0.0 between the two declarations and 2.8e-6 off the truth, CPU VM 2026-10-22,
# pytest tests/contracts/test_inference_problem.py
_SAME_FIT = 1e-12


@pytest.fixture(scope="module")
def decay():
    return decay_recipe("inference_problem_decay")


@pytest.fixture(scope="module")
def plain(decay):
    """The one-experiment problem and its LevenbergMarquardt fit."""
    problem = inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)
    return problem, small_fit(problem, inf.LevenbergMarquardt(max_steps=20))


def test_an_experiment_and_a_study_of_it_declare_the_same_fit(decay, plain):
    """InferenceProblem of an Experiment and of a one-experiment Study fit to the same point."""
    problem, fit = plain
    study = inf.Study(decay_experiment(decay), parameters=ds.ParameterSet(shared=(decay.rate,)))
    declared = inf.InferenceProblem(study)
    other = small_fit(declared, inf.LevenbergMarquardt(max_steps=20))
    assert fit.objective is problem.objective and other.objective is declared.objective
    np.testing.assert_allclose(other.params, fit.params, atol=_SAME_FIT)
    assert float(fit.experiments["decay"][decay.rate]) == float(fit[decay.rate])


def test_clusters_and_a_mesh_free_layout_pass_through_to_the_joint(decay):
    """ParameterSet varying over clusters gives one coordinate per owner; the Layout is kept."""
    experiment = decay_experiment(decay)
    layout = ds.Layout()
    problem = inf.InferenceProblem(
        (experiment, dataclasses.replace(experiment, id="second")),
        parameters=ds.ParameterSet(varying={"particle": (decay.rate,)}),
        clusters={"particle": ("A", "B")},
        layout=layout,
    )
    assert problem.layout is layout
    assert problem.theta_names == ("rate@A", "rate@B")
    assert problem.placement == {"decay": (0, 0), "second": (0, 0)}


def test_the_flat_order_follows_the_declared_parameters():
    """parameters=(source, rate) orders x0, theta_names, the fit and the posterior that way."""
    decay = decay_recipe(
        "inference_problem_order",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.3),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    experiment = decay_experiment(decay, source=0.05)
    problem = inf.InferenceProblem(
        experiment, parameters=(decay.source, decay.rate), objective=inf.MAP()
    )
    assert problem.theta_names == ("source", "rate")
    np.testing.assert_array_equal(problem.x0, [0.0, 0.45])
    fit = small_fit(problem, inf.LevenbergMarquardt(max_steps=20))
    posterior = inf.laplace(problem, fit, curvature="expected")
    for column, handle in enumerate((decay.source, decay.rate)):
        assert float(fit[handle]) == float(fit.params[column])
        assert float(posterior[handle].mean) == float(fit.params[column])


def test_map_requires_a_prior_and_a_bare_model_is_not_an_experiment(decay):
    """MAP without a proper prior and a model in place of an Experiment are refused."""
    experiment = decay_experiment(decay)
    with pytest.raises(ValueError, match="proper prior"):
        inf.InferenceProblem(experiment, parameters=decay.rate, objective=inf.MAP())
    with pytest.raises(TypeError, match="Experiment"):
        inf.InferenceProblem(experiment.model, parameters=decay.rate)


def test_reusing_a_placeholder_lowering_rebuilds_the_residuals_over_new_values(decay):
    """as_inference_problem(reuse=) keeps the lowering and evaluates the new data's rows."""
    placeholder = decay_experiment(decay).as_inference_problem()
    loaded = decay_experiment(decay, rate=0.5)
    reloaded = loaded.as_inference_problem(reuse=placeholder)
    fresh = loaded.as_inference_problem()
    x0 = jnp.asarray(fresh.x0)
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/contracts/test_inference_problem.py
    np.testing.assert_allclose(reloaded.stacked_r_fn(x0), fresh.stacked_r_fn(x0), rtol=1e-12)
    shorter = decay_experiment(decay, times=TIMES[:3])
    with pytest.raises(ValueError, match="same names and observed shapes"):
        shorter.as_inference_problem(reuse=placeholder)


def test_multishoot_declares_its_warm_start_first_step_and_state_prior(decay):
    """Multishoot carries warm_start_steps and FirstStep; a state prior sets the row weight."""
    experiment = decay_experiment(decay)
    trajectory = inf.Multishoot(
        inf.Windows.uniform(2), warm_start_steps=2, first_step=inf.FirstStep.fraction(0.5)
    )
    leaf = experiment.as_inference_problem(trajectory=trajectory)
    assert leaf.theta_only_steps == 2 and leaf.restart_step.mode == "fraction"
    study = inf.Study((experiment,), parameters=ds.ParameterSet(shared=(decay.rate,)))
    assert study.compile_inference_problem(state_prior=inf.Gaussian(0.25)).state_prior_weight == 4.0
    with pytest.raises(ValueError, match="warm_start_steps"):
        inf.Multishoot(inf.Windows.uniform(2), warm_start_steps=-1)
    with pytest.raises(TypeError, match="first_step"):
        inf.Multishoot(inf.Windows.uniform(2), first_step="auto")


def test_windows_from_states_seed_every_window_without_scoring_the_seeds(decay):
    """Windows.from_states starts each window at a given state that enters no residual row."""
    experiment = decay_experiment(decay)
    states = decay_values()[:, None]
    leaf = experiment.as_inference_problem(
        trajectory=inf.Multishoot(inf.Windows.from_states(TIMES, states))
    )
    residual = np.asarray(leaf.stacked_r_fn(jnp.asarray([TRUE_RATE])))
    assert leaf.residual_rows == 4 and residual.shape == (4,)
    # measured 3.0e-5 largest row (rows are 50 times the misfit), CPU VM 2026-10-22,
    # pytest tests/contracts/test_inference_problem.py
    assert float(np.max(np.abs(residual))) < 1e-3
    with pytest.raises(ValueError, match="one full state per frame"):
        inf.Windows.from_states(TIMES, states[:-1])


def test_a_window_initialization_is_stored_once_and_read_back(decay):
    """The window sweep is stored under its content key in DIFFSOUP_CACHE_DIR and reused."""
    directory = Path(os.environ["DIFFSOUP_CACHE_DIR"]) / "window-initialization"
    before = set(directory.glob("*.npz"))
    trajectory = inf.Multishoot(inf.Windows.uniform(2))
    first = decay_experiment(decay, rate=0.55).as_inference_problem(trajectory=trajectory)
    added = set(directory.glob("*.npz")) - before
    assert len(added) == 1
    second = decay_experiment(decay, rate=0.55).as_inference_problem(trajectory=trajectory)
    assert second.window_initialization.provenance["content_key"] == next(iter(added)).stem
    np.testing.assert_array_equal(np.asarray(second.x0), np.asarray(first.x0))
    assert set(directory.glob("*.npz")) - before == added


def test_a_trial_predicate_a_handle_regularizer_and_a_resume_reach_the_problem(
    decay, plain, tmp_path
):
    """The predicate and a per-handle Tikhonov row act on the problem; resume= takes a saved
    fit of the same problem and refuses one of another."""
    regularized = inf.InferenceProblem(
        decay_experiment(decay),
        parameters=decay.rate,
        regularization=inf.Tikhonov(lambda_=0.01, on=decay.rate),
        trial_predicate=lambda x: jnp.all(x < 1.0),
    )
    x0 = jnp.asarray(regularized.x0)
    assert bool(regularized.trial_is_valid(x0)) and not bool(regularized.trial_is_valid(x0 + 5.0))
    rows = regularized.regularization.residual(x0)
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/contracts/test_inference_problem.py
    np.testing.assert_allclose(rows, np.sqrt(0.01) * x0, rtol=1e-12)
    problem, fit = plain
    fit.save(tmp_path / "fit")
    resumed = small_fit(problem, inf.LevenbergMarquardt(max_steps=1), resume=tmp_path / "fit")
    np.testing.assert_array_equal(resumed.initial_params, fit.params)
    foreign = dataclasses.replace(fit, info={"schema_fingerprint": "not-this-problem"})
    other = inf.InferenceProblem(decay_experiment(decay, "other"), parameters=decay.rate)
    with pytest.raises(ValueError, match="same problem"):
        small_fit(other, inf.LevenbergMarquardt(max_steps=1), resume=foreign)


def test_a_type_error_raised_at_a_trial_surfaces_from_optsolve(decay):
    """A TypeError of the trial predicate at the first trial is raised, not taken as a rejection."""
    seen = []

    def predicate(x):
        seen.append(x)
        if len(seen) > 1:
            raise TypeError("the predicate broke at the trial")
        return jnp.asarray(True)

    problem = inf.InferenceProblem(
        decay_experiment(decay), parameters=decay.rate, trial_predicate=predicate
    )
    with pytest.raises(TypeError, match="broke at the trial"):
        small_fit(problem, inf.LevenbergMarquardt(max_steps=2))


def test_a_problem_lowered_onto_a_one_device_mesh_fingerprints_by_its_axes():
    """Two builds on a one-device mesh share a fingerprint, which differs from the unplaced one."""

    def placed(layout):
        decay = decay_recipe("inference_problem_mesh", layout=layout)
        return inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)

    first = placed(ds.Layout(mesh=ds.global_mesh(("data",))))
    second = placed(ds.Layout(mesh=ds.global_mesh(("data",))))
    local = placed(None)
    assert first.parameter_schema.fingerprint == second.parameter_schema.fingerprint
    assert local.parameter_schema.fingerprint != first.parameter_schema.fingerprint


def test_a_fit_at_one_solver_tolerance_starts_the_problem_at_another(decay, plain):
    """with_start(fit) takes a fit of the same recipe solved at another tolerance."""
    _, fit = plain
    # The solver tolerance under test, not a bound: measured, 1.0.0 refused this start with
    # a schema fingerprint ValueError, CPU workstation 2026-10-28, bugfix/we1fix/m.py
    coarse = inf.InferenceProblem(
        decay_experiment(decay, rtol=1e-4, atol=1e-4), parameters=decay.rate
    )
    np.testing.assert_array_equal(coarse.with_start(fit).x0, np.asarray(fit.params))
