"""Batch lowering: windows and cohort lanes run as true batches and reuse their programs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from tests.support.inference import (
    CELLS,
    TIMES,
    decay_experiment,
    decay_recipe,
    decay_study,
)
from tests.support.programs import equations


def _loops(closed):
    return [
        (equation.primitive.name, equation.params.get("length"))
        for equation in equations(closed)
        if equation.primitive.name in {"while", "scan", "map"}
    ]


def test_multishoot_windows_lower_to_one_batched_solve_without_a_window_scan():
    """Four Multishoot windows run in one solve loop; no scan steps over the windows."""
    decay = decay_recipe("batch_lowering_windows")
    problem = inf.InferenceProblem(
        decay_experiment(decay),
        parameters=decay.rate,
        trajectory=inf.Multishoot(inf.Windows.at(TIMES)),
    )
    loops = _loops(jax.make_jaxpr(problem.loss_fn)(jnp.asarray(problem.x0)))
    windows = len(TIMES) - 1
    assert [name for name, _ in loops].count("while") == 1
    assert ("scan", windows) not in loops


@pytest.fixture(scope="module")
def cohort():
    """A three-experiment homogeneous cohort lowered onto lanes, with its prepare report."""
    decay = decay_recipe(
        "batch_lowering_cohort",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    study = decay_study(decay)
    lowered = ds.lower(study, objective=inf.MAP())
    return decay, study, lowered, lowered.prepare()


def test_a_homogeneous_cohort_compiles_one_program_per_mode_and_scans_no_lanes(cohort):
    """Three like experiments share one bucket: one value and one blocks program, no lane scan."""
    _, _, lowered, report = cohort
    assert sorted((record.bucket, record.mode) for record in report.records) == [
        (0, "blocks"),
        (0, "value"),
    ]
    loops = _loops(jax.make_jaxpr(lowered.log_likelihood)(jnp.asarray(lowered.x0)))
    assert ("scan", len(CELLS)) not in loops


def test_a_reordered_or_reloaded_cohort_reuses_every_compiled_program(cohort):
    """balance_owners' reordering and new data of the same structure compile nothing new."""
    decay, study, _, _ = cohort
    reordered = ds.lower(inf.balance_owners(study, 2), objective=inf.MAP())
    reloaded = ds.lower(decay_study(decay, noisy=False), objective=inf.MAP())
    for lowered in (reordered, reloaded):
        assert {record.cache for record in lowered.prepare().records} == {"reused"}
