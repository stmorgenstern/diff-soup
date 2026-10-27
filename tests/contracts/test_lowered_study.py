"""Lowered studies: TrustRegion on ds.lower(study), its verdicts, resume, bootstrap and sampling.

Every fit of a lowered study evaluates through evaluation state of its own (the
step memory that orders its calls, the inputs kept on the device, its ledger),
so two fits of one study make the same attempts. One compiled program gives an
attempt's solve at zero damping, the decrease the model predicts for that full
step and the multiplier search of the trust-region step; it agrees with the
chain of host-driven solves.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference.hierarchical import _level_elimination as elimination
from diff_soup.inference.hierarchical.level_solvers import (
    _full_step_decrease,
    _LaneRoute,
    _scale_candidate,
    _subproblem,
    _trial_point,
)
from tests.support.inference import CELLS, decay_recipe, decay_study, small_fit


@pytest.fixture(scope="module")
def lowered():
    """A three-owner MAP decay study, lowered and prepared once."""
    decay = decay_recipe(
        "lowered_study_decay",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    study = ds.lower(decay_study(decay), objective=inf.MAP())
    return decay, study, study.prepare()


@pytest.fixture(scope="module")
def joint_fit(lowered):
    _, study, _ = lowered
    return small_fit(study, inf.TrustRegion(max_steps=30))


def test_trust_region_defaults_are_the_documented_table():
    """TrustRegion() holds the documented defaults of every setting the table names."""
    solver = inf.TrustRegion()
    settings = {name: getattr(solver, name) for name in _DEFAULTS}
    assert settings == _DEFAULTS


_DEFAULTS = {
    "hessian": "gauss-newton",
    "nested_at": None,
    "radius_init": 1.0,
    "radius_max": 1000.0,
    "eta": 1e-4,
    "grad_tol": None,
    "step_tol": None,
    "cost_tol": None,
    "stationarity_tol": None,
    "max_consecutive_invalid_trials": 5,
    "gradient_floor": "auto",
    "max_steps": 30,
    "max_evaluations": None,
}


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_consecutive_invalid_trials", 0),
        ("hessian", "bogus"),
        ("stationarity_tol", -1.0),
        ("radius_init", 0.0),
        ("gradient_floor", -1.0),
        ("inner_fraction", 1.0),
    ],
)
def test_trust_region_refuses_an_out_of_range_setting_by_name(setting, value):
    """An out-of-range constructor setting is a ConfigurationError naming the setting."""
    with pytest.raises(ds.ConfigurationError, match=setting):
        inf.TrustRegion(**{setting: value})


def test_prepare_compiles_every_bucket_once_and_reports_it(lowered):
    """prepare() returns a PrepareReport of the value and blocks programs; a repeat reuses them."""
    _, study, report = lowered
    assert type(report).__name__ == "PrepareReport"
    assert sorted(record.mode for record in report.records) == ["blocks", "value"]
    assert all(record.cache != "reused" for record in report.records)
    assert {record.cache for record in study.prepare().records} == {"reused"}


def test_joint_and_nested_fits_reach_the_same_optimum(lowered, joint_fit):
    """The joint trust region and the one nested at the owner level end at one point."""
    _, study, _ = lowered
    nested = small_fit(study, inf.TrustRegion(max_steps=30, nested_at="cell"))
    assert joint_fit.converged and nested.converged
    # measured 2.1e-7 absolute difference, CPU VM 2026-10-22,
    # pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(nested.params, joint_fit.params, atol=1e-5)


class _StopAtSecondAttempt(inf.Callback):
    def __init__(self):
        self.attempts = 0

    def on_step(self, info):
        self.attempts += 1
        if self.attempts == 2:
            raise inf.StopFit()


@pytest.fixture(scope="module")
def stopped(lowered, tmp_path_factory):
    _, study, _ = lowered
    directory = tmp_path_factory.mktemp("lowered_checkpoint")
    fit = small_fit(
        study,
        inf.TrustRegion(max_steps=30),
        callbacks=(_StopAtSecondAttempt(),),
        checkpoint=directory,
    )
    return directory, fit


def test_stop_fit_ends_a_checkpointed_fit_after_writing_its_attempts(stopped):
    """A StopFit raised by a callback ends the fit STOPPED with its attempt file written."""
    directory, fit = stopped
    assert fit.status is inf.TerminationStatus.STOPPED and fit.n_steps == 2
    assert [path.name for path in sorted(directory.glob("attempt-*.npz"))] == ["attempt-000001.npz"]


def test_a_fit_resumed_from_its_checkpoint_equals_the_uninterrupted_fit(
    lowered, joint_fit, stopped
):
    """optsolve(checkpoint=dir) resumes the stopped fit and ends where the straight fit ends."""
    _, study, _ = lowered
    directory, _ = stopped
    resumed = small_fit(study, inf.TrustRegion(max_steps=30), checkpoint=directory)
    assert resumed.status == joint_fit.status and resumed.n_steps == joint_fit.n_steps
    # measured 0.0 relative difference, CPU VM 2026-10-22,
    # pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(resumed.params, joint_fit.params, rtol=1e-12)


def test_a_fit_without_other_stops_ends_at_its_derivatives_accuracy(lowered):
    """With derivatives accurate to 1e-3 and the other stops off, the fit ends at that floor."""
    _, study, _ = lowered
    stops_off = {"stationarity_tol": 0.0, "cost_tol": 0.0, "step_tol": 0.0}
    fit = small_fit(study, inf.TrustRegion(max_steps=30, gradient_floor=1e-3, **stops_off))
    assert fit.status is inf.TerminationStatus.DERIVATIVE_ACCURACY and fit.converged
    assert fit.trust_region.stop == "derivative_accuracy" and fit.trust_region.gradient_floor > 0.0


def test_trust_region_max_steps_zero_returns_the_start_point_and_a_negative_budget_is_refused(
    lowered, joint_fit
):
    """max_steps=0 makes no step: the start point, its cost and MAX_STEPS; max_steps=-1 is a
    ConfigurationError."""
    _, study, _ = lowered
    fit = small_fit(study, inf.TrustRegion(max_steps=0))
    assert fit.status is inf.TerminationStatus.MAX_STEPS and fit.n_steps == 0
    np.testing.assert_array_equal(fit.params, joint_fit.initial_params)
    assert fit.cost == joint_fit.cost_history[0]
    with pytest.raises(ds.ConfigurationError, match="max_steps must be an integer >= 0"):
        small_fit(study, inf.TrustRegion(max_steps=-1))


def test_a_step_test_ending_a_fit_its_model_still_improves_reports_stalled(lowered):
    """A step test that ends the loop while the model predicts a decrease is STALLED."""
    _, study, _ = lowered
    fit = small_fit(
        study,
        inf.TrustRegion(
            max_steps=30, radius_init=1e-6, step_tol=1e-3, stationarity_tol=0.0, gradient_floor=None
        ),
    )
    assert fit.status is inf.TerminationStatus.STALLED and not fit.converged
    assert fit.trust_region.stall_reason == "radius_collapsed" and fit.message.startswith("stalled")


def test_the_one_step_bootstrap_sandwich_sums_the_members_influence(lowered, joint_fit):
    """bootstrap(steps=0) keeps every one-step estimate; its sandwich is sum of u_m u_m^T."""
    decay, study, _ = lowered
    boot = inf.bootstrap(study, joint_fit, replicates=3, key=jax.random.key(3), steps=0)
    shared = boot.one_step.influence[:, :1]
    assert boot.n_kept == 3 and boot.outcomes is None
    np.testing.assert_array_equal(boot.values(decay.rate), boot.one_step.estimates[:, 0])
    # measured 0.0 difference on a 4.5e-4 entry, CPU VM 2026-10-22,
    # pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(boot.one_step.sandwich, shared.T @ shared, rtol=1e-12)


def test_sample_draws_the_posterior_of_a_lowered_study(lowered, joint_fit):
    """inf.sample on a lowered study's MAP fit returns Draws of every handle."""
    decay, study, _ = lowered
    draws = inf.sample(study, joint_fit, kernel="rmh", key=jax.random.key(4), chains=1, draws=3)
    assert isinstance(draws, inf.Draws) and draws[decay.rate].shape == (1, 3)
    assert set(draws[decay.source]) == {"cell-a", "cell-b", "cell-c"}


@pytest.fixture(scope="module")
def again(lowered, joint_fit):
    """The joint fit repeated after the module's other fits of the study."""
    _, study, _ = lowered
    return small_fit(study, inf.TrustRegion(max_steps=30))


def test_two_fits_of_one_study_make_the_same_attempts(joint_fit, again):
    """A later fit of the study, after the module's other fits, repeats the first fit's
    attempts and ends at its point and cost to the last bit."""
    counts = ("attempts", "accepted", "blocks", "value")
    # measured 5 attempts, 5 accepted, 6 blocks and 5 value requests, CPU VM 2026-10-22,
    # pytest tests/contracts/test_lowered_study.py
    assert (
        {name: int(again.work[name]) for name in counts}
        == {name: int(joint_fit.work[name]) for name in counts}
        == {"attempts": 5, "accepted": 5, "blocks": 6, "value": 5}
    )
    np.testing.assert_array_equal(again.params, joint_fit.params)
    assert again.cost == joint_fit.cost


def test_each_fit_ledger_holds_its_own_evaluations_only(again):
    """A fit's saved ledger names every lane and counts one evaluation per request of that fit,
    none of the study's earlier fits."""
    requests = int(again.work["blocks"]) + int(again.work["value"])
    lanes = tuple(str(lane) for lane in again.state["ledger.lane_ids"])
    assert lanes == tuple(owner for owner, _ in CELLS)
    np.testing.assert_array_equal(again.state["ledger.evaluations"], requests)


def test_a_run_of_one_process_fits_the_study_and_closes_it(lowered, joint_fit):
    """ds.distributed.run on one process yields the study lowered in one process, its fit equals
    the fit of ds.lower(study), and after the block the study is closed, which in one process
    shuts nothing down: closing again does nothing and the study still evaluates."""
    decay, _, _ = lowered
    with ds.distributed.run(decay_study(decay), objective=inf.MAP()) as study:
        assert isinstance(study, inf.LoweredStudy) and study.processes == 1
        fit = small_fit(study, inf.TrustRegion(max_steps=30))
    np.testing.assert_array_equal(fit.params, joint_fit.params)
    assert fit.cost == joint_fit.cost
    assert study.close() is None
    assert study.loss_fn(fit.params) == fit.cost


@pytest.fixture(scope="module")
def attempt(lowered):
    """The start point's system, its subproblem and a radius a tenth of the full step's."""
    _, study, _ = lowered
    route = _LaneRoute(inf.TrustRegion(), study)
    point = np.asarray(study.x0, dtype=np.float64)
    system = route.system(route.evaluate("blocks", point), point)
    gradient, diagonal = route.vectors(system)
    scale = _scale_candidate(system, inf.TrustRegion().x_scale, diagonal)
    coordinates = _subproblem(point, gradient, scale, route.bounds)
    zero = elimination.solve_scaled(system, coordinates.scale, 0.0, coordinates.addition)
    radius = 0.1 * float(np.linalg.norm(zero[0]))
    return route, system, gradient, coordinates, zero, radius, point


def test_the_attempt_program_gives_the_chain_of_solves(attempt):
    """attempt_solve gives the zero-damping solve, the full step's predicted decrease and the
    radius-cut step, decrease and multiplier of the host chain of solves."""
    _, system, gradient, coordinates, zero, radius, _ = attempt
    scale, addition = coordinates.scale, coordinates.addition
    decrease = _full_step_decrease(system, gradient, scale, zero, addition)
    step, predicted, multiplier, _ = elimination.level_secular_step(
        system, scale, radius, first=zero, addition=addition
    )
    fused = elimination.attempt_solve(system, scale, radius, addition)
    assert not fused.failed and multiplier > 0.0  # the radius cut the step: the search ran
    chain = (zero[0], zero[1], decrease, step, predicted, multiplier)
    ours = (fused.z, fused.q_squared, fused.decrease, fused.step, fused.predicted)
    for value, expected in zip((*ours, fused.multiplier), chain, strict=True):
        # measured 0.0 on every entry here, at most 2.4e-16 relative on the step over every
        # attempt of four fits, CPU VM 2026-10-22, pytest tests/contracts/test_lowered_study.py
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)
    assert fused.answers(scale, radius, addition) and not fused.answers(scale, 2 * radius, addition)


def test_a_step_the_search_cannot_bracket_is_refused_where_the_step_is_taken(attempt):
    """At radius 0 the attempt program reports a failed search and keeps its zero-damping
    solve; taking the step raises the host chain's error."""
    route, system, gradient, coordinates, zero, _, point = attempt
    fused = elimination.attempt_solve(system, coordinates.scale, 0.0, coordinates.addition)
    assert fused.failed
    np.testing.assert_array_equal(fused.z, zero[0])
    with pytest.raises(ValueError, match="could not bracket"):
        _trial_point(
            system,
            point,
            gradient,
            coordinates,
            0.0,
            first=fused,
            fixed=np.zeros((0,), dtype=np.int64),
            bounds=route.bounds,
            equality=False,
        )


def test_the_trial_cost_reuses_the_regularizer_at_the_point(attempt):
    """The trial cost is the weighted lane costs plus the priors' penalty, and the system at that
    point takes the regularizer blocks the trial cost made."""
    route, *_, start = attempt
    point = start + 0.01

    def weighted(results):
        return float(np.sum(route.omega * np.asarray(results.cost, dtype=np.float64)[0]))

    results = route.evaluate("value", point)
    penalty = route.penalty(point)
    assert penalty > 0.0
    # measured 0.0 relative, CPU VM 2026-10-22, pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(
        route.trial_cost(results, point), weighted(results) + penalty, rtol=1e-12
    )
    kept = route.regularizer_at(point)
    blocks = route.evaluate("blocks", point)
    system = route.system(blocks, point)
    assert route.regularizer_at(point) is kept
    # measured 0.0 relative, CPU VM 2026-10-22, pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(float(system.cost), weighted(blocks) + kept[1], rtol=1e-12)


class _Records(inf.Callback):
    def __init__(self):
        self.steps = []

    def on_step(self, info):
        self.steps.append((info, dict(info.state)))


def test_nested_attempt_records_split_the_cost_by_owner():
    """A nested fit's record of every attempt holds each owner's cost at the current point and at
    the trial, and the shared remainder; at every attempt they sum to the fit's cost there."""
    from tests.support.inference import NOISE, decay_experiment

    decay = decay_recipe(
        "owner_cost_decay",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    cells = CELLS[:2]
    study = inf.Study(
        tuple(
            decay_experiment(decay, owner, source=source, noise=(index + 1) * NOISE)
            for index, (owner, source) in enumerate(cells)
        ),
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": tuple(owner for owner, _ in cells)},
        independent=True,
    )
    records = _Records()
    fit = small_fit(
        ds.lower(study, objective=inf.MAP()),
        inf.TrustRegion(max_steps=8, nested_at="cell"),
        callbacks=(records,),
    )
    assert records.steps
    current = float(fit.cost_history[0])
    for info, state in records.steps:
        owners = np.asarray(state["record.owner_costs"], dtype=np.float64)
        assert owners.shape == (2,)
        total = float(owners.sum() + state["record.owner_shared_cost"])
        # measured 1.8e-16 relative difference, workstation RTX 4070 Ti 2026-10-25,
        # pytest tests/contracts/test_lowered_study.py -k owner
        np.testing.assert_allclose(total, current, rtol=1e-10)
        assert np.asarray(state["record.owner_predicted"]).shape == (2,)
        trial = np.asarray(state["record.owner_trial_costs"], dtype=np.float64)
        if info.accepted:
            trial_total = float(trial.sum() + state["record.owner_trial_shared_cost"])
            # measured 1.8e-16 relative difference, as above
            np.testing.assert_allclose(trial_total, float(info.cost), rtol=1e-10)
        current = float(info.cost)


def test_a_member_whose_step_cannot_be_bracketed_is_held_when_asked(attempt):
    """A member whose subtree has a non-finite gradient has no finite step inside its radius: the
    members' step raises, or, with hold=True, gives that member a zero step and a nan multiplier
    and every other member the step it takes without the hold."""
    import equinox as eqx
    import jax.numpy as jnp

    _, system, _, coordinates, _, radius, _ = attempt
    gradient = jnp.asarray(system.gradient[1]).at[1].set(jnp.nan)
    poisoned = eqx.tree_at(
        lambda s: s.gradient, system, (None, gradient) + tuple(system.gradient[2:])
    )
    count = system.counts[0]
    radii, active = np.full((count,), radius), np.ones((count,), dtype=bool)
    scale = coordinates.scale
    with pytest.raises(ValueError, match="could not bracket"):
        elimination.inner_secular_step(poisoned, scale, radii, 1, active)
    step, _, multipliers, _ = elimination.inner_secular_step(
        poisoned, scale, radii, 1, active, hold=True
    )
    clean, _, clean_multipliers, _ = elimination.inner_secular_step(system, scale, radii, 1, active)
    owners = elimination.node_of_columns(system, 1)
    assert np.isnan(multipliers[1]) and np.all(step[owners == 1] == 0.0)
    others = (owners >= 0) & (owners != 1)
    np.testing.assert_array_equal(step[others], np.asarray(clean)[others])
    np.testing.assert_array_equal(multipliers[[0, 2]], clean_multipliers[[0, 2]])


def test_a_member_with_non_finite_sensitivities_at_a_finite_cost_retreats_and_fails(tmp_path):
    """A member whose lanes solve at a finite cost but give non-finite Gauss-Newton blocks has its
    lanes recorded as failed: it retreats, an outer trial it cannot solve names it in
    failed_owners, and the fit ends at a finite cost instead of raising."""
    from tests.support.inference import (
        NOISE,
        TIMES,
        decay_experiment,
        decay_values,
        mean_measurement,
        solver,
    )

    decay = decay_recipe(
        "nonfinite_blocks_decay",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    start = jax.numpy.asarray(decay_values(TIMES, rate=0.45, source=0.0))

    def poisoned_mean(state):
        # the mean, plus sqrt(m - m) = 0 whose derivative is infinite, once the prediction leaves
        # the start trajectory: a finite cost with non-finite sensitivities off the start point
        value = jax.numpy.mean(state)
        frozen = jax.lax.stop_gradient(value)
        moved = jax.numpy.min(jax.numpy.abs(frozen - start)) > 1e-4
        return value + jax.numpy.where(moved, jax.numpy.sqrt(value - frozen), 0.0)

    experiments = []
    for index, (owner, source) in enumerate(CELLS):
        experiment = decay_experiment(decay, owner, source=source, noise=(index + 1) * NOISE)
        if owner == "cell-b":
            values = experiment.measurements[0].values
            experiment = inf.Experiment(
                model=decay.model,
                solver=solver(1e-6, 1e-8, "dense"),
                t_span=(0.0, float(TIMES[-1])),
                initial_state=decay.initial_state,
                measurements=(mean_measurement(values, TIMES, observe=poisoned_mean),),
                id=owner,
            )
        experiments.append(experiment)
    study = inf.Study(
        tuple(experiments),
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": tuple(owner for owner, _ in CELLS)},
        independent=True,
    )
    records = _Records()
    fit = small_fit(
        ds.lower(study, objective=inf.MAP()),
        inf.TrustRegion(max_steps=6, nested_at="cell"),
        callbacks=(records,),
    )
    assert np.isfinite(fit.cost)
    named = [info for info, _ in records.steps if info.failed_owners]
    assert named and all("cell-b" in str(info.failed_owners) for info in named)
    assert any(int(np.asarray(state["record.inner_retreats"])[1]) > 0 for _, state in records.steps)
    kinds = {str(k) for _, state in records.steps for k in np.ravel(state["record.failed_kinds"])}
    assert "nonfinite_blocks_at_finite_cost" in kinds


@pytest.mark.parametrize(
    ("stop", "sigma", "weights", "status"),
    [
        ("attempts", 1e-6, [0.0, 1.0, 2.0], "converged"),
        ("attempts", 0.5, [0.0, 1.0, 2.0], "failed"),
        ("attempts", 0.5, [1.0, 1.0, 1.0], "converged"),
        ("evaluation_noise", 1e-6, [0.0, 1.0, 2.0], "converged"),
    ],
)
def test_a_replicate_without_a_step_is_kept_where_it_needs_none(stop, sigma, weights, status):
    """A bootstrap replicate that ends without an accepted step from a converged fit is kept as
    converged when its point passes the converged test or its weights are the original sample's
    (the fit is then its own answer); otherwise it fails with no_step."""
    from types import SimpleNamespace

    from diff_soup.inference.hierarchical._replicates import _Member, _Settings

    member = SimpleNamespace(
        accepted=0,
        noise_changes=[],
        lock=0,
        grad_threshold=None,
        on_rows=lambda: True,
        route=SimpleNamespace(constraint_matrix=np.zeros((0, 1))),
        weights=np.asarray(weights),
        start_converged=True,
    )
    settings = _Settings(
        view=None,
        bounds=None,
        fixed=np.zeros((0,), dtype=np.int64),
        x_scale=None,
        grad_tolerance=None,
        step_tol=0.0,
        cost_tol=1e-4,
        cost_atol=0.0,
        stationarity_tol=None,
        eta=1e-4,
        radius_init=1.0,
        radius_max=1e3,
        max_invalid=5,
        relaxed=False,
        noise_limit=None,
        steps=1,
        max_attempts=4,
    )
    member.passes_converged_test = lambda *a, **k: _Member.passes_converged_test(member, *a, **k)
    member.needs_no_step = lambda *a, **k: _Member.needs_no_step(member, *a, **k)
    _Member.end(member, settings, stop, sigma=sigma)
    assert member.status == status
    assert member.reason == (None if status == "converged" else "no_step")
