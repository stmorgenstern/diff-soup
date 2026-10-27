"""Hierarchies and studies: studies across builds, ParameterSet and Hierarchy, owner balance."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d
from tests.support.inference import (
    CELLS,
    TRUE_RATE,
    decay_experiment,
    decay_recipe,
    decay_study,
    mean_measurement,
    small_fit,
    solver,
)


def _rod(length, label, *, bounds=(0.01, 2.0), name="rate"):
    """A decay on a rod of the given length, built from its own recipe."""
    spec = ds.problem("hierarchy_cross_build")
    spec.domain("rod", cartesian_grid_1d(2, 0.0, length, method="fvm"))
    spec.field("u", mass="differential", scale=1.0)
    handle = spec.param(name, value=0.2, learnable=True, bounds=bounds)

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -getattr(q.p, name) * q.u

    times = np.linspace(0.0, 1.0, 5)
    experiment = inf.Experiment(
        spec.build(),
        solver=solver(),
        t_span=(0.0, 1.0),
        initial_state=jnp.ones(2),
        id=label,
        measurements=(mean_measurement(np.exp(-0.4 * times), likelihood=inf.Gaussian(0.1)),),
    )
    return experiment, handle


def test_a_study_takes_the_first_builds_handle_and_fits_both_models():
    """A handle of the first build selects the same-named parameter of a second build."""
    first, rate = _rod(1.0, "first")
    second, other = _rod(2.0, "second")
    study = inf.Study((first, second), parameters=ds.ParameterSet(shared=(rate,)), independent=True)
    fit = small_fit(inf.InferenceProblem(study), inf.LevenbergMarquardt(max_steps=20))
    # measured 6.5e-6 relative at solver rtol 1e-6, CPU VM 2026-10-22,
    # pytest tests/public_api/test_hierarchy.py
    np.testing.assert_allclose(float(fit[rate]), 0.4, rtol=1e-4)
    assert float(fit[other]) == float(fit[rate])


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "rate.*second.*other"),
        ("shared_bounds", "first.*0.01.*2.0.*second.*0.1.*3.0"),
        ("owner_bounds", "first.*0.01.*2.0.*second.*0.1.*3.0"),
    ],
    ids=("missing", "shared-bounds", "owner-bounds"),
)
def test_a_study_names_a_missing_parameter_and_untied_bounds(case, match):
    """A handle absent from one build, or bounds that differ across builds, is refused by name."""
    first, rate = _rod(1.0, "first")
    name = "other" if case == "missing" else "rate"
    second, _ = _rod(2.0, "second", bounds=(0.1, 3.0), name=name)
    varying = case == "owner_bounds"
    parameters = ds.ParameterSet(
        **({"varying": {"cell": (rate,)}} if varying else {"shared": (rate,)})
    )
    with pytest.raises(ValueError, match=match):
        inf.Study(
            (first, second), parameters=parameters, clusters={"cell": ("A", "A")}, independent=True
        )


def _growth_study():
    """Two owners of ``u' = rate u^2``; a trial rate of 1 or more blows up before t = 1."""
    spec = ds.problem("hierarchy_growth")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=0.5, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def growth(q):
        return q.p.rate * q.u * q.u

    model = spec.build()
    times = np.linspace(0.2, 1.0, 5)

    def experiment(owner, truth):
        values = 1.0 / (1.0 - truth * times)
        return inf.Experiment(
            model=model,
            solver=solver(),
            t_span=(0.0, 1.0),
            initial_state=spec.assemble_state({}),
            measurements=(mean_measurement(values, times, likelihood=inf.Gaussian(0.05)),),
            id=owner,
        )

    study = inf.Study(
        (experiment("steep", 0.95), experiment("mild", 0.3)),
        parameters=ds.ParameterSet(varying={"cell": (rate,)}),
        clusters={"cell": ("steep", "mild")},
        independent=True,
    )
    return study, rate


@pytest.fixture(scope="module")
def growth():
    """A TrustRegion fit of the growth study from a radius of 64, with its step records."""
    study, rate = _growth_study()
    steps = []

    class Record(inf.Callback):
        def on_step(self, info):
            steps.append(info)

    fit = small_fit(
        ds.lower(study), inf.TrustRegion(max_steps=12, radius_init=64.0), callbacks=(Record(),)
    )
    return rate, fit, steps


def test_trust_region_recovers_owner_rates_and_names_a_rejected_trials_owner(growth):
    """TrustRegion recovers each owner's rate; a trial whose solve fails is rejected with
    rho -inf, its owner named and the radius cut to a quarter."""
    rate, fit, steps = growth
    assert fit.converged
    # measured 7e-6 (steep) and 4e-6 (mild) off the truth, CPU VM 2026-10-22,
    # pytest tests/public_api/test_hierarchy.py
    np.testing.assert_allclose([fit[rate]["steep"], fit[rate]["mild"]], [0.95, 0.3], atol=1e-4)
    failed = [info for info in steps if info.failed_owners]
    first = failed[0]
    assert first.failed_owners == ("steep",) and not first.accepted
    assert first.rho == float("-inf") and first.radius == 16.0
    assert "steep" in first.rejection_reason
    assert all(info.rejection_reason is None for info in steps if info.accepted)
    assert int(fit.work["invalid_trials"]) == len(failed)


def test_a_window_sweep_that_fails_is_refused_naming_the_window():
    """A Multishoot sweep whose forward solve blows up in window 0 is refused at construction."""
    spec = ds.problem("hierarchy_blown_sweep")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    hot = spec.param("rate", value=2.5, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: q.p.rate * q.u * q.u)
    experiment = inf.Experiment(
        model=spec.build(),
        solver=solver(),
        t_span=(0.0, 1.0),
        initial_state=spec.assemble_state({}),
        measurements=(mean_measurement(np.ones(5), likelihood=inf.Gaussian(0.05)),),
        id="hot",
    )
    with pytest.raises(ValueError, match="window-initialization sweep failed in window 0"):
        inf.InferenceProblem(
            experiment, parameters=(hot,), trajectory=inf.Multishoot(inf.Windows.uniform(2))
        )


@pytest.fixture(scope="module")
def owners():
    """A joint fit of the three-owner decay study: one shared rate, one source per owner."""
    decay = decay_recipe("hierarchy_owners", source=True)
    study = decay_study(decay, noisy=False)
    return decay, study, small_fit(inf.InferenceProblem(study), inf.LevenbergMarquardt())


def test_a_joint_result_resolves_shared_and_owner_handles(owners):
    """fit[rate] is the shared value; fit[source][owner] each owner's, as its experiment sees it."""
    decay, _, fit = owners
    # measured 1.2e-5 on the rate, 1.0e-5 on a source, CPU VM 2026-10-22,
    # pytest tests/public_api/test_hierarchy.py
    np.testing.assert_allclose(float(fit[decay.rate]), TRUE_RATE, atol=1e-4)
    for owner, source in CELLS:
        assert float(fit.experiments[owner][decay.source]) == float(fit[decay.source][owner])
        # measured 1.0e-5, the run above
        np.testing.assert_allclose(float(fit[decay.source][owner]), source, atol=1e-4)


def test_a_fit_warm_starts_the_same_study_and_refuses_another_schema(owners):
    """with_start(fit) starts at the fit's point; a fit of another schema is refused."""
    _, study, fit = owners
    problem = inf.InferenceProblem(study).with_start(fit)
    np.testing.assert_array_equal(problem.x0, np.asarray(fit.params))
    schema = fit.parameter_schema
    foreign = dataclasses.replace(
        fit, parameter_schema=type(schema)(schema.entries, fingerprint=schema.fingerprint + "-x")
    )
    with pytest.raises(ValueError, match="fingerprint"):
        inf.InferenceProblem(study).with_start(foreign)


def test_a_hierarchy_declares_the_layout_a_parameter_set_declares(owners):
    """Hierarchy(per={"cell": ...}) with members on the experiments orders as ParameterSet does."""
    decay, study, _ = owners
    members = tuple(
        dataclasses.replace(experiment, parameters={"cell": owner})
        for experiment, (owner, _) in zip(study.experiments, CELLS)
    )
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(decay.rate,), per={"cell": (decay.source,)})
    nested = inf.Study(members, parameters=hierarchy)
    assert nested.clusters["cell"] == tuple(owner for owner, _ in CELLS)
    assert inf.InferenceProblem(nested).theta_names == inf.InferenceProblem(study).theta_names


def test_balance_owners_orders_owners_heaviest_first_across_lanes():
    """balance_owners places the heaviest owner first and keeps each owner's experiments together."""
    decay = decay_recipe("hierarchy_balance")
    plan = (("a", 4), ("a", 4), ("b", 9), ("c", 3))
    experiments = tuple(
        decay_experiment(decay, f"{owner}-{index}", times=np.linspace(0.0, 1.0, count))
        for index, (owner, count) in enumerate(plan)
    )
    study = inf.Study(
        experiments,
        parameters=ds.ParameterSet(varying={"cell": (decay.rate,)}),
        clusters={"cell": tuple(owner for owner, _ in plan)},
        independent=True,
    )
    balanced = inf.balance_owners(study, 2)
    assert [item.id for item in balanced.experiments] == ["b-2", "c-3", "a-0", "a-1"]
    assert balanced.clusters["cell"] == ("b", "c", "a", "a")
    assert inf.balance_owners(experiments, 3)[0].id == "b-2"
