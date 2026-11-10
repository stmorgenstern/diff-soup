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
from diff_soup.optimization import inference as inf
from diff_soup.optimization.inference.engine._kernel import _solve
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._lane_route import _LaneRoute
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _full_step_decrease,
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
    # measured 3.0e-8 absolute difference, CPU workstation 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py
    np.testing.assert_allclose(nested.params, joint_fit.params, atol=1e-5)


def test_the_nested_mode_without_polish_makes_the_joint_attempts():
    """With no polish iterations the nested mode's attempts are the joint mode's: at every attempt
    the same acceptance, cost, point and radius, and the fits end with the same stop and status."""
    study = ds.lower(_two_owners("zero_polish_decay"), objective=inf.MAP())
    fits = []
    for solver in (
        inf.TrustRegion(max_steps=8),
        inf.TrustRegion(max_steps=8, nested_at="cell", inner=inf.TrustRegion(max_steps=0)),
    ):
        records = _Records()
        fits.append((small_fit(study, solver, callbacks=(records,)), records.steps))
    (joint, joint_steps), (nested, nested_steps) = fits
    assert (nested.status, nested.trust_region.stop) == (joint.status, joint.trust_region.stop)
    assert [info.accepted for info, _ in nested_steps] == [info.accepted for info, _ in joint_steps]
    rows = [
        [(info.cost, info.radius, *info.params) for info, _ in steps]
        for steps in (joint_steps, nested_steps)
    ]
    # measured 0.0 relative difference over every attempt and field, CPU workstation 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py -k without_polish
    np.testing.assert_allclose(rows[1], rows[0], rtol=1e-10, atol=0.0)


def test_inequality_rows_end_the_joint_fit_at_the_interior_point_solution():
    """InequalityRows on a member's own value, active with a bound: the joint trust region ends at
    the interior point's solution of the same objective, where its own gradient meets the KKT
    conditions."""
    import jax.numpy as jnp

    from diff_soup.domains import cartesian_grid_1d
    from tests.support.inference import NOISE, TIMES, decay_values, solver

    # The lanes observe the exact mean, read from the parameters, so the fit's cost is the
    # closed form below. The rate's lower bound and cell-b's row hold at the solution.
    spec = ds.problem("inequality_rows_decay")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=0.9, learnable=True, bounds=(0.65, 1.2))
    source = spec.param("source", value=0.0, learnable=True, bounds=(-0.25, 0.25))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -q.p.rate * q.u + q.p.source)
    mean = lambda r, s, t: s / r + (1.0 - s / r) * jnp.exp(-r * t)  # noqa: E731
    spec.observe.scalar(name="exact", fn=lambda q: mean(q.p.rate, q.p.source, q.t))
    model = spec.build()
    data = [decay_values(TIMES, 0.6, s) + (i + 1) * NOISE for i, (_, s) in enumerate(CELLS[:2])]
    experiments = tuple(
        inf.Experiment(
            model=model,
            solver=solver(),
            t_span=(0.0, 1.0),
            initial_state=spec.assemble_state({}),
            measurements=(
                inf.Measurement(
                    "exact",
                    times=jnp.asarray(TIMES),
                    values=data[i],
                    observe="exact",
                    likelihood=inf.Gaussian(scale=0.02),
                ),
            ),
            id=owner,
        )
        for i, (owner, _) in enumerate(CELLS[:2])
    )
    study = inf.Study(
        experiments,
        parameters=ds.ParameterSet(shared=(rate,), varying={"cell": (source,)}),
        clusters={"cell": ("cell-a", "cell-b")},
        independent=True,
    )
    lowered = ds.lower(study).with_start(np.array([0.9, 0.0, -0.1]))
    limits = {"cell-a": jnp.asarray(1.0), "cell-b": jnp.asarray(-0.02)}
    rows = inf.InequalityRows(lambda own, w, d: own[0] + 0.01 * w[0] - d, level="cell", data=limits)
    optimizer = inf.TrustRegion(max_steps=40, cost_tol=1e-14, step_tol=1e-14, inequalities=(rows,))
    x = np.asarray(small_fit(lowered, optimizer).params)

    def residual(z, args=None):
        return jnp.concatenate(
            [(data[i] - mean(z[0], z[1 + i], jnp.asarray(TIMES))) / 0.02 for i in range(2)]
        )

    c = lambda z, args=None: z[1:] + 0.01 * z[0] - jnp.asarray([1.0, -0.02])  # noqa: E731
    reference = ds.optimization.LeastSquaresProblem.from_residual(
        jnp.asarray([0.9, 0.0, -0.1]),
        residual,
        bounds=(jnp.asarray([0.65, -0.25, -0.25]), jnp.asarray([1.2, 0.25, 0.25])),
    ).with_constraints(ds.optimization.Constraints(inequality=c, n_inequality=2))
    # measured: the reference meets tol=1e-12 with status 0, workstation CPU 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py -k inequality
    expected, _ = ds.optimization.InteriorPoint(tol=1e-12).run(reference, reference.x0, None)
    # measured 4.2e-13 relative, workstation CPU 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py -k inequality
    np.testing.assert_allclose(x, np.asarray(expected), rtol=1e-8)
    # The gradient against the active bound's and row's normals: g = nu e_rate - mu C_b.
    lowered.prepare(modes=("gradient",))
    g = np.asarray(lowered.value_and_gradient(x).gradients).reshape(-1)
    normals = np.array([[1.0, 0.0, 0.0], [-0.01, 0.0, -1.0]]).T
    weights, *_ = np.linalg.lstsq(normals, g, rcond=None)
    # measured 4.1e-15 relative KKT residual, weights 138.5 and 179.1, as above
    assert np.linalg.norm(g - normals @ weights) <= 1e-8 * np.linalg.norm(g)
    assert np.all(weights > 0.0)


# The linear toy's matrix: its cost is 0.5 (z - m)^T H (z - m) in z = (rate, source_a, source_b).
_TOY_H = np.array([[2.0, -0.9, -0.9], [-0.9, 1.0, 0.0], [-0.9, 0.0, 1.0]])


def _linear_toy(name, m):
    """Two owners whose lanes observe rate * a(t) + source * b(t) exactly, a and b orthonormal
    under the likelihood up to the coupling -0.9, so the cost is the closed form above."""
    import jax.numpy as jnp

    from diff_soup.domains import cartesian_grid_1d
    from tests.support.inference import TIMES, solver

    scale = 0.02

    def a(t):
        return scale / jnp.sqrt(5.0) * jnp.ones_like(t)

    def b(t):
        return scale * (-0.9 / jnp.sqrt(5.0) + jnp.sqrt(0.19) * (4.0 * t - 2.0) / jnp.sqrt(10.0))

    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=0.9, learnable=True)
    source = spec.param("source", value=0.0, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -0.5 * q.u)
    spec.observe.scalar(name="exact", fn=lambda q: q.p.rate * a(q.t) + q.p.source * b(q.t))
    model, times = spec.build(), jnp.asarray(TIMES)
    experiments = tuple(
        inf.Experiment(
            model=model,
            solver=solver(),
            t_span=(0.0, 1.0),
            initial_state=spec.assemble_state({}),
            measurements=(
                inf.Measurement(
                    "exact",
                    times=times,
                    values=m[0] * a(times) + m[1 + i] * b(times),
                    observe="exact",
                    likelihood=inf.Gaussian(scale=scale),
                ),
            ),
            id=owner,
        )
        for i, owner in enumerate(("cell-a", "cell-b"))
    )
    parameters = ds.ParameterSet(shared=(rate,), varying={"cell": (source,)})
    clusters = {"cell": ("cell-a", "cell-b")}
    return ds.lower(
        inf.Study(experiments, parameters=parameters, clusters=clusters, independent=True)
    )


@pytest.mark.parametrize("nested_at", [None, "cell"])
def test_inequality_rows_end_the_fit_at_the_closed_form_constrained_minimum(nested_at):
    """One row per owner on its own value, source <= 1: from a start whose full step crosses both
    rows, the joint and the nested fit end at the constrained minimum, where cell-a's row holds
    with a positive multiplier and cell-b's row, held by the first step, is released."""
    import jax.numpy as jnp

    m = np.array([3.0, 2.0, 1.1])
    limits = {"cell-a": jnp.asarray(1.0), "cell-b": jnp.asarray(1.0)}
    rows = inf.InequalityRows(lambda own, w, d: own[:1] - d, level="cell", data=limits)
    optimizer = inf.TrustRegion(
        max_steps=40, radius_init=100.0, inequalities=(rows,), nested_at=nested_at
    )
    start = np.array([2.0, 0.5, 0.5])
    fit = small_fit(_linear_toy("rows_minimum_toy", m).with_start(start), optimizer)
    # The minimum with source_a = 1 held: the rate and source_b rows of H (z - m) vanish.
    free = [0, 2]
    reduced = _TOY_H[np.ix_(free, free)]
    rest = m[free] - np.linalg.solve(reduced, _TOY_H[free, 1] * (1.0 - m[1]))
    expected = np.array([rest[0], 1.0, rest[1]])
    # measured 4.4e-16 worst difference, joint and nested (0.58 and 0.26 before the active-set
    # method), workstation CPU 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py -k closed_form
    np.testing.assert_allclose(fit.params, expected, rtol=0.0, atol=1e-10)


def test_a_start_off_the_equality_rows_moves_onto_them_and_ends_at_the_constrained_minimum():
    """With the row rate = 1 and the start at the unconstrained minimum, off the row by 2, the fit
    does not return the start: it ends at the minimum on the row."""
    from diff_soup.optimization.inference.statistics import LinearConstraint

    m = np.array([3.0, 1.1, 0.0])
    row = LinearConstraint({"rate": 1.0}, rhs=1.0)
    optimizer = inf.TrustRegion(max_steps=40, radius_init=100.0, constraints=(row,))
    fit = small_fit(_linear_toy("equality_start_toy", m).with_start(m), optimizer)
    # The minimum with rate = 1: each source moves by -H[source, rate] (1 - m_rate) / H[s, s].
    expected = np.array([1.0, m[1] + 0.9 * (1.0 - m[0]), m[2] + 0.9 * (1.0 - m[0])])
    # measured 4.4e-16 worst difference (2.0 when the start was returned), workstation CPU
    # 2026-11-09, pytest tests/contracts/test_lowered_study.py -k equality_rows
    np.testing.assert_allclose(fit.params, expected, rtol=0.0, atol=1e-10)


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


def test_bootstrap_refuses_a_solver_with_inequality_rows(lowered, joint_fit):
    """The replicates step without inequality rows, so a solver= that carries them is refused."""
    _, study, _ = lowered
    rows = inf.InequalityRows(lambda own, w, d: own[:1] - 0.02, level="cell")
    solver = inf.TrustRegion(inequalities=(rows,))
    with pytest.raises(ds.ConfigurationError, match="inequalities"):
        inf.bootstrap(study, joint_fit, replicates=2, key=jax.random.key(3), steps=1, solver=solver)


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


def _two_owners(name):
    """Two owners sharing ``rate``, each with its own ``source``, under priors on both."""
    from tests.support.inference import NOISE, decay_experiment

    decay = decay_recipe(
        name,
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.5),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.2),
    )
    cells = CELLS[:2]
    return inf.Study(
        tuple(
            decay_experiment(decay, owner, source=source, noise=(index + 1) * NOISE)
            for index, (owner, source) in enumerate(cells)
        ),
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": tuple(owner for owner, _ in cells)},
        independent=True,
    )


def test_nested_attempt_records_split_the_cost_by_owner():
    """A nested fit's record of every attempt holds each owner's cost at the current point and at
    the trial, and the shared remainder; at every attempt they sum to the fit's cost there."""
    records = _Records()
    fit = small_fit(
        ds.lower(_two_owners("owner_cost_decay"), objective=inf.MAP()),
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


@pytest.fixture(scope="module")
def frozen():
    """The two-owner study prepared with the frozen mode, a route, and blocks at the start."""
    study = ds.lower(_two_owners("frozen_decay"), objective=inf.MAP())
    study.prepare(modes=("value", "frozen"))
    route = _LaneRoute(inf.TrustRegion(), study)
    point = np.asarray(study.x0, dtype=np.float64)
    return route, point, route.evaluate("blocks", point)


def _kept_times_rows(route, lane, kept_at, point):
    """``(J^T r, |r|^2 / 2)`` of one lane, unbatched: ``J`` its rows' Jacobian at ``kept_at`` in
    narrow coordinates, by forward mode, and ``r`` its rows at ``point``."""
    lanes = route.problem._lanes
    bucket, position = lanes.where[lane]
    entry = lanes.buckets[bucket]
    leaves = jax.device_put([leaf[position] for leaf in entry.leaves])
    inputs = jax.tree_util.tree_unflatten(entry.tree, leaves)
    columns = entry.gather[position]

    def rows(narrow):
        return _solve(entry.program, narrow, inputs)[0]

    jacobian = jax.jacfwd(rows)(jax.device_put(np.asarray(kept_at)[columns]))
    new = rows(jax.device_put(np.asarray(point)[columns]))
    return np.asarray(jacobian.T @ new), float(0.5 * new @ new)


def test_the_frozen_mode_at_the_blocks_point_gives_the_blocks_gradient_and_system(frozen):
    """At the point of the last blocks request a frozen request of every lane, or of one, gives
    the blocks gradient and cost, and the frozen system there is the blocks system."""
    route, point, blocks = frozen
    every = route.evaluate("frozen", point)
    one = route.evaluate("frozen", point, lanes=route.ids[1:])
    # measured 0.0 on every entry, workstation CPU 2026-11-09,
    # pytest tests/contracts/test_lowered_study.py -k frozen
    np.testing.assert_allclose(every.gradient, blocks.gradient, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(every.cost, blocks.cost, rtol=1e-12, atol=0.0)
    # measured 0.0, as above
    np.testing.assert_allclose(one.gradient[0, 0], blocks.gradient[0, 1], rtol=1e-12, atol=0.0)
    system = route.system(blocks, point)
    again = route.frozen_system(system, every, point)
    for ours, expected in zip(route.vectors(again), route.vectors(system)):
        # measured 0.0, as above
        np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0)
    # measured 0.0 relative, as above
    np.testing.assert_allclose(float(again.cost), float(system.cost), rtol=1e-12)


def test_the_frozen_mode_at_a_moved_point_steps_on_the_kept_jacobian(frozen):
    """At a moved point a frozen request of every lane, and of each lane alone, gives each lane's
    cost there and J^T r with J its Jacobian at the blocks point and r its rows there."""
    route, point, _ = frozen
    moved = point + np.asarray([0.03, -0.02, 0.01])
    every = route.evaluate("frozen", moved)
    for column, lane in enumerate(route.ids):
        gradient, cost = _kept_times_rows(route, lane, point, moved)
        alone = route.evaluate("frozen", moved, lanes=(lane,))
        # measured at most 1.5e-14 relative on the gradient and 2.9e-14 on the cost (batched
        # lanes against one unbatched solve), workstation CPU 2026-11-09, -k frozen
        np.testing.assert_allclose(every.gradient[0, column], gradient, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(alone.gradient[0, 0], gradient, rtol=1e-12, atol=0.0)
        # measured as above
        np.testing.assert_allclose(every.cost[0, column], cost, rtol=1e-12)


def test_a_frozen_system_of_some_lanes_replaces_the_gradient_of_their_members_only(frozen):
    """A frozen system from a request of one owner's lanes takes that owner's gradient from a
    request of every lane and keeps every other gradient of the blocks system; its cost is NaN."""
    route, point, blocks = frozen
    moved = point + np.asarray([0.0, -0.02, 0.01])
    system = route.system(blocks, point)
    every = route.frozen_system(system, route.evaluate("frozen", moved), moved)
    some = route.frozen_system(system, route.evaluate("frozen", moved, lanes=route.ids[1:]), moved)
    # measured 0.0, workstation CPU 2026-11-09, -k frozen
    np.testing.assert_allclose(some.gradient[1][1], every.gradient[1][1], rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(some.gradient[1][0], system.gradient[1][0])
    np.testing.assert_array_equal(some.root_gradient, system.root_gradient)
    assert np.isnan(float(some.cost)) and np.isfinite(float(every.cost))


def test_blocks_keep_no_jacobian_unless_the_frozen_mode_is_prepared(frozen):
    """Without the frozen mode prepared a blocks request keeps nothing and a frozen request is
    refused; with it, a lane that has no blocks result yet is refused by name."""
    # A study of its own: a nested fit with a polish prepares the frozen mode on its study.
    study = ds.lower(_two_owners("unkept_decay"), objective=inf.MAP())
    study.prepare()
    route = _LaneRoute(inf.TrustRegion(), study)
    point = np.asarray(study.x0, dtype=np.float64)
    route.evaluate("blocks", point)
    assert route.problem._lanes.kept == {}
    with pytest.raises(ValueError, match="prepare"):
        route.evaluate("frozen", point)
    fresh = _LaneRoute(inf.TrustRegion(), frozen[0].problem)
    with pytest.raises(ValueError, match="no kept Jacobian"):
        fresh.evaluate("frozen", frozen[1])


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


def test_a_member_with_non_finite_sensitivities_at_a_finite_cost_retreats_in_the_polish():
    """A member whose lanes solve at a finite cost but give non-finite Gauss-Newton blocks at its
    polished point has its lanes recorded as failed: it retreats along its polish step and goes
    back to where its lanes solved, at a finite cost; at an accepted step such blocks end the fit
    with an error naming the member."""
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
    lowered = ds.lower(study, objective=inf.MAP())
    fit = small_fit(
        lowered, inf.TrustRegion(max_steps=0, nested_at="cell", inner_start_iterations=2)
    )
    assert np.isfinite(fit.cost) and int(fit.work["inner_retreats"][1]) > 0
    assert {str(kind) for kind in fit.work["failure_kind"]} == {"nonfinite_blocks_at_finite_cost"}
    assert all(fit.work["failure_inner"])
    with pytest.raises(ValueError, match="cell-b"):
        small_fit(lowered, inf.TrustRegion(max_steps=6, nested_at="cell"))


def _banded_toy(name, band, hump=None):
    """The linear two-owner toy of ``_linear_toy`` (m = (3, 2, 1.1)) on one model per owner:
    cell-a's lanes have infinite sensitivities at a finite cost while its source lies in ``band``,
    and with ``hump`` a narrow rise of its cost there; cell-b's lanes are the plain toy's."""
    import jax.numpy as jnp

    from diff_soup.domains import cartesian_grid_1d
    from tests.support.inference import TIMES, solver

    scale, m, times = 0.02, np.array([3.0, 2.0, 1.1]), jnp.asarray(TIMES)

    def a(t):
        return scale / jnp.sqrt(5.0) * jnp.ones_like(t)

    def b(t):
        return scale * (-0.9 / jnp.sqrt(5.0) + jnp.sqrt(0.19) * (4.0 * t - 2.0) / jnp.sqrt(10.0))

    def observed(q, banded):
        value = q.p.rate * a(q.t) + q.p.source * b(q.t)
        if not banded:
            return value
        if hump is not None:
            value = value + 3.0 * scale * jnp.exp(-(((q.p.source - hump) / 0.05) ** 2))
        inside = (q.p.source > band[0]) & (q.p.source < band[1])
        # sqrt(0) adds nothing to the value and an infinite derivative inside the band
        return value + jnp.where(inside, jnp.sqrt(value - jax.lax.stop_gradient(value)), 0.0)

    experiments, handles = [], []
    for i, owner in enumerate(("cell-a", "cell-b")):
        spec = ds.problem(f"{name}_{owner}")
        spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
        handles.append(
            (
                spec.param("rate", value=0.9, learnable=True),
                spec.param("source", value=0.0, learnable=True),
            )
        )
        spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
        spec.equation(writes="u", reads=("u",))(lambda q: -0.5 * q.u)
        spec.observe.scalar(name="exact", fn=lambda q, banded=i == 0: observed(q, banded))
        values = m[0] * a(times) + m[1 + i] * b(times)
        measurement = inf.Measurement(
            "exact", times=times, values=values, observe="exact", likelihood=inf.Gaussian(scale)
        )
        experiments.append(
            inf.Experiment(
                model=spec.build(),
                solver=solver(),
                t_span=(0.0, 1.0),
                initial_state=spec.assemble_state({}),
                measurements=(measurement,),
                id=owner,
            )
        )
    rate, source = handles[0]
    parameters = ds.ParameterSet(shared=(rate,), varying={"cell": (source,)})
    study = inf.Study(
        tuple(experiments),
        parameters=parameters,
        clusters={"cell": ("cell-a", "cell-b")},
        independent=True,
    )
    # cell-b starts at its minimum for the start rate 2: only cell-a moves in the start polish
    return ds.lower(study).with_start(np.array([2.0, 0.5, 0.2]))


def test_a_member_whose_retreat_raises_its_cost_goes_back_to_where_its_polish_started():
    """cell-a's start polish reaches its minimum 1.1 for the start rate, inside the band, and
    retreats halfway, to the hump at 0.8, where its cost is above its start's: it goes back to its
    start 0.5 rather than keep the retreated point."""
    lowered = _banded_toy("retreat_hump_toy", band=(1.0, 1.2), hump=0.8)
    solver = inf.TrustRegion(
        max_steps=0, nested_at="cell", inner_start_iterations=1, radius_init=100.0
    )
    fit = small_fit(lowered, solver)
    assert int(fit.work["inner_retreats"][0]) == 1
    assert float(fit.params[1]) == 0.5


def test_a_member_that_goes_back_to_its_start_takes_its_next_step_on_the_jacobian_there():
    """cell-a's start polish and each of its retreats land inside the band, so it goes back to its
    start; it is evaluated again there, and at every later attempt every member's stationarity
    number is finite, as its Jacobian of the abandoned point would not make it."""
    records = _Records()
    lowered = _banded_toy("retreat_band_toy", band=(0.55, 1.2))
    solver = inf.TrustRegion(
        max_steps=3, nested_at="cell", inner_start_iterations=1, radius_init=100.0
    )
    fit = small_fit(lowered, solver, callbacks=(records,))
    assert int(fit.work["inner_retreats"][0]) == 3
    numbers = np.array([info.member_stationarity for info, _ in records.steps], dtype=float)
    assert numbers.size and np.all(np.isfinite(numbers))


def test_trust_region_takes_a_bare_constraint_or_gauge_as_a_sequence_of_one():
    """constraints= and gauges= take one LinearConstraint or Gauge as a sequence of one."""
    from diff_soup.optimization.inference.statistics import LinearConstraint

    row, gauge = LinearConstraint({"rate": 1.0}, rhs=1.0), inf.Gauge.pin("rate", 1.0)
    solver = inf.TrustRegion(constraints=row, gauges=gauge)
    assert (solver.constraints, solver.gauges) == ((row,), (gauge,))


def test_a_replicate_without_a_step_is_kept_where_it_needs_none(stop, sigma, weights, status):
    """A bootstrap replicate that ends without an accepted step from a converged fit is kept as
    converged when its point passes the converged test or its weights are the original sample's
    (the fit is then its own answer); otherwise it fails with no_step."""
    from types import SimpleNamespace

    from diff_soup.optimization.inference.hierarchical._replicates import _Member, _Settings

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
