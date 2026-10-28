"""Optimizer stop rules: termination statuses, relative stops, stalls, StopFit, checkpoints."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d
from tests.support.inference import (
    NOISE,
    TIMES,
    TRUE_RATE,
    decay_experiment,
    decay_recipe,
    small_fit,
    solver,
)


@pytest.fixture(scope="module")
def decay():
    return decay_recipe("stops_decay")


@pytest.fixture(scope="module")
def problem(decay):
    return inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)


@pytest.fixture(scope="module")
def straight(problem):
    return small_fit(problem, inf.LevenbergMarquardt(max_steps=30))


def test_termination_statuses_name_their_reason_and_whether_they_converged():
    """Every TerminationStatus has its documented value and convergence flag."""
    table = {status.name: (status.value, status.converged) for status in inf.TerminationStatus}
    assert table == {
        "SUCCESS": ("success", True),
        "GRAD_TOL": ("gradient_tolerance", True),
        "STEP_TOL": ("step_tolerance", True),
        "COST_TOL": ("cost_tolerance", True),
        "DERIVATIVE_ACCURACY": ("derivative_accuracy", True),
        "ACCEPTED_STEPS": ("accepted_steps_reached", False),
        "MAX_STEPS": ("max_steps_reached", False),
        "SOLVER_FAILED": ("pde_solver_failed", False),
        "DIVERGED": ("optimizer_diverged", False),
        "STOPPED": ("stopped_by_callback", False),
        "UNKNOWN": ("unknown", False),
        "STALLED": ("stalled", False),
    }


def test_the_cost_change_stop_does_not_depend_on_residual_units(decay, straight):
    """Rows 1e5 times smaller (sigma 2000, cost far below one) end at the same rate."""
    small_rows = inf.InferenceProblem(
        decay_experiment(decay, likelihood=inf.Gaussian(scale=2000.0)), parameters=decay.rate
    )
    fit = small_fit(small_rows, inf.LevenbergMarquardt(max_steps=30, grad_tol=0.0))
    assert fit.converged
    # measured 1.6e-13 difference to the sigma-0.02 fit, both 2.8e-6 off the truth,
    # CPU VM 2026-10-22, pytest tests/public_api/test_optimizer_stops.py
    np.testing.assert_allclose(fit.params, straight.params, atol=1e-7)


# A small misfit on six frames, so that the optimum's cost is not zero and the stationarity
# test, not the cost test, ends the fit.
_WOBBLE = np.asarray([0.0, 3e-3, -2e-3, 4e-3, -1e-3, 2e-3])


def _whitened_study(decay, sigma):
    """One decay experiment on six frames whose rows are whitened by ``sigma``."""
    experiment = decay_experiment(
        decay,
        rate=0.65,
        times=np.linspace(0.0, 1.0, 6),
        noise=_WOBBLE,
        likelihood=inf.Gaussian(scale=sigma),
        # the solver's tolerances, under which the numbers of the test below were measured
        rtol=1e-10,
        atol=1e-12,
    )
    return ds.lower(inf.Study(experiment, parameters=ds.ParameterSet(shared=(decay.rate,))))


def test_trust_region_stationarity_stop_does_not_depend_on_residual_units():
    """Lowered-study fits whitened by sigma 1e-4, 1 and 1e4 (costs 1.4e3, 1.4e-5 and 1.4e-13)
    all end GRAD_TOL by the stationarity test at the same rate; sigma 1 and 1e4 take the same
    steps to the same stationarity number."""
    decay = decay_recipe("stops_units", rate=0.3, rate_bounds=(0.05, 1.5))
    # Every study is held until all are fitted: equal buckets then share one compiled program.
    studies = [_whitened_study(decay, sigma) for sigma in (1e-4, 1.0, 1e4)]
    large, unit, tiny = (small_fit(study, inf.TrustRegion()) for study in studies)
    ends = {(fit.status, fit.trust_region.stop) for fit in (large, unit, tiny)}
    assert ends == {(inf.TerminationStatus.GRAD_TOL, "stationarity")}
    rates = [np.asarray(fit[decay.rate]).item() for fit in (large, unit, tiny)]
    # measured 8.94e-8 spread (0.6473395258, 0.6473394365, 0.6473394365), CPU VM 2026-10-22,
    # pytest tests/public_api/test_optimizer_stops.py
    assert np.ptp(rates) <= 2e-7
    # measured 5 steps each (sigma 1e-4: 9), CPU VM 2026-10-22
    assert unit.n_steps == tiny.n_steps == 5
    # The number is near the square of the distance to the minimizer, so it magnifies any
    # difference of the end points: none on CPU; on a GPU the two whitenings give first steps one
    # ulp apart, which the solves at rtol 1e-10 carry to 5.0e-13 at the end point.
    # measured 2.4e-12 relative (2.167e-10 both), CPU VM 2026-10-22; 2.8e-6 (2.167025e-10 and
    # 2.167019e-10) measured on an RTX 4070 Ti and an L40S 2026-10-23,
    # pytest tests/public_api/test_optimizer_stops.py
    np.testing.assert_allclose(
        tiny.trust_region.stationarity, unit.trust_region.stationarity, rtol=1e-5
    )


def test_a_newton_fit_whose_radius_collapses_short_of_the_minimum_ends_stalled():
    """At a kink of |x0| the trust radius collapses before x1 reaches its minimum at 0.5, and the
    fit ends STALLED, not converged, with the reason radius_collapsed."""
    problem = inf.LeastSquaresProblem.from_residual(
        jnp.asarray([0.3, 0.2]),
        lambda x: jnp.stack([jnp.abs(x[0]) + 1.0, x[0] + 2.0 * x[1] - 1.0]),
    )
    fit = small_fit(problem, inf.NewtonTrustRegion(hessian="gauss-newton"))
    assert (fit.status, fit.converged, fit.info["stall_reason"]) == (
        inf.TerminationStatus.STALLED,
        False,
        "radius_collapsed",
    )


def _growth(start, values, scale, observe):
    """``u' = rate`` from ``u(0) = 1``, observed through ``observe`` at the suite's times."""
    spec = ds.problem("stops_growth")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=start, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def growth(q):
        return q.p.rate * jnp.ones_like(q.u)

    experiment = inf.Experiment(
        model=spec.build(),
        solver=solver(),
        t_span=(0.0, 1.0),
        initial_state=spec.assemble_state({}),
        measurements=(
            inf.Measurement(
                "m",
                times=jnp.asarray(TIMES),
                values=jnp.asarray(values),
                observe=observe,
                likelihood=inf.Gaussian(scale=scale),
            ),
        ),
        id="growth",
    )
    return inf.InferenceProblem(experiment, parameters=rate)


def test_bfgs_classifies_a_stalled_line_search_at_the_optimum_as_converged():
    """At a stiff optimum with a large residual, a stalled line search ends STEP_TOL."""
    values = 1.0 + TRUE_RATE * TIMES + 10.0 * NOISE
    optimum = float(np.sum(TIMES * (values - 1.0)) / np.sum(TIMES**2))
    problem = _growth(optimum + 5e-10, values, 1e-4, lambda state: jnp.mean(state))
    fit = small_fit(problem, inf.BFGS(max_steps=20, grad_tol=1e-8, step_tol=1e-6, ad="forward"))
    assert fit.status is inf.TerminationStatus.STEP_TOL and fit.converged
    # measured 5.0e-10 (the start offset), CPU VM 2026-10-22,
    # pytest tests/public_api/test_optimizer_stops.py
    np.testing.assert_allclose(fit.params, [optimum], atol=1e-8)


def test_bfgs_reports_a_non_finite_objective_as_divergence():
    """A gradient that is not finite at the start ends BFGS DIVERGED, not converged."""
    problem = _growth(
        0.5, np.sqrt(TRUE_RATE * TIMES), 0.1, lambda state: jnp.sqrt(jnp.mean(state) - 1.0)
    )
    fit = small_fit(problem, inf.BFGS(max_steps=5, ad="forward"))
    assert fit.status is inf.TerminationStatus.DIVERGED and not fit.converged


def test_tree_map_over_a_result_maps_its_arrays_and_keeps_its_status(straight):
    """jax.tree.map over an OptResult doubles its point, cost and histories; status and counts stay."""
    doubled = jax.tree.map(lambda leaf: 2 * leaf, straight)
    np.testing.assert_array_equal(doubled.params, 2 * np.asarray(straight.params))
    np.testing.assert_array_equal(doubled.cost_history, 2 * np.asarray(straight.cost_history))
    assert doubled.cost == 2 * straight.cost
    assert doubled.status is straight.status and doubled.n_steps == straight.n_steps
    assert doubled.info is straight.info


def test_a_fit_out_of_steps_reports_max_steps(problem):
    """A fit that exhausts max_steps ends MAX_STEPS after exactly that many steps."""
    fit = small_fit(problem, inf.LevenbergMarquardt(max_steps=2))
    assert fit.status is inf.TerminationStatus.MAX_STEPS and fit.n_steps == 2


@pytest.mark.parametrize(
    "optimizer",
    [
        inf.LevenbergMarquardt,
        functools.partial(
            inf.NewtonTrustRegion, hessian="true", composition="forward-over-forward"
        ),
        functools.partial(inf.NewtonTrustRegion, hessian="gauss-newton"),
        functools.partial(inf.BFGS, ad="forward"),
        functools.partial(inf.GradientDescent, ad="forward"),
    ],
    ids=("levenberg_marquardt", "newton_true", "newton_gauss_newton", "bfgs", "gradient_descent"),
)
def test_max_steps_zero_returns_the_start_point_and_a_negative_budget_is_refused(
    problem, straight, optimizer
):
    """max_steps=0 returns the start point, its cost and MAX_STEPS; max_steps=-1 is refused."""
    fit = small_fit(problem, optimizer(max_steps=0))
    assert fit.status is inf.TerminationStatus.MAX_STEPS and fit.n_steps == 0
    np.testing.assert_array_equal(fit.params, straight.initial_params)
    # another program's autotuned cost: measured 1 ulp off, 7 of 30 runs, RTX 4070 Ti 2026-10-24
    np.testing.assert_allclose(fit.cost, straight.cost_history[0], rtol=1e-15)
    with pytest.raises(ds.ConfigurationError, match="max_steps must be an integer >= 0"):
        small_fit(problem, optimizer(max_steps=-1))


class _StopAtFirstAccepted(inf.Callback):
    def on_step(self, info):
        if info.accepted:
            raise inf.StopFit()


@pytest.fixture(
    scope="module",
    params=[
        functools.partial(inf.LevenbergMarquardt, max_steps=30),
        functools.partial(inf.NewtonTrustRegion, hessian="gauss-newton"),
    ],
    ids=("levenberg_marquardt", "newton_gauss_newton"),
)
def stopped(problem, tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("stops_checkpoint")
    fit = small_fit(
        problem, request.param(), callbacks=(_StopAtFirstAccepted(),), checkpoint=directory
    )
    return directory, fit, request.param


def test_a_fit_stopped_by_stop_fit_resumes_from_its_checkpoint_to_the_same_optimum(
    problem, straight, stopped
):
    """StopFit ends a fit STOPPED; checkpoint=directory continues it from its step file to the
    straight fit's point."""
    directory, fit, optimizer = stopped
    assert fit.status is inf.TerminationStatus.STOPPED and fit.n_accepted == 1
    assert [path.name for path in sorted(directory.glob("step-*.npz"))] == ["step-000000.npz"]
    resumed = small_fit(problem, optimizer(), checkpoint=directory)
    np.testing.assert_array_equal(resumed.initial_params, fit.params)
    assert resumed.converged
    # measured 5.1e-12 relative difference (Levenberg-Marquardt), 3.0e-11 (Gauss-Newton),
    # workstation CPU 2026-10-28, pytest tests/public_api/test_optimizer_stops.py
    np.testing.assert_allclose(resumed.params, straight.params, rtol=1e-10)


def test_a_checkpoint_of_another_parameter_layout_is_refused(stopped):
    """optsolve(checkpoint=...) refuses a directory a fit of other parameters wrote."""
    directory, _fit, optimizer = stopped
    decay = decay_recipe("stops_two_parameters", source=True)
    other = inf.InferenceProblem(decay_experiment(decay), parameters=(decay.rate, decay.source))
    with pytest.raises(ValueError, match=r"laid out as \(rate: 0\), this problem's as"):
        small_fit(other, optimizer(), checkpoint=directory)


def test_log_callback_writes_one_line_per_step_to_its_file(problem, tmp_path):
    """LogCallback(log_file=...) records every optimizer step in the file."""
    path = tmp_path / "fit.log"
    fit = small_fit(
        problem, inf.LevenbergMarquardt(max_steps=3), callbacks=(inf.LogCallback(str(path)),)
    )
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) >= fit.n_steps
