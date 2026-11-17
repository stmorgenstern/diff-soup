"""LinearPenalty: a linear penalty on nonnegative parameters, its exact zeros, and its weights.

The toy study measures every parameter directly: an experiment of member ``m`` integrates
``u' = rho_m`` and ``v' = level`` from zero and observes ``u`` and ``v`` at ``t = 1/2, 1``
with unit noise, against ``a_mk t`` and ``b t``. With ``T = sum t^2 = 5/4`` the objective is
``sum_k T (rho_m - a_mk)^2 / 2`` per member plus the levels' terms, and ``rho_m >= 0`` under
the penalty ``w * rho_m`` has the closed form ``max((T sum_k a_mk - w) / (T K + 1 / s^2), 0)``,
``1 / s^2`` the precision of a MAP prior ``N(0, s)`` and zero under MLE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.optimization import LeastSquaresProblem
from diff_soup.optimization import inference as inf
from tests.support.inference import small_fit, solver

#: The members of the level ``cohort`` and the data ``a_mk`` of their two experiments: A
#: ends above zero, B and D at zero, C starts at zero and leaves it.
DATA = {"A": (2.0, 2.4), "B": (0.1, 0.2), "C": (1.0, 1.2), "D": (-0.6, -0.2)}
#: The data of the levels, one per experiment.
LEVELS = (0.3, -0.2, 0.7, 0.1, -0.4, 0.5, 0.2, -0.1)
#: The prior's standard deviation on every parameter.
PRIOR_SCALE = 10.0
#: The observation times, and the sum of their squares.
TIMES = (0.5, 1.0)
SQUARES = sum(t * t for t in TIMES)


def _study():
    spec = ds.problem("linear_penalty_toy")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    prior = inf.NormalPrior(0.0, PRIOR_SCALE)
    rho = spec.param("rho", value=0.5, learnable=True, bounds=(0.0, None), prior=prior)
    level = spec.param("level", value=0.0, learnable=True, prior=prior)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))
    spec.field("v", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: q.p.rho + 0.0 * q.u)
    spec.equation(writes="v", reads=("v",))(lambda q: q.p.level + 0.0 * q.v)
    model = spec.build()

    def measured(name, value, index):
        return inf.Measurement(
            name,
            times=jnp.asarray(TIMES),
            values=value * jnp.asarray(TIMES),
            observe=lambda state: state[index],
            likelihood=inf.Gaussian(1.0),
        )

    experiments = []
    for member, values in DATA.items():
        for k, value in enumerate(values):
            experiments.append(
                inf.Experiment(
                    model=model,
                    solver=solver(),
                    t_span=(0.0, 1.0),
                    initial_state=spec.assemble_state({}),
                    measurements=(
                        measured("u", value, 0),
                        measured("v", LEVELS[len(experiments)], 1),
                    ),
                    id=f"{member}{k}",
                    parameters={"cohort": member},
                )
            )
    hierarchy = inf.Hierarchy(levels=("cohort",), per={"cohort": (rho,), "experiment": (level,)})
    return inf.Study(tuple(experiments), parameters=hierarchy), rho, level


def _columns(problem, handle) -> dict:
    """The column of every occurrence of ``handle``, by the member or experiment it belongs to."""
    columns = {}
    for occurrence in problem.parameter_schema.entry(handle).occurrences:
        owner = occurrence.owner
        label = owner[-1] if isinstance(owner, tuple) else owner
        columns[str(label)] = int(occurrence.columns[0])
    return columns


def _expected(weight: float, *, prior: bool) -> tuple:
    """The closed-form ``rho`` per member and ``level`` per experiment."""
    precision = 1.0 / PRIOR_SCALE**2 if prior else 0.0
    rho = {
        m: max((SQUARES * sum(a) - weight) / (SQUARES * len(a) + precision), 0.0)
        for m, a in DATA.items()
    }
    levels = [SQUARES * b / (SQUARES + precision) for b in LEVELS]
    return rho, levels


@pytest.fixture(scope="module")
def toy():
    """The toy study lowered under MLE and under MAP, and its handles."""
    study, rho, level = _study()
    return {
        "mle": ds.lower(study),
        "map": ds.lower(study, objective=inf.MAP()),
        "rho": rho,
        "level": level,
    }


def _fit(lowered, rho, weight, solver_=None, *, start_c_at_zero=True):
    problem = lowered.with_regularization(inf.LinearPenalty(weight, on="rho"))
    if start_c_at_zero:
        start = np.array(problem.x0, copy=True)
        start[_columns(problem, rho)["C"]] = 0.0
        problem = problem.with_start(start)
    return small_fit(problem, solver_ or inf.TrustRegion(max_steps=40))


def test_the_gauss_newton_closure_adds_the_linear_value_and_slope_and_no_curvature():
    """The Gauss-Newton statistics of a penalized problem add w . x_chosen and w, not a matrix."""
    a = jnp.asarray([1.0, -2.0, 0.5])
    penalty = inf.LinearPenalty(0.75).on_columns([0, 2])
    plain = LeastSquaresProblem.from_residual(jnp.zeros(3), lambda x: x - a)
    penalized = LeastSquaresProblem.from_residual(
        jnp.zeros(3), lambda x: x - a, regularization=penalty
    )
    x = jnp.asarray([0.3, 0.2, 0.1])
    value, gradient, matrix, _ = plain.closure("gauss_newton")(x, None)
    p_value, p_gradient, p_matrix, _ = penalized.closure("gauss_newton")(x, None)
    # measured an exact match, B200 2026-11-17
    np.testing.assert_allclose(float(p_value - value), 0.75 * (0.3 + 0.1), rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(p_gradient - gradient), [0.75, 0.0, 0.75])
    np.testing.assert_array_equal(np.asarray(p_matrix), np.asarray(matrix))


@pytest.mark.parametrize("weight", (0.5, 10.0), ids=("some-zero", "all-zero"))
def test_the_joint_trust_region_reaches_the_closed_form_with_exact_zeros(toy, weight):
    """Under MLE, rho_m = max(mean(a_m) - w / (2 T), 0): B and D end exactly at zero, C (started
    at zero) leaves it, and a large weight puts every rho exactly at zero."""
    fit = _fit(toy["mle"], toy["rho"], weight)
    expected, levels = _expected(weight, prior=False)
    columns = _columns(toy["mle"], toy["rho"])
    params = np.asarray(fit.params)
    for member, value in expected.items():
        if value == 0.0:
            assert params[columns[member]] == 0.0
        else:
            # measured 8.9e-16 relative off the closed form, B200 2026-11-17 (measure.py)
            np.testing.assert_allclose(params[columns[member]], value, rtol=1e-9)
    level_columns = _columns(toy["mle"], toy["level"])
    found = [params[level_columns[f"{m}{k}"]] for m in DATA for k in range(2)]
    # measured 5.6e-16 off the data, B200 2026-11-17 (measure.py)
    np.testing.assert_allclose(found, levels, atol=1e-9)


@pytest.mark.parametrize("nested_at", (None, "cohort", "experiment"))
def test_a_map_fit_joint_or_nested_reaches_the_closed_form_with_exact_zeros(toy, nested_at):
    """Under MAP with the prior N(0, 10) the joint and the nested trust regions reach
    max((T sum a_m - w) / (2 T + 1/100), 0), the zeros exactly."""
    fit = _fit(toy["map"], toy["rho"], 0.5, inf.TrustRegion(max_steps=40, nested_at=nested_at))
    expected, levels = _expected(0.5, prior=True)
    columns = _columns(toy["map"], toy["rho"])
    params = np.asarray(fit.params)
    assert [params[columns[m]] == 0.0 for m in DATA] == [v == 0.0 for v in expected.values()]
    # measured 7.8e-16 relative off the closed form, B200 2026-11-17 (measure.py)
    np.testing.assert_allclose(
        [params[columns[m]] for m in DATA], list(expected.values()), rtol=1e-9
    )
    level_columns = _columns(toy["map"], toy["level"])
    found = [params[level_columns[f"{m}{k}"]] for m in DATA for k in range(2)]
    # measured 5.6e-16 off the closed form, B200 2026-11-17 (measure.py)
    np.testing.assert_allclose(found, levels, atol=1e-9)


def test_without_a_penalty_the_fit_is_unchanged(toy):
    """with_regularization(None) fits exactly as the lowered study itself."""
    plain = small_fit(toy["map"], inf.TrustRegion(max_steps=40))
    again = small_fit(toy["map"].with_regularization(None), inf.TrustRegion(max_steps=40))
    np.testing.assert_array_equal(np.asarray(again.params), np.asarray(plain.params))
    assert again.cost == plain.cost


def test_a_new_weight_compiles_no_program(toy):
    """A third fit of one lowered study under another weight compiles nothing; a new program
    compiled after it is heard, so the listener hears compiles."""
    heard = []
    counting = [False]

    def listen(event, duration, **_):
        if counting[0] and "compile" in event:
            heard.append(event)

    jax.monitoring.register_event_duration_secs_listener(listen)
    _fit(toy["map"], toy["rho"], 10.0, start_c_at_zero=False)
    _fit(toy["map"], toy["rho"], 0.5, start_c_at_zero=False)
    counting[0] = True
    fit = _fit(toy["map"], toy["rho"], 2.0, start_c_at_zero=False)
    during_fit = list(heard)
    jax.jit(lambda x: 3.0 * x + 1.0)(jnp.ones(3))
    counting[0] = False
    assert fit.converged
    assert during_fit == [] and heard


def test_a_penalty_is_refused_where_it_would_be_dropped_or_unbounded(toy):
    """A Gauss-Newton solver of rows alone refuses the penalty; an unbounded parameter does too."""
    problem = toy["mle"].with_regularization(inf.LinearPenalty(0.5, on="rho"))
    with pytest.raises(ds.ConfigurationError, match="LinearPenalty has none"):
        small_fit(problem, inf.LevenbergMarquardt(max_steps=2))
    with pytest.raises(ValueError, match="no finite lower bound"):
        toy["mle"].with_regularization(inf.LinearPenalty(0.5, on="level"))
