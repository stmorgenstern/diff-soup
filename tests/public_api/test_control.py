"""Control: a control problem over a schedule solve, its numeric problem and its derivative in
the initial state and limits, signal controls lowered exactly with free durations, the receding
horizon under scan and from the host, the estimators."""

from __future__ import annotations

import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.optimization.inference import approx
from tests.support.problems import operational_cell

control = ds.optimization.control

# The charging toy: four intervals of a quarter, the mean of the four states driven to a
# charge, the cost the mean square of the currents.
N, D = 4, 0.25
TIMES = np.linspace(D / 4, N * D, 4 * N)
# Tolerances under which the interior point is measured at 1.2e-9 off the closed form,
# workstation CPU 2026-11-06, pytest tests/public_api/test_control.py.
SOLVER = ds.JIDA(linear="dense", rtol=1e-11, atol=1e-13)  # measured: see above
IP = ds.optimization.InteriorPoint(tol=1e-10)  # measured: see above
STARTS_AT = 2.0
CHARGE = 2.5
# Signal controls: a clamped cubic B-spline current of six control points on three segments
# of a third, sampled at fractions of every segment.
MODES = ("current", "voltage", "power")
FRACTIONS = (0.25, 0.5, 1.0)


def _state(level):
    """The toy's state with every particle at `level`, batched along `level`'s shape."""
    level = jnp.asarray(level, jnp.float64)[..., None]
    shape = level.shape[:-1]
    return jnp.concatenate([jnp.broadcast_to(level, (*shape, 4)), jnp.zeros((*shape, 1))], -1)


def _terminal(solution, u, args):
    return jnp.mean(solution.ys[-1]) - args[2]["charge"]


def _voltage(state, theta):
    return jnp.mean(state[:4]) - 2.0 * state[4]


def _duration(solution, u, args):
    return u[-1]


def _spline_current(free_durations=False):
    """A B-spline current over [0, 1], or over a free duration."""
    spline = approx.BSpline(np.linspace(0.0, 1.0, 4), np.zeros(6))
    return control.SignalControls([(spline, "current", 1.0)], free_durations=free_durations)


def _hold():
    """One constant segment of a half, a block of held current or voltage."""
    return approx.PiecewiseConstant([0.0, 0.5], [0.0])


def _signal(kind, knots):
    """A piecewise signal of `kind` on `knots` with random coefficients."""
    rng, n = np.random.default_rng(0), len(knots)
    if kind == "PiecewiseConstant":
        return approx.PiecewiseConstant(knots, rng.normal(size=n - 1))
    if kind == "PiecewiseLinear":
        return approx.PiecewiseLinear(knots, rng.normal(size=n))
    if kind == "CubicHermite":
        return approx.CubicHermite(knots, rng.normal(size=n), rng.normal(size=n))
    return approx.BSpline(knots, rng.normal(size=n + 2))


def _closed_form(level, charge):
    """The optimal currents: the charge left after the decay of `level`, along the decay weights."""
    edges = np.arange(N + 1) * D
    weights = np.exp(-(N * D - edges[1:])) - np.exp(-(N * D - edges[:-1]))
    return (charge - level * np.exp(-N * D)) * weights / (weights @ weights)


@pytest.fixture(scope="module")
def toy():
    """The toy's model, its cost, and the plant one interval at a time."""
    model = operational_cell().build()

    def cost(solution, u, args):
        return jnp.mean(u**2)

    @jax.jit
    def advance(state, u):
        schedule = control.PiecewiseConstant(mode="current", intervals=1, duration=D).segments(
            jnp.atleast_1d(u), tuple(model.operational_modes())
        )
        solution, _ = ds.solve_schedule(
            model, SOLVER, schedule, state, t0=0.0, t_end=D, times=jnp.asarray([D])
        )
        return jnp.concatenate([solution.terminal.y, solution.terminal.lam])

    def plant(state, u, interval):
        new = advance(state, u)
        return new, new

    return types.SimpleNamespace(model=model, cost=cost, advance=advance, plant=plant)


def charging(toy, level=STARTS_AT, charge=CHARGE):
    """The toy's control problem from `level` to `charge`, both of which may be batched."""
    return control.ControlProblem(
        problem=toy.model,
        controls=control.PiecewiseConstant(mode="current", intervals=N, duration=D),
        cost=toy.cost,
        terminal=_terminal,
        n_terminal=1,
        bounds=(0.0, 5.0),
        save_times=TIMES,
        initial_state=_state(level),
        limits={"charge": jnp.asarray(charge, jnp.float64)},
        solver=SOLVER,
    )


@eqx.filter_jit
def _solve(problem):
    """A control problem's currents by the interior point, one compiled program per configuration."""
    numeric = problem.numeric_problem(IP)
    return IP.run(numeric, numeric.x0, numeric.args)[0]


@pytest.fixture(scope="module")
def closed_loop(toy):
    return control.receding_horizon(charging(toy), toy.plant, control.FullState(), 3, solver=IP)


def test_a_last_save_time_at_the_grids_end_is_the_horizons_end():
    """A save time at the end of a grid of durations scaled by a reference time is the
    horizon's end, though the durations sum to one rounding short of it."""
    # Three intervals of a (k/3)**0.25 grid over 1 s, in units of 0.3 s: the grid's end,
    # 3.3333333333333335, lies one ulp past its durations added in order, 3.333333333333333.
    grid = np.linspace(0.0, 1.0, 4) ** 0.25
    problem = control.ControlProblem(
        problem=operational_cell().build(),
        controls=control.PiecewiseConstant(
            mode="current", intervals=3, duration=np.diff(grid) / 0.3
        ),
        cost=lambda sol, u, args: jnp.sum(sol.ys[-1]),
        save_times=grid[1:] / 0.3,
        initial_state=jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))]),
        solver=ds.JIDA(linear="dense"),
    )
    numeric = problem.numeric_problem(ds.optimization.SQP())
    assert np.isfinite(numeric.value(jnp.ones(3)))


def test_the_interior_point_reaches_the_charging_problems_closed_form(toy):
    """The interior point on the toy's control problem ends at its closed-form currents."""
    result = ds.optimization.optsolve(charging(toy), IP)
    # measured 1.2e-9 off the closed form, workstation CPU 2026-11-06,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(result.params, _closed_form(STARTS_AT, CHARGE), rtol=0, atol=1e-8)


def test_the_currents_differentiate_in_the_initial_state_and_the_limit(toy):
    """jax.jacfwd of the interior point's currents in the starting level and the charge limit,
    through the implicit rule over the schedule solve, is the closed form's."""
    point = jnp.asarray([STARTS_AT, CHARGE])
    jacobian = jax.jacfwd(lambda p: _solve(charging(toy, level=p[0], charge=p[1])))(point)
    base = _closed_form(0.0, 0.0)
    expected = np.stack([_closed_form(1.0, 0.0) - base, _closed_form(0.0, 1.0) - base], axis=1)
    # measured 6.6e-10 off the closed form's derivatives, workstation CPU 2026-11-07,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(jacobian, expected, rtol=0, atol=1e-8)


def test_a_batch_of_control_problems_is_one_vmap(toy):
    """Control problems batched along their initial states and limits solve as one `vmap`,
    each lane at its own closed form."""
    batch = charging(toy, level=[STARTS_AT, 1.8], charge=[CHARGE, 2.4])
    expected = np.stack([_closed_form(STARTS_AT, CHARGE), _closed_form(1.8, 2.4)])
    # measured 3.2e-9 off the closed forms, workstation CPU 2026-11-06,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(eqx.filter_vmap(_solve)(batch), expected, rtol=0, atol=1e-8)


def test_the_host_loop_applies_the_scan_loops_moves(toy, closed_loop):
    """`HostLoop` drives a plant on the host through the same horizons as the scan."""

    def plant(state, u, interval):
        new = np.asarray(toy.advance(jnp.asarray(state), jnp.asarray(u)))
        return new, new

    host = control.receding_horizon(
        charging(toy), None, control.FullState(), 3, solver=IP, driver=control.HostLoop(plant)
    )
    # measured 0.0 between the two drivers' moves, workstation CPU 2026-11-06,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(host.controls, closed_loop.controls, rtol=0, atol=1e-10)


def test_the_closed_loop_differentiates_in_the_limit_under_its_warm_start(toy, closed_loop):
    """jax.jacfwd of the receding horizon's moves in the charge, under the default warm start
    that starts each horizon from the last one's solution and record, is the central
    difference's."""

    def moves(charge, ip=ds.optimization.InteriorPoint(tol=1e-8)):  # measured floor 1.1e-10, GPU
        problem = charging(toy, charge=charge)
        return control.receding_horizon(problem, toy.plant, control.FullState(), 3, solver=ip)

    jacobian = jax.jacfwd(lambda charge: moves(charge).controls)(jnp.asarray(CHARGE))
    # The moves are affine in the charge (the closed form), so a wide difference is exact up
    # to the horizon solves' error.
    step = 0.1
    differences = (moves(CHARGE + step).controls - moves(CHARGE - step).controls) / (2 * step)
    # measured 1.8e-8, workstation CPU 2026-11-09, pytest tests/public_api/test_control.py
    np.testing.assert_allclose(jacobian, differences, rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", ["kalman", "moving_horizon"])
def test_an_estimator_corrects_a_wrong_belief_from_voltage_measurements(toy, kind):
    """Believing the particles at 1.6 when they are at 2.0, each estimator corrects its mean
    within three measured intervals."""
    problem = charging(toy, level=1.6)
    if kind == "kalman":
        estimator = control.ExtendedKalmanFilter(
            problem=problem,
            measured=_voltage,
            process_covariance=1e-8,
            measurement_covariance=1e-6,
            initial_covariance=0.1,
        )
    else:
        estimator = control.MovingHorizonEstimator(
            problem=problem,
            measured=_voltage,
            window=2,
            arrival_covariance=0.1,
            measurement_covariance=1e-6,
            basis=_state(1.0)[:, None],
            # The window fit's KKT error floors at 1.5e-8 here (the schedule solve's rtol
            # whitened by a measurement deviation of 1e-3), measured workstation CPU
            # 2026-11-06, so its tolerance sits above it.
            solver=ds.optimization.SQP(tol=1e-6),  # measured floor 1.5e-8, see above
        )
    update = eqx.filter_jit(estimator.estimate)
    state, truth = estimator.init(problem.args()), _state(2.0)
    for u in (1.0, 1.5, 0.5):
        truth = toy.advance(truth, jnp.asarray(u))
        state = update(state, jnp.asarray(u), _voltage(truth, None), problem.args())
    error = float(jnp.mean(state.plant[:4]) - jnp.mean(truth[:4]))
    # measured -6.5e-6 (Kalman) and -2.0e-11 (moving horizon, every fit at status 0) from
    # 0.4, workstation CPU 2026-11-06, pytest tests/public_api/test_control.py
    assert abs(error) < 1e-4


@pytest.mark.parametrize(
    "kind", ["PiecewiseConstant", "PiecewiseLinear", "CubicHermite", "BSpline"]
)
def test_signal_controls_lower_each_segment_of_a_signal_exactly(kind):
    """A signal's segments lower to schedule segments whose targets, evaluated as the schedule
    evaluates them, are the signal at every time."""
    knots = np.array([0.0, 0.7, 1.1, 2.0, 3.6, 4.0])
    signal = _signal(kind, knots)
    u = jnp.concatenate([jnp.ravel(getattr(signal, path)) for path in signal.learnable_paths()])
    arrays = control.SignalControls([(signal, "current", 1.0)]).segments(u, MODES)
    starts = jnp.asarray(np.concatenate([[0.0], np.cumsum(arrays.duration)[:-1]]))
    times = jnp.asarray(np.concatenate([np.linspace(0.0, 4.0, 401), knots]))
    segment = jnp.clip(jnp.searchsorted(knots, times, side="left") - 1, 0, len(knots) - 2)

    def target(k, t):
        terms = {}
        if arrays.terms is not None:
            terms = {"target_terms": arrays.terms[k], "target_origin": starts[k]}
        return ds.ControlState.operating(mode=0, target=arrays.target[k], **terms).target_at(t)

    # measured 5.6e-16 (CubicHermite; 1.1e-16 PiecewiseLinear, 0.0 the others), workstation
    # CPU 2026-11-09, pytest tests/public_api/test_control.py
    np.testing.assert_allclose(
        jax.vmap(target)(segment, times), jax.vmap(signal)(times), rtol=0, atol=1e-14
    )


def test_the_horizon_differentiates_in_spline_coefficients_and_a_free_duration(toy):
    """The cost, terminal and path rows of a horizon with a B-spline current over a free
    duration, sampled at fractions of every segment, differentiate as central differences do."""
    problem = control.ControlProblem(
        problem=toy.model,
        controls=_spline_current(free_durations=True),
        cost=lambda solution, u, args: jnp.mean(solution.ys**2),
        terminal=_terminal,
        n_terminal=1,
        path=lambda solution, u, args: jnp.mean(solution.ys, axis=1) - 2.0 * solution.lambdas[:, 0],
        n_path=1,
        save_fractions=FRACTIONS,
        initial_state=_state(STARTS_AT),
        limits={"charge": jnp.asarray(CHARGE, jnp.float64)},
        solver=SOLVER,
    )
    numeric = problem.numeric_problem(IP)
    u = jnp.asarray([1.0, 3.0, 2.0, 4.0, 2.5, 1.5, 0.8])
    _, gradient, _, _, terminal, path = jax.jit(numeric.evaluate_with_derivatives)(u)
    jacobian = np.concatenate([gradient[None], terminal, path])
    evaluate = jax.jit(numeric.evaluate)

    def rows(x):
        cost, equality, inequality = evaluate(x)
        return np.concatenate([[cost], equality, inequality])

    step = 1e-3
    differences = np.stack(
        [(rows(u.at[j].add(step)) - rows(u.at[j].add(-step))) / (2 * step) for j in range(u.size)],
        axis=1,
    )
    # measured 9.5e-8 of the largest entry (the duration column 3.1e-8), workstation CPU
    # 2026-11-09, pytest tests/public_api/test_control.py
    scale = np.max(np.abs(jacobian))
    np.testing.assert_allclose(differences, jacobian, rtol=0, atol=1e-6 * scale)


def test_a_current_then_voltage_problem_reaches_its_closed_form(toy):
    """A held current then a held voltage, a half each, the mean square of the two targets
    minimized to the charge: the voltage phase relaxes the mean towards minus the voltage, so the
    charge is linear in the targets and the optimum lies along its coefficients."""
    problem = control.ControlProblem(
        problem=toy.model,
        controls=control.SignalControls([(_hold(), "current", 1.0), (_hold(), "voltage", 1.0)]),
        cost=toy.cost,
        terminal=_terminal,
        n_terminal=1,
        bounds=(-10.0, 10.0),
        save_times=TIMES,
        initial_state=_state(STARTS_AT),
        limits={"charge": jnp.asarray(CHARGE, jnp.float64)},
        solver=SOLVER,
    )
    result = ds.optimization.optsolve(problem, IP)
    # The charge after m' = I - m, then m' = -(m + V) / 2: a I + b V + c.
    a, b = (1 - np.exp(-0.5)) * np.exp(-0.25), np.exp(-0.25) - 1
    expected = (CHARGE - STARTS_AT * np.exp(-0.75)) * np.array([a, b]) / (a * a + b * b)
    # measured 2.3e-9 off the closed form, workstation CPU 2026-11-09,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(result.params, expected, rtol=0, atol=1e-8)


def test_a_free_duration_reaches_the_minimum_charging_time(toy):
    """With the duration free and the spline's control points under a ceiling, the shortest
    charge holds the ceiling throughout: T = log((ceiling - level) / (ceiling - charge))."""
    ceiling = 5.0
    problem = control.ControlProblem(
        problem=toy.model,
        controls=_spline_current(free_durations=True),
        cost=_duration,
        terminal=_terminal,
        n_terminal=1,
        bounds=(np.r_[np.zeros(6), 0.05], np.r_[np.full(6, ceiling), 2.0]),
        save_fractions=(1.0,),
        initial_state=_state(STARTS_AT),
        limits={"charge": jnp.asarray(CHARGE, jnp.float64)},
        solver=SOLVER,
    )
    result = ds.optimization.optsolve(problem, IP)
    expected = np.log((ceiling - STARTS_AT) / (ceiling - CHARGE))
    # measured 7.0e-11 off the closed form, workstation CPU 2026-11-09,
    # pytest tests/public_api/test_control.py
    np.testing.assert_allclose(result.params[-1], expected, rtol=0, atol=1e-8)


def test_the_receding_horizon_refuses_free_durations(toy):
    """The receding horizon steps a grid of fixed intervals, so it refuses free durations."""
    problem = control.ControlProblem(
        problem=toy.model,
        controls=_spline_current(free_durations=True),
        cost=_duration,
        save_fractions=(1.0,),
        initial_state=_state(STARTS_AT),
        solver=SOLVER,
    )
    with pytest.raises(ds.ConfigurationError, match="free durations"):
        control.receding_horizon(problem, toy.plant, control.FullState(), 2, solver=IP)
