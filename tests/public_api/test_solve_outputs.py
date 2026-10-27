"""Solve outputs and control: save grids and SaveAt, named outputs, terminal buffers, events
and affects, operational controls, schedules, and continuing a solve from its history."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.timestepping import Bisection, EventSet
from tests.support.problems import ConstrainedDAE, DecayODE, observed_recipe, operational_cell

_THETA = jnp.asarray(0.4)
#: Solver and root-finder settings (requested tolerances, not asserted bounds).
_TIGHT = {"rtol": 1e-7, "atol": 1e-9}
_ROOT = {"rtol": 1e-10, "atol": 1e-12}
_FINE = {"rtol": 1e-8, "atol": 1e-10}
_VDP = {"rtol": 1e-6, "atol": 1e-8}
_PROJECTED = ds.DenseOutputPolicy(mode="projected")
# measured worst event-time error 1.1e-9 (projected DAE cutoff; 1.0e-13 for a time event),
# CPU VM 2026-10-22, pytest tests/public_api/test_solve_outputs.py
_ROOT_BOUND = 1e-8
# measured 2.7e-13 in the terminal state at the roots 0.4 and 0.6, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_a_terminating_event_keeps_a_canonical_capacity_prefix
_DAE_ROOT_BOUND = 1e-11
# measured 1.2e-9 relative against the closed form, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_an_affect_without_controls_keeps_epoch_zero_and_restarts_the_solver
_AFFECT_BOUND = 1e-6
# measured 0.0 on the held rows and 7.6e-9 in the switch time, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_an_operational_switch_changes_the_row_and_owns_its_samples_by_epoch
_SWITCH_BOUND = 1e-7
# measured first step of the continued solve 1.5e-2 against the configured 1e-6, CPU VM
# 2026-10-22, pytest tests/public_api/test_solve_outputs.py::test_a_solve_given_its_history_continues_the_integrator
_CONTINUED_STEP = 1e-3
# measured 1.004 times the one solve's accepted steps, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_a_solve_given_its_history_continues_the_integrator
_COUNT_BOUND = 1.05
# measured 2.5e-6 against the one solve in the max norm, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_a_solve_given_its_history_continues_the_integrator
_END_BOUND = 1e-4


_ODE, _DAE = DecayODE(), ConstrainedDAE()


def _jndf():
    return ds.JNDF.stock(linear="dense", **_TIGHT)


def _time_stop(time, state, args, *, control=None, **kwargs):
    return time - control["stop"]


def _algebraic_cutoff(time, state, args, *, control=None, **kwargs):
    # On the manifold q + lam = p = exp(-0.4 t): the condition reads the local algebraic
    # field and the algebraic global, and crosses ``level`` at t = ln(1 / level) / 0.4.
    return state[1] + state[2] - control["level"]


#: Events are static solve configuration: one object per declaration keeps one program.
_TIME_STOP = ds.Event(_time_stop, root_finder=Bisection(**_ROOT), direction="up")
_CUTOFF = ds.Event(
    _algebraic_cutoff, root_finder=Bisection(**_ROOT), direction="down", state_policy=_PROJECTED
)
_GRID = ds.SaveAt(ts=jnp.asarray([0.0, 0.25, 0.5, 0.75]))


def _buffer_solve(stop):
    """The DAE stopped by the projected cutoff whose root is at ``stop``."""
    return ds.pdesolve(
        _DAE.problem,
        _jndf(),
        (0.0, 1.0),
        _DAE.initial_state(),
        theta=_THETA,
        control_state0={"level": jnp.exp(-0.4 * jnp.asarray(stop))},
        event=_CUTOFF,
        saveat=_GRID,
        max_steps=400,
    )


@pytest.fixture(scope="module")
def buffers():
    """Fixed-buffer solves of one compiled program, keyed by the event's root time."""
    return {stop: _buffer_solve(stop) for stop in (0.4, 0.6, 2.0)}


def test_an_explicit_grid_with_endpoint_flags_is_one_canonical_request():
    """``ts`` names interior times and ``t0``/``t1`` add the endpoints, the saved initial row is
    the initial state, and times equal after conversion to the solve dtype share one row."""
    case = _ODE
    both = ds.pdesolve(
        case.problem,
        _jndf(),
        (0.0, 0.05),
        case.initial_state(),
        theta=_THETA,
        saveat=ds.SaveAt(ts=jnp.asarray([0.01, 0.025]), t0=True, t1=True),
    )
    assert isinstance(both, ds.Solution) and bool(both.ok)
    np.testing.assert_array_equal(both.ts, [0.0, 0.01, 0.025, 0.05])
    np.testing.assert_array_equal(both.ys[0], case.initial_state())
    single = DecayODE(ds.Precision(compute_dtype="float32"))
    first = np.float64(0.025)
    merged = ds.pdesolve(
        single.problem,
        ds.JNDF.stock(linear="dense"),
        (0.0, 0.05),
        single.initial_state(),
        theta=_THETA,
        saveat=ds.SaveAt(ts=jnp.asarray([first, np.nextafter(first, np.inf)])),
    )
    assert merged.ts.shape == (1,) and int(merged.valid_count) == 1


def test_save_requests_no_solve_can_serve_are_refused_before_integration():
    """Reserved output fields, ``fn``, a step plan with a grid, a step plan under the
    checkpoint adjoint and a non-positive stride are refused with a configuration error."""
    case = _ODE

    def solve(saveat, **keywords):
        return ds.pdesolve(
            case.problem,
            _jndf(),
            (0.0, 0.1),
            case.initial_state(),
            theta=_THETA,
            saveat=saveat,
            **keywords,
        )

    for field in ("solver_state", "controller_state", "made_jump"):
        with pytest.raises(ds.ConfigurationError, match=rf"SaveAt\.{field} is reserved"):
            solve(ds.SaveAt(t1=True, **{field: True}))
    for saveat, keywords, match in (
        (ds.SaveAt(t1=True, fn=lambda t, y, args: y[0]), {}, r"SaveAt\.fn .*outputs="),
        (ds.SaveAt(ts=jnp.asarray([0.02]), steps=True), {}, "cannot be combined"),
        (ds.SaveAt(steps=True), {"adjoint": ds.CheckpointAdjoint()}, "CheckpointAdjoint"),
        (ds.SaveAt(steps=0), {}, "positive integer"),
    ):
        with pytest.raises(ds.ConfigurationError, match=match):
            solve(saveat, **keywords)


def test_a_step_plan_records_the_exact_accepted_states():
    """``SaveAt(steps=True)`` saves every accepted step: the saved times and states are the
    interpolant's nodes bit for bit, the last is the terminal state, the tail is NaN."""
    case = _ODE
    every = ds.pdesolve(
        case.problem,
        _jndf(),
        (0.0, 0.12),
        case.initial_state(),
        theta=_THETA,
        saveat=ds.SaveAt(steps=True, dense=True),
        max_steps=40,
    )
    count = int(every.valid_count)
    assert count == int(every.stats.accepts) and int(every.stats.saveat_interpolations) == 0
    np.testing.assert_array_equal(every.ts[:count], every.interpolation.ts[1 : count + 1])
    np.testing.assert_array_equal(every.ys[:count], every.interpolation.ys[1 : count + 1, :2])
    np.testing.assert_array_equal(every.ys[count - 1], every.terminal.y)
    assert bool(jnp.all(jnp.isnan(every.ts[count:])))


def test_the_save_grid_is_evaluated_only_on_attempts_that_cover_a_save_time():
    """Five save times cost at most five covering attempts of five interpolant evaluations
    each, however many attempts the solve makes: an attempt that covers no save time,
    rejected or accepted, evaluates nothing."""
    solution = ds.pdesolve(
        _ODE.problem,
        _jndf(),
        (0.0, 0.5),
        _ODE.initial_state(),
        theta=_THETA,
        saveat=ds.SaveAt(ts=jnp.linspace(0.1, 0.5, 5)),
    )
    assert int(solution.stats.trial_steps) > 5
    assert 0 < int(solution.stats.saveat_interpolations) <= 5 * 5


@pytest.mark.parametrize(
    "stop,expected",
    ((0.4, (0.0, 0.25, 0.4)), (0.6, (0.0, 0.25, 0.5, 0.6)), (2.0, (0.0, 0.25, 0.5, 0.75))),
    ids=("root-inside-the-grid", "root-after-two-rows", "no-fire"),
)
def test_a_terminating_event_keeps_a_canonical_capacity_prefix(buffers, stop, expected):
    """A projected cutoff reading a local algebraic field and an algebraic global adds one
    reserve slot: outputs are a valid prefix ending at the exact root with the exact terminal
    state, the tail is NaN with metadata -1, and the trimmed view is that prefix."""
    solution, count = buffers[stop], len(expected)
    assert solution.ts.shape == (5,) and int(solution.valid_count) == count
    np.testing.assert_array_equal(solution.valid_mask, np.arange(5) < count)
    np.testing.assert_allclose(solution.ts[:count], expected, rtol=0.0, atol=_ROOT_BOUND)
    for floating in (solution.ts, solution.ys, solution.lambdas):
        assert bool(jnp.all(jnp.isnan(floating[count:])))
    for values in (solution.event_indices, solution.event_kinds, solution.event_directions):
        assert bool(jnp.all(values[count:] == -1))
    fired = stop < 1.0
    assert solution.result == (ds.RESULT.EVENT_STOP if fired else ds.RESULT.SUCCESS)
    last = (int(solution.event_kinds[count - 1]), int(solution.event_directions[count - 1]))
    assert last == ((2, -1) if fired else (-1, -1))
    if fired:
        terminal = jnp.concatenate((solution.terminal.y, solution.terminal.lam))
        np.testing.assert_allclose(terminal, _DAE.analytical(stop, 0.4), atol=_DAE_ROOT_BOUND)
    trimmed = solution.trimmed()
    assert trimmed.ts.shape == (count,) and bool(jnp.all(trimmed.valid_mask))
    np.testing.assert_array_equal(trimmed.event_kinds, solution.event_kinds[:count])


def test_trimming_a_batch_of_buffers_requires_and_selects_one_integral_lane(buffers):
    """A stacked batch of fixed buffers is trimmed only for an explicit integer lane."""
    batched = jax.tree_util.tree_map(
        lambda a, b: jnp.stack((a, b)) if isinstance(a, jax.Array) else a,
        buffers[0.4],
        buffers[0.6],
    )
    with pytest.raises(ValueError, match="requires an explicit lane"):
        batched.trimmed()
    with pytest.raises(ValueError, match="integer batch index"):
        batched.trimmed(lane=0.5)
    selected = batched.trimmed(lane=1).ts
    np.testing.assert_allclose(selected, [0.0, 0.25, 0.5, 0.6], rtol=0.0, atol=_ROOT_BOUND)


_DISCRETE = ds.Event(cond_fn=lambda time, state, args, **kwargs: time >= 0.035)
_TWICE = EventSet((_TIME_STOP, _TIME_STOP))
_SHORT_GRID = ds.SaveAt(ts=jnp.asarray([0.0, 0.02, 0.06]))


@pytest.mark.parametrize("case", ("discrete", "simultaneous"))
def test_the_terminal_row_records_the_event_kind_in_declaration_order(case):
    """A discrete event marks kind 1 and no direction; of two simultaneous continuous events
    the first declared fires, with kind 2, direction up and the event mask on it alone."""
    event, expected = (_DISCRETE, (1, 0)) if case == "discrete" else (_TWICE, (2, 1))
    solution = ds.pdesolve(
        _ODE.problem,
        _jndf(),
        (0.0, 0.1),
        _ODE.initial_state(),
        theta=_THETA,
        control_state0={"stop": jnp.asarray(0.04)},
        event=event,
        saveat=_SHORT_GRID,
        max_steps=400,
    )
    slot = int(solution.valid_count) - 1
    assert int(solution.event_indices[slot]) == 0
    assert (int(solution.event_kinds[slot]), int(solution.event_directions[slot])) == expected
    assert int(solution.result_codes[slot]) == int(solution.terminal.result_code)
    if case == "simultaneous":
        assert (bool(solution.event_mask[0]), bool(solution.event_mask[1])) == (True, False)


_LONG_GRID = ds.SaveAt(ts=jnp.asarray([0.0, 0.02, 0.04, 0.06, 0.08, 0.1]))


def test_mapped_lanes_keep_their_own_valid_prefixes():
    """Under jit and vmap each lane of a fixed grid ends at its own event, with the mask
    exactly the prefix of its own count and NaN after it."""

    def lane(stop):
        solution = ds.pdesolve(
            _ODE.problem,
            _jndf(),
            (0.0, 0.1),
            _ODE.initial_state(),
            theta=_THETA,
            control_state0={"stop": stop},
            event=_TIME_STOP,
            saveat=_LONG_GRID,
            max_steps=400,
        )
        return solution.ts, solution.ys, solution.valid_mask, solution.valid_count

    ts, ys, masks, counts = jax.jit(jax.vmap(lane))(jnp.asarray([0.0, 0.035, 0.075]))
    np.testing.assert_array_equal(counts, [1, 3, 5])
    np.testing.assert_array_equal(masks, np.arange(7) < np.asarray(counts)[:, None])
    np.testing.assert_allclose(ts[1, :3], [0.0, 0.02, 0.035], rtol=0.0, atol=_ROOT_BOUND)
    assert bool(jnp.all(jnp.isnan(ys[1, 3:])))


def _repeating_event():
    def condition(time, state, args, *, control=None, **kwargs):
        return time - control["next"]

    def advance(time, state, args, control):
        return state, {"next": control["next"] + 0.02}

    return ds.Event(condition, advance, Bisection(**_ROOT), "up")


def test_the_event_log_holds_its_capacity_and_flags_an_overflow():
    """Four firings into a log of capacity two keep two and flag the overflow; a capacity
    below one is refused."""
    ode = _ODE

    def solve(capacity):
        return ds.pdesolve(
            ode.problem,
            _jndf(),
            (0.0, 0.095),
            ode.initial_state(),
            theta=_THETA,
            control_state0={"next": jnp.asarray(0.02)},
            event=_repeating_event(),
            max_steps=400,
            max_events=capacity,
        )

    log = solve(2).event_log
    assert (int(log.count), bool(log.overflow), log.ts.shape) == (2, True, (2,))
    with pytest.raises(ds.ConfigurationError, match="max_events must be at least 1"):
        solve(0)


def test_an_affect_without_controls_keeps_epoch_zero_and_restarts_the_solver():
    """An affect that folds into the parameters fires once, keeps every epoch at zero, calls
    a fresh setup right after the event, and integrates the doubled rate exactly."""
    ode, event_time = _ODE, 0.025

    def condition(time, state, args, *, control=None, **kwargs):
        return jnp.where(control < 0.6, time - event_time, -1.0)

    solution = ds.pdesolve(
        ode.problem,
        ds.JNDF.stock(linear="dense", **_FINE),
        (0.0, 0.05),
        ode.initial_state(),
        theta=_THETA,
        event=ds.Event(condition, lambda t, y, args, c: (y, 2.0 * c), Bisection(**_ROOT), "up"),
        saveat=ds.SaveAt(ts=jnp.asarray([0.01, 0.04, 0.05])),
        max_steps=300,
        trace=True,
    )
    assert solution.control_state is None and int(solution.event_log.count) == 1
    np.testing.assert_array_equal(solution.control_epochs, 0)
    times = np.asarray(solution.trace.header.t)
    restart = np.flatnonzero((times > event_time) & (times < event_time + 1e-3))
    assert bool(np.asarray(solution.trace.call_setup)[restart[0]])
    np.testing.assert_allclose(
        solution.ys[-1, 0], np.exp(-0.4 * event_time - 0.8 * 0.025), rtol=_AFFECT_BOUND
    )


def test_an_operational_switch_changes_the_row_and_owns_its_samples_by_epoch():
    """A voltage crossing switches the table from current to voltage mode: the epoch advances
    once, samples before it hold the current and after it the voltage, and a dense output is
    refused across the switch."""
    problem = operational_cell().build()
    target = 0.25
    level = (2.0 - target) * np.exp(-0.2) - target

    def cutoff(time, state, args, *, control=None, **kwargs):
        return jnp.mean(state[:4]) - 2.0 * state[-1] - level

    def switch(time, state, args, control):
        return state, control.with_operating(mode=1, target=level, current_seed=state[-1])

    def solve(dense=False):
        return ds.pdesolve(
            problem,
            ds.JNDF.stock(linear="dense", **_FINE),
            (0.0, 0.5),
            jnp.concatenate([jnp.full((4,), 2.0), jnp.zeros((1,))]),
            saveat=ds.SaveAt(ts=jnp.asarray([0.1, 0.2, 0.3]), output=_PROJECTED, dense=dense),
            control_state0=ds.ControlState.operating(mode=0, target=target),
            event=ds.Event(cutoff, switch, Bisection(**_ROOT), "down", _PROJECTED),
            max_steps=400,
        )

    switched = solve()
    currents = switched.lambdas[:, 0]
    voltages = jnp.mean(switched.ys, axis=1) - 2.0 * currents
    assert bool(switched.ok) and (
        int(switched.control_state.mode),
        int(switched.control_state.epoch),
    ) == (1, 1)
    np.testing.assert_array_equal(switched.control_epochs, [0, 1, 1])
    np.testing.assert_allclose(
        [currents[0], *voltages[1:], switched.event_log.ts[0]],
        [target, level, level, 0.2],
        atol=_SWITCH_BOUND,
    )
    with pytest.raises(ds.ConfigurationError, match="single smooth interpolant"):
        solve(dense=True)


def test_schedules_and_segments_are_value_types_that_validate_their_entries():
    """Segments compare and hash by value and read a polynomial target; a schedule sums its
    durations and refuses an empty or a non-segment entry."""
    held = ds.Segment(current=1e-3, duration=3600.0)
    ramp = ds.Segment(voltage=(4.0, 1e-4), duration=1800.0)
    assert held == ds.Segment(current=1e-3, duration=3600.0) and hash(held) == hash(
        ds.Segment(current=1e-3, duration=3600.0)
    )
    assert (ramp.mode, ramp.target) == ("voltage", (4.0, 1e-4))
    schedule = ds.Schedule([held, ramp])
    assert (len(schedule), schedule.duration, schedule.degree) == (2, 5400.0, 1)
    with pytest.raises(ValueError, match="at least one segment"):
        ds.Schedule([])
    with pytest.raises(TypeError):
        ds.Schedule([held, "hold"])


def _van_der_pol():
    return ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack(
            (y[1], theta * ((1.0 - y[0] ** 2) * y[1] - y[0]))
        ),
        mass=ds.MassOperator.identity(2),
        structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),
    )


def _vdp(solver, span, y0, **keywords):
    """Van der Pol with ``mu = 8``: stiff, and on its slow branch over ``(0, 4)``."""
    return ds.pdesolve(
        _van_der_pol(),
        solver,
        span,
        jnp.asarray(y0),
        theta=jnp.asarray(8.0),
        dt0=1e-6,
        max_steps=2000,
        **keywords,
    )


@pytest.fixture(scope="module")
def histories():
    solver = ds.JIDA.stock(linear="dense", **_VDP)
    whole = _vdp(solver, (0.0, 4.0), (2.0, 0.0))
    first = _vdp(solver, (0.0, 2.0), (2.0, 0.0), history=True)
    second = _vdp(solver, (2.0, 4.0), first.terminal.y, history=first.history)
    return whole, first, second


def test_a_solve_given_its_history_continues_the_integrator(histories):
    """Two solves joined by ``history=`` take about the steps of one, start the second from the
    step the first reached, and end where the one solve ends; another solver refuses the
    record."""
    whole, first, second = histories
    assert whole.history is None and bool(np.asarray(first.history.continues))
    assert bool(second.ok) and bool(np.asarray(second.history.continues))
    assert float(second.stats.first_step) > _CONTINUED_STEP
    assert int(first.stats.accepts) + int(second.stats.accepts) <= _COUNT_BOUND * int(
        whole.stats.accepts
    )
    np.testing.assert_allclose(second.terminal.y, whole.terminal.y, rtol=0.0, atol=_END_BOUND)
    with pytest.raises(TypeError, match="record of another solver"):
        _vdp(
            ds.JNDF.stock(linear="dense", **_VDP),
            (2.0, 2.1),
            first.terminal.y,
            history=first.history,
        )


_OBSERVED, _OBSERVED_Y0, _RATE = observed_recipe()
#: Eight states and one global on 24 save times; the named outputs are two scalars.
_SAVE_TIMES = jnp.linspace(0.05, 1.0, 24)
_NAMED = ("mean_u", "load_gap")
#: Root-finder settings (requested tolerances, not asserted bounds).
_EVENT_ROOT = {"atol": 1e-12}
_CHECKPOINTED = ds.JNDF.stock(
    linear="dense", adjoint=ds.CheckpointAdjoint(), **{"rtol": 1e-9, "atol": 1e-11}
)
# measured 0.0 for the named observations against observe() on the full save, CPU VM
# 2026-10-22, pytest tests/public_api/test_solve_outputs.py::test_named_outputs_equal_the_observed_full_save_in_less_compiled_memory
_NAMED_BOUND = 1e-11
# measured 3.7e-9 between the observed level at the terminal sample and the event level (the
# located root's tolerance), CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs.py::test_named_outputs_hold_the_terminal_sample_and_refuse_undeclared_names[terminal-event]
_LEVEL_BOUND = 1e-8


def _observed_solve(value, outputs, *, ts=_SAVE_TIMES, event=None):
    theta = ds.default_theta(_OBSERVED, {_RATE: value})
    return ds.pdesolve(
        _OBSERVED,
        _CHECKPOINTED,
        (0.0, 1.0),
        _OBSERVED_Y0,
        theta=theta,
        saveat=ds.SaveAt(ts=ts, outputs=outputs),
        event=event,
    )


def _observe(save, name):
    """observe() on every row of a full save."""
    return jax.vmap(lambda y, lam, t: _OBSERVED.observe(y, t, lam)[name])(
        save["ys"], save["lambdas"], save["ts"]
    )


def _stored(solution):
    """What a solve stores at its save times."""
    return {key: getattr(solution, key) for key in ("ts", "ys", "lambdas", "outputs")}


@pytest.fixture(scope="module")
def saved():
    """The full save and the named save of one problem as compiled programs: their solutions
    and their compiled memory analyses, keyed by the output request."""

    def compiled(outputs):
        value = jnp.asarray(1.3)
        program = jax.jit(lambda v: _stored(_observed_solve(v, outputs))).lower(value).compile()
        return program(value), program.memory_analysis()

    return {outputs: compiled(outputs) for outputs in (None, _NAMED)}


def test_named_outputs_equal_the_observed_full_save_in_less_compiled_memory(saved):
    """Naming two observations stores them alone (no ``ys``, no ``lambdas``) and equal to
    observe() on the full save; the compiled outputs shrink by exactly the rows not stored and
    the whole compiled footprint is smaller."""
    (full, full_memory), (named, named_memory) = saved[None], saved[_NAMED]
    assert (named["ys"], named["lambdas"], full["outputs"]) == (None, None, None)
    assert tuple(named["outputs"]) == _NAMED
    for name in _NAMED:
        np.testing.assert_allclose(named["outputs"][name], _observe(full, name), atol=_NAMED_BOUND)
    # 24 rows of eight states and one global against 24 rows of two scalars
    stored = full_memory.output_size_in_bytes - named_memory.output_size_in_bytes
    assert stored == 24 * (8 + 1 - 2) * 8

    def total(memory):
        return (
            memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes
        )

    # measured 784,707 B (full save) against 782,819 B (named), CPU VM 2026-10-22,
    # pytest tests/public_api/test_solve_outputs.py::test_named_outputs_equal_the_observed_full_save_in_less_compiled_memory
    assert total(named_memory) < total(full_memory)


def _level(time, state, args, **kwargs):
    return jnp.mean(state[:4]) - 1.0


_SHORT_TIMES = jnp.asarray([0.1, 0.2, 0.3, 0.5, 0.8])
_LEVEL_STOP = ds.Event(cond_fn=_level, root_finder=Bisection(**_EVENT_ROOT), direction="down")


@pytest.mark.parametrize("case", ("terminal-event", "unknown-name", "bare-string"))
def test_named_outputs_hold_the_terminal_sample_and_refuse_undeclared_names(case):
    """A terminating event's sample is observed at the event and its stored state is the state
    in ``terminal``, a field is that state's slice and the rows after it are NaN; an
    undeclared name is refused listing the declared observations and fields, and a bare
    string is refused."""
    value = jnp.asarray(1.3)
    if case == "terminal-event":
        request = ("mean_u", "c_e", "state")
        named = _observed_solve(value, request, ts=_SHORT_TIMES, event=_LEVEL_STOP)
        last = int(named.valid_count) - 1
        state = named.outputs["state"]
        assert named.result == ds.RESULT.EVENT_STOP and named.lambdas is None
        assert float(named.ts[last]) == float(named.terminal.t)
        np.testing.assert_allclose(named.outputs["mean_u"][last], 1.0, atol=_LEVEL_BOUND)
        np.testing.assert_array_equal(state[last], named.terminal.y)
        np.testing.assert_array_equal(named.outputs["c_e"][: last + 1], state[: last + 1, 4:])
        assert bool(jnp.all(jnp.isnan(named.outputs["mean_u"][last + 1 :])))
    elif case == "unknown-name":
        declared = r"'voltage'.*\['mean_u', 'load_gap'\].*\['c_e', 'u'\]"
        with pytest.raises(ds.ConfigurationError, match=declared):
            _observed_solve(value, ("voltage",))
    else:
        with pytest.raises(ds.ConfigurationError, match="tuple of output names"):
            ds.SaveAt(outputs="mean_u")
