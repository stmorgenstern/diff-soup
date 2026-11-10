"""The receding horizon: one step body under `lax.scan`, or stepped from the host."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.control._horizon import sample_times
from diff_soup.optimization.solvers import SolveRecord
from diff_soup.optimization.solvers.sqp import SQP

if TYPE_CHECKING:
    from diff_soup.optimization._numeric import OptimizationProblem
    from diff_soup.optimization.control.estimators import Estimator
    from diff_soup.optimization.control.problem import ControlProblem
    from diff_soup.optimization.solvers import CompiledSolver

#: The carry of `mpc_step`: the estimator's state, the next horizon's start, the record its
#: multipliers come from (`None` for a cold solve), and the plant's state.
Carry = tuple[eqx.Module, Float[Array, " n"], SolveRecord | None, PyTree]

_WARM_STARTS = ("shift", "previous", "cold")


class ClosedLoop(eqx.Module):
    """What a receding-horizon run returns, every leaf an array stacked over the intervals.

    Attributes:
        controls: The applied first move of each interval.
        estimates: The estimator's state before each horizon.
        plant_states: The plant state after each interval.
        measurements: The measurement after each interval.
        records: The `SolveRecord` of each horizon.
        solve_seconds: The wall time of each horizon solve when driven by `HostLoop`, compile excluded, else `None`.
        compile_seconds: The wall time of compiling the horizon solve when driven by `HostLoop`, else `None`.
    """

    controls: Float[Array, "*intervals"]
    estimates: eqx.Module
    plant_states: PyTree
    measurements: PyTree
    records: SolveRecord
    solve_seconds: Float[np.ndarray, " intervals"] | None = None
    compile_seconds: float | None = eqx.field(static=True, default=None)


class HostLoop(eqx.Module):
    """Drive `mpc_step` from Python for a plant outside JAX, such as hardware or an external simulator.

    The horizon solve and the estimator update are each one compiled program for every
    interval; the plant runs on the host.

    Args:
        plant: `plant(state, u, interval) -> (state, measurement)`, called on the host.
    """

    plant: Callable = eqx.field(static=True)


def mpc_step(
    carry: Carry,
    inputs: Int[ArrayLike, ""],
    *,
    control_problem: ControlProblem,
    solver: CompiledSolver,
    plant: Callable,
    estimator: Estimator,
    warm_start: str = "shift",
) -> tuple[Carry, ClosedLoop]:
    """One interval: estimate, solve the horizon warm-started, apply the first move, advance the plant.

    Args:
        carry: `(estimator_state, start, record, plant_state)`: the horizon starts from `start` with the multipliers of `record`, a `SolveRecord`, both already moved to this interval; a `record` of `None` solves the horizon cold. `receding_horizon` starts from the problem's start and a record of zeros.
        inputs: The interval's index.
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem].
        solver: A `CompiledSolver`, `SQP` by default.
        plant: `plant(state, u, interval) -> (state, measurement)` as a JAX function.
        estimator: An [Estimator][diff_soup.optimization.control.Estimator].
        warm_start: `"shift"`, `"previous"` or `"cold"`, as for `receding_horizon`.

    Returns:
        The next carry and the interval's output, the slice of a `ClosedLoop`.
    """
    numeric = control_problem.numeric_problem(solver)
    return _step(carry, inputs, control_problem, numeric, solver, plant, estimator, warm_start)


def receding_horizon(
    control_problem: ControlProblem,
    plant: Callable | None,
    estimator: Estimator,
    n_intervals: int,
    *,
    solver: CompiledSolver | None = None,
    warm_start: str = "shift",
    driver: HostLoop | None = None,
) -> ClosedLoop:
    """Run the closed loop for `n_intervals`, under `lax.scan` unless `driver` is a `HostLoop`.

    The loop is one compiled program, cached on its inputs: a second call with the same
    control problem configuration, plant, estimator and solver runs at solve time. The first
    horizon starts warm from a record of zeros, so every horizon is one solve in the program.

    Args:
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem].
        plant: `plant(state, u, interval) -> (state, measurement)`; a JAX function, or `None` with a `HostLoop` driver.
        estimator: An [Estimator][diff_soup.optimization.control.Estimator].
        n_intervals: The number of intervals to run.
        solver: The horizon solver, `SQP()` when `None`.
        warm_start: `"shift"` starts each horizon from the previous solution shifted by one interval, which needs intervals of one length; `"previous"` from it unshifted; `"cold"` from the problem's start.
        driver: `None` for `lax.scan`, or a [HostLoop][diff_soup.optimization.control.HostLoop].

    Returns:
        The [ClosedLoop][diff_soup.optimization.control.ClosedLoop].
    """
    if warm_start not in _WARM_STARTS:
        raise ConfigurationError(
            f"receding_horizon warm_start must be one of {_WARM_STARTS}; got {warm_start!r}."
        )
    if int(n_intervals) < 1:
        raise ConfigurationError(f"receding_horizon needs n_intervals >= 1; got {n_intervals}.")
    if control_problem.controls.free_durations:
        raise ConfigurationError(
            "receding_horizon moves a horizon of fixed intervals one interval at a time, and these "
            "controls have free durations. Solve the problem with optsolve, or give the receding "
            "horizon PiecewiseConstant controls."
        )
    durations = control_problem.controls.durations
    if warm_start == "shift" and any(d != durations[0] for d in durations):
        raise ConfigurationError(
            "receding_horizon warm_start='shift' moves the solution by one interval, which "
            f"needs intervals of one length; the controls have durations {durations}. Use "
            "warm_start='previous' or 'cold'."
        )
    solver = SQP() if solver is None else solver
    if driver is None:
        return _closed_loop(control_problem, plant, estimator, solver, int(n_intervals), warm_start)
    if not isinstance(driver, HostLoop):
        raise ConfigurationError(
            f"receding_horizon driver must be None or a HostLoop; got {type(driver).__name__}."
        )
    return _host_loop(
        control_problem, driver.plant, estimator, int(n_intervals), solver, warm_start
    )


@eqx.filter_jit
def _closed_loop(control_problem, plant, estimator, solver, n_intervals, warm_start) -> ClosedLoop:
    """The loop under `lax.scan`: one horizon solve in the program, the first from a zero record."""
    numeric = control_problem.numeric_problem(solver)
    args = control_problem.args()

    def step(carry, interval):
        return _step(
            carry, interval, control_problem, numeric, solver, plant, estimator, warm_start
        )

    carry = (estimator.init(args), numeric.x0, _zero_record(numeric), args[0])
    _, outputs = jax.lax.scan(step, carry, jnp.arange(n_intervals, dtype=jnp.int32))
    return outputs


def _step(carry, interval, control_problem, numeric, solver, plant, estimator, warm_start):
    """One interval of the loop over a lowered horizon; see `mpc_step`."""
    estimate, start, record, plant_state = carry
    solution, record = _solve_horizon(numeric, solver, estimate, start, record, warm_start)
    move = solution[0]
    plant_state, measurement = plant(plant_state, move, interval)
    next_estimate = estimator.estimate(estimate, move, measurement, control_problem.args())
    output = ClosedLoop(
        controls=move,
        estimates=estimate,
        plant_states=plant_state,
        measurements=measurement,
        records=record,
    )
    start, handed = _next_start(control_problem, numeric, solution, record, warm_start)
    return (next_estimate, start, handed, plant_state), output


def _solve_horizon(numeric, solver, estimate, start, record, warm_start):
    """Solve the horizon from the estimate: cold from the problem's start, else warm from `start` and `record`.

    The start and the record carry no derivative into the solution (`CompiledSolver.run`), so
    their tangents, which the previous horizon's solution hands them, are stopped here.
    """
    args = (estimate.plant, *numeric.args[1:])
    if record is None or warm_start == "cold":
        start = jax.lax.stop_gradient(jnp.asarray(numeric.x0))
        solution, solved = solver.run(numeric, start, args)
    else:
        start, warm = jax.lax.stop_gradient((start, record))
        solution, solved = solver.run(numeric, start, args, warm_start=warm)
    if record is not None:
        # The loop's carry keeps one type; a record holds counts and codes of fixed widths.
        solved = jax.tree.map(lambda new, old: new.astype(old.dtype), solved, record)
    return solution, solved


def _next_start(control_problem, numeric, solution, record, warm_start):
    """The next horizon's start and record: moved one interval for `"shift"`, as they are else."""
    if warm_start == "cold":
        return numeric.x0, record
    if warm_start == "previous":
        return solution, record
    moves = _shift_map(control_problem.controls.size)
    rows = _row_shift_map(control_problem)
    lower, upper = record.bound_multipliers
    shifted = dataclasses.replace(
        record,
        multipliers_inequality=record.multipliers_inequality[rows],
        slacks=record.slacks[rows],
        bound_multipliers=(lower[moves], upper[moves]),
    )
    return solution[moves], shifted


def _zero_record(numeric: OptimizationProblem) -> SolveRecord:
    """A record of zeros sized for `numeric`, which the warm path starts the first horizon from."""
    return SolveRecord.zeros(
        numeric.n, numeric.n_equality, numeric.n_inequality, dtype=jnp.result_type(numeric.x0)
    )


def _shift_map(intervals: int) -> np.ndarray:
    """Where each interval's value comes from after one interval passes: the next, the last repeated."""
    return np.minimum(np.arange(intervals) + 1, intervals - 1)


def _row_shift_map(control_problem: ControlProblem) -> np.ndarray:
    """Where each inequality row's multiplier comes from after one interval passes.

    A path row at saved time `t` takes the row at `t + duration` where that time is saved, and
    keeps its own otherwise; rows of `Constraints` on `u` keep theirs.
    """
    duration = control_problem.controls.interval
    if control_problem.save_times is None:
        durations = np.asarray(control_problem.controls.durations, dtype=np.float64)
        times = sample_times(durations, control_problem.save_fractions, xp=np)[1]
    else:
        times = np.asarray(control_problem.save_times, dtype=np.float64)
    later = np.searchsorted(times, times + duration)
    found = (later < times.size) & np.isclose(
        times[np.minimum(later, times.size - 1)], times + duration, rtol=0.0, atol=1e-9 * duration
    )
    source = np.where(found, later, np.arange(times.size))
    n_path = control_problem.n_path
    path = (source[:, None] * n_path + np.arange(n_path)[None, :]).ravel()
    constraints = control_problem.constraints
    extra = 0 if constraints is None else constraints.n_inequality
    return np.concatenate([path, path.size + np.arange(extra)]).astype(np.int32)


@eqx.filter_jit
def _solve_one(numeric, solver, estimate, start, record, warm_start):
    """One horizon solve of the host loop, compiled once for every interval."""
    return _solve_horizon(numeric, solver, estimate, start, record, warm_start)


@eqx.filter_jit
def _estimate(estimator, state, move, measurement, args):
    """One estimator update of the host loop, compiled once for every interval."""
    return estimator.estimate(state, move, measurement, args)


def _host_loop(control_problem, plant, estimator, n_intervals, solver, warm_start) -> ClosedLoop:
    """Step the loop from Python: the horizon solve and the estimate compiled, the plant on the host.

    The horizon solve is compiled ahead of the first interval and its time reported apart, so
    every entry of `solve_seconds` is a solve.
    """
    numeric = control_problem.numeric_problem(solver)
    args = control_problem.args()
    estimate = estimator.init(args)
    start, record = numeric.x0, _zero_record(numeric)
    tick = time.perf_counter()
    solve = _solve_one.lower(numeric, solver, estimate, start, record, warm_start).compile()
    compile_seconds = time.perf_counter() - tick
    plant_state = args[0]
    outputs, seconds = [], []
    for interval in range(n_intervals):
        tick = time.perf_counter()
        solution, solved = jax.block_until_ready(
            solve(numeric, solver, estimate, start, record, warm_start)
        )
        seconds.append(time.perf_counter() - tick)
        move = solution[0]
        plant_state, measurement = plant(plant_state, np.asarray(move), interval)
        outputs.append(
            ClosedLoop(
                controls=move,
                estimates=estimate,
                plant_states=jnp.asarray(plant_state),
                measurements=jnp.asarray(measurement),
                records=solved,
            )
        )
        estimate = _estimate(estimator, estimate, move, jnp.asarray(measurement), args)
        start, record = _next_start(control_problem, numeric, solution, solved, warm_start)
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *outputs)
    return dataclasses.replace(
        stacked, solve_seconds=np.asarray(seconds), compile_seconds=compile_seconds
    )
