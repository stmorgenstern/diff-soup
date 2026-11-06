"""The receding horizon: one step body under `lax.scan`, or stepped from the host."""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

_WARM_STARTS = ("shift", "previous", "cold")


class ClosedLoop(eqx.Module):
    """What a receding-horizon run returns, every leaf an array stacked over the intervals.

    Attributes:
        controls: The applied first move of each interval.
        estimates: The estimator's state before each horizon.
        plant_states: The plant state after each interval.
        measurements: The measurement after each interval.
        records: The `SolveRecord` of each horizon.
        solve_seconds: The wall time of each horizon solve when driven by `HostLoop`, else `None`.
    """

    controls: Any
    estimates: Any
    plant_states: Any
    measurements: Any
    records: Any
    solve_seconds: Any = None


class HostLoop(eqx.Module):
    """Drive `mpc_step` from Python for a plant outside JAX, such as hardware or an external simulator.

    Args:
        plant: `plant(state, u, interval) -> (state, measurement)`, called on the host.
    """

    plant: Any = eqx.field(static=True)


def mpc_step(
    carry: Any,
    inputs: Any,
    *,
    control_problem: Any,
    solver: Any,
    plant: Any,
    estimator: Any,
    warm_start: str = "shift",
) -> tuple[Any, Any]:
    """One interval: estimate, solve the horizon warm-started, apply the first move, advance the plant.

    Args:
        carry: `(estimator_state, previous_solution, previous_record, plant_state)`; a `previous_record` of `None` solves the horizon cold.
        inputs: The interval's index.
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem].
        solver: A `CompiledSolver`, `SQP` by default.
        plant: `plant(state, u, interval) -> (state, measurement)` as a JAX function.
        estimator: An [Estimator][diff_soup.optimization.control.Estimator].
        warm_start: `"shift"`, `"previous"` or `"cold"`, as for `receding_horizon`.

    Returns:
        The next carry and the interval's output, the slice of a `ClosedLoop`.
    """
    estimate, previous, record, plant_state = carry
    solution, record = _solve_horizon(
        control_problem, solver, estimate, previous, record, warm_start=warm_start
    )
    move = solution[0]
    plant_state, measurement = plant(plant_state, move, inputs)
    next_estimate = estimator.estimate(estimate, move, measurement, control_problem.args())
    output = ClosedLoop(
        controls=move,
        estimates=estimate,
        plant_states=plant_state,
        measurements=measurement,
        records=record,
    )
    return (next_estimate, solution, record, plant_state), output


def receding_horizon(
    control_problem: Any,
    plant: Any,
    estimator: Any,
    n_intervals: int,
    *,
    solver: Any = None,
    warm_start: str = "shift",
    driver: Any = None,
) -> ClosedLoop:
    """Run the closed loop for `n_intervals`, under `lax.scan` unless `driver` is a `HostLoop`.

    Args:
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem].
        plant: `plant(state, u, interval) -> (state, measurement)`; a JAX function, or the host plant of `driver`.
        estimator: An [Estimator][diff_soup.optimization.control.Estimator].
        n_intervals: The number of intervals to run.
        solver: The horizon solver, `SQP()` when `None`.
        warm_start: `"shift"` starts each horizon from the previous solution shifted by one interval; `"previous"` from it unshifted; `"cold"` from the problem's start.
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
    if solver is None:
        from diff_soup.optimization.solvers.sqp import SQP

        solver = SQP()
    if driver is not None:
        if not isinstance(driver, HostLoop):
            raise ConfigurationError(
                f"receding_horizon driver must be None or a HostLoop; got {type(driver).__name__}."
            )
        return _host_loop(
            control_problem, driver.plant, estimator, int(n_intervals), solver, warm_start
        )

    @eqx.filter_jit
    def closed_loop(args):
        # The args are traced; the model, the functionals and the closures stay as they are.
        initial_state, theta, limits = args
        problem = dataclasses.replace(
            control_problem, initial_state=initial_state, theta=theta, limits=limits
        )
        step = functools.partial(
            mpc_step,
            control_problem=problem,
            solver=solver,
            plant=plant,
            estimator=estimator,
            warm_start=warm_start,
        )
        carry = (estimator.init(args), _start(problem, solver), None, initial_state)
        carry, first = step(carry, jnp.asarray(0, dtype=jnp.int32))
        if n_intervals == 1:
            return jax.tree_util.tree_map(lambda leaf: leaf[None], first)
        _, rest = jax.lax.scan(step, carry, jnp.arange(1, n_intervals, dtype=jnp.int32))
        return jax.tree_util.tree_map(
            lambda head, tail: jnp.concatenate([head[None], tail]), first, rest
        )

    return closed_loop(control_problem.args())


def _start(control_problem: Any, solver: Any) -> Any:
    """The first guess of a cold horizon, the numeric problem's start."""
    return jnp.asarray(control_problem.numeric_problem(solver).x0)


def _solve_horizon(control_problem, solver, estimate, previous, record, *, warm_start, cold=None):
    """Solve the horizon from the estimate, warm-started from the previous one unless cold.

    With `cold` a traced boolean, both starts are one program and `cold` picks the branch, so a
    host loop compiles once for its cold first horizon and its warm later ones.
    """
    numeric = control_problem.numeric_problem(solver)
    _, theta, limits = control_problem.args()
    args = (estimate.plant, theta, limits)

    def cold_run():
        return solver.run(numeric, jnp.asarray(numeric.x0), args)

    if record is None or warm_start == "cold":
        return cold_run()
    if warm_start == "shift":
        moves = _shift_map(control_problem.controls.intervals)
        rows = _row_shift_map(control_problem)
        previous = previous[moves]
        record = _shifted_record(record, moves, rows)

    def warm_run():
        return solver.run(numeric, previous, args, warm_start=record)

    if cold is None:
        return warm_run()
    return jax.lax.cond(cold, cold_run, warm_run)


def _record_template(control_problem: Any, solver: Any, estimate: Any) -> Any:
    """A record of zeros shaped as the horizon solve's, which a cold first horizon is handed."""
    shapes = jax.eval_shape(
        lambda e: _solve_horizon(control_problem, solver, e, None, None, warm_start="cold")[1],
        estimate,
    )
    return jax.tree_util.tree_map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)


@functools.lru_cache(maxsize=64)
def _shift_map(intervals: int) -> np.ndarray:
    """Where each interval's value comes from after one interval passes: the next, the last repeated."""
    return np.minimum(np.arange(intervals) + 1, intervals - 1)


def _row_shift_map(control_problem: Any) -> np.ndarray:
    """Where each inequality row's multiplier comes from after one interval passes.

    A path row at saved time `t` takes the row at `t + duration` where that time is saved, and
    keeps its own otherwise; rows of `Constraints` on `u` keep theirs.
    """
    times = np.asarray(control_problem.save_times, dtype=np.float64).ravel()
    duration = control_problem.controls.interval
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


def _shifted_record(record: Any, moves: np.ndarray, rows: np.ndarray) -> Any:
    """`record` with its inequality multipliers, slacks and bound multipliers shifted one interval."""
    lower, upper = record.bound_multipliers
    return dataclasses.replace(
        record,
        multipliers_inequality=record.multipliers_inequality[rows],
        slacks=record.slacks[rows],
        bound_multipliers=(lower[moves], upper[moves]),
    )


def _host_loop(control_problem, plant, estimator, n_intervals, solver, warm_start) -> ClosedLoop:
    """Step the loop from Python: the horizon solve and the estimate compiled, the plant on the host.

    Every horizon is one compiled program: a cold loop never hands a record on, and a warm one
    hands the first horizon a record of zeros with the flag that runs it cold.
    """
    solve = eqx.filter_jit(
        lambda estimate, previous, record, cold: _solve_horizon(
            control_problem, solver, estimate, previous, record, warm_start=warm_start, cold=cold
        )
    )
    # The estimator is closed over, not traced: a model it holds keeps its concrete arrays.
    update = eqx.filter_jit(
        lambda state, move, measurement, args: estimator.estimate(state, move, measurement, args)
    )
    args = control_problem.args()
    estimate = estimator.init(args)
    previous = _start(control_problem, solver)
    warm = warm_start != "cold"
    handed = _record_template(control_problem, solver, estimate) if warm else None
    plant_state = args[0]
    outputs, seconds = [], []
    for interval in range(n_intervals):
        tick = time.perf_counter()
        cold = jnp.asarray(interval == 0) if warm else None
        previous, record = jax.block_until_ready(solve(estimate, previous, handed, cold))
        seconds.append(time.perf_counter() - tick)
        handed = record if warm else None
        move = previous[0]
        plant_state, measurement = plant(plant_state, np.asarray(move), interval)
        outputs.append(
            ClosedLoop(
                controls=move,
                estimates=estimate,
                plant_states=jnp.asarray(plant_state),
                measurements=jnp.asarray(measurement),
                records=record,
            )
        )
        estimate = update(estimate, move, jnp.asarray(measurement), args)
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *outputs)
    return dataclasses.replace(stacked, solve_seconds=np.asarray(seconds))
