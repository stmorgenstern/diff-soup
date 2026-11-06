"""The receding horizon: one step body under `lax.scan`, or stepped from the host."""

from __future__ import annotations

from typing import Any

import equinox as eqx


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
) -> tuple[Any, Any]:
    """One interval: estimate, solve the horizon warm-started, apply the first move, advance the plant.

    Args:
        carry: `(estimator_state, previous_solution, previous_record, plant_state)`.
        inputs: The per-interval inputs, such as the measurement when driven from the host.
        control_problem: The [ControlProblem][diff_soup.optimization.control.ControlProblem].
        solver: A [CompiledSolver][diff_soup.optimization.CompiledSolver], `SQP` by default.
        plant: `plant(state, u, interval) -> (state, measurement)` as a JAX function.
        estimator: An [Estimator][diff_soup.optimization.control.Estimator].

    Returns:
        The next carry and the interval's output, the slice of a `ClosedLoop`.
    """
    raise NotImplementedError


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
    raise NotImplementedError
