"""A control problem: cost, path and terminal functionals of a model's solution over a horizon."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx

from diff_soup.optimization.problem import AbstractOptimizationProblem


class PiecewiseConstant(eqx.Module):
    """Controls that hold one target of an operational row per interval of a fixed grid.

    Args:
        mode: The row of `spec.control.operational` the targets drive: `"current"`, `"voltage"`, `"power"` or `"resistance"`.
        intervals: The number of intervals, the length of the decision vector.
        duration: The length of each interval, in the model's time unit.
    """

    mode: str = eqx.field(static=True)
    intervals: int = eqx.field(static=True)
    duration: float = eqx.field(static=True)

    def segments(self, u: Any) -> Any:
        """The schedule arrays of the horizon for the decision vector `u`, traced, never retraced for a new `u`."""
        raise NotImplementedError


class ControlProblem(AbstractOptimizationProblem):
    """Minimize a cost over the controls of a model under path and terminal constraints.

    The horizon runs through `ds.solve_schedule` with the segment targets as traced arrays of
    `u`. The path rows' Jacobian is forward mode through the schedule tangents; the cost
    gradient is reverse mode under `CheckpointAdjoint` when it pays. `args` carries the initial
    state, `theta` and the limits, so a batch of cells or scenarios is one `vmap`.

    Args:
        problem: A built diff-soup problem whose operational table has the row `controls` drives.
        controls: The parameterization of the decision vector, such as [PiecewiseConstant][diff_soup.optimization.control.PiecewiseConstant].
        cost: `cost(sol, u, args) -> scalar`, with `sol` the `Solution` over the horizon at `save_times`.
        path: `path(sol, u, args) -> rows` at or below 0 at every saved time, or `None`.
        terminal: `terminal(sol, u, args) -> rows` at 0 at the end of the horizon, or `None`.
        n_path: The number of rows `path` returns per saved time.
        n_terminal: The number of rows `terminal` returns.
        bounds: `(lower, upper)` on `u`, broadcast, or `None`.
        save_times: The times the solution is sampled at for the functionals.
        initial_state: The state the horizon starts from; an `args` leaf.
        theta: The model parameters; an `args` leaf.
        solver: The time-stepping solver and its tolerances.
        adjoint: The differentiation strategy of the horizon solve, `None` for forward mode.
        layout: The placement of a batch of horizons.
    """

    problem: Any = eqx.field(static=True, default=None)
    controls: Any = eqx.field(static=True, default=None)
    cost: Callable | None = eqx.field(static=True, default=None)
    path: Callable | None = eqx.field(static=True, default=None)
    terminal: Callable | None = eqx.field(static=True, default=None)
    n_path: int = eqx.field(static=True, default=0)
    n_terminal: int = eqx.field(static=True, default=0)
    save_times: Any = None
    initial_state: Any = None
    theta: Any = None
    solver: Any = eqx.field(static=True, default=None)
    adjoint: Any = eqx.field(static=True, default=None)
    layout: Any = eqx.field(static=True, default=None)

    def numeric_problem(self, optimizer: Any) -> Any:
        """Lower to closures over `u` and `args`; see [AbstractOptimizationProblem.numeric_problem][diff_soup.optimization.AbstractOptimizationProblem.numeric_problem]."""
        raise NotImplementedError

    def args(self) -> Any:
        """`(initial_state, theta, limits)`, the leaves a batch varies."""
        raise NotImplementedError
