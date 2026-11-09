"""A control problem: cost, path and terminal functionals of a model's solution over a horizon."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup._schedule import Schedule, ScheduleArrays, Segment, schedule_arrays
from diff_soup.optimization._binding import Held, hold
from diff_soup.optimization._derived import as_bounds
from diff_soup.optimization._numeric import OptimizationProblem
from diff_soup.optimization.constraints import Constraints
from diff_soup.optimization.control._horizon import Configuration, Horizon
from diff_soup.optimization.problem import AbstractOptimizationProblem

if TYPE_CHECKING:
    from diff_soup._adjoints import AbstractAdjoint
    from diff_soup._layout import Layout
    from diff_soup.optimization._abc import OptSolver
    from diff_soup.timestepping.vsvo.classes import JBDF, JIDA, JNDF

#: How far above the horizon a save time may lie and still be its end: Higham's bound on
#: the rounding error of a sum of `n` positive terms added in order, `(n - 1) * u * sum`
#: with `u` the unit roundoff (Higham, Accuracy and Stability of Numerical Algorithms,
#: 2nd ed., eq. 4.4), taken as `n * eps * horizon`.
_END_ROUNDING = np.finfo(np.float64).eps


def _lengths(duration: ArrayLike | Sequence[float]) -> float | tuple:
    """A duration as Python floats, a scalar or a tuple, so it is static and hashable."""
    if np.ndim(duration) == 0:
        return float(duration)
    return tuple(float(d) for d in np.asarray(duration, dtype=np.float64).ravel())


class PiecewiseConstant(eqx.Module):
    """Controls that hold one target of an operational row per interval of a fixed grid.

    Interval `k` holds the target `scale * u[k]`.

    Args:
        mode: The row of `spec.control.operational` the targets drive: `"current"`, `"voltage"`, `"power"` or `"resistance"`.
        intervals: The number of intervals, the length of the decision vector.
        duration: The length of each interval in the model's time unit, or a tuple of one length per interval.
        carry_step: As `ds.Schedule(carry_step=)`: `None` continues the integrator's step into the next interval, `False` starts each afresh.
        scale: The factor from the decision vector to the model's target, such as a C-rate's current with its sign.
    """

    mode: str = eqx.field(static=True)
    intervals: int = eqx.field(static=True)
    duration: float | tuple = eqx.field(static=True, converter=_lengths)
    carry_step: bool | None = eqx.field(static=True, default=None)
    scale: float = eqx.field(static=True, default=1.0, converter=float)

    def __check_init__(self) -> None:
        """Refuse a grid whose lengths do not match `intervals` or are not positive."""
        durations = np.asarray(self.durations, dtype=np.float64)
        if durations.shape != (self.intervals,) or not np.all(durations > 0):
            raise ConfigurationError(
                f"PiecewiseConstant needs {self.intervals} positive durations; got "
                f"{self.duration!r}."
            )

    @property
    def durations(self) -> tuple:
        """The length of each interval."""
        if isinstance(self.duration, tuple):
            return tuple(float(d) for d in self.duration)
        return (float(self.duration),) * self.intervals

    @property
    def horizon(self) -> float:
        """The end of the horizon: the durations added in order, as the schedule adds them."""
        end = 0.0
        for duration in self.durations:
            end += duration
        return end

    @property
    def interval(self) -> float:
        """The one length of every interval, which the receding horizon and the estimators step by."""
        durations = self.durations
        if any(d != durations[0] for d in durations):
            raise ConfigurationError(
                "This needs intervals of one length; the PiecewiseConstant has "
                f"durations {durations}."
            )
        return durations[0]

    def interval_controls(self) -> PiecewiseConstant:
        """The parameterization of one interval of this grid, keeping its mode, scale and step carry."""
        return dataclasses.replace(self, intervals=1, duration=self.interval)

    def segments(self, u: ArrayLike, modes: tuple[str, ...]) -> ScheduleArrays:
        """The schedule arrays of the horizon for the decision vector `u`, traced, never retraced for a new `u`.

        Args:
            u: The decision vector, `(intervals,)`.
            modes: The model's mode names, `problem.operational_modes()`.

        Returns:
            The [ScheduleArrays][diff_soup.ScheduleArrays] whose `target` leaf is `scale * u`.
        """
        template = schedule_arrays(
            Schedule(
                [Segment(**{self.mode: 0.0}, duration=d) for d in self.durations],
                carry_step=self.carry_step,
            ),
            modes,
        )
        u = jnp.asarray(u)
        if u.shape != (self.intervals,):
            raise ConfigurationError(
                f"PiecewiseConstant over {self.intervals} intervals takes a decision vector of "
                f"shape ({self.intervals},); got {u.shape}."
            )
        return dataclasses.replace(template, target=self.scale * u)


def _static_times(times: Any) -> tuple[float, ...] | None:
    """Save times as Python floats, static and hashable; a traced value is refused."""
    if times is None:
        return None
    if isinstance(times, jax.core.Tracer):
        raise ConfigurationError(
            "ControlProblem save_times are static: give concrete times, not traced ones."
        )
    return tuple(float(t) for t in np.asarray(times, dtype=np.float64).ravel())


def _static_start(start: Any) -> tuple[float, ...] | None:
    """The first guess as Python floats, static; a traced value is refused."""
    if start is None:
        return None
    if isinstance(start, jax.core.Tracer):
        raise ConfigurationError(
            "ControlProblem start is static: give a concrete first guess, not a traced one."
        )
    return tuple(float(value) for value in np.asarray(start, dtype=np.float64).ravel())


class ControlProblem(AbstractOptimizationProblem, eqx.Module):
    """Minimize a cost over the controls of a model under path and terminal constraints.

    The horizon runs through `ds.solve_schedule` with the segment targets as traced arrays of
    `u`. The path rows' Jacobian is forward mode through the schedule tangents; the cost
    gradient is reverse mode under `CheckpointAdjoint` when it pays. `args` carries the initial
    state, `theta` and the limits, so a batch of cells or scenarios is one `vmap`. The model,
    the save times and the first guess are static: the model is held by identity, so two
    problems over one model, one configuration and different `args` share a compile.

    Args:
        problem: A built diff-soup problem whose operational table has the row `controls` drives.
        controls: The parameterization of the decision vector, such as [PiecewiseConstant][diff_soup.optimization.control.PiecewiseConstant].
        cost: `cost(sol, u, args) -> scalar`, with `sol` the `Solution` over the horizon at `save_times`.
        path: `path(sol, u, args) -> rows` at or below 0 at every saved time, or `None`.
        terminal: `terminal(sol, u, args) -> rows` at 0 at the end of the horizon, or `None`.
        n_path: The number of rows `path` returns per saved time.
        n_terminal: The number of rows `terminal` returns.
        bounds: `(lower, upper)` on `u`, broadcast, or `None`.
        constraints: [Constraints][diff_soup.optimization.Constraints] on `u` itself, functions of `(u, args)`, or `None`. Their scales are padded with ones over the terminal and path rows, and their `jacobian` mode differentiates their own rows.
        save_times: The times the solution is sampled at for the functionals, after the horizon's start at 0. A last time at the horizon's end, within the rounding of its sum, is the schedule's own end.
        initial_state: The state the horizon starts from; an `args` leaf.
        theta: The model parameters; an `args` leaf.
        limits: What the functionals read as limits, such as a voltage ceiling; an `args` leaf.
        start: The first guess of `u`; the middle of the bounds, or zeros, when `None`.
        solver: The time-stepping solver and its tolerances.
        adjoint: The differentiation strategy of the horizon solve, `None` for forward mode.
        layout: The placement of a batch of horizons.
    """

    problem: Held = eqx.field(static=True, converter=hold)
    constraints: Constraints | None = None
    bounds: tuple[ArrayLike | None, ArrayLike | None] | None = None
    controls: PiecewiseConstant | None = eqx.field(static=True, default=None)
    cost: Callable | None = eqx.field(static=True, default=None)
    path: Callable | None = eqx.field(static=True, default=None)
    terminal: Callable | None = eqx.field(static=True, default=None)
    n_path: int = eqx.field(static=True, default=0)
    n_terminal: int = eqx.field(static=True, default=0)
    save_times: tuple[float, ...] | None = eqx.field(
        static=True, default=None, converter=_static_times
    )
    initial_state: Float[ArrayLike, "*batch n_states"] | None = None
    theta: PyTree = None
    limits: PyTree = None
    start: tuple[float, ...] | None = eqx.field(static=True, default=None, converter=_static_start)
    solver: JIDA | JBDF | JNDF | None = eqx.field(static=True, default=None)
    adjoint: AbstractAdjoint | None = eqx.field(static=True, default=None)
    layout: Layout | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        """Take a last save time within the rounding of the horizon's sum as its end."""
        times = _static_times(self.save_times)
        if times and self.controls is not None:
            end = self.controls.horizon
            rounding = len(self.controls.durations) * _END_ROUNDING * end
            if end < times[-1] <= end + rounding:
                self.save_times = (*times[:-1], end)

    @property
    def model(self) -> object:
        """The built diff-soup problem, held by identity."""
        return self.problem.value

    def numeric_problem(self, optimizer: OptSolver) -> OptimizationProblem:
        """Lower to closures over `u` and `args`; see [AbstractOptimizationProblem.numeric_problem][diff_soup.optimization.AbstractOptimizationProblem.numeric_problem]."""
        horizon = self._closures()
        n = self.controls.intervals
        start = _middle(self.bounds, n) if self.start is None else jnp.asarray(self.start)
        x0 = jnp.reshape(jnp.asarray(start, dtype=jnp.float64), (n,))
        structure = {
            "value_gradient",
            "hessian",
            "constraint_jacobian",
            "lagrangian_hessian",
            "constraint_hessian",
        }
        n_equality, n_inequality = horizon.n_equality, horizon.n_inequality
        if n_equality or n_inequality:
            structure.add("constraints")
        return OptimizationProblem(
            x0=x0,
            structure=frozenset(structure),
            bounds=as_bounds(self.bounds, x0),
            args=self.args(),
            n_equality=n_equality,
            n_inequality=n_inequality,
            constraint_scales=self._row_scales(),
            ad="forward" if self.adjoint is None else "reverse",
            _value=horizon.closure("value"),
            _value_and_gradient=horizon.closure("value_and_gradient"),
            _value_gradient_hessian=horizon.closure("value_gradient_hessian"),
            _constraints=horizon.closure("constraints"),
            _constraints_and_jacobians=horizon.closure("constraints_and_jacobians"),
            _lagrangian_hessian=horizon.closure("lagrangian_hessian"),
            _constraint_hessian=horizon.closure("constraint_hessian"),
            _evaluate=horizon.closure("evaluate"),
            _evaluate_with_derivatives=horizon.closure("evaluate_with_derivatives"),
        )

    def args(self) -> tuple[PyTree, PyTree, PyTree]:
        """`(initial_state, theta, limits)`, the leaves a batch varies."""
        return (self.initial_state, self.theta, self.limits)

    def _closures(self) -> Horizon:
        """The horizon's closures, equal for every control problem of this configuration.

        Problems over the same model with the same functionals, controls, constraints on `u`,
        save times, solver and adjoint have equal closures, so a solver compiled for one
        serves the next without retracing.
        """
        constraints = self.constraints
        if constraints is not None:
            # The scales are arrays the solver reads from the numeric problem, not the closures.
            constraints = dataclasses.replace(constraints, scales=None)
        return Horizon(
            Configuration(
                model=self.problem,
                controls=self.controls,
                cost=self.cost,
                path=self.path,
                terminal=self.terminal,
                n_path=self.n_path,
                n_terminal=self.n_terminal,
                constraints=constraints,
                save_times=self.save_times,
                solver=self.solver,
                adjoint=self.adjoint,
            )
        )

    def _row_scales(self) -> tuple[Float[Array, " n_eq"], Float[Array, " n_ineq"]] | None:
        """The scales of the `Constraints` on `u`, after ones for the terminal and path rows."""
        scales = None if self.constraints is None else self.constraints.row_scales()
        if scales is None:
            return None
        path = self.n_path * len(self.save_times)
        equality, inequality = scales
        return (
            jnp.concatenate([jnp.ones((self.n_terminal,), equality.dtype), equality]),
            jnp.concatenate([jnp.ones((path,), inequality.dtype), inequality]),
        )

    def __check_init__(self) -> None:
        """Refuse a control problem missing what its lowering needs, or with save times off its horizon."""
        missing = [
            name
            for name in ("controls", "cost", "save_times", "initial_state", "solver")
            if getattr(self, name) is None
        ]
        if self.model is None:
            missing.insert(0, "problem")
        if missing:
            raise ConfigurationError(f"ControlProblem needs {', '.join(missing)}.")
        if (self.path is None) != (self.n_path == 0) or (self.terminal is None) != (
            self.n_terminal == 0
        ):
            raise ConfigurationError(
                "ControlProblem: give path with n_path and terminal with n_terminal, or "
                "neither of each pair."
            )
        times = np.asarray(self.save_times, dtype=np.float64)
        horizon = self.controls.horizon
        if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > horizon:
            raise ConfigurationError(
                f"ControlProblem save_times must increase strictly within [0, {horizon}]."
            )


def _middle(bounds: tuple[ArrayLike | None, ArrayLike | None] | None, n: int) -> Float[Array, " n"]:
    """The middle of finite bounds, the finite bound where one side is open, zero where both are."""
    if bounds is None:
        return jnp.zeros((n,))
    lower, upper = (jnp.broadcast_to(jnp.asarray(side, dtype=jnp.float64), (n,)) for side in bounds)
    finite_lower, finite_upper = jnp.isfinite(lower), jnp.isfinite(upper)
    middle = jnp.where(finite_lower & finite_upper, 0.5 * (lower + upper), 0.0)
    middle = jnp.where(finite_lower & ~finite_upper, lower, middle)
    return jnp.where(~finite_lower & finite_upper, upper, middle)
