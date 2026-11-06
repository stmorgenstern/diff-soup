"""A control problem: cost, path and terminal functionals of a model's solution over a horizon."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._schedule import Schedule, Segment, schedule_arrays, solve_schedule
from diff_soup.optimization._abc import OptimizationProblem, _as_bounds
from diff_soup.optimization.control._horizon_cache import cached
from diff_soup.optimization.problem import AbstractOptimizationProblem


class PiecewiseConstant(eqx.Module):
    """Controls that hold one target of an operational row per interval of a fixed grid.

    Args:
        mode: The row of `spec.control.operational` the targets drive: `"current"`, `"voltage"`, `"power"` or `"resistance"`.
        intervals: The number of intervals, the length of the decision vector.
        duration: The length of each interval, in the model's time unit.
        carry_step: As `ds.Schedule(carry_step=)`: `None` continues the integrator's step into the next interval, `False` starts each afresh.
    """

    mode: str = eqx.field(static=True)
    intervals: int = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    carry_step: bool | None = eqx.field(static=True, default=None)

    @property
    def horizon(self) -> float:
        """The length of the horizon, `intervals * duration`."""
        return self.intervals * self.duration

    def segments(self, u: Any, modes: tuple) -> Any:
        """The schedule arrays of the horizon for the decision vector `u`, traced, never retraced for a new `u`.

        Args:
            u: The target of each interval, `(intervals,)`.
            modes: The model's mode names, `problem.operational_modes()`.

        Returns:
            The [ScheduleArrays][diff_soup.ScheduleArrays] whose `target` leaf is `u`.
        """
        template = schedule_arrays(
            Schedule(
                [Segment(**{self.mode: 0.0}, duration=self.duration)] * self.intervals,
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
        return dataclasses.replace(template, target=u)


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
        constraints: [Constraints][diff_soup.optimization.Constraints] on `u` itself, functions of `(u, args)`, or `None`.
        save_times: The times the solution is sampled at for the functionals, after the horizon's start at 0.
        initial_state: The state the horizon starts from; an `args` leaf.
        theta: The model parameters; an `args` leaf.
        limits: What the functionals read as limits, such as a voltage ceiling; an `args` leaf.
        start: The first guess of `u`; the middle of the bounds, or zeros, when `None`.
        solver: The time-stepping solver and its tolerances.
        adjoint: The differentiation strategy of the horizon solve, `None` for forward mode.
        layout: The placement of a batch of horizons.
    """

    # The model holds arrays, so it is a leaf, not static; the closures are keyed by its identity.
    problem: Any = None
    controls: Any = eqx.field(static=True, default=None)
    cost: Callable | None = eqx.field(static=True, default=None)
    path: Callable | None = eqx.field(static=True, default=None)
    terminal: Callable | None = eqx.field(static=True, default=None)
    n_path: int = eqx.field(static=True, default=0)
    n_terminal: int = eqx.field(static=True, default=0)
    save_times: Any = None
    initial_state: Any = None
    theta: Any = None
    limits: Any = None
    start: Any = None
    solver: Any = eqx.field(static=True, default=None)
    adjoint: Any = eqx.field(static=True, default=None)
    layout: Any = eqx.field(static=True, default=None)

    def numeric_problem(self, optimizer: Any) -> Any:
        """Lower to closures over `u` and `args`; see [AbstractOptimizationProblem.numeric_problem][diff_soup.optimization.AbstractOptimizationProblem.numeric_problem]."""
        self._check()
        n = self.controls.intervals
        closures = self.closures()
        start = self.start
        if start is None:
            start = _middle(self.bounds, n)
        x0 = np.asarray(start, dtype=np.float64).reshape((n,))
        structure = {"value_gradient", "hessian", "constraint_jacobian", "lagrangian_hessian"}
        if closures.n_equality or closures.n_inequality:
            structure.add("constraints")
        return OptimizationProblem(
            x0=x0,
            structure=frozenset(structure),
            bounds=_as_bounds(self.bounds, x0),
            args=self.args(),
            n_equality=closures.n_equality,
            n_inequality=closures.n_inequality,
            constraint_scales=None if self.constraints is None else self.constraints.scales,
            ad="forward" if self.adjoint is None else "reverse",
            _value=closures.value,
            _value_and_gradient=closures.value_and_gradient,
            _value_gradient_hessian=closures.value_gradient_hessian,
            _constraints=closures.constraints,
            _constraints_and_jacobians=closures.constraints_and_jacobians,
            _lagrangian_hessian=closures.lagrangian_hessian,
            _evaluate=closures.evaluate,
            _evaluate_with_derivatives=closures.evaluate_with_derivatives,
        )

    def args(self) -> Any:
        """`(initial_state, theta, limits)`, the leaves a batch varies."""
        return (self.initial_state, self.theta, self.limits)

    def closures(self) -> Any:
        """The `(u, args)` closures of the horizon, built once per problem and configuration.

        A second `ControlProblem` over the same model with the same functionals, controls,
        solver and save times reuses them, so a solver compiled for one horizon serves the
        next without retracing.
        """
        self._check()
        times = tuple(float(t) for t in np.asarray(self.save_times, dtype=np.float64).ravel())
        constraints = self.constraints
        key = (
            self.controls,
            self.cost,
            self.path,
            self.terminal,
            self.n_path,
            self.n_terminal,
            None if constraints is None else _constraints_key(constraints),
            times,
            self.solver,
            self.adjoint,
        )
        return cached(self.problem, key, lambda: _horizon_closures(self, times))

    def _check(self) -> None:
        """Refuse a control problem missing what its lowering needs."""
        missing = [
            name
            for name in ("problem", "controls", "cost", "save_times", "initial_state", "solver")
            if getattr(self, name) is None
        ]
        if missing:
            raise ConfigurationError(f"ControlProblem needs {', '.join(missing)}.")
        if (self.path is None) != (self.n_path == 0) or (self.terminal is None) != (
            self.n_terminal == 0
        ):
            raise ConfigurationError(
                "ControlProblem: give path with n_path and terminal with n_terminal, or "
                "neither of each pair."
            )
        times = np.asarray(self.save_times, dtype=np.float64).ravel()
        horizon = self.controls.horizon
        if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > horizon:
            raise ConfigurationError(
                f"ControlProblem save_times must increase strictly within [0, {horizon}]."
            )


@dataclasses.dataclass(frozen=True)
class _Closures:
    """The closures of a horizon, each taking `(u, args)` with `args = (state, theta, limits)`."""

    value: Callable
    value_and_gradient: Callable
    value_gradient_hessian: Callable
    constraints: Callable
    constraints_and_jacobians: Callable
    lagrangian_hessian: Callable
    evaluate: Callable
    evaluate_with_derivatives: Callable
    n_equality: int
    n_inequality: int


def _constraints_key(constraints: Any) -> tuple:
    """The static configuration of a `Constraints`, which keys the closures it builds."""
    return (
        constraints.equality,
        constraints.inequality,
        constraints.n_equality,
        constraints.n_inequality,
    )


def _middle(bounds: Any, n: int) -> np.ndarray:
    """The middle of finite bounds, the finite bound where one side is open, zero where both are."""
    if bounds is None:
        return np.zeros((n,))
    lower, upper = (np.broadcast_to(np.asarray(side, dtype=np.float64), (n,)) for side in bounds)
    middle = np.where(np.isfinite(lower) & np.isfinite(upper), 0.5 * (lower + upper), 0.0)
    middle = np.where(np.isfinite(lower) & ~np.isfinite(upper), lower, middle)
    return np.where(~np.isfinite(lower) & np.isfinite(upper), upper, middle)


def _horizon_closures(control: ControlProblem, times: tuple) -> _Closures:
    """Build the closures: one schedule solve per evaluation, its rows stacked for AD."""
    model, controls, solver, adjoint = (
        control.problem,
        control.controls,
        control.solver,
        control.adjoint,
    )
    cost, path, terminal = control.cost, control.path, control.terminal
    extra = control.constraints
    modes = tuple(model.operational_modes())
    n = controls.intervals
    save_times = np.asarray(times, dtype=np.float64)
    n_path_rows = control.n_path * save_times.size
    n_equality = control.n_terminal + (0 if extra is None else extra.n_equality)
    n_inequality = n_path_rows + (0 if extra is None else extra.n_inequality)
    n_rows = 1 + n_equality + n_inequality
    # Reverse mode needs an adjoint that supports it, and pays when columns exceed rows.
    reverse_all = adjoint is not None and n > n_rows
    reverse_cost = adjoint is not None and n > 1
    reverse_rows = adjoint is not None and n > n_equality + n_inequality
    extra_rows = None if extra is None else extra.lower(None, _identity).constraints

    def rows(u, args):
        state, theta, _limits = args
        solution, _ = solve_schedule(
            model,
            solver,
            controls.segments(u, modes),
            state,
            t0=0.0,
            t_end=controls.horizon,
            times=jnp.asarray(save_times),
            theta=theta,
            adjoint=adjoint,
        )
        value = jnp.reshape(jnp.asarray(cost(solution, u, args)), ())
        equality = [_sized(terminal, solution, u, args, control.n_terminal, "terminal")]
        inequality = [_sized(path, solution, u, args, n_path_rows, "path")]
        if extra_rows is not None:
            on_u = extra_rows(u, args)
            equality.append(on_u[0])
            inequality.append(on_u[1])
        dtype = jnp.result_type(value)
        equality = jnp.concatenate([jnp.asarray(r, dtype) for r in equality])
        inequality = jnp.concatenate([jnp.asarray(r, dtype) for r in inequality])
        # A failed horizon solve is a rejected trial (M5): every output is NaN.
        ok = solution.ok & jnp.all(solution.valid_mask)
        fail = lambda r: jnp.where(ok, r, jnp.nan)  # noqa: E731
        return fail(value), fail(equality), fail(inequality)

    def stacked(u, args):
        value, equality, inequality = rows(u, args)
        return jnp.concatenate([value[None], equality, inequality])

    def split(vector):
        return vector[0], vector[1 : 1 + n_equality], vector[1 + n_equality :]

    def value(u, args):
        return rows(u, args)[0]

    def value_and_gradient(u, args):
        gradient, primal = (jax.jacrev if reverse_cost else jax.jacfwd)(
            lambda x: (value(x, args),) * 2, has_aux=True
        )(u)
        return primal, gradient

    def value_gradient_hessian(u, args):
        gradient = (jax.grad if reverse_cost else jax.jacfwd)(lambda x: value(x, args))
        hessian = jax.jacfwd(gradient)(u)
        primal, first = value_and_gradient(u, args)
        return primal, first, hessian

    def constraints(u, args):
        return rows(u, args)[1:]

    def constraint_vector(u, args):
        return jnp.concatenate(rows(u, args)[1:])

    def constraints_and_jacobians(u, args):
        jacobian, vector = (jax.jacrev if reverse_rows else jax.jacfwd)(
            lambda x: (constraint_vector(x, args),) * 2, has_aux=True
        )(u)
        return (
            vector[:n_equality],
            vector[n_equality:],
            jacobian[:n_equality],
            jacobian[n_equality:],
        )

    def evaluate(u, args):
        return rows(u, args)

    def evaluate_with_derivatives(u, args):
        jacobian, vector = (jax.jacrev if reverse_all else jax.jacfwd)(
            lambda x: (stacked(x, args),) * 2, has_aux=True
        )(u)
        value, equality, inequality = split(vector)
        gradient, jac_eq, jac_ineq = split(jacobian)
        return value, gradient, equality, inequality, jac_eq, jac_ineq

    def lagrangian(u, lam_eq, lam_ineq, args):
        value, equality, inequality = rows(u, args)
        return value + lam_eq @ equality + lam_ineq @ inequality

    def lagrangian_hessian(u, lam_eq, lam_ineq, args):
        first = jax.grad if reverse_cost else jax.jacfwd
        return jax.jacfwd(first(lagrangian))(u, lam_eq, lam_ineq, args)

    return _Closures(
        value=value,
        value_and_gradient=value_and_gradient,
        value_gradient_hessian=value_gradient_hessian,
        constraints=constraints,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=lagrangian_hessian,
        evaluate=evaluate,
        evaluate_with_derivatives=evaluate_with_derivatives,
        n_equality=n_equality,
        n_inequality=n_inequality,
    )


def _identity(u: Any) -> Any:
    """The decision vector as the view the `Constraints` on `u` take."""
    return u


def _sized(function: Callable | None, solution: Any, u: Any, args: Any, count: int, name: str):
    """The rows `function` returns, flattened and checked against `count`; empty without one."""
    if function is None:
        return jnp.zeros((0,))
    values = jnp.reshape(jnp.asarray(function(solution, u, args)), (-1,))
    if values.shape[0] != count:
        raise ConfigurationError(
            f"ControlProblem {name} returned {values.shape[0]} rows; it declares {count}."
        )
    return values
