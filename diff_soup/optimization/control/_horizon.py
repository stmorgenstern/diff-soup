"""The closures of a control horizon: one schedule solve per evaluation, its rows stacked for AD.

A [ControlProblem][diff_soup.optimization.control.ControlProblem] lowers to closures that
read only its static configuration (the model, the functionals, the controls, the save
times, the solver) and take `(u, args)`. `Horizon` holds that configuration and compares
by it, so two control problems of one configuration have equal numeric problems and a
compiled solver reuses its compile, with no cache of closures.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup._schedule import solve_schedule
from diff_soup.optimization._binding import Derived, Held, Lowered
from diff_soup.optimization._derived import identity

if TYPE_CHECKING:
    from diff_soup._adjoints import AbstractAdjoint
    from diff_soup.optimization.constraints import Constraints
    from diff_soup.optimization.control.problem import AbstractControls
    from diff_soup.timestepping.vsvo.classes import JBDF, JIDA, JNDF


class Configuration(NamedTuple):
    """What a horizon's closures read besides `(u, args)`; hashable, and free of arrays."""

    model: Held
    controls: AbstractControls
    cost: Callable
    path: Callable | None
    terminal: Callable | None
    n_path: int
    n_terminal: int
    constraints: Constraints | None
    save_times: tuple[float, ...] | None
    save_fractions: tuple[float, ...] | None
    solver: JIDA | JBDF | JNDF
    adjoint: AbstractAdjoint | None


class Horizon:
    """The closures of one horizon configuration, built on first use and equal by configuration.

    Attributes:
        configuration: The [Configuration][diff_soup.optimization.control._horizon.Configuration].
    """

    __slots__ = ("configuration", "_built", "__weakref__")

    def __init__(self, configuration: Configuration):
        """Hold `configuration`; the closures are built when first called."""
        self.configuration = configuration
        self._built = None

    def __eq__(self, other: object) -> bool:
        """Equal when both hold the same configuration."""
        return isinstance(other, Horizon) and other.configuration == self.configuration

    def __hash__(self) -> int:
        """The configuration's hash."""
        return hash(self.configuration)

    @property
    def n_equality(self) -> int:
        """The terminal rows, then the equality rows of the `Constraints` on `u`."""
        extra = self.configuration.constraints
        return self.configuration.n_terminal + (0 if extra is None else extra.n_equality)

    @property
    def n_samples(self) -> int:
        """The saved times: the save times, or the save fractions in every segment."""
        return _sample_count(self.configuration)

    @property
    def n_inequality(self) -> int:
        """The path rows at every saved time, then the inequality rows of the `Constraints` on `u`."""
        extra = self.configuration.constraints
        path = self.configuration.n_path * self.n_samples
        return path + (0 if extra is None else extra.n_inequality)

    def closure(self, name: str) -> Callable:
        """The closure `name`, taking `args` last, equal across equal horizons."""
        return Derived(_call, self, name)

    def built(self) -> Lowered:
        """The closures by name, built once per object."""
        if self._built is None:
            self._built = _build(self.configuration)
        return self._built


def _call(horizon: Horizon, name: str, *operands: PyTree) -> PyTree:
    """Call the closure `name` of `horizon`."""
    return horizon.built()[name](*operands)


def _build(configuration: Configuration) -> Lowered:
    """The closures of a horizon configuration."""
    model, controls = configuration.model.value, configuration.controls
    n_terminal = configuration.n_terminal
    fractions = configuration.save_fractions
    times = None if fractions is not None else np.asarray(configuration.save_times, np.float64)
    n_path = configuration.n_path * _sample_count(configuration)
    extra = configuration.constraints
    lowered = None if extra is None else extra.lower(None, identity)
    n = controls.size
    modes = tuple(model.operational_modes())
    reverse = configuration.adjoint is not None
    # Reverse mode needs an adjoint that supports it, and pays when columns exceed rows.
    reverse_cost = reverse and n > 1
    reverse_rows = reverse and n > n_terminal + n_path
    reverse_all = reverse and n > 1 + n_terminal + n_path

    def horizon(u, args):
        state, theta, _ = args
        schedule = controls.segments(u, modes)
        if fractions is None:
            t_end, sampled = controls.horizon, jnp.asarray(times)
        else:
            # The schedule and its sample times read one stored copy of the durations: a
            # product fused into their sums (an FMA) would put a segment's end and the sample
            # at its end an ulp apart, and the horizon's end short of the last segment's.
            durations = jax.lax.optimization_barrier(jnp.asarray(schedule.duration))
            schedule = dataclasses.replace(schedule, duration=durations)
            t_end, sampled = sample_times(durations, fractions)
        solution, _ = solve_schedule(
            model,
            configuration.solver,
            schedule,
            state,
            t0=0.0,
            t_end=t_end,
            times=sampled,
            theta=theta,
            adjoint=configuration.adjoint,
        )
        value = jnp.reshape(jnp.asarray(configuration.cost(solution, u, args)), ())
        dtype = jnp.result_type(value)
        terminal = _sized(configuration.terminal, solution, u, args, n_terminal, "terminal")
        path = _sized(configuration.path, solution, u, args, n_path, "path")
        stacked = jnp.concatenate([value[None], terminal.astype(dtype), path.astype(dtype)])
        # A failed horizon solve is a rejected trial: every output is NaN.
        ok = solution.ok & jnp.all(solution.valid_mask)
        return jnp.where(ok, stacked, jnp.nan)

    def split(stacked):
        return stacked[0], stacked[1 : 1 + n_terminal], stacked[1 + n_terminal :]

    def jacobian(function, reverse_mode, u):
        """`(J, f(u))` of a vector function, in one pass."""
        differentiate = jax.jacrev if reverse_mode else jax.jacfwd
        return differentiate(lambda x: (function(x),) * 2, has_aux=True)(u)

    def extra_rows(u, args):
        if lowered is None:
            empty = jnp.zeros((0,), jnp.result_type(u))
            return empty, empty
        return lowered["constraints"](u, args)

    def extra_rows_and_jacobians(u, args):
        if lowered is None:
            empty, block = (
                jnp.zeros((0,), jnp.result_type(u)),
                jnp.zeros((0, n), jnp.result_type(u)),
            )
            return empty, empty, block, block
        return lowered["constraints_and_jacobians"](u, args)

    def joined(terminal, path, extra):
        return jnp.concatenate([terminal, extra[0]]), jnp.concatenate([path, extra[1]])

    def value(u, args):
        return horizon(u, args)[0]

    def value_and_gradient(u, args):
        gradient, primal = jacobian(lambda x: value(x, args), reverse_cost, u)
        return primal, gradient

    def value_gradient_hessian(u, args):
        def gradient(x):
            primal, slope = value_and_gradient(x, args)
            return slope, (primal, slope)

        hessian, (primal, slope) = jax.jacfwd(gradient, has_aux=True)(u)
        return primal, slope, hessian

    def constraints(u, args):
        _, terminal, path = split(horizon(u, args))
        return joined(terminal, path, extra_rows(u, args))

    def constraints_and_jacobians(u, args):
        rows, stacked = jacobian(lambda x: horizon(x, args)[1:], reverse_rows, u)
        eq, ineq, jac_eq, jac_ineq = extra_rows_and_jacobians(u, args)
        return (
            jnp.concatenate([stacked[:n_terminal], eq]),
            jnp.concatenate([stacked[n_terminal:], ineq]),
            jnp.concatenate([rows[:n_terminal], jac_eq]),
            jnp.concatenate([rows[n_terminal:], jac_ineq]),
        )

    def evaluate(u, args):
        cost, terminal, path = split(horizon(u, args))
        return (cost, *joined(terminal, path, extra_rows(u, args)))

    def evaluate_with_derivatives(u, args):
        rows, stacked = jacobian(lambda x: horizon(x, args), reverse_all, u)
        cost, terminal, path = split(stacked)
        gradient, jac_terminal, jac_path = split(rows)
        eq, ineq, jac_eq, jac_ineq = extra_rows_and_jacobians(u, args)
        return (
            cost,
            gradient,
            jnp.concatenate([terminal, eq]),
            jnp.concatenate([path, ineq]),
            jnp.concatenate([jac_terminal, jac_eq]),
            jnp.concatenate([jac_path, jac_ineq]),
        )

    def weighted(u, lam_eq, lam_ineq, args, objective):
        stacked = horizon(u, args)
        weights = jnp.concatenate(
            [
                jnp.asarray([1.0 if objective else 0.0], stacked.dtype),
                lam_eq[:n_terminal],
                lam_ineq[:n_path],
            ]
        )
        return weights @ stacked

    def second_order(u, lam_eq, lam_ineq, args, objective):
        first = jax.grad if reverse_cost else jax.jacfwd
        horizon_part = jax.jacfwd(first(lambda x: weighted(x, lam_eq, lam_ineq, args, objective)))
        hessian = horizon_part(u)
        if lowered is not None:
            hessian = hessian + lowered["lagrangian_hessian"](
                u, lam_eq[n_terminal:], lam_ineq[n_path:], args
            )
        return hessian

    def lagrangian_hessian(u, lam_eq, lam_ineq, args):
        return second_order(u, lam_eq, lam_ineq, args, True)

    def constraint_hessian(u, lam_eq, lam_ineq, args):
        return second_order(u, lam_eq, lam_ineq, args, False)

    return {
        "value": value,
        "value_and_gradient": value_and_gradient,
        "value_gradient_hessian": value_gradient_hessian,
        "constraints": constraints,
        "constraints_and_jacobians": constraints_and_jacobians,
        "lagrangian_hessian": lagrangian_hessian,
        "constraint_hessian": constraint_hessian,
        "evaluate": evaluate,
        "evaluate_with_derivatives": evaluate_with_derivatives,
    }


def _sample_count(configuration: Configuration) -> int:
    """The saved times of a configuration: its save times, or its fractions in every segment."""
    if configuration.save_times is not None:
        return len(configuration.save_times)
    return len(configuration.save_fractions) * len(configuration.controls.durations)


def sample_times(durations, fractions: tuple[float, ...], xp=jnp):
    """The end of a horizon and its times at `fractions` of every segment, segment after segment.

    The segments' starts are the durations added in order, as `solve_schedule` adds them, so
    a fraction of one is a segment's own end and the last such time is the horizon's end.

    Args:
        durations: The length of each segment, traced or not.
        fractions: The positions within a segment, increasing within `(0, 1]`.
        xp: `jax.numpy` in a trace, `numpy` for static durations.

    Returns:
        `(end, times)`, the times of shape `(segments * fractions,)`.
    """
    starts, end = [], durations[0] * 0
    for index in range(durations.shape[0]):
        starts.append(end)
        end = end + durations[index]
    starts = xp.stack(starts)
    within = xp.asarray(fractions, dtype=durations.dtype)
    return end, xp.reshape(starts[:, None] + within[None, :] * durations[:, None], (-1,))


def _sized(
    function: Callable | None, solution: PyTree, u: ArrayLike, args: PyTree, count: int, name: str
):
    """The rows `function` returns, flattened and checked against `count`; empty without one."""
    if function is None:
        return jnp.zeros((0,))
    values = jnp.reshape(jnp.asarray(function(solution, u, args)), (-1,))
    if values.shape[0] != count:
        raise ConfigurationError(
            f"ControlProblem {name} returned {values.shape[0]} rows; it declares {count}."
        )
    return values


__all__ = ["Configuration", "Horizon", "sample_times"]
