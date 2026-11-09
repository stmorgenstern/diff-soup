"""The closures of a control horizon: one schedule solve per evaluation, its rows stacked for AD.

A [ControlProblem][diff_soup.optimization.control.ControlProblem] lowers to closures that
read only its static configuration (the model, the functionals, the controls, the save
times, the solver) and take `(u, args)`. `Horizon` holds that configuration and compares
by it, so two control problems of one configuration have equal numeric problems and a
compiled solver reuses its compile, with no cache of closures.
"""

from __future__ import annotations

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
    save_times: tuple[float, ...]
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
    def n_inequality(self) -> int:
        """The path rows at every save time, then the inequality rows of the `Constraints` on `u`."""
        configuration = self.configuration
        extra = configuration.constraints
        path = configuration.n_path * len(configuration.save_times)
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
    times = np.asarray(configuration.save_times, dtype=np.float64)
    n_path = configuration.n_path * times.size
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
        solution, _ = solve_schedule(
            model,
            configuration.solver,
            controls.segments(u, modes),
            state,
            t0=0.0,
            t_end=controls.horizon,
            times=jnp.asarray(times),
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


__all__ = ["Configuration", "Horizon"]
