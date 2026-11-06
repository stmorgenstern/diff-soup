"""The closures a compiled solver reads from a numeric problem, each taking `(x, args)`.

A compiled solver never calls a problem's bound properties: it binds the raw closures and
passes `args` itself, so one compiled program serves every lane. This module is the single
place that knows where a problem keeps them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp


class Closures(NamedTuple):
    """The raw closures of a numeric problem and its static sizes.

    Attributes:
        value: `(x, args) -> f`.
        value_and_gradient: `(x, args) -> (f, g)`.
        constraints: `(x, args) -> (equality, inequality)`.
        constraints_and_jacobians: `(x, args) -> (equality, inequality, J_eq, J_ineq)`.
        lagrangian_hessian: `(x, lam_eq, lam_ineq, args) -> H`, or `None`.
        is_valid: `(x, args) -> bool`, or `None` when every point is valid.
        n_equality: The number of equality rows.
        n_inequality: The number of inequality rows.
    """

    value: Callable
    value_and_gradient: Callable
    constraints: Callable
    constraints_and_jacobians: Callable
    lagrangian_hessian: Callable | None
    is_valid: Callable | None
    n_equality: int
    n_inequality: int


def closures_of(problem: Any) -> Closures:
    """The raw `(x, args)` closures of `problem`.

    The numeric problem keeps them in its private fields `_value`, `_value_and_gradient`,
    `_constraints`, `_constraints_and_jacobians`, `_lagrangian_hessian` and `_is_valid`, with
    the row counts `n_equality` and `n_inequality` static.
    """
    return Closures(
        value=problem._value,
        value_and_gradient=problem._value_and_gradient,
        constraints=problem._constraints,
        constraints_and_jacobians=problem._constraints_and_jacobians,
        lagrangian_hessian=getattr(problem, "_lagrangian_hessian", None),
        is_valid=getattr(problem, "_is_valid", None),
        n_equality=int(problem.n_equality),
        n_inequality=int(problem.n_inequality),
    )


def bounds_of(problem: Any, x0: Any) -> tuple[Any, Any]:
    """The `(lower, upper)` bounds of `problem` aligned with `x0`, infinite where absent."""
    bounds = getattr(problem, "bounds", None)
    if bounds is None:
        infinite = jnp.full(jnp.shape(x0), jnp.inf, dtype=jnp.result_type(x0))
        return -infinite, infinite
    lower, upper = bounds
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype=jnp.result_type(x0)), jnp.shape(x0))
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype=jnp.result_type(x0)), jnp.shape(x0))
    return lower, upper


class Evaluation(eqx.Module):
    """The objective and constraints with their derivatives at one point.

    Attributes:
        value: The objective.
        gradient: Its gradient, `(n,)`.
        equality: The equality rows, `(n_eq,)`.
        inequality: The inequality rows, `(n_ineq,)`.
        jacobian_equality: `(n_eq, n)`.
        jacobian_inequality: `(n_ineq, n)`.
    """

    value: Any
    gradient: Any
    equality: Any
    inequality: Any
    jacobian_equality: Any
    jacobian_inequality: Any


def evaluate(closures: Closures, x: Any, args: Any) -> Evaluation:
    """The objective, the constraints and their first derivatives at `x`."""
    value, gradient = closures.value_and_gradient(x, args)
    equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(x, args)
    return Evaluation(value, gradient, equality, inequality, jac_eq, jac_ineq)


def trial(closures: Closures, x: Any, args: Any) -> tuple[Any, Any, Any, Any]:
    """`(value, equality, inequality, valid)` at a trial point, without derivatives.

    `valid` is false where the problem's trial predicate refuses the point or any value is
    not finite: a failed evaluation the line search rejects.
    """
    value = closures.value(x, args)
    equality, inequality = closures.constraints(x, args)
    valid = (
        jnp.isfinite(value) & jnp.all(jnp.isfinite(equality)) & jnp.all(jnp.isfinite(inequality))
    )
    if closures.is_valid is not None:
        valid = valid & closures.is_valid(x, args)
    return value, equality, inequality, valid
