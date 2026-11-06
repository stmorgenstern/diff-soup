"""The closures a compiled solver reads from a numeric problem, each taking `(x, args)`.

A compiled solver never calls a problem's bound properties: it binds the raw closures and
passes `args` itself, so one compiled program serves every lane. This module is the single
place that knows where a problem keeps them. Scaling is a transformation of the closures:
[scaled_closures][] returns closures of the scaled problem, which take `(args, scales)`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp

G_MAX = 100.0
SCALE_MIN = 1e-8


class Closures(NamedTuple):
    """The raw closures of a numeric problem and its static sizes.

    Attributes:
        value: `(x, args) -> f`.
        value_and_gradient: `(x, args) -> (f, g)`.
        constraints: `(x, args) -> (equality, inequality)`.
        constraints_and_jacobians: `(x, args) -> (equality, inequality, J_eq, J_ineq)`.
        lagrangian_hessian: `(x, lam_eq, lam_ineq, args) -> H`, or `None`.
        gauss_newton: `(x, args) -> (f, g, J_r^T J_r, failed)` for a least-squares objective, or `None`.
        is_valid: `(x, args) -> bool`, or `None` when every point is valid.
        n_equality: The number of equality rows.
        n_inequality: The number of inequality rows.
        evaluate: `(x, args) -> (f, eq, ineq)` from one evaluation, or `None`.
        evaluate_with_derivatives: `(x, args) -> (f, g, eq, ineq, J_eq, J_ineq)` from one
            evaluation, or `None`.
    """

    value: Callable
    value_and_gradient: Callable
    constraints: Callable
    constraints_and_jacobians: Callable
    lagrangian_hessian: Callable | None
    gauss_newton: Callable | None
    is_valid: Callable | None
    n_equality: int
    n_inequality: int
    evaluate: Callable | None = None
    evaluate_with_derivatives: Callable | None = None


def closures_of(problem: Any) -> Closures:
    """The raw `(x, args)` closures of `problem`.

    The numeric problem keeps them in its private fields `_value`, `_value_and_gradient`,
    `_constraints`, `_constraints_and_jacobians`, `_lagrangian_hessian`, `_gauss_newton` and
    `_is_valid`, with the row counts `n_equality` and `n_inequality` static.
    """
    structure = getattr(problem, "structure", None)
    gauss_newton = getattr(problem, "_gauss_newton", None)
    if structure is not None and "gauss_newton" not in structure:
        gauss_newton = None
    return Closures(
        value=problem._value,
        value_and_gradient=problem._value_and_gradient,
        constraints=problem._constraints,
        constraints_and_jacobians=problem._constraints_and_jacobians,
        lagrangian_hessian=getattr(problem, "_lagrangian_hessian", None),
        gauss_newton=gauss_newton,
        is_valid=getattr(problem, "_is_valid", None),
        n_equality=int(problem.n_equality),
        n_inequality=int(problem.n_inequality),
        evaluate=getattr(problem, "_evaluate", None),
        evaluate_with_derivatives=getattr(problem, "_evaluate_with_derivatives", None),
    )


def _first_derivatives(closures: Closures, x: Any, args: Any) -> tuple:
    """`(f, g, eq, ineq, J_eq, J_ineq)`, from one call when the problem gives one."""
    if closures.evaluate_with_derivatives is not None:
        return closures.evaluate_with_derivatives(x, args)
    value, gradient = closures.value_and_gradient(x, args)
    return (value, gradient, *closures.constraints_and_jacobians(x, args))


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


class Scales(eqx.Module):
    """The factors the objective and each constraint row are multiplied by.

    Attributes:
        objective: A scalar.
        equality: `(n_eq,)`.
        inequality: `(n_ineq,)`.
    """

    objective: Any
    equality: Any
    inequality: Any


def unit_scales(closures: Closures, dtype: Any) -> Scales:
    """Scales of 1: the problem as given."""
    return Scales(
        jnp.ones((), dtype),
        jnp.ones((closures.n_equality,), dtype),
        jnp.ones((closures.n_inequality,), dtype),
    )


def gradient_scales(closures: Closures, x: Any, args: Any) -> Scales:
    """Ipopt's gradient-based scaling at `x`: each gradient's largest entry brought to `G_MAX`.

    A factor is `min(1, G_MAX / ||gradient||_inf)`, at least `SCALE_MIN`, and 1 where the
    gradient is not finite.
    """
    _, gradient, _, _, jac_eq, jac_ineq = _first_derivatives(closures, x, args)

    def factor(norm):
        scale = jnp.clip(G_MAX / jnp.maximum(norm, jnp.finfo(x.dtype).tiny), SCALE_MIN, 1.0)
        return jnp.where(jnp.isfinite(norm), scale, 1.0)

    return Scales(
        factor(jnp.max(jnp.abs(gradient), initial=0.0)),
        factor(jnp.max(jnp.abs(jac_eq), axis=1, initial=0.0)),
        factor(jnp.max(jnp.abs(jac_ineq), axis=1, initial=0.0)),
    )


def problem_scales(problem: Any, closures: Closures, x: Any, args: Any) -> Scales:
    """The problem's own `objective_scale` and `constraint_scales`, else the gradient-based ones."""
    dtype = x.dtype
    objective = getattr(problem, "objective_scale", None)
    rows = getattr(problem, "constraint_scales", None)
    if objective is None and rows is None:
        return gradient_scales(closures, x, args)
    unit = unit_scales(closures, dtype)
    equality, inequality = (unit.equality, unit.inequality) if rows is None else rows
    return Scales(
        jnp.asarray(1.0 if objective is None else objective, dtype),
        jnp.asarray(equality, dtype),
        jnp.asarray(inequality, dtype),
    )


def scaled_closures(closures: Closures) -> Closures:
    """The closures of the scaled problem, which take `args = (args, scales)`."""

    def value(x, packed):
        args, scales = packed
        return scales.objective * closures.value(x, args)

    def value_and_gradient(x, packed):
        args, scales = packed
        f, g = closures.value_and_gradient(x, args)
        return scales.objective * f, scales.objective * g

    def constraints(x, packed):
        args, scales = packed
        equality, inequality = closures.constraints(x, args)
        return scales.equality * equality, scales.inequality * inequality

    def constraints_and_jacobians(x, packed):
        args, scales = packed
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(x, args)
        return (
            scales.equality * equality,
            scales.inequality * inequality,
            scales.equality[:, None] * jac_eq,
            scales.inequality[:, None] * jac_ineq,
        )

    def lagrangian_hessian(x, lam_e, lam_i, packed):
        args, scales = packed
        objective = scales.objective
        return objective * closures.lagrangian_hessian(
            x, lam_e * scales.equality / objective, lam_i * scales.inequality / objective, args
        )

    def gauss_newton(x, packed):
        args, scales = packed
        f, g, matrix, failed = closures.gauss_newton(x, args)
        return scales.objective * f, scales.objective * g, scales.objective * matrix, failed

    def is_valid(x, packed):
        return closures.is_valid(x, packed[0])

    def evaluate_once(x, packed):
        args, scales = packed
        f, equality, inequality = closures.evaluate(x, args)
        return scales.objective * f, scales.equality * equality, scales.inequality * inequality

    def evaluate_with_derivatives(x, packed):
        args, scales = packed
        f, g, equality, inequality, jac_eq, jac_ineq = closures.evaluate_with_derivatives(x, args)
        return (
            scales.objective * f,
            scales.objective * g,
            scales.equality * equality,
            scales.inequality * inequality,
            scales.equality[:, None] * jac_eq,
            scales.inequality[:, None] * jac_ineq,
        )

    return Closures(
        value=value,
        value_and_gradient=value_and_gradient,
        constraints=constraints,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=None if closures.lagrangian_hessian is None else lagrangian_hessian,
        gauss_newton=None if closures.gauss_newton is None else gauss_newton,
        is_valid=None if closures.is_valid is None else is_valid,
        n_equality=closures.n_equality,
        n_inequality=closures.n_inequality,
        evaluate=None if closures.evaluate is None else evaluate_once,
        evaluate_with_derivatives=(
            None if closures.evaluate_with_derivatives is None else evaluate_with_derivatives
        ),
    )


class Evaluation(eqx.Module):
    """The objective and constraints with their derivatives at one point.

    Attributes:
        value: The objective.
        gradient: Its gradient, `(n,)`.
        equality: The equality rows, `(n_eq,)`.
        inequality: The inequality rows, `(n_ineq,)`.
        jacobian_equality: `(n_eq, n)`.
        jacobian_inequality: `(n_ineq, n)`.
        gauss_newton: The Gauss-Newton matrix, `(n, n)`, or `(0, 0)` when not used.
    """

    value: Any
    gradient: Any
    equality: Any
    inequality: Any
    jacobian_equality: Any
    jacobian_inequality: Any
    gauss_newton: Any

    @property
    def finite(self) -> Any:
        """Whether every entry is finite."""
        return (
            jnp.isfinite(self.value)
            & jnp.all(jnp.isfinite(self.gradient))
            & jnp.all(jnp.isfinite(self.equality))
            & jnp.all(jnp.isfinite(self.inequality))
            & jnp.all(jnp.isfinite(self.jacobian_equality))
            & jnp.all(jnp.isfinite(self.jacobian_inequality))
            & jnp.all(jnp.isfinite(self.gauss_newton))
        )


def evaluate(closures: Closures, x: Any, args: Any, gauss_newton: bool = False) -> Evaluation:
    """The objective, the constraints and their first derivatives at `x`.

    With `gauss_newton`, the objective, gradient and Gauss-Newton matrix come from one call of
    the problem's `gauss_newton` closure, and a reported failure makes the value NaN.
    """
    if gauss_newton:
        value, gradient, matrix, failed = closures.gauss_newton(x, args)
        value = jnp.where(jnp.asarray(failed) > 0, jnp.nan, value)
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(x, args)
    else:
        value, gradient, equality, inequality, jac_eq, jac_ineq = _first_derivatives(
            closures, x, args
        )
        matrix = jnp.zeros((0, 0), dtype=x.dtype)
    return Evaluation(value, gradient, equality, inequality, jac_eq, jac_ineq, matrix)


def trial(closures: Closures, x: Any, args: Any) -> tuple[Any, Any, Any, Any]:
    """`(value, equality, inequality, valid)` at a trial point, without derivatives.

    `valid` is false where the problem's trial predicate refuses the point or any value is
    not finite: a failed evaluation the line search rejects.
    """
    if closures.evaluate is not None:
        value, equality, inequality = closures.evaluate(x, args)
    else:
        value = closures.value(x, args)
        equality, inequality = closures.constraints(x, args)
    valid = (
        jnp.isfinite(value) & jnp.all(jnp.isfinite(equality)) & jnp.all(jnp.isfinite(inequality))
    )
    if closures.is_valid is not None:
        valid = valid & closures.is_valid(x, args)
    return value, equality, inequality, valid
