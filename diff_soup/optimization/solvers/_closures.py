"""The closures a compiled solver reads from a numeric problem, each taking `(x, args)`.

A compiled solver never calls a problem's bound properties: it takes the raw closures from
`problem.closure(name)` and passes `args` itself, so one compiled program serves every lane.
Scaling is a transformation of the closures: [scaled_closures][] returns closures of the
scaled problem, which take `(args, scales)`, and [Evaluation.scaled][] scales an evaluation of
the problem as given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._abc import OptimizationProblem

#: Ipopt's `nlp_scaling_max_gradient`: the largest gradient entry after gradient-based scaling.
G_MAX = 100.0
#: Ipopt's `nlp_scaling_min_value`: the least scaling factor.
SCALE_MIN = 1e-8


class Closures(NamedTuple):
    """The raw closures of a numeric problem and its static sizes.

    Attributes:
        evaluate: `(x, args) -> (f, eq, ineq)` from one evaluation.
        evaluate_with_derivatives: `(x, args) -> (f, g, eq, ineq, J_eq, J_ineq)` from one evaluation.
        constraints_and_jacobians: `(x, args) -> (eq, ineq, J_eq, J_ineq)`.
        lagrangian_hessian: `(x, lam_eq, lam_ineq, args) -> H`, or `None`.
        constraint_hessian: `(x, lam_eq, lam_ineq, args) -> H` of the constraint terms alone, or `None`.
        gauss_newton: `(x, args) -> (f, g, J_r^T J_r, failed)` for a least-squares objective, or `None`.
        is_valid: `(x, args) -> bool`.
        n_equality: The number of equality rows.
        n_inequality: The number of inequality rows.
        bounded: Whether the problem has bounds.
    """

    evaluate: Callable
    evaluate_with_derivatives: Callable
    constraints_and_jacobians: Callable
    lagrangian_hessian: Callable | None
    constraint_hessian: Callable | None
    gauss_newton: Callable | None
    is_valid: Callable
    n_equality: int
    n_inequality: int
    bounded: bool


def _optional(problem: OptimizationProblem, name: str) -> Callable | None:
    """`problem.closure(name)`, or `None` when the problem does not provide it."""
    try:
        return problem.closure(name)
    except ConfigurationError:
        return None


def closures_of(problem: OptimizationProblem) -> Closures:
    """The raw `(x, args)` closures of `problem`, each from `problem.closure(name)`.

    The constraint terms of the Lagrangian Hessian come from the problem's private
    `_constraint_hessian` closure when it has one.
    """
    constraint_hessian = _optional(problem, "constraint_hessian")
    if constraint_hessian is None:
        constraint_hessian = getattr(problem, "_constraint_hessian", None)
    return Closures(
        evaluate=problem.closure("evaluate"),
        evaluate_with_derivatives=problem.closure("evaluate_with_derivatives"),
        constraints_and_jacobians=problem.closure("constraints_and_jacobians"),
        lagrangian_hessian=_optional(problem, "lagrangian_hessian"),
        constraint_hessian=constraint_hessian,
        gauss_newton=_optional(problem, "gauss_newton"),
        is_valid=problem.closure("is_valid"),
        n_equality=int(problem.n_equality),
        n_inequality=int(problem.n_inequality),
        bounded=problem.bounds is not None,
    )


def bounds_of(
    problem: OptimizationProblem, x0: Float[ArrayLike, "n"]
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """The `(lower, upper)` bounds of `problem` aligned with `x0`, infinite where absent."""
    dtype = jnp.result_type(x0)
    if problem.bounds is None:
        infinite = jnp.full(jnp.shape(x0), jnp.inf, dtype=dtype)
        return -infinite, infinite
    lower, upper = problem.bounds
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype=dtype), jnp.shape(x0))
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype=dtype), jnp.shape(x0))
    return lower, upper


class Scales(eqx.Module):
    """The factors the objective and each constraint row are multiplied by.

    Attributes:
        objective: The objective's factor.
        equality: One factor per equality row.
        inequality: One factor per inequality row.
    """

    objective: Float[Array, ""]
    equality: Float[Array, "n_eq"]
    inequality: Float[Array, "n_ineq"]


class Evaluation(eqx.Module):
    """The objective and constraints with their derivatives at one point.

    Attributes:
        value: The objective.
        gradient: Its gradient.
        equality: The equality rows.
        inequality: The inequality rows.
        jacobian_equality: The equality rows' Jacobian.
        jacobian_inequality: The inequality rows' Jacobian.
        gauss_newton: The Gauss-Newton matrix, or `(0, 0)` when not used.
    """

    value: Float[Array, ""]
    gradient: Float[Array, "n"]
    equality: Float[Array, "n_eq"]
    inequality: Float[Array, "n_ineq"]
    jacobian_equality: Float[Array, "n_eq n"]
    jacobian_inequality: Float[Array, "n_ineq n"]
    gauss_newton: Float[Array, "k k"]

    @property
    def rows_finite(self) -> Bool[Array, ""]:
        """Whether the constraint rows and their Jacobians are finite."""
        return (
            jnp.all(jnp.isfinite(self.equality))
            & jnp.all(jnp.isfinite(self.inequality))
            & jnp.all(jnp.isfinite(self.jacobian_equality))
            & jnp.all(jnp.isfinite(self.jacobian_inequality))
        )

    @property
    def finite(self) -> Bool[Array, ""]:
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

    def scaled(self, scales: Scales) -> Evaluation:
        """This evaluation of the problem multiplied by `scales`, as the scaled problem gives it."""
        objective = scales.objective
        return Evaluation(
            objective * self.value,
            objective * self.gradient,
            scales.equality * self.equality,
            scales.inequality * self.inequality,
            scales.equality[:, None] * self.jacobian_equality,
            scales.inequality[:, None] * self.jacobian_inequality,
            objective * self.gauss_newton,
        )


def unit_scales(closures: Closures, dtype: jnp.dtype) -> Scales:
    """Scales of 1: the problem as given."""
    return Scales(
        jnp.ones((), dtype),
        jnp.ones((closures.n_equality,), dtype),
        jnp.ones((closures.n_inequality,), dtype),
    )


def gradient_scales(evaluation: Evaluation) -> Scales:
    """Ipopt's gradient-based scaling: each gradient's largest entry brought to `G_MAX`.

    A factor is `min(1, G_MAX / ||gradient||_inf)`, at least `SCALE_MIN`, and 1 where the
    gradient is not finite.
    """
    tiny = jnp.finfo(evaluation.gradient.dtype).tiny

    def factor(norm):
        scale = jnp.clip(G_MAX / jnp.maximum(norm, tiny), SCALE_MIN, 1.0)
        return jnp.where(jnp.isfinite(norm), scale, 1.0)

    return Scales(
        factor(jnp.max(jnp.abs(evaluation.gradient), initial=0.0)),
        factor(jnp.max(jnp.abs(evaluation.jacobian_equality), axis=1, initial=0.0)),
        factor(jnp.max(jnp.abs(evaluation.jacobian_inequality), axis=1, initial=0.0)),
    )


def problem_scales(
    problem: OptimizationProblem, closures: Closures, evaluation: Evaluation
) -> Scales:
    """The problem's own `objective_scale` and `constraint_scales`, else the gradient-based ones."""
    objective, rows = problem.objective_scale, problem.constraint_scales
    if objective is None and rows is None:
        return gradient_scales(evaluation)
    dtype = evaluation.gradient.dtype
    unit = unit_scales(closures, dtype)
    equality, inequality = (unit.equality, unit.inequality) if rows is None else rows
    return Scales(
        jnp.asarray(1.0 if objective is None else objective, dtype),
        jnp.asarray(equality, dtype),
        jnp.asarray(inequality, dtype),
    )


def scaled_closures(closures: Closures) -> Closures:
    """The closures of the scaled problem, which take `args = (args, scales)`."""

    def evaluate(x, packed):
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

    def constraint_hessian(x, lam_e, lam_i, packed):
        args, scales = packed
        return closures.constraint_hessian(
            x, lam_e * scales.equality, lam_i * scales.inequality, args
        )

    def gauss_newton(x, packed):
        args, scales = packed
        f, g, matrix, failed = closures.gauss_newton(x, args)
        return scales.objective * f, scales.objective * g, scales.objective * matrix, failed

    def is_valid(x, packed):
        return closures.is_valid(x, packed[0])

    return Closures(
        evaluate=evaluate,
        evaluate_with_derivatives=evaluate_with_derivatives,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=None if closures.lagrangian_hessian is None else lagrangian_hessian,
        constraint_hessian=None if closures.constraint_hessian is None else constraint_hessian,
        gauss_newton=None if closures.gauss_newton is None else gauss_newton,
        is_valid=is_valid,
        n_equality=closures.n_equality,
        n_inequality=closures.n_inequality,
        bounded=closures.bounded,
    )


def evaluate(
    closures: Closures, x: Float[Array, "n"], args: PyTree, gauss_newton: bool = False
) -> Evaluation:
    """The objective, the constraints and their first derivatives at `x`, from one evaluation.

    With `gauss_newton`, the objective, gradient and Gauss-Newton matrix come from the
    problem's `gauss_newton` closure, and a reported failure makes the value NaN.
    """
    if gauss_newton:
        value, gradient, matrix, failed = closures.gauss_newton(x, args)
        value = jnp.where(jnp.asarray(failed) > 0, jnp.nan, value)
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(x, args)
    else:
        value, gradient, equality, inequality, jac_eq, jac_ineq = (
            closures.evaluate_with_derivatives(x, args)
        )
        matrix = jnp.zeros((0, 0), dtype=x.dtype)
    return Evaluation(value, gradient, equality, inequality, jac_eq, jac_ineq, matrix)


def trial(
    closures: Closures,
    x: Float[Array, "n"],
    args: PyTree,
    value_required: bool | Bool[Array, ""] = True,
) -> tuple[Float[Array, ""], Float[Array, "n_eq"], Float[Array, "n_ineq"], Bool[Array, ""]]:
    """`(value, equality, inequality, valid)` at a trial point, without derivatives.

    `valid` is false where the problem's trial predicate refuses the point or a row is not
    finite, or the objective is not finite where `value_required`: a failed evaluation the
    line search rejects.
    """
    value, equality, inequality = closures.evaluate(x, args)
    valid = (
        (jnp.isfinite(value) | ~jnp.asarray(value_required))
        & jnp.all(jnp.isfinite(equality))
        & jnp.all(jnp.isfinite(inequality))
        & jnp.asarray(closures.is_valid(x, args))
    )
    return value, equality, inequality, valid
