"""Fixed variables: a coordinate whose lower and upper bounds are equal leaves the solve.

As Ipopt's `fixed_variable_treatment="make_parameter"` (its default) does, the coordinate is
a parameter of the solve rather than a variable: it starts at its value, every closure reads
that value there, and its gradient entry and its columns of the constraint Jacobians are 0.
The shapes stay those of the problem, so one compiled program serves bounds that fix
different coordinates; the curvature is the identity on a fixed coordinate, so the step
system stays regular and the coordinate's step is exactly 0. Its bound is removed, so no
barrier or projection reads a width of 0. After the solve the coordinate holds its value and
its bound multipliers are the gradient of the Lagrangian there, split by sign, as Ipopt
reports them for `make_parameter`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, PyTree

from diff_soup.optimization.solvers._barrier import Bounds
from diff_soup.optimization.solvers._closures import Closures
from diff_soup.optimization.solvers._protocol import SolveRecord


def pinned(bounds: Bounds) -> Bool[Array, " n"]:
    """Where a finite lower bound equals its upper bound."""
    return bounds.has_lower & bounds.has_upper & (bounds.lower == bounds.upper)


def fix(closures: Closures, bounds: Bounds) -> tuple[Closures, Bounds]:
    """The closures with the coordinates `bounds` fix as parameters, and the bounds without them.

    A problem without bounds is returned as it is.
    """
    if not closures.bounded:
        return closures, bounds
    fixed = pinned(bounds)
    value = bounds.lower
    free = (~fixed).astype(value.dtype)
    identity = jnp.diag(fixed.astype(value.dtype))

    def at(x):
        return jnp.where(fixed, value, x)

    def columns(jacobian):
        return jacobian * free[None, :]

    def masked(matrix):
        return free[:, None] * matrix * free[None, :]

    def curvature(matrix):
        return masked(matrix) + identity

    def evaluate(x, args):
        return closures.evaluate(at(x), args)

    def evaluate_with_derivatives(x, args):
        f, g, equality, inequality, jac_eq, jac_ineq = closures.evaluate_with_derivatives(
            at(x), args
        )
        return f, g * free, equality, inequality, columns(jac_eq), columns(jac_ineq)

    def constraints_and_jacobians(x, args):
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(at(x), args)
        return equality, inequality, columns(jac_eq), columns(jac_ineq)

    def lagrangian_hessian(x, lam_e, lam_i, args):
        return curvature(closures.lagrangian_hessian(at(x), lam_e, lam_i, args))

    def constraint_hessian(x, lam_e, lam_i, args):
        return masked(closures.constraint_hessian(at(x), lam_e, lam_i, args))

    def gauss_newton(x, args):
        f, g, matrix, failed = closures.gauss_newton(at(x), args)
        return f, g * free, curvature(matrix), failed

    def is_valid(x, args):
        return closures.is_valid(at(x), args)

    def residual(x, args):
        return closures.residual(at(x), args)

    def residual_and_jacobian(x, args):
        return closures.residual_and_jacobian(at(x), args)

    def optional(fn, own):
        return None if fn is None else own

    wrapped = closures._replace(
        evaluate=evaluate,
        evaluate_with_derivatives=evaluate_with_derivatives,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=optional(closures.lagrangian_hessian, lagrangian_hessian),
        constraint_hessian=optional(closures.constraint_hessian, constraint_hessian),
        gauss_newton=optional(closures.gauss_newton, gauss_newton),
        is_valid=is_valid,
        residual=optional(closures.residual, residual),
        residual_and_jacobian=optional(closures.residual_and_jacobian, residual_and_jacobian),
        fixed=fixed,
    )
    unbounded = bounds._replace(
        has_lower=bounds.has_lower & ~fixed, has_upper=bounds.has_upper & ~fixed
    )
    return wrapped, unbounded


def start(closures: Closures, bounds: Bounds, x: Float[Array, " n"]) -> Float[Array, " n"]:
    """`x` with each fixed coordinate at its value."""
    if closures.fixed is None:
        return x
    return jnp.where(closures.fixed, bounds.lower, x)


def restore(
    raw: Closures,
    fixed: Bool[Array, " n"] | None,
    bounds: Bounds,
    x: Float[Array, " n"],
    record: SolveRecord,
    args: PyTree,
) -> tuple[Float[Array, " n"], SolveRecord]:
    """The solution with each fixed coordinate at its value and its bound multipliers reported.

    Args:
        raw: The problem's own closures, before [fix][].
        fixed: Where a coordinate is fixed, or `None` for a problem without bounds.
        bounds: The bounds [fix][] returned, which keep the fixed values.
        x: The solution.
        record: Its record, whose multipliers are of the problem as given.
        args: The problem's `args`.

    Returns:
        `(x, record)`: `z_lower = max(g, 0)` and `z_upper = max(-g, 0)` at a fixed coordinate,
        with `g` the gradient of the Lagrangian there.
    """
    if fixed is None:
        return x, record
    x = jnp.where(fixed, bounds.lower, x)

    def gradient():
        _, g, _, _, jac_eq, jac_ineq = raw.evaluate_with_derivatives(x, args)
        return (
            g + jac_eq.T @ record.multipliers_equality + jac_ineq.T @ record.multipliers_inequality
        )

    g = lax.cond(jnp.any(fixed), gradient, lambda: jnp.zeros_like(x))
    lower, upper = record.bound_multipliers
    multipliers = (
        jnp.where(fixed, jnp.maximum(g, 0.0), lower),
        jnp.where(fixed, jnp.maximum(-g, 0.0), upper),
    )
    return x, eqx.tree_at(lambda r: r.bound_multipliers, record, multipliers)
