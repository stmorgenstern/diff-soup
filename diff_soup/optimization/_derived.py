"""The closures a numeric problem derives from the ones it is given.

Each closure here is a module-level rule. A problem holds `Derived(rule, *inputs)`, which
equals another derived by the same rule from equal inputs
([Derived][diff_soup.optimization._binding.Derived]), so a problem lowered or posed twice has
equal static parts and `filter_jit` reuses its compile. Every closure takes `args` last.
The checks of a starting point and of bounds that the constructors share sit here too.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._binding import Derived


def identity(x: Any) -> Any:
    """`x` itself: the flat vector as the view of a problem without one."""
    return x


@functools.partial(jax.jit, static_argnums=0)
def _compiled(fn: Callable, *operands: Any) -> Any:
    """`fn(*operands)` under `jax.jit`, one program per closure."""
    return fn(*operands)


def jit(fn: Callable | None) -> Callable | None:
    """`fn` compiled when called on its own: one program per closure, shared by equal ones."""
    return None if fn is None else Derived(_compiled, fn)


def valid_everywhere(x: Any, args: Any) -> Any:
    """The trial predicate of a problem without one: every point is valid."""
    return jnp.asarray(True)


def no_constraints(x: Any, args: Any) -> tuple[Any, Any]:
    """The rows of a problem without constraints: two empty vectors."""
    empty = jnp.zeros((0,), dtype=jnp.result_type(x))
    return empty, empty


def no_constraint_jacobians(x: Any, args: Any) -> tuple[Any, Any, Any, Any]:
    """The rows and Jacobians of a problem without constraints: empty, with `n` columns."""
    empty = jnp.zeros((0,), dtype=jnp.result_type(x))
    jacobian = jnp.zeros((0, jnp.shape(x)[-1]), dtype=jnp.result_type(x))
    return empty, empty, jacobian, jacobian


def no_constraint_hessian(x: Any, lam_eq: Any, lam_ineq: Any, args: Any) -> Any:
    """The constraint terms of the Lagrangian Hessian of a problem without constraints: zeros."""
    n = jnp.shape(x)[-1]
    return jnp.zeros((n, n), dtype=jnp.result_type(x))


def first(value_and_gradient: Callable, x: Any, args: Any) -> Any:
    """The value, from a value-and-gradient closure."""
    return value_and_gradient(x, args)[0]


def rows(constraints_and_jacobians: Callable, x: Any, args: Any) -> tuple[Any, Any]:
    """The constraint rows, from a rows-and-Jacobians closure."""
    return constraints_and_jacobians(x, args)[:2]


def hessian_times(value_gradient_hessian: Callable, x: Any, v: Any, args: Any) -> Any:
    """The Hessian-vector product, from a value-gradient-Hessian closure."""
    return value_gradient_hessian(x, args)[2] @ v


def lagrangian_times(
    lagrangian_hessian: Callable, x: Any, lam_eq: Any, lam_ineq: Any, v: Any, args: Any
) -> Any:
    """The Lagrangian Hessian-vector product, from the Lagrangian Hessian."""
    return lagrangian_hessian(x, lam_eq, lam_ineq, args) @ v


def objective_hessian(
    value_gradient_hessian: Callable, x: Any, lam_eq: Any, lam_ineq: Any, args: Any
) -> Any:
    """The Lagrangian Hessian of a problem without constraints: the objective's."""
    return value_gradient_hessian(x, args)[2]


def plus_objective_hessian(
    constraint_part: Callable,
    value_gradient_hessian: Callable,
    x: Any,
    lam_eq: Any,
    lam_ineq: Any,
    args: Any,
) -> Any:
    """The Lagrangian Hessian: the objective's Hessian plus the constraint terms'."""
    return value_gradient_hessian(x, args)[2] + constraint_part(x, lam_eq, lam_ineq, args)


def plus_objective_hessian_vector(
    constraint_part: Callable,
    hessian_vector: Callable,
    x: Any,
    lam_eq: Any,
    lam_ineq: Any,
    v: Any,
    args: Any,
) -> Any:
    """The Lagrangian Hessian-vector product: the objective's plus the constraint terms'."""
    return hessian_vector(x, v, args) + constraint_part(x, lam_eq, lam_ineq, v, args)


def evaluate(value: Callable, constraints: Callable, x: Any, args: Any) -> tuple:
    """`(f, eq, ineq)`, from the objective and the constraint rows."""
    return (value(x, args), *constraints(x, args))


def evaluate_with_derivatives(
    value_and_gradient: Callable, constraints_and_jacobians: Callable, x: Any, args: Any
) -> tuple:
    """`(f, g, eq, ineq, J_eq, J_ineq)`, from the two first-order closures."""
    return (*value_and_gradient(x, args), *constraints_and_jacobians(x, args))


def value_and_gradient(value: Callable, x: Any, args: Any) -> tuple[Any, Any]:
    """The value and its gradient, reverse mode."""
    return jax.value_and_grad(value)(x, args)


def value_gradient_hessian(value: Callable, x: Any, args: Any) -> tuple[Any, Any, Any]:
    """The value, its gradient, and its Hessian forward over reverse."""
    cost, gradient = jax.value_and_grad(value)(x, args)
    return cost, gradient, jax.jacfwd(jax.grad(value))(x, args)


def hessian_vector(value: Callable, x: Any, v: Any, args: Any) -> Any:
    """The Hessian of the value times `v`, forward over reverse."""
    return jax.jvp(lambda y: jax.grad(value)(y, args), (x,), (v,))[1]


def half_sum_of_squares(r: Any) -> Any:
    """`0.5 * sum(r**2)`, the least-squares value of the residual `r`."""
    return 0.5 * jnp.sum(r * r)


def half_squared(residual: Callable, x: Any, args: Any) -> Any:
    """The least-squares value, from a residual closure."""
    return half_sum_of_squares(residual(x, args))


def gradient(residual_and_jacobian: Callable, x: Any, args: Any) -> tuple[Any, Any]:
    """The least-squares value and its gradient `J^T r`, from a residual-and-Jacobian closure."""
    r, jacobian = residual_and_jacobian(x, args)
    return half_sum_of_squares(r), jacobian.T @ r


def gauss_newton(residual_and_jacobian: Callable, x: Any, args: Any) -> tuple:
    """The Gauss-Newton statistics `(value, J^T r, J^T J, 0)`, from a residual-and-Jacobian closure."""
    r, jacobian = residual_and_jacobian(x, args)
    return half_sum_of_squares(r), jacobian.T @ r, jacobian.T @ jacobian, 0


def residual_and_jacobian(
    residual: Callable, jacobian: Callable | None, x: Any, args: Any
) -> tuple[Any, Any]:
    """The residual and its Jacobian, forward mode when no Jacobian is given."""
    if jacobian is None:
        return jax.jacfwd(lambda y: (residual(y, args),) * 2, has_aux=True)(x)[::-1]
    return residual(x, args), jacobian(x, args)


def as_vector(x0: Any) -> Any:
    """`x0` as a 1-D array; a `ConfigurationError` for any other shape."""
    if not isinstance(x0, (jax.Array, np.ndarray)):
        x0 = np.asarray(x0)
    if x0.ndim != 1:
        raise ConfigurationError(
            f"an optimization problem's x0 must be a 1-D array; got shape {tuple(x0.shape)}."
        )
    return x0


def as_bounds(bounds: Any, x0: Any) -> tuple[Any, Any] | None:
    """`bounds` as the `(lower, upper)` pair broadcast to `x0`'s shape and dtype, or `None`."""
    if bounds is None:
        return None
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "bounds must be None or a (lb, ub) pair of arrays aligned with x0."
        ) from exc
    dtype = jnp.result_type(jnp.asarray(x0))
    shape = np.shape(x0)
    lower = jnp.broadcast_to(jnp.asarray(-jnp.inf if lower is None else lower, dtype), shape)
    upper = jnp.broadcast_to(jnp.asarray(jnp.inf if upper is None else upper, dtype), shape)
    return lower, upper
