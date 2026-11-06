"""A regularization's penalty folded into the closures of a numeric problem.

A regularization (`Tikhonov`, a MAP prior's rows) gives extra least-squares rows
`residual(x)` and the penalty `penalty(x) = 0.5 * ||residual(x)||^2`. A host-loop solver
adds it to the problem's objective itself; a compiled solver reads the closures
`OptimizationProblem.closure` returns, which carry it: the value, the gradient, the
Gauss-Newton statistics and every Hessian. The residual rows stay the data's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from diff_soup.optimization._binding import Derived, Held


def _penalty(held: Held, x: Any) -> Any:
    """The penalty and its gradient at `x`."""
    return jax.value_and_grad(held.value.penalty)(x)


def _penalty_hessian(held: Held, x: Any) -> Any:
    """The Hessian of the penalty at `x`."""
    return jax.hessian(held.value.penalty)(x)


def _penalty_hessian_vector(held: Held, x: Any, v: Any) -> Any:
    """The Hessian of the penalty at `x` times `v`."""
    return jax.jvp(jax.grad(held.value.penalty), (x,), (v,))[1]


def _rows(held: Held, x: Any) -> tuple[Any, Any]:
    """The regularization's rows and their Jacobian at `x`."""
    residual = held.value.residual
    return residual(x), jax.jacfwd(residual)(x)


def _value(value: Callable, held: Held, x: Any, args: Any) -> Any:
    return value(x, args) + held.value.penalty(x)


def _value_and_gradient(value_and_gradient: Callable, held: Held, x: Any, args: Any) -> Any:
    value, gradient = value_and_gradient(x, args)
    penalty, slope = _penalty(held, x)
    return value + penalty, gradient + slope


def _value_gradient_hessian(value_gradient_hessian: Callable, held: Held, x: Any, args: Any):
    value, gradient, hessian = value_gradient_hessian(x, args)
    penalty, slope = _penalty(held, x)
    return value + penalty, gradient + slope, hessian + _penalty_hessian(held, x)


def _hessian_vector(hessian_vector: Callable, held: Held, x: Any, v: Any, args: Any) -> Any:
    return hessian_vector(x, v, args) + _penalty_hessian_vector(held, x, v)


def _gauss_newton(gauss_newton: Callable, held: Held, x: Any, args: Any) -> Any:
    value, gradient, matrix, failed = gauss_newton(x, args)
    rows, jacobian = _rows(held, x)
    return (
        value + 0.5 * jnp.sum(rows * rows),
        gradient + jacobian.T @ rows,
        matrix + jacobian.T @ jacobian,
        failed,
    )


def _evaluate(evaluate: Callable, held: Held, x: Any, args: Any) -> Any:
    value, equality, inequality = evaluate(x, args)
    return value + held.value.penalty(x), equality, inequality


def _evaluate_with_derivatives(evaluate: Callable, held: Held, x: Any, args: Any) -> Any:
    value, gradient, *rows = evaluate(x, args)
    penalty, slope = _penalty(held, x)
    return (value + penalty, gradient + slope, *rows)


def _lagrangian_hessian(lagrangian_hessian: Callable, held: Held, x, lam_eq, lam_ineq, args):
    return lagrangian_hessian(x, lam_eq, lam_ineq, args) + _penalty_hessian(held, x)


def _lagrangian_hessian_vector(product: Callable, held: Held, x, lam_eq, lam_ineq, v, args):
    return product(x, lam_eq, lam_ineq, v, args) + _penalty_hessian_vector(held, x, v)


#: The closures of the objective, each with the rule that folds the penalty into it.
_RULES = {
    "value": _value,
    "value_and_gradient": _value_and_gradient,
    "value_gradient_hessian": _value_gradient_hessian,
    "hessian_vector": _hessian_vector,
    "gauss_newton": _gauss_newton,
    "evaluate": _evaluate,
    "evaluate_with_derivatives": _evaluate_with_derivatives,
    "lagrangian_hessian": _lagrangian_hessian,
    "lagrangian_hessian_vector": _lagrangian_hessian_vector,
}


def penalized(name: str, closure: Callable, regularization: Held) -> Callable:
    """The closure `name` with the regularization's penalty folded in; unchanged when it has none.

    Args:
        name: The closure's name, as `OptimizationProblem.closure` takes it.
        closure: The closure of the objective without the penalty, taking `args` last.
        regularization: The held regularization, whose value is `None` for none.

    Returns:
        The closure of the regularized objective, equal to another folded from equal inputs.
    """
    rule = _RULES.get(name)
    if rule is None or regularization.value is None:
        return closure
    return Derived(rule, closure, regularization)


__all__ = ["penalized"]
