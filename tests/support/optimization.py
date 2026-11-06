"""Small constrained problems with closed-form solutions, as the numeric problems a compiled solver reads.

A compiled solver reads a problem's raw ``(x, args)`` closures: ``_value``,
``_value_and_gradient``, ``_constraints``, ``_constraints_and_jacobians`` and
``_lagrangian_hessian``, the static row counts ``n_equality`` and ``n_inequality``, and
``bounds``. Each toy's ``args`` is what varies between lanes (a constraint limit or an
objective weight), and ``solution(args)`` gives ``(x, lam_eq, lam_ineq, z_lower, z_upper)``
exactly, differentiably in ``args``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class NumericProblem(eqx.Module):
    """The closures and sizes a compiled solver reads, built from plain functions."""

    bounds: Any
    x0: Any
    args: Any
    _value: Callable = eqx.field(static=True)
    _value_and_gradient: Callable = eqx.field(static=True)
    _constraints: Callable = eqx.field(static=True)
    _constraints_and_jacobians: Callable = eqx.field(static=True)
    _lagrangian_hessian: Callable = eqx.field(static=True)
    n_equality: int = eqx.field(static=True)
    n_inequality: int = eqx.field(static=True)


def numeric_problem(objective, equality, inequality, n_eq, n_ineq, x0, args, bounds=None):
    """A ``NumericProblem`` with derivatives by ``jax.jacfwd`` and ``jax.hessian``."""

    def constraints(x, a):
        eq = equality(x, a) if n_eq else jnp.zeros((0,), x.dtype)
        ineq = inequality(x, a) if n_ineq else jnp.zeros((0,), x.dtype)
        return eq, ineq

    def constraints_and_jacobians(x, a):
        eq, ineq = constraints(x, a)
        jac_eq = jax.jacfwd(lambda y: constraints(y, a)[0])(x)
        jac_ineq = jax.jacfwd(lambda y: constraints(y, a)[1])(x)
        return eq, ineq, jac_eq, jac_ineq

    def lagrangian_hessian(x, lam_eq, lam_ineq, a):
        def lagrangian(y):
            eq, ineq = constraints(y, a)
            return objective(y, a) + lam_eq @ eq + lam_ineq @ ineq

        return jax.hessian(lagrangian)(x)

    return NumericProblem(
        bounds=None if bounds is None else tuple(jnp.asarray(b, jnp.float64) for b in bounds),
        x0=jnp.asarray(x0, jnp.float64),
        args=jnp.asarray(args, jnp.float64),
        _value=objective,
        _value_and_gradient=jax.value_and_grad(objective),
        _constraints=constraints,
        _constraints_and_jacobians=constraints_and_jacobians,
        _lagrangian_hessian=lagrangian_hessian,
        n_equality=n_eq,
        n_inequality=n_ineq,
    )


class Toy(NamedTuple):
    """A problem, a batch of its ``args`` (one per lane) and its exact solution."""

    problem: NumericProblem
    args_batch: Any
    solution: Callable


def hs7() -> Toy:
    """Hock and Schittkowski 7, the equality's right-hand side ``a`` in ``args`` (4 in HS)."""

    def solution(a):
        r = jnp.sqrt(a - 1.0)
        return (
            jnp.stack([0.0 * r, r]),
            jnp.stack([0.5 / r]),
            jnp.zeros(0),
            jnp.zeros(2),
            jnp.zeros(2),
        )

    problem = numeric_problem(
        lambda x, a: jnp.log1p(x[0] ** 2) - x[1],
        lambda x, a: jnp.stack([(1 + x[0] ** 2) ** 2 + x[1] ** 2 - a]),
        None,
        1,
        0,
        [2.0, 2.0],
        4.0,
    )
    return Toy(problem, jnp.array([4.0, 2.0, 10.0, 50.0]), solution)


_H35 = jnp.array([[4.0, 2.0, 2.0], [2.0, 4.0, 0.0], [2.0, 0.0, 2.0]])
_G35 = jnp.array([-8.0, -6.0, -4.0])
_A35 = jnp.array([1.0, 1.0, 2.0])


def hs35() -> Toy:
    """Hock and Schittkowski 35, the inequality's limit ``a`` in ``args`` (3 in HS), x >= 0."""

    def solution(a):
        kkt = jnp.zeros((4, 4)).at[:3, :3].set(_H35).at[:3, 3].set(_A35).at[3, :3].set(_A35)
        sol = jnp.linalg.solve(kkt, jnp.concatenate([-_G35, jnp.stack([a])]))
        return sol[:3], jnp.zeros(0), sol[3:], jnp.zeros(3), jnp.zeros(3)

    problem = numeric_problem(
        lambda x, a: 9.0 + _G35 @ x + 0.5 * x @ _H35 @ x,
        None,
        lambda x, a: jnp.stack([_A35 @ x - a]),
        0,
        1,
        [0.5, 0.5, 0.5],
        3.0,
        bounds=(jnp.zeros(3), jnp.full(3, jnp.inf)),
    )
    return Toy(problem, jnp.array([3.0, 2.5, 3.4, 2.9]), solution)


def hs21() -> Toy:
    """Hock and Schittkowski 21, the objective weight ``a`` in ``args`` (0.01 in HS); a bound active."""

    def solution(a):
        return (
            jnp.array([2.0, 0.0]),
            jnp.zeros(0),
            jnp.zeros(1),
            jnp.stack([4.0 * a, 0.0 * a]),
            jnp.zeros(2),
        )

    problem = numeric_problem(
        lambda x, a: a * x[0] ** 2 + x[1] ** 2 - 100.0,
        None,
        lambda x, a: jnp.stack([10.0 - 10.0 * x[0] + x[1]]),
        0,
        1,
        [-1.0, -1.0],
        0.01,
        bounds=(jnp.array([2.0, -50.0]), jnp.array([50.0, 50.0])),
    )
    return Toy(problem, jnp.array([0.01, 1.0, 0.1, 10.0]), solution)


def duplicate_rows() -> NumericProblem:
    """``min |x - 2|^2`` with ``x1 + x2 <= a`` twice: both rows active, ``x* = (a, a) / 2``."""
    return numeric_problem(
        lambda x, a: jnp.sum((x - 2.0) ** 2),
        None,
        lambda x, a: jnp.stack([x[0] + x[1] - a, x[0] + x[1] - a]),
        0,
        2,
        [0.0, 0.0],
        1.0,
    )


def infeasible() -> NumericProblem:
    """``min x1 + x2`` with ``x1^2 + x2^2 + a <= 0``, which no point satisfies for ``a > 0``."""
    return numeric_problem(
        lambda x, a: x[0] + x[1], None, lambda x, a: jnp.stack([x @ x + a]), 0, 1, [1.0, 1.0], 1.0
    )


def nan_region() -> NumericProblem:
    """``(x - 2)^2`` below ``x = a`` and NaN from there on, so the iterates crowd the edge."""
    return numeric_problem(
        lambda x, a: jnp.where(x[0] < a, (x[0] - 2.0) ** 2, jnp.nan), None, None, 0, 0, [0.0], 1.0
    )
