"""Small constrained problems with closed-form solutions, posed through the public problem API.

Each toy is an objective of `x` alone, posed with `OptimizationProblem.from_value` and its exact
Hessian, under `with_constraints`. The constraints read `args`, a limit or a right-hand side,
which is what varies between lanes, and `solution(args)` gives
`(x, lam_eq, lam_ineq, z_lower, z_upper)` exactly, differentiably in `args`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp

from diff_soup.optimization import Constraints, LeastSquaresProblem, OptimizationProblem


class Toy(NamedTuple):
    """A problem, a batch of its `args` (one per lane) and its exact solution."""

    problem: OptimizationProblem
    args_batch: Any
    solution: Callable


def objective(x0, value, bounds=None, dtype=jnp.float64) -> OptimizationProblem:
    """`value(x)` from `x0`, with its gradient and exact Hessian by autodiff."""
    return OptimizationProblem.from_value(
        jnp.asarray(x0, dtype), value, bounds=bounds, hessian=True
    )


def constrained(
    problem, args, *, equality=None, n_eq=0, inequality=None, n_ineq=0, dtype=jnp.float64
):
    """`problem` under `equality(x, args) = 0` and `inequality(x, args) <= 0`, holding `args`."""
    constraints = Constraints(
        equality=equality, inequality=inequality, n_equality=n_eq, n_inequality=n_ineq
    )
    return problem.with_constraints(constraints).with_args(jnp.asarray(args, dtype))


def hs7() -> Toy:
    """Hock and Schittkowski 7, the equality's right-hand side `a` in `args` (4 in HS)."""

    def solution(a):
        r = jnp.sqrt(a - 1.0)
        return (
            jnp.stack([0.0 * r, r]),
            jnp.stack([0.5 / r]),
            jnp.zeros(0),
            jnp.zeros(2),
            jnp.zeros(2),
        )

    problem = constrained(
        objective([2.0, 2.0], lambda x: jnp.log1p(x[0] ** 2) - x[1]),
        4.0,
        equality=lambda x, a: jnp.stack([(1 + x[0] ** 2) ** 2 + x[1] ** 2 - a]),
        n_eq=1,
    )
    return Toy(problem, jnp.array([4.0, 2.0, 10.0, 50.0]), solution)


_H35 = jnp.array([[4.0, 2.0, 2.0], [2.0, 4.0, 0.0], [2.0, 0.0, 2.0]])
_G35 = jnp.array([-8.0, -6.0, -4.0])
_A35 = jnp.array([1.0, 1.0, 2.0])


def hs35(dtype=jnp.float64) -> Toy:
    """Hock and Schittkowski 35, the inequality's limit `a` in `args` (3 in HS), x >= 0, posed
    in `dtype`."""
    h, g, row = (matrix.astype(dtype) for matrix in (_H35, _G35, _A35))

    def solution(a):
        kkt = jnp.zeros((4, 4)).at[:3, :3].set(_H35).at[:3, 3].set(_A35).at[3, :3].set(_A35)
        sol = jnp.linalg.solve(kkt, jnp.concatenate([-_G35, jnp.stack([a])]))
        return sol[:3], jnp.zeros(0), sol[3:], jnp.zeros(3), jnp.zeros(3)

    problem = constrained(
        objective(
            [0.5, 0.5, 0.5],
            lambda x: 9.0 + g @ x + 0.5 * x @ h @ x,
            (jnp.zeros(3, dtype), None),
            dtype,
        ),
        3.0,
        inequality=lambda x, a: jnp.stack([row @ x - a]),
        n_ineq=1,
        dtype=dtype,
    )
    return Toy(problem, jnp.array([3.0, 2.5, 3.4, 2.9]), solution)


def bound_active() -> Toy:
    """`min (x0 + 1)^2 + x1^2 + x2^2` with `x0 + x1 + x2 = a` and `x0 >= 0`, the bound active for `a < 2`.

    `x* = (0, a/2, a/2)`, `lam_eq = -a` and `z_lower = (2 - a, 0, 0)`.
    """

    def solution(a):
        return (
            jnp.stack([0.0 * a, 0.5 * a, 0.5 * a]),
            jnp.stack([-a]),
            jnp.zeros(0),
            jnp.stack([2.0 - a, 0.0 * a, 0.0 * a]),
            jnp.zeros(3),
        )

    lower = jnp.array([0.0, -jnp.inf, -jnp.inf])
    problem = constrained(
        objective(
            [1.0, 1.0, 1.0], lambda x: (x[0] + 1.0) ** 2 + x[1] ** 2 + x[2] ** 2, (lower, None)
        ),
        1.0,
        equality=lambda x, a: jnp.stack([jnp.sum(x) - a]),
        n_eq=1,
    )
    return Toy(problem, jnp.array([1.0, 0.5, 1.5, 0.2]), solution)


def least_squares() -> Toy:
    """`min |x - 1|^2 / 2` with `x0 + x1 <= a`, a least-squares objective: `x* = (a, a) / 2` for `a < 2`."""

    def solution(a):
        return (
            jnp.stack([0.5 * a, 0.5 * a]),
            jnp.zeros(0),
            jnp.stack([1.0 - 0.5 * a]),
            jnp.zeros(2),
            jnp.zeros(2),
        )

    objective = LeastSquaresProblem.from_residual(jnp.zeros(2, jnp.float64), lambda x: x - 1.0)
    problem = constrained(
        objective, 1.0, inequality=lambda x, a: jnp.stack([x[0] + x[1] - a]), n_ineq=1
    )
    return Toy(problem, jnp.array([1.0, 0.5, 1.5, 0.2]), solution)
