"""Constraints on a problem's decision variables, and their lowering to closures.

Two structure keys join `STRUCTURE_KEYS`:

- `"constraint_jacobian"`: `constraints_and_jacobians(x, args) -> (eq, ineq, J_eq, J_ineq)`,
  dense Jacobians in forward or reverse mode by shape.
- `"lagrangian_hessian"`: `lagrangian_hessian(x, lam_eq, lam_ineq, args) -> H` and
  `lagrangian_hessian_vector(x, lam_eq, lam_ineq, v, args)`, the exact second derivatives of
  `f + lam_eq . c_E + lam_ineq . c_I`.

The existing `"constraints"` key, `constraints(x, args) -> (equality, inequality)`, keeps its
meaning: an equality row is satisfied at 0, an inequality row at or below 0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx

CONSTRAINT_JACOBIAN = "constraint_jacobian"
LAGRANGIAN_HESSIAN = "lagrangian_hessian"


class ConstraintClosures(NamedTuple):
    """The closures a `Constraints` lowers to, each taking `(x, args)` on the flat vector.

    Attributes:
        constraints: `(x, args) -> (equality, inequality)`.
        constraints_and_jacobians: `(x, args) -> (equality, inequality, J_eq, J_ineq)`.
        lagrangian_hessian: `(x, lam_eq, lam_ineq, args) -> H` for the objective given at lowering.
        lagrangian_hessian_vector: `(x, lam_eq, lam_ineq, v, args) -> H v`.
        n_equality: The number of equality rows.
        n_inequality: The number of inequality rows.
    """

    constraints: Callable
    constraints_and_jacobians: Callable
    lagrangian_hessian: Callable
    lagrangian_hessian_vector: Callable
    n_equality: int
    n_inequality: int


class Constraints(eqx.Module):
    """Equality and inequality constraints on the unflattened parameter view.

    Args:
        equality: `equality(params, args)` returning rows satisfied at 0, or `None`.
        inequality: `inequality(params, args)` returning rows satisfied at or below 0, or `None`.
        n_equality: The number of rows `equality` returns.
        n_inequality: The number of rows `inequality` returns.
        jacobian: `"auto"` takes forward mode when rows exceed columns and reverse mode otherwise; `"forward"` and `"reverse"` force one.
        scales: One positive scale per row, `(equality_scales, inequality_scales)`, or `None` for the solver's own scaling.
    """

    equality: Callable | None = eqx.field(static=True, default=None)
    inequality: Callable | None = eqx.field(static=True, default=None)
    n_equality: int = eqx.field(static=True, default=0)
    n_inequality: int = eqx.field(static=True, default=0)
    jacobian: str = eqx.field(static=True, default="auto")
    scales: Any = None

    def lower(self, objective: Callable, unflatten: Callable) -> ConstraintClosures:
        """Lower to closures on the flat vector, with Jacobians and Lagrangian Hessians by AD.

        Args:
            objective: `objective(x, args)` on the flat vector, whose Hessian joins the Lagrangian's.
            unflatten: The map from the flat vector to the parameter view the functions take.

        Returns:
            The [ConstraintClosures][diff_soup.optimization.ConstraintClosures].
        """
        raise NotImplementedError
