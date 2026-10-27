"""The two public implicit problem frontends.

Both state the same internal equation

    F(t, y, ydot, theta, control) = 0

with explicit variable roles, and both expose it as
``problem.residual(t, y, ydot, theta, control)``. They differ in what the
author promises:

- [MassProblem][diff_soup.problem.implicit.problems.MassProblem] promises the mass form
  ``F = D(t, y, theta, control) @ ydot - f(t, y, theta, control)`` with ``D``
  a [MassOperator][diff_soup.problem.implicit.operator.MassOperator] — linear
  in ``ydot`` with declared coefficient structure.
- [ImplicitProblem][diff_soup.problem.implicit.problems.ImplicitProblem] promises only a residual plus its declared
  derivative structure — the honest interface for nonlinear dependence on
  ``ydot``.

All dynamic numerical inputs — time, state, derivative candidate, the
parameter pytree ``theta``, and the control pytree — are explicit call
arguments. A problem never closes over a value that should vary with them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import jax
import numpy as np

from diff_soup._precision import Precision
from diff_soup.linear import SparseTopology
from diff_soup.problem.implicit.operator import MassOperator
from diff_soup.problem.implicit.residual import ResidualLinearization, ResidualSparsity
from diff_soup.problem.implicit.structure import DAEStructure

__all__ = ["MassProblem", "ImplicitProblem"]


def _reject_algebraic_columns(pattern: SparseTopology, structure: DAEStructure, what: str) -> None:
    """Reject derivative support in a column declared algebraic."""
    cols = np.asarray(pattern.cols)
    algebraic = ~structure.differential_vars
    bad = np.unique(cols[algebraic[cols]])
    if bad.size:
        raise ValueError(
            f"{what} has support in algebraic-variable column(s) {bad[:5].tolist()}"
            f"{'...' if bad.size > 5 else ''}: a variable declared algebraic "
            f"(differential_vars False) must have a structurally zero ydot column in "
            f"every equation. Mark the variable differential, or remove its column "
            f"from the declared support."
        )


def _default_precision(value):
    """``None`` means the default all-``float64`` policy."""
    return Precision() if value is None else value


def _check_sizes(n_pattern: int, structure: DAEStructure, what: str) -> None:
    """Reject a declared object whose size disagrees with the structure."""
    if n_pattern != structure.n:
        raise ValueError(
            f"{what} covers {n_pattern} unknowns but the DAEStructure declares "
            f"{structure.n} unknowns; the sizes must agree."
        )


class MassProblem(eqx.Module):
    """Mass-matrix problem `M(t, y) @ ydot = f(t, y)`, one of the two problem types the solvers integrate.

    The sizes of `mass`, `rhs_sparsity` and `structure` must agree. An algebraic
    variable's column of the mass must be empty.

    Attributes:
        rhs: Function `(t, y, theta, control)` returning `f`.
        mass: [`MassOperator`][diff_soup.problem.MassOperator] `M`.
        structure: [`DAEStructure`][diff_soup.problem.DAEStructure] marking the differential variables.
        rhs_sparsity: [`SparseTopology`][diff_soup.linear.SparseTopology] holding every possible nonzero of `df/dy`, or `"dense"`.
        linearization: Optional [`ResidualLinearization`][diff_soup.problem.ResidualLinearization] supplying Jacobian values.
        precision: [`Precision`][diff_soup.Precision] of the solve, the default when `None`.
    """

    rhs: Callable = eqx.field(static=True)
    mass: MassOperator = eqx.field(static=True)
    structure: DAEStructure = eqx.field(static=True)
    rhs_sparsity: Union[SparseTopology, str] = eqx.field(static=True, default="dense")
    linearization: Optional[ResidualLinearization] = eqx.field(static=True, default=None)
    precision: Precision = eqx.field(static=True, default=None, converter=_default_precision)

    def __post_init__(self):
        """Validate operator/structure/sparsity agreement at construction."""
        if self.precision is not None and not isinstance(self.precision, Precision):
            raise TypeError(
                f"MassProblem precision must be a Precision; got {type(self.precision).__name__}."
            )
        _check_sizes(self.mass.n, self.structure, "the mass operator")
        _reject_algebraic_columns(
            self.mass.coefficient_sparsity,
            self.structure,
            "the mass operator's coefficient support",
        )
        if isinstance(self.rhs_sparsity, SparseTopology):
            _check_sizes(int(self.rhs_sparsity.n_dofs), self.structure, "rhs_sparsity")
        elif self.rhs_sparsity != "dense":
            raise ValueError(
                "rhs_sparsity must be a SparseTopology containing every possible "
                f"nonzero of df/dy, or the explicit string 'dense'; got "
                f"{self.rhs_sparsity!r}."
            )

    @property
    def n(self) -> int:
        """Number of unknowns."""
        return self.structure.n

    def residual(self, t, y, ydot, theta, control) -> jax.Array:
        """The residual ``F = D @ ydot - f`` at the given point."""
        return self.mass.apply(t, y, theta, control, ydot) - self.rhs(t, y, theta, control)


class ImplicitProblem(eqx.Module):
    """Fully implicit problem `F(t, y, ydot) = 0`, one of the two problem types the solvers integrate.

    Use it when the residual is nonlinear in `ydot`. An algebraic variable's
    column of the declared `dF/dydot` support must be empty.

    Args:
        residual: Function `(t, y, ydot, theta, control)` returning `F`.
        structure: [`DAEStructure`][diff_soup.problem.DAEStructure] marking the differential variables.
        linearization: [`ResidualLinearization`][diff_soup.problem.ResidualLinearization] declaring the Jacobian supports.
        precision: [`Precision`][diff_soup.Precision] of the solve, or `None` for the default.
    """

    residual_fn: Callable = eqx.field(static=True)
    structure: DAEStructure = eqx.field(static=True)
    linearization: ResidualLinearization = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual: Callable,
        structure: DAEStructure,
        linearization: ResidualLinearization,
        precision: Optional[Precision] = None,
    ):
        """Validate the declared derivative structure against the variable roles."""
        if precision is not None and not isinstance(precision, Precision):
            raise TypeError(
                f"ImplicitProblem precision must be a Precision; got {type(precision).__name__}."
            )
        self.precision = Precision() if precision is None else precision
        self.residual_fn = residual
        self.structure = structure
        self.linearization = linearization
        sparsity: ResidualSparsity = linearization.sparsity
        if isinstance(sparsity.dy, SparseTopology):
            _check_sizes(int(sparsity.dy.n_dofs), structure, "the declared F_y support")
        if isinstance(sparsity.dydot, SparseTopology):
            _check_sizes(int(sparsity.dydot.n_dofs), structure, "the declared F_ydot support")
            _reject_algebraic_columns(sparsity.dydot, structure, "the declared F_ydot support")

    @property
    def n(self) -> int:
        """Number of unknowns."""
        return self.structure.n

    def residual(self, t, y, ydot, theta, control) -> jax.Array:
        """The residual ``F`` at the given point."""
        return self.residual_fn(t, y, ydot, theta, control)
