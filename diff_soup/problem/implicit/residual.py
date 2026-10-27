"""Declared derivative structure of a fully implicit residual.

[ResidualSparsity][diff_soup.problem.implicit.residual.ResidualSparsity]
states the support of ``F_y`` and ``F_ydot``;
[ResidualLinearization][diff_soup.problem.implicit.residual.
ResidualLinearization] bundles it with optional exact derivative
accelerators. Sparse patterns must contain **every** possible nonzero — a
missing edge makes colored assembly silently wrong — and a user who cannot
bound the support declares ``"dense"`` explicitly.

Callback conventions (all optional; omitting one selects the automatic exact
builder, never a weaker derivative):

- ``dy_values`` returns ``F_y`` values over the declared ``dy`` support.
- ``dydot_values`` returns ``F_ydot`` values over the declared ``dydot``
  support.
- ``combined_values(..., cj)`` returns ``F_y + cj * F_ydot`` **already
  combined** over the union support; the consumer never applies another
  ``cj`` factor.
- ``jvp(..., cj, v)`` returns ``(F_y + cj * F_ydot) @ v``.
- ``transpose_jvp(..., cj, w)`` returns ``(F_y + cj * F_ydot).T @ w``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx

from diff_soup.linear import SparseTopology

__all__ = ["ResidualSparsity", "ResidualLinearization"]

PatternLike = Union[SparseTopology, str]


def _check_pattern(value, what: str) -> PatternLike:
    """Coerce a declared support to a pattern or the explicit 'dense'."""
    if isinstance(value, SparseTopology):
        return value
    if value == "dense":
        return "dense"
    raise ValueError(
        f"{what} must be a SparseTopology containing every possible nonzero, or the "
        f"explicit string 'dense'; got {value!r}."
    )


class ResidualSparsity(eqx.Module):
    """Supports of the two residual Jacobians, `dF/dy` and `dF/dydot`.

    Args:
        dy: [`SparseTopology`][diff_soup.linear.SparseTopology] holding every possible nonzero of `dF/dy`, or `"dense"`.
        dydot: [`SparseTopology`][diff_soup.linear.SparseTopology] holding every possible nonzero of `dF/dydot`, or `"dense"`.
    """

    dy: PatternLike = eqx.field(static=True)
    dydot: PatternLike = eqx.field(static=True)

    def __init__(self, *, dy, dydot):
        """Validate that each declared support is a pattern or 'dense'."""
        self.dy = _check_pattern(dy, "ResidualSparsity dy")
        self.dydot = _check_pattern(dydot, "ResidualSparsity dydot")


class ResidualLinearization(eqx.Module):
    """Jacobian supports of a residual, with optional callbacks for their values.

    A callback left `None` is replaced by automatic differentiation over the
    declared support.

    Attributes:
        sparsity: [`ResidualSparsity`][diff_soup.problem.ResidualSparsity] of `dF/dy` and `dF/dydot`.
        dy_values: Function `(t, y, ydot, theta, control)` returning `dF/dy` on its support, or `None`.
        dydot_values: Function `(t, y, ydot, theta, control)` returning `dF/dydot` on its support, or `None`.
        combined_values: Function `(t, y, ydot, theta, control, cj)` returning `dF/dy + cj * dF/dydot` on the union support, or `None`.
        max_fy_reuse_steps: Most steps a solver may reuse one `dF/dy`, nonnegative, or `None` for the solver's default.
    """

    sparsity: ResidualSparsity = eqx.field(static=True)
    dy_values: Optional[Callable] = eqx.field(static=True, default=None)
    dydot_values: Optional[Callable] = eqx.field(static=True, default=None)
    combined_values: Optional[Callable] = eqx.field(static=True, default=None)
    max_fy_reuse_steps: Optional[int] = eqx.field(static=True, default=None)
