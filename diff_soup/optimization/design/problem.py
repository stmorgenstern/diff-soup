"""The design problem: the design variables and a criterion over `inference.design`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import PyTree

from diff_soup.optimization.problem import AbstractOptimizationProblem

if TYPE_CHECKING:
    from diff_soup.optimization._abc import OptimizationProblem, OptSolver
    from diff_soup.optimization.constraints import Constraints


class DesignProblem(AbstractOptimizationProblem, eqx.Module):
    """Choose a design that optimizes a criterion of the Fisher information, under constraints.

    The design code is `diff_soup.optimization.inference.design`; `numeric_problem` raises
    `NotImplementedError`.

    Args:
        design: The design variables in their own terms, such as a protocol's targets and durations.
        criterion: `criterion(design, args) -> scalar` built from `inference.design`'s criteria, to minimize.
    """

    constraints: Constraints | None = None
    bounds: PyTree = None
    design: PyTree = None
    criterion: Callable | None = eqx.field(static=True, default=None)

    def numeric_problem(self, optimizer: OptSolver) -> OptimizationProblem:
        """Raise `NotImplementedError`: a design problem has no lowering."""
        raise NotImplementedError("DesignProblem has no lowering to a numeric problem.")
