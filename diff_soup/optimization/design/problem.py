"""The design problem, a stub: the design variables and a criterion over `inference.design`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx

from diff_soup.optimization.problem import AbstractOptimizationProblem


class DesignProblem(AbstractOptimizationProblem):
    """Choose a design that optimizes a criterion of the Fisher information, under constraints.

    A stub: the existing design code stays in `diff_soup.optimization.inference.design`, and
    `numeric_problem` is not implemented.

    Args:
        design: The design variables in their own terms, such as a protocol's targets and durations.
        criterion: `criterion(design, args) -> scalar` built from `inference.design`'s criteria, to minimize.
    """

    design: Any = None
    criterion: Callable | None = eqx.field(static=True, default=None)

    def numeric_problem(self, optimizer: Any) -> Any:
        """Not implemented: the design problem is a stub."""
        raise NotImplementedError("DesignProblem is a stub; its lowering is not implemented.")
