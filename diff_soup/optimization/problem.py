"""The base of every problem a solver can be asked to minimize."""

from __future__ import annotations

import abc
from typing import Any


class AbstractOptimizationProblem(abc.ABC):
    """A problem a user poses, which lowers to the numeric problem a solver takes.

    `InferenceProblem` and `ControlProblem` are its subclasses. Each declares
    its decision variables in its own terms and lowers them, with its objective, constraints
    and bounds, to an [OptimizationProblem][diff_soup.optimization.OptimizationProblem] of
    closures over one flat vector and `args`. `optsolve` takes any of them.

    It is a plain abstract class, not an `eqx.Module`, because `InferenceProblem` and the
    lowered study are host objects that cannot subclass a Module. `ControlProblem` is a
    Module through its second base; once `InferenceProblem` is one too,
    this base can become a Module. It holds no state of its own.

    Attributes:
        constraints: The [Constraints][diff_soup.optimization.Constraints] on the decision variables, or `None`.
        bounds: The `(lower, upper)` bounds in the problem's own parameterization, or `None`.
    """

    constraints: Any = None
    bounds: Any = None

    @abc.abstractmethod
    def numeric_problem(self, optimizer: Any) -> Any:
        """Lower to the numeric problem `optimizer` solves.

        Args:
            optimizer: The solver, whose `requires` says which closures to build.

        Returns:
            An [OptimizationProblem][diff_soup.optimization.OptimizationProblem] whose closures
            take `(x, args)` and whose `args` leaf holds what varies between lanes.
        """
        raise NotImplementedError

    def args(self) -> Any:
        """The dynamic leaves the numeric problem carries: limits, initial states, outer parameters.

        A batch of problems is a batch of `args`; an outer loop differentiates through them.
        """
        return None
