"""The solver base of ``diff_soup.optimization``, and the numeric problems it solves.

- [OptSolver][diff_soup.optimization._abc.OptSolver]: a solver, which names in
  ``requires`` the structure it needs.
- [Basis][diff_soup.optimization._abc.Basis]: the marker base of function-approximation
  bases.

The numeric problems a solver reads, [OptimizationProblem][diff_soup.optimization.OptimizationProblem]
and [LeastSquaresProblem][diff_soup.optimization.LeastSquaresProblem], and the structure
keys live in their own modules and are imported here too. Import this module for
annotations: it imports no solver, problem or fitting machinery.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import equinox as eqx

from diff_soup.optimization._least_squares import LeastSquaresProblem
from diff_soup.optimization._numeric import OptimizationProblem
from diff_soup.optimization._structure import STRUCTURE_KEYS

if TYPE_CHECKING:
    from diff_soup.optimization.inference.fitting.callbacks import Callback
    from diff_soup.optimization.inference.fitting.result import OptResult

#: The entry of a solver's ``requires`` that names a capability of the solver
#: rather than a structure of the problem: its ``solve`` takes ``state=``, the
#: solver state a step record or a saved result carries, and continues from it.
RESUME = "resume"


class OptSolver(eqx.Module):
    """Base class for an optimizer that `ds.optimization.optsolve` runs.

    A subclass implements `solve` and sets `requires`. `ds.optimization.optsolve` refuses a
    problem that cannot provide what the solver requires.

    Attributes:
        requires: The names of what `solve` needs from the problem, such as `"value_gradient"`, plus `"resume"` when `solve` takes `state=`.
    """

    #: The structure keys the solver needs, and ``RESUME`` when ``solve``
    #: takes ``state=``. The empty set takes every problem.
    requires: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def solve(
        self,
        problem: OptimizationProblem,
        *,
        callbacks: tuple[Callback, ...] = (),
    ) -> OptResult:
        """Minimize the problem and return the result.

        Args:
            problem: The [OptimizationProblem][diff_soup.optimization.OptimizationProblem] to minimize.
            callbacks: The [Callback][diff_soup.optimization.inference.Callback] objects to call during the run.

        Returns:
            The [OptResult][diff_soup.optimization.OptResult] of the run.
        """
        raise NotImplementedError


class Basis(eqx.Module):
    """Marker base for function-approximation bases.

    A ``Basis`` is an ``eqx.Module`` pytree that holds differentiable
    coefficient leaves.  Subclass this to mark a class as a function-
    approximation basis compatible with diff_soup's series and collocation
    solvers.

    No abstract methods are required; the class serves as a structural marker
    and a pytree node via the Equinox module system.
    """


__all__ = [
    "RESUME",
    "STRUCTURE_KEYS",
    "Basis",
    "LeastSquaresProblem",
    "OptSolver",
    "OptimizationProblem",
]
