"""Narrow structural protocols consumed by the optimization composer."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from diff_soup.inference.fitting.result import OptResult


@runtime_checkable
class StatisticalInferenceProblem(Protocol):
    """A statistical problem that lowers to an optimization problem and finalizes its fit.

    ``optsolve`` solves ``numeric_problem(optimizer)`` and hands the result to
    ``finalize_opt_result``; the end-of-fit diagnostics read
    ``diagnostic_problem``.
    """

    @property
    def diagnostic_problem(self) -> Any:
        """The data-only problem the end-of-fit diagnostics evaluate."""

    def numeric_problem(self, optimizer: Any) -> Any:
        """The [OptimizationProblem][diff_soup.inference.OptimizationProblem] ``optimizer`` solves.

        A [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem] whose
        ``structure`` holds what this problem provides for ``optimizer``.
        """

    def joint_view(self) -> Any:
        """The joint problem with its level structure, at this problem's start."""

    def finalize_opt_result(self, result: OptResult, *, optimizer: Any) -> OptResult:
        """Attach domain provenance to a completed optimization result."""


__all__ = ["StatisticalInferenceProblem"]
