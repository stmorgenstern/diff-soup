"""Sequential quadratic programming on the interior point's KKT step and filter line search."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx

from diff_soup.optimization.kkt import DenseKKT
from diff_soup.optimization.solvers._protocol import CompiledSolver, SolveRecord, SolverState


class SQP(CompiledSolver):
    """SQP: each step solves the QP of the quasi-Newton model and the linearized constraints, then a filter line search.

    It warm-starts from `(x, lam)` natively, which makes it the receding-horizon workhorse.

    Args:
        tol: The scaled optimality error at which the run ends.
        max_steps: The step budget.
        hessian: `"auto"`, `"bfgs"`, `"exact"` or `"gauss-newton"`.
        qp_tol: The tolerance of the interior point solve of each QP.
        qp_max_steps: The step budget of each QP solve.
        max_failed_evaluations: Consecutive failed trial evaluations before the run ends with `SOLVER_FAILED`.
        kkt: The [KKTSolve][diff_soup.optimization.KKTSolve] backend.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=100)
    hessian: str = eqx.field(static=True, default="auto")
    qp_tol: float = eqx.field(static=True, default=1e-10)
    qp_max_steps: int = eqx.field(static=True, default=50)
    max_failed_evaluations: int = eqx.field(static=True, default=10)
    kkt: Any = eqx.field(static=True, default_factory=DenseKKT)

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """The state at `x0`, with the multipliers of `warm_start` when given."""
        raise NotImplementedError

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One SQP step: the QP, the line search, the quasi-Newton update."""
        raise NotImplementedError

    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """End on the scaled optimality error, the budget, or a failure."""
        raise NotImplementedError

    def run(
        self,
        problem: Any,
        x0: Any,
        args: Any,
        *,
        max_steps: int | None = None,
        warm_start: Any = None,
    ) -> tuple[Any, SolveRecord]:
        """The compiled loop; see [CompiledSolver.run][diff_soup.optimization.CompiledSolver.run]."""
        raise NotImplementedError
