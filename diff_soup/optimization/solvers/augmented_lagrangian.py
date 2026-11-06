"""An augmented Lagrangian with a compiled projected Newton trust region inside."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx

from diff_soup.optimization.solvers._protocol import CompiledSolver, SolveRecord, SolverState


class AugmentedLagrangian(CompiledSolver):
    """Powell-Hestenes-Rockafellar multipliers with the LANCELOT update rules.

    Each outer step minimizes the augmented Lagrangian under the bounds with a compiled
    projected Newton trust region, then updates the multipliers or raises the penalty. It is
    the fallback for degenerate constraints and replaces `constrained_optsolve`.

    Args:
        tol: The scaled optimality error at which the run ends.
        feasibility_tol: The constraint violation at which the run ends.
        max_steps: The outer step budget.
        rho_init: The first penalty.
        rho_factor: The factor the penalty grows by when the violation did not fall enough.
        inner_max_steps: The step budget of each inner minimization.
        hessian: `"auto"`, `"bfgs"`, `"exact"` or `"gauss-newton"` for the inner Newton.
        max_failed_evaluations: Consecutive failed trial evaluations before the run ends with `SOLVER_FAILED`.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    feasibility_tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=50)
    rho_init: float = eqx.field(static=True, default=10.0)
    rho_factor: float = eqx.field(static=True, default=10.0)
    inner_max_steps: int = eqx.field(static=True, default=200)
    hessian: str = eqx.field(static=True, default="auto")
    max_failed_evaluations: int = eqx.field(static=True, default=10)

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """The state at `x0` with zero multipliers, or those of `warm_start`."""
        raise NotImplementedError

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One outer step: the inner minimization, then the multiplier or penalty update."""
        raise NotImplementedError

    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """End on the optimality error and the violation, the budget, or a failure."""
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
