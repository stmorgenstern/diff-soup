"""A primal-dual interior point method with a filter line search, after Wächter and Biegler 2006."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx

from diff_soup.optimization.kkt import DenseKKT
from diff_soup.optimization.solvers._protocol import CompiledSolver, SolveRecord, SolverState


class InteriorPoint(CompiledSolver):
    """Interior point with slacks on the inequalities, the barrier on the slacks and the bounds, and Ipopt's filter line search.

    Args:
        tol: The scaled optimality error at which the run ends.
        max_steps: The step budget.
        mu_init: The first barrier parameter; a warm start takes a smaller one from the given point's complementarity.
        mu_min: The barrier parameter is never reduced below this.
        kappa_mu: The linear factor of the monotone barrier update.
        theta_mu: The superlinear exponent of the monotone barrier update.
        tau_min: The least fraction to the boundary.
        bound_push: How far inside its bounds the start point is pushed, absolute.
        bound_frac: How far inside its bounds the start point is pushed, relative to the bound width.
        hessian: `"auto"`, `"bfgs"`, `"exact"` or `"gauss-newton"`.
        scaling: `"gradient"` for Ipopt's gradient-based scaling at the start point, `"none"` for the problem's own.
        restoration: Whether the feasibility restoration phase runs when the line search fails.
        max_failed_evaluations: Consecutive failed trial evaluations before the run ends with `SOLVER_FAILED`.
        kkt: The [KKTSolve][diff_soup.optimization.KKTSolve] backend.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    tol: float = eqx.field(static=True, default=1e-8)
    max_steps: int = eqx.field(static=True, default=300)
    mu_init: float = eqx.field(static=True, default=0.1)
    mu_min: float = eqx.field(static=True, default=1e-11)
    kappa_mu: float = eqx.field(static=True, default=0.2)
    theta_mu: float = eqx.field(static=True, default=1.5)
    tau_min: float = eqx.field(static=True, default=0.99)
    bound_push: float = eqx.field(static=True, default=0.01)
    bound_frac: float = eqx.field(static=True, default=0.01)
    hessian: str = eqx.field(static=True, default="auto")
    scaling: str = eqx.field(static=True, default="gradient")
    restoration: bool = eqx.field(static=True, default=True)
    max_failed_evaluations: int = eqx.field(static=True, default=10)
    kkt: Any = eqx.field(static=True, default_factory=DenseKKT)

    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """Push `x0` inside its bounds, set the slacks from the inequalities, estimate the multipliers."""
        raise NotImplementedError

    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One barrier step: the KKT solve, fraction to the boundary, the filter line search."""
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
