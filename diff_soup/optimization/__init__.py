"""Optimization problems, their constraints and their solvers.

Every user-facing problem shares one base,
[AbstractOptimizationProblem][diff_soup.optimization.AbstractOptimizationProblem], and
lowers to the numeric [OptimizationProblem][diff_soup.optimization.OptimizationProblem]
a solver takes. `diff_soup.optimization.inference` fits models to measurements,
`diff_soup.optimization.control` poses and runs control problems, and
`diff_soup.optimization.design` is reserved for experiment design.

The contract is batjax's `docs/2026-11-06-optimization-module.md`. Until the inference
package moves under this one, the numeric types are imported from their old home.
"""

from diff_soup.inference._abc import (
    STRUCTURE_KEYS,
    LeastSquaresProblem,
    OptimizationProblem,
    OptSolver,
)
from diff_soup.optimization.constraints import (
    CONSTRAINT_JACOBIAN,
    LAGRANGIAN_HESSIAN,
    ConstraintClosures,
    Constraints,
)
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.lanes import lanes
from diff_soup.optimization.problem import AbstractOptimizationProblem
from diff_soup.optimization.solvers import (
    SQP,
    AugmentedLagrangian,
    CompiledSolver,
    InteriorPoint,
    SolveRecord,
    SolveStatus,
)

__all__ = [
    "AbstractOptimizationProblem",
    "AugmentedLagrangian",
    "CONSTRAINT_JACOBIAN",
    "CompiledSolver",
    "ConstraintClosures",
    "Constraints",
    "DenseKKT",
    "InteriorPoint",
    "KKTSolve",
    "LAGRANGIAN_HESSIAN",
    "LeastSquaresProblem",
    "OptSolver",
    "OptimizationProblem",
    "SQP",
    "STRUCTURE_KEYS",
    "SolveRecord",
    "SolveStatus",
    "lanes",
]
