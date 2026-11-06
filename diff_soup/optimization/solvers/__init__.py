"""The compiled constrained solvers and the protocol they share.

The five host-loop solvers (`NewtonTrustRegion`, `LevenbergMarquardt`, `BFGS`,
`GradientDescent`, `OptaxSolver`) move here from `diff_soup.inference.solvers` with the
inference package.
"""

from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
    SolveStatus,
)
from diff_soup.optimization.solvers.augmented_lagrangian import AugmentedLagrangian
from diff_soup.optimization.solvers.interior_point import InteriorPoint
from diff_soup.optimization.solvers.sqp import SQP

__all__ = [
    "AugmentedLagrangian",
    "CompiledSolver",
    "InteriorPoint",
    "SQP",
    "SolveRecord",
    "SolveStatus",
    "SolverState",
]
