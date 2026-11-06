"""The optimizers `optsolve` runs: the host-loop solvers and the compiled constrained ones.

Every solver is an [OptSolver][diff_soup.optimization.solvers.OptSolver] that `optsolve` drives
through `solver.solve(problem, callbacks=...)`. `NewtonTrustRegion`,
`LevenbergMarquardt`, `BFGS`, `GradientDescent` and `Optax` loop on the host;
`InteriorPoint`, `SQP` and `AugmentedLagrangian` are
[CompiledSolver][diff_soup.optimization.solvers.CompiledSolver]s whose whole loop compiles.
`value_and_jacfwd` and its tiled variants in `tangent` are the forward-mode Jacobian
building blocks the host-loop solvers share.
"""

from diff_soup.optimization._abc import STRUCTURE_KEYS, OptSolver
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
)
from diff_soup.optimization.solvers.augmented_lagrangian import AugmentedLagrangian
from diff_soup.optimization.solvers.bfgs import BFGS
from diff_soup.optimization.solvers.gradient_descent import GradientDescent
from diff_soup.optimization.solvers.interior_point import InteriorPoint
from diff_soup.optimization.solvers.levenberg_marquardt import LevenbergMarquardt
from diff_soup.optimization.solvers.newton_trust_region import NewtonTrustRegion
from diff_soup.optimization.solvers.optax_solver import Optax
from diff_soup.optimization.solvers.sqp import SQP
from diff_soup.optimization.solvers.tangent import (
    normalize_tangent_tile,
    value_and_jacfwd,
    value_and_jacfwd_tiled,
)

__all__ = [
    "AugmentedLagrangian",
    "BFGS",
    "CompiledSolver",
    "DenseKKT",
    "GradientDescent",
    "InteriorPoint",
    "KKTSolve",
    "LevenbergMarquardt",
    "NewtonTrustRegion",
    "Optax",
    "OptSolver",
    "SQP",
    "STRUCTURE_KEYS",
    "SolveRecord",
    "SolverState",
    "normalize_tangent_tile",
    "value_and_jacfwd",
    "value_and_jacfwd_tiled",
]
