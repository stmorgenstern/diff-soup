"""Optimization problems, their constraints and their solvers.

Every user-facing problem shares one base,
[AbstractOptimizationProblem][diff_soup.optimization.AbstractOptimizationProblem], and
lowers to the numeric [OptimizationProblem][diff_soup.optimization.OptimizationProblem]
a solver takes. `diff_soup.optimization.inference` fits models to measurements, and
`diff_soup.optimization.control` poses and runs control problems.

`diff_soup.inference` is a deprecated alias of `diff_soup.optimization.inference`, kept
for one release.
"""

# The inference package loads first: its fitting stack and the host-loop solvers import
# each other, in the order its own __init__ sets.
from diff_soup.optimization import inference  # noqa: I001
from diff_soup.optimization._abc import LeastSquaresProblem, OptimizationProblem
from diff_soup.optimization.constraints import Constraints
from diff_soup.optimization.lanes import lanes
from diff_soup.optimization.problem import AbstractOptimizationProblem
from diff_soup.optimization.solvers import (
    BFGS,
    SQP,
    AugmentedLagrangian,
    GradientDescent,
    InteriorPoint,
    LevenbergMarquardt,
    NewtonTrustRegion,
    Optax,
    SolveRecord,
)
from diff_soup.optimization.inference.fitting._optsolve import optsolve
from diff_soup.optimization.inference.fitting.result import OptResult, TerminationStatus
from diff_soup.optimization import control

__all__ = [
    "AbstractOptimizationProblem",
    "AugmentedLagrangian",
    "BFGS",
    "Constraints",
    "GradientDescent",
    "InteriorPoint",
    "LeastSquaresProblem",
    "LevenbergMarquardt",
    "NewtonTrustRegion",
    "OptResult",
    "Optax",
    "OptimizationProblem",
    "SQP",
    "SolveRecord",
    "TerminationStatus",
    "control",
    "inference",
    "lanes",
    "optsolve",
]
