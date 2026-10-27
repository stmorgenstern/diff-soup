"""Optimizer engines of ``diff_soup.inference``.

Every engine is an [OptSolver][diff_soup.inference._abc.OptSolver] object that
``fit`` drives through ``solver.solve(problem, callbacks=...)``; inject the
object, not a string. ``LevenbergMarquardt`` is the least-squares engine used by
default, ``BFGS`` and ``GradientDescent`` are first-order alternatives, and
``Optax`` adapts an external optimizer.

``value_and_jacfwd`` and the tiled variants in
[tangent][diff_soup.inference.solvers.tangent] are the forward-mode Jacobian
building blocks the engines share.
"""

from diff_soup.inference._abc import OptSolver
from diff_soup.inference.solvers.bfgs import BFGS
from diff_soup.inference.solvers.gradient_descent import GradientDescent
from diff_soup.inference.solvers.levenberg_marquardt import LevenbergMarquardt
from diff_soup.inference.solvers.newton_trust_region import NewtonTrustRegion
from diff_soup.inference.solvers.optax_solver import Optax
from diff_soup.inference.solvers.tangent import (
    normalize_tangent_tile,
    value_and_jacfwd,
    value_and_jacfwd_tiled,
)

__all__ = [
    "OptSolver",
    "LevenbergMarquardt",
    "NewtonTrustRegion",
    "BFGS",
    "GradientDescent",
    "Optax",
    "value_and_jacfwd",
    "value_and_jacfwd_tiled",
    "normalize_tangent_tile",
]
