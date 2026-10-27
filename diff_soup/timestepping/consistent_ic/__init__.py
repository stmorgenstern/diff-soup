"""Euler-Newton consistent initial conditions for the solver suite.

The Euler-Newton kernels start a differential-algebraic solve from a state
on the algebraic manifold: solve the algebraic components of ``y`` and the
differential components of ``ydot`` (``ALG_AND_YDOT``), or solve all of
``y`` at ``cj = 0`` given ``ydot`` (``ALL_Y_GIVEN_YDOT``). Both are
differentiable through the converged root by the implicit-function rule
(never through the Newton iteration). The shared records [ICMode][],
[ICResult][], and [ICConfig][] describe the mode, the result, and
the tolerances/constants.

The default startup projection — pin the differential rows, solve the local
and global algebraic values — runs the capability-routed
initialization solve on the solver problem instead
([SolverView.consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state]).
"""

from diff_soup.timestepping.consistent_ic.api import (
    ICRequest,
    ICState,
    InitMode,
    consistent_init,
)
from diff_soup.timestepping.consistent_ic.euler_newton import euler_newton_ic
from diff_soup.timestepping.consistent_ic.pin_differential import (
    ICConfig,
    ICMode,
    ICResult,
)

__all__ = [
    "consistent_init",
    "ICRequest",
    "ICState",
    "InitMode",
    "euler_newton_ic",
    "ICResult",
    "ICMode",
    "ICConfig",
]
