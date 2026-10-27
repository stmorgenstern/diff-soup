"""diff_soup.nonlinear: nonlinear solver contract + adjoint family.

The public contract face lives in [`diff_soup.nonlinear._api`][diff_soup.nonlinear._api]; the fused
kernel + IFT rules in [`diff_soup.nonlinear.solver`][diff_soup.nonlinear.solver] /
[`diff_soup.nonlinear.box_newton`][diff_soup.nonlinear.box_newton].

``nonlinear/`` exposes nothing at the top-level ``diff_soup`` namespace.
``Newton`` / ``BoxConstrainedActiveSet`` surface where users meet them — as
timestepping correctors — re-exported from ``diff_soup.timestepping``.

``__all__`` is the curated 7-symbol surface; the kernel entry points and box
config/AD entries below stay importable for internal use and the test suite
but are NOT public surface. The raw box primal
``box_constrained_solve_primal_unified`` is intentionally not re-exported:
production callers go through the AD-aware
``box_constrained_solve_{vjp,jvp}`` entries.
"""

from diff_soup._adjoints import AbstractAdjoint, ForwardAdjoint, ImplicitAdjoint
from diff_soup.nonlinear._api import (
    BoxConstrainedActiveSet,
    Newton,
    NewtonInfo,
    NonlinearSolver,
)
from diff_soup.nonlinear._api import (
    NonlinearConfig as NonlinearConfig,
)
from diff_soup.nonlinear._api import (
    NonlinearState as NonlinearState,
)
from diff_soup.nonlinear.box_bounds import BoxBounds as BoxBounds
from diff_soup.nonlinear.box_newton import (
    BoxConstrainedUnifiedConfig as BoxConstrainedUnifiedConfig,
)
from diff_soup.nonlinear.box_newton import (
    BoxConstrainedUnifiedInfo as BoxConstrainedUnifiedInfo,
)
from diff_soup.nonlinear.box_newton import (
    box_constrained_solve_jvp as box_constrained_solve_jvp,
)
from diff_soup.nonlinear.box_newton import (
    box_constrained_solve_vjp as box_constrained_solve_vjp,
)
from diff_soup.nonlinear.solver import (
    nonlinear_solve_ift_jvp as nonlinear_solve_ift_jvp,
)
from diff_soup.nonlinear.solver import (
    nonlinear_solve_ift_vjp as nonlinear_solve_ift_vjp,
)
from diff_soup.nonlinear.solver import (
    nonlinear_solve_primal as nonlinear_solve_primal,
)

__all__ = [
    # Curated public surface (7 symbols)
    "NonlinearSolver",
    "Newton",
    "BoxConstrainedActiveSet",
    "NewtonInfo",
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "ForwardAdjoint",
]
