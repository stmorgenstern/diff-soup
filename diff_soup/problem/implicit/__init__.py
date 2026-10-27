"""diff_soup.problem.implicit: the solver problem model.

Every supported transient problem lowers to one internal equation:

    F(t, y, ydot, theta, control) = 0

with explicit variable roles ([DAEStructure][diff_soup.problem.implicit.DAEStructure])
and explicit dependency declarations
([Dependencies][diff_soup.problem.implicit.Dependencies]). Two public frontends
state genuinely different contracts: [MassProblem][diff_soup.problem.implicit.MassProblem]
for the mass form ``F = D(t, y, theta, control) @ ydot - f(t, y, theta, control)``
built on a [MassOperator][diff_soup.problem.implicit.MassOperator], and
[ImplicitProblem][diff_soup.problem.implicit.ImplicitProblem] for a residual
written directly, with its exact derivative structure declared through
[ResidualSparsity][diff_soup.problem.implicit.ResidualSparsity] /
[ResidualLinearization][diff_soup.problem.implicit.ResidualLinearization].
"""

from diff_soup.problem.implicit.consistent_init import (
    ConsistentInitPlan,
    ConsistentInitResult,
    build_consistent_init_plan,
    check_consistent_init,
    consistent_initial_state,
)
from diff_soup.problem.implicit.operator import MassOperator
from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem
from diff_soup.problem.implicit.residual import ResidualLinearization, ResidualSparsity
from diff_soup.problem.implicit.root_ift import (
    IterationMatrixCertificate,
    implicit_step_root,
)
from diff_soup.problem.implicit.signature import StructureSignature
from diff_soup.problem.implicit.solver_problem import (
    SolverView,
    get_or_build_solver_view,
)
from diff_soup.problem.implicit.structure import DAEStructure, Dependencies
from diff_soup.problem.implicit.theta_codec import ThetaCodec

__all__ = [
    "ConsistentInitPlan",
    "ConsistentInitResult",
    "DAEStructure",
    "Dependencies",
    "ImplicitProblem",
    "IterationMatrixCertificate",
    "MassOperator",
    "MassProblem",
    "ResidualLinearization",
    "ResidualSparsity",
    "SolverView",
    "StructureSignature",
    "ThetaCodec",
    "build_consistent_init_plan",
    "check_consistent_init",
    "consistent_initial_state",
    "get_or_build_solver_view",
    "implicit_step_root",
]
