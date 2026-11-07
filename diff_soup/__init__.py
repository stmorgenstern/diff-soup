"""diff_soup: differentiable, GPU-native JAX library for stiff PDEs and DAEs.

Three layers with two public scientific namespaces:

* Layer 1 (public):  ``domains``, ``optimization``
* Layer 2 (problem): ``problem``
* Layer 3 (private): ``timestepping``, ``nonlinear``, ``linear``

**Curated top-level namespace.** This file re-exports the subset of symbols
that users import directly from ``diff_soup``.  Every other symbol lives
behind its sub-namespace. ``diff_soup.optimization.inference`` is not imported eagerly:
``ds.optimization.inference`` resolves it on first access, so the idiomatic spelling is
``ds.optimization.inference.x``.

Import direction: L1 → L2 → L3, never upward. Enforced by the
contract test ``tests/contracts/test_architecture.py`` and, for module-level
imports between first-level packages, by ``.github/scripts/check_layers.py``.

Design principles:
- A deprecated path warns with a ``DeprecationWarning`` and lasts one release.
- No string-flag dispatch in the public signature.
- One verb per job: ``pdesolve``, ``lower``, and ``optimization.optsolve``.
- One AD entry per verb: ``pdesolve`` → solver's ``solve``, ``lower`` → IFT
  chain.
- Underscore = private; the contract test walks the import surface.
"""

import sys as _sys
import types as _types

# First import of the package: makes the changes to the process that
# ``diff_soup._environment`` lists (the XLA flags, JAX's constant hoisting, the
# typed-FFI custom-call builder) before anything else imports ``jax``.
import diff_soup._environment  # noqa: F401

# ---------------------------------------------------------------------------
# Root-private modules: errors, layout, schedules
# ---------------------------------------------------------------------------
from diff_soup import distributed

# ``compile`` compiles one ``pdesolve`` call ahead of time without running it.
from diff_soup._compile import compile
from diff_soup._distributed import global_mesh, mesh_groups
from diff_soup._errors import (
    BackendError,
    ConfigurationError,
    DiffSoupError,
    LoweringError,
    OptionalDependencyError,
)
from diff_soup._layout import Layout
from diff_soup._precision import Precision

# ``Schedule``/``Segment`` describe an operating protocol: segments that each hold
# one row of a model's operational control table at a constant target.
from diff_soup._schedule import (
    Schedule,
    ScheduleArrays,
    Segment,
    SegmentEnds,
    schedule_arrays,
    solve_schedule,
)
from diff_soup._solution import RESULT
from diff_soup._version import __version__ as __version__

# ---------------------------------------------------------------------------
# Layer 2 — problem IR + lower() seam
# ---------------------------------------------------------------------------
# ``ds.problem`` is the ``diff_soup.problem`` package and is also callable as
# the Recipe API verb (see ``_CallableProblemPackage`` below), so both
# ``ds.problem(name).domain(...).field(...)`` and ``ds.problem.implicit`` work.
from diff_soup.domains import cartesian_grid_1d
from diff_soup.domains.measures import domain_measure, surface_measure
from diff_soup.domains.operators import Operators

# ``pdesolve`` is the public forward-simulation entry point.
from diff_soup.pdesolve import pdesolve
from diff_soup.problem import (
    BoundaryIntegralConstraint,
    BoundarySet,
    Constraint,
    DAEStructure,
    Dependencies,
    DomainIntegralConstraint,
    EquationSet,
    Family,
    FamilyGraph,
    FamilyGraphExchange,
    GalvanostaticConstraint,
    ImplicitProblem,
    LoweredProblem,
    MassOperator,
    MassProblem,
    MergedView,
    NodalExtractOp,
    NodalInjectOp,
    OperationalControlConstraint,
    ParameterSet,
    ProblemSpec,
    ReduceResidualConstraint,
    ResidualLinearization,
    ResidualSparsity,
    RowOwner,
    Scales,
    SolverView,
    TopologyReport,
    TransferOp,
    VarRef,
    default_theta,
    jacobian_nnz,
    jacobian_pattern,
    lower,
    parameter_gradient,
    solver_view,
    tags,
    topology_report,
)
from diff_soup.problem._spec.recipe import recipe

# ``project_algebraic`` is the public full-state algebraic projection: pin the
# differential fields and globals, solve the algebraic fields and globals onto
# the constraint manifold, with paired implicit forward/reverse derivatives.
from diff_soup.projection import (
    AlgebraicProjectionError,
    AlgebraicProjectionOptions,
    AlgebraicProjectionResult,
    AlgebraicProjectionStatus,
    project_algebraic,
)

# ---------------------------------------------------------------------------
# Layer 3 — the scalar solver suite + events + output policy
# ---------------------------------------------------------------------------
from diff_soup.timestepping import (
    BoxConstrainedActiveSet,
    CheckpointAdjoint,
    ControlState,
    DenseOutputPolicy,
    DiffraxStepper,
    Event,
    ForwardAdjoint,
    Newton,
    ResolvedTolerances,
    SaveAt,
    ToleranceSpec,
)

# ``JNDF``/``JBDF``/``JIDA`` are the public scalar solver suite (the vsvo
# engine classes). All three return the shared ``Solution`` record below.
from diff_soup.timestepping.vsvo.classes import JBDF, JIDA, JNDF

# ``Solution``/``SolverStats`` are the records every solver in the
# public suite returns/carries (``timestepping.vsvo.solution``); ``RESULT``
# (imported above) is the one terminal-status enum every record reuses.
from diff_soup.timestepping.vsvo.solution import Solution, SolverStats


class _CallableProblemPackage(_types.ModuleType):
    """The ``diff_soup.problem`` package, callable as the Recipe API verb.

    The Recipe API starts with ``ds.problem("name")``, and ``ds.problem`` is
    also the package that holds the problem model (``ds.problem.implicit``,
    ``ds.problem.ProblemSpec``). Giving the package module this class lets one
    attribute serve both: calling the package calls its ``problem`` function,
    and every attribute lookup is the ordinary module lookup.
    ``from diff_soup.problem import problem`` is the plain function.
    """

    def __call__(self, *args, **kwargs):
        """Start a recipe: forwards to ``diff_soup.problem.problem``."""
        return self.problem(*args, **kwargs)

    @property
    def __signature__(self):
        """The signature of ``diff_soup.problem.problem``, for ``inspect.signature``."""
        import inspect

        return inspect.signature(self.problem)


_sys.modules["diff_soup.problem"].__class__ = _CallableProblemPackage
problem = _sys.modules["diff_soup.problem"]

# ---------------------------------------------------------------------------
# __all__ — the curated public surface
# ---------------------------------------------------------------------------

# ``optimization`` and ``learnables`` are resolved on first access by the module
# ``__getattr__`` below; every other name is bound when ``diff_soup`` imports.
__all__ = [
    "distributed",
    "optimization",
    "problem",
    "global_mesh",
    "mesh_groups",
    "recipe",
    # ---- root-private pillars ----
    "Layout",
    "Precision",
    # ---- equation-authoring surface ----
    # ``Operators`` is the equation-authoring surface (defined in
    # ``diff_soup.domains.operators.Operators``); top-level re-export so
    # ``ds.Operators(mesh)`` works without a domains import.
    "Operators",
    # ---- grid factories ----
    "cartesian_grid_1d",
    # ---- integration measures (host-side reductions) ----
    "domain_measure",
    "surface_measure",
    # ---- result types ----
    "RESULT",
    "Solution",
    "SolverStats",
    "ResolvedTolerances",
    "ToleranceSpec",
    # ---- errors ----
    "DiffSoupError",
    "ConfigurationError",
    "LoweringError",
    "BackendError",
    "OptionalDependencyError",
    # ---- Layer 2: problem spec + IR + seam ----
    "ProblemSpec",
    "ParameterSet",
    "lower",
    "default_theta",
    "parameter_gradient",
    "learnables",
    # ---- Layer 2: the solver problem model ----
    "MassProblem",
    "ImplicitProblem",
    "DAEStructure",
    "Dependencies",
    "MassOperator",
    "ResidualSparsity",
    "ResidualLinearization",
    # ---- cross-component transfer operators ----
    "TransferOp",
    "NodalExtractOp",
    "NodalInjectOp",
    # ---- iteration-matrix structure diagnostics ----
    "topology_report",
    "jacobian_nnz",
    "jacobian_pattern",
    "solver_view",
    "SolverView",
    "TopologyReport",
    "RowOwner",
    # ---- forward-simulation entry point ----
    "pdesolve",
    "compile",
    # ---- full-state algebraic projection ----
    "project_algebraic",
    "AlgebraicProjectionOptions",
    "AlgebraicProjectionResult",
    "AlgebraicProjectionStatus",
    "AlgebraicProjectionError",
    "EquationSet",
    "LoweredProblem",
    "Constraint",
    "GalvanostaticConstraint",
    "DomainIntegralConstraint",
    "ReduceResidualConstraint",
    "BoundaryIntegralConstraint",
    "OperationalControlConstraint",
    "Family",
    "FamilyGraph",
    "FamilyGraphExchange",
    "VarRef",
    "MergedView",
    "Scales",
    # ---- boundary-set selectors ----
    "BoundarySet",
    "tags",
    # ---- Layer 3: the scalar solver suite ----
    "JNDF",
    "JBDF",
    "JIDA",
    "DiffraxStepper",
    # ---- Layer 3: nonlinear (re-exported from timestepping) ----
    "Newton",
    "BoxConstrainedActiveSet",
    # ---- save/event ----
    "SaveAt",
    "DenseOutputPolicy",
    "Event",
    "ControlState",
    # ---- operating protocols ----
    "Schedule",
    "Segment",
    "ScheduleArrays",
    "SegmentEnds",
    "schedule_arrays",
    "solve_schedule",
    # ---- differentiation strategies ----
    "ForwardAdjoint",
    "CheckpointAdjoint",
]


def __getattr__(name: str):
    """Resolve optimization, the learnable selector and the deprecated inference alias.

    The inference package imports the whole fitting and statistics stack, so it
    is not loaded with ``import diff_soup``; ``ds.optimization.inference`` imports it on
    demand and returns the package. ``ds.inference`` imports the deprecated alias
    ``diff_soup.inference``, which warns once and binds itself here. Any other unknown
    attribute raises ``AttributeError`` as usual.
    """
    if name == "inference":
        import diff_soup.inference as inference

        return inference
    if name == "learnables":
        import importlib

        return importlib.import_module("diff_soup.optimization.inference").learnables
    if name == "optimization":
        import importlib

        return importlib.import_module("diff_soup.optimization")
    raise AttributeError(f"module 'diff_soup' has no attribute {name!r}")
