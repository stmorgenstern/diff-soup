"""Layer-2 EquationSet IR node hierarchy.

The IR consumed by ``lower()``: the ``EquationSet`` composition
root, the typed ``Constraint`` / ``Observation`` / ``Control`` discriminated
unions, ``Variable`` / ``GlobalVariable`` handles, boundary-condition nodes,
owner-aware ``VarRef`` / ``MergedView`` reference tracking, the structural
``content_hash`` fingerprinter, and ``Scales`` non-dim bookkeeping.

Every node is a frozen, pytree-registered ``eqx.Module``; host-side metadata
lives in ``eqx.field(static=True)`` and array data in runtime fields.
``object.__setattr__`` is forbidden package-wide.  The ``Constraint`` union is
closed — adding a member requires extending this package and all dispatch sites.
``LoweredProblem`` carries ``layout`` and ``precision`` pillars.

``coupling.py`` provides the ``Coupling`` node and the ``TransferOp`` hierarchy
(``NodalInjectOp`` / ``NodalExtractOp`` / ``SphericalSurfaceTraceOp`` /
``RadialSurfaceInjectOp`` / ``FamilyScatterOp``) along with the typed
[Reduce][] union (``Sum`` | ``Mean``) replacing the legacy 3-value
``reduce`` string.  ``Family.domain`` is a runtime pytree leaf so the per-member
domain arrays remain differentiable.  The slim single-domain path never populates
``couplings`` or ``families``.  ``Submodel`` is not exported — problems are
authored using the Recipe API (``ds.problem(...)``).

Each node type is in its own module; ``content_hash.py`` consolidates the full
structural fingerprinter for the whole IR in one place.
"""

from __future__ import annotations

from ._merged_view import MergedView
from .bc import (
    BoundaryCondition,
    BoundarySpec,
    DirichletBC,
    FluxBC,
    PeriodicBC,
    RobinBC,
    SymmetryBC,
    TractionBC,
    _BoundaryFn,
)
from .component import Component
from .constraint import (
    BoundaryIntegralConstraint,
    Constraint,
    DomainIntegralConstraint,
    FluxTotalConstraint,
    GalvanostaticConstraint,
    OperationalControlConstraint,
    ReduceResidualConstraint,
)
from .content_hash import content_hash
from .control import Control, DerivedControl, DirectControl
from .coupling import (
    Coupling,
    FamilyScatterOp,
    Mean,
    NodalExtractOp,
    NodalInjectOp,
    RadialSurfaceInjectOp,
    Reduce,
    SphericalSurfaceTraceOp,
    Sum,
    TransferOp,
)
from .equation import Equation, MultiEquation
from .equation_set import EquationSet
from .family import Family
from .family_graph import FamilyGraph, FamilyGraphExchange
from .lowered_problem import LoweredProblem
from .observation import (
    BoundaryMeanObservation,
    FieldObservation,
    Observable,
    Observation,
    ScalarObservation,
    SpatialMeanObservation,
)
from .scales import Scales, identity_scales
from .var_ref import AmbiguousVariableError, UnregisteredGlobalError, VarRef, build_var_id_to_ref
from .variable import GlobalVariable, Variable

__all__ = [
    # Variable handles
    "Variable",
    "GlobalVariable",
    # Boundary conditions
    "BoundarySpec",
    "BoundaryCondition",
    "DirichletBC",
    "FluxBC",
    "_BoundaryFn",
    "RobinBC",
    "TractionBC",
    "SymmetryBC",
    "PeriodicBC",
    # Equations
    "Equation",
    "MultiEquation",
    # Composition
    "Component",
    "EquationSet",
    "Family",
    "FamilyGraph",
    "FamilyGraphExchange",
    "LoweredProblem",
    "content_hash",
    # Cross-component / family coupling IR (TransferOps + typed Reduce)
    "Coupling",
    "TransferOp",
    "NodalInjectOp",
    "NodalExtractOp",
    "FamilyScatterOp",
    "RadialSurfaceInjectOp",
    "SphericalSurfaceTraceOp",
    "Reduce",
    "Sum",
    "Mean",
    # Constraint discriminated union (closed)
    "Constraint",
    "GalvanostaticConstraint",
    "DomainIntegralConstraint",
    "ReduceResidualConstraint",
    "BoundaryIntegralConstraint",
    "FluxTotalConstraint",
    "OperationalControlConstraint",
    # Observation discriminated union
    "Observation",
    "Observable",
    "ScalarObservation",
    "FieldObservation",
    "BoundaryMeanObservation",
    "SpatialMeanObservation",
    # Control discriminated union
    "Control",
    "DirectControl",
    "DerivedControl",
    # Non-dimensionalisation bookkeeping
    "Scales",
    "identity_scales",
    # Owner-aware references / view
    "VarRef",
    "MergedView",
    "AmbiguousVariableError",
    "UnregisteredGlobalError",
    "build_var_id_to_ref",
]
