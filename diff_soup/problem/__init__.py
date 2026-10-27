"""diff_soup.problem: problem IR and the ``lower()`` compilation seam.

This module forms the problem layer (Layer 2) of diff_soup's layered
architecture.  It imports only from Layer 3 (infrastructure) and never from
Layer 1 (physics).

Public exports
--------------
- ``ProblemSpec`` / ``problem`` / ``lower`` — Recipe API builder and the
  lowering entry point; ``ParameterSet`` selects its returned
  parameter/function handles for optimization and UQ.
- ``EquationSet``, ``LoweredProblem`` — compiled IR roots.
- ``Constraint``, ``GalvanostaticConstraint``, ``DomainIntegralConstraint``,
  ``ReduceResidualConstraint``, ``BoundaryIntegralConstraint`` — constraint
  union types used in the IR.
- ``VarRef``, ``MergedView``, ``Family``, ``Scales`` — IR identity and scaling
  primitives.
- ``MassProblem`` / ``ImplicitProblem`` — the authored solver-problem
  contracts consumed by every runtime engine.
- ``InconsistentInitializationError`` — raised when a DAE's consistent-initial-
  condition projection cannot start from the supplied state.
- ``ConsistentInitProblem`` — the runtime-checkable solver view surface
  retained consistent-initialization kernels consume.

``Submodel`` is intentionally *not* exported; use the Recipe API
(``ProblemSpec`` / ``problem()``) to author problems instead.
"""

from diff_soup.domains.boundary_set import BoundarySet, tags
from diff_soup.problem._consistency import InconsistentInitializationError
from diff_soup.problem._ir import (
    BoundaryIntegralConstraint,
    Constraint,
    DomainIntegralConstraint,
    EquationSet,
    Family,
    FamilyGraph,
    FamilyGraphExchange,
    GalvanostaticConstraint,
    LoweredProblem,
    MergedView,
    OperationalControlConstraint,
    ReduceResidualConstraint,
    Scales,
    VarRef,
)
from diff_soup.problem._ir.coupling import (
    NodalExtractOp,
    NodalInjectOp,
    TransferOp,
)
from diff_soup.problem._spec import (
    ProblemSpec,
    lower,
    problem,
)
from diff_soup.problem.consistent_init import ConsistentInitProblem
from diff_soup.problem.implicit import (
    DAEStructure,
    Dependencies,
    ImplicitProblem,
    MassOperator,
    MassProblem,
    ResidualLinearization,
    ResidualSparsity,
    get_or_build_solver_view,
)
from diff_soup.problem.implicit.topology_report import (
    RowOwner,
    TopologyReport,
    topology_report,
)
from diff_soup.problem.parameters import ParameterSet
from diff_soup.timestepping.vsvo.common.contract import register_problem_coercion


def default_theta(problem, values=None):
    """Return a problem's default parameter values, with chosen parameters replaced.

    Args:
        problem: A built problem, such as a [`LoweredProblem`][diff_soup.problem.LoweredProblem].
        values: Mapping from a parameter returned by `spec.param` or `spec.function` to its value, broadcast to that parameter's shape and dtype.

    Returns:
        The parameter pytree, ready to pass as `theta=`.

    Raises:
        ValueError: If a parameter in `values` is not declared on this problem.
    """
    surface = get_or_build_solver_view(problem).default_theta
    theta = surface() if callable(surface) else surface
    if not values:
        return theta

    import jax
    import jax.numpy as jnp

    from diff_soup.problem._registry import handle_leaf_ids

    replacements: dict = {}
    for handle, value in values.items():
        for leaf_id in handle_leaf_ids(problem, handle):
            replacements[leaf_id] = value

    def _replace(leaf):
        value = replacements.get(id(leaf))
        if value is None:
            return leaf
        return jnp.broadcast_to(jnp.asarray(value, dtype=leaf.dtype), jnp.shape(leaf))

    return jax.tree_util.tree_map(_replace, theta)


__all__ = [
    # Builder + lowering
    "ProblemSpec",
    "ParameterSet",
    "problem",
    "lower",
    # Default differentiable-parameter pytree accessor
    "default_theta",
    # Boundary-set selectors (where a BC / boundary integral applies)
    "BoundarySet",
    "tags",
    # IR roots
    "EquationSet",
    "LoweredProblem",
    # IR nodes
    "Constraint",
    "GalvanostaticConstraint",
    "DomainIntegralConstraint",
    "ReduceResidualConstraint",
    "BoundaryIntegralConstraint",
    "OperationalControlConstraint",
    "Family",
    "FamilyGraph",
    "FamilyGraphExchange",
    # Cross-component transfer operators (the read-through coupling contract)
    "TransferOp",
    "NodalExtractOp",
    "NodalInjectOp",
    # Iteration-matrix structure diagnostics
    "topology_report",
    "TopologyReport",
    "RowOwner",
    # Identity/ownership
    "VarRef",
    "MergedView",
    # Scaling
    "Scales",
    # Consistent-IC precondition exception
    "InconsistentInitializationError",
    # The solver view surface retained consistent-init kernels consume
    "ConsistentInitProblem",
    # The solver problem model
    "MassProblem",
    "ImplicitProblem",
    "DAEStructure",
    "Dependencies",
    "MassOperator",
    "ResidualSparsity",
    "ResidualLinearization",
]

# Wire the scalar solver suite's problem-coercion seam so direct engine calls
# and the top-level entry resolve the same solver problem. Registering at
# problem-layer import time keeps the import direction problem -> timestepping.
register_problem_coercion(get_or_build_solver_view)
