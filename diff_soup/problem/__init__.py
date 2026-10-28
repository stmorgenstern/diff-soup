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

    A scalar or array value is broadcast to the parameter's shape and dtype, and
    for a function into each of its arrays. A function also takes a module of
    its declared structure, whose arrays replace the declared ones.

    Args:
        problem: A built problem, such as a [`LoweredProblem`][diff_soup.problem.LoweredProblem].
        values: Mapping from a parameter returned by `spec.param` or `spec.function` to its value.

    Returns:
        The parameter pytree, ready to pass as `theta=`.

    Raises:
        TypeError: If a key is not a parameter, or a value is not numeric or is complex for a real parameter.
        ValueError: If a parameter is not declared on this problem or holds no array, or its value is `None` or does not fit.
    """
    own = getattr(problem, "default_theta", None)
    if callable(own):
        # The problem's own leaves: a problem equal in content to another one
        # shares that one's solver view, whose theta holds the other's leaves.
        theta = own()
    else:
        surface = get_or_build_solver_view(problem).default_theta
        theta = surface() if callable(surface) else surface
    if values is None:
        return theta
    from collections.abc import Mapping

    if not isinstance(values, Mapping):
        raise TypeError(
            "default_theta: values must map parameters returned by spec.param or "
            f"spec.function to their values; got {type(values).__name__}."
        )
    if not values:
        return theta

    import jax

    replacements: dict = {}
    owned: list = []
    for handle, value in values.items():
        pairs = _handle_replacements(problem, handle, value)
        owned.append((handle.name, frozenset(id(leaf) for leaf, _ in pairs)))
        for leaf, new in pairs:
            replacements[id(leaf)] = new

    found: set = set()

    def _replace(leaf):
        key = id(leaf)
        if key not in replacements:
            return leaf
        found.add(key)
        return replacements[key]

    theta = jax.tree_util.tree_map(_replace, theta)
    for name, ids in owned:
        if not ids & found:
            raise ValueError(
                f"default_theta: parameter {name!r} matches no array of this problem's "
                "parameters; pass the parameter returned when this problem was built."
            )
    return theta


def _handle_replacements(problem, handle, value) -> list:
    """Return ``(declared_leaf, new_leaf)`` pairs setting ``handle`` to ``value`` on ``problem``."""
    import equinox as eqx
    import jax

    from diff_soup.problem._registry import declared_handle_value
    from diff_soup.problem.parameters import _function_parameter_partition

    if not (hasattr(handle, "identity") and hasattr(handle, "name")):
        raise TypeError(
            "default_theta: keys must be parameters returned by spec.param or "
            f"spec.function, not {type(handle).__name__} {handle!r}."
        )
    name = handle.name
    if value is None:
        raise ValueError(
            f"default_theta: the value for {name!r} is None; pass a scalar, an array "
            "or, for a function, a module."
        )
    kind, declared = declared_handle_value(problem, handle)
    if kind == "function":
        declared, _ = _function_parameter_partition(declared)
    old = [leaf for leaf in jax.tree_util.tree_leaves(declared) if eqx.is_inexact_array(leaf)]
    if not old:
        raise ValueError(
            f"default_theta: {name!r} holds no floating-point array, so theta cannot "
            "change it. Declare its value as an array, or a function as an eqx.Module "
            "that holds its arrays."
        )
    if _is_array_like(value) or (kind != "function" and eqx.is_array(declared)):
        return [(leaf, _leaf_value(name, value, leaf, broadcast=True)) for leaf in old]
    new = _function_parameter_partition(value)[0] if kind == "function" else value
    if jax.tree_util.tree_structure(new) != jax.tree_util.tree_structure(declared):
        shapes = [tuple(getattr(leaf, "shape", ())) for leaf in jax.tree_util.tree_leaves(new)]
        raise ValueError(
            f"default_theta: the value for {name!r} must have the structure of its "
            f"declared value, a {type(declared).__name__} with arrays of shapes "
            f"{[tuple(leaf.shape) for leaf in old]}; got a {type(value).__name__} "
            f"with leaves of shapes {shapes}."
        )
    return [
        (leaf, _leaf_value(name, new_leaf, leaf, broadcast=False))
        for leaf, new_leaf in zip(
            jax.tree_util.tree_leaves(declared), jax.tree_util.tree_leaves(new)
        )
        if eqx.is_inexact_array(leaf)
    ]


def _is_array_like(value) -> bool:
    """Whether ``value`` is a number or an array, as opposed to a module or container."""
    import numbers

    import jax
    import numpy as np

    return isinstance(value, (numbers.Number, np.ndarray, np.generic, jax.Array))


def _leaf_value(name, value, leaf, *, broadcast):
    """Return ``value`` cast to ``leaf``'s dtype and checked against (or broadcast to) its shape."""
    import jax.numpy as jnp
    import numpy as np

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        try:
            dtype = np.asarray(value).dtype
        except Exception:
            dtype = np.dtype(object)
    if dtype == np.bool_ or not jnp.issubdtype(dtype, jnp.number):
        raise TypeError(
            f"default_theta: the value for {name!r} must hold numbers, got "
            f"{type(value).__name__} of dtype {dtype}."
        )
    if jnp.issubdtype(dtype, jnp.complexfloating) and not jnp.issubdtype(
        leaf.dtype, jnp.complexfloating
    ):
        raise TypeError(
            f"default_theta: the value for {name!r} is complex, but the parameter is "
            f"real ({leaf.dtype})."
        )
    shape = tuple(jnp.shape(value))
    if broadcast:
        try:
            return jnp.broadcast_to(jnp.asarray(value, dtype=leaf.dtype), leaf.shape)
        except ValueError:
            pass
    elif shape == tuple(leaf.shape):
        return jnp.asarray(value, dtype=leaf.dtype)
    raise ValueError(
        f"default_theta: the value for {name!r} has an array of shape {shape}, which "
        f"does not {'broadcast to' if broadcast else 'match'} the declared shape "
        f"{tuple(leaf.shape)}."
    )


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
