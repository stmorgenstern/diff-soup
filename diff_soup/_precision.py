"""Precision: where every dtype decision in a solve lives.

Root-private leaf — imports only jax/jnp/numpy; any layer may import it.

The default ``Precision()`` uses all-fp64 behavior. Frozen, array-free,
hashable — safe as a static field and in content-hash cache keys.

A problem owns its precision: ``spec.build(precision=...)``,
``ds.lower(eqset, precision=...)``, ``MassProblem(..., precision=...)`` and
``ImplicitProblem(..., precision=...)`` all record one policy, and every
solve of that problem runs under it. [resolve_precision][] merges the
problem's policy with a solver's (still accepted) ``precision=`` argument,
and [cast_to_compute_dtype][] casts the inputs of a solve on entry.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

_SOLVE_MODES: frozenset[str] = frozenset({"fp64", "fp32_ir", "fp32", "adaptive"})
_SENSITIVITY_SOLVE_MODES: frozenset[str] = frozenset({"fp64", "fp32_ir", "fp32"})

DTypeSpec: TypeAlias = np.dtype | type | str
"""Any spelling of a dtype a caller may reasonably write.

``jnp.float32``, ``np.float32``, ``np.dtype("float32")`` and ``"float32"``
all name the same dtype; every dtype field of :class:`Precision` accepts all
four and stores the one canonical form.
"""


def _canonical_dtype(value: DTypeSpec | None) -> np.dtype | DTypeSpec | None:
    """Reduce one dtype spelling to the single form :class:`Precision` stores.

    :class:`Precision` is a cache key: two policies naming the same dtypes
    must compare equal *and* hash equal, or a solve compiled under one
    spelling is invisible to a lookup made with another and the compilation
    cache silently splits into duplicate entries. The spellings do not hash
    alike on their own — ``hash("float32") != hash(np.dtype("float32"))`` even
    though the two compare equal — so each dtype is reduced to a
    ``numpy.dtype`` instance as it is stored.

    ``None`` passes through: it is the documented "inherit the forward dtype"
    value of the optional fields, and it is *not* a dtype spelling even though
    ``numpy.dtype(None)`` would read it as ``float64``.

    A value numpy cannot read as a dtype also passes through untouched, so
    that validation reports it against the field it was supplied for rather
    than as a bare conversion failure.

    Args:
        value: A dtype in any accepted spelling, or ``None``.

    Returns:
        The equivalent ``numpy.dtype`` instance; ``None`` and unreadable
        values unchanged.
    """
    if value is None:
        return None
    try:
        return np.dtype(value)
    except (TypeError, ValueError):
        return value


class Precision(eqx.Module):
    """The dtype policy of a solve, declared once on the problem.

    Pass it as `precision=` when building the problem, for example `spec.build(precision=...)`. `Precision()` is float64 compute with the `"fp64"` solve mode, the library's default. `"fp32_ir"` refines float32 factors against a float64 residual, and `"adaptive"` switches between it and `"fp64"` at runtime.

    Args:
        compute_dtype: Floating dtype of states, parameters, residuals, and Jacobians inside the solve.
        solve_mode: Linear solve mode, one of `"fp64"`, `"fp32"`, `"fp32_ir"`, or `"adaptive"`, with `"fp64"` read as `"fp32"` under a float32 `compute_dtype`.
        adjoint_compute_dtype: Dtype of the reverse sweep, at least as wide as `compute_dtype`, which it inherits when `None`.
        adjoint_ir_tol: Positive refinement target of the reverse implicit-function solves, inheriting the forward target when `None`.
        forward_sensitivity_solve_mode: Solve mode of forward sensitivities, inheriting `solve_mode` when `None`.
        reverse_sensitivity_solve_mode: Solve mode of the reverse sweep, inheriting `solve_mode` when `None`.
        jacobian_values_dtype: Dtype of the Jacobian entries alone, no wider than `compute_dtype`.
    """

    compute_dtype: DTypeSpec = eqx.field(converter=_canonical_dtype, default=jnp.float64)
    solve_mode: Literal["fp64", "fp32_ir", "fp32", "adaptive"] = "fp64"
    adjoint_compute_dtype: DTypeSpec | None = eqx.field(converter=_canonical_dtype, default=None)
    adjoint_ir_tol: float | None = None
    forward_sensitivity_solve_mode: Literal["fp64", "fp32_ir", "fp32"] | None = None
    reverse_sensitivity_solve_mode: Literal["fp64", "fp32_ir", "fp32"] | None = None
    jacobian_values_dtype: DTypeSpec | None = eqx.field(converter=_canonical_dtype, default=None)

    def __post_init__(self) -> None:
        """Store the factor mode a single-precision compute dtype actually runs.

        Values computed in ``float32`` are factored in ``float32``; recording
        ``"fp64"`` for them would misreport the solve in every statistic.
        """
        if _is_single(self.compute_dtype) and self.solve_mode == "fp64":
            self.solve_mode = "fp32"

    def __check_init__(self) -> None:
        """Validate every dtype and solve-mode choice eagerly."""
        if self.solve_mode not in _SOLVE_MODES:
            raise ConfigurationError(
                f"Precision.solve_mode must be one of {sorted(_SOLVE_MODES)}; "
                f"got {self.solve_mode!r}."
            )
        for field_name in (
            "forward_sensitivity_solve_mode",
            "reverse_sensitivity_solve_mode",
        ):
            mode = getattr(self, field_name)
            if mode is not None and mode not in _SENSITIVITY_SOLVE_MODES:
                raise ConfigurationError(
                    f"Precision.{field_name} must be None or one of "
                    f"{sorted(_SENSITIVITY_SOLVE_MODES)}; got {mode!r}."
                )
        if self.adjoint_ir_tol is not None:
            try:
                tol = float(self.adjoint_ir_tol)
                finite = bool(np.isfinite(tol))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "Precision.adjoint_ir_tol must be a finite positive relative-residual "
                    f"target, or None to inherit the forward refinement config; got "
                    f"{self.adjoint_ir_tol!r}."
                ) from exc
            if not finite or tol <= 0.0:
                raise ConfigurationError(
                    "Precision.adjoint_ir_tol must be a finite positive relative-residual "
                    f"target, or None to inherit the forward refinement config; got "
                    f"{self.adjoint_ir_tol!r}."
                )
        for field_name in ("compute_dtype",):
            value = getattr(self, field_name)
            try:
                dtype = jnp.dtype(value)
            except TypeError as exc:
                raise ConfigurationError(
                    f"Precision.{field_name} must be a floating dtype; got {value!r}."
                ) from exc
            if not jnp.issubdtype(dtype, jnp.floating):
                raise ConfigurationError(
                    f"Precision.{field_name} must be a floating dtype; got {value!r}."
                )
        if _is_single(self.compute_dtype):
            for field_name in (
                "solve_mode",
                "forward_sensitivity_solve_mode",
                "reverse_sensitivity_solve_mode",
            ):
                mode = getattr(self, field_name)
                if mode in ("fp32_ir", "adaptive"):
                    raise ConfigurationError(
                        f"Precision.{field_name}={mode!r} refines against a "
                        "double-precision residual, which a float32 compute dtype "
                        "cannot supply. Use solve_mode='fp32' for an end-to-end "
                        "float32 policy, or keep compute_dtype=float64 for a mixed "
                        "policy (for example Precision.fp32_ir())."
                    )
        if self.jacobian_values_dtype is not None:
            try:
                jac = jnp.dtype(self.jacobian_values_dtype)
                ok = bool(jnp.issubdtype(jac, jnp.floating))
            except TypeError:
                ok = False
            if not ok:
                raise ConfigurationError(
                    "Precision.jacobian_values_dtype must be a floating dtype (for "
                    f"example float32) or None; got {self.jacobian_values_dtype!r}."
                )
            if jac.itemsize > jnp.dtype(self.compute_dtype).itemsize:
                raise ConfigurationError(
                    f"Precision.jacobian_values_dtype={jac} is wider than "
                    f"compute_dtype={jnp.dtype(self.compute_dtype)}; Jacobian values "
                    "computed wider than the state would promote the solve. Leave it "
                    "None, or choose a dtype no wider than compute_dtype."
                )
            if self.solve_mode == "adaptive":
                raise ConfigurationError(
                    "the adaptive solve mode manages linear-solve precision "
                    "internally and always assembles Jacobian values in the state "
                    "dtype; jacobian_values_dtype requires a concrete solve_mode "
                    "('fp64', 'fp32', 'fp32_ir')."
                )
        if self.adjoint_compute_dtype is not None:
            try:
                adjoint = jnp.dtype(self.adjoint_compute_dtype)
                forward = jnp.dtype(self.compute_dtype)
            except TypeError as exc:
                raise ConfigurationError(
                    "Precision.adjoint_compute_dtype must be a floating dtype "
                    f"or None; got {self.adjoint_compute_dtype!r}."
                ) from exc
            if not jnp.issubdtype(adjoint, jnp.floating):
                raise ConfigurationError(
                    "Precision.adjoint_compute_dtype must be a floating dtype "
                    f"or None; got {self.adjoint_compute_dtype!r}."
                )
            if adjoint.itemsize < forward.itemsize:
                raise ConfigurationError(
                    "Precision.adjoint_compute_dtype must be at least as wide "
                    "as compute_dtype: the adjoint sweep can only run at or "
                    "above the forward precision (a reduced-precision adjoint "
                    f"transpose is rejected by design). Got adjoint_compute_"
                    f"dtype={adjoint} below compute_dtype={forward}; raise "
                    "adjoint_compute_dtype or leave it None to inherit "
                    "compute_dtype."
                )

    @classmethod
    def fp64(cls) -> "Precision":
        """Return the policy of float64 compute with float64 factors, the default.

        Returns:
            A `Precision` with float64 compute and the `"fp64"` solve mode.
        """
        return cls(
            compute_dtype=jnp.dtype(jnp.float64),
            solve_mode="fp64",
            forward_sensitivity_solve_mode="fp64",
        )

    @classmethod
    def fp32(cls) -> "Precision":
        """Return the policy of float32 compute with float32 factors, end to end.

        States, parameters, factors, and both derivative directions run in float32.

        Returns:
            A `Precision` with float32 compute and the `"fp32"` solve mode.
        """
        return cls(compute_dtype=jnp.dtype(jnp.float32), solve_mode="fp32")

    @classmethod
    def fp32_factor(cls) -> "Precision":
        """Return the policy of float64 compute with unrefined float32 factors.

        Forward sensitivities also use float32 factors, and the reverse sweep solves in float64.

        Returns:
            A `Precision` with float64 compute and the `"fp32"` solve mode.
        """
        return cls(
            compute_dtype=jnp.dtype(jnp.float64),
            solve_mode="fp32",
            forward_sensitivity_solve_mode="fp32",
            reverse_sensitivity_solve_mode="fp64",
        )

    @classmethod
    def fp32_ir(cls) -> "Precision":
        """Return the policy of float64 compute with float32 factors refined against a float64 residual.

        Forward sensitivities refine the same way, and the reverse sweep solves in float64.

        Returns:
            A `Precision` with float64 compute and the `"fp32_ir"` solve mode.
        """
        return cls(
            compute_dtype=jnp.dtype(jnp.float64),
            solve_mode="fp32_ir",
            forward_sensitivity_solve_mode="fp32_ir",
            reverse_sensitivity_solve_mode="fp64",
        )


def _is_single(dtype: Any) -> bool:
    """Whether ``dtype`` is a floating dtype of at most four bytes."""
    try:
        resolved = np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    return bool(np.issubdtype(resolved, np.floating)) and resolved.itemsize <= 4


_DEFAULT_PRECISION = Precision()


def problem_precision(problem: Any) -> Precision:
    """The precision a problem declares, or the default all-fp64 policy.

    Reads ``problem.precision`` on a problem, a solver view, or any authored
    object that carries one; objects without the attribute use the default.
    """
    value = getattr(problem, "precision", None)
    return value if isinstance(value, Precision) else _DEFAULT_PRECISION


def resolve_precision(
    problem: Any,
    solver_precision: Precision | None = None,
    solver_jacobian_values_dtype: Any = None,
) -> Precision:
    """The one precision a solve runs under.

    The problem is the source of precision. A solver whose ``precision`` is
    left at the default inherits the problem's policy, including its solve
    mode and sensitivity overrides. A solver argument is still honoured when
    the problem declares nothing, or when both agree. A solver's
    ``jacobian_values_dtype`` argument folds into its policy first.

    Args:
        problem: The problem (or its solver view) being solved.
        solver_precision: The solver's ``precision`` field, or ``None``.
        solver_jacobian_values_dtype: The solver's ``jacobian_values_dtype``
            field, or ``None``.

    Returns:
        The resolved [Precision][diff_soup._precision.Precision].

    Raises:
        ConfigurationError: If the problem and the solver declare different
            policies.
    """
    declared = problem_precision(problem)
    solver = _DEFAULT_PRECISION if solver_precision is None else solver_precision
    if solver_jacobian_values_dtype is not None:
        requested = np.dtype(solver_jacobian_values_dtype)
        if solver.jacobian_values_dtype is not None and solver.jacobian_values_dtype != requested:
            raise ConfigurationError(
                f"the solver sets jacobian_values_dtype={requested} but its precision "
                f"sets jacobian_values_dtype={solver.jacobian_values_dtype}; set it "
                "once, on the problem's Precision."
            )
        solver = _replace(solver, jacobian_values_dtype=requested)
    if solver == _DEFAULT_PRECISION:
        return declared
    if declared == _DEFAULT_PRECISION or declared == solver:
        return solver
    raise ConfigurationError(
        "precision is set on both the problem and the solver, and they disagree: "
        f"the problem declares {describe(declared)}, the solver {describe(solver)}. Precision "
        "belongs on the problem: build it with spec.build(precision=...), "
        "ds.lower(eqset, precision=...), MassProblem(..., precision=...) or "
        "ImplicitProblem(..., precision=...), and leave the solver's precision "
        "and jacobian_values_dtype unset."
    )


def describe(precision: Precision) -> str:
    """A one-line description of a policy: its non-default fields."""
    default = _DEFAULT_PRECISION
    parts = []
    for name in (
        "compute_dtype",
        "solve_mode",
        "adjoint_compute_dtype",
        "adjoint_ir_tol",
        "forward_sensitivity_solve_mode",
        "reverse_sensitivity_solve_mode",
        "jacobian_values_dtype",
    ):
        value = getattr(precision, name)
        if value != getattr(default, name):
            value = str(value) if isinstance(value, np.dtype) else repr(value)
            parts.append(f"{name}={value}")
    return f"Precision({', '.join(parts)})"


def _replace(precision: Precision, **changes: Any) -> Precision:
    """A copy of ``precision`` with fields changed, validated like a new policy."""
    fields = {
        name: getattr(precision, name)
        for name in (
            "compute_dtype",
            "solve_mode",
            "adjoint_compute_dtype",
            "adjoint_ir_tol",
            "forward_sensitivity_solve_mode",
            "reverse_sensitivity_solve_mode",
            "jacobian_values_dtype",
        )
    }
    fields.update(changes)
    return Precision(**fields)


def compute_dtype_of(precision: Precision) -> np.dtype:
    """The concrete array dtype of ``precision.compute_dtype`` in this process.

    With ``jax_enable_x64`` disabled a ``float64`` policy cannot be honoured
    and resolves to JAX's canonical ``float32``.
    """
    return np.dtype(jax.dtypes.canonicalize_dtype(precision.compute_dtype))


def cast_to_compute_dtype(tree: Any, precision: Precision, *, what: str = "input") -> Any:
    """Cast every real floating (and integer array) leaf of ``tree`` to the compute dtype.

    Boolean leaves and Python scalars that are not arrays pass through; a
    complex leaf is rejected, since dropping its imaginary part would be
    silent. Integer *arrays* are cast only when ``what`` names a state (the
    initial state of a solve is numeric data, never an index).

    Args:
        tree: A pytree of arrays (or ``None``).
        precision: The resolved policy.
        what: Description used in the error message.

    Returns:
        ``tree`` with its numeric leaves at the compute dtype.
    """
    if tree is None:
        return None
    target = compute_dtype_of(precision)
    state_like = what in ("initial state", "bordered globals")
    if state_like and isinstance(tree, (int, float, list, tuple)):
        tree = jnp.asarray(tree)

    def cast(leaf: Any) -> Any:
        if not hasattr(leaf, "dtype"):
            return leaf
        dtype = np.dtype(leaf.dtype)
        if np.issubdtype(dtype, np.complexfloating):
            raise ConfigurationError(
                f"the {what} has complex dtype {dtype}; diff_soup solves real "
                "systems. Pass real values."
            )
        if np.issubdtype(dtype, np.floating) or (state_like and np.issubdtype(dtype, np.integer)):
            return leaf if dtype == target else jnp.asarray(leaf).astype(target)
        return leaf

    return jax.tree_util.tree_map(cast, tree)


def require_double_precision(problem: Any, solver_name: str) -> None:
    """Reject a problem whose precision a double-only solver cannot honour.

    Args:
        problem: The problem to be solved.
        solver_name: The solver's name for the message.

    Raises:
        ConfigurationError: If the problem declares a compute dtype other
            than ``float64`` or a reduced linear solve mode.
    """
    declared = problem_precision(problem)
    if np.dtype(declared.compute_dtype) != np.dtype(np.float64) or declared.solve_mode != "fp64":
        raise ConfigurationError(
            f"{solver_name} integrates in double precision only, but the problem "
            f"declares {describe(declared)}; running it would silently change the "
            "problem's precision. Use JNDF, JBDF or JIDA for a reduced-precision "
            "problem, or build the problem with the default Precision."
        )


def cast_floating_leaves(tree: Any, dtype: Any) -> Any:
    """Cast every real floating array leaf of ``tree`` to ``dtype``.

    Array fields of plain dataclass leaves are cast too. Non-array leaves,
    integer and boolean arrays pass through unchanged.
    """
    target = np.dtype(dtype)

    def cast(leaf: Any) -> Any:
        if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            # A plain (non-pytree) dataclass, such as a compiled prior block,
            # is a single leaf; its array fields are cast one by one.
            changes = {
                field.name: cast_floating_leaves(getattr(leaf, field.name), target)
                for field in dataclasses.fields(leaf)
                if field.init
            }
            try:
                return dataclasses.replace(leaf, **changes)
            except (TypeError, ValueError):
                return leaf
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is None or not np.issubdtype(np.dtype(leaf_dtype), np.floating):
            return leaf
        return leaf if np.dtype(leaf_dtype) == target else jnp.asarray(leaf).astype(target)

    return jax.tree_util.tree_map(cast, tree)
