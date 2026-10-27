"""``lower()`` — the EquationSet → LoweredProblem seam.

Three responsibilities:

1. **Structure cache and value copies.** The analysis of an EquationSet's
   structure (equations, domains, boundary conditions, declared names and
   shapes, scales, layout and precision) runs once per structure; the bounded
   content-hash cache keeps it with a private template that is never handed
   out.  Every call returns a [LoweredProblem][] carrying the declared values
   of its own EquationSet: equal structure and equal values return the same
   object, other values another object, and a returned problem never changes
   afterwards.

2. **Layout / Precision recording.** Both are recorded on the
   [LoweredProblem][] (the single source of truth — integrator configs do
   not carry them) and folded into the cache key so that a changed
   Layout/Precision triggers a recompile.  Default values reproduce prior jaxprs
   bit-identically.

3. **Boundary-cast law.** Closure outputs are cast up to
   ``precision.compute_dtype`` where they enter the residual IR.  At the default
   fp64 this is a no-op; for a mixed-precision config it prevents forward-pass
   precision loss from corrupting the factor dtype.

The L1-dependent host analysis (operator-bank residual / Jacobian kernels,
boundary geometry, FEM sparsity stencils) is injected via a ``LoweringOps``
provider — see [_OPS_PROVIDER][] and [set_lowering_ops][].  The
``problem`` layer may not import ``domains`` directly (the import-direction law,
AST-enforced).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import LoweringError
from diff_soup._layout import Layout
from diff_soup._precision import Precision
from diff_soup.problem import _registry
from diff_soup.problem._cache import compose_content_key

from .content_hash import content_hash
from .equation_set import EquationSet
from .lowered_problem import LoweredProblem

# ---------------------------------------------------------------------------
# Operator-provider injection seam.
# ---------------------------------------------------------------------------
#
# The residual / Jacobian / sparsity / mass-diagonal kernels for an EquationSet
# are built from the ``domains.operators`` bank. The ``problem`` layer cannot
# import ``domains`` directly (the import-direction law); a layer that may import
# ``domains`` (the public surface or a domains-side registration) supplies a
# provider here.  The provider protocol is the single function:
#
#     analyze(eqset, *, precision) -> _LoweringAnalysis
#
# which performs the host analysis pass (layout, bcs, sparsity, mass_diagonal,
# residual_fn, jac_blocks_fn, observe_fn, controls_fn), applying the
# boundary-cast law at closure entry.  ``set_lowering_ops`` registers it.  The
# only mutable module state is the registered provider, set once at import of
# the wiring layer.
_OPS_PROVIDER: Optional[Callable[..., "_LoweringAnalysis"]] = None


class _LoweringAnalysis:
    """Host-side analysis artefacts for an eqset (the provider's output).

    Cached by content hash and reused across calls, but holds no traced closure
    values.  The runtime IR-node tuples (equations, value_fns, constraints,
    observations, controls) are read from the eqset on every ``lower()`` call so
    traced ``eqx.Module`` parameters reach the AD boundary: static kernels take
    the runtime IR-node tuples as arguments.
    """

    __slots__ = (
        "layout_entries",
        "global_layout_entries",
        "topo",
        "mass_diagonal",
        "constraint_dim",
        "state_size",
        "alg_indices",
        "bc_dof_indices_static",
        "dirichlet_value_fns",
        "flux_bc_callables",
        "robin_bc_callables",
        "traction_bc_callables",
        "initial_lambda_si",
        "residual_fn",
        "jac_blocks_fn",
        "observe_fn",
        "controls_fn",
    )

    def __init__(self, **kw: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


def set_lowering_ops(provider: Optional[Callable[..., _LoweringAnalysis]]) -> None:
    """Register the ``domains``-aware analysis provider for ``lower()`` (the injection seam).

    Called once by the wiring layer — the layer that may import
    ``domains.operators``.  The provider performs the host analysis pass and
    returns a
    [_LoweringAnalysis][diff_soup.problem._ir.lower._LoweringAnalysis].  Pass
    ``None`` to clear the seam (the
    test / session-reset hook).

    Args:
        provider: ``analyze(eqset, *, precision) -> _LoweringAnalysis``, or
            ``None`` to clear the registered provider.
    """
    global _OPS_PROVIDER
    _OPS_PROVIDER = provider


def _build_analysis(eqset: EquationSet, *, precision: Precision) -> _LoweringAnalysis:
    """Run the host analysis pass via the injected provider.

    The residual / Jacobian / sparsity / mass-diagonal construction requires
    ``domains.operators``, which the ``problem`` layer cannot import directly.
    When no provider is registered, raises a [LoweringError][]
    directing the caller to register one via [set_lowering_ops][].

    Args:
        eqset: The EquationSet to analyse.
        precision: The dtype policy (its ``compute_dtype`` drives the
            boundary-cast law at closure entry).

    Returns:
        The [_LoweringAnalysis][].

    Raises:
        LoweringError: When no analysis provider is registered.
    """
    if _OPS_PROVIDER is None:
        raise LoweringError(
            "lower(): no operator-bank analysis provider is registered. "
            "The residual / Jacobian / sparsity / mass-diagonal kernels are "
            "built from diff_soup.domains.operators, which the problem layer "
            "may not import directly (import-direction law, AST-enforced). "
            "A layer that may import domains.operators must register a provider "
            "via diff_soup.problem._ir.lower.set_lowering_ops(provider) before "
            "calling lower()."
        )
    return _OPS_PROVIDER(eqset, precision=precision)


def _assemble_lowered_problem(
    analysis: _LoweringAnalysis,
    eqset: EquationSet,
    *,
    layout: Layout,
    precision: Precision,
    structure_key: Any,
    boundary: Optional[dict] = None,
    initial_lambda_si: Any = None,
) -> LoweredProblem:
    """Build a fresh [LoweredProblem][] from analysis + current eqset closures.

    A fresh object is built on every call — no ``object.__setattr__`` refresh.
    The runtime IR-node tuples (``_equations`` etc.) live as non-static fields so
    ``eqx.partition`` recurses into any ``eqx.Module`` closure and surfaces its
    parameter arrays for AD; the kernels are static and capture nothing traced.
    ``boundary`` replaces the analysis's boundary-callable tuples by slot name,
    and ``initial_lambda_si`` its initial bordered values, for an eqset other
    than the one the analysis was built from.
    """
    boundary = boundary or {}
    initial = analysis.initial_lambda_si if initial_lambda_si is None else initial_lambda_si
    all_equations = tuple(eq for comp in eqset.components for eq in comp.equations)
    return LoweredProblem(
        _state_size=analysis.state_size,
        _constraint_dim=analysis.constraint_dim,
        _mass_diagonal=analysis.mass_diagonal,
        _topo=analysis.topo,
        _equations=all_equations,
        _dirichlet_value_fns=boundary.get("_dirichlet_value_fns", analysis.dirichlet_value_fns)
        or (),
        _flux_bc_callables=boundary.get("_flux_bc_callables", analysis.flux_bc_callables) or (),
        _robin_bc_callables=analysis.robin_bc_callables or (),
        _traction_bc_callables=analysis.traction_bc_callables or (),
        _constraints=tuple(eqset.constraints),
        _observations=tuple(eqset.observations),
        _controls=tuple(eqset.controls),
        _couplings=tuple(eqset.couplings),
        _physics_residual_fn=analysis.residual_fn,
        _physics_jac_blocks_fn=analysis.jac_blocks_fn,
        _initial_lambda=initial,
        _bc_dof_indices_static=analysis.bc_dof_indices_static or (),
        _observe_fn=analysis.observe_fn,
        _controls_fn=analysis.controls_fn,
        _scales=eqset.scales,
        _layout_entries=tuple(analysis.layout_entries or ()),
        _global_layout_entries=tuple(analysis.global_layout_entries or ()),
        _initial_lambda_si=initial,
        _alg_indices=tuple(analysis.alg_indices or ()),
        _families=tuple(eqset.families),
        layout=layout,
        precision=precision,
        quad_fields=getattr(eqset, "quad_fields", frozenset()),
        _structure_key=structure_key,
    )


def lower(
    eqset: Any,
    *,
    layout: Optional[Layout] = None,
    precision: Optional[Precision] = None,
    **options: Any,
) -> Any:
    """Compile an `EquationSet` into a problem, or a `Study` into a lowered study.

    Equation sets equal in structure, values, layout and precision share one
    lowered problem.

    Args:
        eqset: An [`EquationSet`][diff_soup.problem.EquationSet] or a [`Study`][diff_soup.inference.Study].
        layout: Device placement, a [`Layout`][diff_soup.Layout], or `None` for the default.
        precision: [`Precision`][diff_soup.Precision] of an equation set, or `None` for the default, refused for a study.
        **options: Keywords for a study, such as `chunk`, refused for an equation set.

    Returns:
        A [`LoweredProblem`][diff_soup.problem.LoweredProblem], or a [`LoweredStudy`][diff_soup.inference.LoweredStudy] for a study.

    Raises:
        LoweringError: If `eqset` is neither an equation set nor a study.
        TypeError: If `precision` is given for a study, or `options` for an equation set.
    """
    if not isinstance(eqset, EquationSet):
        if getattr(type(eqset), "_ds_lower", None) is None:
            raise LoweringError(
                f"lower() takes an EquationSet or a Study; got {type(eqset).__name__}. "
                "Build a spec via ds.problem(...) and call .to_equation_set(), construct an "
                "EquationSet directly, or declare the experiments to fit in a "
                "diff_soup.inference.Study."
            )
        if precision is not None:
            raise TypeError(
                "lower() takes precision= for an EquationSet only; the experiments of a "
                f"{type(eqset).__name__} keep the precision of their own models. Remove "
                "precision=."
            )
        return eqset._ds_lower(layout=layout, **options)
    if options:
        raise TypeError(
            f"lower() of an EquationSet takes layout= and precision= only; got "
            f"{sorted(options)}, keywords that only a study's lowering takes."
        )
    return lower_declared(eqset, layout=layout, precision=precision)


def lower_declared(
    eqset: EquationSet,
    *,
    layout: Optional[Layout] = None,
    precision: Optional[Precision] = None,
    bounds: Optional[tuple] = None,
    declarations: Optional[tuple] = None,
    geometry_scales: Optional[tuple] = None,
) -> LoweredProblem:
    """Lower ``eqset`` to a problem carrying its values, bounds and declarations.

    [lower][] with the facts a Recipe build adds: the declared bounds take part
    in the returned problem's identity (two builds equal in values but not in
    bounds return two problems), and the declarations and grid-scale names ride
    on it for the inference layer.  ``ProblemSpec.build`` lowers through here.

    Args:
        eqset: The EquationSet to lower.
        layout: Device-placement policy (default = single device).
        precision: Dtype policy (default = fp64).
        bounds: ``(handle, (lower, upper), learnable)`` per bounded declared
            parameter; ``None`` attaches none.
        declarations: The declaration handles of the build, matched by name.
        geometry_scales: Names of the declared parameters that scale a grid.

    Returns:
        The [LoweredProblem][] holding this build's values; the same object
        for an equal structure, equal values and equal bounds.

    Raises:
        LoweringError: As [lower][].
    """
    layout = Layout() if layout is None else layout
    precision = Precision() if precision is None else precision
    # Content-hash cache key folds the pillar policies: a changed Layout /
    # Precision / Scales is a deliberate recompile, never a silent reuse.
    # Default constructions (Layout() / Precision() / Scales()) hash identically
    # across calls, so a default build always re-hits the cache.
    structure_hash = content_hash(eqset)
    key = compose_content_key(
        structure_hash, layout=layout, precision=precision, scales=eqset.scales
    )
    built: list = []

    def _build() -> tuple:
        analysis = _build_analysis(eqset, precision=precision)
        template = _assemble_lowered_problem(
            analysis,
            eqset,
            layout=layout,
            precision=precision,
            structure_key=key,
        )
        built.append(template)
        return template, analysis

    # The cached template is private: it is never returned and never written to.
    template, analysis = _registry.lowering_cache.get_or_build(key, id(eqset), _build)
    if built:
        theta = eqx.partition(template, eqx.is_inexact_array)[0]
    else:
        current = _assemble_lowered_problem(
            analysis,
            eqset,
            layout=layout,
            precision=precision,
            structure_key=key,
            boundary=_rewrapped_boundary_slots(analysis, eqset),
            initial_lambda_si=_initial_bordered_values(eqset, analysis.initial_lambda_si),
        )
        theta = _values_on_template(current, template)
    return _value_copy(
        template,
        theta,
        key,
        bounds=bounds,
        declarations=declarations,
        geometry_scales=geometry_scales,
    )


def with_theta(problem: LoweredProblem, theta: Any, *, bounds: Optional[tuple] = None):
    """Return a problem equal to ``problem`` in structure, holding the values ``theta``.

    ``theta`` is the inexact-leaf partition of ``problem``
    (``eqx.partition(problem, eqx.is_inexact_array)[0]``, or the ``inner`` tree
    of ``default_theta``) with some leaves replaced.  The result is a problem of
    its own: the same object for an equal ``theta`` and equal bounds, and
    ``problem`` is left as it was.  The declared bounds, declarations and
    grid-scale names of ``problem`` ride along (``bounds`` replaces the
    bounds), and so do its registered parameter handles.  A pytree rewrite of
    the result (``eqx.tree_at``) drops those attributes, as it does for any
    lowered problem.

    Args:
        problem: A problem returned by [lower][] or ``ProblemSpec.build``.
        theta: Its inexact-leaf partition, with the new values.
        bounds: The declared bounds of the result; ``None`` keeps
            ``problem``'s.

    Returns:
        The [LoweredProblem][] holding ``theta``.

    Raises:
        TypeError: If ``problem`` carries no structure key (it was not
            returned by [lower][]).
    """
    getter = getattr(problem, "structure_key", None)
    key = getter() if callable(getter) else None
    if key is None:
        raise TypeError(
            "with_theta() takes a problem returned by lower() or ProblemSpec.build(); got "
            f"{type(problem).__name__} without a structure key."
        )
    if bounds is None:
        bounds = getattr(problem, "_declared_param_bounds", None)
    copy = _value_copy(
        problem,
        theta,
        key,
        bounds=bounds,
        declarations=getattr(problem, "_recipe_declarations", None),
        geometry_scales=getattr(problem, "_geometry_scales", None),
    )
    for namespace, handles in _registry.parameter_handles(problem).items():
        _registry.register_parameter_handles(copy, namespace, handles)
    return copy


def _values_on_template(current: LoweredProblem, template: LoweredProblem) -> Any:
    """The inexact leaves of ``current``, placed in the tree of ``template``'s.

    Both problems share one content key, so their inexact leaves line up path
    by path; ``current``'s static parts (its own closures, equal to the
    template's by the key) are left behind.

    Raises:
        LoweringError: If a leaf's path, dtype or shape differs from the
            template's, naming the first such path.
    """
    current_leaves = jax.tree_util.tree_flatten_with_path(
        eqx.partition(current, eqx.is_inexact_array)[0]
    )[0]
    template_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        eqx.partition(template, eqx.is_inexact_array)[0]
    )
    for index in range(max(len(current_leaves), len(template_leaves))):
        here = current_leaves[index] if index < len(current_leaves) else None
        there = template_leaves[index] if index < len(template_leaves) else None
        if (
            here is None
            or there is None
            or here[0] != there[0]
            or _leaf_signature(here[1]) != _leaf_signature(there[1])
        ):
            raise LoweringError(
                "lower(): the declared values of this EquationSet do not line up with the "
                f"lowered structure of its content key: value leaf {index} is "
                f"{_leaf_description(here)} here and {_leaf_description(there)} in the "
                "lowered structure. Equal structure needs the same value leaves, by path, "
                "dtype and shape; lower the EquationSet built from an unmodified spec."
            )
    return jax.tree_util.tree_unflatten(treedef, [leaf for _, leaf in current_leaves])


def _leaf_signature(leaf: Any) -> tuple:
    """A value leaf's ``(dtype, shape)``."""
    return (str(leaf.dtype), tuple(int(size) for size in leaf.shape))


def _leaf_description(entry: Any) -> str:
    """A ``(path, leaf)`` pair as text for an error message."""
    if entry is None:
        return "absent"
    path, leaf = entry
    dtype, shape = _leaf_signature(leaf)
    return f"{jax.tree_util.keystr(path)} ({dtype}, shape {shape})"


def _value_copy(
    source: LoweredProblem,
    theta: Any,
    key: Any,
    *,
    bounds: Optional[tuple],
    declarations: Optional[tuple],
    geometry_scales: Optional[tuple],
) -> LoweredProblem:
    """The problem of ``source``'s structure holding ``theta``, memoized by value.

    The copy is ``eqx.combine(theta, static)`` with ``source``'s static
    partition: new tuple, dict and module nodes down to every leaf, so it shares
    only immutable things (arrays, functions, static fields) with ``source`` and
    with other copies.  Equal values and bounds return the memoized object,
    whose handle metadata (the bounds' and declarations' handles, matched by
    name) becomes the caller's; a traced value (lowering inside a
    transformation) has no bytes to digest, so its copy is not memoized.
    """
    digest = _value_digest(theta, bounds)
    made: list = []

    def _make() -> LoweredProblem:
        static = eqx.partition(source, eqx.is_inexact_array)[1]
        copy = eqx.combine(theta, static)
        # Not pytree fields: lowering metadata a fit reads (bounds) and the
        # inference layer resolves names from (declarations, grid scales).
        if bounds is not None:
            object.__setattr__(copy, "_declared_param_bounds", tuple(bounds))
        if declarations is not None:
            object.__setattr__(copy, "_recipe_declarations", tuple(declarations))
        if geometry_scales is not None:
            object.__setattr__(copy, "_geometry_scales", tuple(geometry_scales))
        made.append(copy)
        return copy

    if digest is None:
        return _make()
    copy = _registry.value_copy_cache.get_or_build((key, digest), None, _make)
    if not made:
        # An equal problem built again, by this spec or another one equal in
        # structure, values and bounds: its values stay; the handles it carries
        # (matched by name) become this build's, so the handles of a discarded
        # spec are not kept alive by the shared problem.
        if bounds is not None:
            object.__setattr__(copy, "_declared_param_bounds", tuple(bounds))
        if declarations is not None:
            object.__setattr__(copy, "_recipe_declarations", tuple(declarations))
        if geometry_scales is not None:
            object.__setattr__(copy, "_geometry_scales", tuple(geometry_scales))
    return copy


def _value_digest(theta: Any, bounds: Optional[tuple]) -> Optional[str]:
    """SHA-256 of the value leaves (dtype, shape and bytes, in order) and the bounds.

    ``None`` when a leaf is traced: a value inside a transformation has no bytes.
    """
    h = hashlib.sha256()
    # A leaf object several closures share is read from the device once.
    read: dict = {}
    try:
        for leaf in jax.tree_util.tree_leaves(theta):
            if id(leaf) not in read:
                read[id(leaf)] = np.asarray(leaf)
            _digest_value(h, read[id(leaf)])
        if bounds is None:
            h.update(b"|bounds:none")
        else:
            for handle, limits, learnable in bounds:
                h.update(f"|bound:{getattr(handle, 'name', handle)}:{bool(learnable)}".encode())
                for limit in tuple(limits):
                    _digest_value(h, limit)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None
    return h.hexdigest()


def _digest_value(h: Any, value: Any) -> None:
    """Feed one value's dtype, shape and bytes (or its absence) to the hash ``h``."""
    if value is None:
        h.update(b"|none")
        return
    array = np.asarray(value)
    if array.dtype.hasobject:
        h.update(f"|object:{value!r}".encode())
        return
    h.update(f"|{array.dtype.str}:{array.shape}:".encode())
    h.update(np.ascontiguousarray(array).tobytes())


def _initial_bordered_values(eqset: EquationSet, reference: Any) -> Any:
    """This eqset's initial bordered-global vector, in the analysis's layout.

    One slice per closing row: its global's ``init``, broadcast to the row's
    width, in the dtype of ``reference`` (the analysis's vector, built the same
    way from the first eqset of the structure); ``reference`` itself when it is
    absent or empty.
    """
    if reference is None or int(np.size(reference)) == 0 or not eqset.constraints:
        return reference
    parts = []
    for c in eqset.constraints:
        value = jnp.asarray(c.closes.init).reshape(-1)
        if int(value.size) != int(c.dim):
            value = jnp.broadcast_to(value.reshape(()), (c.dim,))
        parts.append(value.astype(reference.dtype))
    return jnp.concatenate(parts)


def _inexact_signature(value) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Dtype/shape fingerprint of every inexact array leaf, in traversal order."""
    return tuple(
        (str(leaf.dtype), tuple(int(size) for size in leaf.shape))
        for leaf in jax.tree_util.tree_leaves(value)
        if eqx.is_inexact_array(leaf)
    )


def _rewrapped_boundary_callable(wrapper: Any, current_fn: Any, *, label: str, mismatch) -> Any:
    """Return ``wrapper`` carrying ``current_fn``'s inexact leaves.

    A boundary wrapper adds only static configuration (unit scaling, the clock
    reference) around the callable a component declared, so the wrapper's
    inexact leaves are exactly that callable's, in the same order.  Swapping the
    leaves therefore installs the current values while keeping the wrapper's own
    configuration.
    """
    cached_dynamic, cached_static = eqx.partition(wrapper, eqx.is_inexact_array)
    cached_leaves, cached_treedef = jax.tree_util.tree_flatten(cached_dynamic)
    if not cached_leaves:
        return wrapper
    current_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(current_fn) if eqx.is_inexact_array(leaf)
    ]
    cached_signature = _inexact_signature(wrapper)
    current_signature = _inexact_signature(current_fn)
    if cached_signature != current_signature:
        mismatch(f"{label} inexact leaf signature {cached_signature!r} != {current_signature!r}")
    if all(old is new for old, new in zip(cached_leaves, current_leaves)):
        return wrapper
    return eqx.combine(
        jax.tree_util.tree_unflatten(cached_treedef, current_leaves),
        cached_static,
    )


def _rewrapped_boundary_slots(analysis: _LoweringAnalysis, eqset: EquationSet) -> dict:
    """The analysis's boundary wrappers around this eqset's boundary callables.

    Dirichlet value functions and boundary flux callables are wrapped once, when
    the boundary geometry is resolved, around the runtime callables their
    component carries.  The content hash covers a boundary callable's static
    spine but not its values, so another eqset of the same structure re-wraps the
    analysis's wrappers around its own callables' leaves (new objects; the
    analysis's are left as they are).

    Raises:
        LoweringError: If a slot's count or a callable's leaf signature differs
            from the analysis's.
    """

    def mismatch(detail: str) -> None:
        raise LoweringError(
            f"lower(): the boundary callables of this EquationSet do not line up with the "
            f"lowered structure of its content key ({detail}). Equal structure needs the "
            "same boundary callables, by count, dtype and shape of their values; lower the "
            "EquationSet built from an unmodified spec."
        )

    slots = {}
    for slot, attribute, source_field, label in (
        (
            "_dirichlet_value_fns",
            "dirichlet_value_fns",
            "dirichlet_value_fns",
            "Dirichlet value function",
        ),
        ("_flux_bc_callables", "flux_bc_callables", "flux_callables", "boundary flux callable"),
    ):
        wrappers = tuple(getattr(analysis, attribute, None) or ())
        if not wrappers:
            continue
        current = tuple(
            fn
            for component in eqset.components
            for fn in (getattr(component, source_field, ()) or ())
        )
        if len(wrappers) != len(current):
            mismatch(f"{label} count {len(wrappers)} != current count {len(current)}")
        slots[slot] = tuple(
            _rewrapped_boundary_callable(wrapper, fn, label=f"{label} {index}", mismatch=mismatch)
            for index, (wrapper, fn) in enumerate(zip(wrappers, current))
        )
    return slots


def _apply_boundary_cast(value: Any, compute_dtype: Any) -> Any:
    """Cast a closure output up to ``compute_dtype`` at the residual IR boundary.

    A closure's forward precision enters the residual noise floor and therefore
    Newton / controller convergence.  Casting its output up to the solver's
    ``compute_dtype`` prevents precision loss from corrupting the factor dtype in
    mixed-precision configurations.  At the default fp64 this is a no-op that
    does not perturb the jaxpr — a same-dtype ``astype`` is elided by XLA.

    Exposed for the injected analysis provider to apply at every per-equation /
    BC / constraint closure boundary where closures enter the residual IR.

    Args:
        value: A closure output (array-like / pytree leaf).
        compute_dtype: ``precision.compute_dtype`` — the residual/Jacobian
            assembly dtype.

    Returns:
        ``value`` cast up to ``compute_dtype`` (a no-op when already that dtype).
    """
    arr = jnp.asarray(value)
    if arr.dtype == compute_dtype:
        return arr  # no-op: same dtype, jaxpr-identical (the fp64-default path).
    # Cast UP only (fp32/bf16 closure -> fp64 solver). Never down-cast a solver
    # array into a closure dtype.
    return arr.astype(compute_dtype)
