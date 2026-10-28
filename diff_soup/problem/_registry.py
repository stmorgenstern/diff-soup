"""The lowering's cache registry: the registry's three caches and the parameter handles.

The registry's three caches are
[BoundedContentCache][diff_soup.problem._cache.BoundedContentCache] instances
owned here. The other instances live next to the code that builds their
values: the solver-view cache in ``problem/implicit/solver_problem.py``, the
solver problems built from a lowering in ``problem/implicit/solver_lowering.py``
and the two coloring caches in ``problem/_ad/sparse_jac.py``.

The registry's three caches:

* ``lowering_cache`` — the ``lower()`` structure cache: per content key (the
  structure of an EquationSet with its layout, precision and scales), the
  analysis and a private template
  [LoweredProblem][diff_soup.problem._ir.LoweredProblem] that is never handed
  out.
* ``value_copy_cache`` — the lowered problems ``lower()`` hands out, one per
  content key and digest of declared values, so an equal value set returns the
  same object.
* ``analysis_cache`` — symbolic-analysis amortisation (reserved for the
  ``analyze`` reuse path).

All are bounded LRU caches; capacities are module constants so eviction
telemetry has a stable denominator.
[reset_registry][diff_soup.problem._registry.reset_registry] clears the registry's
three caches and the handle metadata associated with cached problems; it is the recommended
test / session-reset hook.
"""

from __future__ import annotations

import weakref
from types import MappingProxyType
from typing import Any, Mapping

import equinox as eqx
import jax

from diff_soup.problem._cache import BoundedContentCache

# Bounded capacities. 128 spec variants is generous for a single
# inversion session (one problem family + parameter sweeps); LRU eviction past
# that is telemetered, never silent.
_LOWERING_CACHE_CAPACITY = 128
_VALUE_COPY_CACHE_CAPACITY = 128
_ANALYSIS_CACHE_CAPACITY = 128

# The registry's three caches.
lowering_cache: BoundedContentCache = BoundedContentCache(capacity=_LOWERING_CACHE_CAPACITY)
value_copy_cache: BoundedContentCache = BoundedContentCache(capacity=_VALUE_COPY_CACHE_CAPACITY)
analysis_cache: BoundedContentCache = BoundedContentCache(capacity=_ANALYSIS_CACHE_CAPACITY)
_HandleRefs = tuple[weakref.ReferenceType, ...]
_parameter_handles_by_problem: dict[int, tuple[weakref.ReferenceType, dict[str, _HandleRefs]]] = {}
# Secondary index from a lowered problem's value-independent structure key to
# the identity of the registered problem carrying that structure. It lets a
# structural copy of a registered problem (an ``eqx.tree_at`` rewrite, say) read
# the same declarations; entries are removed with the problem they point at.
_parameter_handle_problems_by_structure: dict[Any, int] = {}


def _structure_key(problem: Any) -> Any:
    """Return a lowered problem's value-independent structure key, or ``None``."""
    getter = getattr(problem, "structure_key", None)
    if not callable(getter):
        return None
    try:
        key = getter()
        hash(key)
    except Exception:
        return None
    return key


def _live_handles(namespaces: Mapping[str, _HandleRefs]) -> dict[str, tuple]:
    """Resolve stored handle references, dropping scopes with none left alive."""
    live: dict[str, tuple] = {}
    for namespace, refs in namespaces.items():
        handles = tuple(handle for handle in (ref() for ref in refs) if handle is not None)
        if handles:
            live[namespace] = handles
    return live


def register_parameter_handles(problem: Any, namespace: str, handles: tuple) -> None:
    """Associate immutable Recipe handles with an identity-stable lowered problem.

    The mapping is cache metadata, not a field on the frozen numerical model,
    so it cannot alter the model's pytree, structural signature, or compiled
    solver view. A weak reference removes metadata after an evicted problem
    is no longer live.

    Recipes equal in structure and declared values share one lowered problem,
    so several builders may register against the same instance. Their handles
    are held weakly and a scope whose handles have all been collected is
    dropped, so rebuilding one Recipe in a loop leaves behind only the scopes
    whose handles a caller still holds.
    """
    identity = id(problem)
    current = _parameter_handles_by_problem.get(identity)
    namespaces = dict(current[1]) if current is not None and current[0]() is problem else {}
    namespaces[str(namespace)] = tuple(weakref.ref(handle) for handle in handles)
    namespaces = {
        name: refs for name, refs in namespaces.items() if any(ref() is not None for ref in refs)
    }
    structure = _structure_key(problem)

    def _discard(dead_ref, *, key=identity, structure=structure):
        entry = _parameter_handles_by_problem.get(key)
        if entry is not None and entry[0] is dead_ref:
            _parameter_handles_by_problem.pop(key, None)
        if _parameter_handle_problems_by_structure.get(structure) == key:
            _parameter_handle_problems_by_structure.pop(structure, None)

    problem_ref = weakref.ref(problem, _discard)
    _parameter_handles_by_problem[identity] = (problem_ref, namespaces)
    if structure is not None:
        _parameter_handle_problems_by_structure[structure] = identity


def parameter_handles(problem: Any) -> Mapping[str, tuple]:
    """Return immutable Recipe-handle metadata registered for ``problem``.

    A problem that is itself unregistered but structurally identical to a
    registered one — a pytree rewrite of it, for instance — reads that
    problem's declarations, so a handle keeps resolving across such a copy.
    """
    entry = _parameter_handles_by_problem.get(id(problem))
    if entry is None or entry[0]() is not problem:
        entry = _structural_entry(problem)
    if entry is None:
        return MappingProxyType({})
    return MappingProxyType(_live_handles(entry[1]))


def _structural_entry(problem: Any):
    """Return the registry entry of a registered problem with the same structure."""
    structure = _structure_key(problem)
    if structure is None:
        return None
    identity = _parameter_handle_problems_by_structure.get(structure)
    if identity is None:
        return None
    entry = _parameter_handles_by_problem.get(identity)
    if entry is None or entry[0]() is None:
        return None
    return entry


def handle_declarations(problem: Any, handle: Any) -> tuple:
    """Return the declarations registered for ``handle``'s declaring scope.

    An empty tuple means ``problem`` carries no registered declarations at all,
    which leaves the caller free to resolve the handle by name.

    Args:
        problem: A lowered problem.
        handle: A Recipe handle returned by ``spec.param(...)`` or
            ``spec.function(...)``.

    Returns:
        Every declaration registered alongside ``handle``, or ``()`` when
        ``problem`` has no registered declarations.

    Raises:
        ValueError: When ``problem`` carries registered declarations but none
            from the scope that declared ``handle``.
    """
    registry = parameter_handles(problem)
    if not registry:
        return ()
    identity = getattr(handle, "identity", None)
    namespace = identity[0] if identity is not None else None
    declarations = tuple(registry.get(namespace, ()))
    if any(candidate.identity == identity for candidate in declarations):
        return declarations
    raise ValueError(
        f"parameter handle {getattr(handle, 'name', handle)!r} comes from a Recipe "
        "builder that declared nothing on this problem. Pass the handle returned "
        "by the spec.param(...) / spec.function(...) call of the build() that "
        "produced this problem."
    )


def declared_handle_value(problem: Any, handle: Any) -> tuple[str, Any]:
    """Return ``(kind, value)``: ``handle``'s declaration kind and ``problem``'s current value for it.

    Resolves through the registry by handle identity, so a handle from a
    structurally identical Recipe rebuild still finds the live declaration. The
    value is read from ``problem``'s own closures, so its array leaves are the
    objects ``problem``'s theta holds. A function's value is the whole declared
    callable, not its partition.

    Args:
        problem: A lowered problem.
        handle: A Recipe handle returned by ``spec.param(...)`` or
            ``spec.function(...)``.

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    identity = getattr(handle, "identity", None)
    namespace = identity[0] if identity is not None else None
    registry = parameter_handles(problem)
    declared = registry.get(namespace, ()) if namespace is not None else ()
    match = next((d for d in declared if getattr(d, "identity", None) == identity), None)
    if match is None:
        raise ValueError(
            f"parameter handle {getattr(handle, 'name', handle)!r} must be a parameter "
            "declared on this problem, returned by its own spec.param(...) or "
            "spec.function(...) call."
        )
    kind = getattr(match, "kind", None)
    value = match.value if kind in ("scalar", "array") else match.fn
    from diff_soup.problem._ir.content_hash import declared_named_values

    params, functions = declared_named_values(problem)
    namespace_values = params if kind in ("scalar", "array") else functions
    if namespace_values is not None and match.name in namespace_values:
        value = namespace_values[match.name]
    return kind, value


def handle_leaves(problem: Any, handle: Any) -> tuple:
    """Return ``problem``'s current theta leaves owned by ``handle``, in pytree order.

    A function owns the inexact leaves at its declared learnable paths (every
    inexact leaf when it declares none); a parameter owns the inexact leaves of
    its value. Works for any declared parameter or function handle, learnable
    or not.

    Args:
        problem: A lowered problem.
        handle: A Recipe handle returned by ``spec.param(...)`` or
            ``spec.function(...)``.

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    kind, value = declared_handle_value(problem, handle)
    if kind == "function":
        from diff_soup.problem.parameters import _function_parameter_partition

        value, _ = _function_parameter_partition(value)
    return tuple(leaf for leaf in jax.tree_util.tree_leaves(value) if eqx.is_inexact_array(leaf))


def handle_leaf_ids(problem: Any, handle: Any) -> frozenset:
    """Return the ``id()`` set of [handle_leaves][diff_soup.problem._registry.handle_leaves].

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    return frozenset(id(leaf) for leaf in handle_leaves(problem, handle))


def reset_registry() -> None:
    """Clear every registry-owned cache and reset telemetry counters.

    Use this as a test or session-reset hook to ensure a clean cache state.
    The cache *instances* are preserved (their ``id`` remains stable across the
    session); only their contents and counters are reset.
    """
    lowering_cache.clear()
    value_copy_cache.clear()
    analysis_cache.clear()
    _parameter_handles_by_problem.clear()
    _parameter_handle_problems_by_structure.clear()
