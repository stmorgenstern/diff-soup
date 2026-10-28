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

import functools
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


# ---------------------------------------------------------------------------
# The theta path table: which places of theta hold each declaration's arrays
# ---------------------------------------------------------------------------
#
# Theta is every inexact array of a lowered problem, in pytree order. A
# declared parameter read by K closures sits at K places (one per closure's
# parameter namespace), and a function's array that a boundary condition also
# holds sits at one more. The table records those places once per problem, as
# key paths, so reading or writing a declaration never compares array objects:
# a theta that went through ``tree_map``, a device transfer or a trace keeps
# its paths. The places holding one array are a slot; a declaration owns one
# slot per array it declares, and every other slot is "not a parameter".

_THETA_TABLES: dict[int, tuple[weakref.ReferenceType, "ThetaTable"]] = {}
_MISSING = object()


def _plain_key(key: Any) -> tuple:
    """A JAX key-path entry as a plain, picklable ``(kind, value)`` pair."""
    if isinstance(key, jax.tree_util.GetAttrKey):
        return ("attr", key.name)
    if isinstance(key, jax.tree_util.DictKey):
        return ("key", key.key)
    if isinstance(key, jax.tree_util.SequenceKey):
        return ("idx", key.idx)
    if isinstance(key, jax.tree_util.FlattenedIndexKey):
        return ("flat", key.key)
    return ("other", str(key))


def plain_path(path: tuple) -> tuple:
    """A JAX key path as a tuple of plain ``(kind, value)`` pairs."""
    return tuple(_plain_key(key) for key in path)


def dotted_path(path: tuple) -> str:
    """A plain key path in the dotted grammar of ``learnable_paths``."""
    return ".".join(str(value) for _kind, value in path)


class ThetaTable:
    """The key paths of every array place in a theta, grouped by the array they hold.

    ``slots[i]`` is the tuple of paths (relative to the problem, which is
    ``theta.inner``) holding one array, in pytree order; the first is the one a
    read uses. ``parameter[i]`` is whether a declaration owns slot ``i``.
    ``declarations`` holds one ``(name, kind, value_path, slot_indices)`` row per
    declared parameter (``kind="parameter"``) or function (``"function"``), with
    the slots of its arrays in declared order (a function's learnable arrays
    only). Rows are keyed by declaration, never by array object, so two
    declarations written from one array keep their own places.

    Static and hashable, so it rides on theta's tree structure: equal for value
    copies of one lowering, and kept by ``tree_map``, ``jit`` and ``grad``.
    """

    __slots__ = ("slots", "parameter", "declarations", "_hash", "_slot_of", "_rows", "_layouts")

    def __init__(self, slots: tuple, parameter: tuple, declarations: tuple):
        self.slots = tuple(tuple(paths) for paths in slots)
        self.parameter = tuple(bool(flag) for flag in parameter)
        self.declarations = tuple(declarations)
        self._hash = hash((self.slots, self.parameter, self.declarations))
        self._slot_of = {path: index for index, paths in enumerate(self.slots) for path in paths}
        self._rows = {(row[0], row[1]): row for row in self.declarations}
        self._layouts: dict = {}

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, ThetaTable) or self._hash != other._hash:
            return False
        return (self.slots, self.parameter, self.declarations) == (
            other.slots,
            other.parameter,
            other.declarations,
        )

    def __repr__(self) -> str:
        return f"ThetaTable({len(self.declarations)} declarations, {len(self.slots)} slots)"

    def __reduce__(self):
        return (ThetaTable, (self.slots, self.parameter, self.declarations))

    def row(self, name: str, kind: str):
        """The ``(name, kind, value_path, slot_indices)`` row of a declaration, or ``None``."""
        return self._rows.get((name, kind))

    def key_of(self, path: tuple) -> tuple:
        """The first path of the slot holding ``path``, or ``path`` itself outside every slot."""
        index = self._slot_of.get(path)
        return path if index is None else self.slots[index][0]

    def slot_keys(self, slot_indices) -> frozenset:
        """The first path of each slot in ``slot_indices``."""
        return frozenset(self.slots[index][0] for index in slot_indices)

    def layout(self, tree: Any, *, strip: int) -> tuple:
        """``(leaves, treedef, paths)`` of ``tree``, with ``strip`` leading path entries dropped.

        The paths are cached per tree structure, so a repeated read or write
        flattens without paths.
        """
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        paths = self._layouts.get((treedef, strip))
        if paths is None:
            paths = tuple(
                plain_path(path)[strip:]
                for path, _leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
            )
            if len(self._layouts) > 16:
                self._layouts.clear()
            self._layouts[(treedef, strip)] = paths
        return leaves, treedef, paths


def _copies_agree(leaves: list) -> bool:
    """Whether the arrays at one slot's places are equal (always for tracers)."""
    first = leaves[0]
    for leaf in leaves[1:]:
        if leaf is first or isinstance(leaf, jax.core.Tracer) or isinstance(first, jax.core.Tracer):
            continue
        if jax.numpy.shape(leaf) != jax.numpy.shape(first):
            return False
        if not bool(jax.numpy.array_equal(leaf, first)):
            return False
    return True


def build_theta_table(tree: Any) -> ThetaTable:
    """Build the [ThetaTable][diff_soup.problem._registry.ThetaTable] of a concrete pytree.

    Places come from the pytree's structure: every parameter namespace of a
    lowered closure names its declarations, so the K closures reading a
    declaration give its K places. A place outside every namespace that holds
    the same array object as a declaration's place (a boundary condition given
    the declared signal) joins that declaration's slot, and the remaining
    places are grouped by the array they hold. This is the one place array
    identity is read, on the problem itself, once.
    """
    from diff_soup.problem._spec.build import _NamedValues
    from diff_soup.problem.parameters import _function_parameter_partition

    places = [
        (plain_path(path), leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if eqx.is_inexact_array(leaf)
    ]
    order = {path: index for index, (path, _leaf) in enumerate(places)}
    leaf_at = dict(places)

    owned: dict[tuple, list] = {}
    value_paths: dict[tuple, tuple] = {}
    namespaces = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda node: isinstance(node, _NamedValues)
    )[0]
    for path, node in namespaces:
        if not isinstance(node, _NamedValues):
            continue
        base = plain_path(path)
        kind = {("attr", "params"): "parameter", ("attr", "functions"): "function"}.get(
            base[-1] if base else None
        )
        if kind is None:
            continue
        for name in node.names:
            value = node.values[name]
            if kind == "function":
                value = _function_parameter_partition(value)[0]
            relative = [
                plain_path(leaf_path)
                for leaf_path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]
                if eqx.is_inexact_array(leaf)
            ]
            prefix = base + (("attr", "values"), ("key", name))
            key = (name, kind)
            if key not in owned:
                owned[key] = [[] for _ in relative]
                value_paths[key] = prefix
            if len(owned[key]) != len(relative):
                continue
            for index, rel in enumerate(relative):
                if prefix + rel in order:
                    owned[key][index].append(prefix + rel)

    slots: list[list] = []
    parameter: list[bool] = []
    slot_of: dict[tuple, int] = {}
    declared_slots: dict[tuple, list] = {}
    for key, leaves in owned.items():
        declared_slots[key] = []
        for paths in leaves:
            if not paths:
                continue
            fresh = [path for path in paths if path not in slot_of]
            if not fresh:
                declared_slots[key].append(slot_of[paths[0]])
                continue
            slot_of.update((path, len(slots)) for path in fresh)
            declared_slots[key].append(len(slots))
            slots.append(fresh)
            parameter.append(True)
    by_object = {id(leaf_at[path]): index for index, paths in enumerate(slots) for path in paths}
    for path, leaf in places:
        if path in slot_of:
            continue
        index = by_object.get(id(leaf))
        if index is None:
            index = len(slots)
            by_object[id(leaf)] = index
            slots.append([])
            parameter.append(False)
        slots[index].append(path)
        slot_of[path] = index
    for index, paths in enumerate(slots):
        paths.sort(key=order.__getitem__)
        assert not parameter[index] or _copies_agree([leaf_at[p] for p in paths]), (
            f"theta places {[dotted_path(p) for p in paths]} hold one declared array "
            "but disagree in value."
        )

    rank = sorted(range(len(slots)), key=lambda index: order[slots[index][0]])
    renumber = {old: new for new, old in enumerate(rank)}
    declarations = tuple(
        (
            name,
            kind,
            value_paths[(name, kind)],
            tuple(renumber[i] for i in declared_slots[(name, kind)]),
        )
        for name, kind in owned
    )
    return ThetaTable(
        tuple(tuple(slots[index]) for index in rank),
        tuple(parameter[index] for index in rank),
        declarations,
    )


def theta_table(problem: Any) -> ThetaTable:
    """Return ``problem``'s [ThetaTable][diff_soup.problem._registry.ThetaTable], built once per problem."""
    entry = _THETA_TABLES.get(id(problem))
    if entry is not None and entry[0]() is problem:
        return entry[1]
    table = build_theta_table(problem)
    identity = id(problem)

    def _discard(dead_ref, *, key=identity):
        current = _THETA_TABLES.get(key)
        if current is not None and current[0] is dead_ref:
            _THETA_TABLES.pop(key, None)

    try:
        _THETA_TABLES[identity] = (weakref.ref(problem, _discard), table)
    except TypeError:
        pass
    return table


def theta_view(theta: Any) -> tuple:
    """Return ``(table, strip)``: ``theta``'s table and how many path entries lead to the problem.

    A Recipe theta carries its table and holds the problem under ``inner``. Any
    other pytree gets a table of its own places, grouped by the array they hold.
    """
    table = getattr(theta, "table", None)
    if isinstance(table, ThetaTable) and hasattr(theta, "inner"):
        return table, 1
    return build_theta_table(theta), 0


def theta_leaf_keys(theta: Any) -> tuple:
    """Return, for each leaf of ``theta`` in flattening order, the key of the slot holding it.

    A slot's key is its first path, relative to the problem, so it is the same
    for every theta of one structure and for the problem's own
    ``*_leaf_ids`` sets.
    """
    table, strip = theta_view(theta)
    _leaves, _treedef, paths = table.layout(theta, strip=strip)
    return tuple(table.key_of(path) for path in paths)


def theta_mask(theta: Any, keys: frozenset) -> Any:
    """Return a boolean pytree over ``theta``, ``True`` at the inexact leaves whose slot key is in ``keys``."""
    table, strip = theta_view(theta)
    leaves, treedef, paths = table.layout(theta, strip=strip)
    return jax.tree_util.tree_unflatten(
        treedef,
        [
            eqx.is_inexact_array(leaf) and table.key_of(path) in keys
            for leaf, path in zip(leaves, paths)
        ],
    )


def _declaration_key(handle: Any, caller: str) -> tuple:
    """``(name, kind)`` of a parameter or function handle, refusing anything else."""
    if not (hasattr(handle, "identity") and hasattr(handle, "name")):
        raise TypeError(
            f"{caller}: keys must be parameters returned by spec.param or "
            f"spec.function, not {type(handle).__name__} {handle!r}."
        )
    return handle.name, "function" if getattr(handle, "kind", None) == "function" else "parameter"


def _row(table: ThetaTable, handle: Any, caller: str) -> tuple:
    """The table row of ``handle``, refusing a handle this theta does not declare."""
    name, kind = _declaration_key(handle, caller)
    row = table.row(name, kind)
    if row is None:
        raise ValueError(
            f"{caller}: parameter {name!r} matches no array of this problem's parameters; "
            "pass the parameter returned when this problem was built."
        )
    if not row[3]:
        raise ValueError(
            f"{caller}: {name!r} holds no floating-point array, so theta cannot "
            "change it. Declare its value as an array, or a function as an eqx.Module "
            "that holds its arrays."
        )
    return row


def _declared_part(tree: Any, row: tuple) -> Any:
    """The part of a declaration's value that theta holds: its arrays, or a function's learnable ones."""
    from diff_soup.problem._spec.build import _NamedValues
    from diff_soup.problem.parameters import _function_parameter_partition

    name, kind, value_path, _slots = row
    namespace_path = value_path[:-2]
    for path, node in jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda node: isinstance(node, _NamedValues)
    )[0]:
        if isinstance(node, _NamedValues) and plain_path(path) == namespace_path:
            value = node.values[name]
            if kind == "function":
                return _function_parameter_partition(value)[0]
            return eqx.filter(value, eqx.is_inexact_array)
    raise ValueError(f"theta holds no namespace for {name!r}.")  # pragma: no cover


@functools.lru_cache(maxsize=None)
def _copy_reader(count: int):
    """A read of the first of ``count`` equal copies whose tangent is their mean.

    The value is the first copy exactly. The tangent is linear in the copies,
    so reverse mode splits a cotangent evenly over them, and the derivative
    through ``theta.replace(handle, x)``, which writes every copy, is exact.
    """

    @jax.custom_jvp
    def read(*copies):
        return copies[0]

    @read.defjvp
    def _read_jvp(primals, tangents):
        total = tangents[0]
        for tangent in tangents[1:]:
            total = total + tangent
        return primals[0], total / count

    return read


def theta_value(theta: Any, handle: Any, *, caller: str = "theta[...]") -> Any:
    """Return the value ``theta`` holds for ``handle``: an array, or a function's learnable arrays."""
    table, strip = theta_view(theta)
    row = _row(table, handle, caller)
    leaves, _treedef, paths = table.layout(theta, strip=strip)
    index = {path: position for position, path in enumerate(paths)}

    def read(slot):
        try:
            copies = [leaves[index[path]] for path in table.slots[slot]]
        except KeyError:
            raise ValueError(
                f"{caller}: this theta does not hold every array of {row[0]!r}; "
                "read it from a whole theta, not a partition of one."
            ) from None
        return copies[0] if len(copies) == 1 else _copy_reader(len(copies))(*copies)

    values = [read(slot) for slot in row[3]]
    if row[1] == "parameter" and len(row[3]) == 1 and table.slots[row[3][0]][0] == row[2]:
        return values[0]
    tree = theta.inner if strip else theta
    part = _declared_part(tree, row)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(part), values)


def theta_replace(theta: Any, values: Mapping, *, caller: str) -> Any:
    """Return ``theta`` with each handle's arrays set to its value at every place it is read.

    A scalar or array value is broadcast to each array's shape and dtype. A
    function also takes a module of its declared structure, whose learnable
    arrays replace the declared ones in order. One new array is written to all
    places of a slot, so the places keep holding one array.
    """
    table, strip = theta_view(theta)
    leaves, treedef, paths = table.layout(theta, strip=strip)
    leaves = list(leaves)
    index = {path: position for position, path in enumerate(paths)}
    for handle, value in values.items():
        name, kind = _declaration_key(handle, caller)
        if value is None:
            raise ValueError(
                f"{caller}: the value for {name!r} is None; pass a scalar, an array "
                "or, for a function, a module."
            )
        row = _row(table, handle, caller)
        try:
            positions = [[index[path] for path in table.slots[slot]] for slot in row[3]]
        except KeyError:
            raise ValueError(
                f"{caller}: this theta does not hold every array of {name!r}; set it "
                "on a whole theta, not a partition of one."
            ) from None
        old = [leaves[places[0]] for places in positions]
        single = kind == "parameter" and len(row[3]) == 1 and table.slots[row[3][0]][0] == row[2]
        if _is_array_like(value) or single:
            new = [_leaf_value(caller, name, value, leaf, broadcast=True) for leaf in old]
        else:
            from diff_soup.problem.parameters import _function_parameter_partition

            declared = _declared_part(theta.inner if strip else theta, row)
            given = _function_parameter_partition(value)[0] if kind == "function" else value
            if jax.tree_util.tree_structure(given) != jax.tree_util.tree_structure(declared):
                shapes = [
                    tuple(getattr(leaf, "shape", ())) for leaf in jax.tree_util.tree_leaves(given)
                ]
                raise ValueError(
                    f"{caller}: the value for {name!r} must have the structure of its "
                    f"declared value, a {type(declared).__name__} with arrays of shapes "
                    f"{[tuple(leaf.shape) for leaf in old]}; got a {type(value).__name__} "
                    f"with leaves of shapes {shapes}."
                )
            new = [
                _leaf_value(caller, name, given_leaf, leaf, broadcast=False)
                for leaf, given_leaf in zip(old, jax.tree_util.tree_leaves(given))
            ]
        for places, array in zip(positions, new):
            for place in places:
                leaves[place] = array
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _is_array_like(value: Any) -> bool:
    """Whether ``value`` is a number or an array, as opposed to a module or container."""
    import numbers

    import numpy as np

    return isinstance(value, (numbers.Number, np.ndarray, np.generic, jax.Array))


def _leaf_value(caller: str, name: str, value: Any, leaf: Any, *, broadcast: bool) -> Any:
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
            f"{caller}: the value for {name!r} must hold numbers, got "
            f"{type(value).__name__} of dtype {dtype}."
        )
    if jnp.issubdtype(dtype, jnp.complexfloating) and not jnp.issubdtype(
        leaf.dtype, jnp.complexfloating
    ):
        raise TypeError(
            f"{caller}: the value for {name!r} is complex, but the parameter is "
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
        f"{caller}: the value for {name!r} has an array of shape {shape}, which "
        f"does not {'broadcast to' if broadcast else 'match'} the declared shape "
        f"{tuple(leaf.shape)}."
    )


def handle_leaf_keys(problem: Any, handle: Any) -> frozenset:
    """Return the slot keys of ``handle``'s arrays in ``problem``'s theta.

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    declared_handle_value(problem, handle)
    name, kind = _declaration_key(handle, "parameter selection")
    row = theta_table(problem).row(name, kind)
    return frozenset() if row is None else theta_table(problem).slot_keys(row[3])


def handle_leaves(problem: Any, handle: Any) -> tuple:
    """Return ``problem``'s current theta leaves owned by ``handle``, in declared order.

    A function owns the inexact leaves at its declared learnable paths (every
    inexact leaf when it declares none); a parameter owns the inexact leaves of
    its value. Works for any declared parameter or function handle, learnable
    or not.

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    declared_handle_value(problem, handle)
    table = theta_table(problem)
    name, kind = _declaration_key(handle, "parameter selection")
    row = table.row(name, kind)
    if row is None:
        return ()
    leaves, _treedef, paths = table.layout(problem, strip=0)
    index = {path: position for position, path in enumerate(paths)}
    return tuple(leaves[index[table.slots[slot][0]]] for slot in row[3])


def handle_leaf_ids(problem: Any, handle: Any) -> frozenset:
    """Return the slot keys of [handle_leaves][diff_soup.problem._registry.handle_leaves].

    A key is the first path of the theta places holding one array, so it names
    the array in every theta of ``problem``'s structure.

    Raises:
        ValueError: If ``handle`` was not declared on ``problem``.
    """
    return handle_leaf_keys(problem, handle)


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
    _THETA_TABLES.clear()
