"""Owner-aware merged view accessor for residual/observation closures.

User closures index this view as ``view[Variable]`` / ``view[VarRef]`` /
``view[(comp_name, var_name)]`` for state, and ``view[GlobalVariable]`` for
galvanostatic globals.  Variable lookups resolve by ``id()`` exact match first,
then structural fallback.  Globals follow the same two-tier pattern (id() then
name+shape).

The structural fallback is required because closures may construct fresh
``Variable(name=..., shape=..., mass=...)`` handles inside lambdas; their
``id()`` is not present in the lower-time identity map.

Read-through / no-DOF MergedView injection arms used by family/coupling lowering
callers live in the lowering pass, not here.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax

from .var_ref import AmbiguousVariableError, UnregisteredGlobalError, VarRef
from .variable import GlobalVariable, Variable


def _struct_key(v: Variable) -> tuple:
    """Full structural key matching Variable equality: (name, shape, mass)."""
    return (v.name, v.shape, v.mass)


def _global_struct_key(g: GlobalVariable) -> tuple:
    """Structural key for GlobalVariable: (name, init.shape)."""
    return (g.name, tuple(g.init.shape))


class MergedView:
    """Owner-aware accessor for per-component variable arrays + globals.

    Resolution order for ``view[key]``:

    1. ``VarRef`` or ``(comp_name, var_name)`` tuple → direct state lookup.
    2. ``Variable`` whose ``id()`` is in ``var_id_to_ref`` → exact state match.
    3. ``Variable`` not seen before → structural fallback over
       ``(name, shape, mass)``.  Exactly one match returns its slot; multiple
       matches raise [AmbiguousVariableError][diff_soup.problem._ir.var_ref.AmbiguousVariableError]; zero matches raise
       ``KeyError``.
    4. ``GlobalVariable`` whose ``id()`` is in ``global_id_to_name`` → exact
       global match.
    5. ``GlobalVariable`` not seen before → structural fallback over
       ``(name, init.shape)``.  Same-rule semantics as state Variables.

    The structural fallback uses **full structural equality**, not just name
    equality.  Two Variables that share a name but differ in shape or mass are
    distinguishable.  Globals fall back by name+shape since structurally equal
    globals from different eqsets share name space.

    Args:
        by_ref: Mapping from VarRef to the state-array slot.
        var_id_to_ref: ``id(Variable) -> VarRef`` for exact-id state lookup.
        ref_to_var: ``VarRef -> Variable`` for structural-fallback keys.
        globals_by_name: ``global_name -> jax.Array`` slot for global lookup.
        global_id_to_name: ``id(GlobalVariable) -> name`` for exact-id global lookup.
        name_to_global: ``global_name -> GlobalVariable`` for structural-key building.
    """

    __slots__ = (
        "_by_ref",
        "_var_id_to_ref",
        "_struct_index",
        "_globals_by_name",
        "_global_id_to_name",
        "_global_struct_index",
    )

    def __init__(
        self,
        *,
        by_ref: Mapping[VarRef, jax.Array],
        var_id_to_ref: Mapping[int, VarRef],
        ref_to_var: Mapping[VarRef, Variable],
        globals_by_name: Mapping[str, jax.Array] | None = None,
        global_id_to_name: Mapping[int, str] | None = None,
        name_to_global: Mapping[str, GlobalVariable] | None = None,
    ):
        self._by_ref = dict(by_ref)
        self._var_id_to_ref = dict(var_id_to_ref)
        self._struct_index: dict[tuple, list[VarRef]] = {}
        for ref, var in ref_to_var.items():
            if ref not in self._by_ref:
                continue
            self._struct_index.setdefault(_struct_key(var), []).append(ref)

        self._globals_by_name = dict(globals_by_name) if globals_by_name else {}
        self._global_id_to_name = dict(global_id_to_name) if global_id_to_name else {}
        self._global_struct_index: dict[tuple, list[str]] = {}
        if name_to_global:
            for name, g in name_to_global.items():
                if name not in self._globals_by_name:
                    continue
                self._global_struct_index.setdefault(_global_struct_key(g), []).append(name)

    def __getitem__(self, key):
        if isinstance(key, VarRef):
            return self._by_ref[key]
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(s, str) for s in key):
            return self._by_ref[VarRef(*key)]
        if isinstance(key, Variable):
            ref = self._var_id_to_ref.get(id(key))
            if ref is not None:
                return self._by_ref[ref]
            candidates = self._struct_index.get(_struct_key(key), [])
            if len(candidates) == 1:
                return self._by_ref[candidates[0]]
            if len(candidates) > 1:
                raise AmbiguousVariableError(
                    f"Variable {key.name!r} (shape={key.shape}, mass={key.mass!r}) "
                    f"is ambiguous across components "
                    f"{[r.component_name for r in candidates]!r}; "
                    f"index by VarRef('component', {key.name!r}) or by "
                    f"(component, {key.name!r}) tuple instead."
                )
            raise KeyError(
                f"Variable {key.name!r} (shape={key.shape}, mass={key.mass!r}) "
                f"not registered with this view."
            )
        if isinstance(key, GlobalVariable):
            name = self._global_id_to_name.get(id(key))
            if name is not None:
                return self._globals_by_name[name]
            candidates = self._global_struct_index.get(_global_struct_key(key), [])
            if len(candidates) == 1:
                return self._globals_by_name[candidates[0]]
            if len(candidates) > 1:
                raise AmbiguousVariableError(
                    f"GlobalVariable {key.name!r} is ambiguous; index by name string instead."
                )
            raise UnregisteredGlobalError(key.name)
        raise TypeError(f"Cannot index merged view with {type(key).__name__}: {key!r}")

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except (KeyError, AmbiguousVariableError, TypeError):
            return False
