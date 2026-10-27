"""Tree rewrites for ``EquationSet.with_(replace=, add=, remove=)``.

Three orthogonal operations:

1. ``replace=((Type, new_node), ...)`` — every node matching ``Type`` is
   replaced by ``new_node``.  BCs nested inside ``Component.bcs`` are descended
   into and rebuilt; top-level kinds (constraint, observation, control,
   coupling, global) are swapped in their tuple.
2. ``add_<kind>=tuple`` — append to the corresponding tuple.
3. ``remove_<kind>=callable`` — drop entries where ``callable(item)`` is truthy.

All three are applied in order ``replace → remove → add`` so an ``add`` cannot
be silently undone by a later ``remove``.  The result is rebuilt via
[EquationSet.__init__][diff_soup.problem._ir.equation_set.EquationSet.__init__] so per-kind name uniqueness is enforced
(``ValueError`` on collision).

The ``couplings`` kind stays in ``_KINDS`` (touched via ``getattr`` + tuple ops
only — no ``coupling.py`` import); on the slim single-domain path it is always
an empty tuple.

Used internally by [with_][diff_soup.problem._ir.equation_set.EquationSet.with_]; do not call directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .bc import BoundaryCondition
from .component import Component

_KINDS = (
    "components",
    "globals",
    "constraints",
    "observations",
    "controls",
    "couplings",
)


def _replace_in_tuple(items, replacements: dict) -> tuple:
    """Substitute every entry whose isinstance type matches a key in replacements.

    ``replacements`` is ``{Type: new_node}``.
    """
    out = []
    for x in items:
        new = None
        for cls, repl in replacements.items():
            if isinstance(x, cls):
                new = repl
                break
        out.append(x if new is None else new)
    return tuple(out)


def _replace_inside_components(components, replacements: dict) -> tuple:
    """Descend into ``Component.bcs`` and substitute by isinstance."""
    if not any(issubclass(cls, BoundaryCondition) for cls in replacements):
        return components
    bc_replacements = {
        cls: r for cls, r in replacements.items() if issubclass(cls, BoundaryCondition)
    }
    out = []
    for comp in components:
        # Substitute into the conditions that still carry their array-bearing
        # callables: ``comp.bcs`` holds array-stripped Dirichlet / flux
        # callables, and rebuilding from those would record the array-free
        # versions as the component's runtime callables.
        source_bcs = comp.runtime_bcs()
        new_bcs = _replace_in_tuple(source_bcs, bc_replacements)
        if all(a is b for a, b in zip(new_bcs, source_bcs)):
            out.append(comp)
            continue
        # Rebuild the Component (eqx.Module is frozen).
        out.append(
            Component(
                name=comp.name,
                domain=comp.domain,
                variables=comp.variables,
                equations=comp.equations,
                bcs=new_bcs,
                operator_factory=comp.operator_factory,
                geometry_scale=comp.geometry_scale,
            )
        )
    return tuple(out)


def _filter_tuple(items, predicate: Optional[Callable[[Any], bool]]) -> tuple:
    if predicate is None:
        return tuple(items)
    return tuple(x for x in items if not predicate(x))


def _apply_rewrite(eqset, *, replace, adds, removes):
    """Apply replace → remove → add to an EquationSet.

    Parameters
    ----------
    eqset : EquationSet
    replace : tuple[tuple[type, Any], ...]
        ``((Type, replacement), ...)``.  Replacement applies inside
        ``Component.bcs`` (for BC types) and inside top-level kind tuples for
        non-BC types.
    adds : dict[str, tuple]
        Per-kind add tuples keyed by ``"<kind>"`` (e.g. ``"globals"``).
    removes : dict[str, Optional[Callable]]
        Per-kind remove predicates keyed the same way.
    """
    from .equation_set import EquationSet

    replacements = dict(replace) if replace else {}
    state = {kind: tuple(getattr(eqset, kind)) for kind in _KINDS}

    if replacements:
        state["components"] = _replace_inside_components(state["components"], replacements)
        for kind in ("globals", "constraints", "observations", "controls", "couplings"):
            state[kind] = _replace_in_tuple(state[kind], replacements)

    for kind in _KINDS:
        state[kind] = _filter_tuple(state[kind], removes.get(kind))

    for kind in _KINDS:
        extra = adds.get(kind, ())
        if extra:
            state[kind] = state[kind] + tuple(extra)

    return EquationSet(
        components=state["components"],
        globals=state["globals"],
        constraints=state["constraints"],
        observations=state["observations"],
        controls=state["controls"],
        couplings=state["couplings"],
    )
