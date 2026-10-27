"""Owner-aware variable references.

[Variable][diff_soup.problem._ir.variable.Variable] uses structural equality
``(name, shape, mass)`` which is useful for content-hashing variable
declarations.  Ownership-sensitive maps inside ``lower()`` cannot use
[Variable][diff_soup.problem._ir.variable.Variable] as a key because two distinct
components may own structurally equal Variables (e.g. ``A.c`` and ``B.c`` in
coupled battery models).

[VarRef][diff_soup.problem._ir.var_ref.VarRef] is the owner-aware
reference: it carries the owning component name and the variable name.
[build_var_id_to_ref][diff_soup.problem._ir.var_ref.build_var_id_to_ref] produces
a ``dict[int, VarRef]`` keyed by Python ``id(variable)`` from sequences of
components and optional family populations, suitable for resolving Variables to
VarRefs at lower-time.  A family member Variable is flagged with ``family=True``
so its owner qualifier stays distinct from a structurally identical
component-owned Variable of the same name.

The optional ``families`` argument defaults to ``()`` so the slim single-domain
path (no families) can call
[build_var_id_to_ref][diff_soup.problem._ir.var_ref.build_var_id_to_ref] with a
single argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class AmbiguousVariableError(KeyError):
    """Raised when a Variable lookup in a multi-component view is ambiguous.

    Subclasses ``KeyError`` so existing ``except KeyError`` handlers in user
    closures still catch it.
    """


class UnregisteredGlobalError(KeyError):
    """Raised when a view lookup names a GlobalVariable the view does not carry.

    Subclasses ``KeyError`` for the same reason as
    :class:`AmbiguousVariableError`; callers that can say *why* the global is
    absent (for example an observation evaluated without the bordered
    globals) catch it and raise a typed configuration error instead.

    Attributes:
        name: The looked-up global's name.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"GlobalVariable {name!r} not registered with this view.")
        self.name = str(name)


@dataclass(frozen=True, slots=True)
class VarRef:
    """Address of a variable by its owner's name and its own.

    Attributes:
        component_name: Name of the component or family that owns the variable.
        variable_name: Name of the variable.
        family: Whether the owner is a family.
    """

    component_name: str
    variable_name: str
    family: bool = False


def build_var_id_to_ref(components: Sequence, families: Sequence = ()) -> dict[int, VarRef]:
    """Build an ``id(variable) -> VarRef`` map for ownership lookup.

    Args:
        components: Sequence of Component objects whose ``variables`` attribute
            is a tuple of Variable instances.
        families: Optional sequence of family populations whose
            ``member_variables`` template Variables are mapped to
            family-flavored ``VarRef`` (``family=True``).  Defaults to ``()``
            so the single-argument call (no families) stays valid.

    Returns:
        Mapping from Python object identity of a Variable to the ``VarRef`` that
        names its owning component (or family) and slot.
    """
    out: dict[int, VarRef] = {}
    for comp in components:
        for v in comp.variables:
            out[id(v)] = VarRef(comp.name, v.name)
    for fam in families:
        for v in fam.member_variables:
            out[id(v)] = VarRef(fam.name, v.name, family=True)
    return out
