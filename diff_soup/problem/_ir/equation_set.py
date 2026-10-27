"""Top-level EquationSet IR container.

[EquationSet][diff_soup.problem._ir.equation_set.EquationSet] is the composition root consumed by ``lower()``.  The
``couplings`` and ``families`` slots are populated via duck-typing (no module-level
imports of ``coupling.Coupling`` / ``family.Family`` here, keeping this node
import-cycle free); the content-hash fold and ``with_()`` rewrite both operate
on them.  The slim single-domain path never populates these slots.

``residual_scale`` is stored as a plain ``dict`` — ``MappingProxyType`` is
unhashable, so a plain dict is used and the frozen ``eqx.Module`` field blocks
reassignment.  Its values participate in structural identity via ``content_hash``.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx

from .component import Component
from .constraint import Constraint
from .control import Control
from .observation import Observation
from .scales import Scales, identity_scales
from .variable import GlobalVariable


class EquationSet(eqx.Module):
    """Frozen description of a problem, its components, globals, constraints, observations, controls, couplings and families.

    `spec.to_equation_set()` builds one, and [`lower`][diff_soup.problem.lower]
    compiles it. `a | b` concatenates two sets, and `with_(...)` returns a copy
    with nodes added, removed or replaced.

    Attributes:
        components: Components, each a domain with its fields, equations and boundary conditions, at least one.
        globals: Global unknowns.
        constraints: Constraint records that close the globals.
        observations: Declared observations.
        controls: Declared controls.
        couplings: Couplings that move values between components.
        families: Families of identical members.
        scales: [`Scales`][diff_soup.problem.Scales] of the problem.
        residual_scale: Divisor of each field's residual rows, keyed by `(component, field)`.
        quad_fields: Names of the fields that `q` reads at quadrature points on a finite-element domain.
    """

    # The closure-bearing slots are NON-static (pytree) fields, like
    # ``globals`` and ``families``: each node's own static-vs-leaf split is
    # internal (equation / constraint / observation / control closures are
    # runtime fields on their nodes; component domains carry mesh arrays), so
    # parameter and geometry arrays surface as EquationSet leaves —
    # differentiable, vmappable, and reachable by path-based selectors — while
    # every node spine stays in its own treedef.  Content identity comes from
    # the explicit ``content_hash`` walk, which folds these slots by attribute
    # rather than by field staticness.
    components: tuple[Component, ...] = ()
    globals: tuple[GlobalVariable, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    observations: tuple[Observation, ...] = ()
    controls: tuple[Control, ...] = ()
    # ``couplings`` keeps the slot (content-hash + with_() reference it); its
    # element type is ``coupling.Coupling`` (Variable + TransferOp members, all
    # array-free/hashable).  Always () on the slim single-domain path.
    couplings: tuple = eqx.field(static=True, default=())
    # NON-static (pytree) like ``globals``: each Family's own static-vs-leaf
    # split is internal, so the member parameter arrays and the member domain's
    # arrays surface as EquationSet leaves (differentiable / vmappable) while
    # the Family spine stays in the Family treedef.  Always () on the slim path.
    families: tuple = ()
    scales: Scales = eqx.field(static=True, default_factory=identity_scales)
    residual_scale: Mapping = eqx.field(static=True, default_factory=dict)
    # Set of field names declared ``spec.field(quad=True)`` (or defaulting to
    # ``True`` on FEM).  Stored here so ``lower()`` can thread it to
    # ``LoweredProblem.quad_fields`` without an extra kwarg at the lowering seam.
    # Always empty on the FVM/FDM path; no downstream IR changes for non-FEM specs.
    quad_fields: "frozenset[str]" = eqx.field(static=True, default_factory=frozenset)

    def __init__(
        self,
        *,
        components: tuple[Component, ...],
        globals: tuple[GlobalVariable, ...] = (),
        constraints: tuple[Constraint, ...] = (),
        observations: tuple[Observation, ...] = (),
        controls: tuple[Control, ...] = (),
        couplings: tuple = (),
        families: tuple = (),
        scales: Scales | None = None,
        residual_scale: Mapping | None = None,
        quad_fields: "frozenset[str] | None" = None,
    ):
        if not components:
            raise ValueError("EquationSet requires at least one Component")
        _check_unique([c.name for c in components], "component")
        _check_unique([g.name for g in globals], "global")
        _check_unique([c.name for c in constraints], "constraint")
        _check_unique([o.name for o in observations], "observation")
        _check_unique([c.name for c in controls], "control")
        _check_unique([cp.name for cp in couplings], "coupling")
        _check_unique([p.name for p in families], "family")
        scales = scales if scales is not None else identity_scales()
        _validate_scales_keys(scales, components, globals, constraints, families)
        self.components = tuple(components)
        self.globals = tuple(globals)
        self.constraints = tuple(constraints)
        self.observations = tuple(observations)
        self.controls = tuple(controls)
        self.couplings = tuple(couplings)
        self.families = tuple(families)
        self.scales = scales
        # ``residual_scale`` is static config whose values participate in
        # structural identity (mirrors how ``scales`` is treated).  Stored as a
        # plain dict (MappingProxyType is unhashable).  The eqx-frozen field
        # blocks reassignment of the slot.
        self.residual_scale = dict(residual_scale) if residual_scale is not None else {}
        self.quad_fields = quad_fields if quad_fields is not None else frozenset()

    def __or__(self, other: "EquationSet") -> "EquationSet":
        """Concatenate two EquationSets along every named slot.

        Validates per-kind name uniqueness via
        [__init__][diff_soup.problem._ir.equation_set.EquationSet.__init__]
        so any collision
        (component / global / constraint / observation / control / coupling)
        raises ``ValueError``.  Returns ``NotImplemented`` for non-EquationSet
        operands so Python tries ``other.__ror__(self)`` next.

        If both operands carry non-identity ``Scales`` and they differ, raise —
        the caller must bring both sides to the same ``Scales`` first.  Otherwise
        the non-identity side wins.

        ``families`` concatenate (uniqueness re-validated by ``__init__``).
        ``residual_scale`` mappings merge ``{**self, **other}`` (right wins on
        key collision), consistent with the non-identity-side-wins rule for
        ``scales``.
        """
        if not isinstance(other, EquationSet):
            return NotImplemented
        identity = identity_scales()
        if self.scales != other.scales and self.scales != identity and other.scales != identity:
            raise ValueError(
                "cannot concatenate EquationSets with conflicting scales; give both "
                "sides the same Scales, or leave one of them at identity scales."
            )
        merged_scales = other.scales if self.scales == identity else self.scales
        merged_residual_scale = {**self.residual_scale, **other.residual_scale}
        merged_quad_fields = self.quad_fields | other.quad_fields
        return EquationSet(
            components=self.components + other.components,
            globals=self.globals + other.globals,
            constraints=self.constraints + other.constraints,
            observations=self.observations + other.observations,
            controls=self.controls + other.controls,
            couplings=self.couplings + other.couplings,
            families=self.families + other.families,
            scales=merged_scales,
            residual_scale=merged_residual_scale,
            quad_fields=merged_quad_fields,
        )

    def with_(
        self,
        *,
        replace: tuple = (),
        add_components: tuple = (),
        add_globals: tuple = (),
        add_constraints: tuple = (),
        add_observations: tuple = (),
        add_controls: tuple = (),
        add_couplings: tuple = (),
        remove_components=None,
        remove_globals=None,
        remove_constraints=None,
        remove_observations=None,
        remove_controls=None,
        remove_couplings=None,
        families: tuple | None = None,
        scales: Scales | None = None,
        residual_scale: Mapping | None = None,
    ) -> "EquationSet":
        """Tree-rewrite: replace by isinstance, add to tuples, remove by predicate.

        Operations apply in the order ``replace → remove → add`` so an ``add``
        cannot be silently undone by ``remove``.  The result is rebuilt via
        [EquationSet.__init__][diff_soup.problem._ir.equation_set.EquationSet.__init__],
        which enforces per-kind name uniqueness.

        ``replace`` accepts ``((Type, replacement), ...)``: every node matching
        ``Type`` (including BCs nested in ``Component.bcs``) is substituted with
        ``replacement``.

        Used for HER VC↔CC mode swaps (``add_globals + add_constraints +
        add_controls``) and any inversion-time mode rewrite.

        Optional ``scales`` / ``families`` / ``residual_scale`` keywords swap
        those slots on the result.  ``families`` and ``residual_scale`` are
        preserved from the source when their keyword is omitted.
        """
        from .tree_rewrite import _apply_rewrite

        adds = {
            "components": add_components,
            "globals": add_globals,
            "constraints": add_constraints,
            "observations": add_observations,
            "controls": add_controls,
            "couplings": add_couplings,
        }
        removes = {
            "components": remove_components,
            "globals": remove_globals,
            "constraints": remove_constraints,
            "observations": remove_observations,
            "controls": remove_controls,
            "couplings": remove_couplings,
        }
        result = _apply_rewrite(self, replace=replace, adds=adds, removes=removes)
        resolved_families = self.families if families is None else families
        resolved_residual_scale = self.residual_scale if residual_scale is None else residual_scale
        if scales is not None or families is not None or residual_scale is not None:
            result = EquationSet(
                components=result.components,
                globals=result.globals,
                constraints=result.constraints,
                observations=result.observations,
                controls=result.controls,
                couplings=result.couplings,
                families=resolved_families,
                scales=scales if scales is not None else self.scales,
                residual_scale=resolved_residual_scale,
                quad_fields=self.quad_fields,
            )
        elif self.families or self.residual_scale or self.quad_fields:
            # Preserve families / residual_scale / quad_fields across a plain
            # rewrite (``_apply_rewrite`` rebuilds via __init__ and would otherwise
            # drop them to defaults).
            result = EquationSet(
                components=result.components,
                globals=result.globals,
                constraints=result.constraints,
                observations=result.observations,
                controls=result.controls,
                couplings=result.couplings,
                families=resolved_families,
                scales=self.scales,
                residual_scale=resolved_residual_scale,
                quad_fields=self.quad_fields,
            )
        return result


def _check_unique(names: list[str], kind: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate {kind} name in EquationSet")


def _validate_scales_keys(
    scales: Scales,
    components: tuple[Component, ...],
    globals_: tuple[GlobalVariable, ...],
    constraints: tuple[Constraint, ...],
    families: tuple = (),
) -> None:
    """Reject Scales that name unknown components / variables / constraints.

    Missing entries are NOT errors — they default to identity at the boundary.
    Extra entries (typos) are errors.

    Globals are encoded as ``("", global_name)`` in ``state_ref`` /
    ``state_offset``.

    A ``(owner, var_name)`` key whose ``owner`` matches a family name is accepted
    when ``var_name`` is one of that family's member variables — families share
    the component owner-namespace for scaling.
    """
    comp_by_name = {c.name: c for c in components}
    global_names = {g.name for g in globals_}
    constraint_names = {c.name for c in constraints}
    fam_member_vars = {fam.name: {v.name for v in fam.member_variables} for fam in families}

    def _is_known_owner_var(comp_name: str, var_name: str) -> bool:
        if comp_name in comp_by_name:
            known = {v.name for v in comp_by_name[comp_name].variables}
            return var_name in known
        if comp_name in fam_member_vars:
            return var_name in fam_member_vars[comp_name]
        return False

    for comp_name, var_name in scales.state_ref:
        if comp_name == "":
            if var_name not in global_names:
                raise ValueError(f"Scales.state_ref names unknown global {var_name!r}")
            continue
        if comp_name not in comp_by_name and comp_name not in fam_member_vars:
            raise ValueError(f"Scales.state_ref names unknown component {comp_name!r}")
        if not _is_known_owner_var(comp_name, var_name):
            raise ValueError(
                f"Scales.state_ref names unknown variable {var_name!r} in component {comp_name!r}"
            )

    for comp_name, var_name in scales.state_offset:
        if comp_name == "":
            if var_name not in global_names:
                raise ValueError(f"Scales.state_offset names unknown global {var_name!r}")
            continue
        if comp_name not in comp_by_name and comp_name not in fam_member_vars:
            raise ValueError(f"Scales.state_offset names unknown component {comp_name!r}")
        if not _is_known_owner_var(comp_name, var_name):
            raise ValueError(
                f"Scales.state_offset names unknown variable "
                f"{var_name!r} in component {comp_name!r}"
            )

    for c_name in scales.integrand_ref:
        if c_name not in constraint_names:
            raise ValueError(f"Scales.integrand_ref names unknown constraint {c_name!r}")
