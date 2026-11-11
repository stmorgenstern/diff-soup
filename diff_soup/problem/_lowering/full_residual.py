"""The full residual of a spec with a family layer: its equations, boundaries, couplings, constraints and controls."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.domains.grid.core import _METRIC_DIMENSION
from diff_soup.domains.mesh.fem._boundary import facet_average, facet_shares
from diff_soup.domains.operators.dispatch import dispatch_lumped_mass, finalize_residual
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.lower import _apply_boundary_cast
from diff_soup.problem._ir.variable import GlobalVariable
from diff_soup.problem._lowering.provider_context import (
    _assembly_dtype,
    _check_multi_equation_return,
    _Ctx,
    _face_signature_flux,
    _geometry_scale_value,
    _periodic_axes_of_component,
    _resolve_direct_controls,
)
from diff_soup.problem._spec.derivative_expr import (
    current_slopes,
    slope_context_includes_residual,
)

from .constraints import _eval_constraint_residual
from .couplings import _read_through_alias_owner, _resolve_family_couplings
from .family_residual import _build_family_residual
from .layout import _BlockLayout
from .operator_resolve import resolve_operator

# ======================================================================
# Family / coupling / bordered-global path (full path).
#
# The slim builders above apply only to a single component with no BCs,
# families, couplings, globals, or constraints.  The builders below are
# selected when the full path is needed (``_has_family_layer`` returns True):
# they add the batched family residual (one trace, one vmap'd primitive),
# the family scatter / gather couplings, bordered globals threaded into member
# and global-row views, and the bordered residual ``g`` + dense Schur
# ``B``/``C``/``D`` blocks.
# ======================================================================


def _build_full_residual_fn(
    eqset: EquationSet,
    layout: _BlockLayout,
    compute_dtype,
    bcs,
    *,
    skip_constraints: bool = False,
    restrict_rows_to=None,
    restrict_constraints_to=None,
    hold_residual_currents=None,
    border_probe=None,
):
    """Build the multi-component family/coupling/bordered residual ``(f, g)``.

    Supports N components + boundary conditions + families + scatter/gather +
    read-through couplings + differential/algebraic globals. The kernel takes
    IR-node tuples (``equations``, compatibility-named ``constraints``,
    ``couplings``, ``families``) + BC value-fn / callable tuples as runtime args
    so ``eqx.Module`` closure leaves reach AD; per-family ``member_params`` are
    read off the runtime ``families`` arg (traced leaves) rather than the
    build-time captured family.

    With ``skip_constraints=True`` the bordered-row host resolution (ctx ops,
    boundary measures, flux-total links) is skipped and the returned
    kernel must be called with an empty runtime ``constraints`` tuple; the
    state residual ``f`` is unaffected (globals still thread into the views
    from ``lam``).  Used by the build-time stencil probe, which only needs
    ``f`` and passes no resolved boundary conditions.

    ``restrict_rows_to`` (a set of ``(owner, field)`` keys) builds the
    bordered-global variant of the kernel: only the state-residual rows
    owned by those fields are assembled — their equations, the boundary
    kernels and family scatters landing on them, and the Dirichlet pins —
    while every other equation and family block is never evaluated.  The
    assembled rows and the bordered residual ``g`` are numerically identical
    to the unrestricted kernel's (row contributions are disjoint per field),
    so differentiating this kernel yields exactly ``C = dg/du`` /
    ``D = dg/dlam`` without differentiating unrelated physics.
    ``restrict_constraints_to`` (global-row names) limits which ``g`` rows are
    evaluated (the rest are zeros — their derivative rows come from the
    scatter-sum shortcut, never from this kernel).  A row whose closed global a
    derived-quantity control drives (``binds_to=``) is likewise left at zero:
    the residual replaces that whole row with ``control fn(t) - lambda``, so
    evaluating the row's own integrand could only produce a discarded value.
    ``hold_residual_currents`` (operational-table names) holds the residual-sum
    current of those tables constant under differentiation; the derivative
    kernel adds that current's Jacobian row from the assembled blocks.
    ``border_probe``
    (a [_ConstraintBorderProbe][diff_soup.problem._lowering.full_jacobian._ConstraintBorderProbe])
    records, at trace time, which equations, family blocks, and bordered-global
    rows this kernel evaluated.
    """
    from diff_soup.problem._ir._merged_view import MergedView
    from diff_soup.problem._ir.var_ref import VarRef

    comps = tuple(eqset.components)

    def _residual_ops(c, domain=None):
        domain = c.domain if domain is None else domain
        base = resolve_operator(domain, c.operator_factory).for_residual()
        axes = _periodic_axes_of_component(c)
        if not axes:
            return base
        # A periodic declaration reaches whichever bank the component resolved
        # to, built-in or recipe-supplied: the view composes that bank rather
        # than replacing it.  The wrap travels as a boundary-condition list, so
        # a bank whose laplacian takes no ``bcs`` has no channel for it and is
        # refused here rather than silently computing the non-periodic answer.
        from diff_soup.domains.operators.base import _accepts_bcs, _PeriodicOperatorView

        wrappable = {"laplacian", "div_flux", "advect_upwind"} & set(
            getattr(base, "provides", frozenset())
        )
        for op_name in sorted(wrappable):
            if not _accepts_bcs(getattr(base, op_name)):
                raise ConfigurationError(
                    f"operator bank {type(base).__name__} cannot carry the periodic "
                    f"wrap declared on component {c.name!r}: its {op_name} takes no "
                    f"bcs argument. A bank used under a periodic declaration must "
                    f"accept bcs= so the wrap for grid axes {list(axes)} can be "
                    f"merged into it."
                )
        return _PeriodicOperatorView(domain=base.domain, inner=base, periodic_axes=axes)

    comp_tag_ops = {c.name: _residual_ops(c) for c in comps}
    comp_plain_ops = {c.name: resolve_operator(c.domain, c.operator_factory) for c in comps}
    comp_domains = {c.name: c.domain for c in comps}
    # Components whose grid is scaled by a declared parameter (``spec.geometry``)
    # keep only their unit grid here; every evaluation reads the parameter and
    # rebuilds their banks on the scaled grid (``_evaluation_banks``). Captured
    # host geometry of theirs is multiplied by the matching power of the scale.
    scaled_comps = {c.name: c for c in comps if getattr(c, "geometry_scale", None) is not None}
    scale_dimension = {
        name: _METRIC_DIMENSION[c.domain.coord_system] for name, c in scaled_comps.items()
    }

    def _evaluation_banks(boundary_context):
        """Banks, domains and scale values of one evaluation.

        Unscaled components keep the banks built here once; a scaled component
        gets its grid at the parameter's current value.
        """
        if not scaled_comps:
            return comp_tag_ops, comp_plain_ops, comp_domains, {}
        tag_ops, plain_ops, domains = dict(comp_tag_ops), dict(comp_plain_ops), dict(comp_domains)
        lengths = {}
        for name, c in scaled_comps.items():
            length = _geometry_scale_value(boundary_context, c)
            grid = c.domain.scaled(length)
            domains[name] = grid
            plain_ops[name] = resolve_operator(grid, c.operator_factory)
            tag_ops[name] = _residual_ops(c, grid)
            lengths[name] = length
        return tag_ops, plain_ops, domains, lengths

    per_comp_vars = {c.name: tuple(c.variables) for c in comps}
    total = int(layout.total_size)
    residual_scale = {k: float(v) for k, v in dict(getattr(eqset, "residual_scale", {})).items()}
    # Flat equation order matches ``LoweredProblem._equations``
    # (``tuple(eq for comp in components for eq in comp.equations)``).
    eq_index_to_comp_name = tuple(c.name for c in comps for _eq in c.equations)

    # ------------------------------------------------------------------
    # Bordered-global row restriction (host-side, static). Flags are
    # aligned with the runtime tuples' order: flat equations (component
    # declaration order), ``EquationSet.families`` order, and
    # ``EquationSet.constraints`` order.
    restricted = restrict_rows_to is not None
    restrict_rows_to = frozenset(restrict_rows_to) if restricted else None

    def _eq_out_keys(cname, eq):
        outs = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
        return tuple((cname, v.name) for v in outs)

    flat_eq_keys = tuple(_eq_out_keys(c.name, eq) for c in comps for eq in c.equations)
    if restricted:
        active_eq_flags = tuple(any(k in restrict_rows_to for k in keys) for keys in flat_eq_keys)
        active_fam_flags = tuple(
            any((fam.name, v.name) in restrict_rows_to for v in fam.member_variables)
            for fam in eqset.families
        )
        active_row_offsets = frozenset(int(layout.offset_of(k)) for k in restrict_rows_to)
    else:
        active_eq_flags = tuple(True for _keys in flat_eq_keys)
        active_fam_flags = tuple(True for _fam in eqset.families)
        active_row_offsets = None
    # Constraint targets are callbacks and run on the dimensional clock (the same
    # convention as controls, Dirichlet values, and the boundary callables the
    # extractors wrap); the residual kernel hands this factor to the
    # constraint dispatcher.
    from .bcs import _callback_time_ref

    callback_time_ref = _callback_time_ref(eqset)
    # Direct-control reach: the static name tuple every closure surface
    # resolves against — q.control in equations and constraint closures,
    # ctx.controls in nodal flux closures, the face view in face fluxes.
    direct_control_names = tuple(
        str(ctl.name)
        for ctl in (getattr(eqset, "controls", ()) or ())
        if getattr(ctl, "binds_to", None) is None
    )
    if restrict_constraints_to is not None:
        active_constraint_flags = tuple(
            c.name in frozenset(restrict_constraints_to) for c in eqset.constraints
        )
    else:
        active_constraint_flags = tuple(True for _c in eqset.constraints)
    held_current_names = frozenset(hold_residual_currents or ())
    hold_current_flags = tuple(c.name in held_current_names for c in eqset.constraints)

    # A control authored with ``binds_to=<global>`` DRIVES that global: the
    # residual overwrites the whole closing row of the constraint
    # that closes it with ``control fn(t) - lambda``.  Whatever that
    # constraint's own integrand evaluates to is discarded, so its row is
    # dropped here and its integrand, ctx and measure never enter the trace.
    # A direct control drives no global and so drops no row.
    bound_globals = frozenset(
        str(ctl.binds_to) for ctl in eqset.controls if getattr(ctl, "binds_to", None) is not None
    )
    if bound_globals:
        active_constraint_flags = tuple(
            active and getattr(getattr(c, "closes", None), "name", None) not in bound_globals
            for active, c in zip(active_constraint_flags, eqset.constraints)
        )

    # Host-time GlobalVariable -> (lam_off, dim) map (global-row order).
    global_to_lam_off: dict = {}
    _off_l = 0
    for c in eqset.constraints:
        global_to_lam_off[c.closes] = (_off_l, int(c.dim))
        _off_l += int(c.dim)

    # Per-(comp,var) lookup of every component Variable (by name).
    comp_var_by_name = {(c.name, v.name): v for c in comps for v in c.variables}

    # Pre-resolve each constraint's ``field`` Variable to a host-side
    # ``(offset, size)`` slice, by Python ``is`` against the owning component.
    constraint_field_slices = []
    for c in eqset.constraints:
        field_var = getattr(c, "field", None)
        if field_var is None:
            constraint_field_slices.append(None)
            continue
        owner_name = None
        for comp in comps:
            if any(v is field_var for v in comp.variables):
                owner_name = comp.name
                break
        if owner_name is None:
            constraint_field_slices.append(None)
            continue
        off = layout.offset_of((owner_name, field_var.name))
        constraint_field_slices.append((off, int(field_var.size)))
    constraint_field_slices = tuple(constraint_field_slices)

    # Family residual blocks built once (host-side, in EquationSet.families order).
    # The build-time declared namespaces let the blocks' build-time member
    # evaluations read ``q.p``; each residual call passes its own.
    from .bcs import _boundary_context

    build_context = _boundary_context(tuple(eq for c in comps for eq in c.equations))
    family_blocks = []
    for fam in eqset.families:
        fam_base = layout.offset_of((fam.name, fam.member_variables[0].name))
        block = _build_family_residual(
            fam, fam_base, total_size=total, time_ref=callback_time_ref, context=build_context
        )
        family_blocks.append((fam_base, block))
    fam_index_by_name = {fam.name: i for i, fam in enumerate(eqset.families)}

    # Resolve family couplings ONCE (host-side, by identity).
    family_couplings = _resolve_family_couplings(eqset)
    scatter_couplings = tuple(fc for fc in family_couplings if fc.direction == "scatter")
    gather_couplings = tuple(fc for fc in family_couplings if fc.direction == "gather")

    # ------------------------------------------------------------------
    # Component<->component (NON read-through) Couplings: resolve each
    # endpoint's owning component by identity; the 1:1 SET injection threads the
    # source slice (transformed by ``op.apply``) into the target component's
    # var_view as ``view[cp.target]``.  Family-touching + read-through couplings
    # are filtered out (their own branches below).  ``component_coupling_names``
    # selects the DOF subset from the runtime ``couplings`` arg so the zip with
    # ``coupling_comp_names`` stays aligned (id() is not stable across tracing).
    member_var_ids = {id(v) for fam in eqset.families for v in fam.member_variables}
    _coupling_resolved_comp_names: list = []
    _component_coupling_ids: set = set()
    for cp in eqset.couplings:
        if id(cp.source) in member_var_ids or id(cp.target) in member_var_ids:
            continue  # family scatter/gather — handled by the family branch.
        if getattr(cp, "read_through", False):
            continue  # read-through alias — injected into the var_view below.
        src_comp_name = None
        tgt_comp_name = None
        for comp in comps:
            for v in comp.variables:
                if v is cp.source:
                    src_comp_name = comp.name
                if v is cp.target:
                    tgt_comp_name = comp.name
        if src_comp_name is None or tgt_comp_name is None:
            raise ValueError(
                f"_build_full_residual_fn: coupling {cp.name!r} references "
                f"variables not found by identity in any component "
                f"(source={cp.source.name!r}, target={cp.target.name!r})."
            )
        _coupling_resolved_comp_names.append((src_comp_name, tgt_comp_name))
        _component_coupling_ids.add(cp.name)
    coupling_comp_names = tuple(_coupling_resolved_comp_names)
    component_coupling_names = frozenset(_component_coupling_ids)

    # Scatter specs: (fam_slab_base, member_state, count, tgt_off, tgt_size, op).
    scatter_specs = []
    for fc in scatter_couplings:
        if restricted and (fc.comp_name, fc.comp_var_name) not in restrict_rows_to:
            continue  # scatter lands on an unassembled row — not needed.
        fam = eqset.families[fam_index_by_name[fc.fam_name]]
        fam_slab_base = layout.offset_of((fc.fam_name, fc.source_var_name))
        tgt_off = layout.offset_of((fc.comp_name, fc.comp_var_name))
        tgt_size = int(comp_var_by_name[(fc.comp_name, fc.comp_var_name)].size)
        scatter_specs.append(
            (
                fam_slab_base,
                int(fam.member_state_size),
                int(fam.count),
                tgt_off,
                tgt_size,
                fc.op,
            )
        )
    scatter_specs = tuple(scatter_specs)

    # Gather specs: (blk_i, macro_var, src_off, src_size, count, op).
    gather_specs = []
    for fc in gather_couplings:
        blk_i = fam_index_by_name[fc.fam_name]
        if restricted and not active_fam_flags[blk_i]:
            continue  # only the (skipped) family block consumes this gather.
        fam = eqset.families[blk_i]
        macro_var = comp_var_by_name[(fc.comp_name, fc.comp_var_name)]
        src_off = layout.offset_of((fc.comp_name, fc.comp_var_name))
        gather_specs.append((blk_i, macro_var, src_off, int(fc.source_size), int(fam.count), fc.op))
    gather_specs = tuple(gather_specs)

    # ------------------------------------------------------------------
    # Read-through couplings: NO-DOF MergedView injection.
    # The transferred value is injected into the CONSUMING component's var_view
    # (so its equation reads ``view[alias]`` by name) AND into the constraint
    # MergedView, with zero state DOF / row / IC.  Two directions, resolved
    # host-side by the (multi-component-aware) alias-owner walk:
    #
    #  * family -> component (surface): pull each member's surface DOF off the
    #    family slab (count,) -> ``op.apply`` places at ``at_indices`` ->
    #    (alias_size,).  Carried as a ``gather_surface`` FamilyCoupling.
    #  * component -> component: source is a component variable slice; resolve
    #    the source's owning component then ``op.apply(src_arr, alias_size)``.
    _rt_alias_owner = _read_through_alias_owner(eqset)
    fam_by_name = {fam.name: fam for fam in eqset.families}
    # family -> component (surface) read-through specs:
    #   (consuming_comp, alias_var, fam_slab_base, member_state, count, op, size).
    rt_fc_specs = []
    for fc in family_couplings:
        if fc.direction != "gather_surface":
            continue
        fam = fam_by_name[fc.fam_name]
        fam_slab_base = layout.offset_of((fc.fam_name, fc.source_var_name))
        alias_var = fc.alias
        rt_fc_specs.append(
            (
                fc.comp_name,
                alias_var,
                fam_slab_base,
                int(fam.member_state_size),
                int(fam.count),
                fc.op,
                int(alias_var.size),
            )
        )
    rt_fc_specs = tuple(rt_fc_specs)
    # component -> component read-through specs:
    #   (consuming_comp, alias_var, src_off, src_size, op, alias_size).
    rt_cc_specs = []
    for cp in eqset.couplings:
        if not getattr(cp, "read_through", False):
            continue
        if id(cp.source) in member_var_ids:
            continue  # family-source read-through (gather_surface, above).
        owner = _rt_alias_owner.get(id(cp.target))
        if owner is None:
            raise ValueError(
                f"_build_full_residual_fn: read-through coupling {cp.name!r} "
                f"alias {cp.target.name!r} has no consuming component: it is read "
                "by no equation and names no onto= target owner."
            )
        consuming_comp, alias_var = owner
        src_comp_name = None
        for comp in comps:
            for v in comp.variables:
                if v is cp.source:
                    src_comp_name = comp.name
        if src_comp_name is None:
            raise ValueError(
                f"_build_full_residual_fn: read-through coupling {cp.name!r} "
                f"source {cp.source.name!r} is not found by identity in any "
                "component."
            )
        src_off = layout.offset_of((src_comp_name, cp.source.name))
        rt_cc_specs.append(
            (
                consuming_comp,
                alias_var,
                src_off,
                int(cp.source.size),
                cp.op,
                int(alias_var.size),
            )
        )
    rt_cc_specs = tuple(rt_cc_specs)

    # ------------------------------------------------------------------
    # Boundary-condition device geometry (cached host-side so each residual
    # call does not re-upload host arrays).  Dirichlet DOF indices are runtime
    # ``bc_dof_indices``; flux/robin/traction geometry per the resolved BCs.
    bc_dof_indices = tuple(d.dof_indices for d in bcs.dirichlet)
    dirichlet_callback_geometry = tuple(d.callback_geometry for d in bcs.dirichlet)
    robin_callback_geometry = tuple(r.callback_geometry for r in bcs.robin)
    traction_callback_geometry = tuple(r.callback_geometry for r in bcs.traction)
    dirichlet_callback_projection = []
    for d in bcs.dirichlet:
        if d.callback_geometry is None:
            dirichlet_callback_projection.append(None)
            continue
        cname, fi, _, _, _ = d.callback_geometry
        off = layout.offset_of((cname, d.var.name))
        width = int(np.prod(d.var.shape[1:])) if len(d.var.shape) > 1 else 1
        nodes = (d.dof_indices - off) // width
        incidence = (nodes[:, None, None] == fi[None, :, :]).any(axis=-1).astype(float)
        dirichlet_callback_projection.append(
            jnp.asarray(incidence / incidence.sum(axis=1, keepdims=True))
        )
    dirichlet_callback_projection = tuple(dirichlet_callback_projection)
    flux_geom = tuple(
        (
            f.comp_name,
            f.permitted_names,
            f.var_offset,
            f.var_size,
            f.var_shape,
            f.is_fem,
            jnp.asarray(f.face_indices, dtype=jnp.int32) if f.face_indices is not None else None,
            jnp.asarray(f.face_lengths) if f.face_lengths is not None else None,
            jnp.asarray(f.face_normals) if f.face_normals is not None else None,
            jnp.asarray(f.node_indices, dtype=jnp.int32) if f.node_indices is not None else None,
            jnp.asarray(f.face_factor) if f.face_factor is not None else None,
            bool(f.is_face_flux),
            jnp.asarray(f.face_centroids) if f.face_centroids is not None else None,
            bool(getattr(f, "cross_facet", False)),
            jnp.asarray(f.face_weights) if f.face_weights is not None else None,
        )
        for f in bcs.flux
    )
    # The ``residual_scale`` of the row each boundary flux, Robin or traction
    # condition adds to (``None`` when the field has none): the contribution is
    # divided by it exactly as the equation output of that row is, so one scale
    # covers the whole row.
    scale_at_offset = {
        int(off): residual_scale[key]
        for key, off, _size, _shape in layout.entries
        if key in residual_scale
    }
    flux_row_scales = tuple(scale_at_offset.get(int(f.var_offset)) for f in bcs.flux)
    robin_row_scales = tuple(scale_at_offset.get(int(r_.var_offset)) for r_ in bcs.robin)
    traction_row_scales = tuple(scale_at_offset.get(int(t_.var_offset)) for t_ in bcs.traction)
    # Names of declared globals (read by a field-dependent flux's ``face`` view as
    # scalars, distinguishing them from per-facet field face-means).
    # The face view answers globals and resolved direct controls by name,
    # as-is; field names answer with per-facet means.
    flux_global_names = frozenset(g.name for g in eqset.globals) | frozenset(direct_control_names)
    robin_geom = tuple(
        (
            r_.var_offset,
            r_.var_size,
            r_.is_fem,
            jnp.asarray(r_.face_indices, dtype=jnp.int32) if r_.face_indices is not None else None,
            jnp.asarray(r_.face_lengths) if r_.face_lengths is not None else None,
            jnp.asarray(r_.face_normals) if r_.face_normals is not None else None,
            jnp.asarray(r_.node_indices, dtype=jnp.int32) if r_.node_indices is not None else None,
            jnp.asarray(r_.face_factor) if r_.face_factor is not None else None,
            r_.comp_name,
            jnp.asarray(r_.face_weights) if r_.face_weights is not None else None,
        )
        for r_ in bcs.robin
    )
    traction_geom = tuple(
        (
            t_.var_offset,
            t_.var_size,
            t_.var_shape,
            jnp.asarray(t_.face_indices, dtype=jnp.int32),
            jnp.asarray(t_.face_lengths),
            jnp.asarray(t_.face_normals),
            jnp.asarray(t_.face_weights) if t_.face_weights is not None else None,
        )
        for t_ in bcs.traction
    )
    # Symmetry rows: ``(rows, cols, coefficients, constants)`` per condition,
    # imposing ``n·u = 0`` at each tagged node.
    symmetry_geom = tuple(
        (
            jnp.asarray(s.rows, dtype=jnp.int32),
            jnp.asarray(s.cols, dtype=jnp.int32),
            jnp.asarray(s.coefficients),
            jnp.asarray(s.constants),
        )
        for s in bcs.symmetry
    )
    # Resolve each constraint to the component/domain owning its representative
    # field. Integral constraints also carry a host-resolved measure — a nodal
    # surface measure for boundary integrals, the domain's lumped volume measure
    # for volumetric ones — so the geometry enters the computed integral, not
    # just the IR hash.
    from diff_soup.problem._ir.constraint import (
        BoundaryIntegralConstraint,
        DomainIntegralConstraint,
        FluxTotalConstraint,
    )

    from .bcs import _boundary_integral_measure

    # Map each named flux's stable name to its position in flux order
    # (== flux_geom / flux_callables order) so a flux-total closure reuses the
    # SAME resolved flux + facet table that drives the natural boundary condition.
    flux_index_by_name: dict = {}
    flux_static_by_index: dict = {}
    for k, fr in enumerate(bcs.flux):
        if fr.bc_name is not None:
            if fr.bc_name in flux_index_by_name:
                raise ValueError(
                    f"duplicate flux boundary name {fr.bc_name!r}; flux names linked by "
                    "spec.constraint.flux_total(from_bc=...) must be unique"
                )
            flux_index_by_name[fr.bc_name] = k
        flux_static_by_index[k] = fr

    constraint_owner_names = []
    constraint_boundary_measures = []
    constraint_domain_measures = []
    constraint_flux_total_static = []
    for constraint in () if skip_constraints else eqset.constraints:
        field_var = getattr(constraint, "field", None)
        owner = next(
            (
                comp
                for comp in comps
                if field_var is not None and any(v is field_var for v in comp.variables)
            ),
            comps[0],
        )
        constraint_owner_names.append(owner.name)
        if isinstance(constraint, BoundaryIntegralConstraint):
            selector = getattr(constraint, "selector", None)
            if selector is None and getattr(constraint, "tag", None) is not None:
                from diff_soup.domains.boundary_set import tags

                selector = tags(constraint.tag)
            constraint_boundary_measures.append(
                _boundary_integral_measure(
                    owner.domain,
                    selector,
                    constraint_desc=f"constraint {constraint.name!r}",
                )
                if selector is not None
                else None
            )
        else:
            constraint_boundary_measures.append(None)
        if isinstance(constraint, DomainIntegralConstraint):
            # Lumped volume measure of the owning domain: grid cell volumes on a
            # finite-volume / finite-difference grid, ``∫ N_i dV`` on a mesh.
            constraint_domain_measures.append(jnp.asarray(dispatch_lumped_mass(owner.domain)))
        else:
            constraint_domain_measures.append(None)
        if isinstance(constraint, FluxTotalConstraint):
            links = []
            for bc_name in constraint.from_bc:
                idx = flux_index_by_name.get(bc_name)
                if idx is None:
                    available = sorted(flux_index_by_name)
                    raise ValueError(
                        f"flux_total {constraint.name!r}: from_bc names {bc_name!r}, but no "
                        "flux boundary condition declares that name. Author the flux with "
                        "spec.bc.flux(..., name=...). Available flux names: "
                        f"{available}."
                    )
                fr = flux_static_by_index[idx]
                if not fr.is_fem:
                    raise ValueError(
                        f"flux_total {constraint.name!r}: flux {bc_name!r} is not on a tagged "
                        "FEM mesh; a flux-total closure integrates over boundary facets."
                    )
                links.append(
                    (
                        idx,
                        fr.comp_name,
                        jnp.asarray(fr.face_indices, dtype=jnp.int32),
                        jnp.asarray(fr.face_lengths),
                        jnp.asarray(fr.face_normals),
                        jnp.asarray(fr.face_centroids),
                        bool(fr.is_face_flux),
                        tuple(fr.permitted_names),
                        bool(getattr(fr, "cross_facet", False)),
                    )
                )
            constraint_flux_total_static.append(tuple(links))
        else:
            constraint_flux_total_static.append(None)
    constraint_owner_names = tuple(constraint_owner_names)
    constraint_boundary_measures = tuple(constraint_boundary_measures)
    constraint_domain_measures = tuple(constraint_domain_measures)
    constraint_flux_total_static = tuple(constraint_flux_total_static)

    def physics_residual_fn(
        equations,
        dirichlet_value_fns,
        flux_callables,
        robin_callables,
        traction_callables,
        constraints,
        couplings,
        u,
        lam,
        t,
        families=(),
        controls=(),
        control=None,
    ):
        from .bcs import _boundary_context

        boundary_context = _boundary_context(equations)
        tag_ops, plain_ops, domains, lengths = _evaluation_banks(boundary_context)
        resolved_controls = _resolve_direct_controls(
            direct_control_names,
            controls,
            callback_time_ref,
            t,
            control,
            dtype=u.dtype,
        )
        # Per-component var_view (id-keyed) + a flat ``(comp,var)`` view.
        view = layout.unpack(u)
        var_views = {
            cname: {v: view[(cname, v.name)] for v in vars_}
            for cname, vars_ in per_comp_vars.items()
        }
        # Thread globals into EVERY component's var_view as phantom lookups so
        # the component equations / constraint closures read ``view[g]``.
        for g, (off, dim) in global_to_lam_off.items():
            val = lam[off] if dim == 1 else lam[off : off + dim]
            for cv in var_views.values():
                cv[g] = val

        # Component<->component (non read-through) couplings: source slice ->
        # op.apply -> target slot (1:1 SET).  The runtime ``couplings`` arg
        # carries ALL of eqset.couplings; select the DOF subset by name so the
        # zip with ``coupling_comp_names`` stays aligned.
        component_cps = [cp for cp in couplings if cp.name in component_coupling_names]
        for cp, (src_comp, tgt_comp) in zip(component_cps, coupling_comp_names):
            src_arr = var_views[src_comp][cp.source]
            transferred = cp.op.apply(src_arr, target_size=cp.target.size)
            var_views[tgt_comp][cp.target] = transferred

        # Gather couplings: per-member gathered component slice ONCE per coupling.
        member_gathers_by_block: dict = {}
        for blk_i, macro_var, src_off, src_size, count, op in gather_specs:
            src_arr = u[src_off : src_off + src_size]
            gathered = op.apply(src_arr, target_size=count)  # (count,)
            member_gathers_by_block.setdefault(blk_i, {})[macro_var] = gathered

        # Read-through couplings: compute each alias value ONCE per
        # residual call and (a) inject into the consuming component's var_view so
        # its per-equation closures read ``view[alias]`` by name, and (b) stash a
        # ``{VarRef(consuming_comp, alias): value}`` dict + the alias Variable so
        # the constraint MergedView can resolve ``view[alias]`` by id too.  NO
        # state DOF, NO residual row.
        rt_injections_by_ref: dict = {}
        rt_alias_vars: dict = {}
        # component -> component: source slice -> op.apply -> alias value.
        for consuming_comp, alias_var, src_off, src_size, op, alias_size in rt_cc_specs:
            src_arr = u[src_off : src_off + src_size]
            transferred = op.apply(src_arr, target_size=alias_size)
            var_views[consuming_comp][alias_var] = transferred
            ref = VarRef(consuming_comp, alias_var.name)
            rt_injections_by_ref[ref] = transferred
            rt_alias_vars[ref] = alias_var
        # family -> component (surface): pull each member's surface DOF off the
        # family slab -> (count,) -> op.apply places at at_indices -> (alias,).
        for (
            consuming_comp,
            alias_var,
            fam_slab_base,
            member_state,
            count,
            op,
            alias_size,
        ) in rt_fc_specs:
            surf_idx = fam_slab_base + jnp.arange(count) * member_state + op.member_surface_dof
            surface_traces = u[surf_idx]  # (count,)
            transferred = op.apply(surface_traces, target_size=alias_size)
            var_views[consuming_comp][alias_var] = transferred
            ref = VarRef(consuming_comp, alias_var.name)
            rt_injections_by_ref[ref] = transferred
            rt_alias_vars[ref] = alias_var

        f = jnp.zeros((total,), dtype=u.dtype)
        asm_dtype = _assembly_dtype(u.dtype, compute_dtype)
        for i, eq in enumerate(equations):
            if not active_eq_flags[i]:
                continue  # writes no assembled row — never evaluated.
            cname = eq_index_to_comp_name[i]
            dom = domains[cname]
            ctx = _Ctx(tag_ops[cname], resolved_controls)
            if border_probe is not None:
                border_probe.count_equation(getattr(eq, "name", None) or f"equation_{i}")
            r = eq.rhs_fn(var_views[cname], ctx, t)
            if isinstance(eq, MultiEquation):
                _check_multi_equation_return(eq, r)
                for v in eq.vars:
                    if restricted and (cname, v.name) not in restrict_rows_to:
                        continue  # this output's rows are not assembled.
                    arr = jnp.asarray(
                        _apply_boundary_cast(finalize_residual(dom, r[v.name]), asm_dtype)
                    ).reshape(-1)
                    _rs = residual_scale.get((cname, v.name))
                    if _rs is not None:
                        arr = arr / _rs
                    ofs = layout.offset_of((cname, v.name))
                    f = f.at[ofs : ofs + v.size].add(arr)
            else:
                arr = jnp.asarray(
                    _apply_boundary_cast(finalize_residual(dom, r), asm_dtype)
                ).reshape(-1)
                _rs = residual_scale.get((cname, eq.var.name))
                if _rs is not None:
                    arr = arr / _rs
                ofs = layout.offset_of((cname, eq.var.name))
                f = f.at[ofs : ofs + eq.var.size].add(arr)

        # Family residual blocks — ONE batched (R-vmap) assembly per Family,
        # AFTER the per-equation Component loop.  Member equations read
        # galvanostatic globals through the merged-view global slot; member
        # params ride the runtime ``families`` arg (traced leaves, not captured constants).
        if family_blocks:
            member_globals = {
                g.name: (lam[off] if dim == 1 else lam[off : off + dim])
                for g, (off, dim) in global_to_lam_off.items()
            }
            runtime_families = tuple(families)
            for blk_i, (fam_base, block) in enumerate(family_blocks):
                if not active_fam_flags[blk_i]:
                    continue  # writes no assembled row — never evaluated.
                if border_probe is not None:
                    border_probe.count_family(eqset.families[blk_i].name)
                runtime_fam = runtime_families[blk_i] if blk_i < len(runtime_families) else None
                f = block(
                    f,
                    u,
                    t,
                    member_globals,
                    runtime_fam=runtime_fam,
                    member_gathers=member_gathers_by_block.get(blk_i),
                    boundary_context={**boundary_context, "controls": resolved_controls},
                )

        # Scatter couplings (family member surface trace -> component target):
        # extract each member's surface DOF off the family slab, scatter-ADD the
        # per-member contribution into the target component rows (ADD so co-
        # located members ACCUMULATE).  Runs AFTER the family blocks so the slab
        # is the same state the members were assembled from.
        for fam_slab_base, member_state, count, tgt_off, tgt_size, op in scatter_specs:
            surf_idx = fam_slab_base + jnp.arange(count) * member_state + op.member_surface_dof
            surface_traces = u[surf_idx]  # (count,)
            contribution = op.apply(surface_traces, target_size=tgt_size)
            f = f.at[tgt_off : tgt_off + tgt_size].add(contribution.astype(u.dtype))

        # Boundary kernels — applied BEFORE the Dirichlet override so the
        # Dirichlet rows still pin to ``u[bc] - target(t)`` even if a FluxBC and
        # a DirichletBC share a node (DirichletBC wins).  String-keyed
        # view-of-bulk-state for boundary closures.
        owner_views = {
            cname: {key.name: val for key, val in cv.items()} for cname, cv in var_views.items()
        }

        def boundary_value(value_fn, geometry):
            from diff_soup.problem._spec.q_accessor import _BoundaryQ

            from .bcs import _eval_boundary_value

            if geometry is None:
                return value_fn(t)
            cname, fi, normals, centroids, _ = geometry
            if cname in lengths:
                centroids = jnp.asarray(centroids) * lengths[cname]
            q = _BoundaryQ(
                {**owner_views[cname], **resolved_controls},
                flux_global_names,
                fi,
                normals,
                centroids,
                ops=plain_ops[cname],
                controls=resolved_controls,
                t=t,
                **boundary_context,
            )
            return _eval_boundary_value(value_fn, q, t)

        # Slope modes (the residual-form lowering, see ``slope_context``): the
        # boundary kernels and the symmetry and Dirichlet overrides are free of
        # time derivatives. The derivative-action mode excludes them, and the
        # full-residual mode subtracts the kernels, so a residual-form row
        # reads ``residual = derivative_part + free_terms - flux``, as the
        # family member kernel does.
        slopes_active = current_slopes() is not None
        derivative_action_only = slopes_active and not slope_context_includes_residual()
        bc_sign = -1.0 if slopes_active else 1.0

        if (flux_geom or robin_geom or traction_geom) and not derivative_action_only:
            # Owner-aware string-keyed views built AFTER coupling transfer: each
            # component's post-transfer var_view (local fields + coupled-in
            # targets SET above + injected read-through aliases + globals) keyed by
            # NAME.  Per-owner so duplicate field names across components never
            # collide — each flux closure reads its OWN component's state, and the
            # values are post-transfer so a flux can read a coupled-in variable.
            owner_views: dict = {
                cname: {key.name: val for key, val in cv.items()} for cname, cv in var_views.items()
            }
            for kern, fn, row_scale in zip(flux_geom, flux_callables, flux_row_scales):
                (
                    cname_owner,
                    permitted_names,
                    v_off,
                    v_size,
                    v_shape,
                    is_fem,
                    fi,
                    fl,
                    _fn_arr,
                    ni,
                    factor,
                    is_face_flux,
                    face_centroids,
                    cross_facet,
                    fw,
                ) = kern
                if restricted and int(v_off) not in active_row_offsets:
                    continue  # scatters onto unassembled rows only.
                if cname_owner in lengths:
                    # A scaled grid: face_area / cell_volume scales as 1 / R and
                    # a face position as R.
                    factor = factor / lengths[cname_owner]
                    if face_centroids is not None:
                        face_centroids = face_centroids * lengths[cname_owner]
                # Control names answer from the evaluation's resolved values,
                # never from the owner's state view.
                bc_view = {
                    name: owner_views[cname_owner][name]
                    for name in permitted_names
                    if name not in resolved_controls
                }
                if resolved_controls:
                    bc_view = {**bc_view, **resolved_controls}
                if is_face_flux:
                    from .bcs import _facet_local_flux_value, _FaceView  # noqa: PLC0415

                    # ``fi`` holds one row per facet: an element facet's vertices
                    # on a mesh, the single boundary cell on a grid, so the view's
                    # row mean is the right boundary representative for both.
                    if cross_facet:
                        face = _FaceView(
                            bc_view,
                            flux_global_names,
                            fi,
                            _fn_arr,
                            face_centroids,
                            ops=plain_ops[cname_owner],
                            controls=resolved_controls,
                            t=t,
                            **boundary_context,
                        )
                        flux_val = fn(face, t)
                    else:
                        # Facet-local by construction: one facet per call, so a
                        # cross-facet read cannot exist and the facet stencil
                        # declared for this flux is exact.
                        flux_val = _facet_local_flux_value(
                            fn,
                            bc_view,
                            flux_global_names,
                            fi,
                            _fn_arr,
                            face_centroids,
                            t,
                            ops=plain_ops[cname_owner],
                            controls=resolved_controls,
                            **boundary_context,
                        )
                else:
                    flux_val = fn(
                        bc_view,
                        _Ctx(plain_ops[cname_owner], resolved_controls),
                        t,
                    )
                if is_fem:
                    fv = jnp.broadcast_to(jnp.asarray(flux_val), fl.shape)
                    face_contrib = fv * fl
                    contribution = jnp.zeros((v_size,), dtype=u.dtype)
                    contribution = contribution.at[fi.reshape(-1)].add(
                        facet_shares(face_contrib, fi, fw).reshape(-1).astype(u.dtype)
                    )
                else:
                    fv = jnp.broadcast_to(jnp.asarray(flux_val), ni.shape)
                    contribution = jnp.zeros((v_size,), dtype=u.dtype)
                    contribution = contribution.at[ni].add((fv * factor).astype(u.dtype))
                if row_scale is not None:
                    contribution = contribution / row_scale
                f = f.at[v_off : v_off + v_size].add(bc_sign * contribution)

            for kern, fns, callback_geom, row_scale in zip(
                robin_geom, robin_callables, robin_callback_geometry, robin_row_scales
            ):
                (v_off, v_size, is_fem, fi, fl, _fn_arr, ni, factor, robin_comp, fw) = kern
                if restricted and int(v_off) not in active_row_offsets:
                    continue  # scatters onto unassembled rows only.
                if robin_comp in lengths:
                    factor = factor / lengths[robin_comp]  # face_area / cell_volume ~ 1 / R
                alpha_fn, beta_fn, value_fn = fns
                alpha_val = jnp.asarray(alpha_fn(t))
                beta_val = jnp.asarray(beta_fn(t))
                value_val = jnp.asarray(boundary_value(value_fn, callback_geom))
                u_slice = u[v_off : v_off + v_size]
                # ``alpha*u + beta*n.grad(u) = value`` prescribes the outward
                # normal derivative ``(value - alpha*u)/beta``, which enters the
                # residual as a boundary flux density positive into the domain —
                # the same sign convention and the same measure the FluxBC
                # branch above uses.  ``beta`` is nonzero: a zero beta states a
                # Dirichlet condition and is rejected when the BC is resolved.
                if is_fem:
                    u_at_faces = facet_average(u_slice, fi, fw)
                    inward = (value_val - alpha_val * u_at_faces) / beta_val
                    per_face = inward * fl
                    contribution = jnp.zeros((v_size,), dtype=u.dtype)
                    contribution = contribution.at[fi.reshape(-1)].add(
                        facet_shares(per_face, fi, fw).reshape(-1).astype(u.dtype)
                    )
                else:
                    u_at_nodes = u_slice[ni]
                    inward = (value_val - alpha_val * u_at_nodes) / beta_val
                    per_node = inward * factor
                    contribution = jnp.zeros((v_size,), dtype=u.dtype)
                    contribution = contribution.at[ni].add(per_node.astype(u.dtype))
                if row_scale is not None:
                    contribution = contribution / row_scale
                f = f.at[v_off : v_off + v_size].add(bc_sign * contribution)

            for kern, value_fn, callback_geom, row_scale in zip(
                traction_geom, traction_callables, traction_callback_geometry, traction_row_scales
            ):
                v_off, v_size, v_shape, fi, fl, _fn_arr, fw = kern
                if restricted and int(v_off) not in active_row_offsets:
                    continue  # scatters onto unassembled rows only.
                n_components = int(v_shape[-1])
                t_vec = jnp.asarray(boundary_value(value_fn, callback_geom))
                t_vec = jnp.broadcast_to(t_vec, (fi.shape[0], n_components))
                n_nodes = v_size // n_components
                per_face = t_vec * fl[:, None]
                contribution = jnp.zeros((n_nodes, n_components), dtype=u.dtype)
                contribution = contribution.at[fi.reshape(-1)].add(
                    facet_shares(per_face, fi, fw).reshape(-1, n_components).astype(u.dtype)
                )
                if row_scale is not None:
                    contribution = contribution / row_scale
                f = f.at[v_off : v_off + v_size].add(bc_sign * contribution.reshape(-1))

        # Symmetry override: replace the row of each tagged node's dominant
        # normal component with the normal projection ``n·u``, leaving the
        # node's other rows on their assembled physics (free tangential
        # traction).  Applied BEFORE the Dirichlet override so an explicit pin
        # on the same DOF still wins.
        for sym_rows, sym_cols, sym_coeffs, sym_consts in (
            () if derivative_action_only else symmetry_geom
        ):
            projection = jnp.sum(sym_coeffs.astype(u.dtype) * u[sym_cols], axis=1)
            f = f.at[sym_rows].set(projection + sym_consts.astype(u.dtype))

        # Dirichlet override: pin each tagged DOF to ``u[bc] - target(t)``.
        for dof_idx, value_fn, geometry, projection in zip(
            () if derivative_action_only else bc_dof_indices,
            dirichlet_value_fns,
            dirichlet_callback_geometry,
            dirichlet_callback_projection,
        ):
            val = jnp.asarray(boundary_value(value_fn, geometry), dtype=u.dtype)
            if projection is not None and val.ndim > 0:
                val = projection @ jnp.broadcast_to(val.reshape(-1), (projection.shape[1],))
            val_b = jnp.broadcast_to(val, dof_idx.shape)
            f = f.at[dof_idx].set(u[dof_idx] - val_b)

        if not constraints:
            return f, jnp.zeros((0,), dtype=u.dtype)

        # Owner-aware merged view for the constraint closures, spanning ALL
        # components (component vars by id, globals by name + id).  Built from
        # the runtime ``equations`` so cache-rebuilt Variable identities line up
        # with the user's lambdas.
        by_ref: dict = {}
        var_id_to_ref: dict = {}
        ref_to_var: dict = {}
        for i, eq in enumerate(equations):
            cname = eq_index_to_comp_name[i]
            eq_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
            for v in eq_vars:
                ref = VarRef(cname, v.name)
                var_id_to_ref[id(v)] = ref
                ref_to_var.setdefault(ref, v)
                if ref not in by_ref:
                    by_ref[ref] = view[(cname, v.name)]
        for cname, vars_ in per_comp_vars.items():
            for v in vars_:
                ref = VarRef(cname, v.name)
                if ref not in by_ref:
                    by_ref[ref] = view[(cname, v.name)]
                    ref_to_var.setdefault(ref, v)
        # Component<->component coupling targets: read the post-coupling slot.
        for cp, (src_comp, tgt_comp) in zip(component_cps, coupling_comp_names):
            tgt_ref = VarRef(tgt_comp, cp.target.name)
            var_id_to_ref[id(cp.target)] = tgt_ref
            ref_to_var.setdefault(tgt_ref, cp.target)
            by_ref[tgt_ref] = var_views[tgt_comp][cp.target]
        # Read-through aliases: inject the NO-DOF transferred values by VarRef so
        # constraints can read them by name too.  Register each alias Variable's
        # id so ``view[alias]`` resolves by id-exact match.
        for ref, value in rt_injections_by_ref.items():
            by_ref[ref] = value
            alias_var = rt_alias_vars.get(ref)
            if alias_var is not None:
                var_id_to_ref[id(alias_var)] = ref
                ref_to_var.setdefault(ref, alias_var)
        globals_by_name: dict = {}
        global_id_to_name: dict = {}
        name_to_global: dict = {}
        for c in constraints:
            g = getattr(c, "closes", None)
            if g is None or not isinstance(g, GlobalVariable):
                continue
            if g.name in globals_by_name or g not in global_to_lam_off:
                continue
            off, dim = global_to_lam_off[g]
            globals_by_name[g.name] = lam[off] if dim == 1 else lam[off : off + dim]
            global_id_to_name[id(g)] = g.name
            name_to_global.setdefault(g.name, g)
        merged_view = MergedView(
            by_ref=by_ref,
            var_id_to_ref=var_id_to_ref,
            ref_to_var=ref_to_var,
            globals_by_name=globals_by_name,
            global_id_to_name=global_id_to_name,
            name_to_global=name_to_global,
        )
        # Owner-aware name-keyed nodal views (post-coupling) so a flux-total
        # closure rebuilds the SAME ``face`` view its natural BC used.
        nodal_by_owner = {
            cname: {key.name: val for key, val in cv.items()} for cname, cv in var_views.items()
        }
        parts = []
        off_l = 0
        for i, c in enumerate(constraints):
            dim_c = int(c.dim)
            if not active_constraint_flags[i]:
                # Either this constraint's derivative rows come from the
                # scatter-sum shortcut rather than this kernel, or a bound
                # control replaces its closing row outright — either way its
                # residual is never read, so it is not evaluated.
                parts.append(jnp.zeros((dim_c,), dtype=u.dtype))
                off_l += dim_c
                continue
            if border_probe is not None:
                border_probe.count_constraint(getattr(c, "name", None) or f"constraint_{i}")
            lam_slice = lam[off_l : off_l + dim_c]
            flux_total_link = None
            static_link = constraint_flux_total_static[i]
            if static_link is not None:
                flux_total_link = tuple(
                    (
                        _face_signature_flux(
                            flux_callables[idx],
                            is_face_flux_link,
                            nodal_by_owner[comp_name_link],
                            permitted_link,
                            plain_ops[comp_name_link],
                            resolved_controls,
                            boundary_context,
                        ),
                        fi_link,
                        fl_link,
                        normals_link,
                        centroids_link,
                        {**nodal_by_owner[comp_name_link], **resolved_controls}
                        if resolved_controls
                        else nodal_by_owner[comp_name_link],
                        flux_global_names,
                        # A facet-local face flux integrates exactly as it
                        # scatters: one facet per evaluation.
                        bool(is_face_flux_link) and not cross_facet_link,
                    )
                    for (
                        idx,
                        comp_name_link,
                        fi_link,
                        fl_link,
                        normals_link,
                        centroids_link,
                        is_face_flux_link,
                        permitted_link,
                        cross_facet_link,
                    ) in static_link
                )
            owner_name = constraint_owner_names[i]
            boundary_measure = constraint_boundary_measures[i]
            domain_measure = constraint_domain_measures[i]
            if owner_name in lengths:
                # A scaled grid: a boundary measure (face area) scales as R**d,
                # a domain measure (cell volume) as R**(d + 1).
                length, d = lengths[owner_name], scale_dimension[owner_name]
                if boundary_measure is not None:
                    boundary_measure = dataclasses.replace(
                        boundary_measure,
                        nodal_measure=jnp.asarray(boundary_measure.nodal_measure) * length**d,
                    )
                if domain_measure is not None:
                    domain_measure = domain_measure * length ** (d + 1)
            parts.append(
                _eval_constraint_residual(
                    c,
                    merged_view,
                    _Ctx(plain_ops[owner_name], resolved_controls),
                    lam_slice,
                    t,
                    f=f,
                    field_slice=constraint_field_slices[i],
                    boundary_measure=boundary_measure,
                    domain_measure=domain_measure,
                    flux_total_link=flux_total_link,
                    control=control,
                    time_ref=callback_time_ref,
                    hold_residual_current=hold_current_flags[i],
                ).astype(u.dtype)
            )
            off_l += dim_c
        g = jnp.concatenate(parts)
        return f, g

    return physics_residual_fn
