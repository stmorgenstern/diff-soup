"""Lowering analysis provider: builds residual/Jacobian/sparsity/mass kernels.

``lower()`` (in ``diff_soup.problem``) builds a ``LoweredProblem`` from an
``EquationSet`` and calls [analyze][diff_soup.problem._lowering.provider.analyze]
for the residual / Jacobian / sparsity / mass-diagonal kernels, which this module
builds from the ``diff_soup.domains.operators`` bank. ``lower()`` imports it on
first call, since this module imports the IR that ``lower()`` belongs to.

Two branches, selected by ``_has_family_layer``:

* the **slim path** for a single-component equation set with no BCs, families,
  couplings, globals, or constraints — produces the residual/Jacobian kernels
  for simple single-domain problems efficiently; and
* the **full path** — multi-component physics + boundary conditions + families
  + scatter / gather / read-through couplings + globals + constraints
  (galvanostatic Schur arm) — for multi-physics and multi-domain problems.

The residual-row scaling verb (``EquationSet.residual_scale``) and the
internal-time non-dim mass division (``Scales.time_ref``) are applied on both
paths.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._precision import Precision
from diff_soup.linear import SparseTopology
from diff_soup.problem._ad.pattern import dedup_coo
from diff_soup.problem._ir.derivative_ir import capture_forms
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.lower import _LoweringAnalysis
from diff_soup.problem._ir.variable import GlobalVariable
from diff_soup.problem._lowering.full_jacobian import _build_full_jac_blocks_fn
from diff_soup.problem._lowering.full_residual import _build_full_residual_fn
from diff_soup.problem._lowering.provider_context import _Ctx, _geometry_scale_value
from diff_soup.problem._lowering.residual_fn import _build_jac_blocks_fn, _build_residual_fn

from .family_validate import _validate_families
from .layout import _build_global_layout_entries, _layout
from .mass import _mass_diagonal as _mass_diagonal_full
from .operator_resolve import resolve_operator
from .sparsity import _sparsity as _sparsity_full
from .stencil_infer import _infer_stencil_offsets, _walked_stencil_offsets


def _has_family_layer(eqset: EquationSet) -> bool:
    """Whether ``eqset`` requires the full (N-component) lowering path.

    The full path handles ANY of: multiple components, boundary conditions,
    families, couplings, globals, or constraints (including the galvanostatic
    Schur arm).  The slim path applies only to a SINGLE component with none of
    these features.
    """
    if len(eqset.components) != 1:
        return True
    if eqset.components[0].bcs:
        return True
    return bool(
        eqset.families
        or eqset.couplings
        or eqset.globals
        or eqset.constraints
        or getattr(eqset, "controls", ())
    )


def _build_layout(eqset: EquationSet):
    """Block layout entries + per-key offset map for a single component.

    Returns ``(entries, offset_of, total_size)`` where ``entries`` is the
    ``((comp, var), off, size, shape)`` tuple recorded on ``LoweredProblem``.
    """
    comp = eqset.components[0]
    entries = []
    off = 0
    for var in comp.variables:
        entries.append(((comp.name, var.name), off, var.size, var.shape))
        off += var.size
    offset_of = {k: o for k, o, _s, _sh in entries}
    return tuple(entries), offset_of, off


def _mass_diagonal(eqset: EquationSet) -> jax.Array:
    """Assemble the diagonal mass for a single same-domain component.

    FVM rows are already volume-normalised by the kernels (``I du/dt = f``), so
    the differential mass is ``ones``; FEM rows carry the consistent / lumped
    mass.  Algebraic rows are exactly zero.  A differential field carrying a
    residual-row scale ``S`` has its mass slice divided by ``S`` as well, since
    the row scaling multiplies ``M du/dt = f`` through by ``1/S``: both sides
    move, so the trajectory is untouched and only the conditioning changes.
    Algebraic rows stay exactly zero under it.  The whole differential diagonal
    is divided by ``Scales.time_ref`` (internal-time convention ``t_phys =
    time_ref * t_int``), defaulting to 1.0 (identity).

    Raises:
        ValueError: when a ``differential`` variable's assembled mass slice is
            all zero (degenerate: declare it ``mass='algebraic'`` if you mean
            ``0=f``, or supply a domain with positive lumped/consistent mass).
    """
    comp = eqset.components[0]
    is_fem = comp.domain.discretization_type == "fem"
    if is_fem:
        if hasattr(comp.domain, "lumped_mass"):
            weight = jnp.asarray(comp.domain.lumped_mass())
        else:
            weight = resolve_operator(comp.domain, comp.operator_factory).lumped_mass()
    else:
        weight = jnp.ones(comp.domain.num_nodes)
    # ``(component_name, var_name) -> factor``: the same keys the residual rows
    # are divided by, so both sides of a scaled row move together.
    res_scale = {k: float(v) for k, v in dict(getattr(eqset, "residual_scale", {})).items()}
    parts = []
    for var in comp.variables:
        if var.mass == "differential":
            if var.size == comp.domain.num_nodes:
                slice_ = weight
            else:
                slice_ = jnp.broadcast_to(weight[..., None], var.shape).reshape(-1)
            scale = res_scale.get((comp.name, var.name))
            if scale is not None:
                slice_ = slice_ / scale
            if np.all(np.asarray(slice_) == 0):
                raise ValueError(
                    f"variable {var.name!r} in component {comp.name!r} is "
                    f"declared mass='differential' but its assembled mass "
                    f"diagonal is all zero, which is degenerate (really "
                    f"algebraic). Declare it mass='algebraic' if you mean 0=f, "
                    f"or supply a domain with positive (lumped/consistent) mass."
                )
            parts.append(slice_)
        else:
            parts.append(jnp.zeros((var.size,), dtype=weight.dtype))
    scales = getattr(eqset, "scales", None)
    time_ref = float(getattr(scales, "time_ref", 1.0)) if scales is not None else 1.0
    return jnp.concatenate(parts) / time_ref


def _build_observe_fn(eqset: EquationSet, offset_of=None, per_vars=None):  # noqa: ARG001
    """Build the observe() closure (``None`` when no observations).

    Each observation evaluates with the operator bank and grid of its ``owner``
    component, which the build records on the observation node; the unit bank
    of every owner is resolved once here, and a bank whose grid is scaled by a
    declared parameter is rescaled at the parameter's current value on each
    call.  The closure evaluates exactly the ``observations`` it receives, so a
    subset of the declared observations evaluates only that subset.

    The view is built across ALL components from the family-aware ``_layout``
    (member-major), so the closure is correct for both the slim single-component
    path (``offset_of``/``per_vars`` args are unused — the unpack is identical)
    and the multi-component path.

    The bordered globals are read from ``lam_si`` in the global-row order the
    residual and
    [_build_global_layout_entries][diff_soup.problem._lowering.layout._build_global_layout_entries]
    use, so an observation reads ``view[global]`` exactly as the residual
    closures do.  An observation that reads a global when no ``lam_si`` was
    passed raises a
    [ConfigurationError][diff_soup._errors.ConfigurationError] naming the
    missing argument, never a bare ``KeyError``.
    """
    if not eqset.observations:
        return None
    from diff_soup._errors import ConfigurationError
    from diff_soup.problem._ir._merged_view import MergedView
    from diff_soup.problem._ir.var_ref import UnregisteredGlobalError, VarRef

    comps = tuple(eqset.components)
    comp_by_name = {c.name: c for c in comps}
    layout = _layout(eqset)
    unit_ops: dict = {}

    def unit_ops_of(cname):
        if cname not in unit_ops:
            c = comp_by_name[cname]
            unit_ops[cname] = resolve_operator(c.domain, c.operator_factory)
        return unit_ops[cname]

    # Each observation evaluates with the operator bank of the component it
    # names as its owner, so a stencil and spacing never cross grids.
    owner_of_obs = {
        obs.name: getattr(obs, "owner", "") or comps[0].name for obs in eqset.observations
    }
    for cname in {comps[0].name, *owner_of_obs.values()}:
        unit_ops_of(cname)
    per_comp_vars = {c.name: tuple(c.variables) for c in comps}

    def observe(equations, constraints, observations, u, t, lam_si=None):  # noqa: ARG001
        ops_by_owner = dict(unit_ops)
        for cname in ops_by_owner:
            c = comp_by_name[cname]
            if getattr(c, "geometry_scale", None) is not None:
                # The observed grid is scaled by a declared parameter: observe
                # on the grid at the parameter's current value.
                from .bcs import _boundary_context

                length = _geometry_scale_value(_boundary_context(equations), c)
                ops_by_owner[cname] = resolve_operator(c.domain.scaled(length), c.operator_factory)
        view = layout.unpack(u)
        # Bordered globals in global-row order (one closing row per global).
        globals_by_name: dict = {}
        global_id_to_name: dict = {}
        name_to_global: dict = {}
        if lam_si is not None:
            lam = jnp.asarray(lam_si)
            expected = sum(int(c.dim) for c in constraints)
            if lam.ndim != 1 or int(lam.shape[0]) != expected:
                raise ConfigurationError(
                    f"observe() received lam_internal with shape {tuple(lam.shape)}, but the "
                    f"problem's bordered globals form a vector of {expected} entries "
                    "(see global_layout()); pass the solution's lambdas at that time."
                )
            off_l = 0
            for c in constraints:
                dim = int(c.dim)
                g = getattr(c, "closes", None)
                if isinstance(g, GlobalVariable) and g.name not in globals_by_name:
                    globals_by_name[g.name] = lam[off_l] if dim == 1 else lam[off_l : off_l + dim]
                    global_id_to_name[id(g)] = g.name
                    name_to_global[g.name] = g
                off_l += dim
        by_ref: dict = {}
        var_id_to_ref: dict = {}
        ref_to_var: dict = {}
        eq_index_to_comp_name = tuple(c.name for c in comps for _eq in c.equations)
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
        merged_view = MergedView(
            by_ref=by_ref,
            var_id_to_ref=var_id_to_ref,
            ref_to_var=ref_to_var,
            globals_by_name=globals_by_name,
            global_id_to_name=global_id_to_name,
            name_to_global=name_to_global,
        )
        out = {}
        for obs in observations:
            ctx = _Ctx(ops_by_owner[owner_of_obs.get(obs.name, comps[0].name)])
            try:
                out[obs.name] = obs.fn(merged_view, ctx, t)
            except UnregisteredGlobalError as exc:
                if lam_si is not None:
                    raise
                raise ConfigurationError(
                    f"observation {obs.name!r} reads the bordered global "
                    f"{exc.name!r}, but observe() was called without the "
                    "bordered-global vector; pass lam_internal (the solution's "
                    "lambdas at that time, in global_layout() order) so the "
                    "globals are in the view."
                ) from exc
        return out

    return observe


class _ControlsEvaluator:
    """Evaluate registered controls; carries the compiled control bindings.

    Calling the evaluator with the runtime ``controls`` tuple and an internal
    time ``t`` returns ``{name: value}`` for every control, direct or
    derived-quantity — the public ``LoweredProblem.controls`` surface.
    Controls are authored in dimensional units, so each ``fn`` is called at ``time_ref * t``
    and its dimensional return value is passed back unchanged.  ``bindings`` is the
    compiled
    [ControlBindingSet][diff_soup.problem._lowering.control_bindings.ControlBindingSet]
    — the operational form of the declarations: the driven algebraic-global
    slice ``(lam_offset, dim)`` per derived control, the direct controls'
    names, and the ``bound_values`` / ``internal_values`` evaluation surface
    the residual reads.
    """

    __slots__ = ("bindings", "time_ref")

    def __init__(self, bindings, time_ref: float = 1.0) -> None:
        self.bindings = bindings
        self.time_ref = float(time_ref)

    def __call__(self, controls, t):
        t_si = t if self.time_ref == 1.0 else self.time_ref * t
        return {ctl.name: ctl.fn(t_si) for ctl in controls}


def _build_controls_fn(eqset: EquationSet):
    """Build the controls() evaluator (``None`` when no controls).

    Compiling the control bindings here (not at first use) surfaces a
    ``binds_to`` declaration error at lowering time, and resolves each driven
    global to its algebraic-global slice once per lowered problem.
    """
    if not eqset.controls:
        return None
    from .control_bindings import compile_control_bindings

    return _ControlsEvaluator(compile_control_bindings(eqset), float(eqset.scales.time_ref))


def _initial_lambda(eqset: EquationSet, dtype) -> jax.Array:
    """Concatenated initial bordered-global vector (one slice per global).

    Each differential equation or algebraic constraint closes a
    [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable] whose
    ``init`` seeds the slice; the slices line up with
    ``_build_global_layout_entries``'s global-row order.
    """
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    if not eqset.constraints:
        return jnp.zeros((0,), dtype=dtype)
    parts = []
    for c in eqset.constraints:
        v = jnp.asarray(c.closes.init).reshape(-1)
        if int(v.size) != int(c.dim):
            v = jnp.broadcast_to(v.reshape(()), (c.dim,))
        parts.append(v.astype(dtype))
    return jnp.concatenate(parts)


def _component_form_cells(eqset: EquationSet):
    """The derivative-form cell of every component equation that carries one."""
    for comp in eqset.components:
        for eq in comp.equations:
            cell = getattr(eq.rhs_fn, "_derivative_form", None)
            if cell is not None:
                yield cell


def _capture_component_forms(eqset: EquationSet, probe_f, state_size: int, n_lam: int, dtype):
    """Run the build-time capture pass over the component equations.

    A component equation authored with ``q.dt`` records its derivative
    structure only inside this pass (see
    [capture_forms][diff_soup.problem._ir.derivative_ir.capture_forms]), the
    same pass family member equations go through. It evaluates the probe
    residual once, eagerly, at a generic positive state (the family pass's
    convention), on a grid and on a mesh alike.
    """
    dtype = jax.dtypes.canonicalize_dtype(dtype)

    def evaluate():
        n_u, n_l = max(int(state_size), 1), max(int(n_lam), 1)
        u = jnp.linspace(0.75, 1.25, n_u).astype(dtype)[: int(state_size)]
        lam = jnp.linspace(0.75, 1.25, n_l).astype(dtype)[: int(n_lam)]
        probe_f(u, lam, jnp.asarray(0.0, dtype=dtype))

    capture_forms(_component_form_cells(eqset), evaluate)


def _analyze_family_layer(
    eqset: EquationSet, compute_dtype, probe_stencils: bool = True
) -> _LoweringAnalysis:
    """Host analysis for the multi-component family / coupling / BC path.

    Uses the family-aware [_layout][diff_soup.problem._lowering.layout._layout]
    (member-major slabs after components), the multi-component family
    self-block + cross-block `_sparsity_full`, the family mass-diagonal
    branch (`_mass_diagonal_full`), the resolved boundary conditions
    ([_bc_dofs][diff_soup.problem._lowering.bcs._bc_dofs]: Dirichlet pin rows + mass-zero +
    topology ``bc_indices`` + FEM RobinBC face-stencil pairs + symmetry
    normal-projection rows and their cross-component columns), and the full
    residual / Jacobian builders (N-component physics + BCs + family residual +
    scatter/gather/read-through couplings + constraint Schur ``B``/``C``/``D``).
    """
    from .bcs import (
        _bc_dofs,
        _extract_flux_callables,
        _extract_robin_callables,
        _extract_traction_callables,
        _robin_face_stencil_pairs,
        _symmetry_row_column_pairs,
    )

    layout = _layout(eqset)
    total = int(layout.total_size)
    bcs = _bc_dofs(eqset, layout)
    # Infer per-block stencil reach BEFORE declaring the topology.  The probe
    # residual is built WITHOUT the resolved boundary conditions: BC row
    # scatters (e.g. a flux closure reading the whole field) have their own
    # dedicated cross-block declarations, and letting them contaminate the
    # probe would widen — or flip dense — the interior stencil blocks.
    # Periodic wraps are unaffected (they are applied inside the operator
    # calls, not as BC row scatters), and Dirichlet rows probed un-pinned
    # only ever ADD structure (a safe superset; the pin itself needs only the
    # always-declared diagonal).
    from .bcs import _ResolvedBCs  # noqa: PLC0415

    probe_residual_fn = _build_full_residual_fn(
        eqset,
        layout,
        compute_dtype,
        _ResolvedBCs(dirichlet=()),
        skip_constraints=True,
    )
    probe_equations = tuple(eq for comp in eqset.components for eq in comp.equations)
    probe_couplings = tuple(eqset.couplings)
    probe_families = tuple(eqset.families)
    probe_controls = tuple(getattr(eqset, "controls", ()) or ())

    def _probe_f(u, lam, t):
        return probe_residual_fn(
            probe_equations,
            (),
            (),
            (),
            (),
            (),
            probe_couplings,
            u,
            lam,
            t,
            probe_families,
            probe_controls,
        )[0]

    measure = _infer_stencil_offsets if probe_stencils else _walked_stencil_offsets
    stencil_offsets = measure(
        eqset,
        layout.offset_of,
        total,
        int(sum(int(c.dim) for c in eqset.constraints)),
        _probe_f,
        compute_dtype,
    )
    _capture_component_forms(
        eqset,
        _probe_f,
        total,
        int(sum(int(c.dim) for c in eqset.constraints)),
        compute_dtype,
    )
    rows, cols = _sparsity_full(
        eqset, layout, resolved_fluxes=bcs.flux, stencil_offsets=stencil_offsets
    )
    # FEM RobinBC weak ``alpha*u`` terms induce boundary-face vertex-vertex
    # off-diagonals, and every symmetry row couples its node's components; the
    # reads-derived sparsity walk sees neither (both empty for DFN / FVM).  Fold
    # them into the topology so the colored assembly captures them.
    rb_rows, rb_cols = _robin_face_stencil_pairs(eqset, layout)
    sym_rows, sym_cols = _symmetry_row_column_pairs(bcs)
    callback_rows, callback_cols = [], []
    from .couplings import _resolve_coupling_endpoints

    coupling_endpoints = _resolve_coupling_endpoints(eqset)
    components = {comp.name: comp for comp in eqset.components}
    for resolved in (*bcs.dirichlet, *bcs.robin, *bcs.traction):
        geometry = resolved.callback_geometry
        if geometry is None:
            continue
        cname, fi, _, _, accessed = geometry
        comp = components[cname]
        if hasattr(resolved, "dof_indices"):
            target_rows = np.asarray(resolved.dof_indices)
        else:
            width = int(np.prod(getattr(resolved, "var_shape", (1,))[1:]))
            target_rows = resolved.var_offset + (
                np.unique(fi)[:, None] * width + np.arange(width)
            ).reshape(-1)
        for name in set(value.removeprefix("nodal:") for value in accessed):
            endpoint = coupling_endpoints.get((cname, name))
            if endpoint is not None:
                source_comp, source_var, size, _ = endpoint
                offset = layout.offset_of((source_comp, source_var))
            else:
                variable = next((var for var in comp.variables if var.name == name), None)
                if variable is None:
                    continue
                offset, size = layout.offset_of((cname, name)), variable.size
            callback_rows.append(np.repeat(target_rows, size))
            callback_cols.append(np.tile(offset + np.arange(size), target_rows.size))
    bc_pair_rows = np.concatenate([rb_rows, sym_rows, *callback_rows])
    bc_pair_cols = np.concatenate([rb_cols, sym_cols, *callback_cols])
    if bc_pair_rows.size:
        rows, cols, _ = dedup_coo(
            np.concatenate([rows, bc_pair_rows]),
            np.concatenate([cols, bc_pair_cols]),
            total,
            first_seen=True,
        )
    bc_indices = None
    identity_pins = [d.dof_indices for d in bcs.dirichlet if d.callback_geometry is None]
    if identity_pins:
        bc_indices = np.concatenate(identity_pins).astype(np.int32)
    topo = SparseTopology(rows=rows, cols=cols, n_dofs=total, bc_indices=bc_indices)
    mass = _mass_diagonal_full(eqset, layout)
    # Dirichlet DOFs become algebraic identity rows ``u[bc] - target(t)`` (mass
    # zeroed).  A symmetry row is algebraic in the same way — it states
    # ``n·u = 0``, not a rate — but it is NOT a Dirichlet row: its Jacobian
    # entries are the normal's components, so it stays out of the topology's
    # ``bc_indices`` (which the linear backends replace with an identity row).
    for d in bcs.dirichlet:
        mass = mass.at[d.dof_indices].set(0.0)
    for s in bcs.symmetry:
        mass = mass.at[s.rows].set(0.0)
    residual_fn = _build_full_residual_fn(eqset, layout, compute_dtype, bcs)
    jac_fn = _build_full_jac_blocks_fn(
        eqset, residual_fn, topo, total, compute_dtype=compute_dtype, bcs=bcs
    )
    # Observe / controls reuse the multi-component closures (vars by name);
    # both return ``None`` when the eqset declares none (the gate fixtures).
    observe_fn = _build_observe_fn(eqset, None, None)
    controls_fn = _build_controls_fn(eqset)
    constraint_dim = int(sum(int(c.dim) for c in eqset.constraints))
    initial_lambda = _initial_lambda(eqset, compute_dtype)
    alg_indices = tuple(int(i) for i in np.where(np.asarray(mass) == 0.0)[0].tolist())
    bc_dof_indices_static = tuple(
        tuple(int(i) for i in d.dof_indices.tolist()) for d in bcs.dirichlet
    )
    return _LoweringAnalysis(
        layout_entries=tuple(layout.entries),
        global_layout_entries=_build_global_layout_entries(eqset),
        topo=topo,
        mass_diagonal=mass,
        constraint_dim=constraint_dim,
        state_size=total,
        alg_indices=alg_indices,
        bc_dof_indices_static=bc_dof_indices_static,
        dirichlet_value_fns=tuple(d.value_fn for d in bcs.dirichlet),
        flux_bc_callables=_extract_flux_callables(eqset),
        robin_bc_callables=_extract_robin_callables(eqset),
        traction_bc_callables=_extract_traction_callables(eqset),
        initial_lambda_si=initial_lambda,
        residual_fn=residual_fn,
        jac_blocks_fn=jac_fn,
        observe_fn=observe_fn,
        controls_fn=controls_fn,
    )


def analyze(
    eqset: EquationSet, *, precision: Optional[Precision] = None, probe_stencils: bool = True
) -> _LoweringAnalysis:
    """Run the host analysis pass for ``lower()`` (the lowering provider).

    Builds the layout, sparse topology, mass diagonal, residual / Jacobian /
    observe / controls kernels for the slim single-component path OR the full
    multi-component path with families, couplings, boundary conditions, and
    galvanostatic constraints.

    Args:
        eqset: The
            [EquationSet][diff_soup.problem._ir.equation_set.EquationSet] to
            analyse.
        precision: Dtype policy; its ``compute_dtype`` drives the boundary-cast
            at closure entry (default fp64).
        probe_stencils: Whether to measure each block's stencil at probe states, or else
            read it from the residual's tangent map.

    Returns:
        The
        [_LoweringAnalysis][diff_soup.problem._ir.lower._LoweringAnalysis]
        consumed by
        ``lower._assemble_lowered_problem``.

    Raises:
        NotImplementedError: family<->family (many:many) coupling is out of
            scope; use separate families or an explicit component bridge.
        ValueError: an illegal family-scatter coupling (at= range / injectivity).
    """
    # Run family validation before building any kernels.  It also covers an
    # EquationSet built directly from IR nodes rather than by spec.build(), so
    # an invalid coupling is reported here with a clear diagnostic.
    _validate_families(eqset)
    precision = Precision() if precision is None else precision
    compute_dtype = precision.compute_dtype

    if _has_family_layer(eqset):
        return _analyze_family_layer(eqset, compute_dtype, probe_stencils)

    entries, offset_of, total = _build_layout(eqset)
    residual_fn = _build_residual_fn(eqset, offset_of, total, compute_dtype)
    # Infer per-block stencil reach from the assembled residual BEFORE the
    # topology is declared, so a body whose dependency is wider than the
    # grid's radius-1 stencil (e.g. a composed laplacian) is not silently
    # under-declared.  The slim path has no boundary conditions, so the
    # residual is probed as-is.
    slim_equations = tuple(eqset.components[0].equations)

    def _probe_f(u, lam, t):
        return residual_fn(slim_equations, (), (), (), (), (), (), u, lam, t, ())[0]

    measure = _infer_stencil_offsets if probe_stencils else _walked_stencil_offsets
    stencil_offsets = measure(
        eqset,
        lambda key: offset_of[key],
        total,
        0,
        _probe_f,
        compute_dtype,
    )
    _capture_component_forms(eqset, _probe_f, total, 0, compute_dtype)
    rows, cols = _sparsity_full(eqset, _layout(eqset), stencil_offsets=stencil_offsets)
    topo = SparseTopology(rows=rows, cols=cols, n_dofs=total, bc_indices=None)
    mass = _mass_diagonal(eqset)
    jac_fn = _build_jac_blocks_fn(residual_fn, topo, total)
    comp = eqset.components[0]
    per_vars = tuple(comp.variables)
    observe_fn = _build_observe_fn(eqset, offset_of, per_vars)
    controls_fn = _build_controls_fn(eqset)
    alg_indices = tuple(int(i) for i in np.where(np.asarray(mass) == 0.0)[0].tolist())
    return _LoweringAnalysis(
        layout_entries=entries,
        global_layout_entries=(),
        topo=topo,
        mass_diagonal=mass,
        constraint_dim=0,
        state_size=total,
        alg_indices=alg_indices,
        bc_dof_indices_static=(),
        dirichlet_value_fns=(),
        flux_bc_callables=(),
        robin_bc_callables=(),
        traction_bc_callables=(),
        initial_lambda_si=jnp.zeros((0,)),
        residual_fn=residual_fn,
        jac_blocks_fn=jac_fn,
        observe_fn=observe_fn,
        controls_fn=controls_fn,
    )
