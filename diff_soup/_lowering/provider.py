"""Lowering analysis provider: builds residual/Jacobian/sparsity/mass kernels.

``lower()`` (in ``diff_soup.problem``) builds a ``LoweredProblem`` from an
``EquationSet`` but cannot author the residual / Jacobian / sparsity /
mass-diagonal kernels itself: those require the ``diff_soup.domains.operators``
bank, which ``problem`` cannot import directly.  The injection seam —
``diff_soup.problem._ir.lower.set_lowering_ops(provider)`` — accepts an
``analyze(eqset, *, precision) -> _LoweringAnalysis`` callable from a layer
that may freely import both sides.  This module is that provider.

``diff_soup/_lowering/`` sits outside the layer-rank map, so it may legally
import both ``diff_soup.domains`` and ``diff_soup.problem._ir``.  It wires the
seam at import time (see [diff_soup._lowering][diff_soup._lowering]).

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

import dataclasses
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._precision import Precision
from diff_soup.domains.grid.core import _METRIC_DIMENSION
from diff_soup.domains.mesh.fem._boundary import facet_average, facet_shares
from diff_soup.domains.operators.dispatch import dispatch_lumped_mass, finalize_residual
from diff_soup.linear import SparseTopology
from diff_soup.problem._ad.sparse_jac import SparseJacobian
from diff_soup.problem._ir.constraint import (
    OperationalControlConstraint,
    ReduceResidualConstraint,
)
from diff_soup.problem._ir.derivative_ir import capture_forms
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.lower import _apply_boundary_cast, _LoweringAnalysis
from diff_soup.problem._ir.variable import GlobalVariable, Variable
from diff_soup.problem._spec.derivative_expr import (
    current_slopes,
    slope_context_includes_residual,
)

from .constraints import _eval_constraint_residual
from .couplings import _read_through_alias_owner, _resolve_family_couplings
from .family_residual import _build_family_residual
from .family_validate import _validate_families
from .layout import _BlockLayout, _build_global_layout_entries, _layout
from .mass import _mass_diagonal as _mass_diagonal_full
from .operator_resolve import resolve_operator
from .sparsity import _sparsity as _sparsity_full
from .stencil_infer import _infer_stencil_offsets

_MAX_DENSE_FALLBACK_NNZ = 1_000_000


def _check_multi_equation_return(eq: MultiEquation, r) -> None:
    """Reject a bundled-equation return that does not cover its written fields.

    A [MultiEquation][diff_soup.problem._ir.equation.MultiEquation] closure must return a dict with one residual per
    written field name; anything else fails here with the equation named,
    rather than as a downstream key or shape error.
    """
    written = tuple(v.name for v in eq.vars)
    if not isinstance(r, dict):
        raise TypeError(
            f"equation {eq.name!r} writes the fields {written!r} and must "
            f"return a dict keyed by their names; got {type(r).__name__}."
        )
    missing = [name for name in written if name not in r]
    if missing:
        raise ValueError(
            f"equation {eq.name!r} returned no residual for {missing!r}; its "
            f"dict must provide one entry per written field {written!r}."
        )


class _Ctx:
    """Minimal residual/observation context carrying a component's ops.

    The per-equation closures read ``ctx.ops`` for the operator bank and
    ``ctx.controls`` for the evaluation's resolved dimensional control values (``None``
    on surfaces controls do not reach).  A plain attribute holder —
    ``object.__setattr__`` is not overridden.
    """

    def __init__(self, ops: Any, controls: Any = None) -> None:
        self.ops = ops
        self.controls = controls


def _geometry_scale_value(boundary_context, comp) -> jax.Array:
    """The current value of the declared parameter that scales ``comp``'s grid.

    ``boundary_context`` is the evaluation's declared namespaces
    ([_boundary_context][diff_soup._lowering.bcs._boundary_context]), read from
    the runtime equation leaves, so the value is the one this evaluation carries
    (a fixed input, a fitted iterate, or a traced lane value).
    """
    params = (boundary_context or {}).get("params") or {}
    name = comp.geometry_scale
    if name not in params:
        raise ConfigurationError(
            f"component {comp.name!r} is scaled by the parameter {name!r}, but the "
            f"evaluation carries no declared parameter of that name. Declare the "
            f"scale with spec.geometry({name!r}) so its value reaches the kernels."
        )
    return jnp.asarray(params[name])


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


def _periodic_axes_of_component(comp) -> tuple[int, ...]:
    """Union of periodic axes declared on ``comp`` (empty if none)."""
    from diff_soup.problem._ir.bc import PeriodicBC

    axes: set[int] = set()
    for bc in comp.bcs:
        if isinstance(bc, PeriodicBC):
            axes.update(int(a) for a in bc.axes)
    return tuple(sorted(axes))


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


def _sparsity(eqset: EquationSet, offset_of: dict, total: int, stencil_offsets=None):
    """Per-equation ``(out_var, inp_var)`` node-field block pattern + diagonal.

    Same-domain node fields tile the domain's ``sparsity_pattern`` stencil;
    multi-dim variables fall back to a dense local block (subject to the cap).
    Globals are skipped (they feed the dense Schur ``B`` block, absent on the
    slim path).  The mandatory diagonal completion guarantees the implicit-mass
    / BC-identity term has a slot.

    ``stencil_offsets`` carries the probe-inferred stencils (see
    [_infer_stencil_offsets][diff_soup._lowering.stencil_infer._infer_stencil_offsets]): a
    ``(comp, write, read)`` entry replaces that block's default radius-1
    pattern with the measured offset set, or with the dense block when the
    measured dependency is not a local stencil.  Blocks without an entry keep
    the domain's default radius-1 pattern.
    """
    comp = eqset.components[0]
    num_nodes_attr = getattr(comp.domain, "num_nodes", None)
    num_nodes = int(num_nodes_attr) if num_nodes_attr is not None else None
    has_sparsity = hasattr(comp.domain, "sparsity_pattern")
    all_rows: list = []
    all_cols: list = []
    for eq in comp.equations:
        output_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
        for out_var in output_vars:
            row_offset = offset_of[(comp.name, out_var.name)]
            row_size = out_var.size
            for inp in eq.inputs:
                if isinstance(inp, GlobalVariable):
                    continue
                col_offset = offset_of[(comp.name, inp.name)]
                col_size = inp.size
                out_d = _node_components(out_var, num_nodes) if has_sparsity else None
                inp_d = _node_components(inp, num_nodes) if has_sparsity else None
                key = (comp.name, out_var.name, inp.name)
                if out_d is not None and inp_d is not None:
                    if stencil_offsets and key in stencil_offsets:
                        block_offsets = stencil_offsets[key]
                        if block_offsets is None:
                            r_local, c_local = _dense_fallback_block(
                                row_size, col_size, eq.name, inp.name
                            )
                        else:
                            r_local, c_local = _node_field_block(
                                comp.domain, out_d, inp_d, block_offsets=block_offsets
                            )
                    else:
                        r_local, c_local = _node_field_block(comp.domain, out_d, inp_d)
                else:
                    r_local, c_local = _dense_fallback_block(row_size, col_size, eq.name, inp.name)
                all_rows.append(r_local + row_offset)
                all_cols.append(c_local + col_offset)
    diag = np.arange(total, dtype=np.int64)
    all_rows.append(diag)
    all_cols.append(diag)
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    # Canonicalise row-major + dedup.
    pair_key = rows.astype(np.int64) * np.int64(total) + cols.astype(np.int64)
    _uniq, uniq_idx = np.unique(pair_key, return_index=True)
    order = np.sort(uniq_idx)
    return rows[order].astype(np.int32), cols[order].astype(np.int32)


def _node_components(var, num_nodes) -> Optional[int]:
    """Per-node component count if ``var`` is a node field, else ``None``.

    1-D ``(n,)`` -> 1 component; 2-D ``(n, d)`` -> ``d``; otherwise ``None``
    (dense fallback).
    """
    if num_nodes is None:
        return None
    s = var.shape
    if len(s) == 1 and s[0] == num_nodes:
        return 1
    if len(s) == 2 and s[0] == num_nodes:
        return int(s[1])
    return None


def _node_field_block(domain, d_out: int, d_in: int, block_offsets=None):
    """Kronecker tiling of the domain node-pair stencil across components.

    Node-major DOF layout: ``d*node + comp``.  When ``block_offsets`` is
    given (a probe-inferred ``(n_offsets, ndim)`` stencil offset set), the
    node-pair stencil is generated from those offsets instead of the domain's
    default radius-1 pattern.
    """
    if block_offsets is not None:
        from diff_soup.domains.grid.core import stencil_sparsity_nd  # noqa: PLC0415

        from .stencil_infer import _grid_shape  # noqa: PLC0415

        r1, c1 = stencil_sparsity_nd(_grid_shape(domain), block_offsets)
    else:
        r1, c1 = domain.sparsity_pattern(n_vars=1)
    r1 = np.asarray(r1, dtype=np.int64)
    c1 = np.asarray(c1, dtype=np.int64)
    a = np.arange(d_out, dtype=np.int64)
    b = np.arange(d_in, dtype=np.int64)
    rows = d_out * r1[:, None, None] + a[None, :, None]
    cols = d_in * c1[:, None, None] + b[None, None, :]
    rows = np.broadcast_to(rows, (r1.size, d_out, d_in)).reshape(-1)
    cols = np.broadcast_to(cols, (c1.size, d_out, d_in)).reshape(-1)
    return rows, cols


def _dense_fallback_block(row_size: int, col_size: int, eq_name: str, inp_name: str):
    """Dense ``(row_size, col_size)`` local block for a non-stencil equation.

    Raises:
        ValueError: when the dense block exceeds ``_MAX_DENSE_FALLBACK_NNZ``.
    """
    dense_nnz = int(row_size) * int(col_size)
    if dense_nnz > _MAX_DENSE_FALLBACK_NNZ:
        raise ValueError(
            "lower(): refusing to materialize a dense fallback sparsity block "
            f"with {dense_nnz} entries for equation {eq_name!r} input "
            f"{inp_name!r}. Provide an explicit local sparsity path or split "
            "the coupling through a GlobalVariable/Constraint."
        )
    r_local = np.repeat(np.arange(row_size, dtype=np.int64), col_size)
    c_local = np.tile(np.arange(col_size, dtype=np.int64), row_size)
    return r_local, c_local


def _assembly_dtype(state_dtype, compute_dtype):
    """The residual-assembly dtype for one evaluation: follow a lowered state.

    The boundary cast raises closure outputs to the assembly dtype so a
    closure computed at reduced precision cannot corrupt the residual noise
    floor. When the CALLER deliberately feeds a reduced-precision state — the
    Jacobian-values seam evaluates the residual at fp32 inside its
    differentiation seeds — raising outputs back to fp64 would silently undo
    that choice, so the assembly dtype follows the state downward. It never
    exceeds the configured compute dtype, and with the default fp64 state it
    IS the compute dtype (the emitted program is unchanged).
    """
    sd = jnp.dtype(state_dtype)
    cd = jnp.dtype(compute_dtype)
    return sd if sd.itemsize < cd.itemsize else cd


def _resolve_direct_controls(names, controls, time_ref, t, payload, *, dtype=None):
    """Resolve each direct control's dimensional value for one residual evaluation.

    ``names`` is the static tuple of declared direct-control names.  Each value
    is the override payload's entry when one rides the evaluation, otherwise
    the authored ``fn`` at dimensional time — the same resolution order the solve-level
    control channel applies, so every surface reads one value.

    ``dtype`` is the dtype the evaluation's field values carry.  A control is
    handed to the closures beside those values — a name in the flux face view,
    an entry of ``ctx.controls`` — and multiplies them there, so an authored
    callable returning double precision would widen the whole closure on a
    single-precision state.  Each resolved value is therefore rounded to the
    field dtype exactly once, here; a value that already carries it (the
    solve-level channel casts its payload before it arrives) is passed through
    untouched, so no value is rounded twice.  ``None`` keeps whatever dtype the
    payload or the authored callable produced.
    """
    if not names:
        return {}
    if payload is not None:
        values = getattr(payload, "values", None)
        if hasattr(payload, "epoch") and not callable(values):
            payload = values
    by_name = {str(ctl.name): ctl.fn for ctl in controls}
    t_si = t if time_ref == 1.0 else jnp.asarray(time_ref) * jnp.asarray(t)
    want = None if dtype is None else jnp.dtype(dtype)

    def in_field_dtype(value):
        arr = jnp.asarray(value)
        return arr if want is None or arr.dtype == want else arr.astype(want)

    out = {}
    for name in names:
        if payload is not None and name in payload:
            out[name] = in_field_dtype(payload[name])
            continue
        fn = by_name.get(name)
        if fn is None:
            raise ValueError(
                f"the runtime controls tuple has no entry named {name!r}; it "
                f"must carry every declared control (available: "
                f"{sorted(by_name)!r})."
            )
        out[name] = in_field_dtype(fn(t_si))
    return out


def _face_signature_flux(
    flux_fn,
    is_face_flux,
    owner_view,
    permitted_names,
    plain_ops,
    controls=None,
    boundary_context=None,
):
    """Present one resolved flux callable to a flux-total closure as ``fn(face, t)``.

    A flux authored as ``fn=lambda face, t: ...`` already has that signature.
    The constant and time-dependent forms lower to a ``(view, ctx, t)`` callable
    reading nodal values instead of a face view, so they are wrapped to ignore
    the face view and evaluate against their own component's post-transfer
    values.  Either way the flux-total integral and the natural boundary
    condition evaluate the one density the recipe declared.
    """
    if is_face_flux:

        def boundary_flux(q, t):
            from diff_soup.problem._spec.q_accessor import _BoundaryQ

            query = _BoundaryQ(
                owner_view,
                q._globals,
                q._fi,
                q.geom.normal,
                q.geom.x,
                ops=plain_ops,
                controls=controls,
                t=t,
                **(boundary_context or {}),
            )
            return flux_fn(query, t)

        return boundary_flux
    bc_view = {name: owner_view[name] for name in permitted_names}
    ctx = _Ctx(plain_ops, controls)

    def nodal_flux_as_face_flux(face, t):  # noqa: ARG001 — a nodal flux reads no face
        return flux_fn(bc_view, ctx, t)

    return nodal_flux_as_face_flux


def _build_residual_fn(eqset: EquationSet, offset_of: dict, total: int, compute_dtype):
    """Build the static same-domain residual kernel ``(f, g)``.

    Takes the IR-node ``equations`` tuple as a runtime arg so ``eqx.Module``
    closure leaves reach AD (the hybrid-cache contract).  Per equation:
    evaluates ``rhs_fn(var_view, ctx, t)`` with ``ctx.ops`` the implicit-mass
    the weak-form residual view, runs ``finalize_residual`` on leftover-nodal terms,
    applies the residual-row scaling verb (``residual_scale``; identity default),
    casts up to ``compute_dtype``, and scatters into ``f``.  ``g`` (constraint
    residual) is empty on the slim path.
    """
    comp = eqset.components[0]
    unit_dom = comp.domain
    unit_tag_ops = resolve_operator(unit_dom, comp.operator_factory).for_residual()
    scaled = getattr(comp, "geometry_scale", None) is not None
    per_vars = tuple(comp.variables)
    residual_scale = {k: float(v) for k, v in dict(getattr(eqset, "residual_scale", {})).items()}

    def physics_residual_fn(
        equations,
        dirichlet_value_fns,  # noqa: ARG001 — slim path: no BCs.
        flux_callables,  # noqa: ARG001
        robin_callables,  # noqa: ARG001
        traction_callables,  # noqa: ARG001
        constraints,  # noqa: ARG001 — slim path: no constraints.
        couplings,  # noqa: ARG001 — slim path: no couplings.
        u,
        lam,  # noqa: ARG001 — slim path: no globals.
        t,
        families=(),  # noqa: ARG001 — slim path: no families.
        controls=(),  # noqa: ARG001 — slim path: controls force the full path.
        control=None,  # noqa: ARG001 — slim path: no control-driven rows.
    ):
        view = {}
        o = 0
        for v in per_vars:
            view[v.name] = u[o : o + v.size].reshape(v.shape)
            o += v.size
        var_view = {v: view[v.name] for v in per_vars}
        dom, tag_ops = unit_dom, unit_tag_ops
        if scaled:
            # The grid is scaled by a declared parameter: evaluate on the grid
            # at the parameter's current value.
            from .bcs import _boundary_context

            length = _geometry_scale_value(_boundary_context(equations), comp)
            dom = unit_dom.scaled(length)
            tag_ops = resolve_operator(dom, comp.operator_factory).for_residual()
        ctx = _Ctx(tag_ops)
        f = jnp.zeros((total,), dtype=u.dtype)
        asm_dtype = _assembly_dtype(u.dtype, compute_dtype)
        for eq in equations:
            r = eq.rhs_fn(var_view, ctx, t)
            if isinstance(eq, MultiEquation):
                _check_multi_equation_return(eq, r)
                for v in eq.vars:
                    arr = jnp.asarray(
                        _apply_boundary_cast(finalize_residual(dom, r[v.name]), asm_dtype)
                    ).reshape(-1)
                    if arr.size != v.size:
                        raise ValueError(
                            f"equation residual for variable {v.name!r} has size "
                            f"{int(arr.size)} but the field has {int(v.size)} "
                            f"DOFs; the rhs_fn must return one value per DOF "
                            f"(shape {v.shape})."
                        )
                    _rs = residual_scale.get((comp.name, v.name))
                    if _rs is not None:
                        arr = arr / _rs
                    ofs = offset_of[(comp.name, v.name)]
                    f = f.at[ofs : ofs + v.size].add(arr)
            else:
                arr = jnp.asarray(
                    _apply_boundary_cast(finalize_residual(dom, r), asm_dtype)
                ).reshape(-1)
                if arr.size != eq.var.size:
                    raise ValueError(
                        f"equation residual for variable {eq.var.name!r} has size "
                        f"{int(arr.size)} but the field has {int(eq.var.size)} "
                        f"DOFs; the rhs_fn must return one value per DOF "
                        f"(shape {eq.var.shape})."
                    )
                _rs = residual_scale.get((comp.name, eq.var.name))
                if _rs is not None:
                    arr = arr / _rs
                ofs = offset_of[(comp.name, eq.var.name)]
                f = f.at[ofs : ofs + eq.var.size].add(arr)
        return f, jnp.zeros((0,), dtype=u.dtype)

    return physics_residual_fn


def _build_jac_blocks_fn(residual_fn, topo: SparseTopology, total: int):
    """Build the static same-domain KKT-Jacobian kernel ``(Avals, B, C, D, topo)``.

    ``Avals`` = sparse ``df/du`` via CPR-coloured ``SparseJacobian.evaluate``
    (AD of ``residual_fn``).  On the slim path there are no constraints, so
    ``B``/``C``/``D`` are empty.
    """
    sparse_jac = SparseJacobian.from_topology(topo)

    def physics_jac_blocks_fn(
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
        def f_of_u(u_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u_,
                lam,
                t,
                families,
                controls,
                control,
            )
            return f

        Avals = sparse_jac.evaluate(f_of_u, u)
        return (
            Avals,
            jnp.zeros((total, 0), dtype=u.dtype),
            jnp.zeros((0, total), dtype=u.dtype),
            jnp.zeros((0, 0), dtype=u.dtype),
            topo,
        )

    return physics_jac_blocks_fn


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
    (a [_ConstraintBorderProbe][diff_soup._lowering.provider._ConstraintBorderProbe])
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
    family_blocks = []
    for fam in eqset.families:
        fam_base = layout.offset_of((fam.name, fam.member_variables[0].name))
        family_blocks.append(
            (
                fam_base,
                _build_family_residual(fam, fam_base, total_size=total, time_ref=callback_time_ref),
            )
        )
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


def _is_plain_sum(reduce_fn) -> bool:
    """Return True iff ``reduce_fn`` is a plain element sum.

    Accepts ``jnp.sum`` / ``numpy.sum`` by identity and any wrapping callable
    named ``"sum"`` (e.g. ``lambda x: jnp.sum(x)``).  This ensures the same
    reduction always selects the same C-block formula regardless of how it was
    authored.
    """
    if reduce_fn is jnp.sum:
        return True
    if reduce_fn is np.sum:
        return True
    return getattr(reduce_fn, "__name__", "") == "sum"


class _AccessTrackingView:
    """View proxy that records which keys a residual closure reads.

    Used at lower-time to discover the true column support of a field
    equation's residual so the ``ReduceResidualConstraint`` C-block can decide
    whether the declared-topology scatter-sum shortcut is safe (every read is
    declared in ``inputs``) or whether it must fall back to dense jacrev.
    """

    def __init__(self, backing: dict):
        self._backing = backing
        self.accessed: set = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return self._backing[key]

    def __contains__(self, key):
        return key in self._backing


def _threaded_globals(eqset) -> tuple:
    """The bordered globals the residual places in every component's view.

    Returns ``(global, dim)`` pairs in global-row order: the residual reads a
    global closed by a row of one entry as a scalar and a wider one as a
    ``(dim,)`` slice of the bordered unknowns.
    """
    return tuple(
        (c.closes, int(c.dim))
        for c in eqset.constraints
        if isinstance(getattr(c, "closes", None), GlobalVariable)
    )


def _field_eq_is_fully_declared(comp, field_var, ops, threaded_globals=()) -> bool:
    """Return True iff every Variable read by ``field_var``'s equation is declared.

    Traces the residual closure that writes ``field_var`` against an
    access-tracking view, with ``ops`` the operator bank the residual hands the
    component's equations, and confirms every local ``Variable`` it reads
    appears in the equation's ``inputs``.  When the equation under-declares a
    column the declared-topology ``Avals`` would miss it, so the C-block
    scatter-sum shortcut is unsafe and the caller must use dense jacrev.

    The view holds the component's variables and every bordered global in
    ``threaded_globals`` (``(global, dim)`` pairs, as the residual threads
    them), so an equation that reads a global traces as it runs.  A global's
    column is ``B = df/dlam``, which the kernel differentiates in full, so a
    read of a global never makes the shortcut unsafe.

    A read the view does not hold (a ``KeyError`` from the view, or an
    ``AttributeError`` from the ``q`` accessor) returns ``False`` so the safe
    dense-jacrev path is taken; any other error the closure raises propagates.
    """
    target_eq = None
    for eq in comp.equations:
        out_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
        if any(v is field_var for v in out_vars):
            target_eq = eq
            break
    if target_eq is None:
        return False
    declared = {inp for inp in target_eq.inputs if isinstance(inp, Variable)}
    try:
        backing = {v: jnp.zeros(v.shape, dtype=jnp.float64) for v in comp.variables}
        # Globals are read through the view but never enter the sparse df/du
        # topology: a global's value has the shape the residual gives it.
        for g, dim in threaded_globals:
            backing.setdefault(g, jnp.zeros(() if dim == 1 else (dim,), dtype=jnp.float64))
        for inp in target_eq.inputs:
            if isinstance(inp, GlobalVariable) and inp not in backing:
                backing[inp] = jnp.zeros((), dtype=jnp.float64)
        view = _AccessTrackingView(backing)
        ctx = _Ctx(ops)
        target_eq.rhs_fn(view, ctx, jnp.zeros(()))
    except (KeyError, AttributeError):
        return False
    read_vars = {k for k in view.accessed if isinstance(k, Variable)}
    return read_vars.issubset(declared)


class _ConstraintBorderProbe:
    """Trace-time work counters for the bordered-global derivative kernel.

    The compatibility type name predates differential globals. Each count is
    incremented when the border kernel (the row-restricted
    residual used to build ``C = dg/du`` / ``D = dg/dlam``) invokes an
    equation closure, a Family block, or a bordered residual DURING TRACING.
    Under an outer ``jax.jit`` the counts therefore reflect compilations, not
    executions — which is the meaningful measure: physics absent from the
    trace is absent from the compiled derivative program.
    """

    __slots__ = ("equation_evals", "family_evals", "constraint_evals")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero all counters (start a fresh measurement window)."""
        self.equation_evals: dict = {}
        self.family_evals: dict = {}
        self.constraint_evals: dict = {}

    def count_equation(self, name: str) -> None:
        self.equation_evals[name] = self.equation_evals.get(name, 0) + 1

    def count_family(self, name: str) -> None:
        self.family_evals[name] = self.family_evals.get(name, 0) + 1

    def count_constraint(self, name: str) -> None:
        self.constraint_evals[name] = self.constraint_evals.get(name, 0) + 1


class _ConstraintBorderPlan(NamedTuple):
    """Static evaluation plan of the bordered-global derivative kernel.

    Attributes:
        required_rows: ``(owner, field)`` residual slices the border kernel
            assembles (the residual-reducing fallbacks' declared fields).
        required_by: per-fallback-global-row ``(name, rows)`` pairs; an
            observable-only fallback (e.g. galvanostatic) requires no slice.
        fallback_constraints: compatibility-named global rows whose ``C``/``D`` come from
            dense differentiation of the border kernel.
        shortcut_constraints: algebraic constraints whose rows come from the
            scatter-sum-over-``Avals`` / slice-of-``B`` shortcut.  An
            operational table whose current is a field's residual sum is in
            both tuples: its ``current`` row comes from the shortcut, its
            other rows from the border kernel, which holds that current fixed.
        skipped_equations: equations the border kernel never evaluates.
        skipped_families: families the border kernel never evaluates.
        restricted: ``False`` when the border falls back to the full residual
            kernel (a fallback global row's residual slice could not be
            resolved to an owning component field).
    """

    required_rows: tuple
    required_by: tuple
    fallback_constraints: tuple
    shortcut_constraints: tuple
    skipped_equations: tuple
    skipped_families: tuple
    restricted: bool


def _build_full_jac_blocks_fn(
    eqset,
    residual_fn,
    topo: SparseTopology,
    total: int,
    *,
    compute_dtype=None,
    bcs=None,
):
    """Build the family/coupling KKT-Jacobian kernel ``(Avals, B, C, D, topo)``.

    ``Avals`` = sparse ``df/du`` via CPR-coloured ``SparseJacobian.evaluate``;
    ``B = df/dlam`` (``jacfwd`` over the small global axis). ``C = dg/du`` /
    ``D = dg/dlam`` are built per bordered-global row: a ``ReduceResidualConstraint`` with
    a plain-sum reduction over a fully-declared field uses a
    scatter-sum-from-``Avals`` + slice-and-sum-from-``B`` shortcut; every other
    row (including a ``DifferentialGlobalEquation``, ``GalvanostaticConstraint``,
    or any under-declared/non-sum reducer) takes the dense ``jacrev`` /
    ``jacfwd`` fallback.  An ``OperationalControlConstraint`` whose current is
    the residual sum of a fully-declared field, in a table without a ``power``
    mode, combines the two: the border kernel differentiates its row with that
    current held fixed (the ``voltage`` row's dependence), and the shortcut row
    is added when the operating mode is ``current``.

    The fallback differentiates the bordered-global kernel — the residual
    restricted to the fallback rows' declared residual slices and
    observables (built when ``compute_dtype`` and ``bcs`` are provided) — so
    ``C``/``D`` construction never differentiates unrelated component or
    Family physics while remaining numerically identical to differentiating
    the full residual's ``g``.  ``B = df/dlam`` stays a full physics
    derivative.  The returned kernel publishes its border introspection as
    ``constraint_border_plan``
    (a [_ConstraintBorderPlan][diff_soup._lowering.provider._ConstraintBorderPlan])
    and ``constraint_border_probe``
    (a [_ConstraintBorderProbe][diff_soup._lowering.provider._ConstraintBorderProbe]).
    """
    sparse_jac = SparseJacobian.from_topology(topo)
    n_c = int(sum(int(c.dim) for c in eqset.constraints))
    n_s = int(total)

    # ------------------------------------------------------------------
    # Host-time per-global-row Schur shortcut analysis. Each entry:
    #   - ("sum", row_lo, row_hi, lam_off, dim) for the scatter-sum path
    #   - ("fallback", lam_off, dim)            for the dense jacrev/jacfwd path
    #   - ("operational_sum", row_lo, row_hi, lam_off, dim, current_index, name) for
    #     an operational table whose current is a residual sum: the dense path
    #     with that current held fixed, plus the scatter-sum row whenever the
    #     operating mode index equals ``current_index`` (``None``: no such mode)
    # The constrained field is resolved to its owning component by Python ``is``
    # (host-side) so a structurally-equal Variable cannot misroute the slice.
    # ------------------------------------------------------------------
    layout = _layout(eqset)
    threaded_globals = _threaded_globals(eqset)

    def _residual_bank(comp):
        # The bank the residual hands the owner's equations as ``q.ops``: on
        # FEM the weak-form view, whose operators return the shapes the
        # equation's terms combine in.
        return resolve_operator(comp.domain, comp.operator_factory).for_residual()

    shortcut_meta: list = []
    shortcut_names: list = []
    fallback_names: list = []
    held_names: list = []
    required_by: list = []
    border_resolvable = True
    _lam_off = 0
    for c in eqset.constraints:
        dim_c = int(c.dim)
        if isinstance(c, ReduceResidualConstraint):
            field_var = c.field
            owner_comp = None
            owner_name = None
            for comp in eqset.components:
                if any(v is field_var for v in comp.variables):
                    owner_comp = comp
                    owner_name = comp.name
                    break
            if _is_plain_sum(c.reduce_fn):
                if owner_name is None:
                    raise ValueError(
                        f"constraint {c.name!r}: field {field_var.name!r} not found "
                        f"by identity in any component's variables"
                    )
                row_lo = layout.offset_of((owner_name, field_var.name))
                row_hi = row_lo + int(field_var.size)
                if _field_eq_is_fully_declared(
                    owner_comp, field_var, _residual_bank(owner_comp), threaded_globals
                ):
                    shortcut_meta.append(("sum", row_lo, row_hi, _lam_off, dim_c))
                    shortcut_names.append(c.name)
                else:
                    # Under-declared field equation: declared-topology Avals
                    # would drop the undeclared column.  Dense fallback over
                    # the field's residual slice keeps C == dg/du exact.
                    shortcut_meta.append(("fallback", _lam_off, dim_c))
                    fallback_names.append(c.name)
                    required_by.append((c.name, ((owner_name, field_var.name),)))
            else:
                shortcut_meta.append(("fallback", _lam_off, dim_c))
                fallback_names.append(c.name)
                if owner_name is None:
                    # The declared slice cannot be pinned to a component field;
                    # the border must evaluate the full residual to stay exact.
                    border_resolvable = False
                    required_by.append((c.name, ()))
                else:
                    required_by.append((c.name, ((owner_name, field_var.name),)))
        elif isinstance(c, OperationalControlConstraint) and c.field is not None:
            field_var = c.field
            owner_comp = next(
                (comp for comp in eqset.components if any(v is field_var for v in comp.variables)),
                None,
            )
            if owner_comp is None:
                raise ValueError(
                    f"operational control {c.name!r}: field {field_var.name!r} not found "
                    f"by identity in any component's variables"
                )
            row_lo = layout.offset_of((owner_comp.name, field_var.name))
            row_hi = row_lo + int(field_var.size)
            fallback_names.append(c.name)
            if "power" not in c.modes and _field_eq_is_fully_declared(
                owner_comp, field_var, _residual_bank(owner_comp), threaded_globals
            ):
                # The current enters its row linearly, so the row's derivative
                # is the border kernel's (current held fixed) plus, in the
                # ``current`` mode, the field's Jacobian rows summed.
                current_index = c.modes.index("current") if "current" in c.modes else None
                shortcut_meta.append(
                    ("operational_sum", row_lo, row_hi, _lam_off, dim_c, current_index, c.name)
                )
                shortcut_names.append(c.name)
                held_names.append(c.name)
                required_by.append((c.name, ()))
            else:
                # A power row multiplies the current by the voltage, and an
                # under-declared field equation would drop a column from
                # ``Avals``: differentiate the field's residual rows directly.
                shortcut_meta.append(("fallback", _lam_off, dim_c))
                required_by.append((c.name, ((owner_comp.name, field_var.name),)))
        else:
            shortcut_meta.append(("fallback", _lam_off, dim_c))
            fallback_names.append(c.name)
            required_by.append((c.name, ()))
        _lam_off += dim_c
    any_fallback = any(m[0] in ("fallback", "operational_sum") for m in shortcut_meta)

    # ------------------------------------------------------------------
    # Bordered-global kernel: the fallback C/D derivatives differentiate a
    # residual restricted to the fallback rows' declared slices, so
    # unrelated physics rows never enter the derivative trace.
    # ------------------------------------------------------------------
    required_rows = tuple(sorted({k for _name, keys in required_by for k in keys}))
    border_probe = _ConstraintBorderProbe()
    border_restricted = (
        any_fallback and border_resolvable and compute_dtype is not None and bcs is not None
    )
    if held_names and not border_restricted:
        # The border kernel is then the full residual, which holds no current
        # fixed: differentiating it gives these rows whole.
        shortcut_meta = [
            ("fallback", m[3], m[4]) if m[0] == "operational_sum" else m for m in shortcut_meta
        ]
        shortcut_names = [name for name in shortcut_names if name not in held_names]
        held_names = []
    if border_restricted:
        border_residual_fn = _build_full_residual_fn(
            eqset,
            layout,
            compute_dtype,
            bcs,
            restrict_rows_to=frozenset(required_rows),
            restrict_constraints_to=frozenset(fallback_names),
            hold_residual_currents=frozenset(held_names),
            border_probe=border_probe,
        )
    else:
        border_residual_fn = residual_fn
    if border_restricted:
        required_set = frozenset(required_rows)
        skipped_equations = []
        for comp in eqset.components:
            for eq in comp.equations:
                outs = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
                if not any((comp.name, v.name) in required_set for v in outs):
                    skipped_equations.append(getattr(eq, "name", None) or eq.var.name)
        skipped_families = [
            fam.name
            for fam in eqset.families
            if not any((fam.name, v.name) in required_set for v in fam.member_variables)
        ]
    else:
        skipped_equations = []
        skipped_families = []
    border_plan = _ConstraintBorderPlan(
        required_rows=required_rows,
        required_by=tuple(required_by),
        fallback_constraints=tuple(fallback_names),
        shortcut_constraints=tuple(shortcut_names),
        skipped_equations=tuple(skipped_equations),
        skipped_families=tuple(skipped_families),
        restricted=border_restricted,
    )
    # Device-side topology arrays so the scatter-sum path closes over them
    # without re-uploading host arrays per call (XLA folds them at trace time).
    topo_rows_dev = jnp.asarray(topo.rows, dtype=jnp.int32)
    topo_cols_dev = jnp.asarray(topo.cols, dtype=jnp.int32)

    def physics_jac_blocks_fn(
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
        def f_of_u(u_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u_,
                lam,
                t,
                families,
                controls,
                control,
            )
            return f

        Avals = sparse_jac.evaluate(f_of_u, u)
        if n_c == 0:
            return (
                Avals,
                jnp.zeros((n_s, 0), dtype=u.dtype),
                jnp.zeros((0, n_s), dtype=u.dtype),
                jnp.zeros((0, 0), dtype=u.dtype),
                topo,
            )

        def f_of_lam(lam_):
            f, _g = residual_fn(
                equations,
                dirichlet_value_fns,
                flux_callables,
                robin_callables,
                traction_callables,
                constraints,
                couplings,
                u,
                lam_,
                t,
                families,
                controls,
                control,
            )
            return f

        # B = df/dlam via forward-mode (lam is constraint-dim, small); jacfwd
        # avoids the n_state-dim backward tape jacrev would build.
        B = jax.jacfwd(f_of_lam)(lam)

        # C = dg/du, D = dg/dlam.  Dense jacrev/jacfwd only when at least one
        # constraint takes the fallback; the plain-sum-over-fully-declared rows
        # take the scatter-sum-from-Avals + slice-and-sum-from-B shortcut.
        # The fallback differentiates the bordered-global kernel (the
        # residual restricted to the fallback rows' declared slices),
        # never the unrelated physics rows.
        full_C = None
        full_D = None
        if any_fallback:

            def g_of_u(u_):
                _f, g = border_residual_fn(
                    equations,
                    dirichlet_value_fns,
                    flux_callables,
                    robin_callables,
                    traction_callables,
                    constraints,
                    couplings,
                    u_,
                    lam,
                    t,
                    families,
                    controls,
                    control,
                )
                return g

            def g_of_lam(lam_):
                _f, g = border_residual_fn(
                    equations,
                    dirichlet_value_fns,
                    flux_callables,
                    robin_callables,
                    traction_callables,
                    constraints,
                    couplings,
                    u,
                    lam_,
                    t,
                    families,
                    controls,
                    control,
                )
                return g

            full_C = jax.jacrev(g_of_u)(u)
            full_D = jax.jacfwd(g_of_lam)(lam)

        def residual_sum_rows(row_lo, row_hi, dim_c):
            """The ``C`` and ``D`` rows of the sum of the residual rows ``[row_lo, row_hi)``."""
            # C row: scatter-sum Avals over the field's residual rows into
            # the u-columns: dg/du_j = sum_{i in [row_lo,row_hi)} A[i, j].
            row_mask = (topo_rows_dev >= row_lo) & (topo_rows_dev < row_hi)
            contrib = jnp.where(row_mask, Avals, jnp.zeros_like(Avals))
            C_row = jnp.zeros((dim_c, n_s), dtype=u.dtype)
            C_row = C_row.at[0, topo_cols_dev].add(contrib)
            # D row: sum B's field-row slice along axis 0.
            D_row = jnp.sum(B[row_lo:row_hi, :], axis=0, keepdims=True)
            return C_row, D_row

        c_rows: list = []
        d_rows: list = []
        for meta in shortcut_meta:
            if meta[0] == "sum":
                _, row_lo, row_hi, _lam_off_local, dim_c = meta
                C_row, D_row = residual_sum_rows(row_lo, row_hi, dim_c)
                c_rows.append(C_row)
                d_rows.append(D_row)
            elif meta[0] == "operational_sum":
                _, row_lo, row_hi, _lam_off_local, dim_c, current_index, table_name = meta
                C_row = full_C[_lam_off_local : _lam_off_local + dim_c]
                D_row = full_D[_lam_off_local : _lam_off_local + dim_c]
                if current_index is not None:
                    mode = getattr(control, "mode", None)
                    if mode is None:
                        raise ConfigurationError(
                            f"this problem declares an operational control table "
                            f"({table_name!r}); evaluate it with an operating control "
                            "state: ControlState.operating(mode=..., target=...)."
                        )
                    C_sum, D_sum = residual_sum_rows(row_lo, row_hi, dim_c)
                    on = jnp.asarray(mode, dtype=jnp.int32) == current_index
                    C_row = jnp.where(on, C_row + C_sum, C_row)
                    D_row = jnp.where(on, D_row + D_sum, D_row)
                c_rows.append(C_row)
                d_rows.append(D_row)
            else:  # fallback
                _, _lam_off_local, dim_c = meta
                c_rows.append(full_C[_lam_off_local : _lam_off_local + dim_c])
                d_rows.append(full_D[_lam_off_local : _lam_off_local + dim_c])
        C = jnp.concatenate(c_rows, axis=0)
        D = jnp.concatenate(d_rows, axis=0)
        return Avals, B, C, D, topo

    # Border introspection: the static evaluation plan and the trace-time work
    # counters of the bordered-global derivative kernel.
    physics_jac_blocks_fn.constraint_border_plan = border_plan
    physics_jac_blocks_fn.constraint_border_probe = border_probe
    return physics_jac_blocks_fn


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
    [_build_global_layout_entries][diff_soup._lowering.layout._build_global_layout_entries]
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
    [ControlBindingSet][diff_soup._lowering.control_bindings.ControlBindingSet]
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


def _analyze_family_layer(eqset: EquationSet, compute_dtype) -> _LoweringAnalysis:
    """Host analysis for the multi-component family / coupling / BC path.

    Uses the family-aware [_layout][diff_soup._lowering.layout._layout]
    (member-major slabs after components), the multi-component family
    self-block + cross-block `_sparsity_full`, the family mass-diagonal
    branch (`_mass_diagonal_full`), the resolved boundary conditions
    ([_bc_dofs][diff_soup._lowering.bcs._bc_dofs]: Dirichlet pin rows + mass-zero +
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

    stencil_offsets = _infer_stencil_offsets(
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
        rows = np.concatenate([rows, bc_pair_rows.astype(np.int32)])
        cols = np.concatenate([cols, bc_pair_cols.astype(np.int32)])
        pair_key = rows.astype(np.int64) * np.int64(total) + cols.astype(np.int64)
        _uniq, uniq_idx = np.unique(pair_key, return_index=True)
        order = np.sort(uniq_idx)
        rows = rows[order].astype(np.int32)
        cols = cols[order].astype(np.int32)
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


def analyze(eqset: EquationSet, *, precision: Optional[Precision] = None) -> _LoweringAnalysis:
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
        return _analyze_family_layer(eqset, compute_dtype)

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

    stencil_offsets = _infer_stencil_offsets(
        eqset,
        lambda key: offset_of[key],
        total,
        0,
        _probe_f,
        compute_dtype,
    )
    _capture_component_forms(eqset, _probe_f, total, 0, compute_dtype)
    rows, cols = _sparsity(eqset, offset_of, total, stencil_offsets=stencil_offsets)
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
