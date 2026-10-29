"""Sparse Jacobian topology construction (component + Family arms).

``_sparsity`` builds the (rows, cols) tuple representing the static Jacobian
topology from per-equation node-field rules + coupling cross-blocks.  Used by
the provider's ``analyze`` to construct the [SparseTopology][diff_soup._lowering.sparsity.SparseTopology] that
``physics_jacobian_blocks`` evaluates against.

This module lives under ``diff_soup/_lowering`` and may import
``diff_soup.problem._ir`` IR nodes.  The family self-block stamps ONE member
stencil ``count`` times via ``SparseJacobian.from_tiled_coo`` (no dense
fallback), and the family<->component cross-blocks are resolved once per
coupling.
"""

from __future__ import annotations

import numpy as np

from diff_soup.problem._ad.sparse_jac import SparseJacobian
from diff_soup.problem._ir.bc import FluxBC
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.equation_set import EquationSet
from diff_soup.problem._ir.variable import GlobalVariable

from .couplings import (
    _compose_cross_block_sparsity,
    _dense_cross_block,
    _op_declares_empty_sparsity,
    _read_through_alias_owner,
    _resolve_coupling_endpoints,
)
from .layout import _BlockLayout

_MAX_DENSE_FALLBACK_NNZ = 1_000_000

# Maximum number of columns a single Jacobian row (one residual equation) may
# couple to in any cross-block.  The implicit-NDF backend recovers df/du by
# coloring COLUMNS so that no two same-colored columns share a nonzero row;
# memory is O(n_colors * n_dofs) and n_colors is bounded below by the largest
# row degree.  A near-dense row (one residual row coupled to ~n_dofs columns)
# therefore collapses the coloring to roughly one color per degree of freedom
# and the assembly to the dense size.  Every shipped coupling / flux / family
# cross-block couples a residual row only to a transfer op's node-local source
# columns (tens), so this absolute limit sits far above legitimate blocks while
# rejecting a collapsing near-dense row.  Only the ROW degree is bounded: the
# coloring is over columns, so a dense COLUMN is harmless and a legitimate
# broadcast coupling (a family gather/scatter that touches many members) has
# large column degree but small row degree and must pass.
_MAX_BLOCK_DEGREE = 1024


def _guard_block_degree(rows, cols, *, where: str, remedy: str | None = None) -> None:
    """Reject a cross-block whose densest row exceeds the per-row column limit.

    Computes the maximum ROW degree (the number of UNIQUE columns coupled to a
    single residual row) of the local ``(rows, cols)`` block and raises when it
    exceeds ``_MAX_BLOCK_DEGREE``.  Several assembly paths emit duplicate
    ``(row, col)`` pairs that are later canonicalized away, so the pairs are
    deduplicated before the degree is measured — counting raw row occurrences
    would over-estimate the degree and reject a block that is within the limit.
    Column degree is intentionally not bounded: the sparse-Jacobian recovery
    colors columns, so a dense column is benign while a near-dense row collapses
    the coloring to one color per degree of freedom.

    Raises:
        ValueError: when the densest row of ``(rows, cols)`` couples to more
            than ``_MAX_BLOCK_DEGREE`` unique columns.
    """
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    if rows.size == 0:
        return
    unique_pairs = np.unique(np.stack((rows, cols), axis=1), axis=0)
    max_row_degree = int(np.unique(unique_pairs[:, 0], return_counts=True)[1].max())
    if max_row_degree > _MAX_BLOCK_DEGREE:
        if remedy is None:
            remedy = (
                "Give the transfer op a tighter sparsity() or compose the "
                "block against the reading stencil so each residual row "
                "depends only on its node-local source columns."
            )
        raise ValueError(
            f"lower(): {where} declares a Jacobian row coupled to "
            f"{max_row_degree} columns, exceeding the per-row limit of "
            f"{_MAX_BLOCK_DEGREE}. A near-dense Jacobian row collapses the "
            "sparse-Jacobian coloring to one color per degree of freedom, so "
            f"the assembly grows to the dense size. {remedy}"
        )


def _dense_fallback_block(
    row_size: int, col_size: int, eq_name: str, inp_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(row_size, col_size)`` local block for a non-stencil equation.

    Multi-dim Variables (e.g. P2D particles with shape ``(n_x, n_r)``) do not
    fit the 1-D-field domain sparsity pattern; this materialises the full dense
    block subject to the global cap.  Named (vs inlined) so callers — and the
    family-path tests — can detect when this branch is taken via a sentinel.

    Raises:
        ValueError: when the dense block exceeds ``_MAX_DENSE_FALLBACK_NNZ``.
    """
    dense_nnz = int(row_size) * int(col_size)
    if dense_nnz > _MAX_DENSE_FALLBACK_NNZ:
        raise ValueError(
            "lower(): refusing to materialize a dense fallback "
            f"sparsity block with {dense_nnz} entries for "
            f"equation {eq_name!r} input {inp_name!r}. "
            "Provide an explicit local sparsity path or split "
            "the coupling through a GlobalVariable/Constraint."
        )
    r_local = np.repeat(np.arange(row_size, dtype=np.int64), col_size)
    c_local = np.tile(np.arange(col_size, dtype=np.int64), row_size)
    return r_local, c_local


def _component_periodic_axes(comp) -> tuple[int, ...]:
    """Axes on which any field of ``comp`` is periodic (empty if none)."""
    from diff_soup.problem._ir.bc import PeriodicBC

    axes: set[int] = set()
    for bc in comp.bcs:
        if isinstance(bc, PeriodicBC):
            axes.update(int(a) for a in bc.axes)
    return tuple(sorted(axes))


def _node_field_block(
    domain,
    d_out: int,
    d_in: int,
    periodic_axes: tuple[int, ...] = (),
    block_offsets=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular Kronecker tiling of a domain node-pair stencil across components.

    ``sparsity_pattern(n_vars=1)`` returns the per-node coupling pairs (DOF index
    == node index for 1 var/node).  Expand each pair ``(i, j)`` to all
    ``d_out x d_in`` component combinations, node-major (DOF = ``d*node + comp``)
    to match the layout's contiguous ``(n, d)`` reshape.  ``d_out == d_in == 1``
    returns the node-pair pattern itself.

    ``periodic_axes`` (non-empty only on a component whose grid is declared
    periodic) adds the wrap coupling between the first and last node/cell on
    each listed axis, on every node-field block — a periodic grid domain is
    periodic for every field it hosts.

    ``block_offsets`` (a probe-inferred or declared ``(n_offsets, ndim)``
    stencil offset set) replaces the domain's default radius-1 node-pair
    pattern with the pattern generated from those offsets; the offsets wrap on
    ``periodic_axes`` and clip at the other axes' boundaries.
    """
    if block_offsets is not None:
        from diff_soup.domains.grid.core import stencil_sparsity_nd  # noqa: PLC0415

        from .stencil_infer import _grid_shape  # noqa: PLC0415

        r1, c1 = stencil_sparsity_nd(
            _grid_shape(domain), block_offsets, periodic_axes=periodic_axes
        )
    elif periodic_axes and hasattr(domain, "axes"):
        r1, c1 = domain.sparsity_pattern(n_vars=1, periodic_axes=periodic_axes)
    elif periodic_axes:
        # Grid1D: single axis; periodic iff axis 0 is listed.
        r1, c1 = domain.sparsity_pattern(n_vars=1, periodic=(0 in periodic_axes))
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


def _node_components(var, num_nodes):
    """Return the per-node component count if ``var`` is a node field, else None.

    Node field == leading axis equals ``num_nodes``.  1-D ``(n,)`` -> 1
    component; 2-D ``(n, d)`` -> ``d`` components.  Higher-rank or
    non-node-leading shapes (e.g. P2D ``(n_x, n_r)`` where ``n_x != num_nodes``)
    return None -> dense fallback (unchanged behaviour).
    """
    if num_nodes is None:
        return None
    s = var.shape
    if len(s) == 1 and s[0] == num_nodes:
        return 1
    if len(s) == 2 and s[0] == num_nodes:
        return int(s[1])
    return None


def _sparsity(
    eqset: EquationSet,
    layout: _BlockLayout,
    resolved_fluxes=(),
    stencil_offsets=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full ``(rows, cols)`` Jacobian topology for ``eqset``.

    Walks the per-equation node-field stencils + coupling cross-blocks + the
    mandatory diagonal completion, then (when families are present) stamps each
    family's self-block via ``SparseJacobian.from_tiled_coo`` and adds the
    family<->component cross-blocks.  On a no-family single-component eqset this
    is the per-equation node-field block plus the diagonal.

    ``stencil_offsets`` carries the probe-inferred stencils (see
    [_infer_stencil_offsets][diff_soup._lowering.stencil_infer._infer_stencil_offsets]): a
    ``(comp, write, read)`` entry replaces that block's default radius-1
    pattern with the measured offset set, or with the dense block when the
    measured dependency is not a local stencil.  Blocks without an entry keep
    the domain's default radius-1 pattern.
    """
    all_rows = []
    all_cols = []
    coupling_endpoints = _resolve_coupling_endpoints(eqset)
    read_through_alias_ids = {
        id(cp.target) for cp in eqset.couplings if getattr(cp, "read_through", False)
    }
    for comp in eqset.components:
        periodic_axes = _component_periodic_axes(comp)
        for eq in comp.equations:
            output_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
            for out_var in output_vars:
                row_offset = layout.offset_of((comp.name, out_var.name))
                row_size = out_var.size
                for inp in eq.inputs:
                    if isinstance(inp, GlobalVariable):
                        # Globals contribute to B = df/dlam (dense Schur block),
                        # not to the sparse df/du topology.
                        continue
                    if id(inp) in read_through_alias_ids:
                        # NO-DOF read-through alias — no local column; the source
                        # dependency is a dedicated cross-block (below).
                        continue
                    col_offset = layout.offset_of((comp.name, inp.name))
                    col_size = inp.size
                    num_nodes_attr = getattr(comp.domain, "num_nodes", None)
                    num_nodes = int(num_nodes_attr) if num_nodes_attr is not None else None
                    has_sparsity = hasattr(comp.domain, "sparsity_pattern")
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
                                    comp.domain,
                                    out_d,
                                    inp_d,
                                    periodic_axes,
                                    block_offsets=block_offsets,
                                )
                        else:
                            r_local, c_local = _node_field_block(
                                comp.domain, out_d, inp_d, periodic_axes
                            )
                    else:
                        r_local, c_local = _dense_fallback_block(
                            row_size, col_size, eq.name, inp.name
                        )
                    all_rows.append(r_local + row_offset)
                    all_cols.append(c_local + col_offset)

                    inp_key = (comp.name, inp.name)
                    if inp_key in coupling_endpoints:
                        src_comp_name, src_var_name, col_size_src, transfer_op = coupling_endpoints[
                            inp_key
                        ]
                        col_offset_src = layout.offset_of((src_comp_name, src_var_name))
                        if _op_declares_empty_sparsity(
                            transfer_op,
                            source_size=col_size_src,
                            target_size=col_size,
                        ):
                            r_cross, c_cross = _dense_cross_block(r_local, source_size=col_size_src)
                            cross_nnz = int(r_cross.size)
                            if cross_nnz > _MAX_DENSE_FALLBACK_NNZ:
                                raise ValueError(
                                    "lower(): TransferOp "
                                    f"{type(transfer_op).__name__!r} on "
                                    f"coupling into target {inp.name!r} "
                                    f"(component {comp.name!r}, sourced from "
                                    f"{src_var_name!r}) declares an empty "
                                    "sparsity(), so lowering falls back to a "
                                    "dense d(target)/d(source) cross-block "
                                    f"with {cross_nnz} entries, which exceeds "
                                    f"the dense cap of {_MAX_DENSE_FALLBACK_NNZ}. "
                                    "Override sparsity() to return the actual "
                                    "(target_rows, source_cols) pattern of your "
                                    "apply() so the cross-block stays sparse."
                                )
                        else:
                            r_cross, c_cross = _compose_cross_block_sparsity(
                                r_local,
                                c_local,
                                transfer_op,
                                source_size=col_size_src,
                                target_size=col_size,
                            )
                            cross_nnz = int(r_cross.size)
                            if cross_nnz > _MAX_DENSE_FALLBACK_NNZ:
                                raise ValueError(
                                    "lower(): coupling-induced cross-block with "
                                    f"{cross_nnz} entries exceeds the dense cap "
                                    f"for equation {eq.name!r} reading coupling "
                                    f"target {inp.name!r} sourced from "
                                    f"{src_var_name!r}."
                                )
                        if cross_nnz > 0:
                            _guard_block_degree(
                                r_cross,
                                c_cross,
                                where=(
                                    f"coupling into target {inp.name!r} "
                                    f"(component {comp.name!r}) sourced from "
                                    f"{src_var_name!r}"
                                ),
                            )
                            all_rows.append(r_cross + row_offset)
                            all_cols.append(c_cross + col_offset_src)
    # Read-through component->component cross-blocks: each consuming-equation
    # output row that reads a read-through alias depends on the op-selected
    # source columns (the NO-DOF alias has no local column).
    rt_rows, rt_cols = _read_through_cc_cross_block(eqset, layout)
    if rt_rows.size > 0:
        all_rows.append(rt_rows)
        all_cols.append(rt_cols)
    # Boundary flux closures: a FluxBC's density may read any field of its own
    # component, a coupled-in variable / read-through alias, or a global.  The
    # residual scatters into the BC's boundary rows, so declare those rows'
    # dependency on the read field's columns (a component field) or the transfer
    # source columns (a coupled-in value).
    flux_rows, flux_cols = _flux_bc_cross_block_sparsity(
        eqset,
        layout,
        resolved_fluxes=resolved_fluxes,
    )
    if flux_rows.size > 0:
        all_rows.append(flux_rows)
        all_cols.append(flux_cols)
    # Every state DOF needs a diagonal ``(i, i)`` slot.  The implicit NDF
    # iteration matrix places the mass term / BC-row identity onto pre-existing
    # diagonal slots; deriving the topology purely from ``reads`` would drop
    # these for any DOF whose equation does not read itself.  Emit the full
    # diagonal unconditionally; the ``np.unique`` below dedups against the
    # reads-derived blocks.
    n_state = int(layout.total_size)
    if n_state > 0:
        diag = np.arange(n_state, dtype=np.int64)
        all_rows.append(diag)
        all_cols.append(diag)

    # Family populations contribute (i) a batched per-member SELF-block and
    # (ii) family<->component CROSS-blocks for couplings touching the family.
    # Both are emitted AFTER the Component loop (families live strictly after all
    # components in the layout).  The self-block stamps ONE
    # member stencil ``count`` times via ``SparseJacobian.from_tiled_coo`` (no
    # dense fallback, no per-member ``sparsity_pattern`` call); the cross-blocks
    # resolve ONCE per coupling (not ``count`` times).
    families = getattr(eqset, "families", ())
    if families:
        fam_cross_rows, fam_cross_cols = _family_cross_block_sparsity(eqset, layout)
        if fam_cross_rows.size > 0:
            all_rows.append(fam_cross_rows)
            all_cols.append(fam_cross_cols)
        for fam in families:
            graph_rows, graph_cols = _family_graph_sparsity(fam, layout)
            if graph_rows.size > 0:
                all_rows.append(graph_rows)
                all_cols.append(graph_cols)
        # Border = macro (components) ∪ coupling (family cross-blocks) ∪ Schur
        # (globals contribute nothing to the sparse df/du topology) ∪ diagonal,
        # i.e. everything accumulated so far.  ``from_tiled_coo`` stamps each
        # family's tiles and unions with this border, deduping row-major to
        # match the component-path canonicalisation.
        border_rows = (
            np.concatenate(all_rows).astype(np.int32) if all_rows else np.zeros(0, dtype=np.int32)
        )
        border_cols = (
            np.concatenate(all_cols).astype(np.int32) if all_cols else np.zeros(0, dtype=np.int32)
        )
        for fam in families:
            sj = _family_self_block_sparsity(fam, layout, border_rows, border_cols, n_state)
            # Accumulate the prior family's tiles into the border for the next
            # family so the final union covers every family's self-block.
            border_rows = np.asarray(sj.rows, dtype=np.int32)
            border_cols = np.asarray(sj.cols, dtype=np.int32)
        # ``from_tiled_coo`` already deduped the full union row-major.
        return border_rows.astype(np.int32), border_cols.astype(np.int32)

    if not all_rows:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    rows = np.concatenate(all_rows).astype(np.int64)
    cols = np.concatenate(all_cols).astype(np.int64)
    pair = rows * np.int64(layout.total_size) + cols
    _unique, unique_idx = np.unique(pair, return_index=True)
    return rows[unique_idx].astype(np.int32), cols[unique_idx].astype(np.int32)


def _declared_member_stencil_offsets(fam) -> np.ndarray | None:
    """Offset set of a family's declared ``stencil_radius``, else ``None``.

    ``stencil_radius=r`` on a structured member grid declares every offset
    with at most ``r`` cells per axis (the ``(2r+1)**ndim`` box, which contains
    every composed axis-aligned operator of reach ``r``); ``None`` keeps the
    member domain's own radius-1 pattern.

    Raises:
        ValueError: when a radius is declared but the member domain is not a
            structured grid (no per-axis cell counts to generate offsets from).
    """
    radius = getattr(fam, "stencil_radius", None)
    if radius is None:
        return None
    from .stencil_infer import _grid_shape  # noqa: PLC0415

    shape = _grid_shape(fam.domain)
    if shape is None:
        raise ValueError(
            f"family {fam.name!r} declares stencil_radius={int(radius)}, but its "
            f"member domain {type(fam.domain).__name__!r} is not a structured "
            "grid, so no per-axis stencil can be generated from the radius. "
            "Use a structured grid member domain, or omit stencil_radius and "
            "declare the member topology through the domain's sparsity_pattern()."
        )
    ndim = len(shape)
    axis = np.arange(-int(radius), int(radius) + 1, dtype=np.int64)
    grids = np.meshgrid(*([axis] * ndim), indexing="ij")
    return np.stack([g.reshape(-1) for g in grids], axis=1)


def _family_self_block_sparsity(
    fam,
    layout: _BlockLayout,
    border_rows: np.ndarray,
    border_cols: np.ndarray,
    n_dofs: int,
) -> SparseJacobian:
    """Batched per-member self-block topology via a single stamped tile.

    Builds the ONE-member stencil (host-side, once — not per member) in the
    member's BLOCKED per-field coordinates and stamps it ``count`` times at
    member-major offsets ``fam_base + member * member_state_size`` through
    [SparseJacobian.from_tiled_coo][diff_soup._lowering.sparsity.SparseJacobian.from_tiled_coo], unioned with the supplied border
    (macro ∪ coupling ∪ Schur ∪ diagonal).  Returns the resulting
    [SparseJacobian][diff_soup._lowering.sparsity.SparseJacobian] (its ``.rows`` / ``.cols`` carry the full deduped
    COO topology).

    One member's slab blocks its fields in declaration order: member field
    ``v`` occupies the contiguous band ``[var_local_off : var_local_off +
    v.size]`` (see ``layout._layout`` / ``_FamilyResidualBlock``).  The tile is
    therefore built in THOSE blocked coordinates — for every declared
    ``(out_field, in_field)`` pair the domain node-stencil block is placed at
    the fields' local offsets, exactly the construction the component path
    uses ([_node_field_block][diff_soup._lowering.sparsity._node_field_block]).
    Flux callbacks have no reads tuple, so their written rows conservatively
    include every member field. The tile must NOT use the interleaved
    ``sparsity_pattern(n_vars=fields_per_member)`` (which orders DOFs
    node-major, ``n_vars*node + var``): stamping that onto the blocked member
    layout scatters every cross-field coupling (e.g. ``df_c/dmu``) to the
    wrong member columns, yielding a structurally wrong — and colour-
    compression-corrupting — Jacobian for any family with more than one member
    field.  A single-field member reduces to the plain scalar stencil exactly,
    so single-field families are unchanged.

    A family declaring ``stencil_radius=r`` replaces the domain's radius-1
    node-pair pattern in every node-field block by the pattern of all offsets
    within ``r`` cells per axis of the structured member grid (member kernels
    are not probed, so the declaration is the only way a wider member operator
    enters the topology).  A per-member scalar field (size 1) is not a node
    field: its blocks against the node fields are the exact ``1 x n`` /
    ``n x 1`` all-pairs blocks — a scalar couples to the whole member — and
    they are never counted as a dense fallback.

    There is no *implicit* dense fallback: a tile produced because a field
    cannot be interpreted on the member domain is rejected when it becomes the
    dense product. A domain may explicitly declare a dense one-member stencil;
    stamping that small tile across multiple independent members still exposes
    the block structure and retains bounded coloring.

    Raises:
        ValueError: if the member domain has no ``sparsity_pattern``, an
            implicit fallback is the dense product, or a ``stencil_radius``
            is declared on a member domain that is not a structured grid.
    """
    member_state_size = int(fam.member_state_size)
    domain = fam.domain
    if not hasattr(domain, "sparsity_pattern"):
        raise ValueError(
            f"family {fam.name!r}: member domain {type(domain).__name__!r} "
            "exposes no sparsity_pattern(); a Family self-block requires a "
            "sparse repeated stencil (no dense fallback)."
        )
    block_offsets = _declared_member_stencil_offsets(fam)
    # Member-field local offsets within one member's blocked slab.
    num_nodes_attr = getattr(domain, "num_nodes", None)
    num_nodes = int(num_nodes_attr) if num_nodes_attr is not None else None
    field_offsets: list[tuple[object, int]] = []
    loc = 0
    for v in fam.member_variables:
        field_offsets.append((v, loc))
        loc += int(v.size)
    # Declared cross-field tile in blocked coords. Member equations name their
    # state dependencies in ``inputs``; emit only those (out, in) blocks. A
    # member FluxBC has no separate reads declaration, so its written field is
    # conservatively allowed to read every member field. The global diagonal
    # was already added to the border, including constant rows with no reads.
    member_by_id = {id(variable): variable for variable in fam.member_variables}
    declared_pairs: set[tuple[int, int]] = set()
    for equation in fam.member_equations:
        outputs = equation.vars if isinstance(equation, MultiEquation) else (equation.var,)
        inputs = tuple(
            member_by_id[id(value)] for value in equation.inputs if id(value) in member_by_id
        )
        for output in outputs:
            for input_ in inputs:
                declared_pairs.add((id(output), id(input_)))
    for bc in fam.member_bcs:
        if isinstance(bc, FluxBC):
            for input_ in fam.member_variables:
                declared_pairs.add((id(bc.var), id(input_)))

    tile_r_list: list[np.ndarray] = []
    tile_c_list: list[np.ndarray] = []
    used_dense_fallback = False
    for out_v, out_off in field_offsets:
        out_d = _node_components(out_v, num_nodes)
        for in_v, in_off in field_offsets:
            if (id(out_v), id(in_v)) not in declared_pairs:
                continue
            in_d = _node_components(in_v, num_nodes)
            if out_d is not None and in_d is not None:
                r_blk, c_blk = _node_field_block(domain, out_d, in_d, block_offsets=block_offsets)
            elif int(out_v.size) == 1 or int(in_v.size) == 1:
                # A per-member scalar couples to every DOF of the other field
                # (and to itself): the all-pairs block is exact, not a fallback.
                r_blk = np.repeat(np.arange(int(out_v.size), dtype=np.int64), int(in_v.size))
                c_blk = np.tile(np.arange(int(in_v.size), dtype=np.int64), int(out_v.size))
            else:
                used_dense_fallback = True
                r_blk, c_blk = _dense_fallback_block(
                    int(out_v.size),
                    int(in_v.size),
                    f"{fam.name}.{out_v.name}",
                    in_v.name,
                )
            tile_r_list.append(np.asarray(r_blk, dtype=np.int64) + out_off)
            tile_c_list.append(np.asarray(c_blk, dtype=np.int64) + in_off)
    r1 = np.concatenate(tile_r_list) if tile_r_list else np.empty((0,), dtype=np.int64)
    c1 = np.concatenate(tile_c_list) if tile_c_list else np.empty((0,), dtype=np.int64)
    # Reject only an implicit dense fallback. An explicitly declared dense
    # member stencil remains a genuine repeated block when ``count > 1`` and
    # ``from_tiled_coo`` colors the stamped blocks together.
    if used_dense_fallback and int(r1.size) >= member_state_size * member_state_size:
        raise ValueError(
            f"family {fam.name!r}: member stencil declares the dense product "
            f"({r1.size} entries for a {member_state_size}-DOF member); "
            "refusing the dense fallback; supply a sparse member stencil."
        )
    fam_base = layout.offset_of((fam.name, fam.member_variables[0].name))
    base = fam_base + np.arange(int(fam.count), dtype=np.int64) * member_state_size
    return SparseJacobian.from_tiled_coo(
        tile_rows=r1,
        tile_cols=c1,
        tile_size=member_state_size,
        tile_count=int(fam.count),
        tile_offsets=base,
        border_rows=border_rows,
        border_cols=border_cols,
        n_dofs=int(n_dofs),
    )


def _family_graph_sparsity(fam, layout: _BlockLayout) -> tuple[np.ndarray, np.ndarray]:
    """Exact endpoint blocks declared by a family's graph exchanges.

    A callback receives whole endpoint fields, so one written field row block
    conservatively depends on every DOF of each field named in ``reads`` or
    ``weight_reads`` at both incident members. No other cross-member block is
    emitted. The existing row-degree guard applies to high-degree hubs.
    """
    exchanges = tuple(getattr(fam, "graph_exchanges", ()))
    if not exchanges:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    graph_by_name = {graph.name: graph for graph in getattr(fam, "graphs", ())}
    field_info: dict[str, tuple[int, int]] = {}
    local = 0
    for variable in fam.member_variables:
        field_info[variable.name] = (local, int(variable.size))
        local += int(variable.size)
    member_state = int(fam.member_state_size)
    fam_base = layout.offset_of((fam.name, fam.member_variables[0].name))
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []

    for exchange in exchanges:
        graph = graph_by_name.get(exchange.graph)
        if graph is None:
            raise ValueError(
                f"family {fam.name!r} graph exchange {exchange.name!r} names "
                f"unknown graph {exchange.graph!r}; declare the graph on the "
                "same family."
            )
        write_off, write_size = field_info[exchange.writes]
        dependencies = tuple(dict.fromkeys(exchange.reads + exchange.weight_reads))
        exchange_rows: list[np.ndarray] = []
        exchange_cols: list[np.ndarray] = []
        for source, target in zip(graph.edge_sources, graph.edge_targets):
            for row_member in (int(source), int(target)):
                row_band = (
                    fam_base
                    + row_member * member_state
                    + write_off
                    + np.arange(write_size, dtype=np.int64)
                )
                for dependency in dependencies:
                    dep_off, dep_size = field_info[dependency]
                    for col_member in (int(source), int(target)):
                        col_band = (
                            fam_base
                            + col_member * member_state
                            + dep_off
                            + np.arange(dep_size, dtype=np.int64)
                        )
                        exchange_rows.append(np.repeat(row_band, dep_size))
                        exchange_cols.append(np.tile(col_band, write_size))
        if exchange_rows:
            rows = np.concatenate(exchange_rows)
            cols = np.concatenate(exchange_cols)
            _guard_block_degree(
                rows,
                cols,
                where=(f"family graph exchange {exchange.name!r} on graph {graph.name!r}"),
                remedy=(
                    "Reduce or split the graph's hub degree, or localize "
                    "reads/weight_reads so each endpoint residual depends "
                    "on fewer source columns."
                ),
            )
            rows_out.append(rows)
            cols_out.append(cols)

    if not rows_out:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty
    return (
        np.concatenate(rows_out).astype(np.int32),
        np.concatenate(cols_out).astype(np.int32),
    )


def _read_through_node_local_cross_block(
    eqset: EquationSet,
    layout: _BlockLayout,
    comp_name: str,
    alias_var,
    op,
    *,
    source_size: int,
    src_off: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-block topology for a node-local read-through alias.

    A read-through alias carries no state DOF; the consuming closure reads it by
    name inside its view.  The read is node-local: a reader's output at node ``i``
    depends on the alias at node ``i`` and that node's domain-stencil neighbours.
    So each consuming equation's node-pair stencil (output DOF -> alias DOF),
    composed with the transfer op's (alias row -> source col) sparsity, gives the
    exact source columns the reader rows depend on -- the per-node fan-in, not the
    whole source population.

    For each equation in ``comp_name`` whose ``inputs`` contains ``alias_var`` (by
    identity) and each of its output Variables, build the consuming domain's
    node-field block (output DOF -> alias DOF) and compose it with ``op`` over the
    alias rows, offsetting rows by the output Variable's offset and columns by
    ``src_off``.  When the consuming domain exposes no stencil, or the output or
    alias Variable is not a node field, fall back to the conservative superset:
    every reader output row depends on every op-selected source column.

    ``source_size`` is the flat size of the transfer source (a component variable
    or a whole family slab) and ``src_off`` its column offset in the global state.
    Returns ``(rows, cols)`` in global indexing (empty when the alias is read by no
    equation in ``comp_name`` or the op selects no source column).
    """
    comp = next((c for c in eqset.components if c.name == comp_name), None)
    if comp is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    num_nodes_attr = getattr(comp.domain, "num_nodes", None)
    num_nodes = int(num_nodes_attr) if num_nodes_attr is not None else None
    has_sparsity = hasattr(comp.domain, "sparsity_pattern")
    alias_d = _node_components(alias_var, num_nodes) if has_sparsity else None
    _transfer_rows, transfer_cols = op.sparsity(int(source_size), int(alias_var.size))
    op_cols_all = np.unique(np.asarray(transfer_cols, dtype=np.int64))

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for eq in comp.equations:
        if not any(inp is alias_var for inp in eq.inputs):
            continue
        out_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
        for out_var in out_vars:
            row_off = layout.offset_of((comp.name, out_var.name))
            out_d = _node_components(out_var, num_nodes) if has_sparsity else None
            if out_d is not None and alias_d is not None:
                r_local, c_local = _node_field_block(comp.domain, out_d, alias_d)
                r_cross, c_cross = _compose_cross_block_sparsity(
                    r_local,
                    c_local,
                    op,
                    source_size=int(source_size),
                    target_size=int(alias_var.size),
                )
                if r_cross.size == 0:
                    continue
                _guard_block_degree(
                    r_cross,
                    c_cross,
                    where=(
                        f"read-through alias {alias_var.name!r} read in component {comp_name!r}"
                    ),
                )
                rows_list.append(np.asarray(r_cross, dtype=np.int64) + row_off)
                cols_list.append(np.asarray(c_cross, dtype=np.int64) + src_off)
            elif op_cols_all.size > 0:
                out_rows = np.arange(int(out_var.size), dtype=np.int64) + row_off
                _guard_block_degree(
                    np.repeat(out_rows, op_cols_all.size),
                    np.tile(op_cols_all, out_rows.size) + src_off,
                    where=(
                        f"read-through alias {alias_var.name!r} read in component {comp_name!r}"
                    ),
                )
                rows_list.append(np.repeat(out_rows, op_cols_all.size))
                cols_list.append(np.tile(op_cols_all, out_rows.size) + src_off)

    if not rows_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows_list), np.concatenate(cols_list)


def _read_through_cc_cross_block(
    eqset: EquationSet, layout: _BlockLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-block topology for component->component read-through couplings.

    A read-through alias is injected by name into the consuming closure's view
    with no state DOF, so the dependency ``d(consuming_row)/d(source_col)`` must
    be declared from the coupling geometry.  Each consuming equation reads the
    alias node-locally: its output node ``i`` depends on the alias at node ``i``
    and at the consuming domain's stencil neighbours of ``i``.  For each
    read-through coupling whose source is a component variable: compose that
    consuming-domain node stencil (reader output DOF -> alias DOF) with the op's
    ``(alias_row, source_col)`` sparsity so each reader row depends only on the
    source columns its node-local read reaches, not the whole source field.  When
    the consuming domain carries no node stencil, or the alias / output variable
    is not a node field on it, fall back to the conservative dependency of every
    reader row on every source column the op reads.

    Returns ``(rows, cols)`` in global indexing (empty if no comp->comp
    read-through coupling exists).
    """
    rt_alias_owner = _read_through_alias_owner(eqset)
    if not rt_alias_owner:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    family_member_ids = {
        id(v) for fam in getattr(eqset, "families", ()) for v in fam.member_variables
    }
    comp_var_owner: dict[int, tuple[str, str, int]] = {}
    for comp in eqset.components:
        for v in comp.variables:
            comp_var_owner[id(v)] = (comp.name, v.name, int(v.size))

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for cp in eqset.couplings:
        if not getattr(cp, "read_through", False):
            continue
        if id(cp.source) in family_member_ids:
            continue  # family->component surface read-through (handled elsewhere).
        owner = rt_alias_owner.get(id(cp.target))
        src_info = comp_var_owner.get(id(cp.source))
        if owner is None or src_info is None:
            continue
        tgt_comp_name, alias_var = owner
        src_comp, src_var, src_size = src_info
        src_off = layout.offset_of((src_comp, src_var))
        r, c = _read_through_node_local_cross_block(
            eqset,
            layout,
            tgt_comp_name,
            alias_var,
            cp.op,
            source_size=int(src_size),
            src_off=src_off,
        )
        if r.size:
            rows_list.append(r)
            cols_list.append(c)

    if not rows_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows_list), np.concatenate(cols_list)


def _flux_self_stencil(resolved, name: str, fi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local ``(row, col)`` stencil of an FEM flux reading the field ``name``.

    A facet-local flux (the default) is evaluated once per facet, so its
    value can depend only on its own facet's vertices: the facet
    vertex-to-vertex stencil is exact, with nothing to probe and nothing a
    hidden branch can hide.  ``(rows, cols)`` are in the read field's local
    indexing.  A ``cross_facet=True`` flux never reaches this stencil — its
    caller declares the conservative boundary-wide pattern instead.
    """
    del resolved, name
    nv = fi.shape[1]  # vertices per face
    local_rows = np.repeat(fi, nv, axis=1).reshape(-1)
    local_cols = np.tile(fi, (1, nv)).reshape(-1)
    return local_rows, local_cols


def _flux_bc_cross_block_sparsity(
    eqset: EquationSet,
    layout: _BlockLayout,
    *,
    resolved_fluxes=(),
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian topology induced by boundary flux closures.

    A component ``FluxBC``'s density may read any field of its own component, a
    variable coupled into that component (an ordinary coupling target or a
    NO-DOF read-through alias), or a global (see
    [_flux_fn_accessed_names][diff_soup._lowering.bcs._flux_fn_accessed_names]).  The residual
    scatters the flux into the BC's boundary rows, so those rows depend on:

    * the read field's own columns — the per-face vertex-vertex stencil on FEM
      domains (the flux is a per-face quantity scattered to that face's
      vertices, so ``dR_a/du_b`` is nonzero only for vertices ``a, b`` sharing a
      face — exactly [_robin_face_stencil_pairs][diff_soup._lowering.bcs._robin_face_stencil_pairs]).
      Non-FEM (grid/FVM) fluxes carry no facet geometry, so they fall back to the
      conservative whole field (``boundary_rows × all columns``);
    * the transfer SOURCE columns for a coupled-in value — the post-transfer
      value the closure reads is a function of the source.  A flux reads that
      value at its boundary FACE nodes, so on an FEM domain a boundary row
      depends on a source column only through the alias values at the nodes
      sharing a face with it: compose the per-face vertex stencil with the op's
      ``sparsity`` so boundary row ``a`` depends on source col ``k`` exactly when
      a face node ``b`` shares a face with ``a`` and the op maps node ``b`` to
      col ``k``.  This holds when the alias is a full per-node field on the BC
      domain (its node index matches the BC field's); a compact or reduced
      alias, an op that declares no source sparsity, or a non-FEM grid with no
      facet geometry keeps the conservative ``boundary_rows × source_cols``
      block.

    Globals feed the dense ``df/dlam`` Schur block (built by ``jax.jacfwd``), not
    the sparse ``df/du`` topology, so they contribute nothing here.  Declaring a
    superset is safe under the CPR-coloured Jacobian (extra entries recover as
    zero) but a WHOLE-field self block makes every boundary row conflict with
    every column, collapsing the colouring to ``n_colors == n_dofs`` and the
    vmap-over-colours assembly to ``O(n_dofs^2)``; the precise FEM facet stencil
    avoids this while staying a correct superset.  Under-declaring corrupts the
    colour compression, so the op-composed source columns remain mandatory.

    Returns ``(rows, cols)`` in global indexing (empty when no ``FluxBC`` reads
    state).
    """
    if not resolved_fluxes:
        from .bcs import _bc_dofs  # noqa: PLC0415

        resolved_fluxes = _bc_dofs(eqset, layout).flux

    coupling_endpoints = _resolve_coupling_endpoints(eqset)
    rt_alias_owner = _read_through_alias_owner(eqset)
    alias_size_by_ref = {
        (owner, alias_var.name): int(alias_var.size) for owner, alias_var in rt_alias_owner.values()
    }
    global_names = {g.name for g in eqset.globals}

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    comp_by_name = {comp.name: comp for comp in eqset.components}
    for resolved in resolved_fluxes:
        comp = comp_by_name[resolved.comp_name]
        local_var_by_name = {v.name: v for v in comp.variables}
        boundary_rows = np.asarray(resolved.residual_rows, dtype=np.int64)
        for name in resolved.accessed_names:
            nodal_read = name.startswith("nodal:")
            if nodal_read:
                name = name.removeprefix("nodal:")
                variable = local_var_by_name.get(name)
                if variable is not None:
                    offset = layout.offset_of((comp.name, name))
                    columns = np.arange(variable.size, dtype=np.int64) + offset
                    rows_list.append(np.repeat(boundary_rows, columns.size))
                    cols_list.append(np.tile(columns, boundary_rows.size))
                    continue
            if name in global_names:
                continue  # df/dlam (dense Schur block), not df/du.
            endpoint = coupling_endpoints.get((comp.name, name))
            if endpoint is not None:
                # Coupled-in target / read-through alias: the read value is a
                # function of the transfer SOURCE, so depend on source cols.
                src_comp, src_var, src_size, op = endpoint
                if name in local_var_by_name:
                    target_size = int(local_var_by_name[name].size)
                else:
                    target_size = alias_size_by_ref.get((comp.name, name))
                if target_size is None:
                    continue
                src_off = layout.offset_of((src_comp, src_var))
                op_empty = _op_declares_empty_sparsity(
                    op,
                    source_size=int(src_size),
                    target_size=int(target_size),
                )
                fi = getattr(resolved, "face_indices", None)
                fi = None if fi is None else np.asarray(fi, dtype=np.int64)
                num_nodes = int(getattr(comp.domain, "num_nodes", 0) or 0)
                # A flux reads a coupled-in value at its boundary FACE nodes, so a
                # boundary row depends on the source only through the alias values
                # at the nodes sharing a face with it. When the alias is a full
                # per-node field on this domain (its node index matches the BC
                # field's), compose the per-face vertex stencil with the op's
                # source sparsity: boundary row a depends on source col k iff a
                # face node b shares a face with a AND the op maps node b -> col k.
                # Otherwise (a non-FEM grid with no facet geometry, a compact or
                # reduced alias, or an op that declares no source sparsity) keep
                # the conservative boundary_rows x source-cols block.
                if (
                    resolved.is_fem
                    and not nodal_read
                    and fi is not None
                    and fi.size
                    and not op_empty
                    and num_nodes > 0
                    and int(target_size) == num_nodes
                    and int(resolved.var_size) == num_nodes
                ):
                    # Boundary row a depends on source col k iff some face node b
                    # the flux reads at a face containing a maps (via the op) to k.
                    # The read nodes b come from the flux's AD-measured per-face
                    # read columns (the per-face stencil for a face-local read,
                    # wider for a non-local one), composed with the op.
                    s_rows, s_cols = _flux_self_stencil(resolved, name, fi)
                    r_cross, c_cross = _compose_cross_block_sparsity(
                        s_rows,
                        s_cols,
                        op,
                        source_size=int(src_size),
                        target_size=int(target_size),
                    )
                    if r_cross.size == 0:
                        continue
                    _flux_guard_dense_cap(
                        int(r_cross.size),
                        comp.name,
                        resolved.var_name,
                        name,
                    )
                    _guard_block_degree(
                        r_cross,
                        c_cross,
                        where=(
                            f"flux on {resolved.var_name!r} (component "
                            f"{comp.name!r}) reading coupled-in {name!r}"
                        ),
                    )
                    rows_list.append(np.asarray(r_cross, dtype=np.int64) + resolved.var_offset)
                    cols_list.append(np.asarray(c_cross, dtype=np.int64) + src_off)
                    continue
                if op_empty:
                    op_cols = np.arange(int(src_size), dtype=np.int64)
                else:
                    _op_rows, op_cols = op.sparsity(
                        int(src_size),
                        int(target_size),
                    )
                    op_cols = np.unique(np.asarray(op_cols, dtype=np.int64))
                if op_cols.size == 0:
                    continue
                _flux_guard_dense_cap(
                    int(boundary_rows.size) * int(op_cols.size),
                    comp.name,
                    resolved.var_name,
                    name,
                )
                _guard_block_degree(
                    np.repeat(boundary_rows, op_cols.size),
                    np.tile(op_cols, boundary_rows.size) + src_off,
                    where=(
                        f"flux on {resolved.var_name!r} (component "
                        f"{comp.name!r}) reading coupled-in {name!r}"
                    ),
                )
                rows_list.append(np.repeat(boundary_rows, op_cols.size))
                cols_list.append(np.tile(op_cols, boundary_rows.size) + src_off)
            elif name in local_var_by_name:
                # State-dependent flux reading a field of its own component —
                # the BC's own field or a sibling evaluated at the same boundary.
                # The flux is a per-face quantity scattered to that face's
                # vertices, so dR_a/du_b is nonzero only for vertices a,b sharing
                # a face: the per-face vertex-vertex stencil (exactly as RobinBC,
                # see _robin_face_stencil_pairs), NOT the whole field. Declaring
                # whole-field is a correct superset but collapses the CPR
                # colouring to n_colors == n_dofs (every boundary row conflicts
                # with every column), which makes the vmap-over-colours Jacobian
                # assembly O(n_dofs^2). On FEM domains use the precise facet
                # stencil, which indexes rows and columns alike by node and so
                # needs the read field to share the BC field's node indexing;
                # non-FEM (grid/FVM) fluxes carry no facet geometry, so they keep
                # the conservative whole-field.
                field_var = local_var_by_name[name]
                row_off = int(resolved.var_offset)
                col_off = layout.offset_of((comp.name, name))
                where = f"flux on {resolved.var_name!r} (component {comp.name!r}) reading {name!r}"
                fi = getattr(resolved, "face_indices", None)
                fi = None if fi is None else np.asarray(fi, dtype=np.int64)
                if (
                    resolved.is_fem
                    and fi is not None
                    and fi.size
                    and int(field_var.size) == int(resolved.var_size)
                    and not getattr(resolved, "cross_facet", False)
                ):
                    # A facet-local flux is evaluated per facet, so the facet
                    # vertex-vertex stencil is its exact pattern; a declared
                    # cross_facet flux falls to the conservative whole-field
                    # branch below.
                    s_rows, s_cols = _flux_self_stencil(resolved, name, fi)
                    _flux_guard_dense_cap(
                        int(s_rows.size),
                        comp.name,
                        resolved.var_name,
                        name,
                    )
                    _guard_block_degree(s_rows, s_cols, where=where)
                    rows_list.append(s_rows + row_off)
                    cols_list.append(s_cols + col_off)
                else:
                    field_cols = np.arange(int(field_var.size), dtype=np.int64)
                    _flux_guard_dense_cap(
                        int(boundary_rows.size) * int(field_cols.size),
                        comp.name,
                        resolved.var_name,
                        name,
                    )
                    _guard_block_degree(
                        np.repeat(boundary_rows, field_cols.size),
                        np.tile(field_cols, boundary_rows.size) + col_off,
                        where=where,
                    )
                    rows_list.append(np.repeat(boundary_rows, field_cols.size))
                    cols_list.append(np.tile(field_cols, boundary_rows.size) + col_off)
            # else: unreachable — the probe fails closed on any other name.

    if not rows_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows_list), np.concatenate(cols_list)


def _flux_guard_dense_cap(nnz: int, comp_name: str, bc_var_name: str, read_name: str) -> None:
    """Reject a flux-induced cross-block that exceeds the dense nnz cap.

    Raises:
        ValueError: when ``nnz`` exceeds ``_MAX_DENSE_FALLBACK_NNZ``.
    """
    if nnz > _MAX_DENSE_FALLBACK_NNZ:
        raise ValueError(
            f"lower(): FluxBC on {bc_var_name!r} (component {comp_name!r}) reading "
            f"{read_name!r} induces a Jacobian cross-block with {nnz} entries, "
            f"which exceeds the dense cap of {_MAX_DENSE_FALLBACK_NNZ}. Restrict "
            "the flux to a smaller boundary or give the transfer op a tighter "
            "sparsity()."
        )


def _family_cross_block_sparsity(
    eqset: EquationSet, layout: _BlockLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-block topology for couplings touching a Family.

    Each coupling whose source OR target is a Family member contributes a
    single off-diagonal block, resolved ONCE per coupling (not ``count``
    times).  Direction:

    * **scatter** (family source → component target): the component equation
      reads the scattered value, so component target rows depend on the
      per-member source columns.  The ``FamilyScatterOp.sparsity`` already
      declares the population pattern (one nonzero per member); composing it
      with the identity reading-pattern on the target rows (via
      ``_compose_cross_block_sparsity``) reproduces it exactly.
    * **gather** (component source → family target): each member reads the
      gathered component value, so family target rows depend on component
      source columns; same single composition over the member axis.

    Returns ``(rows, cols)`` in global indexing (empty arrays if no family
    coupling exists).
    """
    family_var_owner: dict[int, tuple[str, str]] = {}
    member_state_by_fam: dict[str, int] = {}
    for fam in eqset.families:
        member_state_by_fam[fam.name] = int(fam.member_state_size)
        for v in fam.member_variables:
            family_var_owner[id(v)] = (fam.name, v.name)
    comp_var_owner: dict[int, tuple[str, str, int]] = {}
    for comp in eqset.components:
        for v in comp.variables:
            comp_var_owner[id(v)] = (comp.name, v.name, int(v.size))
    # Read-through family-surface aliases are NOT component variables; resolve
    # the consuming component + the per-equation reader rows so the cross-block
    # ``d(consuming_row)/d(family_slab_col)`` is declared.
    rt_alias_owner = _read_through_alias_owner(eqset)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for cp in eqset.couplings:
        src_is_fam = id(cp.source) in family_var_owner
        tgt_is_fam = id(cp.target) in family_var_owner
        if not (src_is_fam or tgt_is_fam):
            continue  # pure component↔component coupling — handled above.
        if src_is_fam and tgt_is_fam:
            raise ValueError(
                f"coupling {cp.name!r} connects two Family members; "
                "family↔family couplings are out of scope (separate families)."
            )

        if src_is_fam and getattr(cp, "read_through", False):
            # gather_surface read-through: family member surface trace injected
            # into a component closure's MergedView under an alias.  The writer
            # rows are the consuming component equations that READ the alias; the
            # source columns are the per-member surface DOFs off the family slab.
            owner = rt_alias_owner.get(id(cp.target))
            if owner is None:
                continue  # orphan alias — endpoint resolution errors elsewhere.
            tgt_comp_name, alias_var = owner
            fam_name, src_var = family_var_owner[id(cp.source)]
            fam = next(f for f in eqset.families if f.name == fam_name)
            source_size = fam.count * member_state_by_fam[fam_name]
            src_off = layout.offset_of((fam_name, src_var))
            # The alias is read NODE-LOCALLY: a reader's output at node i depends
            # on the alias at node i and that node's domain-stencil neighbours.
            # Composing the consuming domain's node-pair stencil (reader output
            # DOF -> alias DOF) with the scatter op's (alias row -> family-slab
            # col) pattern yields the true per-node fan-in.  Crossing every reader
            # row with the whole member population would instead make each reader
            # row conflict with every member surface column, collapsing the CPR
            # column colouring toward one colour per DOF.
            rt_rows, rt_cols = _read_through_node_local_cross_block(
                eqset,
                layout,
                tgt_comp_name,
                alias_var,
                cp.op,
                source_size=source_size,
                src_off=src_off,
            )
            if rt_rows.size > 0:
                rows_list.append(rt_rows)
                cols_list.append(rt_cols)
            continue

        if src_is_fam:
            # scatter: family source -> component target.  Writer = component
            # (reads the scattered value); source = family slab.  The
            # ``FamilyScatterOp.sparsity`` is a POPULATION pattern (one nonzero
            # per member, columns already in family-slab-local indexing), so a
            # single composition over the identity reading-pattern on the
            # component target rows reproduces it exactly (NOT per member).
            fam_name, src_var = family_var_owner[id(cp.source)]
            fam = next(f for f in eqset.families if f.name == fam_name)
            source_size = fam.count * member_state_by_fam[fam_name]
            tgt_comp, tgt_var, target_size = comp_var_owner[id(cp.target)]
            writer_off = layout.offset_of((tgt_comp, tgt_var))
            src_off = layout.offset_of((fam_name, src_var))
            op_rows, _op_cols = cp.op.sparsity(source_size, target_size)
            op_rows = np.unique(np.asarray(op_rows, dtype=np.int64))
            r_cross, c_cross = _compose_cross_block_sparsity(
                op_rows,
                op_rows,
                cp.op,
                source_size=source_size,
                target_size=target_size,
            )
            if r_cross.size == 0:
                continue
            _guard_block_degree(
                r_cross,
                c_cross,
                where=(
                    f"family scatter into {tgt_var!r} (component {tgt_comp!r}) "
                    f"from family {fam_name!r}"
                ),
            )
            rows_list.append(np.asarray(r_cross, dtype=np.int64) + writer_off)
            cols_list.append(np.asarray(c_cross, dtype=np.int64) + src_off)
        else:
            # gather: component source -> family target.  The op maps a
            # per-member gathered value (``target_size == count``) to its
            # component source column; a SINGLE composition over the member
            # axis gives ``(member, source_col)``.  Each member reads its
            # gathered value inside its own slab, so the dependency is the
            # safe Jacobian superset: member ``m``'s whole slab depends on the
            # gathered source column (the exact member-row support is the
            # residual's job — over-declaring here is correct topology).
            fam_name, tgt_var = family_var_owner[id(cp.target)]
            fam = next(f for f in eqset.families if f.name == fam_name)
            member_state = member_state_by_fam[fam_name]
            fam_base = layout.offset_of((fam_name, tgt_var))
            src_comp, src_var, source_size = comp_var_owner[id(cp.source)]
            src_off = layout.offset_of((src_comp, src_var))
            count = int(fam.count)
            op_rows, _op_cols = cp.op.sparsity(source_size, count)
            op_rows = np.unique(np.asarray(op_rows, dtype=np.int64))
            member_idx, src_cols = _compose_cross_block_sparsity(
                op_rows,
                op_rows,
                cp.op,
                source_size=source_size,
                target_size=count,
            )
            if member_idx.size == 0:
                continue
            member_idx = np.asarray(member_idx, dtype=np.int64)
            src_cols = np.asarray(src_cols, dtype=np.int64)
            # Expand over the member's own DOF axis: member m's full slab rows
            # [fam_base + m*member_state, +member_state) all depend on src_col.
            local = np.arange(member_state, dtype=np.int64)
            exp_rows = (fam_base + (member_idx[:, None] * member_state) + local[None, :]).reshape(
                -1
            )
            exp_cols = np.repeat(src_cols, member_state)
            _guard_block_degree(
                exp_rows,
                exp_cols + src_off,
                where=(
                    f"family gather into {fam_name!r} from {src_var!r} (component {src_comp!r})"
                ),
            )
            rows_list.append(exp_rows)
            cols_list.append(exp_cols + src_off)

    if not rows_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows_list), np.concatenate(cols_list)
