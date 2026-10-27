"""Cross-component coupling resolution.

- ``_resolve_coupling_endpoints``: owner-aware (VarRef-keyed) lookup for
  every coupling's source/target component pairs.
- ``_resolve_family_couplings``: owner-aware resolution of the family-
  touching couplings (scatter / gather) into
  [FamilyCoupling][diff_soup._lowering.couplings.FamilyCoupling]
  records, used by the residual-assembly family branch.
- ``_compose_cross_block_sparsity``: composes the equation-local
  dependency rows with the transfer-op's source columns to produce the
  cross-block (rows, cols) tuple.

This module lives under ``diff_soup/_lowering`` and may import
``diff_soup.problem._ir`` IR nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diff_soup.problem._ir.equation_set import EquationSet


@dataclass(frozen=True)
class FamilyCoupling:
    """Resolved family-touching coupling (scatter or gather).

    A coupling whose source OR target is a
    [Family][diff_soup.problem._ir.family.Family] member Variable.
    Resolution is by Python ``is`` against
    ``eqset.families`` / ``eqset.components`` (host-side, before tracing) so a
    structurally-equal Variable in another owner cannot misroute the endpoint.

    Direction:

    * ``"scatter"`` — family member surface trace -> component target row.  The
      ``op`` (a
      [FamilyScatterOp][diff_soup.problem._ir.coupling.FamilyScatterOp]) reads
      each member's surface DOF off the family slab and scatter-**ADDs** the
      per-member contribution into the target component residual rows.
    * ``"gather"`` — component source -> family member target.  The ``op`` (a
      [NodalExtractOp][diff_soup.problem._ir.coupling.NodalExtractOp]) maps
      the component
      source to one value per member; each member reads its own slice inside the
      batched member kernel.
    * ``"gather_surface"`` — READ-THROUGH family member surface trace ->
      component closure (``read_through=True``, source is a family member).
      Same slab geometry as ``"scatter"`` (each member's surface DOF mapped via
      ``op.apply`` to the target nodes) but the result is INJECTED into the
      consuming component closure's MergedView under the alias, NOT scatter-ADDed
      into residual rows — so it allocates NO state DOF.

    ``read_through`` is ``True`` for a ``"gather_surface"`` record (no DOF/row;
    injected by name), ``False`` for ``"scatter"`` / ``"gather"``.

    Attributes:
        name: The originating [Coupling][diff_soup.problem._ir.coupling.Coupling]
            name.
        direction: ``"scatter"`` or ``"gather"``.
        fam_name: The owning [Family][diff_soup.problem._ir.family.Family] name.
        family: Always ``True`` (the family-owner qualifier; mirrors
            [VarRef.family][diff_soup.problem._ir.var_ref.VarRef.family]).
        comp_name: The COMPONENT endpoint name — the target component for a
            scatter, the source component for a gather.  Used by the residual
            branch to look up the component variable slice.
        comp_var_name: The component-endpoint Variable name.
        source_var_name: Source Variable name (the member var for scatter, the
            component var for gather).
        source_size: Source slab size (the full family slab for scatter; the
            component var size for gather).
        target_comp: For a scatter, the target Component name; for a gather, the
            family name (the member is the target).
        target_var_name: Target Variable name (the component var for scatter, the
            member var for gather).
        op: The [TransferOp][diff_soup.problem._ir.coupling.TransferOp].
    """

    name: str
    direction: str
    fam_name: str
    family: bool
    comp_name: str
    comp_var_name: str
    source_var_name: str
    source_size: int
    target_comp: str
    target_var_name: str
    op: object
    read_through: bool = False
    alias: object = None


def _read_through_alias_owner(eqset: EquationSet) -> dict[int, tuple[str, object]]:
    """Map each read-through alias ``id`` to ``(consuming_comp_name, alias_var)``.

    A read-through coupling's ``target`` (the alias) is NOT a member of any
    ``Component.variables`` — it has no state DOF.  Its owning component is taken
    from the coupling's explicit ``target_owner`` (the authoring ``onto=``) when
    set; this is authoritative and needs no equation scan, so an alias read ONLY
    by a boundary-condition callback — and by no equation — still resolves.

    When ``target_owner`` is ``None`` the owner is inferred by scanning each
    equation's ``inputs`` for the alias (by Python ``is``).  An alias inferred to
    be read by equations in more than one component is rejected (a read-through
    alias is component-local by construction; name the consuming component with
    ``onto=`` if you need the same source in two).

    Returns ``{id(alias): (comp_name, alias_var)}`` for every read-through
    coupling whose alias is owned or read.  Aliases that are neither (orphans)
    are omitted; callers raise a clearer error.
    """
    rt_couplings = [cp for cp in eqset.couplings if getattr(cp, "read_through", False)]
    alias_ids = {id(cp.target): cp.target for cp in rt_couplings}
    if not alias_ids:
        return {}
    owner: dict[int, tuple[str, object]] = {}
    # Explicit target_owner (authoring onto=) is authoritative — no equation scan.
    for cp in rt_couplings:
        target_owner = getattr(cp, "target_owner", None)
        if target_owner is not None:
            owner[id(cp.target)] = (target_owner, cp.target)
    # Inferred owner for an alias left without an explicit owner: the
    # consuming component is the one whose equation reads it.
    seen_comp: dict[int, str] = {}
    for comp in eqset.components:
        for eq in comp.equations:
            for inp in eq.inputs:
                aid = id(inp)
                if aid not in alias_ids or aid in owner:
                    continue
                prior = seen_comp.get(aid)
                if prior is not None and prior != comp.name:
                    raise ValueError(
                        f"read-through alias {alias_ids[aid].name!r} is read by "
                        f"equations in both component {prior!r} and "
                        f"{comp.name!r}; a read-through alias is component-local "
                        "(declare one alias per consuming component, or set "
                        "onto= to name the consuming component explicitly)."
                    )
                seen_comp[aid] = comp.name
                owner[aid] = (comp.name, alias_ids[aid])
    return owner


def _resolve_coupling_endpoints(
    eqset: EquationSet,
) -> dict[tuple[str, str], tuple[str, str, int, object]]:
    """Resolve every coupling's (target_comp, target_var_name) to its source.

    Resolution is by Python ``is`` against ``eqset.components``
    (host-side, before tracing) so structurally-equal Variables in
    different components do not collide.  Returns
    ``{(tgt_comp_name, tgt_var_name): (src_comp_name, src_var_name,
    src_size, transfer_op)}``.
    """
    family_member_ids = {
        id(v) for fam in getattr(eqset, "families", ()) for v in fam.member_variables
    }
    out: dict[tuple[str, str], tuple[str, str, int, object]] = {}
    rt_alias_owner = _read_through_alias_owner(eqset)
    comp_var_owner_full: dict[int, tuple[str, str, int]] = {}
    for comp in eqset.components:
        for v in comp.variables:
            comp_var_owner_full[id(v)] = (comp.name, v.name, int(v.size))
    # Unified duplicate-target guard: a target slot (component name, var name)
    # may be injected into by at most ONE coupling — component OR family.  The
    # second injector would silently overwrite (component SET) or double-count
    # (family scatter ADD) the first.  ``_resolve_family_couplings`` records its
    # component scatter targets into the SAME registry so a comp-coupling and a
    # family-scatter into one target collide here too.  A single reduce='sum'
    # FamilyScatterOp with co-located ``at_indices`` is ONE coupling, so it
    # never trips this guard (the co-located members accumulate internally).
    seen_targets: dict[tuple[str, str], str] = {}
    for cp in eqset.couplings:
        if id(cp.source) in family_member_ids or id(cp.target) in family_member_ids:
            continue
        if getattr(cp, "read_through", False):
            src_info = comp_var_owner_full.get(id(cp.source))
            owner = rt_alias_owner.get(id(cp.target))
            if src_info is None or owner is None:
                continue  # family source or orphan alias — handled elsewhere.
            src_comp_name, src_var_name, src_size = src_info
            tgt_comp_name, alias_var = owner
            key = (tgt_comp_name, alias_var.name)
            _guard_duplicate_target(
                seen_targets,
                (
                    tgt_comp_name,
                    f"__readthrough__:{src_comp_name}.{src_var_name}->{alias_var.name}",
                ),
                cp.name,
            )
            out[key] = (src_comp_name, src_var_name, int(src_size), cp.op)
            continue
        src_comp_name: str | None = None
        tgt_comp_name: str | None = None
        for comp in eqset.components:
            for v in comp.variables:
                if v is cp.source:
                    src_comp_name = comp.name
                if v is cp.target:
                    tgt_comp_name = comp.name
        if src_comp_name is None or tgt_comp_name is None:
            raise ValueError(
                f"coupling {cp.name!r} references variables not found by "
                f"identity in any component "
                f"(source={cp.source.name!r}, target={cp.target.name!r})."
            )
        key = (tgt_comp_name, cp.target.name)
        _guard_duplicate_target(seen_targets, key, cp.name)
        out[key] = (src_comp_name, cp.source.name, int(cp.source.size), cp.op)
    # Resolve the family couplings too so a family scatter into a component
    # target that a different coupling ALSO targets is rejected by the same
    # unified guard.  Discarded here (the residual branch calls
    # ``_resolve_family_couplings`` directly); only the guard side-effect
    # matters for ``_sparsity``'s consumption of the component endpoints.
    _resolve_family_couplings(eqset, seen_targets=seen_targets)
    return out


def _guard_duplicate_target(
    seen_targets: dict[tuple[str, str], str],
    key: tuple[str, str],
    coupling_name: str,
) -> None:
    """Reject a second DISTINCT coupling injecting into the same target slot.

    ``key`` is the ``(target_owner_name, target_var_name)`` pair.  ``seen_targets``
    maps each already-seen key to the coupling name that claimed it; a second,
    distinct coupling claiming the same key raises ``ValueError``.  One coupling
    (e.g. a reduce='sum' family scatter whose co-located members accumulate) is
    fine — it touches the key exactly once.
    """
    prior = seen_targets.get(key)
    if prior is not None and prior != coupling_name:
        raise ValueError(
            f"two distinct couplings inject into the same target "
            f"({key[0]!r}, {key[1]!r}): {prior!r} and {coupling_name!r}; the IR "
            "forbids this since the second would silently overwrite / double-"
            "count the first. Merge them into one coupling (a single "
            "reduce='sum' family scatter accumulates co-located members)."
        )
    seen_targets[key] = coupling_name


def _resolve_family_couplings(
    eqset: EquationSet,
    seen_targets: dict[tuple[str, str], str] | None = None,
) -> tuple[FamilyCoupling, ...]:
    """Resolve every family-touching coupling into a FamilyCoupling.

    A coupling is family-touching when its source OR target is a
    [Family][diff_soup.problem._ir.family.Family] member Variable (resolved by
    Python ``is`` against ``fam.member_variables``).  Two directions:

    * **scatter** — source is a family member, target is a component Variable.
      Resolves the source to ``(fam.name, var, family=True)`` and the target to
      its owning component.  The component target is registered in
      ``seen_targets`` so the unified duplicate-target guard (shared with
      ``_resolve_coupling_endpoints``) rejects a SECOND distinct coupling into
      the same component target while PERMITTING one reduce='sum' scatter.
    * **gather** — target is a family member, source is a component Variable.
      Resolves the target to ``(fam.name, var, family=True)`` and the source to
      its owning component.

    A coupling connecting two family members is out of scope and raises
    (mirrors ``_family_cross_block_sparsity``).

    Returns the resolved records in ``eqset.couplings`` order (empty when no
    family coupling exists).
    """
    if seen_targets is None:
        seen_targets = {}
    fam_owner: dict[int, tuple[str, str, int]] = {}
    member_state_by_fam: dict[str, int] = {}
    for fam in getattr(eqset, "families", ()):
        member_state_by_fam[fam.name] = int(fam.member_state_size)
        for v in fam.member_variables:
            fam_owner[id(v)] = (fam.name, v.name, int(fam.count))
    comp_owner: dict[int, tuple[str, str, int]] = {}
    for comp in eqset.components:
        for v in comp.variables:
            comp_owner[id(v)] = (comp.name, v.name, int(v.size))
    # Read-through alias targets are NOT component variables; resolve their
    # owning component as the one whose equation reads the alias (the consuming
    # closure).  ``(comp_name, alias_var)`` — distinct from ``comp_owner`` since
    # the alias has no state slot.
    alias_owner = _read_through_alias_owner(eqset)

    records: list[FamilyCoupling] = []
    for cp in eqset.couplings:
        src_is_fam = id(cp.source) in fam_owner
        tgt_is_fam = id(cp.target) in fam_owner
        if not (src_is_fam or tgt_is_fam):
            continue  # pure component<->component — handled by endpoints.
        if src_is_fam and tgt_is_fam:
            raise ValueError(
                f"coupling {cp.name!r} connects two Family members; "
                "family<->family couplings are out of scope (separate families)."
            )
        if src_is_fam and getattr(cp, "read_through", False):
            # gather_surface: READ-THROUGH family member surface trace ->
            # component closure alias.  Same slab geometry as scatter but routed
            # to MergedView injection (no DOF / no row).  The alias target is NOT
            # a component variable, so resolve its owning component from the
            # consuming equation's inputs.
            fam_name, src_var, count = fam_owner[id(cp.source)]
            source_size = count * member_state_by_fam[fam_name]
            owner = alias_owner.get(id(cp.target))
            if owner is None:
                raise ValueError(
                    f"read-through family coupling {cp.name!r} target alias "
                    f"{cp.target.name!r} is not read by any component equation; "
                    "a read-through alias must be consumed (declare it in the "
                    "reading equation's inputs)."
                )
            tgt_comp, alias_var = owner
            # Guard on the readable alias identity (NOT a DOF/row): an alias may
            # be read by many closures in the SAME component, so key on the
            # source surface trace so the same family surface cannot be aliased
            # twice into the same component under the same name, while permitting
            # distinct aliases.
            _guard_duplicate_target(
                seen_targets,
                (tgt_comp, f"__readthrough__:{fam_name}.{src_var}->{alias_var.name}"),
                cp.name,
            )
            records.append(
                FamilyCoupling(
                    name=cp.name,
                    direction="gather_surface",
                    fam_name=fam_name,
                    family=True,
                    comp_name=tgt_comp,
                    comp_var_name=alias_var.name,
                    source_var_name=src_var,
                    source_size=source_size,
                    target_comp=tgt_comp,
                    target_var_name=alias_var.name,
                    op=cp.op,
                    read_through=True,
                    alias=alias_var,
                )
            )
        elif src_is_fam:
            # scatter: family member surface trace -> component target.
            fam_name, src_var, count = fam_owner[id(cp.source)]
            source_size = count * member_state_by_fam[fam_name]
            if id(cp.target) not in comp_owner:
                raise ValueError(
                    f"family scatter coupling {cp.name!r} target "
                    f"{cp.target.name!r} not found by identity in any component."
                )
            tgt_comp, tgt_var, _tgt_size = comp_owner[id(cp.target)]
            _guard_duplicate_target(seen_targets, (tgt_comp, tgt_var), cp.name)
            records.append(
                FamilyCoupling(
                    name=cp.name,
                    direction="scatter",
                    fam_name=fam_name,
                    family=True,
                    comp_name=tgt_comp,
                    comp_var_name=tgt_var,
                    source_var_name=src_var,
                    source_size=source_size,
                    target_comp=tgt_comp,
                    target_var_name=tgt_var,
                    op=cp.op,
                )
            )
        else:
            # gather: component source -> family member target.
            fam_name, tgt_var, _count = fam_owner[id(cp.target)]
            if id(cp.source) not in comp_owner:
                raise ValueError(
                    f"family gather coupling {cp.name!r} source "
                    f"{cp.source.name!r} not found by identity in any component."
                )
            src_comp, src_var, src_size = comp_owner[id(cp.source)]
            # A GATHER makes the component SOURCE readable inside the member
            # kernel (keyed by the macro source Variable, not the family target
            # row — the smart-couple facade routes every gather's nominal target
            # to ``members[0]``, so guarding on ``(fam, members[0])`` would
            # falsely reject a SECOND distinct gather (e.g. c_e AND Phi_e AND
            # Psi_n onto one particle family).  Guard on the readable identity
            # ``(fam, source)`` instead: that still rejects gathering the SAME
            # source twice onto one family while permitting distinct sources.
            _guard_duplicate_target(
                seen_targets, (fam_name, f"__gather__:{src_comp}.{src_var}"), cp.name
            )
            records.append(
                FamilyCoupling(
                    name=cp.name,
                    direction="gather",
                    fam_name=fam_name,
                    family=True,
                    comp_name=src_comp,
                    comp_var_name=src_var,
                    source_var_name=src_var,
                    source_size=int(src_size),
                    target_comp=fam_name,
                    target_var_name=tgt_var,
                    op=cp.op,
                )
            )
    return tuple(records)


def _op_declares_empty_sparsity(op, source_size: int, target_size: int) -> bool:
    """Whether ``op.sparsity()`` declares zero ``d(target)/d(source)`` nonzeros.

    A user-defined [TransferOp][diff_soup.problem._ir.coupling.TransferOp]
    subclass (the documented extension
    point) may return empty ``sparsity()`` arrays.  When the coupling is
    *real* — i.e. some equation actually reads the target — an empty op
    sparsity would make the composed cross-block empty.  ``_sparsity`` uses
    this to fall back to the safe dense cross-block (every writer row that
    reads the target depends on every source column) instead of silently
    dropping it.  All four shipped ops declare non-empty sparsity for
    non-degenerate sizes, so they never trip the fallback.
    """
    transfer_rows, _transfer_cols = op.sparsity(source_size, target_size)
    return int(np.asarray(transfer_rows).size) == 0


def _dense_cross_block(local_rows: np.ndarray, source_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``d(writer)/d(source)`` cross-block for an empty-sparsity op.

    When a [TransferOp][diff_soup.problem._ir.coupling.TransferOp] declares
    an empty ``sparsity()`` but its
    ``apply()`` genuinely consumes the source, we cannot know which source
    columns each target row depends on.  The safe Jacobian-topology
    superset is: every *writer* row that reads the coupling target (the
    unique rows in ``local_rows``) depends on *every* source column in
    ``[0, source_size)``.  Returns ``(rows, cols)`` in component-local
    coordinates (writer rows; source columns), suitable for offsetting by
    the writer's row offset and the source's column offset respectively.
    """
    writer_rows = np.unique(np.asarray(local_rows, dtype=np.int64))
    n_writer = int(writer_rows.size)
    src = np.arange(int(source_size), dtype=np.int64)
    rows = np.repeat(writer_rows, int(source_size))
    cols = np.tile(src, n_writer)
    return rows, cols


def _compose_cross_block_sparsity(
    local_rows: np.ndarray,
    local_cols: np.ndarray,
    op,
    source_size: int,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose writer-local target dependency with transfer-op source sparsity.

    The writer's local block declares which target columns each writer row
    depends on (``local_rows × local_cols`` in component-local coordinates).
    The transfer op's
    [sparsity][diff_soup.problem._ir.coupling.TransferOp.sparsity] declares
    which source columns each
    target row depends on.  Their join is the cross-block: writer row r
    depends on source col k iff there exists a target row tc such that
    ``(r, tc)`` is in the local pattern AND ``(tc, k)`` is in the op pattern.

    Replacing the dense ``writer_rows × source_size`` fallback with
    this composition typically shrinks declared nnz by 1-2 orders of
    magnitude on real coupled systems (NodalExtractOp collapses to one
    source col per target row; SphericalSurfaceTraceOp to a single col).
    """
    transfer_rows, transfer_cols = op.sparsity(source_size, target_size)
    transfer_rows = np.asarray(transfer_rows, dtype=np.int64)
    transfer_cols = np.asarray(transfer_cols, dtype=np.int64)
    target_row_to_srcs: dict[int, list[int]] = {}
    for tr, sc in zip(transfer_rows.tolist(), transfer_cols.tolist()):
        target_row_to_srcs.setdefault(int(tr), []).append(int(sc))
    cross_rows_list: list[int] = []
    cross_cols_list: list[int] = []
    for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
        srcs = target_row_to_srcs.get(int(lc))
        if not srcs:
            continue
        for s in srcs:
            cross_rows_list.append(int(lr))
            cross_cols_list.append(int(s))
    return (
        np.asarray(cross_rows_list, dtype=np.int64),
        np.asarray(cross_cols_list, dtype=np.int64),
    )
