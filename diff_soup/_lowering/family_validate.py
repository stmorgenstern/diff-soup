"""Lowering-time family coupling validation.

Runs static validation on [EquationSet][diff_soup.problem._ir.equation_set.EquationSet] family couplings before residual assembly.  Fires for any
``EquationSet`` — including those built without the Recipe API facade (e.g.
a raw-IR ``EquationSet``) — so invalid family couplings are
rejected with clear diagnostics.

Three checks:

* A family targeting ANOTHER family (source AND target both family members)
  raises [NotImplementedError][builtins.NotImplementedError] (family<->family couplings are out of
  scope; use separate families or an explicit component bridge).
* A ``FamilyScatterOp.at_indices`` value out of range for the target component
  variable's ``num_nodes`` raises [ValueError][builtins.ValueError].
* A ``FamilyScatterOp`` with ``injective=True`` whose ``at_indices`` contain a
  repeated target node raises [ValueError][builtins.ValueError] (each target node must receive
  at most one contribution when ``reduce='none'``).
"""

from __future__ import annotations

import numpy as np

from diff_soup.problem._ir.coupling import FamilyScatterOp
from diff_soup.problem._ir.equation_set import EquationSet


def _validate_families(eqset: EquationSet) -> None:
    """Run the lowering-time family error taxonomy (raise on illegal coupling).

    Raises:
        NotImplementedError: family→family (many:many) coupling (out of scope;
            use separate families or an explicit component bridge).
        ValueError: a family-scatter ``at=`` index out of range for the target,
            or an ``injective=True`` scatter whose targets collide.
    """
    families = getattr(eqset, "families", ())
    if not families:
        return
    fam_member_ids = {id(v) for fam in families for v in fam.member_variables}
    comp_var_size_by_id: dict[int, tuple[str, int]] = {}
    for comp in eqset.components:
        for v in comp.variables:
            comp_var_size_by_id[id(v)] = (v.name, int(v.size))

    for cp in eqset.couplings:
        src_is_fam = id(cp.source) in fam_member_ids
        tgt_is_fam = id(cp.target) in fam_member_ids
        if not (src_is_fam or tgt_is_fam):
            continue
        # Row 3: family -> family (many:many) is out of scope.
        if src_is_fam and tgt_is_fam:
            raise NotImplementedError(
                f"coupling {cp.name!r} targets a family from another family "
                f"({cp.source.name!r} -> {cp.target.name!r}); family<->family "
                "(many:many) couplings are out of scope — use separate "
                "families or an explicit component bridge."
            )
        if not isinstance(cp.op, FamilyScatterOp):
            continue
        # Remaining rows concern a family-SCATTER into a component target.
        if not src_is_fam:
            continue
        target_info = comp_var_size_by_id.get(id(cp.target))
        if target_info is None:
            continue  # endpoint resolution surfaces a clearer error downstream.
        target_name, num_nodes = target_info
        at_indices = np.asarray(cp.op.at_indices, dtype=np.int64)
        # Row 2: at= out of range for the target num_nodes.
        if at_indices.size and ((at_indices < 0).any() or (at_indices >= num_nodes).any()):
            raise ValueError(
                f"spec.couple: at= index out of range for target "
                f"{target_name!r} (min={int(at_indices.min())}, "
                f"max={int(at_indices.max())}, num_nodes={num_nodes})"
            )
        # Injective scatter — reject a target-node collision with the facade-verb
        # string; keyed on op.injective (the explicit bool on FamilyScatterOp).
        if cp.op.injective:
            counts: dict[int, int] = {}
            for node in at_indices.tolist():
                counts[int(node)] = counts.get(int(node), 0) + 1
            for node, k in counts.items():
                if k > 1:
                    raise ValueError(
                        f"spec.couple(reduce='none'): target node {node} "
                        f"receives {k} contributions; use reduce='sum' or 'mean'"
                    )
