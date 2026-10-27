"""State layout for the lowered problem (component + Family blocks).

- [_BlockLayout][diff_soup._lowering.layout._BlockLayout]: dataclass describing per-component variable offsets
  and per-family member-major slab offsets in the assembled state vector,
  plus the global-layout helper used by the bordered-global vector.
- [_layout][]: build a [_BlockLayout][diff_soup._lowering.layout._BlockLayout] from an [EquationSet][diff_soup._lowering.layout.EquationSet].
- [_build_global_layout_entries][]: per-global ``(name, off, size)`` for
  the bordered-vector rewrap (used by ``_initial_lambda`` and the IFT path).

A [Family][diff_soup.problem._ir.family.Family] lowers to a member-major
contiguous slab placed AFTER all [Component][diff_soup.problem._ir.component.Component] blocks, in ``EquationSet.families`` order — this pins the flat
layout so content_hash / codec / ``assemble_state`` concatenation agree.
Each member field emits ONE entry keyed ``(fam.name, field_name)`` of size
``count * field_size`` and shape ``(count, *field_shape)``; a multi-field
family records a per-entry stride so [unpack][diff_soup._lowering.layout._BlockLayout.unpack] reconstructs
the member-major interleave (``[m0_f0, m0_f1, m1_f0, m1_f1, ...]``).

This module lives under ``diff_soup/_lowering`` and may import
``diff_soup.problem._ir`` IR nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from diff_soup.problem._ir.equation_set import EquationSet


@dataclass(frozen=True)
class _BlockLayout:
    """Per-component + per-family-member offsets in the assembled state vector.

    Attributes:
        entries: ``((comp_or_fam_name, var_name), offset, size, shape)`` per
            component variable and per family member field, in layout order
            (components first, then families in ``EquationSet.families`` order).
        total_size: Assembled state size (after all components + families).
        member_strides: Per-family-entry stride metadata so ``unpack``
            reconstructs a member-major slab when a Family has MORE THAN ONE
            member field (the field arrays interleave). Keyed by
            ``(fam.name, field_name)`` → ``(member_state_size, var_local_off,
            member_field_size)``. Empty for component-only / single-field-family
            layouts, so the component path stays byte-identical (no stride
            bookkeeping).
    """

    entries: tuple[tuple[tuple[str, str], int, int, tuple[int, ...]], ...]
    total_size: int
    member_strides: dict[tuple[str, str], tuple[int, int, int]] = field(default_factory=dict)

    def offset_of(self, key: tuple[str, str]) -> int:
        """Flat offset of the block keyed ``(owner_name, var_name)``.

        Raises:
            KeyError: when ``key`` is not a registered block.
        """
        for k, off, _size, _shape in self.entries:
            if k == key:
                return off
        raise KeyError(key)

    def unpack(self, u) -> dict[tuple[str, str], jax.Array]:
        """Split the flat state ``u`` into a ``{(owner, var): array}`` view.

        Component (and single-field family) entries are contiguous slices.
        A multi-field family entry gathers the member-major stride: member
        ``m``'s field block starts at ``off + m * member_state`` (``off``
        already includes the field's ``var_local_off``).
        """
        out = {}
        for key, off, size, shape in self.entries:
            stride = self.member_strides.get(key)
            if stride is None:
                # Component (or single-field family) entry: contiguous slice.
                out[key] = u[off : off + size].reshape(shape)
            else:
                # Multi-field family entry: gather the member-major stride.
                member_state, _var_local_off, member_field_size = stride
                count = shape[0]
                base = off + jnp.arange(count) * member_state  # (count,)
                idx = base[:, None] + jnp.arange(member_field_size)[None, :]
                out[key] = u[idx.reshape(-1)].reshape(shape)
        return out


def _layout(eqset: EquationSet) -> _BlockLayout:
    """Build the assembled-state [_BlockLayout][diff_soup._lowering.layout._BlockLayout] for ``eqset``.

    Components first (variable order), then families in ``EquationSet.families``
    order. Each family's whole population slab is contiguous and MEMBER-MAJOR
    (member 0's full profile, then member 1's, ...), so a member var's global
    offset is ``fam_base + member * member_state_size + var_local_off``.
    """
    entries = []
    off = 0
    for comp in eqset.components:
        for var in comp.variables:
            key = (comp.name, var.name)
            entries.append((key, off, var.size, var.shape))
            off += var.size
    member_strides: dict[tuple[str, str], tuple[int, int, int]] = {}
    for fam in getattr(eqset, "families", ()):
        fam_base = off
        member_state_size = fam.member_state_size
        multi_field = len(fam.member_variables) > 1
        var_local_off = 0
        for var in fam.member_variables:
            key = (fam.name, var.name)
            size = fam.count * var.size
            shape = (fam.count, *var.shape)
            entries.append((key, fam_base + var_local_off, size, shape))
            if multi_field:
                # Member-major interleave needs a strided gather in unpack.
                member_strides[key] = (member_state_size, var_local_off, var.size)
            var_local_off += var.size
        off += fam.count * member_state_size
    return _BlockLayout(entries=tuple(entries), total_size=off, member_strides=member_strides)


def _build_global_layout_entries(eqset: EquationSet) -> tuple:
    """Per-global ``(name, offset, size)`` for bordered-vector rewrap.

    Walks ``eqset.constraints`` in the same order ``_initial_lambda`` does so
    the offsets line up with the concatenated bordered-global vector. Each
    differential equation or algebraic constraint closes one
    [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable], so the
    slice is tagged by that global's name.
    """
    entries = []
    offset = 0
    for c in eqset.constraints:
        size = int(c.dim)
        entries.append((c.closes.name, offset, size))
        offset += size
    return tuple(entries)
