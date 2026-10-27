"""Public, pure-jnp accessor over a Family's member-major state slab.

The lowered problem exposes ``problem.family_view(name) -> FamilyView`` — a
frozen, internals-free accessor for post-solve analysis:

* ``.extract(u)``      → ``(count, member_state_size)`` slab gather
* ``.field_slice(f)``  → per-member ``slice`` for field ``f``
* ``.member_weights``  → ``(count,)`` volume / quadrature weights (``None`` →
  uniform)

The slab is the contiguous member-major block the layout places after all
components: member ``m``'s full member_state lives at
``[base + m*member_state : base + (m+1)*member_state]``, so ``extract`` is a
single reshape and ``field_slice(f)`` is the field's column band within one
member (``[var_local_off : var_local_off + var.size]``).

This view is built from host-side offsets (``base``, ``member_state_size``,
``count``) and a ``{field -> slice}`` map, all plain Python / numpy scalars,
plus an optional ``(count,)`` weights array.

This module lives under ``diff_soup/_lowering``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax


@dataclass(frozen=True)
class FamilyView:
    """Frozen accessor over one Family's member-major state slab.

    Attributes:
        name: The family name this view addresses.
        base: Flat offset of the family's slab in the assembled state vector.
        count: Number of members in the population.
        member_state_size: Per-member state size (sum of member-field sizes).
        field_slices: ``{field_name -> slice}`` — the per-member column band of
            each member field within one member's flat block.
        member_weights: ``(count,)`` per-member volume / quadrature weights, or
            ``None`` for a uniform population.
    """

    name: str
    base: int
    count: int
    member_state_size: int
    field_slices: dict[str, slice]
    member_weights: Optional[jax.Array] = None
    graphs: dict[str, object] = field(default_factory=dict)

    def extract(self, u: jax.Array) -> jax.Array:
        """Gather the family's member-major slab from the flat state ``u``.

        Args:
            u: Assembled state vector ``(state_size,)``.

        Returns:
            ``(count, member_state_size)`` — member ``m``'s full member-state on
            row ``m`` (a single contiguous reshape, pure-jnp / differentiable).
        """
        n = self.count * self.member_state_size
        return u[self.base : self.base + n].reshape(self.count, self.member_state_size)

    def field_slice(self, field: str) -> slice:
        """Per-member column ``slice`` for ``field`` within one member's block.

        Args:
            field: A member-field name.

        Returns:
            The ``slice`` into a member row (axis 1 of
            [extract][diff_soup._lowering.family_view.FamilyView.extract]'s output).

        Raises:
            KeyError: when ``field`` is not a member field of this family.
        """
        try:
            return self.field_slices[field]
        except KeyError as exc:
            raise KeyError(
                f"family {self.name!r} has no member field {field!r}; member "
                f"fields are {sorted(self.field_slices)!r}"
            ) from exc

    def field(self, u: jax.Array, field: str) -> jax.Array:
        """Gather one member field as a ``(count, field_size)`` slab.

        Convenience over ``extract(u)[:, field_slice(field)]`` (pure-jnp).
        """
        return self.extract(u)[:, self.field_slice(field)]

    def graph(self, name: str):
        """Return one canonical graph record by its authored name."""
        try:
            return self.graphs[name]
        except KeyError as exc:
            raise KeyError(
                f"family {self.name!r} has no graph {name!r}; graphs are {sorted(self.graphs)!r}."
            ) from exc


def _build_family_view(fam, base: int) -> FamilyView:
    """Build the [FamilyView][diff_soup._lowering.family_view.FamilyView] for one family.

    Builds the view for one [Family][diff_soup.problem._ir.family.Family] at slab
    offset ``base``.

    The per-member field slices are the member-field column bands within one
    member's flat block (the layout's member-major convention). ``member_weights``
    is ``None`` (uniform) — the Family node carries no per-member volume metadata
    (a future weighted-population path supplies it without changing this view's
    public surface).
    """
    field_slices: dict[str, slice] = {}
    loc = 0
    for var in fam.member_variables:
        size = int(var.size)
        field_slices[var.name] = slice(loc, loc + size)
        loc += size
    return FamilyView(
        name=fam.name,
        base=int(base),
        count=int(fam.count),
        member_state_size=int(fam.member_state_size),
        field_slices=field_slices,
        member_weights=None,
        graphs={graph.name: graph for graph in getattr(fam, "graphs", ())},
    )
