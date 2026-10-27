"""Member-mean couplings of a family as a low-rank term of the state Jacobian.

A family member ``m`` whose equations read the weighted mean of one of its
site fields ``f``,

    ybar_{m, f} = sum_i w_i y_{m, f, i},

has a dense block per member in its state Jacobian: every row that reads the
mean depends on every site of ``f``. That block is the member's stencil part
plus a rank-one term ``u_{m, f} w_f^T``, where ``w_f`` holds the weights on the
member's sites of ``f`` and ``u_{m, f} = dF / d ybar_{m, f}`` is supported on
the member's own rows. With several mean fields per member the term has rank
``count * len(fields)``, one coupling per member and field.

A residual that adds a per-member offset ``o_{m, f}`` (zero in every solve) to
each mean it reads exposes ``U`` through forward-mode derivatives: because no
member reads another member's offset, one derivative along the direction that
is one on every member's offset of field ``f`` gives, on member ``m``'s rows,
exactly the column ``u_{m, f}``. So ``len(fields)`` forward-mode derivatives
give every entry of ``U``.

[MemberMeanCoupling][diff_soup.problem._ir._low_rank.MemberMeanCoupling]
declares the coupling; with the lowered problem's
[FamilyView][diff_soup._lowering.family_view.FamilyView],
[member_mean_term][diff_soup.problem._ir._low_rank.member_mean_term] builds
the [LowRankTerm][diff_soup.linear.LowRankTerm],
[member_mean_values][diff_soup.problem._ir._low_rank.member_mean_values]
gathers the entries of ``U`` from the offset derivatives in the term's order,
and [member_block_layout][diff_soup.problem._ir._low_rank.member_block_layout]
describes the family's members for a
[MemberBlock][diff_soup.linear.MemberBlock] backend. The operator ``A_b + U
W^T`` is then the base topology's values joined with those entries on
``term.topology(base)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from diff_soup._errors import ConfigurationError

__all__ = [
    "MemberMeanCoupling",
    "member_block_layout",
    "member_mean_term",
    "member_mean_values",
]


@dataclass(frozen=True)
class MemberMeanCoupling:
    """Declare that every member of a family reads the means of some of its site fields.

    The residual contract: member ``m``'s equations read each mean exactly
    (differentiably) plus a per-member offset that is zero in every solve,
    and no equation outside member ``m`` reads member ``m``'s means or
    offsets. The state Jacobian is then ``A_b + U W^T`` with ``A_b`` the
    family's stencil pattern, ``W`` the structural mean weights and ``U`` the
    derivatives of the residual along the offsets.

    Attributes:
        family: Name of the family.
        fields: The site fields whose member means enter the residual (one
            coupling per member and field).
        readers: The member fields whose rows read the means (``None``: every
            member field).
        weights: Optional per-site weights of the means, one tuple per field
            in ``fields``; ``None`` gives the plain mean ``1 / n_sites``.

    Raises:
        ConfigurationError: If ``family`` is empty, ``fields`` is empty, or
            ``weights`` does not give one tuple per field.
    """

    family: str
    fields: Any
    readers: Optional[Any] = None
    weights: Optional[Any] = None

    def __post_init__(self) -> None:
        """Normalise the field tuples and check the declaration."""
        if not isinstance(self.family, str) or not self.family:
            raise ConfigurationError(
                f"MemberMeanCoupling.family must be a non-empty string; got {self.family!r}."
            )
        fields = (self.fields,) if isinstance(self.fields, str) else tuple(self.fields)
        if not fields or not all(isinstance(name, str) and name for name in fields):
            raise ConfigurationError(
                "MemberMeanCoupling.fields must name at least one member site field; got "
                f"{self.fields!r}."
            )
        object.__setattr__(self, "fields", fields)
        if self.readers is not None:
            readers = (self.readers,) if isinstance(self.readers, str) else tuple(self.readers)
            object.__setattr__(self, "readers", readers)
        if self.weights is not None:
            weights = tuple(tuple(float(w) for w in field) for field in self.weights)
            if len(weights) != len(fields):
                raise ConfigurationError(
                    f"MemberMeanCoupling.weights must give one weight tuple per field "
                    f"({len(fields)}); got {len(weights)}."
                )
            object.__setattr__(self, "weights", weights)


def _member_rows(view: Any, member: int, field: str) -> np.ndarray:
    """Global unknowns of ``field`` of member ``member`` in a family view."""
    band = view.field_slice(field)
    start = int(view.base) + member * int(view.member_state_size)
    return np.arange(start + band.start, start + band.stop, dtype=np.int64)


def member_mean_term(view: Any, coupling: MemberMeanCoupling, *, n_dofs: int) -> Any:
    """The low-rank term of a family's member-mean coupling.

    Coupling ``m * len(fields) + j`` belongs to member ``m`` and field
    ``fields[j]``: its ``W`` support is the member's sites of that field with
    the mean weights, its ``U`` support the member's rows of the reader
    fields (in the order of ``readers``, then site order).

    Args:
        view: The family's view (``problem.family_view(coupling.family)``).
        coupling: The declaration.
        n_dofs: Size of the operator (the state size, or a larger bordered
            size that keeps the state first).

    Returns:
        The [LowRankTerm][diff_soup.linear.LowRankTerm].

    Raises:
        ConfigurationError: If a field is not a member field of the family,
            or the weights do not match a field's site count.
    """
    from diff_soup.linear import LowRankTerm

    names = tuple(view.field_slices)
    readers = coupling.readers if coupling.readers is not None else names
    for name in (*coupling.fields, *readers):
        if name not in view.field_slices:
            raise ConfigurationError(
                f"family {view.name!r} has no member field {name!r}; its member fields are "
                f"{sorted(names)!r}."
            )
    reader_rows, groups, weights = [], [], []
    for member in range(int(view.count)):
        rows = np.concatenate([_member_rows(view, member, name) for name in readers])
        for j, name in enumerate(coupling.fields):
            sites = _member_rows(view, member, name)
            if coupling.weights is None:
                w = np.full(sites.shape, 1.0 / sites.size)
            else:
                w = np.asarray(coupling.weights[j], np.float64)
                if w.shape != sites.shape:
                    raise ConfigurationError(
                        f"MemberMeanCoupling.weights[{j}] has {w.size} entries but field "
                        f"{name!r} has {sites.size} sites per member."
                    )
            reader_rows.append(rows)
            groups.append(sites)
            weights.append(w)
    return LowRankTerm.group_means(n_dofs, readers=reader_rows, groups=groups, weights=weights)


def member_mean_values(view: Any, coupling: MemberMeanCoupling, derivatives: ArrayLike) -> Array:
    """The entries of ``U``, in the term's order, from the offset derivatives.

    Args:
        view: The family's view.
        coupling: The declaration.
        derivatives: ``(len(fields), n)``: row ``j`` is the derivative of the
            residual along the direction that is one on every member's
            offset of ``fields[j]`` and zero elsewhere.

    Returns:
        The ``n_u`` entries of ``U`` that follow the base values.
    """
    derivatives = jnp.asarray(derivatives)
    names = tuple(view.field_slices)
    readers = coupling.readers if coupling.readers is not None else names
    pieces = []
    for member in range(int(view.count)):
        rows = np.concatenate([_member_rows(view, member, name) for name in readers])
        for j in range(len(coupling.fields)):
            pieces.append(derivatives[j, rows])
    return jnp.concatenate(pieces)


def member_block_layout(
    view: Any,
    fields: Any,
    *,
    n_dofs: int,
    term: Any = None,
    retained: Any = None,
) -> Any:
    """The [MemberBlockLayout][diff_soup.linear.MemberBlockLayout] of a family's site fields.

    Member ``m`` has one site per entry of the fields (all of one size) and
    one unknown per field at each site; every other unknown is retained
    unless ``retained`` lists the retained set.

    Args:
        view: The family's view.
        fields: The member site fields, in their order inside a site.
        n_dofs: Size of the operator.
        term: Optional member term (for example from
            [member_mean_term][diff_soup.problem._ir._low_rank.member_mean_term]).
        retained: Optional explicit retained set.

    Returns:
        The layout.

    Raises:
        ConfigurationError: If the fields have different site counts.
    """
    from diff_soup.linear import MemberBlockLayout

    fields = (fields,) if isinstance(fields, str) else tuple(fields)
    sizes = {view.field_slice(name).stop - view.field_slice(name).start for name in fields}
    if len(sizes) != 1:
        raise ConfigurationError(
            f"member_block_layout needs site fields of one size; {fields!r} of family "
            f"{view.name!r} have sizes {sorted(sizes)!r}."
        )
    members = [
        np.stack([_member_rows(view, member, name) for name in fields], axis=1)
        for member in range(int(view.count))
    ]
    names = [f"{view.name}[{member}]" for member in range(int(view.count))]
    return MemberBlockLayout(members, n_dofs=n_dofs, retained=retained, term=term, names=names)
