"""Elimination of members with their own scalars onto a sparse shared system.

A *member* (one particle of a population) is ``n_i`` sites of ``b`` unknowns
that couple at most one site apart, plus ``k_i`` scalars of its own whose
rows and columns may be full over the member. Every other unknown is
*shared*. A member scalar that shares an entry with another member (a link)
is moved to the shared set when the pattern is classified. In member order

    A = [[blockdiag_m(A_m), B], [C, A_SS]],     A_m = [[T_m, E_m], [F_m, G_m]],

with ``T_m`` block tridiagonal. The elimination of one lane is

1. ``T_m = Lt Ut`` for all members at once ([chain_factor][]);
2. ``Z_E = T^-1 E`` and the pivoted LU of each member's ``S_m = G - F Z_E``;
3. ``W_m = A_m^-1 B_m`` over the shared columns member ``m`` reads;
4. the values of the shared Schur complement ``S = A_SS - sum_m C_m W_m`` on
   a sparse pattern fixed at analysis, for a backend of the caller's choice.

The four lane functions [eliminate][], [reduce_rhs][], [back_substitute][]
and the host [member_schur_plan][] are what the ``MemberSchur`` backend
composes with the shared backend.

Nothing is padded to the longest member. The sites of all members are stored
back to back (``S = sum(n_m)`` sites, a [ChainLayout][]), followed by ``k``
scalar places per member: the *local* unknowns, ``L = S b + M k`` of them.
The factor holds ``Z_E`` ``(S, b, k)``, ``W`` ``(L, kc)`` and the chain factor;
``E``, ``F`` and ``C`` are read from the values where they are used. Values
reach their places by gathers through index maps built once per pattern, and
sums over a member's sites are reductions over equal lengths or segment sums
over different ones. Every step is a fixed-shape ``jax.numpy`` expression on
plain transposes, so real and complex values, ``jit`` and ``vmap`` all work.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.linear._api import SparseTopology
from diff_soup.linear._factor import SolveStatus, dirichlet_rows
from diff_soup.linear._structured._common import index_constant
from diff_soup.linear._structured._member_chain import (
    ELIMINATIONS,
    ChainLayout,
    chain_factor,
    chain_solve,
)
from diff_soup.linear._structured._small_lu import (
    bmm,
    small_lu_factor,
    small_lu_solve,
    small_lu_solve_transpose,
)

__all__ = [
    "MemberFactor",
    "MemberSchurLayout",
    "back_substitute",
    "eliminate",
    "member_schur_plan",
    "reduce_rhs",
]


def _int_bytes(arrays: Any) -> bytes:
    """Concatenated index arrays as contiguous int32 host bytes."""
    if not arrays:
        return b""
    flat = np.concatenate([np.asarray(a, dtype=np.int64).reshape(-1) for a in arrays])
    return np.ascontiguousarray(flat.astype(np.int32)).tobytes()


class MemberSchurLayout(eqx.Module):
    """The members, their own scalars and the shared unknowns of a `MemberSchur` solver.

    Every unknown is a site of one member, a scalar of one member, or shared.

    Args:
        members: One `(n_sites, block_size)` index array per member, site by site, with one
            `block_size` for all. Members may differ in `n_sites`.
        n_dofs: The number of unknowns.
        scalars: One index array per member with that member's own scalar unknowns, or `None`
            for none. Members may differ in how many they have.
        names: One name per member, used in error messages.
    """

    n_dofs: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    lengths: tuple = eqx.field(static=True)
    scalar_counts: tuple = eqx.field(static=True)
    names: tuple = eqx.field(static=True)
    _sites: bytes = eqx.field(static=True)
    _scalars: bytes = eqx.field(static=True)

    def __init__(self, members: Any, *, n_dofs: int, scalars: Any = None, names: Any = None):
        """Validate the members and their scalars, and freeze them."""
        arrays = [np.asarray(member, dtype=np.int64) for member in members]
        if not arrays:
            raise ConfigurationError("MemberSchurLayout needs at least one member.")
        for k, array in enumerate(arrays):
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise ConfigurationError(
                    f"member {k} has an index array of shape {array.shape}; each member needs an "
                    "(n_sites, b) array with n_sites >= 1 and b >= 1."
                )
            if array.shape[1] != arrays[0].shape[1]:
                raise ConfigurationError(
                    f"member {k} has {array.shape[1]} unknowns per site but member 0 has "
                    f"{arrays[0].shape[1]}; every member of one layout has the same number."
                )
        if scalars is None:
            own = [np.zeros((0,), np.int64) for _ in arrays]
        else:
            own = [np.asarray(s, dtype=np.int64).reshape(-1) for s in scalars]
            if len(own) != len(arrays):
                raise ConfigurationError("scalars must give one index array per member.")
        n = int(n_dofs)
        flat = np.concatenate([a.reshape(-1) for a in arrays] + own)
        if flat.min() < 0 or flat.max() >= n:
            raise ConfigurationError(f"member indices must lie in [0, {n}) for n_dofs={n}.")
        if np.unique(flat).size != flat.size:
            raise ConfigurationError(
                "members must be disjoint: an unknown is a site or a scalar of two members, or "
                "appears twice in one."
            )
        self.n_dofs = n
        self.block_size = int(arrays[0].shape[1])
        self.lengths = tuple(int(a.shape[0]) for a in arrays)
        self.scalar_counts = tuple(int(s.size) for s in own)
        self.names = tuple(str(name) for name in names) if names is not None else ()
        if self.names and len(self.names) != len(arrays):
            raise ConfigurationError("names must give one name per member.")
        self._sites = _int_bytes(arrays)
        self._scalars = _int_bytes(own)

    @property
    def n_members(self) -> int:
        """Number of members."""
        return len(self.lengths)

    @property
    def sites(self) -> np.ndarray:
        """``(total sites, b)`` unknowns of every member site, member by member (host int32)."""
        return np.frombuffer(self._sites, dtype=np.int32).reshape(-1, self.block_size)

    @property
    def scalars(self) -> np.ndarray:
        """Every member's own scalars, member by member (host int32)."""
        return np.frombuffer(self._scalars, dtype=np.int32)

    def member_label(self, member: int) -> str:
        """The member's name, or ``"member <k>"``."""
        return self.names[member] if self.names else f"member {member}"

    def __hash__(self) -> int:
        """Hash by content."""
        return hash((self.n_dofs, self.block_size, self.lengths, self.scalar_counts, self._sites, self._scalars))

    def __eq__(self, other: object) -> bool:
        """Equal when the members and their scalars are equal."""
        if not isinstance(other, MemberSchurLayout):
            return NotImplemented
        return (
            self.n_dofs == other.n_dofs
            and self.block_size == other.block_size
            and self.lengths == other.lengths
            and self.scalar_counts == other.scalar_counts
            and self._sites == other._sites
            and self._scalars == other._scalars
        )


class _Plan:
    """Host maps from a value vector to the member elimination (one per analysis)."""


def _gather_map(size: int, dst: np.ndarray, src: np.ndarray, units: Any = ()) -> np.ndarray:
    """For each of ``size`` places, the entry that lands there: ``-1`` for none, ``-2`` for a one.

    ``(size,)`` when no place receives two entries, else ``(size, d)`` with the entries of a
    place along the last axis (a pattern that lists an entry twice sums its values).
    """
    counts = np.bincount(dst, minlength=size) if dst.size else np.zeros((size,), np.int64)
    depth = int(max(1, counts.max() if size else 1))
    out = np.full((size, depth), -1, np.int64)
    order = np.argsort(dst, kind="stable")
    ordered = dst[order]
    rank = np.arange(ordered.size) - (np.cumsum(counts) - counts)[ordered]
    out[ordered, rank] = src[order]
    units = np.asarray(units, dtype=np.int64)
    out[units, 0] = -2
    return index_constant(out if depth > 1 else out[:, 0])


def member_schur_plan(
    layout: MemberSchurLayout, topology: SparseTopology, elimination: str = "thomas"
) -> _Plan:
    """Classify every pattern entry against the layout and build the shared pattern.

    Args:
        layout: The members and their scalars.
        topology: The pattern of the operator; its Dirichlet rows are honoured.
        elimination: The chain kernel, one of ``"thomas"``, ``"cyclic"``, ``"native"``.

    A topology larger than the layout (a derivative system that appends unknowns to the
    state) keeps the layout's members, and its trailing unknowns are shared. A topology
    smaller than the layout is another system in another numbering (the algebraic rows of a
    starting-state solve): no member is eliminated and the whole operator is the shared system.

    Returns:
        The plan: ``n``, ``n_shared``, ``shared`` (the shared unknowns, ascending),
        ``shared_topology`` (the pattern the shared values follow) and the index maps.

    Raises:
        ConfigurationError: Naming the mismatch: an entry that joins sites of two members,
            or sites of one member more than one apart.
    """
    if elimination not in ELIMINATIONS:
        raise ConfigurationError(f"elimination must be one of {ELIMINATIONS}; got {elimination!r}.")
    b, nm = layout.block_size, layout.n_members
    n = int(topology.n_dofs)
    if n < layout.n_dofs:
        return _whole_plan(topology, elimination)
    rows = np.asarray(topology.rows, dtype=np.int64)
    cols = np.asarray(topology.cols, dtype=np.int64)
    lengths = np.asarray(layout.lengths, dtype=np.int64)
    counts = np.asarray(layout.scalar_counts, dtype=np.int64)
    chain = ChainLayout(lengths)
    n_sites = chain.n_sites
    first = np.cumsum(lengths) - lengths

    # Owner, site and field of every unknown (-1 where it does not apply).
    owner = np.full((n,), -1, np.int64)
    site = np.full((n,), -1, np.int64)
    field = np.full((n,), -1, np.int64)
    sites = layout.sites.astype(np.int64)
    member_of_site = np.repeat(np.arange(nm), lengths)
    site_in_member = np.arange(n_sites) - np.repeat(first, lengths)
    owner[sites] = member_of_site[:, None]
    site[sites] = site_in_member[:, None]
    field[sites] = np.arange(b)[None, :]
    scalars = layout.scalars.astype(np.int64)
    scalar_owner = np.repeat(np.arange(nm), counts)
    owner[scalars] = scalar_owner
    is_scalar = np.zeros((n,), bool)
    is_scalar[scalars] = True

    # Links: an entry between two members moves the scalar(s) it touches to the shared set.
    ro, co = owner[rows], owner[cols]
    linked = (ro >= 0) & (co >= 0) & (ro != co)
    if np.any(linked):
        both_sites = linked & ~is_scalar[rows] & ~is_scalar[cols]
        if np.any(both_sites):
            e = int(np.flatnonzero(both_sites)[0])
            raise ConfigurationError(
                f"entry ({rows[e]}, {cols[e]}) joins a site of {layout.member_label(int(ro[e]))} "
                f"to a site of {layout.member_label(int(co[e]))}; MemberSchur needs members that "
                "couple through shared unknowns or through their own scalars."
            )
        promoted = np.unique(
            np.concatenate([rows[linked & is_scalar[rows]], cols[linked & is_scalar[cols]]])
        )
        is_scalar[promoted] = False
        owner[promoted] = -1

    # Local slot of every remaining member scalar, and the shared position of the rest.
    local_scalars = scalars[is_scalar[scalars]]
    local_owner = owner[local_scalars]
    kept = np.bincount(local_owner, minlength=nm) if local_scalars.size else np.zeros(nm, np.int64)
    k_max = int(kept.max()) if nm else 0
    starts = np.cumsum(kept) - kept
    slot = np.full((n,), -1, np.int64)
    slot[local_scalars] = np.arange(local_scalars.size) - starts[local_owner]
    shared = np.flatnonzero(owner < 0)
    ns = int(shared.size)
    pos = np.full((n,), -1, np.int64)
    pos[shared] = np.arange(ns)

    # The local place of every member unknown: its site unknowns back to back, then k scalar
    # places per member.
    nb = n_sites * b
    n_local = nb + nm * k_max
    members_of = owner >= 0
    at_site = site >= 0
    local = np.full((n,), -1, np.int64)
    local[at_site] = (first[owner[at_site]] + site[at_site]) * b + field[at_site]
    at_scalar = members_of & ~at_site
    local[at_scalar] = nb + owner[at_scalar] * k_max + slot[at_scalar]
    loc_idx = np.full((n_local,), -1, np.int64)
    loc_idx[local[members_of]] = np.flatnonzero(members_of)

    plan = _Plan()
    plan.elimination = elimination
    plan.whole = False
    plan.chain = chain
    plan.n, plan.b, plan.nm, plan.k = n, b, nm, k_max
    plan.n_sites, plan.n_local, plan.n_shared = n_sites, n_local, ns
    plan.nnz = int(rows.size)
    plan.shared = index_constant(shared)
    plan.loc_idx = index_constant(loc_idx)
    # Where each unknown's solution sits in [local places, shared positions].
    place = np.empty((n,), np.int64)
    place[members_of] = local[members_of]
    place[shared] = n_local + np.arange(ns)
    plan.place = index_constant(place)

    # Every entry of a Dirichlet row is dropped and its diagonal place holds a one.
    mask, diagonal = dirichlet_rows(topology)
    live = np.ones((int(rows.size),), bool) if mask is None else ~np.asarray(mask, bool)
    diagonal = diagonal.astype(np.int64)

    rl, cl = owner[rows] >= 0, owner[cols] >= 0
    rs_, cs_ = site[rows] >= 0, site[cols] >= 0

    # Site-site entries of one member: the chain blocks, flat over (3, S, b, b).
    k = np.flatnonzero(rs_ & cs_)
    offset = site[cols[k]] - site[rows[k]]
    far = np.abs(offset) > 1
    if np.any(far):
        e = int(k[far][0])
        raise ConfigurationError(
            f"entry ({rows[e]}, {cols[e]}) joins sites {int(site[rows[e]])} and "
            f"{int(site[cols[e]])} of {layout.member_label(int(owner[rows[e]]))}; MemberSchur needs "
            "a member stencil of radius 1 in the site index (declare a quantity every site reads "
            "as a member scalar)."
        )
    keep = live[k]
    d_site = diagonal[site[diagonal] >= 0]
    plan.tt_map = _gather_map(
        3 * nb * b,
        (((offset + 1) * nb + local[rows[k]]) * b + field[cols[k]])[keep],
        k[keep],
        (nb + local[d_site]) * b + field[d_site],
    )

    # Member scalars: E and F stored site-major (S b, k), and G (M, k, k).
    if k_max:
        k = np.flatnonzero(rs_ & cl & ~cs_ & live)
        plan.e_map = _gather_map(nb * k_max, local[rows[k]] * k_max + slot[cols[k]], k)
        k = np.flatnonzero(rl & ~rs_ & cs_ & live)
        plan.f_map = _gather_map(nb * k_max, local[cols[k]] * k_max + slot[rows[k]], k)
        k = np.flatnonzero(rl & ~rs_ & cl & ~cs_ & live)
        pad_m, pad_j = np.nonzero(np.arange(k_max)[None, :] >= kept[:, None])
        d_scal = diagonal[(owner[diagonal] >= 0) & (site[diagonal] < 0)]
        plan.g_map = _gather_map(
            nm * k_max * k_max,
            (owner[rows[k]] * k_max + slot[rows[k]]) * k_max + slot[cols[k]],
            k,
            np.concatenate(
                [
                    (pad_m * k_max + pad_j) * k_max + pad_j,
                    (owner[d_scal] * k_max + slot[d_scal]) * k_max + slot[d_scal],
                ]
            ),
        )

    # Member rows reading shared columns, compressed per member: B over (L, kc).
    k = np.flatnonzero(rl & ~cl)
    owners, shared_cols = owner[rows[k]], pos[cols[k]]
    pairs = (
        np.unique(np.stack([owners, shared_cols], axis=1), axis=0)
        if k.size
        else np.zeros((0, 2), np.int64)
    )
    read = np.bincount(pairs[:, 0], minlength=nm) if pairs.size else np.zeros(nm, np.int64)
    kc = int(max(1, read.max() if read.size else 1))
    colmap = np.full((nm, kc), ns, dtype=np.int64)
    start = np.cumsum(read) - read
    pair_local = np.arange(pairs.shape[0]) - start[pairs[:, 0]] if pairs.size else pairs[:, 0]
    if pairs.size:
        colmap[pairs[:, 0], pair_local] = pairs[:, 1]
        key = pairs[:, 0] * max(ns, 1) + pairs[:, 1]
        which = pair_local[np.searchsorted(key, owners * max(ns, 1) + shared_cols)]
    else:
        which = np.zeros(0, np.int64)
    plan.kc = kc
    plan.colmap = index_constant(colmap)
    # The shared columns each local place reads: its member's, (L, kc).
    member_of_local = np.concatenate([np.repeat(member_of_site, b), np.repeat(np.arange(nm), k_max)])
    plan.local_cols = index_constant(colmap[member_of_local])
    keep = live[k]
    plan.b_map = _gather_map(n_local * kc, (local[rows[k]] * kc + which)[keep], k[keep])

    # Shared rows reading member columns: C as entries.
    k = np.flatnonzero(~rl & cl)
    rm_row, rm_member = pos[rows[k]], owner[cols[k]]

    # The shared pattern: A_SS, the diagonal, and the fill of every member's elimination.
    kss = np.flatnonzero(~rl & ~cl)
    fill_rows = np.repeat(rm_row, kc)
    fill_cols = colmap[rm_member].reshape(-1)
    real = fill_cols < ns
    keys = np.concatenate(
        [
            pos[rows[kss]] * max(ns, 1) + pos[cols[kss]],
            np.arange(ns) * (ns + 1),
            fill_rows[real] * max(ns, 1) + fill_cols[real],
        ]
    )
    pattern = np.unique(keys)
    nnz_s = int(pattern.size)
    plan.nnz_shared = nnz_s
    plan.shared_topology = SparseTopology(
        pattern // max(ns, 1), pattern % max(ns, 1), n_dofs=ns
    )
    keep = live[kss]
    d_shared = diagonal[owner[diagonal] < 0]
    plan.ss_map = _gather_map(
        nnz_s,
        np.searchsorted(pattern, pos[rows[kss]] * max(ns, 1) + pos[cols[kss]])[keep],
        kss[keep],
        np.searchsorted(pattern, pos[d_shared] * (ns + 1)),
    )
    fill_slot = np.full((rm_row.size * kc,), nnz_s, np.int64)
    fill_slot[real] = np.searchsorted(pattern, fill_rows[real] * max(ns, 1) + fill_cols[real])
    keep = live[k]
    plan.c_src = index_constant(k[keep])
    plan.c_row = index_constant(rm_row[keep])
    plan.c_col = index_constant(local[cols[k]][keep])
    plan.c_fill = index_constant(fill_slot.reshape(-1, kc)[keep])
    return plan


def _whole_plan(topology: SparseTopology, elimination: str) -> _Plan:
    """The plan of an operator none of whose unknowns is eliminated: all of it is shared."""
    n = int(topology.n_dofs)
    rows = np.asarray(topology.rows, dtype=np.int64)
    cols = np.asarray(topology.cols, dtype=np.int64)
    plan = _Plan()
    plan.elimination = elimination
    plan.whole = True
    plan.n = plan.n_shared = n
    plan.nm = 0
    plan.nnz = int(rows.size)
    plan.shared = index_constant(np.arange(n))
    mask, diagonal = dirichlet_rows(topology)
    live = np.ones((plan.nnz,), bool) if mask is None else ~np.asarray(mask, bool)
    pattern = np.unique(np.concatenate([rows * max(n, 1) + cols, np.arange(n) * (n + 1)]))
    plan.nnz_shared = int(pattern.size)
    plan.shared_topology = SparseTopology(pattern // max(n, 1), pattern % max(n, 1), n_dofs=n)
    plan.ss_map = _gather_map(
        plan.nnz_shared,
        np.searchsorted(pattern, rows * max(n, 1) + cols)[live],
        np.flatnonzero(live),
        np.searchsorted(pattern, diagonal.astype(np.int64) * (n + 1)),
    )
    return plan


class MemberFactor(NamedTuple):
    """The elimination of one lane (absent parts are ``None``)."""

    chain: Any
    ze: Optional[Array]
    s_lu: Optional[Array]
    s_perm: Optional[Array]
    w: Array


def _take(values: Array, index: np.ndarray) -> Array:
    """The values that land on each place of a [_gather_map][]: a gather, never a scatter.

    A scatter of complex values ran one element at a time on a graphics card, and a scatter
    with zero fill wrote every padded place on the processor.
    """
    picked = values.at[np.maximum(index, 0)].get(mode="promise_in_bounds")
    if np.any(index < 0):
        zero = jnp.zeros((), values.dtype)
        other = jnp.where(index == -2, jnp.ones((), values.dtype), zero) if np.any(index == -2) else zero
        picked = jnp.where(index >= 0, picked, other)
    return jnp.sum(picked, axis=-1) if index.ndim == 2 else picked


def _segment_sum(rows: Array, index: np.ndarray, count: int, ordered: bool = False) -> Array:
    """``(count, ...)`` sums of ``rows`` ``(r, ...)`` over equal ``index``.

    A scatter of complex values ran serially on a graphics card: the two parts are summed apart.
    """

    def total(part):
        return jax.ops.segment_sum(part, index, num_segments=count, indices_are_sorted=ordered)

    if jnp.iscomplexobj(rows):
        return jax.lax.complex(total(rows.real), total(rows.imag))
    return total(rows)


def _sub_rows(target: Array, index: np.ndarray, updates: Array) -> Array:
    """``target`` ``(r, m)`` with the rows ``updates`` ``(..., m)`` subtracted at rows ``index``.

    The updates are summed as they are and the sum is subtracted behind a barrier. A
    scatter of negated updates onto one repeated row (a shared row read by every member)
    ran serially on a graphics card: 1.3 s for 200,000 rows against 1.7 ms this way.
    """
    rows = updates.reshape(-1, target.shape[1])
    summed = _segment_sum(rows, index.reshape(-1), target.shape[0])
    return target - jax.lax.optimization_barrier(summed)


def _member_sum(plan: _Plan, x: Array) -> Array:
    """``(M, ...)`` sums of site rows ``x`` ``(S, ...)`` over each member's sites."""
    chain = plan.chain
    if chain.uniform:
        return jnp.sum(x.reshape((chain.n_members, chain.n_max) + x.shape[1:]), axis=1)
    return _segment_sum(x, chain.member_of_site, chain.n_members, ordered=True)


def _per_site(plan: _Plan, x: Array) -> Array:
    """Member rows ``x`` ``(M, ...)`` repeated at every site of the member, ``(S, ...)``."""
    chain = plan.chain
    if chain.uniform:
        shape = (chain.n_members, chain.n_max) + x.shape[1:]
        return jnp.broadcast_to(x[:, None], shape).reshape((chain.n_sites,) + x.shape[1:])
    return x[chain.member_of_site]


def _mm(a: Array, x: Array) -> Array:
    """Batched ``a @ x``, written as a product and a sum.

    A batched matrix product of complex values ran one member at a time on a graphics card
    (0.7 ms per member in a solve); this form is one fused kernel for real and complex values.
    """
    return bmm(a, x)


def _tmm(a: Array, x: Array) -> Array:
    """Batched ``a^T @ x`` with the plain transpose."""
    return bmm(jnp.swapaxes(a, -1, -2), x)


def _chain(plan: _Plan, tree: MemberFactor, v: Array, transpose: bool) -> Array:
    """``T^-1 v`` (or ``T^-T v``) for site columns ``v`` ``(S, b, m)``."""
    return chain_solve(tree.chain, v, plan.chain, transpose, plan.elimination)


def _local_solve(plan: _Plan, values: Array, tree: MemberFactor, v: Array, transpose: bool):
    """``A_m^-1 v`` (or ``A_m^-T v``) for local columns ``v`` ``(L, m)``."""
    ns_, b, k, nm = plan.n_sites, plan.b, plan.k, plan.nm
    nb, m = ns_ * b, v.shape[-1]
    v_sites = v[:nb].reshape(ns_, b, m)
    if not k:
        return _chain(plan, tree, v_sites, transpose).reshape(nb, m)
    v_scal = v[nb:].reshape(nm, k, m)
    f = _take(values, plan.f_map).reshape(ns_, b, k)
    if not transpose:
        g = _chain(plan, tree, v_sites, False)
        x_scal = small_lu_solve(tree.s_lu, tree.s_perm, v_scal - _member_sum(plan, _tmm(f, g)))
        x_sites = g - _mm(tree.ze, _per_site(plan, x_scal))
    else:
        x_scal = small_lu_solve_transpose(
            tree.s_lu, tree.s_perm, v_scal - _member_sum(plan, _tmm(tree.ze, v_sites))
        )
        x_sites = _chain(plan, tree, v_sites - _mm(f, _per_site(plan, x_scal)), True)
    return jnp.concatenate([x_sites.reshape(nb, m), x_scal.reshape(nm * k, m)], axis=0)


def eliminate(plan: _Plan, values: Array) -> tuple:
    """Eliminate every member of one lane.

    Args:
        plan: The [member_schur_plan][] result.
        values: ``(nnz,)`` entry values in the topology's order.

    Returns:
        ``(tree, shared_values, status)``: the [MemberFactor][], the values of the shared
        Schur complement on ``plan.shared_topology`` and an ``int32`` status
        (``SINGULAR`` when a member has a negligible or non-finite pivot).
    """
    dtype = values.dtype
    shared = _take(values, plan.ss_map)
    if plan.whole:
        return (), shared, jnp.asarray(int(SolveStatus.SUCCESS), jnp.int32)
    nm, ns_, b, k, kc = plan.nm, plan.n_sites, plan.b, plan.k, plan.kc
    nb = ns_ * b
    lower, diag, upper = _take(values, plan.tt_map).reshape(3, ns_, b, b)
    chain, bad = chain_factor(lower, diag, upper, plan.chain, plan.elimination)
    bad = jnp.any(bad)

    b_all = _take(values, plan.b_map).reshape(plan.n_local, kc)
    b_sites = b_all[:nb].reshape(ns_, b, kc)
    tree = MemberFactor(chain=chain, ze=None, s_lu=None, s_perm=None, w=b_all)
    if k:
        e = _take(values, plan.e_map).reshape(ns_, b, k)
        f = _take(values, plan.f_map).reshape(ns_, b, k)
        z = _chain(plan, tree, jnp.concatenate([e, b_sites], axis=-1), False)
        ze, zb = z[..., :k], z[..., k:]
        reduced = _member_sum(plan, _tmm(f, z))
        g = _take(values, plan.g_map).reshape(nm, k, k)
        s_lu, s_perm, singular = small_lu_factor(g - reduced[..., :k])
        bad = bad | jnp.any(singular)
        w_scal = small_lu_solve(s_lu, s_perm, b_all[nb:].reshape(nm, k, kc) - reduced[..., k:])
        w_sites = zb - _mm(ze, _per_site(plan, w_scal))
        w = jnp.concatenate([w_sites.reshape(nb, kc), w_scal.reshape(nm * k, kc)], axis=0)
        tree = MemberFactor(chain=chain, ze=ze, s_lu=s_lu, s_perm=s_perm, w=w)
    else:
        w = _chain(plan, tree, b_sites, False).reshape(nb, kc)
        tree = tree._replace(w=w)

    if plan.c_src.size:
        update = values[plan.c_src][:, None] * w[plan.c_col]
        fill = _segment_sum(update.reshape(-1), plan.c_fill.reshape(-1), plan.nnz_shared + 1)
        shared = shared - jax.lax.optimization_barrier(fill)[: plan.nnz_shared]
    finite = jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(shared))
    status = jnp.where(bad | ~finite, int(SolveStatus.SINGULAR), int(SolveStatus.SUCCESS))
    return tree, shared, status.astype(jnp.int32)


def reduce_rhs(plan: _Plan, values: Array, tree: MemberFactor, columns: Array, transpose: bool):
    """The right-hand side of the shared system for columns ``(n, m)``.

    Returns:
        ``(carry, rhs_shared)``: what [back_substitute][] needs, and ``(n_shared, m)``.
    """
    if plan.whole:
        return (), columns
    ns = plan.n_shared
    m = columns.shape[1]
    c_local = columns[np.maximum(plan.loc_idx, 0)]
    if np.any(plan.loc_idx < 0):
        # A member with fewer scalars than the most leaves scalar places empty.
        c_local = jnp.where((plan.loc_idx >= 0)[:, None], c_local, jnp.zeros((), columns.dtype))
    c_shared = columns[plan.shared]
    if not transpose:
        y = _local_solve(plan, values, tree, c_local, False)
        if plan.c_src.size:
            taken = values[plan.c_src][:, None] * y[plan.c_col]
            c_shared = _sub_rows(c_shared, plan.c_row, taken)
        return y, c_shared
    # W^T c per member, then onto the shared columns each member reads.
    moved = tree.w[:, :, None] * c_local[:, None, :]
    nb = plan.n_sites * plan.b
    total = _member_sum(plan, jnp.sum(moved[:nb].reshape(plan.n_sites, plan.b, plan.kc, m), axis=1))
    if plan.k:
        total = total + jnp.sum(moved[nb:].reshape(plan.nm, plan.k, plan.kc, m), axis=1)
    rhs = jnp.concatenate([c_shared, jnp.zeros((1, m), columns.dtype)], axis=0)
    rhs = _sub_rows(rhs, plan.colmap, total)
    return c_local, rhs[:ns]


def back_substitute(
    plan: _Plan, values: Array, tree: MemberFactor, carry: Array, x_shared: Array, transpose: bool
) -> Array:
    """The solution columns ``(n, m)`` from the shared solution ``(n_shared, m)``."""
    if plan.whole:
        return x_shared
    m = x_shared.shape[1]
    if not transpose:
        padded = jnp.concatenate([x_shared, jnp.zeros((1, m), x_shared.dtype)], axis=0)
        x_local = carry - jnp.sum(tree.w[:, :, None] * padded[plan.local_cols], axis=1)
    else:
        v = carry
        if plan.c_src.size:
            v = _sub_rows(v, plan.c_col, values[plan.c_src][:, None] * x_shared[plan.c_row])
        x_local = _local_solve(plan, values, tree, v, True)
    # A gather from the stacked local and shared columns (a scatter of complex values ran
    # serially on a graphics card).
    stacked = jnp.concatenate([x_local, x_shared.astype(x_local.dtype)], axis=0)
    return stacked[plan.place]
