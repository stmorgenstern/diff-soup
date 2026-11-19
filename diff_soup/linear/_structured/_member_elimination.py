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
composes with the shared backend. Members shorter than the longest are padded
with identity rows; no padded value reaches a result. Every step is a
fixed-shape ``jax.numpy`` expression on plain transposes, so real and complex
values, ``jit`` and ``vmap`` all work.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.linear._api import SparseTopology
from diff_soup.linear._factor import SolveStatus, dirichlet_rows
from diff_soup.linear._structured._common import index_constant
from diff_soup.linear._structured._member_chain import ELIMINATIONS, chain_factor, chain_solve
from diff_soup.linear._structured._small_lu import (
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
    n_max = int(lengths.max())

    # Owner, site and field of every unknown (-1 where it does not apply).
    owner = np.full((n,), -1, np.int64)
    site = np.full((n,), -1, np.int64)
    field = np.full((n,), -1, np.int64)
    sites = layout.sites.astype(np.int64)
    member_of_site = np.repeat(np.arange(nm), lengths)
    site_in_member = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
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

    width = n_max * b + k_max
    local = np.where(site >= 0, site * b + field, n_max * b + slot)
    loc_idx = np.full((nm, width), n, np.int64)
    members_of = owner >= 0
    loc_idx[owner[members_of], local[members_of]] = np.flatnonzero(members_of)

    plan = _Plan()
    plan.elimination = elimination
    plan.whole = False
    plan.n, plan.b, plan.nm, plan.n_max, plan.k = n, b, nm, n_max, k_max
    plan.width, plan.n_shared = width, ns
    plan.nnz = int(rows.size)
    plan.shared = index_constant(shared)
    plan.loc_idx = index_constant(loc_idx)

    mask, diagonal = dirichlet_rows(topology)
    keep = np.ones((int(rows.size),), np.float64)
    if mask is not None:
        keep[mask] = 0.0
    keep.flags.writeable = False
    plan.keep = keep
    plan.has_bc = mask is not None
    diagonal = diagonal.astype(np.int64)

    rl, cl = owner[rows] >= 0, owner[cols] >= 0
    rs_, cs_ = site[rows] >= 0, site[cols] >= 0
    entries = np.arange(rows.size, dtype=np.int64)

    # Site-site entries of one member: the chain blocks, flat over (3, M, N, b, b).
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

    def chain_slot(off, member, s, fr, fc):
        return ((((off + 1) * nm + member) * n_max + s) * b + fr) * b + fc

    plan.tt_src = index_constant(entries[k])
    plan.tt_dst = index_constant(
        chain_slot(offset, owner[rows[k]], site[rows[k]], field[rows[k]], field[cols[k]])
    )
    # Unit diagonals of the chain: padded sites and Dirichlet rows.
    pad_m, pad_s = np.nonzero(np.arange(n_max)[None, :] >= lengths[:, None])
    pad_units = chain_slot(
        0, np.repeat(pad_m, b), np.repeat(pad_s, b), np.tile(np.arange(b), pad_m.size),
        np.tile(np.arange(b), pad_m.size),
    )
    d_site = diagonal[site[diagonal] >= 0]
    bc_units = chain_slot(0, owner[d_site], site[d_site], field[d_site], field[d_site])
    plan.tt_units = index_constant(np.concatenate([pad_units, bc_units]))

    # Member scalars: E (site rows), F (scalar rows over sites), G (scalar-scalar).
    row_local, col_local = local[rows], local[cols]
    k = np.flatnonzero(rs_ & cl & ~cs_)
    plan.e_src = index_constant(entries[k])
    plan.e_dst = index_constant(
        (owner[rows[k]] * (n_max * b) + row_local[k]) * max(k_max, 1) + slot[cols[k]]
    )
    k = np.flatnonzero(rl & ~rs_ & cs_)
    plan.f_src = index_constant(entries[k])
    plan.f_dst = index_constant(
        (owner[rows[k]] * max(k_max, 1) + slot[rows[k]]) * (n_max * b) + col_local[k]
    )
    k = np.flatnonzero(rl & ~rs_ & cl & ~cs_)
    plan.g_src = index_constant(entries[k])
    plan.g_dst = index_constant((owner[rows[k]] * k_max + slot[rows[k]]) * k_max + slot[cols[k]])
    pad_m, pad_j = np.nonzero(np.arange(k_max)[None, :] >= kept[:, None])
    d_scal = diagonal[(owner[diagonal] >= 0) & (site[diagonal] < 0)]
    plan.g_units = index_constant(
        np.concatenate(
            [
                (pad_m * k_max + pad_j) * k_max + pad_j,
                (owner[d_scal] * k_max + slot[d_scal]) * k_max + slot[d_scal],
            ]
        )
    )

    # Member rows reading shared columns, compressed per member: B over (M, width, kc).
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
    first = np.cumsum(read) - read
    pair_local = np.arange(pairs.shape[0]) - first[pairs[:, 0]] if pairs.size else pairs[:, 0]
    if pairs.size:
        colmap[pairs[:, 0], pair_local] = pairs[:, 1]
        key = pairs[:, 0] * max(ns, 1) + pairs[:, 1]
        which = pair_local[np.searchsorted(key, owners * max(ns, 1) + shared_cols)]
    else:
        which = np.zeros(0, np.int64)
    plan.kc = kc
    plan.colmap = index_constant(colmap)
    plan.b_src = index_constant(entries[k])
    plan.b_dst = index_constant((owners * width + row_local[k]) * kc + which)

    # Shared rows reading member columns: C as entries.
    k = np.flatnonzero(~rl & cl)
    rm_row, rm_member = pos[rows[k]], owner[cols[k]]
    plan.c_src = index_constant(entries[k])
    plan.c_row = index_constant(rm_row)
    plan.c_col = index_constant(rm_member * width + col_local[k])

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
    plan.ss_src = index_constant(entries[kss])
    plan.ss_dst = index_constant(
        np.searchsorted(pattern, pos[rows[kss]] * max(ns, 1) + pos[cols[kss]])
    )
    d_shared = diagonal[owner[diagonal] < 0]
    plan.ss_units = index_constant(np.searchsorted(pattern, pos[d_shared] * (ns + 1)))
    fill_slot = np.full((rm_row.size * kc,), nnz_s, np.int64)
    fill_slot[real] = np.searchsorted(pattern, fill_rows[real] * max(ns, 1) + fill_cols[real])
    plan.c_fill = index_constant(fill_slot.reshape(-1, kc))
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
    keep = np.ones((plan.nnz,), np.float64)
    if mask is not None:
        keep[mask] = 0.0
    keep.flags.writeable = False
    plan.keep, plan.has_bc = keep, mask is not None
    pattern = np.unique(np.concatenate([rows * max(n, 1) + cols, np.arange(n) * (n + 1)]))
    plan.nnz_shared = int(pattern.size)
    plan.shared_topology = SparseTopology(pattern // max(n, 1), pattern % max(n, 1), n_dofs=n)
    plan.ss_dst = index_constant(np.searchsorted(pattern, rows * max(n, 1) + cols))
    plan.ss_units = index_constant(
        np.searchsorted(pattern, diagonal.astype(np.int64) * (n + 1))
    )
    return plan


class MemberFactor(NamedTuple):
    """The elimination of one lane (absent parts are ``None``)."""

    chain: Any
    ze: Optional[Array]
    s_lu: Optional[Array]
    s_perm: Optional[Array]
    w: Array


def _masked(plan: _Plan, values: Array) -> Array:
    """The values with every entry of a Dirichlet row set to zero."""
    return values * jnp.asarray(plan.keep, values.dtype) if plan.has_bc else values


def _scatter(size: int, dst: np.ndarray, src: Array) -> Array:
    """A flat array of ``size`` zeros with ``src`` added at ``dst``."""
    return jnp.zeros((size,), src.dtype).at[dst].add(src)


def _scalar_rows(plan: _Plan, base: Array) -> Optional[Array]:
    """``F`` ``(M, k, N b)``: every member's scalar rows over its sites (``None`` without scalars)."""
    if not plan.k:
        return None
    nm, nb, k = plan.nm, plan.n_max * plan.b, plan.k
    return _scatter(nm * k * nb, plan.f_dst, base[plan.f_src]).reshape(nm, k, nb)


def _chain(plan: _Plan, tree: MemberFactor, v: Array, transpose: bool) -> Array:
    """``T^-1 v`` (or ``T^-T v``) for site columns ``v`` ``(M, N b, m)``."""
    shape = (plan.nm, plan.n_max, plan.b, v.shape[-1])
    x = chain_solve(tree.chain, v.reshape(shape), transpose, plan.elimination)
    return x.reshape(v.shape)


def _local_solve(plan: _Plan, tree: MemberFactor, f: Any, v: Array, transpose: bool):
    """``A_m^-1 v`` (or ``A_m^-T v``) for local columns ``v`` ``(M, width, m)``."""
    nb = plan.n_max * plan.b
    if not plan.k:
        return _chain(plan, tree, v, transpose)
    v_sites, v_scal = v[:, :nb], v[:, nb:]
    if not transpose:
        g = _chain(plan, tree, v_sites, False)
        x_scal = small_lu_solve(tree.s_lu, tree.s_perm, v_scal - jnp.einsum("mkl,mlj->mkj", f, g))
        x_sites = g - jnp.einsum("mlk,mkj->mlj", tree.ze, x_scal)
    else:
        x_scal = small_lu_solve_transpose(
            tree.s_lu, tree.s_perm, v_scal - jnp.einsum("mlk,mlj->mkj", tree.ze, v_sites)
        )
        x_sites = _chain(plan, tree, v_sites - jnp.einsum("mkl,mkj->mlj", f, x_scal), True)
    return jnp.concatenate([x_sites, x_scal], axis=1)


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
    base = _masked(plan, values)
    if plan.whole:
        shared = _scatter(plan.nnz_shared, plan.ss_dst, base)
        if plan.ss_units.size:
            shared = shared.at[plan.ss_units].add(jnp.ones((), dtype))
        return (), shared, jnp.asarray(int(SolveStatus.SUCCESS), jnp.int32)
    nm, n_max, b, k, kc, width = plan.nm, plan.n_max, plan.b, plan.k, plan.kc, plan.width
    nb = n_max * b
    blocks = _scatter(3 * nm * nb * b, plan.tt_dst, base[plan.tt_src])
    if plan.tt_units.size:
        blocks = blocks.at[plan.tt_units].add(jnp.ones((), dtype))
    lower, diag, upper = blocks.reshape(3, nm, n_max, b, b)
    chain, bad = chain_factor(lower, diag, upper, plan.elimination)
    bad = jnp.any(bad)

    b_all = _scatter(nm * width * kc, plan.b_dst, base[plan.b_src]).reshape(nm, width, kc)
    tree = MemberFactor(chain=chain, ze=None, s_lu=None, s_perm=None, w=b_all)
    if k:
        f = _scalar_rows(plan, base)
        e = _scatter(nm * nb * k, plan.e_dst, base[plan.e_src]).reshape(nm, nb, k)
        z = _chain(plan, tree, jnp.concatenate([e, b_all[:, :nb]], axis=-1), False)
        ze, zb = z[..., :k], z[..., k:]
        g = _scatter(nm * k * k, plan.g_dst, base[plan.g_src])
        if plan.g_units.size:
            g = g.at[plan.g_units].add(jnp.ones((), dtype))
        s_local = g.reshape(nm, k, k) - jnp.einsum("mkl,mlj->mkj", f, ze)
        s_lu, s_perm, singular = small_lu_factor(s_local)
        bad = bad | jnp.any(singular)
        w_scal = small_lu_solve(s_lu, s_perm, b_all[:, nb:] - jnp.einsum("mkl,mlc->mkc", f, zb))
        w = jnp.concatenate([zb - jnp.einsum("mlk,mkc->mlc", ze, w_scal), w_scal], axis=1)
        tree = MemberFactor(chain=chain, ze=ze, s_lu=s_lu, s_perm=s_perm, w=w)
    else:
        w = _chain(plan, tree, b_all, False)
        tree = tree._replace(w=w)

    shared = _scatter(plan.nnz_shared + 1, plan.ss_dst, base[plan.ss_src])
    if plan.ss_units.size:
        shared = shared.at[plan.ss_units].add(jnp.ones((), dtype))
    if plan.c_src.size:
        update = -base[plan.c_src][:, None] * w.reshape(nm * width, kc)[plan.c_col]
        shared = shared.at[plan.c_fill].add(update)
    shared = shared[: plan.nnz_shared]
    finite = jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(shared))
    status = jnp.where(bad | ~finite, int(SolveStatus.SINGULAR), int(SolveStatus.SUCCESS))
    return tree, shared, status.astype(jnp.int32)


def _gather_local(plan: _Plan, columns: Array) -> Array:
    """Member columns ``(M, width, m)`` of ``columns`` ``(n, m)``, zero at padded places."""
    padded = jnp.concatenate([columns, jnp.zeros((1, columns.shape[1]), columns.dtype)], axis=0)
    return padded[plan.loc_idx]


def reduce_rhs(plan: _Plan, values: Array, tree: MemberFactor, columns: Array, transpose: bool):
    """The right-hand side of the shared system for columns ``(n, m)``.

    Returns:
        ``(carry, rhs_shared)``: what [back_substitute][] needs, and ``(n_shared, m)``.
    """
    if plan.whole:
        return (), columns
    nm, width, ns = plan.nm, plan.width, plan.n_shared
    m = columns.shape[1]
    base = _masked(plan, values)
    c_local = _gather_local(plan, columns)
    c_shared = columns[plan.shared]
    if not transpose:
        y = _local_solve(plan, tree, _scalar_rows(plan, base), c_local, False)
        if plan.c_src.size:
            taken = base[plan.c_src][:, None] * y.reshape(nm * width, m)[plan.c_col]
            c_shared = c_shared.at[plan.c_row].add(-taken)
        return y, c_shared
    moved = jnp.einsum("mlc,mlj->mcj", tree.w, c_local)
    rhs = jnp.zeros((ns + 1, m), columns.dtype).at[:ns].set(c_shared)
    rhs = rhs.at[plan.colmap].add(-moved)
    return c_local, rhs[:ns]


def back_substitute(
    plan: _Plan, values: Array, tree: MemberFactor, carry: Array, x_shared: Array, transpose: bool
) -> Array:
    """The solution columns ``(n, m)`` from the shared solution ``(n_shared, m)``."""
    if plan.whole:
        return x_shared
    nm, width, n = plan.nm, plan.width, plan.n
    m = x_shared.shape[1]
    if not transpose:
        padded = jnp.concatenate([x_shared, jnp.zeros((1, m), x_shared.dtype)], axis=0)
        x_local = carry - jnp.einsum("mlc,mcj->mlj", tree.w, padded[plan.colmap])
    else:
        base = _masked(plan, values)
        v = carry.reshape(nm * width, m)
        if plan.c_src.size:
            v = v.at[plan.c_col].add(-base[plan.c_src][:, None] * x_shared[plan.c_row])
        f = _scalar_rows(plan, base)
        x_local = _local_solve(plan, tree, f, v.reshape(nm, width, m), True)
    out = jnp.zeros((n + 1, m), x_local.dtype)
    out = out.at[plan.loc_idx.reshape(-1)].set(x_local.reshape(nm * width, m))
    return out.at[plan.shared].set(x_shared.astype(x_local.dtype))[:n]
