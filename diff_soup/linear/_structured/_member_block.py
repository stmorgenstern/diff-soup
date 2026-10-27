"""Member-block Schur solver: banded member blocks, their low-rank terms and a retained set.

A *member* is a group of unknowns laid out on ``n_sites`` sites with ``b``
unknowns per site (one per member field), for example one particle of a
particle population. When every entry that couples two unknowns of one
member joins sites at most one apart, and no entry couples two members,
ordering each member's unknowns site by site makes its block ``T_m`` block
tridiagonal with ``b x b`` blocks. Every other unknown is *retained* (host
fields, globals, per-member scalars). With ``M`` the member unknowns and
``R`` the retained ones,

    A = [[A_MM, A_MR], [A_RM, A_RR]],     A_MM = blockdiag_m(T_m) + U W^T,

where ``U W^T`` is an optional low-rank term whose couplings each stay inside
one member (a member's mean feeding back into its own rows, for example).
The factorization

1. factors every ``T_m`` by block cyclic reduction over one flat chain of all
   members' sites (members are independent segments of the chain);
2. folds the member terms in by the Woodbury identity, one small ``r x r``
   capacitance matrix ``C_m = I + W_m^T T_m^-1 U_m`` per member (``r`` the
   largest number of couplings of a member);
3. forms ``Z = A_MM^-1 A_MR``, stored per member over the retained columns its
   rows read, and the dense Schur complement ``S = A_RR - A_RM Z``, scaled on
   both sides and factored by pivoted LU.

A solve with ``A`` is ``y = A_MM^-1 b_M``, ``x_R = S^-1 (b_R - A_RM y)``,
``x_M = y - Z x_R``. A solve with ``A^T`` uses the same factor: ``x_R = S^-T
(c_R - Z^T c_M)`` and ``x_M = A_MM^-T (c_M - A_RM^T x_R)``, with ``A_MM^-T v =
T^-T (v - W C^-T Y^T v)`` for ``Y = T^-1 U``. Both run one chain solve.

Every solve checks the backward error of its result against the whole
operator (low-rank term and Dirichlet rows included): componentwise on every
row, except rows whose ``|A| |x| + |b|`` sits at the rounding floor, which
are held to the row-normwise bound (the two-category criterion of Arioli,
Demmel and Duff, 1989). A solve that fails it runs up to
``refinement_sweeps`` sweeps of fixed-precision iterative refinement with the
same factor; a lane that still fails returns ``nan`` with
``SolveStatus.SINGULAR``. No other backend is consulted: the caller chooses
what to do with a failed solve.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.linear._api import SparseTopology
from diff_soup.linear._factor import Analysis, SolveStatus, dirichlet_rows
from diff_soup.linear._primitive import pattern_matvec
from diff_soup.linear._structured._block_tridiagonal import (
    ChainFactor,
    block_tridiagonal_factor,
    block_tridiagonal_solve,
)
from diff_soup.linear._structured._common import (
    ArrayStructuredSolver,
    index_constant,
    refuse_options,
)
from diff_soup.linear._structured._low_rank import LowRankTerm, low_rank_matvec
from diff_soup.linear._structured._small_lu import SINGULAR_PIVOT_ROUNDOFFS

__all__ = ["MemberBlock", "MemberBlockLayout"]

_SUCCESS = int(SolveStatus.SUCCESS)

#: Rows whose componentwise scale ``(|A| |x| + |b|)_i`` falls below
#: ``_ROUNDING_FLOOR * n * eps`` times their row-normwise scale are judged by
#: the row-normwise backward error instead.
_ROUNDING_FLOOR = 1000.0


# --------------------------------------------------------------------------- #
# Layout descriptor
# --------------------------------------------------------------------------- #
class MemberBlockLayout(eqx.Module):
    """The members and the retained unknowns of a `MemberBlock` solver.

    Every unknown is either a site of exactly one member or retained.

    Args:
        members: One `(n_sites, block_size)` index array per member, with one `block_size` for
            all.
        n_dofs: The number of unknowns.
        retained: The retained unknowns, by default every unknown in no member.
        term: A `LowRankTerm` whose couplings each lie inside one member, or `None`.
        names: One name per member, used in error messages.
    """

    n_dofs: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    lengths: tuple = eqx.field(static=True)
    names: tuple = eqx.field(static=True)
    term: Optional[LowRankTerm] = eqx.field(static=True)
    _indices: bytes = eqx.field(static=True)
    _retained: bytes = eqx.field(static=True)

    def __init__(
        self,
        members: Any,
        *,
        n_dofs: int,
        retained: Any = None,
        term: Optional[LowRankTerm] = None,
        names: Any = None,
    ):
        """Validate the members, the retained set and the term, and freeze them."""
        arrays = [np.asarray(member, dtype=np.int64) for member in members]
        if not arrays:
            raise ConfigurationError("MemberBlockLayout needs at least one member.")
        b = None
        for k, array in enumerate(arrays):
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise ConfigurationError(
                    f"member {k} has an index array of shape {array.shape}; each member needs an "
                    "(n_sites, b) array with n_sites >= 1 and b >= 1."
                )
            if b is None:
                b = int(array.shape[1])
            elif int(array.shape[1]) != b:
                raise ConfigurationError(
                    f"member {k} has {array.shape[1]} fields per site but member 0 has {b}; every "
                    "member of one layout carries the same number of fields."
                )
        n = int(n_dofs)
        flat = np.concatenate([array.reshape(-1) for array in arrays])
        if flat.min() < 0 or flat.max() >= n:
            raise ConfigurationError(f"member indices must lie in [0, {n}) for n_dofs={n}.")
        if np.unique(flat).size != flat.size:
            raise ConfigurationError(
                "members must be disjoint: an unknown belongs to two members or appears twice in "
                "one member."
            )
        if retained is None:
            kept = np.setdiff1d(np.arange(n), flat)
        else:
            kept = np.asarray(retained, dtype=np.int64).reshape(-1)
            both = np.intersect1d(kept, flat)
            if both.size:
                raise ConfigurationError(
                    f"unknown {int(both[0])} is both a member site and retained; the members and "
                    "the retained set must be disjoint."
                )
            if np.unique(kept).size != kept.size or (
                kept.size and (kept.min() < 0 or kept.max() >= n)
            ):
                raise ConfigurationError(
                    f"the retained set must list distinct unknowns in [0, {n})."
                )
        if flat.size + kept.size != n:
            missing = np.setdiff1d(np.arange(n), np.concatenate([flat, kept]))
            raise ConfigurationError(
                f"the members and the retained set cover {flat.size + kept.size} of {n} unknowns; "
                f"unknown {int(missing[0])} is in neither. Every unknown is a member site or "
                "retained."
            )
        self.n_dofs = n
        self.block_size = int(b)
        self.lengths = tuple(int(array.shape[0]) for array in arrays)
        self.names = tuple(str(name) for name in names) if names is not None else ()
        if self.names and len(self.names) != len(arrays):
            raise ConfigurationError("names must give one name per member.")
        self._indices = np.ascontiguousarray(flat.astype(np.int32)).tobytes()
        self._retained = np.ascontiguousarray(np.sort(kept).astype(np.int32)).tobytes()
        if term is not None:
            if not isinstance(term, LowRankTerm) or term.n_dofs != n:
                raise ConfigurationError(
                    f"MemberBlockLayout term must be a LowRankTerm of size {n}; got {term!r}."
                )
            _member_couplings(self, term)
        self.term = term

    @classmethod
    def contiguous(
        cls,
        starts: Any,
        n_sites: Any,
        block_size: int,
        *,
        n_dofs: int,
        order: str = "field_major",
        retained: Any = None,
        term: Optional[LowRankTerm] = None,
        names: Any = None,
    ) -> MemberBlockLayout:
        """Members stored contiguously, from ``(start, n_sites)`` pairs.

        Args:
            starts: First global unknown of every member.
            n_sites: Site count of every member.
            block_size: Unknowns per site ``b``.
            n_dofs: Size of the operator.
            order: ``"field_major"`` (field ``f`` of site ``i`` at ``start + f *
                n_sites + i``) or ``"site_major"`` (at ``start + i * b + f``).
            retained: The retained set (``None``: every other unknown).
            term: Optional low-rank term of the member block.
            names: Optional member names.

        Returns:
            The layout.

        Raises:
            ConfigurationError: For an unknown ``order`` or an invalid layout.
        """
        if order not in ("field_major", "site_major"):
            raise ConfigurationError(f"order must be 'field_major' or 'site_major'; got {order!r}.")
        b = int(block_size)
        members = []
        for start, count in zip(starts, n_sites):
            i = np.arange(int(count))[:, None]
            f = np.arange(b)[None, :]
            if order == "field_major":
                members.append(int(start) + f * int(count) + i)
            else:
                members.append(int(start) + i * b + f)
        return cls(members, n_dofs=n_dofs, retained=retained, term=term, names=names)

    @property
    def indices(self) -> np.ndarray:
        """``(n_chain, b)`` global unknowns of every member site, member by member (host int32)."""
        return np.frombuffer(self._indices, dtype=np.int32).reshape(-1, self.block_size)

    @property
    def retained(self) -> np.ndarray:
        """The retained unknowns, ascending (host int32)."""
        return np.frombuffer(self._retained, dtype=np.int32)

    @property
    def n_members(self) -> int:
        """Number of members."""
        return len(self.lengths)

    @property
    def n_chain(self) -> int:
        """Total number of member sites."""
        return int(sum(self.lengths))

    def member_label(self, member: int) -> str:
        """The member's name, or ``"member <k>"``."""
        return self.names[member] if self.names else f"member {member}"

    def __hash__(self) -> int:
        """Hash by content."""
        return hash(
            (self.n_dofs, self.block_size, self.lengths, self._indices, self._retained, self.term)
        )

    def __eq__(self, other: object) -> bool:
        """Equal when the members, the retained set and the term are equal."""
        if not isinstance(other, MemberBlockLayout):
            return NotImplemented
        return (
            self.n_dofs == other.n_dofs
            and self.block_size == other.block_size
            and self.lengths == other.lengths
            and self._indices == other._indices
            and self._retained == other._retained
            and self.term == other.term
        )


def _site_maps(layout: MemberBlockLayout) -> tuple:
    """``(site_of, field_of, member_of_site)``: per unknown its site and field (``-1`` if retained)."""
    n, b = layout.n_dofs, layout.block_size
    idx = layout.indices.astype(np.int64)
    site_of = np.full((n,), -1, dtype=np.int64)
    field_of = np.full((n,), -1, dtype=np.int64)
    site_of[idx.reshape(-1)] = np.repeat(np.arange(idx.shape[0]), b)
    field_of[idx.reshape(-1)] = np.tile(np.arange(b), idx.shape[0])
    member_of_site = np.repeat(np.arange(layout.n_members), layout.lengths)
    return site_of, field_of, member_of_site


def _member_couplings(layout: MemberBlockLayout, term: LowRankTerm) -> tuple:
    """``(owner, local, rank)``: each coupling's member, its index inside the member, the width.

    Raises:
        ConfigurationError: If a row of ``U`` or ``W`` is not a member site,
            or a coupling's rows span two members.
    """
    site_of, _, member_of_site = _site_maps(layout)
    owner = np.full((term.rank,), -1, dtype=np.int64)
    for name, rows, cols in (("U", term.u_rows, term.u_cols), ("W", term.w_rows, term.w_cols)):
        rows = rows.astype(np.int64)
        outside = np.flatnonzero(site_of[rows] < 0)
        if outside.size:
            raise ConfigurationError(
                f"row {int(rows[outside[0]])} of {name} in the member term is not a member site; "
                "every coupling of a member term lies inside one member (use Woodbury over "
                "MemberBlock for couplings that reach retained unknowns or several members)."
            )
        members = member_of_site[site_of[rows]]
        for coupling, member in zip(cols.astype(np.int64), members):
            if owner[coupling] not in (-1, member):
                raise ConfigurationError(
                    f"coupling {int(coupling)} of the member term spans "
                    f"{layout.member_label(int(owner[coupling]))} and "
                    f"{layout.member_label(int(member))}; every coupling of a member term lies "
                    "inside one member (use Woodbury over MemberBlock for couplings between "
                    "members)."
                )
            owner[coupling] = member
    local = np.zeros((term.rank,), dtype=np.int64)
    counts = np.zeros((layout.n_members,), dtype=np.int64)
    for coupling in range(term.rank):
        local[coupling] = counts[owner[coupling]]
        counts[owner[coupling]] += 1
    return owner, local, int(counts.max())


# --------------------------------------------------------------------------- #
# Host plan
# --------------------------------------------------------------------------- #
class _PlainView:
    """An analysis-like view of a topology without a backend: the plain pattern matvec."""

    __slots__ = ("topology",)

    def __init__(self, topology: SparseTopology):
        """Hold the topology."""
        self.topology = topology


class _MemberPlan:
    """Host maps from a value vector to the member-block factorization (one per analysis)."""


def _member_plan(layout: MemberBlockLayout, topology: SparseTopology) -> _MemberPlan:
    """Classify every pattern entry against the layout.

    Raises:
        ConfigurationError: Naming the mismatch: another size, an entry that
            couples two members, an entry that joins sites more than one
            apart, a topology whose trailing entries are not the term's, or a
            Dirichlet row without its diagonal in the base pattern.
    """
    n, b = layout.n_dofs, layout.block_size
    if int(topology.n_dofs) != n:
        raise ConfigurationError(
            f"MemberBlock is bound to a layout of size {n} but received a topology of size "
            f"{int(topology.n_dofs)}; build a MemberBlockLayout for this system and a MemberBlock "
            "backend bound to it."
        )
    term = layout.term
    base = term.base_topology(topology, caller="MemberBlock") if term is not None else topology
    rows = np.asarray(base.rows, dtype=np.int64)
    cols = np.asarray(base.cols, dtype=np.int64)
    nnz_base = int(rows.size)
    site_of, field_of, member_of_site = _site_maps(layout)
    nc, nm = layout.n_chain, layout.n_members
    retained = layout.retained.astype(np.int64)
    nr = int(retained.size)
    ret_pos = np.full((n,), -1, dtype=np.int64)
    ret_pos[retained] = np.arange(nr)

    plan = _MemberPlan()
    plan.n, plan.b, plan.nc, plan.nm, plan.nr = n, b, nc, nm, nr
    plan.nnz_base = nnz_base
    plan.base = base
    plan.term = term
    plan.chain_idx = index_constant(layout.indices)
    plan.retained = index_constant(retained)
    plan.member_of_site = index_constant(member_of_site)

    mask, diagonal = dirichlet_rows(topology)
    keep = np.ones((int(topology.rows.size),), np.float64)
    if mask is not None:
        keep[mask] = 0.0
    base_diagonal = dirichlet_rows(base)[1]
    bc = topology.bc_indices
    if bc is not None and int(bc.size):
        missing = np.setdiff1d(np.unique(bc), base_diagonal)
        if missing.size:
            raise ConfigurationError(
                f"the Dirichlet row {int(missing[0])} has no diagonal entry in the base pattern; "
                "the base topology must include the diagonal entry (i, i) of every Dirichlet row i."
            )
    plan.keep_base = keep[:nnz_base].copy()
    plan.keep_u = keep[nnz_base:].copy()
    plan.diagonal = index_constant(diagonal)

    sr, sc = site_of[rows], site_of[cols]
    fr, fc = field_of[rows], field_of[cols]
    entries = np.arange(nnz_base, dtype=np.int64)
    in_mm = (sr >= 0) & (sc >= 0)
    in_mr = (sr >= 0) & (sc < 0)
    in_rm = (sr < 0) & (sc >= 0)
    in_rr = (sr < 0) & (sc < 0)

    # Member-member entries: the same member, sites at most one apart.
    k = np.flatnonzero(in_mm)
    other = member_of_site[sr[k]] != member_of_site[sc[k]]
    if np.any(other):
        e = k[other][0]
        raise ConfigurationError(
            f"entry ({rows[e]}, {cols[e]}) couples "
            f"{layout.member_label(int(member_of_site[sr[e]]))} to "
            f"{layout.member_label(int(member_of_site[sc[e]]))}; MemberBlock needs members that "
            "couple only through retained unknowns (or Woodbury over MemberBlock for a low-rank "
            "coupling between members)."
        )
    offset = sc[k] - sr[k]
    far = np.abs(offset) > 1
    if np.any(far):
        e = k[far][0]
        raise ConfigurationError(
            f"entry ({rows[e]}, {cols[e]}) joins sites {int(sr[e])} and {int(sc[e])} of "
            f"{layout.member_label(int(member_of_site[sr[e]]))}; MemberBlock needs a member "
            "stencil of radius 1 in the site index (a member mean belongs in the member term)."
        )
    plan.mm_src = index_constant(entries[k])
    plan.mm_dst = index_constant((((offset + 1) * nc + sr[k]) * b + fr[k]) * b + fc[k])
    chain_units = diagonal[site_of[diagonal] >= 0].astype(np.int64) if diagonal.size else diagonal
    plan.mm_units = index_constant(
        ((nc + site_of[chain_units]) * b + field_of[chain_units]) * b + field_of[chain_units]
    )

    # Member rows reading retained columns, compressed per member.
    k = np.flatnonzero(in_mr)
    owners = member_of_site[sr[k]]
    ret_cols = ret_pos[cols[k]]
    pairs = (
        np.unique(np.stack([owners, ret_cols], axis=1), axis=0)
        if k.size
        else np.zeros((0, 2), np.int64)
    )
    counts = np.bincount(pairs[:, 0], minlength=nm) if pairs.size else np.zeros(nm, np.int64)
    k_max = int(max(1, counts.max() if counts.size else 1))
    colmap = np.zeros((nm, k_max), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    local_of_pair = np.arange(pairs.shape[0]) - starts[pairs[:, 0]] if pairs.size else pairs[:, 0]
    if pairs.size:
        colmap[pairs[:, 0], local_of_pair] = pairs[:, 1]
        key = pairs[:, 0] * max(nr, 1) + pairs[:, 1]
        local = local_of_pair[np.searchsorted(key, owners * max(nr, 1) + ret_cols)]
    else:
        local = np.zeros(0, np.int64)
    plan.k_max = k_max
    plan.colmap = index_constant(colmap)
    plan.mr_src = index_constant(entries[k])
    plan.mr_dst = index_constant((sr[k] * b + fr[k]) * k_max + local)

    # Retained rows reading member columns.
    k = np.flatnonzero(in_rm)
    plan.rm_src = index_constant(entries[k])
    plan.rm_row = index_constant(ret_pos[rows[k]])
    plan.rm_col = index_constant(sc[k] * b + fc[k])
    plan.rm_target = index_constant(colmap[member_of_site[sc[k]]])

    # Retained-retained entries and the Dirichlet units of retained rows.
    k = np.flatnonzero(in_rr)
    plan.rr_src = index_constant(entries[k])
    plan.rr_dst = index_constant(ret_pos[rows[k]] * nr + ret_pos[cols[k]])
    ret_units = diagonal[ret_pos[diagonal] >= 0].astype(np.int64) if diagonal.size else diagonal
    plan.rr_units = index_constant(ret_pos[ret_units] * (nr + 1))

    # The member term: U and W as (nc, b, r) blocks over member-local couplings.
    plan.rank = 0
    if term is not None:
        owner, local_coupling, rank = _member_couplings(layout, term)
        u_rows = term.u_rows.astype(np.int64)
        w_rows = term.w_rows.astype(np.int64)
        plan.rank = rank
        plan.p_dst = index_constant(
            (site_of[u_rows] * b + field_of[u_rows]) * rank + local_coupling[term.u_cols]
        )
        q = np.zeros((nc * b * rank,), np.float64)
        np.add.at(
            q,
            (site_of[w_rows] * b + field_of[w_rows]) * rank + local_coupling[term.w_cols],
            term.w_values,
        )
        q = q.reshape(nc, b, rank)
        q.flags.writeable = False
        plan.q = q
    for name in ("keep_base", "keep_u"):
        getattr(plan, name).flags.writeable = False
    return plan


# --------------------------------------------------------------------------- #
# Factor
# --------------------------------------------------------------------------- #
class _MemberFactor(eqx.Module):
    """The member-block factor of one lane (absent parts are ``None``)."""

    chain: ChainFactor
    y: Optional[Array]
    c_lu: Optional[Array]
    c_piv: Optional[Array]
    z: Array
    s_lu: Optional[Array]
    s_piv: Optional[Array]
    row_scale: Optional[Array]
    col_scale: Optional[Array]


def _segment(x: Array, plan: _MemberPlan) -> Array:
    """Sum a per-site array ``(nc, ...)`` over the sites of each member: ``(nm, ...)``."""
    return jax.ops.segment_sum(
        x, jnp.asarray(plan.member_of_site), num_segments=plan.nm, indices_are_sorted=True
    )


def _small_solve(lu: Array, piv: Array, rhs: Array, transpose: bool = False) -> Array:
    """Per-member ``C_m^-1 rhs_m`` (or ``C_m^-T``) for ``(nm, r, r)`` LUs and ``(nm, r, m)``."""
    trans = 1 if transpose else 0
    return jax.vmap(lambda a, p, r: jax.scipy.linalg.lu_solve((a, p), r, trans=trans))(lu, piv, rhs)


def _pivots_bad(lu: Array, scale: Any) -> Array:
    """Whether a pivoted LU has a non-finite entry or a pivot at most 16 roundoffs of ``scale``."""
    roundoff = SINGULAR_PIVOT_ROUNDOFFS * jnp.finfo(lu.dtype).eps / 2
    pivots = jnp.abs(jnp.diagonal(lu, axis1=-2, axis2=-1))
    return ~jnp.all(jnp.isfinite(lu)) | jnp.any(pivots <= roundoff * scale)


class MemberBlock(ArrayStructuredSolver):
    """A direct solver in JAX for chains of sites that couple only through retained unknowns.

    A member is a chain of sites whose entries join each site only to itself and its
    neighbours. Each solve checks its residual row by row, and a solve that misses after its
    sweeps reports `SolveStatus.SINGULAR`.

    Args:
        layout: The `MemberBlockLayout` of the members and the retained unknowns.
        residual_tolerance: The relative residual each row of a solve must reach.
        refinement_sweeps: The most correction sweeps a solve runs to reach it.
    """

    backend_name = "member_block"

    layout: MemberBlockLayout = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    refinement_sweeps: int = eqx.field(static=True)

    def __init__(
        self,
        layout: MemberBlockLayout,
        *,
        residual_tolerance: float = 1.0e-10,
        refinement_sweeps: int = 3,
    ):
        """Bind the backend to a layout and configure the backward-error check."""
        if not isinstance(layout, MemberBlockLayout):
            raise ConfigurationError(
                f"MemberBlock needs a diff_soup.linear.MemberBlockLayout; got {type(layout).__name__}."
            )
        if not float(residual_tolerance) > 0.0:
            raise ConfigurationError(
                f"MemberBlock needs residual_tolerance > 0; got {residual_tolerance!r}."
            )
        if int(refinement_sweeps) < 0:
            raise ConfigurationError(
                f"MemberBlock needs refinement_sweeps >= 0; got {refinement_sweeps!r}."
            )
        self.layout = layout
        self.residual_tolerance = float(residual_tolerance)
        self.refinement_sweeps = int(refinement_sweeps)

    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _MemberPlan:
        """Classify the pattern against the layout.

        Raises:
            ConfigurationError: For backend options or a topology that does
                not match the layout.
        """
        del layout, precision
        refuse_options("MemberBlock", options)
        return _member_plan(self.layout, topology)

    # ---- the operator --------------------------------------------------------------
    def pattern_matvec(
        self, analysis: Analysis, dvalues: ArrayLike, x: ArrayLike, *, transpose: bool
    ) -> Array:
        """``D x`` (or ``D^T x``) for entry values ``dvalues``, the member term included.

        The family's derivative rules and residuals read the operator through
        this method; every entry in a Dirichlet row is dropped.
        """
        plan = analysis.backend_analysis
        if plan.term is None:
            return pattern_matvec(_PlainView(plan.base), dvalues, x, transpose=transpose)
        base, u = plan.term.split(dvalues)
        applied = pattern_matvec(_PlainView(plan.base), base, x, transpose=transpose)
        low_rank = low_rank_matvec(plan.term, plan.keep_u, u, x, transpose=transpose)
        return applied + low_rank.astype(applied.dtype)

    def _operator(self, analysis: Analysis, values: Array, x: Array, transpose: bool) -> Array:
        """``A_bc x`` (or ``A_bc^T x``) for columns ``x`` ``(n, m)``."""
        plan = analysis.backend_analysis
        applied = self.pattern_matvec(analysis, values, x.T, transpose=transpose).T
        if plan.diagonal.size:
            applied = applied.at[plan.diagonal].add(x[plan.diagonal])
        return applied

    def _abs_operator(self, analysis: Analysis, values: Array, x: Array, transpose: bool) -> Array:
        """``|A_bc| x`` (or its transpose) for nonnegative columns ``x`` ``(n, m)``."""
        plan = analysis.backend_analysis
        term = plan.term
        if term is None:
            base, u = values, None
        else:
            base, u = term.split(values)
        out = pattern_matvec(_PlainView(plan.base), jnp.abs(base), x.T, transpose=transpose).T
        if u is not None:
            u_abs = jnp.abs(u) * jnp.asarray(plan.keep_u, u.dtype)
            w_abs = jnp.abs(jnp.asarray(term.w_values, x.dtype))
            if transpose:
                gathered = x[term.u_rows] * u_abs[:, None]
                s = jnp.zeros((term.rank, x.shape[1]), x.dtype).at[term.u_cols].add(gathered)
                out = out.at[term.w_rows].add(w_abs[:, None] * s[term.w_cols])
            else:
                gathered = x[term.w_rows] * w_abs[:, None]
                s = jnp.zeros((term.rank, x.shape[1]), x.dtype).at[term.w_cols].add(gathered)
                out = out.at[term.u_rows].add(u_abs[:, None] * s[term.u_cols])
        if plan.diagonal.size:
            out = out.at[plan.diagonal].add(x[plan.diagonal])
        return out

    # ---- factorization ---------------------------------------------------------------
    def _factor_tree(self, analysis: Analysis, values: Array, precision: str) -> tuple:
        """The member-block factor of ``A_bc`` and its status."""
        del precision
        plan = analysis.backend_analysis
        nc, b, k_max, rank, nr = plan.nc, plan.b, plan.k_max, plan.rank, plan.nr
        dtype = values.dtype
        base = values[: plan.nnz_base] * jnp.asarray(plan.keep_base, dtype)
        blocks = jnp.zeros((3 * nc * b * b,), dtype).at[plan.mm_dst].add(base[plan.mm_src])
        if plan.mm_units.size:
            blocks = blocks.at[plan.mm_units].add(jnp.ones((), dtype))
        lower, diag, upper = blocks.reshape(3, nc, b, b)
        chain, bad = block_tridiagonal_factor(lower, diag, upper)

        z0 = jnp.zeros((nc * b * k_max,), dtype).at[plan.mr_dst].add(base[plan.mr_src])
        z0 = z0.reshape(nc, b, k_max)
        y = c_lu = c_piv = None
        if rank:
            u = values[plan.nnz_base :] * jnp.asarray(plan.keep_u, dtype)
            p = jnp.zeros((nc * b * rank,), dtype).at[plan.p_dst].add(u).reshape(nc, b, rank)
            solved = block_tridiagonal_solve(chain, jnp.concatenate([p, z0], axis=-1))
            y, t = solved[..., :rank], solved[..., rank:]
            q = jnp.asarray(plan.q, dtype)
            capacitance = jnp.eye(rank, dtype=dtype) + _segment(
                jnp.einsum("sbi,sbj->sij", q, y), plan
            )
            c_lu, c_piv = jax.vmap(jax.scipy.linalg.lu_factor)(capacitance)
            c_piv = c_piv.astype(jnp.int32)
            bad = bad | _pivots_bad(
                c_lu, jnp.maximum(1.0, jnp.max(jnp.abs(capacitance - jnp.eye(rank, dtype=dtype))))
            )
            corr = _small_solve(c_lu, c_piv, _segment(jnp.einsum("sbi,sbk->sik", q, t), plan))
            z = t - jnp.einsum("sbi,sik->sbk", y, corr[plan.member_of_site])
        else:
            z = block_tridiagonal_solve(chain, z0)

        s_lu = s_piv = row_scale = col_scale = None
        if nr:
            s = jnp.zeros((nr * nr,), dtype).at[plan.rr_dst].add(base[plan.rr_src])
            if plan.rr_units.size:
                s = s.at[plan.rr_units].add(jnp.ones((), dtype))
            s = s.reshape(nr, nr)
            z_flat = z.reshape(nc * b, k_max)
            update = -base[plan.rm_src][:, None] * z_flat[plan.rm_col]
            s = s.at[plan.rm_row[:, None], plan.rm_target].add(update)
            row_max = jnp.max(jnp.abs(s), axis=1)
            row_scale = jnp.where(row_max > 0, 1.0 / row_max, 1.0).astype(dtype)
            col_max = jnp.max(jnp.abs(s * row_scale[:, None]), axis=0)
            col_scale = jnp.where(col_max > 0, 1.0 / col_max, 1.0).astype(dtype)
            s_lu, s_piv = jax.scipy.linalg.lu_factor(s * row_scale[:, None] * col_scale[None, :])
            s_piv = s_piv.astype(jnp.int32)
            bad = bad | _pivots_bad(s_lu, 1.0)
        tree = _MemberFactor(
            chain=chain,
            y=y,
            c_lu=c_lu,
            c_piv=c_piv,
            z=z,
            s_lu=s_lu,
            s_piv=s_piv,
            row_scale=row_scale,
            col_scale=col_scale,
        )
        finite = jnp.all(
            jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)])
        )
        status = jnp.where(bad | ~finite, int(SolveStatus.SINGULAR), _SUCCESS)
        return tree, status

    # ---- solves ------------------------------------------------------------------------
    def _amm_solve(self, plan: _MemberPlan, tree: _MemberFactor, v: Array, transpose: bool):
        """``A_MM^-1 v`` (or ``A_MM^-T v``) for chain columns ``v`` ``(nc, b, m)``."""
        if not plan.rank:
            return block_tridiagonal_solve(tree.chain, v, transpose=transpose)
        q = jnp.asarray(plan.q, v.dtype)
        sites = plan.member_of_site
        if not transpose:
            t = block_tridiagonal_solve(tree.chain, v)
            corr = _small_solve(
                tree.c_lu, tree.c_piv, _segment(jnp.einsum("sbi,sbm->sim", q, t), plan)
            )
            return t - jnp.einsum("sbi,sim->sbm", tree.y, corr[sites])
        s = _small_solve(
            tree.c_lu, tree.c_piv, _segment(jnp.einsum("sbi,sbm->sim", tree.y, v), plan), True
        )
        return block_tridiagonal_solve(
            tree.chain, v - jnp.einsum("sbi,sim->sbm", q, s[sites]), transpose=True
        )

    def _structured(self, analysis, values, tree, columns, transpose):
        """The structured solve of ``A_bc x = columns`` (or the transpose), ``(n, m)``."""
        plan = analysis.backend_analysis
        nc, b, nr = plan.nc, plan.b, plan.nr
        base = values[: plan.nnz_base] * jnp.asarray(plan.keep_base, values.dtype)
        c_m = columns[plan.chain_idx]
        out = jnp.zeros_like(columns)
        if not nr:
            x_m = self._amm_solve(plan, tree, c_m, transpose)
            return out.at[plan.chain_idx.reshape(-1)].set(x_m.reshape(nc * b, -1))
        a_rm = base[plan.rm_src][:, None]
        c_r = columns[plan.retained]
        lu = (tree.s_lu, tree.s_piv)
        if not transpose:
            y = self._amm_solve(plan, tree, c_m, False)
            y_flat = y.reshape(nc * b, -1)
            rhs_r = c_r.at[plan.rm_row].add(-a_rm * y_flat[plan.rm_col])
            x_r = tree.col_scale[:, None] * jax.scipy.linalg.lu_solve(
                lu, tree.row_scale[:, None] * rhs_r
            )
            per_site = x_r[plan.colmap][plan.member_of_site]
            x_m = y - jnp.einsum("sbk,skm->sbm", tree.z, per_site)
        else:
            zt_c = _segment(jnp.einsum("sbk,sbm->skm", tree.z, c_m), plan)
            rhs_r = c_r - jnp.zeros_like(c_r).at[plan.colmap.reshape(-1)].add(
                zt_c.reshape(-1, c_r.shape[1])
            )
            x_r = tree.row_scale[:, None] * jax.scipy.linalg.lu_solve(
                lu, tree.col_scale[:, None] * rhs_r, trans=1
            )
            v = c_m.reshape(nc * b, -1).at[plan.rm_col].add(-a_rm * x_r[plan.rm_row])
            x_m = self._amm_solve(plan, tree, v.reshape(nc, b, -1), True)
        out = out.at[plan.chain_idx.reshape(-1)].set(x_m.reshape(nc * b, -1))
        return out.at[plan.retained].set(x_r)

    def _check(self, analysis, values, columns, x, transpose, row_abs):
        """``(residual, accepted)`` of the backward-error check for columns ``x``."""
        plan = analysis.backend_analysis
        residual = self._operator(analysis, values, x, transpose) - columns
        abs_x = jnp.abs(x)
        componentwise = self._abs_operator(analysis, values, abs_x, transpose) + jnp.abs(columns)
        row_normwise = componentwise + row_abs[:, None] * jnp.max(abs_x, axis=0, keepdims=True)
        floor = _ROUNDING_FLOOR * plan.n * jnp.finfo(x.dtype).eps
        scale = jnp.where(componentwise > floor * row_normwise, componentwise, row_normwise)
        error = jnp.abs(residual)
        rows_ok = ((scale == 0) & (error == 0)) | (error <= self.residual_tolerance * scale)
        accepted = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(residual)) & jnp.all(rows_ok)
        return residual, accepted

    def _apply_tree(
        self,
        analysis: Analysis,
        values: Array,
        tree: Any,
        status: Array,
        columns: Array,
        transpose: bool,
    ) -> tuple:
        """Structured solve, backward-error check and refinement sweeps."""
        ones = jnp.ones((analysis.n, 1), columns.dtype)
        row_abs = self._abs_operator(analysis, values, ones, transpose)[:, 0]
        x = self._structured(analysis, values, tree, columns, transpose)
        residual, accepted = self._check(analysis, values, columns, x, transpose, row_abs)
        good = status == _SUCCESS

        def rejected(carry):
            sweep, _, _, ok = carry
            return good & ~ok & (sweep < self.refinement_sweeps)

        def refine(carry):
            sweep, xk, rk, _ = carry
            xk = xk - self._structured(analysis, values, tree, rk, transpose)
            rk, ok = self._check(analysis, values, columns, xk, transpose, row_abs)
            return sweep + 1, xk, rk, ok

        if self.refinement_sweeps:
            _, x, _, accepted = jax.lax.while_loop(
                rejected, refine, (jnp.zeros((), jnp.int32), x, residual, accepted)
            )
        status = jnp.where(good & ~accepted, int(SolveStatus.SINGULAR), status)
        return x, status.astype(jnp.int32)
