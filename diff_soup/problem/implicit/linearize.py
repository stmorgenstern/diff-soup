"""Exact linearization of a solver problem's residual.

For the residual ``F(t, y, ydot, theta, control)`` and a leading coefficient
``cj``, the iteration/sensitivity matrix is always

    A(cj) = F_y + cj * F_ydot

with ``cj`` applied exactly once. A [LinearizationPlan][] precomputes the
static machinery — deduplicated supports, their union, scatter maps, and
graph colorings — and evaluates:

- split partial values (``F_y`` and ``F_ydot`` over their own supports);
- combined values over the union support, via a declared already-combined
  callback, an O(nnz) recomposition of split values, or one paired colored
  sweep seeded with ``(v, cj * v)``;
- dense references for validation.

Declared exact callbacks are accelerators only: omitting one selects the
automatic-differentiation builder, never a weaker derivative. Engine-specific
scaling or sign conventions are applied inside an engine, exactly once —
they never leak into these values.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import SparseTopology
from diff_soup.problem._ad.pattern import dedup_coo
from diff_soup.problem._ad.sparse_jac import SparseJacobian
from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem

__all__ = [
    "LinearizationPlan",
    "build_linearization_plan",
]


# --------------------------------------------------------------------------- #
# Structurally dense rows and the coloring.
#
# Column coloring compresses a Jacobian by grouping columns that share no
# nonzero row. A single STRUCTURALLY DENSE ROW — one touching (nearly) every
# column — makes every pair of columns conflict, so the coloring degenerates to
# one color per column and the vmap-over-colors assembly costs O(n^2) in both
# time and memory. A dense *column* is harmless: conflicts are per row.
#
# Bordered-global rows (including algebraic closures such as galvanostatic,
# domain-, or boundary-integral constraints) are kept outside the state block,
# and there are only ``n_lam`` of them. Holding them out of the coloring and recovering each with one
# residual pullback costs ``n_lam`` extra evaluations and leaves the coloring at
# the physics stencil's scale, so total assembly cost stops scaling with ``n``.
#
# The same treatment applies to any other structurally dense row, whatever
# produced it (a nonlocal operator, an under-declared boundary stencil), as long
# as few enough rows are dense that pullbacks are cheaper than the colors they
# save.
#
# A row can set the color count without being dense. A member row that reads
# every site of its member (a reaction rate, a particle mean) makes the color
# count at least the member's size, yet the same rows of different members share
# no column except the globals and host fields, whose entries the coloring keeps.
# One pullback then recovers all of them, so the coloring falls to the stencils.
# --------------------------------------------------------------------------- #
# Ours, each: a row reaching this share of the columns is dense.
_DENSE_ROW_FRACTION = 0.25
# Ours: the cap on pullbacks per assembly, which bounds their seed batch at 256 rows of n.
_MAX_HELD_OUT_ROWS = 256
# Past half the rows held out, the coloring has nothing left to compress.
_MAX_HELD_OUT_FRACTION = 0.5
# Ours: a row this many times wider than the median sets the color count.
_WIDE_ROW_FACTOR = 2
# The colors one pullback is worth: a reverse sweep costs at most four residual
# evaluations (Griewank and Walther, Evaluating Derivatives, 2nd ed., 2008, ch. 4),
# and one color costs about one.
_PULLBACK_COLORS = 4
# Ours: the reader counts tried for a column the coloring takes off the wide rows.
_SHARED_COLUMN_READERS = (2, 4, 8, 16)
# Ours: the passes in which the rows that color a colliding column go to pullbacks.
_RECOLOR_PASSES = 4
_NONE = np.empty((0,), dtype=np.int64)


class _WideRows(NamedTuple):
    """Wide rows that share pullbacks.

    Attributes:
        wide: The rows, sorted.
        groups: Each row's pullback group.
        shared: Columns whose entries on the wide rows stay with the coloring.
    """

    wide: np.ndarray
    groups: np.ndarray
    shared: np.ndarray


class _HeldOut(NamedTuple):
    """Rows kept out of the coloring.

    Attributes:
        held: The rows, sorted.
        seeds: Each row's pullback.
        wide: The held rows that share a pullback.
        shared: Columns whose entries on ``wide`` stay with the coloring.
    """

    held: np.ndarray
    seeds: np.ndarray
    wide: np.ndarray
    shared: np.ndarray


def _colliding_columns(rows, cols, split, coloring) -> np.ndarray:
    """Columns a wide row's pullback recovers that share a color with a colored entry of that row.

    A pullback recovers most of a wide row, so the coloring does not see those
    columns; one in the color of a colored entry of the row would add to that
    entry in the tangent it reads.
    """
    color_of = coloring.column_colors()
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    colors = color_of[cols]
    interior = np.zeros(rows.size, dtype=bool)
    interior[split.interior_take] = True
    on_wide = np.flatnonzero(np.isin(rows, split.wide) & (colors >= 0))
    keys = rows[on_wide] * coloring.n_colors + colors[on_wide]
    taken = keys[interior[on_wide]]
    return np.unique(cols[on_wide[np.isin(keys, taken) & ~interior[on_wide]]])


def _pattern_keys(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Row-major position keys of a COO pattern."""
    return rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)


def _disjoint_groups(rows: np.ndarray, cols: np.ndarray, held: np.ndarray, n: int) -> np.ndarray:
    """Greedily group held-out rows that share no column; each group takes one pullback."""
    order = np.argsort(rows, kind="stable")
    sorted_rows, sorted_cols = rows[order], cols[order]
    starts = np.searchsorted(sorted_rows, held, side="left")
    stops = np.searchsorted(sorted_rows, held, side="right")
    degrees = stops - starts
    groups = np.zeros(held.size, dtype=np.int32)
    occupied: list[np.ndarray] = []
    for position in np.argsort(-degrees, kind="stable"):
        columns = sorted_cols[starts[position] : stops[position]]
        for index, taken in enumerate(occupied):
            if not taken[columns].any():
                break
        else:
            index = len(occupied)
            occupied.append(np.zeros(n, dtype=bool))
        occupied[index][columns] = True
        groups[position] = index
    return groups


def _wide_rows(rows: np.ndarray, cols: np.ndarray, rest: np.ndarray, n: int) -> _WideRows:
    """Rows worth a shared pullback, their pullback groups, and the columns left to the coloring.

    A column many wide rows read (a global, a host field a volume's members
    gather) would keep those rows out of one pullback, so its entries stay with
    the coloring, which recovers a whole column at once. The rest of each wide
    row goes to its group's pullback.

    Args:
        rows: The pattern's row indices.
        cols: The pattern's column indices.
        rest: Row degrees with the rows already held out set to zero.
        n: The number of rows and columns.

    Returns:
        The split, empty when none lowers the cost of an assembly.
    """
    nothing = _WideRows(_NONE, np.empty((0,), dtype=np.int32), _NONE)
    if not rest.any():
        return nothing
    limit = _WIDE_ROW_FACTOR * np.median(rest[rest > 0])
    wide = np.flatnonzero(rest > limit).astype(np.int64)
    if wide.size == 0 or wide.size > _MAX_HELD_OUT_FRACTION * n:
        return nothing
    narrow = int(rest[rest <= limit].max())
    on_wide = np.isin(rows, wide)
    readers = np.bincount(cols[on_wide], minlength=n)
    best_cost, best = int(rest.max()), nothing
    for readers_allowed in _SHARED_COLUMN_READERS:
        shared = np.flatnonzero(readers > readers_allowed).astype(np.int64)
        kept = on_wide & ~np.isin(cols, shared)
        groups = _disjoint_groups(rows[kept], cols[kept], wide, n)
        left = np.bincount(rows[on_wide & ~kept], minlength=n).max(initial=0)
        cost = max(narrow, int(left)) + _PULLBACK_COLORS * (int(groups.max()) + 1)
        if cost < best_cost:
            best_cost, best = cost, _WideRows(wide, groups, shared)
    return best


def _held_out_rows(
    problem, rows: np.ndarray, cols: np.ndarray, n: int, wide_rows: bool = True
) -> _HeldOut:
    """Rows to keep out of the coloring, with the pullback each one takes.

    Declared globals and dense stragglers as described above, plus rows far wider
    than the median when that lowers the cost of an assembly. A member row that
    reads every site of its member (a reaction rate, a particle mean) sets the
    color count to the member's size, yet the same rows of different members
    share no column, so one pullback recovers them all.

    Returns:
        The rows, empty when nothing is worth holding out; each row's pullback is in
        ``range(seeds.max() + 1)``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    none = _NONE
    nothing = _HeldOut(none, np.empty((0,), dtype=np.int32), none, none)
    declared: np.ndarray = none
    structure = getattr(problem, "structure", None)
    for name, lo, hi in getattr(structure, "partitions", ()) or ():
        if name == "global" and int(hi) > int(lo):
            declared = np.arange(int(lo), int(hi), dtype=np.int64)

    degrees = np.bincount(rows, minlength=n)
    dense = np.flatnonzero(degrees >= _DENSE_ROW_FRACTION * n).astype(np.int64)

    held = np.union1d(declared, dense)
    budget = min(_MAX_HELD_OUT_ROWS, max(1, int(_MAX_HELD_OUT_FRACTION * n)))
    if held.size > budget:
        # So much of the matrix is dense that a pullback per held-out row buys
        # no compression over just coloring it (a genuinely dense small block
        # is the usual case) -- leave the pattern whole.
        return nothing

    rest = degrees.copy()
    rest[held] = 0
    wide, groups, shared = (
        _wide_rows(rows, cols, rest, n)
        if wide_rows
        else (none, np.empty((0,), dtype=np.int32), none)
    )
    if held.size == 0 and wide.size == 0:
        return nothing
    return _HeldOut(*_seeded(held, wide, groups), wide, shared)


def _seeded(held: np.ndarray, wide: np.ndarray, groups: np.ndarray) -> tuple:
    """Sorted held-out rows and each one's pullback: one per row of ``held``, one per group of ``wide``."""
    all_held = np.concatenate([held, wide])
    seeds = np.concatenate([np.arange(held.size, dtype=np.int32), groups + held.size])
    order = np.argsort(all_held)
    return all_held[order], seeds[order]


class _RowSplit:
    """A COO pattern split into colored rows and held-out (dense) rows.

    ``interior_take`` / ``border_take`` index the pattern's own entry array, so
    values from the two routes scatter back into the caller's declared order.
    ``border_seed`` maps each held-out entry to its row's pullback in the seed
    batch, and ``border_cols`` to its column. ``row_seeds`` gives each held-out
    row its pullback. Rows that share one have no column in common once the
    ``shared`` columns are set aside, whose entries on ``wide`` rows stay with
    the coloring.
    """

    __slots__ = (
        "interior_take",
        "border_take",
        "border_seed",
        "border_cols",
        "held_rows",
        "row_seeds",
        "wide",
        "n",
    )

    def __init__(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        held_rows: np.ndarray,
        row_seeds: np.ndarray,
        n: int,
        wide: np.ndarray = _NONE,
        shared: np.ndarray = _NONE,
    ):
        rows = np.asarray(rows, dtype=np.int64)
        is_border = np.isin(rows, held_rows) & ~(np.isin(rows, wide) & np.isin(cols, shared))
        self.n = int(n)
        self.held_rows = np.asarray(held_rows, dtype=np.int32)
        self.row_seeds = np.asarray(row_seeds, dtype=np.int32)
        self.interior_take = np.flatnonzero(~is_border).astype(np.int32)
        self.border_take = np.flatnonzero(is_border).astype(np.int32)
        # The pullback of each held-out entry's row.
        self.border_seed = self.row_seeds[np.searchsorted(held_rows, rows[is_border])]
        self.wide = np.asarray(wide, dtype=np.int64)
        self.border_cols = np.asarray(cols, dtype=np.int32)[is_border]

    @property
    def active(self) -> bool:
        """Whether any row is actually held out of the coloring."""
        return self.border_take.size > 0

    def border_values(self, fn, primal, dtype=None):
        """Held-out row values via one residual pullback per group of held-out rows."""
        n_seeds = int(self.row_seeds.max()) + 1
        residual, pull = jax.vjp(fn, primal)
        # A VJP cotangent must match the residual's dtype exactly.  In
        # particular, a user residual may legitimately promote a reduced-
        # precision state (for example through a structural coefficient).
        # Deriving the seed dtype from an independently assembled sparse
        # interior is both unnecessary and incorrect in that case.
        residual_dtype = jnp.asarray(residual).dtype
        seeds = jnp.zeros((n_seeds, self.n), dtype=residual_dtype)
        seeds = seeds.at[jnp.asarray(self.row_seeds), jnp.asarray(self.held_rows)].set(1.0)
        dense_rows = jax.vmap(lambda w: pull(w)[0])(seeds)
        return dense_rows[jnp.asarray(self.border_seed), jnp.asarray(self.border_cols)]

    def merge(self, interior_values, border_values):
        """Scatter both routes into one value array in the declared entry order."""
        interior_values = jnp.asarray(interior_values)
        out = jnp.zeros(
            (self.interior_take.size + self.border_take.size,), dtype=interior_values.dtype
        )
        out = out.at[jnp.asarray(self.interior_take)].set(interior_values)
        return out.at[jnp.asarray(self.border_take)].set(border_values)


class LinearizationPlan:
    """Static linearization machinery for one solver problem.

    Identity-hashed (like the problems it serves) so it can ride a compile
    cache as a static value. Build with [build_linearization_plan][].

    Attributes:
        mode: ``"sparse"`` when both partial supports are declared patterns,
            ``"dense"`` otherwise.
        fy_topology: Deduplicated ``F_y`` support (sparse mode).
        fydot_topology: Deduplicated ``F_ydot`` support (sparse mode).
        union_topology: Deduplicated union support of ``A(cj)`` (sparse mode).
        held_out_rows: Structurally dense rows kept out of the colorings and
            recovered by residual pullback instead — the declared bordered-global
            rows plus any other row dense enough to wreck the column compression.
            Empty when every row is colored.
    """

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, problem):
        """Build supports, unions, scatter maps, and colorings for ``problem``."""
        self._problem = problem
        n = int(problem.n)
        self.n = n

        lin = getattr(problem, "linearization", None)
        self._dy_values = getattr(lin, "dy_values", None)
        self._dydot_values = getattr(lin, "dydot_values", None)
        self._combined_values_cb = getattr(lin, "combined_values", None)
        self.max_fy_reuse_steps = getattr(lin, "max_fy_reuse_steps", None)
        if self.max_fy_reuse_steps is not None and self.max_fy_reuse_steps < 0:
            raise ValueError("max_fy_reuse_steps must be nonnegative when provided.")
        self._coefficient_values_cb = None

        if isinstance(problem, MassProblem):
            op = problem.mass
            self._coefficient_values_cb = op.coefficient_values
            dydot_decl: Any = op.coefficient_sparsity
            if problem.rhs_sparsity == "dense":
                dy_decl: Any = "dense"
            else:
                patterns = [problem.rhs_sparsity]
                if op.state_action_sparsity is not None:
                    patterns.append(op.state_action_sparsity)
                rows = np.concatenate([np.asarray(p.rows, np.int32) for p in patterns])
                cols = np.concatenate([np.asarray(p.cols, np.int32) for p in patterns])
                dy_decl = SparseTopology(rows=rows, cols=cols, n_dofs=n)
        elif isinstance(problem, ImplicitProblem):
            dy_decl = lin.sparsity.dy
            dydot_decl = lin.sparsity.dydot
        else:
            raise TypeError(
                f"linearization plans serve MassProblem or ImplicitProblem; got "
                f"{type(problem).__name__}."
            )

        self.mode = "dense" if (dy_decl == "dense" or dydot_decl == "dense") else "sparse"
        self.fy_topology: Optional[SparseTopology] = None
        self.fydot_topology: Optional[SparseTopology] = None
        self.union_topology: Optional[SparseTopology] = None
        self.held_out_rows: np.ndarray = np.empty((0,), dtype=np.int64)
        if self.mode == "sparse":
            fy_rows, fy_cols, self._fy_declared_map = dedup_coo(dy_decl.rows, dy_decl.cols, n)
            fyd_rows, fyd_cols, self._fydot_declared_map = dedup_coo(
                dydot_decl.rows, dydot_decl.cols, n
            )
            self.fy_topology = SparseTopology(rows=fy_rows, cols=fy_cols, n_dofs=n)
            self.fydot_topology = SparseTopology(rows=fyd_rows, cols=fyd_cols, n_dofs=n)

            fy_keys = _pattern_keys(fy_rows, fy_cols, n)
            fyd_keys = _pattern_keys(fyd_rows, fyd_cols, n)
            union_keys = np.unique(np.concatenate([fy_keys, fyd_keys]))
            u_rows = (union_keys // n).astype(np.int32)
            u_cols = (union_keys % n).astype(np.int32)
            self.union_topology = SparseTopology(rows=u_rows, cols=u_cols, n_dofs=n)
            self._fy_to_union = np.searchsorted(union_keys, fy_keys).astype(np.int32)
            self._fydot_to_union = np.searchsorted(union_keys, fyd_keys).astype(np.int32)

            # Hold structurally dense rows out of every coloring (see above);
            # each pattern colors its own remaining rows and recovers the rest
            # by pullback.
            held, seeds, wide, shared = _held_out_rows(problem, u_rows, u_cols, n)
            patterns = ((fy_rows, fy_cols), (fyd_rows, fyd_cols), (u_rows, u_cols))
            splits = [_RowSplit(r, c, held, seeds, n, wide, shared) for r, c in patterns]
            if any(s.active and s.interior_take.size == 0 for s in splits):
                # A pattern whose every entry sits on a held-out row would leave
                # its coloring with nothing to color; keep that pattern whole.
                held = seeds = wide = shared = np.empty((0,), dtype=np.int64)
                splits = [_RowSplit(r, c, held, seeds, n) for r, c in patterns]
            colorings = [self._coloring(r, c, split, n) for (r, c), split in zip(patterns, splits)]
            if wide.size:
                # A column that collides on a wide row takes no color once the rows that
                # color it go to pullbacks too, and a colored row only ever becomes one.
                declared = np.setdiff1d(held, wide)
                for _ in range(_RECOLOR_PASSES):
                    colliding = np.concatenate(
                        [
                            _colliding_columns(r, c, split, coloring)
                            for (r, c), split, coloring in zip(patterns, splits, colorings)
                        ]
                    )
                    if colliding.size == 0:
                        break
                    readers = u_rows[np.isin(u_cols, colliding) & ~np.isin(u_rows, held)]
                    wide = np.union1d(wide, readers).astype(np.int64)
                    kept = np.isin(u_rows, wide) & ~np.isin(u_cols, shared)
                    groups = _disjoint_groups(u_rows[kept], u_cols[kept], wide, n)
                    held, seeds = _seeded(declared, wide, groups)
                    splits = [_RowSplit(r, c, held, seeds, n, wide, shared) for r, c in patterns]
                    colorings = [
                        self._coloring(r, c, split, n) for (r, c), split in zip(patterns, splits)
                    ]
                else:
                    # Take the rows that really need a pullback and color the rest.
                    held, seeds, _, _ = _held_out_rows(problem, u_rows, u_cols, n, wide_rows=False)
                    wide = shared = np.empty((0,), dtype=np.int64)
                    splits = [_RowSplit(r, c, held, seeds, n) for r, c in patterns]
                    colorings = [
                        self._coloring(r, c, split, n) for (r, c), split in zip(patterns, splits)
                    ]
            self.held_out_rows = held
            self._fy_split, self._fydot_split, self._union_split = splits
            self._fy_coloring, self._fydot_coloring, self._union_coloring = colorings

    @property
    def color_count(self) -> Optional[int]:
        """Colored tangents per assembly of ``A(cj)``, or ``None`` for a dense plan."""
        return None if self.mode == "dense" else int(self._union_coloring.n_colors)

    @property
    def pullback_count(self) -> int:
        """Residual pullbacks per assembly of ``A(cj)``, one per group of held-out rows."""
        if self.mode == "dense" or not self._union_split.active:
            return 0
        return int(self._union_split.row_seeds.max()) + 1

    @staticmethod
    def _coloring(rows, cols, split: "_RowSplit", n: int) -> SparseJacobian:
        """Color a pattern's non-held-out rows (the whole pattern when none is)."""
        if not split.active:
            return SparseJacobian.from_coo(rows, cols, n)
        keep = split.interior_take
        coloring = SparseJacobian.from_coo(rows[keep], cols[keep], n)
        if np.size(split.wide) == 0:
            return coloring
        # A column with no colored entry is left unseeded, so no wide row's colored entry sees it.
        return coloring.seeding_only(np.asarray(cols)[keep])

    def _split_values(self, split: "_RowSplit", coloring, fn, primal):
        """Colored values for the kept rows, pullback values for the held-out ones."""
        interior = coloring.evaluate(fn, primal)
        if not split.active:
            return interior
        border = split.border_values(fn, primal)
        return split.merge(interior, border)

    @property
    def has_exact_fy_values(self) -> bool:
        """Whether the problem supplies exact values for its state partial."""
        return self._dy_values is not None

    # ------------------------------------------------------------------
    # Residual closures
    # ------------------------------------------------------------------
    def _residual(self, t, y, ydot, theta, control):
        return self._problem.residual(t, y, ydot, theta, control)

    def _paired(self, t, y, ydot, theta, control, cj):
        """The paired map whose Jacobian in ``u`` is exactly ``A(cj)``."""

        def paired(u):
            return self._residual(t, u, ydot + cj * (u - y), theta, control)

        return paired

    # ------------------------------------------------------------------
    # Split partial values
    # ------------------------------------------------------------------
    def fy_values(self, t, y, ydot, theta, control):
        """``F_y`` — values over ``fy_topology`` (sparse) or dense ``(n, n)``."""
        if self.mode == "dense":
            if self._dy_values is not None:
                return self._dy_values(t, y, ydot, theta, control)
            return jax.jacfwd(lambda u: self._residual(t, u, ydot, theta, control))(y)
        if self._dy_values is not None:
            declared = jnp.asarray(self._dy_values(t, y, ydot, theta, control))
            out = jnp.zeros((self.fy_topology.rows.size,), dtype=declared.dtype)
            return out.at[self._fy_declared_map].add(declared)
        return self._split_values(
            self._fy_split,
            self._fy_coloring,
            lambda u: self._residual(t, u, ydot, theta, control),
            y,
        )

    def fydot_values(self, t, y, ydot, theta, control):
        """``F_ydot`` — values over ``fydot_topology`` (sparse) or dense ``(n, n)``."""
        if self.mode == "dense":
            if self._dydot_values is not None:
                return self._dydot_values(t, y, ydot, theta, control)
            return jax.jacfwd(lambda v: self._residual(t, y, v, theta, control))(ydot)
        if self._dydot_values is not None:
            declared = jnp.asarray(self._dydot_values(t, y, ydot, theta, control))
        elif self._coefficient_values_cb is not None:
            declared = jnp.asarray(self._coefficient_values_cb(t, y, theta, control))
        else:
            declared = None
        if declared is not None:
            out = jnp.zeros((self.fydot_topology.rows.size,), dtype=declared.dtype)
            return out.at[self._fydot_declared_map].add(declared)
        return self._split_values(
            self._fydot_split,
            self._fydot_coloring,
            lambda v: self._residual(t, y, v, theta, control),
            ydot,
        )

    def recompose(self, fy_vals, fydot_vals, cj):
        """``A(cj)`` union values from cached split partials — O(nnz), no reevaluation."""
        if self.mode == "dense":
            return fy_vals + cj * fydot_vals
        out = jnp.zeros((self.union_topology.rows.size,), dtype=jnp.asarray(fy_vals).dtype)
        out = out.at[self._fy_to_union].add(fy_vals)
        return out.at[self._fydot_to_union].add(cj * fydot_vals)

    # ------------------------------------------------------------------
    # Combined values / actions
    # ------------------------------------------------------------------
    def combined_values(self, t, y, ydot, theta, control, cj):
        """``A(cj)`` values over the union support (already combined, never rescaled)."""
        if self._combined_values_cb is not None:
            return self._combined_values_cb(t, y, ydot, theta, control, cj)
        if self.mode == "dense":
            return self.combined_dense(t, y, ydot, theta, control, cj)
        return self._split_values(
            self._union_split,
            self._union_coloring,
            self._paired(t, y, ydot, theta, control, cj),
            y,
        )

    def combined_dense(self, t, y, ydot, theta, control, cj):
        """Dense ``(n, n)`` ``A(cj)`` — the validation reference and dense-mode value."""
        if self.mode == "sparse":
            vals = jnp.asarray(self.combined_values(t, y, ydot, theta, control, cj))
            topo = self.union_topology
            rows = np.asarray(topo.rows)
            cols = np.asarray(topo.cols)
            dense = jnp.zeros((self.n, self.n), dtype=vals.dtype)
            return dense.at[rows, cols].add(vals)
        return jax.jacfwd(self._paired(t, y, ydot, theta, control, cj))(y)


def build_linearization_plan(problem) -> LinearizationPlan:
    """Build the static [LinearizationPlan][] for a solver problem."""
    return LinearizationPlan(problem)
