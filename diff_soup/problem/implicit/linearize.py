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

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import SparseTopology
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
# --------------------------------------------------------------------------- #
_DENSE_ROW_FRACTION = 0.25  # a row reaching this share of the columns is dense
_MAX_HELD_OUT_ROWS = 256  # absolute cap on pullbacks (and their seed batch) per assembly
_MAX_HELD_OUT_FRACTION = 0.5  # past half the rows there is no compression left to save


def _pattern_keys(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Row-major position keys of a COO pattern."""
    return rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)


def _held_out_rows(problem, rows: np.ndarray, n: int) -> np.ndarray:
    """Rows to keep out of the coloring: declared globals plus dense stragglers.

    The ``"global"`` partition of the problem's [DAEStructure][] is the
    authoritative declaration of the bordered-global rows, so it is honoured
    whatever the row degrees turn out to be. Rows that are dense without being
    declared are added too — the coloring cost is the same either way, and the
    author gets a working problem instead of a silent O(n^2).

    Returns sorted unique row indices, empty when nothing is worth holding out.
    """
    declared: np.ndarray = np.empty((0,), dtype=np.int64)
    structure = getattr(problem, "structure", None)
    for name, lo, hi in getattr(structure, "partitions", ()) or ():
        if name == "global" and int(hi) > int(lo):
            declared = np.arange(int(lo), int(hi), dtype=np.int64)

    degrees = np.bincount(np.asarray(rows, dtype=np.int64), minlength=n)
    dense = np.flatnonzero(degrees >= _DENSE_ROW_FRACTION * n).astype(np.int64)

    held = np.union1d(declared, dense)
    budget = min(_MAX_HELD_OUT_ROWS, max(1, int(_MAX_HELD_OUT_FRACTION * n)))
    if held.size == 0 or held.size > budget:
        # Nothing dense, or so much of the matrix is dense that a pullback per
        # held-out row buys no compression over just coloring it (a genuinely
        # dense small block is the usual case) — leave the pattern whole.
        return np.empty((0,), dtype=np.int64)
    return held


class _RowSplit:
    """A COO pattern split into colored rows and held-out (dense) rows.

    ``interior_take`` / ``border_take`` index the pattern's own entry array, so
    values from the two routes scatter back into the caller's declared order.
    ``border_seed`` maps each held-out entry to its row's position in the
    pullback seed batch, and ``border_cols`` to its column.
    """

    __slots__ = ("interior_take", "border_take", "border_seed", "border_cols", "held_rows", "n")

    def __init__(self, rows: np.ndarray, cols: np.ndarray, held_rows: np.ndarray, n: int):
        rows = np.asarray(rows, dtype=np.int64)
        is_border = np.isin(rows, held_rows)
        self.n = int(n)
        self.held_rows = np.asarray(held_rows, dtype=np.int32)
        self.interior_take = np.flatnonzero(~is_border).astype(np.int32)
        self.border_take = np.flatnonzero(is_border).astype(np.int32)
        # Position of each held-out entry's row within ``held_rows``.
        self.border_seed = np.searchsorted(held_rows, rows[is_border]).astype(np.int32)
        self.border_cols = np.asarray(cols, dtype=np.int32)[is_border]

    @property
    def active(self) -> bool:
        """Whether any row is actually held out of the coloring."""
        return self.border_take.size > 0

    def border_values(self, fn, primal, dtype=None):
        """Held-out row values via one residual pullback per held-out row."""
        n_held = int(self.held_rows.size)
        residual, pull = jax.vjp(fn, primal)
        # A VJP cotangent must match the residual's dtype exactly.  In
        # particular, a user residual may legitimately promote a reduced-
        # precision state (for example through a structural coefficient).
        # Deriving the seed dtype from an independently assembled sparse
        # interior is both unnecessary and incorrect in that case.
        residual_dtype = jnp.asarray(residual).dtype
        seeds = jnp.zeros((n_held, self.n), dtype=residual_dtype)
        seeds = seeds.at[jnp.arange(n_held), jnp.asarray(self.held_rows)].set(1.0)
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


def _dedup(topology: SparseTopology) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate a pattern; return (rows, cols, map_from_declared_entry)."""
    rows = np.asarray(topology.rows, dtype=np.int32)
    cols = np.asarray(topology.cols, dtype=np.int32)
    keys = _pattern_keys(rows, cols, int(topology.n_dofs))
    uniq, inverse = np.unique(keys, return_inverse=True)
    n = int(topology.n_dofs)
    return (uniq // n).astype(np.int32), (uniq % n).astype(np.int32), inverse.astype(np.int32)


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
            fy_rows, fy_cols, self._fy_declared_map = _dedup(dy_decl)
            fyd_rows, fyd_cols, self._fydot_declared_map = _dedup(dydot_decl)
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
            held = _held_out_rows(problem, u_rows, n)
            splits = [
                _RowSplit(fy_rows, fy_cols, held, n),
                _RowSplit(fyd_rows, fyd_cols, held, n),
                _RowSplit(u_rows, u_cols, held, n),
            ]
            if any(s.active and s.interior_take.size == 0 for s in splits):
                # A pattern whose every entry sits on a held-out row would leave
                # its coloring with nothing to color; keep that pattern whole.
                held = np.empty((0,), dtype=np.int64)
                splits = [
                    _RowSplit(fy_rows, fy_cols, held, n),
                    _RowSplit(fyd_rows, fyd_cols, held, n),
                    _RowSplit(u_rows, u_cols, held, n),
                ]
            self.held_out_rows = held
            self._fy_split, self._fydot_split, self._union_split = splits

            self._fy_coloring = self._coloring(fy_rows, fy_cols, self._fy_split, n)
            self._fydot_coloring = self._coloring(fyd_rows, fyd_cols, self._fydot_split, n)
            self._union_coloring = self._coloring(u_rows, u_cols, self._union_split, n)

    @staticmethod
    def _coloring(rows, cols, split: "_RowSplit", n: int) -> SparseJacobian:
        """Color a pattern's non-held-out rows (the whole pattern when none is)."""
        if not split.active:
            return SparseJacobian.from_coo(rows, cols, n)
        keep = split.interior_take
        return SparseJacobian.from_coo(rows[keep], cols[keep], n)

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
