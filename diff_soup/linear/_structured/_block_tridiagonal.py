"""Block-tridiagonal factorization by block cyclic reduction, and its backend.

A block chain is a block-tridiagonal matrix of ``n`` blocks of size ``b x b``,

    T = tridiag(lower[i], diag[i], upper[i]),   i = 0 .. n - 1,

whose block row ``i`` reads blocks ``i - 1``, ``i`` and ``i + 1`` of the
unknown (``lower[i]`` multiplies ``x[i - 1]``, ``upper[i]`` multiplies ``x[i +
1]``). Several independent chains of different lengths stored back to back
are one chain whose coupling blocks at the boundaries are zero (a ragged
chain): every reduction step then adds exact zeros across a boundary, so the
segments never exchange information and no padding enters the layout.

The factorization is odd-even (block cyclic) reduction over the whole flat
chain. A level splits the current chain into its even blocks ``E`` and odd
blocks ``K``; the even blocks couple only to odd neighbours, so with the
block diagonal ``D_E`` and the couplings ``B_EK`` and ``B_KE``

    T = [[I, 0], [B_KE D_E^-1, I]] [[D_E, B_EK], [0, S]],
    S = D_K - B_KE D_E^-1 B_EK,

and ``S`` is again a block chain of half the length. The depth is
``ceil(log2(n))`` whatever the number and lengths of the segments. A level
stores the pivoted LU of every even block, ``G = D_E^-1 B_EK`` (its left and
right blocks) and the odd blocks' couplings ``B_KE``. Pivoting never crosses
a block, so the method is exact while every reduced diagonal block is
nonsingular; a zero or negligible pivot of a reduced block is reported as
singular.

A solve with ``T`` runs the levels down and back: ``g = D_E^-1 r_E``, the
reduced right-hand side ``r_K - B_KE g``, the recursion, and ``x_E = g - G
x_K``. A solve with ``T^T`` uses the same factor in the opposite order:
``T^T = [[D_E^T, 0], [B_EK^T, S^T]] [[I, D_E^-T B_KE^T], [0, I]]``, so the
reduced right-hand side is ``c_K - G^T c_E``, the recursion solves with
``S^T``, and ``x_E = D_E^-T (c_E - B_KE^T x_K)`` with the transposed small
LU. No second factorization is needed for transposed solves.

Every operation is a fixed-shape ``jax.numpy`` expression, so the factor and
the solves are ``jit``-, ``vmap``- and autodiff-transparent.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.linear._api import SparseTopology
from diff_soup.linear._factor import Analysis, SolveStatus, dirichlet_rows
from diff_soup.linear._structured._common import (
    ArrayStructuredSolver,
    index_constant,
    refuse_options,
)
from diff_soup.linear._structured._small_lu import (
    bmm,
    small_lu_factor,
    small_lu_solve,
    small_lu_solve_transpose,
)

__all__ = [
    "BlockTridiagonal",
    "ChainFactor",
    "block_tridiagonal_factor",
    "block_tridiagonal_matvec",
    "block_tridiagonal_solve",
]


class _Level(eqx.Module):
    """One odd-even reduction level of a block chain."""

    lu: Array
    perm: Array
    g_lower: Array
    g_upper: Array
    kept_lower: Array
    kept_upper: Array


class ChainFactor(eqx.Module):
    """Block-cyclic-reduction factor of a block chain.

    Attributes:
        levels: The reduction levels, outermost first.
        final_lu: Pivoted LU of the one block left after the reduction.
        final_perm: Its row permutation.
    """

    levels: tuple
    final_lu: Array
    final_perm: Array


def _swap(x: Array) -> Array:
    """Transpose the two trailing (block) axes."""
    return jnp.swapaxes(x, -1, -2)


def _pad_end(x: Array, count: int) -> Array:
    """Append ``count`` zero blocks along the chain axis (axis ``-3``)."""
    if count == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[-3] = (0, count)
    return jnp.pad(x, pad)


def _pad_start(x: Array, count: int) -> Array:
    """Prepend ``count`` zero blocks along the chain axis (axis ``-3``)."""
    pad = [(0, 0)] * x.ndim
    pad[-3] = (count, 0)
    return jnp.pad(x, pad)


def _interleave(even: Array, odd: Array, n: int) -> Array:
    """Interleave even- and odd-indexed blocks back into a chain of ``n`` blocks."""
    n_e, n_k = even.shape[-3], odd.shape[-3]
    odd = _pad_end(odd, n_e - n_k)
    stacked = jnp.stack([even, odd], axis=-3)
    shape = stacked.shape[:-4] + (2 * n_e,) + stacked.shape[-2:]
    return stacked.reshape(shape)[..., :n, :, :]


def block_tridiagonal_factor(
    lower: ArrayLike, diag: ArrayLike, upper: ArrayLike
) -> tuple[ChainFactor, Array]:
    """Factor a block chain by block cyclic reduction.

    Args:
        lower: ``(..., n, b, b)``; ``lower[..., i]`` multiplies ``x[i - 1]``.
            ``lower[..., 0]`` and every segment-start block must be zero.
        diag: ``(..., n, b, b)`` diagonal blocks.
        upper: ``(..., n, b, b)``; ``upper[..., i]`` multiplies ``x[i + 1]``.
            ``upper[..., n - 1]`` and every segment-end block must be zero.

    Returns:
        ``(factor, singular)``: the [ChainFactor][] and a boolean (one per
        batch entry) that holds where a reduced diagonal block has a zero,
        negligible or non-finite pivot.

    Raises:
        ValueError: For a chain of no blocks.
    """
    lower, diag, upper = jnp.asarray(lower), jnp.asarray(diag), jnp.asarray(upper)
    n = int(diag.shape[-3])
    if n == 0:
        raise ValueError("a block chain must hold at least one block")
    singular = jnp.zeros(diag.shape[:-3], dtype=bool)
    levels = []
    while n > 1:
        n_e, n_k = (n + 1) // 2, n // 2
        d_e, l_e, u_e = diag[..., 0::2, :, :], lower[..., 0::2, :, :], upper[..., 0::2, :, :]
        d_k, l_k, u_k = diag[..., 1::2, :, :], lower[..., 1::2, :, :], upper[..., 1::2, :, :]
        lu, perm, bad = small_lu_factor(d_e)
        singular = singular | jnp.any(bad, axis=-1)
        g_lower = small_lu_solve(lu, perm, l_e)
        g_upper = small_lu_solve(lu, perm, u_e)
        right = n_k + 1 - n_e  # 1 when n is even: the last odd block has no right neighbour
        g_lower_r = _pad_end(g_lower, right)[..., 1 : n_k + 1, :, :]
        g_upper_r = _pad_end(g_upper, right)[..., 1 : n_k + 1, :, :]
        diag = d_k - bmm(l_k, g_upper[..., :n_k, :, :]) - bmm(u_k, g_lower_r)
        new_lower = -bmm(l_k, g_lower[..., :n_k, :, :])
        new_upper = -bmm(u_k, g_upper_r)
        levels.append(
            _Level(
                lu=lu, perm=perm, g_lower=g_lower, g_upper=g_upper, kept_lower=l_k, kept_upper=u_k
            )
        )
        lower, upper, n = new_lower, new_upper, n_k
    final_lu, final_perm, bad = small_lu_factor(diag[..., 0, :, :])
    singular = singular | bad
    factor = ChainFactor(levels=tuple(levels), final_lu=final_lu, final_perm=final_perm)
    return factor, singular


def _solve_forward(factor: ChainFactor, rhs: Array) -> Array:
    """``T x = rhs`` down the levels and back."""

    def down(index: int, r: Array) -> Array:
        if index == len(factor.levels):
            x = small_lu_solve(factor.final_lu, factor.final_perm, r[..., 0, :, :])
            return x[..., None, :, :]
        level = factor.levels[index]
        n = int(r.shape[-3])
        n_e, n_k = (n + 1) // 2, n // 2
        r_e, r_k = r[..., 0::2, :, :], r[..., 1::2, :, :]
        g = small_lu_solve(level.lu, level.perm, r_e)
        g_r = _pad_end(g, n_k + 1 - n_e)[..., 1 : n_k + 1, :, :]
        reduced = r_k - bmm(level.kept_lower, g[..., :n_k, :, :]) - bmm(level.kept_upper, g_r)
        x_k = down(index + 1, reduced)
        # x_e[j] = g[j] - G_L[j] x_k[j - 1] - G_U[j] x_k[j]
        x_k_left = _pad_start(x_k, 1)[..., :n_e, :, :]
        x_k_right = _pad_end(x_k, n_e - n_k)
        x_e = g - bmm(level.g_lower, x_k_left) - bmm(level.g_upper, x_k_right)
        return _interleave(x_e, x_k, n)

    return down(0, rhs)


def _solve_transposed(factor: ChainFactor, rhs: Array) -> Array:
    """``T^T x = rhs`` from the same factor, the levels in the opposite order."""

    def down(index: int, c: Array) -> Array:
        if index == len(factor.levels):
            x = small_lu_solve_transpose(factor.final_lu, factor.final_perm, c[..., 0, :, :])
            return x[..., None, :, :]
        level = factor.levels[index]
        n = int(c.shape[-3])
        n_e, n_k = (n + 1) // 2, n // 2
        c_e, c_k = c[..., 0::2, :, :], c[..., 1::2, :, :]
        right = n_k + 1 - n_e
        # (G^T c_E)[o] = G_U[o]^T c_E[o] + G_L[o + 1]^T c_E[o + 1]
        g_lower_r = _pad_end(level.g_lower, right)[..., 1 : n_k + 1, :, :]
        c_e_r = _pad_end(c_e, right)[..., 1 : n_k + 1, :, :]
        reduced = (
            c_k
            - bmm(_swap(level.g_upper[..., :n_k, :, :]), c_e[..., :n_k, :, :])
            - bmm(_swap(g_lower_r), c_e_r)
        )
        x_k = down(index + 1, reduced)
        # (B_KE^T x_K)[e] = lower_K[e]^T x_K[e] + upper_K[e - 1]^T x_K[e - 1]
        lower_e = _pad_end(level.kept_lower, n_e - n_k)
        x_k_e = _pad_end(x_k, n_e - n_k)
        upper_prev = _pad_start(level.kept_upper, 1)[..., :n_e, :, :]
        x_k_prev = _pad_start(x_k, 1)[..., :n_e, :, :]
        w = c_e - bmm(_swap(lower_e), x_k_e) - bmm(_swap(upper_prev), x_k_prev)
        x_e = small_lu_solve_transpose(level.lu, level.perm, w)
        return _interleave(x_e, x_k, n)

    return down(0, rhs)


def block_tridiagonal_solve(
    factor: ChainFactor, rhs: ArrayLike, *, transpose: bool = False
) -> Array:
    """Solve ``T x = rhs`` (or ``T^T x = rhs``) with a [block_tridiagonal_factor][] result.

    Args:
        factor: The chain factor.
        rhs: ``(..., n, b, m)`` right-hand sides; the batch axes match the
            factor's.
        transpose: Solve with ``T^T`` from the same factor.

    Returns:
        ``x`` with the shape of ``rhs``.
    """
    rhs = jnp.asarray(rhs)
    return _solve_transposed(factor, rhs) if transpose else _solve_forward(factor, rhs)


def block_tridiagonal_matvec(
    lower: ArrayLike, diag: ArrayLike, upper: ArrayLike, x: ArrayLike, *, transpose: bool = False
) -> Array:
    """Apply ``T`` (or ``T^T``) to ``x`` of shape ``(..., n, b, m)``."""
    lower, diag, upper, x = (jnp.asarray(a) for a in (lower, diag, upper, x))
    if transpose:
        lower, diag, upper = (
            _pad_start(_swap(upper), 1)[..., :-1, :, :],
            _swap(diag),
            _pad_end(_swap(lower), 1)[..., 1:, :, :],
        )
    out = bmm(diag, x)
    out = out + bmm(lower, _pad_start(x, 1)[..., :-1, :, :])
    return out + bmm(upper, _pad_end(x, 1)[..., 1:, :, :])


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class _ChainPlan:
    """Host maps from a value vector to the blocks of the chain (one per analysis)."""

    __slots__ = ("n", "n_blocks", "block_size", "slots", "keep", "unit_slots")

    def __init__(self, n, n_blocks, block_size, slots, keep, unit_slots):
        """Record the maps (built by ``BlockTridiagonal._analyze``)."""
        self.n, self.n_blocks, self.block_size = n, n_blocks, block_size
        self.slots, self.keep, self.unit_slots = slots, keep, unit_slots


def chain_plan(topology: SparseTopology, block_size: int, *, caller: str) -> _ChainPlan:
    """Map every pattern entry of ``topology`` to a slot of the chain's blocks.

    Entry ``(r, c)`` lands in block row ``r // b`` and block column ``c //
    b``, which must differ by at most one. Every entry of a Dirichlet row is
    dropped (its value is multiplied by zero) and ``1`` is added at the row's
    diagonal; when ``b`` does not divide ``n`` the chain is padded to whole
    blocks with identity rows.

    Raises:
        ConfigurationError: Naming the first entry outside the block band.
    """
    b = int(block_size)
    n = int(topology.n_dofs)
    n_blocks = max(1, -(-n // b))
    rows = np.asarray(topology.rows, dtype=np.int64)
    cols = np.asarray(topology.cols, dtype=np.int64)
    offset = cols // b - rows // b
    outside = np.flatnonzero(np.abs(offset) > 1)
    if outside.size:
        k = int(outside[0])
        raise ConfigurationError(
            f"{caller} needs every pattern entry inside the block-tridiagonal band of block size "
            f"{b} (block row and block column at most one apart); entry ({rows[k]}, {cols[k]}) "
            f"joins block row {rows[k] // b} to block column {cols[k] // b}. Order the unknowns "
            "so that couplings join neighbouring blocks, or use a general sparse backend."
        )
    kind = offset + 1  # 0 lower, 1 diagonal, 2 upper
    block_area = b * b
    slots = ((kind * n_blocks + rows // b) * block_area + (rows % b) * b + (cols % b)).astype(
        np.int64
    )
    mask, diagonal = dirichlet_rows(topology)
    keep = np.ones(rows.shape, np.float64) if mask is None else np.where(mask, 0.0, 1.0)
    padded = np.arange(n, n_blocks * b, dtype=np.int64)
    units = np.concatenate([np.asarray(diagonal, np.int64), padded])
    unit_slots = (n_blocks + units // b) * block_area + (units % b) * (b + 1)
    keep.flags.writeable = False
    return _ChainPlan(
        n, n_blocks, b, index_constant(slots), keep, index_constant(unit_slots.astype(np.int64))
    )


def chain_blocks(plan: _ChainPlan, values: Array) -> tuple[Array, Array, Array]:
    """The ``(lower, diag, upper)`` blocks ``(n_blocks, b, b)`` of the assembled ``A_bc``."""
    b, nb = plan.block_size, plan.n_blocks
    flat = jnp.zeros((3 * nb * b * b,), values.dtype)
    flat = flat.at[plan.slots].add(values * plan.keep.astype(values.dtype))
    if plan.unit_slots.size:
        flat = flat.at[plan.unit_slots].add(jnp.ones((), values.dtype))
    blocks = flat.reshape(3, nb, b, b)
    return blocks[0], blocks[1], blocks[2]


def chain_columns(plan: _ChainPlan, columns: Array) -> Array:
    """Columns ``(n, m)`` as chain right-hand sides ``(n_blocks, b, m)`` (zero padded)."""
    pad = plan.n_blocks * plan.block_size - plan.n
    if pad:
        columns = jnp.pad(columns, ((0, pad), (0, 0)))
    return columns.reshape(plan.n_blocks, plan.block_size, -1)


class BlockTridiagonal(ArrayStructuredSolver):
    """A direct solver in JAX for block-tridiagonal matrices, on the CPU and on GPUs.

    Every pattern entry must join a block to itself or to a neighbouring block, or `analyze`
    raises. It factors float64 matrices only. Its factor needs no `SolveContext`.

    Args:
        block_size: The unknowns per block, for example `1` for one field on a one-dimensional
            grid.
    """

    backend_name = "block_tridiagonal"

    block_size: int = eqx.field(static=True)

    def __init__(self, block_size: int):
        """Configure the block size."""
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
            raise ConfigurationError(
                f"BlockTridiagonal needs block_size as a positive integer; got {block_size!r}."
            )
        if int(block_size) < 1:
            raise ConfigurationError(
                f"BlockTridiagonal needs block_size >= 1; got {int(block_size)}."
            )
        self.block_size = int(block_size)

    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _ChainPlan:
        """Map the pattern onto the chain's blocks; refuse an entry outside the band.

        Raises:
            ConfigurationError: For backend options, or an entry outside the
                block-tridiagonal band.
        """
        del layout, precision
        refuse_options("BlockTridiagonal", options)
        return chain_plan(topology, self.block_size, caller="BlockTridiagonal")

    def _factor_tree(self, analysis: Analysis, values: Array, precision: str) -> tuple:
        """The chain factor of ``A_bc`` and its status (``SINGULAR`` for a failed pivot)."""
        del precision
        plan = analysis.backend_analysis
        factor, singular = block_tridiagonal_factor(*chain_blocks(plan, values))
        finite = jnp.all(
            jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(factor)])
        )
        bad = singular | ~finite
        return factor, jnp.where(bad, int(SolveStatus.SINGULAR), int(SolveStatus.SUCCESS))

    def _apply_tree(
        self,
        analysis: Analysis,
        values: Array,
        tree: Any,
        status: Array,
        columns: Array,
        transpose: bool,
    ) -> tuple:
        """Solve the chain (or its transpose) for the columns."""
        del values
        plan = analysis.backend_analysis
        rhs = chain_columns(plan, columns)
        x = block_tridiagonal_solve(tree, rhs, transpose=transpose)
        return x.reshape(plan.n_blocks * plan.block_size, -1)[: plan.n], status
