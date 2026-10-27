"""Active-set primitives for the box-constrained Newton corrector.

Pure helpers used inside the box-Newton primal:

* [_build_constrained_mask][diff_soup.nonlinear.active_set._build_constrained_mask] — materialize a Python-time ``bool[n_u]`` mask
  from ``cfg.constrained_indices``.
* [_apply_active_set_to_blocks][diff_soup.nonlinear.active_set._apply_active_set_to_blocks] / [_apply_active_set_to_residual][diff_soup.nonlinear.active_set._apply_active_set_to_residual] —
  zero-out / identity-pin rows of the iteration matrix and RHS for DOFs
  currently at a bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from diff_soup.nonlinear.box_newton import (
        BoxConstrainedUnifiedConfig as BoxConstrainedConfig,
    )


def _build_constrained_mask(cfg: "BoxConstrainedConfig", n_u: int, dtype=jnp.bool_) -> jax.Array:
    """Reconstruct a boolean mask from ``cfg.constrained_indices``.

    Args:
        cfg: Box-corrector config; reads ``cfg.constrained_indices``.
        n_u: Total number of state DOFs.
        dtype: Mask dtype (default ``bool``).

    Returns:
        A ``(n_u,)`` mask, ``True`` at constrained DOF indices.
    """
    mask = jnp.zeros(n_u, dtype=dtype)
    if len(cfg.constrained_indices) > 0:
        idx = jnp.asarray(cfg.constrained_indices, dtype=jnp.int32)
        mask = mask.at[idx].set(True)
    return mask


def _apply_active_set_to_blocks(
    Avals: jax.Array,
    B: jax.Array,
    topo_rows: np.ndarray,
    topo_cols: np.ndarray,
    active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Modify A-block values and B coupling for active-set rows.

    Active rows (``active != 0``) get identity in ``A`` and zero in ``B``.
    Free rows (``active == 0``) are unchanged.

    Args:
        Avals: COO values of the differential block ``A``, shape ``(nnz,)``.
        B: Dense coupling block, shape ``(n_u, n_lam)``.
        topo_rows: COO row indices of ``A``, shape ``(nnz,)`` — host numpy
            (constant-folded under jit, never traced).
        topo_cols: COO col indices of ``A``, shape ``(nnz,)`` — host numpy.
        active: Per-DOF active code in ``{-1, 0, +1}``, shape ``(n_u,)``.

    Returns:
        ``(Avals_out, B_out)`` — the pinned block values and coupling.
    """
    is_active = active != 0
    row_active = is_active[topo_rows]
    is_diag = topo_rows == topo_cols
    A_active = jnp.where(is_diag, 1.0, 0.0)
    Avals_out = jnp.where(row_active, A_active, Avals)
    B_out = jnp.where(is_active[:, None], 0.0, B)
    return Avals_out, B_out


def _apply_active_set_to_residual(
    R: jax.Array,
    u: jax.Array,
    active: jax.Array,
    lb_arr: Optional[jax.Array] = None,
    ub_arr: Optional[jax.Array] = None,
) -> jax.Array:
    """Modify residual for active-set rows.

    Active at lb (``active == -1``): ``R_i = u_i - lb_i``.
    Active at ub (``active == +1``): ``R_i = u_i - ub_i``.
    Free (``active == 0``): unchanged.

    When ``lb_arr`` / ``ub_arr`` are ``None``, the ``[0, 1]`` literals are used.

    Args:
        R: Residual ``R(u)``, shape ``(n_u,)``.
        u: Current iterate, shape ``(n_u,)``.
        active: Per-DOF active code in ``{-1, 0, +1}``, shape ``(n_u,)``.
        lb_arr: Per-DOF lower bound, or ``None`` for the ``0`` literal.
        ub_arr: Per-DOF upper bound, or ``None`` for the ``1`` literal.

    Returns:
        The pinned residual, shape ``(n_u,)``.
    """
    at_lb = active == -1
    at_ub = active == 1
    if lb_arr is None:
        lb_arr = jnp.zeros_like(u)
    if ub_arr is None:
        ub_arr = jnp.ones_like(u)
    R_out = jnp.where(at_lb, u - lb_arr, jnp.where(at_ub, u - ub_arr, R))
    return R_out
