"""domains/fem/assembly.py — Public FEM assembly utilities for multi-DOF problems.

Provides the standard element-level assembly pipeline used by all multi-DOF
FEM problems in diff_soup (Allen-Cahn, Cahn-Hilliard, LFP, etc.):

    elem_residual_fn  →  vmap  →  scatter-add  →  global residual
    elem_residual_fn  →  jacfwd + vmap  →  scatter-add  →  sparse Jacobian

These utilities eliminate the per-problem boilerplate for sparsity pattern
construction, element→sparse index mapping, lumped mass assembly, and
element residual/Jacobian scatter.

Example usage::

    from diff_soup.domains.mesh.fem.assembly import (
        build_dof_sparsity, build_elem_to_sparse_map,
        assemble_lumped_mass, scatter_elem_residual, scatter_elem_jacobian,
    )

    # Setup (once, at problem construction)
    rows, cols = build_dof_sparsity(cells_np, n_nodes, n_vars=4)
    elem_jac_map = build_elem_to_sparse_map(cells_np, rows, cols, n_vars=4)
    mass = assemble_lumped_mass(fe)

    # Assembly (each residual/Jacobian evaluation)
    f_elem = jax.vmap(elem_fn)(cell_sol, grads, jxw)   # (n_cells, n_local)
    R = scatter_elem_residual(f_elem, cells, n_nodes, n_vars=4)

    elem_jacs = jax.vmap(jax.jacfwd(elem_fn))(cell_sol, grads, jxw)
    Jvals = scatter_elem_jacobian(elem_jacs, elem_jac_map, nnz)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from diff_soup.domains.mesh.fem.topology import FiniteElementSpace

Array = jax.Array


# ---------------------------------------------------------------------------
# Sparsity pattern construction (numpy, done once at problem setup)
# ---------------------------------------------------------------------------


def build_dof_sparsity(
    cells_np: np.ndarray,
    n_nodes: int,
    n_vars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """COO sparsity pattern for an ``n_vars``-DOF/node FEM mesh.

    Generates all ``(row, col)`` pairs where both DOFs belong to nodes
    that share at least one element.  Sorted lexicographically.

    Args:
        cells_np: (n_cells, n_nodes_per_cell) int
            Element connectivity.
        n_nodes: Total number of mesh nodes.
        n_vars: Number of DOFs per node (e.g. 4 for [c, mu, ux, uy]).

    Returns:
        rows, cols : int32 arrays of shape ``(nnz,)``
    """
    npc = cells_np.shape[1]
    # All node pairs from connectivity: (n_cells, npc, npc)
    ni = np.broadcast_to(cells_np[:, :, None], (cells_np.shape[0], npc, npc))
    nj = np.broadcast_to(cells_np[:, None, :], (cells_np.shape[0], npc, npc))
    # Pack as composite key and deduplicate
    N = np.int64(n_nodes)
    node_keys = ni.ravel().astype(np.int64) * N + nj.ravel().astype(np.int64)
    unique_keys = np.unique(node_keys)
    node_i = (unique_keys // N).astype(np.int32)
    node_j = (unique_keys % N).astype(np.int32)
    # Expand to DOF pairs via outer product over var indices.
    # Shape: (n_pairs, n_vars, n_vars) for both rows and cols.
    n_pairs = len(node_i)
    vi = np.arange(n_vars, dtype=np.int32)
    vj = np.arange(n_vars, dtype=np.int32)
    rows_3d = n_vars * node_i[:, None, None] + vi[None, :, None]
    cols_3d = n_vars * node_j[:, None, None] + vj[None, None, :]
    rows = np.broadcast_to(rows_3d, (n_pairs, n_vars, n_vars)).ravel()
    cols = np.broadcast_to(cols_3d, (n_pairs, n_vars, n_vars)).ravel()
    order = np.lexsort((cols, rows))
    return rows[order], cols[order]


def build_elem_to_sparse_map(
    cells_np: np.ndarray,
    jac_rows: np.ndarray,
    jac_cols: np.ndarray,
    n_vars: int,
) -> np.ndarray:
    """Element→sparse index mapping for scatter-add of element Jacobians.

    Maps each entry ``(element, local_i, local_j)`` of the element Jacobian
    matrix to the corresponding index in the global sparse COO arrays
    ``(jac_rows, jac_cols)``.

    Local DOFs are interleaved: ``local_k = n_vars * node_local + var``.

    Args:
        cells_np: ``(n_cells, n_nodes_per_cell)`` int array.
        jac_rows: ``(nnz,)`` int32 row indices from
            [build_dof_sparsity][diff_soup.domains.mesh.fem.assembly.build_dof_sparsity].
        jac_cols: ``(nnz,)`` int32 column indices from
            [build_dof_sparsity][diff_soup.domains.mesh.fem.assembly.build_dof_sparsity].
        n_vars: Number of variables per node (interleaving factor).

    Returns:
        ``int32`` array of shape ``(n_cells * n_local_dofs^2,)`` where
        ``n_local_dofs = n_nodes_per_cell * n_vars``.
    """
    n_cells = cells_np.shape[0]
    n_nodes_per_cell = cells_np.shape[1]
    n_local = n_nodes_per_cell * n_vars

    # Build global DOF indices for each local DOF in each element.
    # Local DOF k maps to: n_vars * cells[e, k // n_vars] + (k % n_vars)
    local_node = np.arange(n_local) // n_vars  # (n_local,)
    local_var = np.arange(n_local) % n_vars  # (n_local,)
    # (n_cells, n_local) global DOF indices
    global_dofs = (n_vars * cells_np[:, local_node] + local_var[None, :]).astype(np.int64)

    # Build all (global_i, global_j) pairs: (n_cells, n_local, n_local)
    gi = np.broadcast_to(global_dofs[:, :, None], (n_cells, n_local, n_local))
    gj = np.broadcast_to(global_dofs[:, None, :], (n_cells, n_local, n_local))

    # Composite key for each element entry and for the sparse pattern
    N = np.int64(jac_rows.max() + 1)  # == n_nodes * n_vars
    elem_keys = gi.ravel() * N + gj.ravel()
    sparse_keys = jac_rows.astype(np.int64) * N + jac_cols.astype(np.int64)

    # The sparse pattern is already sorted lexicographically (from
    # build_dof_sparsity), so searchsorted gives the correct index.
    return np.searchsorted(sparse_keys, elem_keys).astype(np.int32)


# ---------------------------------------------------------------------------
# Mass matrix assembly (numpy, done once at problem setup)
# ---------------------------------------------------------------------------

_QUADRATIC_ELES = {"TRI6", "QUAD8", "QUAD9", "TET10", "HEX20", "HEX27"}


def assemble_lumped_mass(
    fe: FiniteElementSpace,
    *,
    scheme: str = "auto",
) -> np.ndarray:
    """Lumped mass for a scalar field on the given FE space.

    Two schemes are supported:

    - ``"row_sum"`` — ``M_i = sum_e sum_q N_i(x_q) |J_e| w_q``.  Textbook
      row-sum lumping.  Produces positive entries only for linear elements
      (TRI3, QUAD4, TET4, HEX8).  Row-summing TRI6/QUAD8/TET10 etc. yields
      *negative* corner-node entries (a known property of quadratic/
      serendipity bases), which wrecks transient solves.
    - ``"hrz"`` — Hinton–Rock–Zienkiewicz diagonal scaling.  Uses the
      diagonal of the consistent mass matrix, then rescales so the lumped
      total equals the element's total mass.  Always positive; the
      recommended scheme for quadratic/serendipity elements.
    - ``"auto"`` (default) — picks ``"row_sum"`` for linear elements and
      ``"hrz"`` for quadratic/serendipity elements to avoid silent negative
      mass entries.

    Args:
        fe: Finite element space.
        scheme: ``"auto"``, ``"row_sum"``, or ``"hrz"``.

    Returns:
        mass: float64 array of shape ``(n_nodes,)``
    """
    cells_np = np.asarray(fe.cells, dtype=np.int32)
    jxw_np = np.asarray(fe.jxw)  # (n_cells, n_q)
    svals_np = np.asarray(fe.shape_vals)  # (n_q, n_local)
    n_nodes = int(fe.num_total_nodes)

    ele_type = str(fe.ele_type).upper()
    if scheme == "auto":
        scheme = "hrz" if ele_type in _QUADRATIC_ELES else "row_sum"

    if scheme == "row_sum":
        mass = np.zeros(n_nodes, dtype=np.float64)
        for e in range(fe.num_cells):
            for qi in range(jxw_np.shape[1]):
                w = jxw_np[e, qi]
                for ni in range(svals_np.shape[1]):
                    mass[cells_np[e, ni]] += svals_np[qi, ni] * w
        if np.any(mass <= 0.0):
            from diff_soup._warnings import warn_user

            if ele_type in _QUADRATIC_ELES:
                cause = (
                    f"Row sums of a quadratic {ele_type!r} basis are negative at the "
                    f"corner nodes; use scheme='hrz' (or scheme='auto', the default)."
                )
            else:
                cause = (
                    f"Row sums of a linear {ele_type!r} basis are positive on every "
                    f"valid element, so an element here has a non-positive measure: "
                    f"check the mesh for inverted or degenerate cells."
                )
            warn_user(
                f"row_sum lumping on element type {ele_type!r} produced "
                f"non-positive mass entries (min = {float(mass.min()):.4e}), which "
                f"give nonphysical dynamics in transient solves. {cause}",
                RuntimeWarning,
            )
        return mass

    if scheme == "hrz":
        # HRZ: per-element, form the diagonal of the consistent mass,
        # then scale each diagonal entry so the element diagonals sum to the
        # element's total mass (preserving total mass).
        mass = np.zeros(n_nodes, dtype=np.float64)
        n_q = svals_np.shape[0]
        n_local = svals_np.shape[1]
        # Precompute N_i(x_q)^2 * w_q summed over q  →  (n_local,) per element.
        # Because jxw depends on element, do it per-element.
        for e in range(fe.num_cells):
            # Element consistent-mass diagonal:
            #   M_ii^e = sum_q N_i(x_q)^2 |J_e| w_q
            diag_elem = np.zeros(n_local, dtype=np.float64)
            total_elem = 0.0
            for qi in range(n_q):
                w = jxw_np[e, qi]
                total_elem += w  # sum of shape funcs = 1 → integral of 1 = vol
                for ni in range(n_local):
                    diag_elem[ni] += svals_np[qi, ni] * svals_np[qi, ni] * w
            diag_sum = float(diag_elem.sum())
            if diag_sum <= 0.0 or total_elem <= 0.0:
                raise ValueError(
                    f"HRZ lumping found element {e} of type {ele_type!r} with a "
                    f"non-positive measure (volume {total_elem:.4e}); the mesh has "
                    f"an inverted or degenerate cell. Fix its node order or remove it."
                )
            scale = total_elem / diag_sum
            diag_elem *= scale
            for ni in range(n_local):
                mass[cells_np[e, ni]] += diag_elem[ni]
        return mass

    raise ValueError(f"Unknown lumping scheme {scheme!r}. Use 'auto', 'row_sum', or 'hrz'.")


# ---------------------------------------------------------------------------
# JAX element residual scatter (called each evaluation, JIT-friendly)
# ---------------------------------------------------------------------------


def scatter_elem_residual(
    f_elem: Array,
    cells: Array,
    n_nodes: int,
    n_vars: int,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Scatter-add element residuals to a global vector.

    Args:
        f_elem: (n_cells, n_local_dofs) — vmapped element residual output
        cells: (n_cells, n_nodes_per_cell) int32 — element connectivity
        n_nodes: int — total mesh nodes
        n_vars: int — DOFs per node
        dtype: optional dtype override

    Returns:
        R: (n_nodes * n_vars,) global residual vector
    """
    if dtype is None:
        dtype = f_elem.dtype
    n_cells = cells.shape[0]
    n_nodes_per_cell = cells.shape[1]
    n_local = n_nodes_per_cell * n_vars
    # Build global DOF indices: (n_cells, n_local)
    global_dofs = (n_vars * cells[:, :, None] + jnp.arange(n_vars)[None, None, :]).reshape(
        n_cells, n_local
    )
    return (
        jnp.zeros(n_nodes * n_vars, dtype=dtype).at[global_dofs.reshape(-1)].add(f_elem.reshape(-1))
    )


def scatter_elem_jacobian(
    elem_jacs: Array,
    elem_jac_map: Array,
    nnz: int,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Scatter-add element Jacobians to global sparse COO values.

    Args:
        elem_jacs: (n_cells, n_local, n_local) — from jacfwd + vmap
        elem_jac_map: (n_cells * n_local^2,) int32 — from
            [build_elem_to_sparse_map][diff_soup.domains.mesh.fem.assembly.build_elem_to_sparse_map]
        nnz: int — number of nonzeros in the global sparse pattern
        dtype: optional dtype override

    Returns:
        Jvals: (nnz,) sparse Jacobian values in COO order
    """
    if dtype is None:
        dtype = elem_jacs.dtype
    return jnp.zeros(nnz, dtype=dtype).at[elem_jac_map].add(elem_jacs.reshape(-1))
