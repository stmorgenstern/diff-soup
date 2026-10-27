"""Sparse Jacobian assembly and coloring for JVP-based recovery.

Utilities used when a residual's Jacobian has a known sparsity pattern but
its entries must be obtained from a JAX-traced function. The coloring routines
partition columns so that one forward-mode JVP per color recovers many
Jacobian entries at once; the evaluators then pack the recovered values into a
flat array matching a COO ``(rows, cols)`` pattern.

Typical flow:

1. Build a COO pattern ``(rows, cols)`` from the FEM/FVM topology.
2. Compute color groups with [build_sparse_jvp_groups][diff_soup.utils.sparse.build_sparse_jvp_groups].
3. Pad to rectangular arrays with [build_padded_color_arrays][diff_soup.utils.sparse.build_padded_color_arrays].
4. Recover values inside JIT with [sparse_jvp_jacobian_values][diff_soup.utils.sparse.sparse_jvp_jacobian_values] (vmap,
   memory-hungry) or [sparse_jvp_jacobian_values_scan][diff_soup.utils.sparse.sparse_jvp_jacobian_values_scan] (scan,
   memory-lean).

See `diff_soup.utils.interpolation` for the analogous JIT-safe
interpolation utilities.
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def build_sparse_jvp_groups(
    rows: np.ndarray,
    cols: np.ndarray,
    n_cols: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Color sparse columns so one JVP can recover several Jacobian entries.

    Uses a greedy graph-coloring algorithm (descending-degree order) on the
    column-adjacency graph induced by the sparsity pattern. Two columns are
    adjacent iff they share at least one row.

    Args:
        rows: COO row indices, shape ``(nnz,)``.
        cols: COO column indices, shape ``(nnz,)``.
        n_cols: Total number of columns in the Jacobian.

    Returns:
        A 3-tuple ``(col_groups, entry_groups, row_groups)`` where each
        element is a tuple-of-tuples:

        * ``col_groups[c]`` — column indices assigned to color ``c``.
        * ``entry_groups[c]`` — COO entry indices whose column has color ``c``.
        * ``row_groups[c]`` — row indices for those entries (same order as
          ``entry_groups[c]``).

    Raises:
        ValueError: If the coloring is invalid (same-color columns share a row).
    """
    from scipy.sparse import csr_matrix

    rows_np = np.asarray(rows, dtype=np.int32)
    cols_np = np.asarray(cols, dtype=np.int32)
    n_rows = int(rows_np.max()) + 1 if rows_np.size else 0
    nc = int(n_cols)

    if rows_np.size == 0:
        return (), (), ()

    # Build the column adjacency graph via A^T @ A where A is the 0/1 CSR
    # incidence matrix. The product counts the rows two columns share and is
    # used only as a nonzero indicator, so its dtype must not wrap: an int8
    # count of exactly 256 shared rows would read as 0, drop the adjacency
    # and color the two columns together (a structurally wrong Jacobian).
    # int32 cannot wrap here — two columns share at most ``n_rows`` rows.
    A = csr_matrix(
        (np.ones(rows_np.size, dtype=np.int32), (rows_np, cols_np)),
        shape=(n_rows, nc),
    )
    AtA = (A.T @ A).tocsr()
    AtA.setdiag(0)
    AtA.eliminate_zeros()

    # Greedy coloring by descending degree.
    degrees = np.diff(AtA.indptr).astype(np.int32)
    colors = -np.ones(nc, dtype=np.int32)
    for col in np.argsort(-degrees, kind="stable"):
        col = int(col)
        neighbors = AtA.indices[AtA.indptr[col] : AtA.indptr[col + 1]]
        used = set(colors[neighbors].tolist()) - {-1}
        color = 0
        while color in used:
            color += 1
        colors[col] = color

    n_colors = int(colors.max()) + 1 if colors.size else 0
    col_groups = []
    entry_groups = []
    row_groups = []
    for color in range(n_colors):
        cols_group = np.nonzero(colors == color)[0].astype(np.int32)
        entry_ids = np.nonzero(colors[cols_np] == color)[0].astype(np.int32)
        row_ids = rows_np[entry_ids]
        if row_ids.size != np.unique(row_ids).size:
            raise ValueError("Sparse JVP coloring failed; same-color columns share a row.")
        col_groups.append(tuple(int(x) for x in cols_group.tolist()))
        entry_groups.append(tuple(int(x) for x in entry_ids.tolist()))
        row_groups.append(tuple(int(x) for x in row_ids.tolist()))
    return tuple(col_groups), tuple(entry_groups), tuple(row_groups)


def build_padded_color_arrays(
    column_groups: tuple[tuple[int, ...], ...],
    entry_groups: tuple[tuple[int, ...], ...],
    row_groups: tuple[tuple[int, ...], ...],
) -> tuple[Array, Array, Array, Array, Array]:
    """Pad color groups to uniform size for ``vmap`` / ``lax.scan`` evaluation.

    Args:
        column_groups: Output of [build_sparse_jvp_groups][diff_soup.utils.sparse.build_sparse_jvp_groups] first element.
        entry_groups: Output of [build_sparse_jvp_groups][diff_soup.utils.sparse.build_sparse_jvp_groups] second element.
        row_groups: Output of [build_sparse_jvp_groups][diff_soup.utils.sparse.build_sparse_jvp_groups] third element.

    Returns:
        5-tuple of JAX arrays:

        * ``col_idx``: shape ``(n_colors, max_col_group)``, int32.
        * ``col_mask``: shape ``(n_colors, max_col_group)``, bool.
        * ``entry_idx``: shape ``(n_colors, max_entry_group)``, int32.
        * ``row_idx``: shape ``(n_colors, max_entry_group)``, int32.
        * ``entry_mask``: shape ``(n_colors, max_entry_group)``, bool.
    """
    n_colors = len(column_groups)
    if n_colors == 0:
        e32 = jnp.zeros((0, 1), dtype=jnp.int32)
        eb = jnp.zeros((0, 1), dtype=bool)
        return e32, eb, e32, e32, eb

    max_col_group = max(len(g) for g in column_groups) or 1
    max_entry_group = max(len(g) for g in entry_groups) or 1

    col_idx = np.zeros((n_colors, max_col_group), dtype=np.int32)
    col_mask = np.zeros((n_colors, max_col_group), dtype=bool)
    entry_idx = np.zeros((n_colors, max_entry_group), dtype=np.int32)
    row_idx = np.zeros((n_colors, max_entry_group), dtype=np.int32)
    entry_mask = np.zeros((n_colors, max_entry_group), dtype=bool)

    for i, (cg, eg, rg) in enumerate(zip(column_groups, entry_groups, row_groups)):
        kc = len(cg)
        ke = len(eg)
        if kc > 0:
            col_idx[i, :kc] = cg
            col_mask[i, :kc] = True
        if ke > 0:
            entry_idx[i, :ke] = eg
            row_idx[i, :ke] = rg
            entry_mask[i, :ke] = True

    return (
        jnp.asarray(col_idx),
        jnp.asarray(col_mask),
        jnp.asarray(entry_idx),
        jnp.asarray(row_idx),
        jnp.asarray(entry_mask),
    )


def sparse_jvp_jacobian_values(
    res_fn: collections.abc.Callable[[Array], Array],
    u: Array,
    col_idx: Array,
    col_mask: Array,
    entry_idx: Array,
    row_idx: Array,
    entry_mask: Array,
    nnz: int,
) -> Array:
    """Recover sparse Jacobian entries via ``vmap(jvp)`` over all colors in parallel.

    All colored JVPs are evaluated simultaneously on GPU.
    Uses O(n_colors * n_dofs) memory for the batched tangent/result arrays.

    Args:
        res_fn: Residual function ``u -> r``, shape ``(n,) -> (n,)``.
        u: Current state, shape ``(n,)``.
        col_idx: Padded column indices per color, shape ``(n_colors, max_col)``.
        col_mask: Valid-entry mask for ``col_idx``, shape ``(n_colors, max_col)``.
        entry_idx: Padded COO entry indices per color, shape ``(n_colors, max_entry)``.
        row_idx: Row indices for each entry, shape ``(n_colors, max_entry)``.
        entry_mask: Valid-entry mask for ``entry_idx``, shape ``(n_colors, max_entry)``.
        nnz: Total number of non-zeros in the COO pattern.

    Returns:
        COO values array, shape ``(nnz,)``, same dtype as ``u``.
    """
    n = u.shape[0]
    n_colors = col_idx.shape[0]

    # Build all tangent vectors: (n_colors, n_dofs).
    # Use .add() not .set() — padding entries (col_idx=0, col_mask=False)
    # add 0.0 harmlessly; .set() could clobber a real 1.0 at column 0.
    tangents = jnp.zeros((n_colors, n), dtype=u.dtype)
    color_ax = jnp.arange(n_colors)[:, None]
    tangents = tangents.at[color_ax, col_idx].add(jnp.where(col_mask, 1.0, 0.0))

    # Batched JVP: all colors in parallel.
    def single_jvp(tangent):
        _, jvp_val = jax.jvp(res_fn, (u,), (tangent,))
        return jvp_val

    all_jvp = jax.vmap(single_jvp)(tangents)  # (n_colors, n_dofs)

    # Extract: for each (color, entry), grab all_jvp[color, row_idx[color, entry]].
    extracted = all_jvp[color_ax, row_idx]  # (n_colors, max_entry_group)

    # Scatter to output.
    values = jnp.zeros(nnz, dtype=u.dtype)
    values = values.at[entry_idx.reshape(-1)].add(jnp.where(entry_mask, extracted, 0.0).reshape(-1))
    return values


def sparse_jvp_jacobian_values_scan(
    res_fn: collections.abc.Callable[[Array], Array],
    u: Array,
    col_idx: Array,
    col_mask: Array,
    entry_idx: Array,
    row_idx: Array,
    entry_mask: Array,
    nnz: int,
) -> Array:
    """Like [sparse_jvp_jacobian_values][diff_soup.utils.sparse.sparse_jvp_jacobian_values] but sequential via ``lax.scan``.

    Lower memory (O(n_dofs) per step) at the cost of sequential color
    evaluation. Use when the vmap version runs out of GPU memory.

    Args:
        res_fn: Residual function ``u -> r``, shape ``(n,) -> (n,)``.
        u: Current state, shape ``(n,)``.
        col_idx: Padded column indices per color, shape ``(n_colors, max_col)``.
        col_mask: Valid-entry mask for ``col_idx``, shape ``(n_colors, max_col)``.
        entry_idx: Padded COO entry indices per color, shape ``(n_colors, max_entry)``.
        row_idx: Row indices for each entry, shape ``(n_colors, max_entry)``.
        entry_mask: Valid-entry mask for ``entry_idx``, shape ``(n_colors, max_entry)``.
        nnz: Total number of non-zeros in the COO pattern.

    Returns:
        COO values array, shape ``(nnz,)``, same dtype as ``u``.
    """
    n = u.shape[0]

    def scan_body(values, color_data):
        c_idx, c_mask, e_idx, r_idx, e_mask = color_data
        tangent = jnp.zeros(n, dtype=u.dtype).at[c_idx].add(jnp.where(c_mask, 1.0, 0.0))
        _, jvp_val = jax.jvp(res_fn, (u,), (tangent,))
        extracted = jvp_val[r_idx]
        values = values.at[e_idx].add(jnp.where(e_mask, extracted, 0.0))
        return values, None

    init = jnp.zeros(nnz, dtype=u.dtype)
    values, _ = jax.lax.scan(scan_body, init, (col_idx, col_mask, entry_idx, row_idx, entry_mask))
    return values
