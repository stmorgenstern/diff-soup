"""geometry/refinement.py — Mesh uniform refinement and solution transfer for TRI3 meshes.

Provides uniform red refinement
([refine_uniform][diff_soup.domains.geometry.refinement.refine_uniform]) and
P1 nodal solution transfer between a coarse mesh and its refined counterpart
([transfer_solution_refined]
[diff_soup.domains.geometry.refinement.transfer_solution_refined]).

All routines operate on pure NumPy arrays (no JAX dependency) and return
[Mesh][diff_soup.domains.mesh.Mesh] objects directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from diff_soup.domains.mesh import Mesh

# ======================================================================
# Uniform red refinement
# ======================================================================


def refine_uniform(mesh: Mesh) -> Tuple[Mesh, np.ndarray, dict]:
    """Uniform red refinement for TRI3 meshes.

    Each triangle is subdivided into four children by inserting midpoint
    nodes on every edge.

    Args:
        mesh: Input mesh with ``ele_type="TRI3"``.

    Returns:
        refined_mesh:
            New [Mesh][diff_soup.domains.mesh.Mesh] with four times as many
            elements and midpoint nodes added.
        parent_map:
            Shape ``(n_new_cells,)`` int32 array where ``parent_map[i]``
            gives the index of the parent cell in the original mesh.
        edge_to_midpoint:
            Dict mapping ``(min_node, max_node)`` edge tuples to the new
            midpoint node index.  Pass to [transfer_solution_refined]
            [diff_soup.domains.geometry.refinement.transfer_solution_refined].

    Raises:
        ValueError: if ``mesh.ele_type`` is not ``"TRI3"``.
    """
    if mesh.ele_type != "TRI3":
        raise ValueError(f"refine_uniform requires TRI3 mesh, got {mesh.ele_type!r}")

    points = mesh.points
    cells = mesh.cells
    n_nodes = points.shape[0]
    n_cells = cells.shape[0]

    # Build unique edges and assign midpoint node indices
    edge_to_mid: Dict[Tuple[int, int], int] = {}
    next_node = n_nodes

    for tri in cells:
        for li, lj in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[li]), int(tri[lj])
            edge = (min(a, b), max(a, b))
            if edge not in edge_to_mid:
                edge_to_mid[edge] = next_node
                next_node += 1

    # Compute midpoint coordinates
    n_midpoints = len(edge_to_mid)
    mid_coords = np.empty((n_midpoints, points.shape[1]), dtype=points.dtype)
    for (a, b), mid_idx in edge_to_mid.items():
        mid_coords[mid_idx - n_nodes] = (points[a] + points[b]) / 2.0

    new_points = np.vstack([points, mid_coords])

    # Subdivide each triangle into 4 children
    new_cells = np.empty((n_cells * 4, 3), dtype=np.int32)
    for ci, tri in enumerate(cells):
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        m01 = edge_to_mid[(min(v0, v1), max(v0, v1))]
        m12 = edge_to_mid[(min(v1, v2), max(v1, v2))]
        m20 = edge_to_mid[(min(v2, v0), max(v2, v0))]

        base = ci * 4
        new_cells[base + 0] = [v0, m01, m20]
        new_cells[base + 1] = [m01, v1, m12]
        new_cells[base + 2] = [m20, m12, v2]
        new_cells[base + 3] = [m01, m12, m20]

    parent_map = np.repeat(np.arange(n_cells, dtype=np.int32), 4)

    refined_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        ele_type="TRI3",
    )
    return refined_mesh, parent_map, edge_to_mid


# ======================================================================
# Solution transfer
# ======================================================================


def transfer_solution_refined(
    u_old: np.ndarray,
    old_mesh: Mesh,
    new_mesh: Mesh,
    parent_map: np.ndarray,
    edge_to_midpoint: Dict[Tuple[int, int], int],
) -> np.ndarray:
    """Transfer a P1 nodal solution from a coarse mesh to a refined mesh.

    Old vertex nodes retain their values.  New midpoint nodes receive
    the average of the two parent edge endpoints.

    Args:
        u_old: Nodal solution on the old mesh, shape ``(n_old_nodes,)`` or
            ``(n_old_nodes, n_vars)``.
        old_mesh: The original (coarse) mesh.
        new_mesh: The refined mesh (output of
            [refine_uniform][diff_soup.domains.geometry.refinement.refine_uniform]
            or `refine_adaptive`).
        parent_map: Parent cell index array (unused directly, kept for API
            consistency; the transfer is purely node-based).
        edge_to_midpoint: Dict mapping canonical edge ``(min_node, max_node)`` to the
            new midpoint node index in ``new_mesh``.

    Returns:
        Nodal solution on the refined mesh, same dtype and number of
        columns as ``u_old``.
    """
    n_old = old_mesh.num_nodes
    n_new = new_mesh.num_nodes

    if u_old.ndim == 1:
        u_new = np.zeros(n_new, dtype=u_old.dtype)
        u_new[:n_old] = u_old
        for (a, b), mid_idx in edge_to_midpoint.items():
            u_new[mid_idx] = (u_old[a] + u_old[b]) / 2.0
    else:
        n_vars = u_old.shape[1]
        u_new = np.zeros((n_new, n_vars), dtype=u_old.dtype)
        u_new[:n_old] = u_old
        for (a, b), mid_idx in edge_to_midpoint.items():
            u_new[mid_idx] = (u_old[a] + u_old[b]) / 2.0

    return u_new
