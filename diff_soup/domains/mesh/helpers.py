"""Geometric node-selection helpers for boundary / pin construction (pure numpy).

These are host-side topology helpers — they operate on ``Mesh.points`` and a
boundary-node index array and return plain Python ints / numpy values.  They do
not participate in a JIT trace.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def select_rigid_body_pins(points, boundary_nodes, strategy: str = "longest_axis"):
    """Pick two boundary nodes to remove the rigid-body modes of a vector field.

    ``longest_axis``: the boundary-node pair with maximum pairwise distance.
    It is the first maximum of
    ``cdist(points[boundary], points[boundary])`` in row-major order, so the
    choice is deterministic for a given node ordering.

    Args:
        points: Node coordinates, shape ``(num_nodes, dim)``.
        boundary_nodes: Global indices of the boundary nodes (1-D int array).
        strategy: Selection rule.  Only ``"longest_axis"`` is supported.

    Returns:
        ``(k1, k2, transverse_axis)`` where ``k1`` / ``k2`` are the global node
        indices of the longest-spanning boundary pair and ``transverse_axis`` is
        the component index (perpendicular to the pin axis) to additionally pin
        at ``k2`` to remove the in-plane rotation.

    Raises:
        ValueError: If ``strategy`` is not ``"longest_axis"``.
    """
    if strategy != "longest_axis":
        raise ValueError(f"unknown strategy {strategy!r}; only 'longest_axis' supported")
    pts_all = np.asarray(points)
    bnd = np.asarray(boundary_nodes, dtype=np.int64).reshape(-1)
    pts = pts_all[bnd]
    dists = cdist(pts, pts)
    i_max, j_max = np.unravel_index(int(dists.argmax()), dists.shape)
    k1, k2 = int(bnd[i_max]), int(bnd[j_max])
    delta = np.abs(pts_all[k2] - pts_all[k1])
    pin_axis = int(np.argmax(delta))  # longest-spanning axis
    if delta.size == 2:
        transverse_axis = 1 - pin_axis
    else:
        transverse_axis = int(np.argmin(delta))
    return k1, k2, transverse_axis
