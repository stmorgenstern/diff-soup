"""QUAD8 (serendipity quadratic quadrilateral) shape functions on [-1, 1]^2."""

from __future__ import annotations

import numpy as np


def _quad8_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QUAD8 serendipity shape functions on [-1,1]^2.

    Nodes: corners (-1,-1),(1,-1),(1,1),(-1,1)
           midedges (0,-1),(1,0),(0,1),(-1,0)
    pts : (n_pts, 2)
    Returns vals (n_pts, 8) and grads (n_pts, 8, 2).
    """
    r, s = pts[:, 0], pts[:, 1]

    # Corner nodes: (r_i, s_i)
    rc = np.array([-1.0, 1.0, 1.0, -1.0])
    sc = np.array([-1.0, -1.0, 1.0, 1.0])

    vals_list = []
    dr_list = []
    ds_list = []

    # Corners i=0..3
    for i in range(4):
        ri, si = rc[i], sc[i]
        rr = 1.0 + ri * r
        ss = 1.0 + si * s
        vals_list.append(0.25 * rr * ss * (ri * r + si * s - 1.0))
        dr_list.append(0.25 * (ri * ss * (ri * r + si * s - 1.0) + rr * ss * ri))
        ds_list.append(0.25 * (rr * si * (ri * r + si * s - 1.0) + rr * ss * si))

    # Midedge 4: (0,-1) — s_i=-1, r_i=0 => midedge at s_i ≠ 0, r_i = 0
    vals_list.append(0.5 * (1.0 - r**2) * (1.0 - s))
    dr_list.append(0.5 * (-2.0 * r) * (1.0 - s))
    ds_list.append(0.5 * (1.0 - r**2) * (-1.0))

    # Midedge 5: (1,0) — r_i=1, s_i=0 => midedge at r_i ≠ 0, s_i = 0
    vals_list.append(0.5 * (1.0 + r) * (1.0 - s**2))
    dr_list.append(0.5 * (1.0 - s**2))
    ds_list.append(0.5 * (1.0 + r) * (-2.0 * s))

    # Midedge 6: (0,1) — s_i=1, r_i=0
    vals_list.append(0.5 * (1.0 - r**2) * (1.0 + s))
    dr_list.append(0.5 * (-2.0 * r) * (1.0 + s))
    ds_list.append(0.5 * (1.0 - r**2) * 1.0)

    # Midedge 7: (-1,0) — r_i=-1, s_i=0
    vals_list.append(0.5 * (1.0 - r) * (1.0 - s**2))
    dr_list.append(0.5 * (-1.0) * (1.0 - s**2))
    ds_list.append(0.5 * (1.0 - r) * (-2.0 * s))

    vals = np.stack(vals_list, axis=1)
    grads = np.stack((np.stack(dr_list, axis=1), np.stack(ds_list, axis=1)), axis=-1)
    return vals, grads
