"""HEX20 (serendipity quadratic hexahedron) shape functions on [-1, 1]^3."""

from __future__ import annotations

import numpy as np


def _hex20_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HEX20 serendipity shape functions on [-1,1]^3.

    Nodes (gmsh ordering):
      0-7: corners (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                    (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)
      8-19: midedges
        8:(0,-1,-1) 9:(1,0,-1) 10:(0,1,-1) 11:(-1,0,-1)
        12:(0,-1,1) 13:(1,0,1) 14:(0,1,1) 15:(-1,0,1)
        16:(-1,-1,0) 17:(1,-1,0) 18:(1,1,0) 19:(-1,1,0)
    pts : (n_pts, 3)
    Returns vals (n_pts, 20) and grads (n_pts, 20, 3).
    """
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]

    # Corner coordinates
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )

    # Midedge coordinates
    midedges = np.array(
        [
            [0, -1, -1],
            [1, 0, -1],
            [0, 1, -1],
            [-1, 0, -1],
            [0, -1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ],
        dtype=float,
    )

    vals_list = []
    dr_list = []
    ds_list = []
    dt_list = []

    # Corners
    for i in range(8):
        ri, si, ti = corners[i]
        rr = 1.0 + ri * r
        ss = 1.0 + si * s
        tt = 1.0 + ti * t
        sm = ri * r + si * s + ti * t - 2.0  # sum term
        v = 0.125 * rr * ss * tt * sm
        vals_list.append(v)
        dr_list.append(0.125 * (ri * ss * tt * sm + rr * ss * tt * ri))
        ds_list.append(0.125 * (rr * si * tt * sm + rr * ss * tt * si))
        dt_list.append(0.125 * (rr * ss * ti * sm + rr * ss * tt * ti))

    # Midedges
    for i in range(12):
        ri, si, ti = midedges[i]
        if ri == 0:
            # r_i=0: N = 0.25*(1-r^2)*(1+s_i*s)*(1+t_i*t)
            ss = 1.0 + si * s
            tt = 1.0 + ti * t
            v = 0.25 * (1.0 - r**2) * ss * tt
            vals_list.append(v)
            dr_list.append(0.25 * (-2.0 * r) * ss * tt)
            ds_list.append(0.25 * (1.0 - r**2) * si * tt)
            dt_list.append(0.25 * (1.0 - r**2) * ss * ti)
        elif si == 0:
            # s_i=0: N = 0.25*(1+r_i*r)*(1-s^2)*(1+t_i*t)
            rr = 1.0 + ri * r
            tt = 1.0 + ti * t
            v = 0.25 * rr * (1.0 - s**2) * tt
            vals_list.append(v)
            dr_list.append(0.25 * ri * (1.0 - s**2) * tt)
            ds_list.append(0.25 * rr * (-2.0 * s) * tt)
            dt_list.append(0.25 * rr * (1.0 - s**2) * ti)
        else:
            # t_i=0: N = 0.25*(1+r_i*r)*(1+s_i*s)*(1-t^2)
            rr = 1.0 + ri * r
            ss = 1.0 + si * s
            v = 0.25 * rr * ss * (1.0 - t**2)
            vals_list.append(v)
            dr_list.append(0.25 * ri * ss * (1.0 - t**2))
            ds_list.append(0.25 * rr * si * (1.0 - t**2))
            dt_list.append(0.25 * rr * ss * (-2.0 * t))

    vals = np.stack(vals_list, axis=1)
    grads = np.stack(
        (
            np.stack(dr_list, axis=1),
            np.stack(ds_list, axis=1),
            np.stack(dt_list, axis=1),
        ),
        axis=-1,
    )
    return vals, grads
