"""HEX27 (triquadratic Lagrange hexahedron) shape functions on [-1, 1]^3."""

from __future__ import annotations

import numpy as np


def _hex27_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HEX27 shape functions (tensor product of 1D quadratic Lagrange) on [-1,1]^3.

    Nodes (gmsh ordering):
      0-7: corners, 8-19: midedges (same as HEX20),
      20-25: face centers, 26: body center.

    The node ordering follows gmsh convention:
      Corners:   0(-1,-1,-1) 1(1,-1,-1) 2(1,1,-1) 3(-1,1,-1)
                 4(-1,-1,1) 5(1,-1,1) 6(1,1,1) 7(-1,1,1)
      Midedges:  8(0,-1,-1) 9(1,0,-1) 10(0,1,-1) 11(-1,0,-1)
                 12(0,-1,1) 13(1,0,1) 14(0,1,1) 15(-1,0,1)
                 16(-1,-1,0) 17(1,-1,0) 18(1,1,0) 19(-1,1,0)
      Faces:     20(0,0,-1) 21(0,0,1) 22(0,-1,0) 23(1,0,0)
                 24(0,1,0) 25(-1,0,0)
      Center:    26(0,0,0)

    pts : (n_pts, 3)
    Returns vals (n_pts, 27) and grads (n_pts, 27, 3).
    """
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]

    # 1D quadratic Lagrange at nodes -1, 0, +1:
    # L0(x) = x(x-1)/2,  L1(x) = 1-x^2,  L2(x) = x(x+1)/2
    # dL0/dx = (2x-1)/2,  dL1/dx = -2x,  dL2/dx = (2x+1)/2

    def _lag(x):
        """Returns (L0, L1, L2) and (dL0, dL1, dL2) for 1D."""
        l0 = x * (x - 1.0) / 2.0
        l1 = 1.0 - x**2
        l2 = x * (x + 1.0) / 2.0
        dl0 = (2.0 * x - 1.0) / 2.0
        dl1 = -2.0 * x
        dl2 = (2.0 * x + 1.0) / 2.0
        return (l0, l1, l2), (dl0, dl1, dl2)

    Lr, dLr = _lag(r)
    Ls, dLs = _lag(s)
    Lt, dLt = _lag(t)

    # gmsh node ordering -> (i_r, i_s, i_t) index into Lagrange basis {-1:0, 0:1, 1:2}
    # Map: coord -> index: -1->0, 0->1, 1->2
    node_coords = [
        # Corners 0-7
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
        # Midedges 8-19
        (0, -1, -1),
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, 1),
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (-1, -1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
        # Face centers 20-25
        (0, 0, -1),
        (0, 0, 1),
        (0, -1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (-1, 0, 0),
        # Body center 26
        (0, 0, 0),
    ]

    coord_to_idx = {-1: 0, 0: 1, 1: 2}

    vals_list = []
    dr_list = []
    ds_list = []
    dt_list = []

    for rc, sc, tc in node_coords:
        ir = coord_to_idx[rc]
        js = coord_to_idx[sc]
        kt = coord_to_idx[tc]
        vals_list.append(Lr[ir] * Ls[js] * Lt[kt])
        dr_list.append(dLr[ir] * Ls[js] * Lt[kt])
        ds_list.append(Lr[ir] * dLs[js] * Lt[kt])
        dt_list.append(Lr[ir] * Ls[js] * dLt[kt])

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
