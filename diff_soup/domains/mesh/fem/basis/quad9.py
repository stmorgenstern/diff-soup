"""QUAD9 (biquadratic Lagrange quadrilateral) shape functions on [-1, 1]^2."""

from __future__ import annotations

import numpy as np


def _quad9_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QUAD9 shape functions on [-1,1]^2, the face of a HEX27.

    Nodes: corners (-1,-1),(1,-1),(1,1),(-1,1)
           midedges (0,-1),(1,0),(0,1),(-1,0)
           centre (0,0)
    pts : (n_pts, 2)
    Returns vals (n_pts, 9) and grads (n_pts, 9, 2).
    """
    r, s = pts[:, 0], pts[:, 1]

    def _lag(x):
        """1D quadratic Lagrange at -1, 0, 1 and its derivative."""
        return (
            np.stack((x * (x - 1.0) / 2.0, 1.0 - x**2, x * (x + 1.0) / 2.0)),
            np.stack(((2.0 * x - 1.0) / 2.0, -2.0 * x, (2.0 * x + 1.0) / 2.0)),
        )

    lr, dlr = _lag(r)
    ls, dls = _lag(s)
    # (index in r, index in s) of each node, -1 -> 0, 0 -> 1, 1 -> 2
    ir = (0, 2, 2, 0, 1, 2, 1, 0, 1)
    js = (0, 0, 2, 2, 0, 1, 2, 1, 1)
    vals = np.stack([lr[i] * ls[j] for i, j in zip(ir, js)], axis=1)
    dr = np.stack([dlr[i] * ls[j] for i, j in zip(ir, js)], axis=1)
    ds = np.stack([lr[i] * dls[j] for i, j in zip(ir, js)], axis=1)
    return vals, np.stack((dr, ds), axis=-1)
