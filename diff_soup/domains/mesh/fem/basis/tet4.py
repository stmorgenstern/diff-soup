"""TET4 (linear tetrahedron) shape functions on the reference tetrahedron."""

from __future__ import annotations

import numpy as np


def _tet4_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TET4 shape functions on the reference tetrahedron.

    Nodes: (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    pts : (n_pts, 3)
    Returns vals (n_pts, 4) and grads (n_pts, 4, 3).
    """
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]
    vals = np.stack((1.0 - r - s - t, r, s, t), axis=1)
    ones = np.ones_like(r)
    zeros = np.zeros_like(r)
    grads = np.stack(
        (
            np.stack((-ones, ones, zeros, zeros), axis=1),
            np.stack((-ones, zeros, ones, zeros), axis=1),
            np.stack((-ones, zeros, zeros, ones), axis=1),
        ),
        axis=-1,
    )
    return vals, grads
