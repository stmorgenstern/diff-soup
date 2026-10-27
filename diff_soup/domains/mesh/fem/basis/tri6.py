"""TRI6 (quadratic triangle) shape functions on the reference triangle."""

from __future__ import annotations

import numpy as np


def _tri6_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRI6 shape functions on the reference triangle (0,0)-(1,0)-(0,1).

    Nodes: (0,0),(1,0),(0,1),(0.5,0),(0.5,0.5),(0,0.5)
    pts : (n_pts, 2)
    Returns vals (n_pts, 6) and grads (n_pts, 6, 2).
    """
    r, s = pts[:, 0], pts[:, 1]
    lam = 1.0 - r - s  # barycentric coord λ

    n0 = lam * (2.0 * lam - 1.0)
    n1 = r * (2.0 * r - 1.0)
    n2 = s * (2.0 * s - 1.0)
    n3 = 4.0 * lam * r
    n4 = 4.0 * r * s
    n5 = 4.0 * s * lam
    vals = np.stack((n0, n1, n2, n3, n4, n5), axis=1)

    # dN/dr
    dn0_dr = -1.0 * (2.0 * lam - 1.0) + lam * (-2.0)  # = -4*lam + 1 = 4*r+4*s-3
    dn1_dr = 4.0 * r - 1.0
    dn2_dr = np.zeros_like(r)
    dn3_dr = 4.0 * (lam - r)  # 4*(1-r-s) - 4*r = 4 - 8r - 4s
    dn4_dr = 4.0 * s
    dn5_dr = -4.0 * s

    # dN/ds
    dn0_ds = -1.0 * (2.0 * lam - 1.0) + lam * (-2.0)  # same as dn0_dr
    dn1_ds = np.zeros_like(r)
    dn2_ds = 4.0 * s - 1.0
    dn3_ds = -4.0 * r
    dn4_ds = 4.0 * r
    dn5_ds = 4.0 * (lam - s)  # 4*(1-r-s) - 4*s = 4 - 4r - 8s

    grads = np.stack(
        (
            np.stack((dn0_dr, dn1_dr, dn2_dr, dn3_dr, dn4_dr, dn5_dr), axis=1),
            np.stack((dn0_ds, dn1_ds, dn2_ds, dn3_ds, dn4_ds, dn5_ds), axis=1),
        ),
        axis=-1,
    )
    return vals, grads
