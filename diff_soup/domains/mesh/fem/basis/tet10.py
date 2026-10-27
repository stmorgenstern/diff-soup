"""TET10 (quadratic tetrahedron) shape functions on the reference tetrahedron."""

from __future__ import annotations

import numpy as np


def _tet10_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TET10 quadratic shape functions on the reference tetrahedron.

    Nodes (gmsh/meshio ordering):
      0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(0,0,1)         [corners]
      4:mid(0,1)=(0.5,0,0)    5:mid(1,2)=(0.5,0.5,0)   [mid-edges]
      6:mid(0,2)=(0,0.5,0)    7:mid(0,3)=(0,0,0.5)
      8:mid(2,3)=(0,0.5,0.5)  9:mid(1,3)=(0.5,0,0.5)

    λ = 1-r-s-t (barycentric coordinate for vertex 0).
    pts : (n_pts, 3)
    Returns vals (n_pts, 10) and grads (n_pts, 10, 3).
    """
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]
    lam = 1.0 - r - s - t

    # Corner shape functions (same as TET4 but quadratic)
    n0 = lam * (2.0 * lam - 1.0)
    n1 = r * (2.0 * r - 1.0)
    n2 = s * (2.0 * s - 1.0)
    n3 = t * (2.0 * t - 1.0)
    # Mid-edge shape functions in gmsh order
    n4 = 4.0 * lam * r  # mid(0,1)
    n5 = 4.0 * r * s  # mid(1,2)
    n6 = 4.0 * lam * s  # mid(0,2)
    n7 = 4.0 * lam * t  # mid(0,3)
    n8 = 4.0 * s * t  # mid(2,3)
    n9 = 4.0 * r * t  # mid(1,3)
    vals = np.stack((n0, n1, n2, n3, n4, n5, n6, n7, n8, n9), axis=1)

    dl = -(4.0 * lam - 1.0)  # d(corner0)/dr = d(corner0)/ds = d(corner0)/dt

    # dN/dr  (dlam/dr = -1)
    dn_dr = np.stack(
        (
            dl,  # n0
            4.0 * r - 1.0,  # n1
            np.zeros_like(r),  # n2
            np.zeros_like(r),  # n3
            4.0 * (lam - r),  # n4 = 4*lam*r
            4.0 * s,  # n5 = 4*r*s
            -4.0 * s,  # n6 = 4*lam*s
            -4.0 * t,  # n7 = 4*lam*t
            np.zeros_like(r),  # n8 = 4*s*t
            4.0 * t,  # n9 = 4*r*t
        ),
        axis=1,
    )

    # dN/ds  (dlam/ds = -1)
    dn_ds = np.stack(
        (
            dl,  # n0
            np.zeros_like(r),  # n1
            4.0 * s - 1.0,  # n2
            np.zeros_like(r),  # n3
            -4.0 * r,  # n4 = 4*lam*r
            4.0 * r,  # n5 = 4*r*s
            4.0 * (lam - s),  # n6 = 4*lam*s
            -4.0 * t,  # n7 = 4*lam*t
            4.0 * t,  # n8 = 4*s*t
            np.zeros_like(r),  # n9 = 4*r*t
        ),
        axis=1,
    )

    # dN/dt  (dlam/dt = -1)
    dn_dt = np.stack(
        (
            dl,  # n0
            np.zeros_like(r),  # n1
            np.zeros_like(r),  # n2
            4.0 * t - 1.0,  # n3
            -4.0 * r,  # n4 = 4*lam*r
            np.zeros_like(r),  # n5 = 4*r*s
            -4.0 * s,  # n6 = 4*lam*s
            4.0 * (lam - t),  # n7 = 4*lam*t
            4.0 * s,  # n8 = 4*s*t
            4.0 * r,  # n9 = 4*r*t
        ),
        axis=1,
    )

    grads = np.stack((dn_dr, dn_ds, dn_dt), axis=-1)
    return vals, grads
