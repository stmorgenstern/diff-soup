"""TRI3 (linear triangle) shape functions on the reference triangle."""

from __future__ import annotations

import numpy as np


def _tri3_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRI3 shape functions on the reference triangle ``(0,0)-(1,0)-(0,1)``.

    Args:
        pts: Reference coordinates ``(r, s)`` with shape ``(n_pts, 2)``.

    Returns:
        ``(vals, grads)`` where ``vals`` has shape ``(n_pts, 3)`` and
        ``grads`` has shape ``(n_pts, 3, 2)`` (constant for linear elements).
    """
    r, s = pts[:, 0], pts[:, 1]
    vals = np.stack((1.0 - r - s, r, s), axis=1)
    ones = np.ones_like(r)
    dn_dr = np.stack((-ones, ones, np.zeros_like(r)), axis=1)
    dn_ds = np.stack((-ones, np.zeros_like(r), ones), axis=1)
    grads = np.stack((dn_dr, dn_ds), axis=-1)  # (n_pts, 3, 2)
    return vals, grads
