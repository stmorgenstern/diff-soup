"""LINE2 (linear line) shape functions on the reference interval."""

from __future__ import annotations

import numpy as np


def _line2_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LINE2 shape functions on the reference interval ``[-1, 1]``.

    The two nodal hats are ``N_0(r) = (1 - r) / 2`` and ``N_1(r) = (1 + r) / 2``,
    so node 0 sits at ``r = -1`` and node 1 at ``r = +1``.

    Args:
        pts: Reference coordinates ``(r,)`` with shape ``(n_pts, 1)``.

    Returns:
        ``(vals, grads)`` where ``vals`` has shape ``(n_pts, 2)`` and
        ``grads`` has shape ``(n_pts, 2, 1)`` (constant for linear elements).
    """
    r = pts[:, 0]
    vals = np.stack((0.5 * (1.0 - r), 0.5 * (1.0 + r)), axis=1)
    half = np.full_like(r, 0.5)
    grads = np.stack((-half, half), axis=1)[:, :, None]  # (n_pts, 2, 1)
    return vals, grads
