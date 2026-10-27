"""QUAD4 (bilinear quadrilateral) shape functions on the reference square."""

from __future__ import annotations

import numpy as np


def _quad4_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate QUAD4 shape values and reference gradients at reference points."""
    r = pts[:, 0]
    s = pts[:, 1]
    n1 = 0.25 * (1.0 - r) * (1.0 - s)
    n2 = 0.25 * (1.0 + r) * (1.0 - s)
    n3 = 0.25 * (1.0 + r) * (1.0 + s)
    n4 = 0.25 * (1.0 - r) * (1.0 + s)
    vals = np.stack((n1, n2, n3, n4), axis=1)

    dn_dr = np.stack(
        (
            -0.25 * (1.0 - s),
            0.25 * (1.0 - s),
            0.25 * (1.0 + s),
            -0.25 * (1.0 + s),
        ),
        axis=1,
    )
    dn_ds = np.stack(
        (
            -0.25 * (1.0 - r),
            -0.25 * (1.0 + r),
            0.25 * (1.0 + r),
            0.25 * (1.0 - r),
        ),
        axis=1,
    )
    grads = np.stack((dn_dr, dn_ds), axis=-1)
    return vals, grads
