"""HEX8 (trilinear hexahedron) shape functions on the reference cube."""

from __future__ import annotations

import numpy as np


def _hex8_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate HEX8 shape values and reference gradients at reference points."""
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]
    signs = np.array(
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
    vals = []
    grads = []
    for sr, ss, st in signs:
        v = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * (1.0 + st * t)
        dr = 0.125 * sr * (1.0 + ss * s) * (1.0 + st * t)
        ds = 0.125 * ss * (1.0 + sr * r) * (1.0 + st * t)
        dt = 0.125 * st * (1.0 + sr * r) * (1.0 + ss * s)
        vals.append(v)
        grads.append(np.stack((dr, ds, dt), axis=-1))
    return np.stack(vals, axis=1), np.stack(grads, axis=1)
