"""Host boundary for cheap, JAX-compatible optimizer trial validation."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def trial_is_valid(problem: Any, point: Any) -> bool:
    """Evaluate an optional scalar ``trial_is_valid`` contract on device."""
    checker = getattr(problem, "trial_is_valid", None)
    if checker is None:
        return True
    value = np.asarray(jax.device_get(checker(point)))
    if value.shape != ():
        raise ValueError(
            "problem.trial_is_valid(point) must return one scalar boolean; "
            f"got shape {value.shape}."
        )
    return bool(value)


__all__ = ["trial_is_valid"]
