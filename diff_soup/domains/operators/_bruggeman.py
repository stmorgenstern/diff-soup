"""Bruggeman tortuosity correction behind ``Operators.tortuosity_corrected``."""

from __future__ import annotations

import jax.numpy as jnp


def tortuosity_corrected(D, eps, beta: float = 1.5):
    """Bruggeman effective diffusivity: ``D_eff = eps^beta * D``."""
    return jnp.asarray(eps) ** beta * jnp.asarray(D)


__all__ = ["tortuosity_corrected"]
