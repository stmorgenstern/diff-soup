"""Dense output for the NDF engine.

The interpolant is the classical Newton backward-difference polynomial
through the accepted step's uniform grid: with ``s = (t - t_new)/dt``,

    P(t_new + s*dt) = D[0] + sum_{j=1..k} c_j(s) * D[j],
    c_j(s) = prod_{i=1..j} (s + i - 1) / i,

a textbook form (see e.g. Hairer, Nørsett & Wanner, *Solving Ordinary
Differential Equations I*, Ch. III.1) evaluated here as a cumulative
product, the same expression SciPy's BDF dense output uses (BSD-3; see
``THIRD_PARTY_NOTICES.md``). The time derivative is obtained by
forward-mode automatic differentiation of the scalar-``s`` closure — no
hand-written derivative recurrence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .tables import MAX_ORDER


def ndf_interpolant(d: jax.Array, k: jax.Array, t_new: jax.Array, dt: jax.Array, t: jax.Array):
    """Evaluate ``(y(t), ydot(t))`` on the last accepted step's interval.

    Args:
        d: The (post-fold) difference table at the accepted point, shape
            ``(D_ROWS, n)``; rows ``0..k`` are read.
        k: The accepted step's order (may be traced).
        t_new: The accepted point's time.
        dt: The accepted step size (the table's grid spacing).
        t: The query time.

    Returns:
        The interpolated state and its time derivative at ``t``.
    """
    dtype = d.dtype
    s0 = ((t - t_new) / dt).astype(dtype)
    idx = jnp.arange(1, MAX_ORDER + 1, dtype=dtype)
    active = jnp.arange(1, MAX_ORDER + 1) <= k

    def y_of_s(s):
        c = jnp.cumprod((s + idx - 1.0) / idx)
        weights = jnp.where(active, c, 0.0)
        return d[0] + weights @ d[1 : MAX_ORDER + 1]

    y, dy_ds = jax.jvp(y_of_s, (s0,), (jnp.ones_like(s0),))
    return y, dy_ds / dt


__all__ = ["ndf_interpolant"]
