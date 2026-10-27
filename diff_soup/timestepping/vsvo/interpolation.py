"""Dense output over the accepted trajectory for the scalar solver suite.

A solve with ``SaveAt(dense=True)`` stores one ``(time, state, derivative)``
node per accepted step and serves arbitrary in-horizon query times through
[CubicHermiteInterpolation][] — a piecewise cubic matching the state
and its derivative at both ends of every accepted step. Between nodes the
interpolant is third-order accurate; the solver's own local error control
governs the node values themselves. All local and bordered values are
interpolated. Algebraic-row derivatives are internal only and are stored as
zero; any future public derivative surface must preserve that convention.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["CubicHermiteInterpolation"]


class CubicHermiteInterpolation(eqx.Module):
    """Piecewise cubic Hermite dense output over the accepted step nodes.

    Attributes:
        ts: Node times, shape ``(capacity,)``; slots at or beyond
            [count][] hold ``+inf`` so segment lookup ignores them.
        ys: Node states, shape ``(capacity, n)``.
        ydots: Node state derivatives, shape ``(capacity, n)``.
        count: Number of valid nodes (``int32``); the covered span is
            ``[ts[0], ts[count - 1]]``.
    """

    ts: jax.Array
    ys: jax.Array
    ydots: jax.Array
    count: jax.Array

    def evaluate(self, t):
        """Evaluate the interpolant at ``t`` (scalar or 1-D array of times).

        Query times are clamped to the covered span; a scalar query returns
        shape ``(n,)`` and a vector query ``(len(t), n)``.
        """
        t_arr = jnp.asarray(t, dtype=self.ts.dtype)

        def _one(tq):
            j = jnp.clip(
                jnp.searchsorted(self.ts, tq, side="right"),
                1,
                jnp.maximum(self.count - 1, 1),
            )
            i = j - 1
            t_lo, t_hi = self.ts[i], self.ts[j]
            h = t_hi - t_lo
            safe_h = jnp.where(h > 0, h, jnp.ones_like(h))
            s = jnp.clip(jnp.where(h > 0, (tq - t_lo) / safe_h, 0.0), 0.0, 1.0)
            h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
            h10 = s * (1.0 - s) ** 2
            h01 = s**2 * (3.0 - 2.0 * s)
            h11 = s**2 * (s - 1.0)
            return (
                h00 * self.ys[i]
                + h10 * safe_h * self.ydots[i]
                + h01 * self.ys[j]
                + h11 * safe_h * self.ydots[j]
            )

        if t_arr.ndim == 0:
            return _one(t_arr)
        return jax.vmap(_one)(t_arr)
