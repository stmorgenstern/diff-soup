"""chebyshev.py — Chebyshev polynomial function approximation.

Provides [ChebyshevExpansion][diff_soup.inference.approx.chebyshev.ChebyshevExpansion], a generic building block for
Chebyshev polynomial expansions.  Same usage pattern as
[LegendreExpansion][diff_soup.inference.approx.legendre.LegendreExpansion].
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ChebyshevExpansion(eqx.Module):
    """Chebyshev polynomial expansion evaluated at x ∈ [−1, 1].

    Uses the three-term recurrence: T_{k+1}(x) = 2x T_k(x) − T_{k−1}(x).

    Attributes:
        params: Coefficients (a_0, a_1, …, a_n), shape ``(degree+1,)``.
            These are JAX array leaves and are the learnable parameters.
    """

    params: jax.Array  # (degree+1,)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate Σ_k params[k] * T_k(x) for x ∈ [−1, 1]."""
        n = len(self.params)
        result = self.params[0] * jnp.ones_like(x)
        if n >= 2:
            result = result + self.params[1] * x
        t_prev = jnp.ones_like(x)
        t_curr = x
        for k in range(2, n):
            t_next = 2.0 * x * t_curr - t_prev
            result = result + self.params[k] * t_next
            t_prev, t_curr = t_curr, t_next
        return result
