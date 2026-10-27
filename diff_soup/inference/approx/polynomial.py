"""polynomial.py — Monomial polynomial function approximation.

Provides [MonomialExpansion][], the core generic building block for
monomial polynomial expansions.  Users compose their own physics-specific
functions from this base.

Convention: concentration inputs c ∈ [0, 1] are used directly (no mapping).
"""

from __future__ import annotations

# (typing imports removed: were only used by the deleted physics-specific wrappers)
import equinox as eqx
import jax
import jax.numpy as jnp


class MonomialExpansion(eqx.Module):
    """Monomial polynomial expansion f(c) = Σ_k params[k] * c^k.

    Attributes:
        params: Coefficients (a_0, a_1, …, a_n), shape ``(degree+1,)``.
            These are JAX array leaves and are the learnable parameters.
    """

    params: jax.Array  # (degree+1,)

    def __call__(self, c: jax.Array) -> jax.Array:
        """Evaluate Σ_k params[k] * c^k using Horner's method."""
        n = len(self.params)
        if n == 0:
            return jnp.zeros_like(c)
        # Horner: a_n*c^n + ... + a_0 = (...((a_n*c + a_{n-1})*c + a_{n-2})*c + ...) + a_0
        result = self.params[n - 1] * jnp.ones_like(c)
        for k in range(n - 2, -1, -1):
            result = result * c + self.params[k]
        return result
