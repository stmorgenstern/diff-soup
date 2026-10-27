"""The error-weight law and the weighted-RMS norm for the solver suite.

Every engine weights errors with the tolerance-ratio law
``ewt = 1/(rtol*|y| + atol)`` and compares normalized error norms to 1,
refreshing the weights on accepted steps. ``wrms_norm`` consumes the weight
vector; an optional differential/algebraic mask suppresses algebraic rows in
error norms (never in Newton norms).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from diff_soup.utils import safe_sqrt


def error_weight(y: jax.Array, rtol, atol) -> jax.Array:
    """Tolerance-ratio error-weight vector ``1/(rtol*|y| + atol)`` (JBDF / JIDA).

    The weights are error-control data (they feed norms whose accept/step
    decisions are gradient-dead), so the state they are built from is cut out
    of differentiation here at the source: at tight tolerances the weight's
    state-Jacobian scales like ``rtol * ewt**2`` (enormous), and any cotangent
    reaching it through a norm would be amplified past overflow.
    """
    y = jax.lax.stop_gradient(y)
    tiny = jnp.finfo(y.dtype).tiny
    denom = jnp.maximum(
        jnp.asarray(rtol, dtype=y.dtype) * jnp.abs(y) + jnp.asarray(atol, dtype=y.dtype),
        tiny,
    )
    return 1.0 / denom


def wrms_norm(x: jax.Array, w: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
    """Weighted RMS norm ``sqrt(mean((x*w)^2))``.

    ``w`` is a weight vector produced by
    [error_weight][diff_soup.timestepping.vsvo.common.weights.error_weight]. ``mask`` (the id vector: 1 differential, 0 algebraic)
    removes masked rows from the sum while the divisor stays the full vector
    length -- the standard masked weighted-RMS convention -- so algebraic rows can be
    suppressed from error norms. The root uses ``safe_sqrt`` so the gradient at
    an exact-zero input is finite.
    """
    prod = x * w
    sq = prod * prod
    if mask is None:
        return safe_sqrt(jnp.mean(sq))
    m = mask.astype(prod.dtype)
    # Masked-norm convention: masked rows leave the sum but the
    # divisor stays the FULL vector length.
    return safe_sqrt(jnp.sum(sq * m) / x.shape[0])
