"""The predictor for the fully-implicit DAE engine.

[ida_predict][] forms the predicted state ``yypredict`` and predicted
derivative ``yppredict`` at the step endpoint from the divided-difference
history ``phi`` and the predictor-derivative coefficients ``gamma``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import MXORDP1


def ida_predict(phi: jax.Array, gamma: jax.Array, kk: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the predicted ``(yypredict, yppredict)`` at the step endpoint.

    ``yypredict`` is the plain sum of the history rows through the current
    order, ``sum_{j=0..kk} phi[j]``; ``yppredict`` is the ``gamma``-weighted
    sum of the higher-order rows, ``sum_{j=1..kk} gamma[j] * phi[j]``.

    Both sums run over the fixed leading axis with a compile-time mask, so the
    order ``kk`` is a traced integer while ``phi`` keeps its static
    ``(MXORDP1, n)`` shape. The coefficient masks are control quantities and
    carry no gradient; ``phi`` is *not* stopped, so both predictors keep the
    history's tangent (each is a linear combination of the ``phi`` rows) and
    the accepted trajectory's gradient flows through them.

    Args:
        phi: The divided-difference history, shape ``(MXORDP1, n)``.
        gamma: The predictor-derivative coefficients, shape ``(MXORDP1,)``.
        kk: The integration order for this step attempt.

    Returns:
        The predicted state and derivative, each shape ``(n,)``.
    """
    j = jnp.arange(MXORDP1)
    kk = jnp.asarray(kk)
    zero = jnp.zeros((), dtype=phi.dtype)
    yy_mask = (j <= kk)[:, None]
    yp_mask = ((j >= 1) & (j <= kk))[:, None]
    yypredict = jnp.sum(jnp.where(yy_mask, phi, zero), axis=0)
    yppredict = jnp.sum(jnp.where(yp_mask, gamma[:, None] * phi, zero), axis=0)
    return yypredict, yppredict


__all__ = ["ida_predict"]
