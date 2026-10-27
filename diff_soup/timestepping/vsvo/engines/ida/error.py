"""The local-error test for the fully-implicit DAE engine.

[ida_test_error][diff_soup.timestepping.vsvo.engines.ida.error.ida_test_error]
forms the raw and sigma-scaled local-error estimates
at the current order and the two orders below it, decides whether the step
is accepted, and proposes the order the next attempt should use, given the
post-corrector history and the converged Newton correction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup.timestepping.vsvo.common.weights import wrms_norm

from .state import MXORDP1

_HALF = 0.5
"""The order-lower threshold factor applied only at order 2."""


def ida_test_error(
    ck: jax.Array,
    ee: jax.Array,
    phi: jax.Array,
    sigma: jax.Array,
    kk: jax.Array,
    ewt: jax.Array,
    id_mask: jax.Array,
    suppressalg: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the local-error accept/reject test and propose the next order.

    Forms the raw weighted-RMS norm of the Newton correction ``ee`` and the
    sigma-scaled error estimates at the current order ``kk`` and, when the
    order allows, at ``kk - 1`` and ``kk - 2``. The accept/reject verdict
    compares ``ck`` against the *raw* norm at order ``kk`` -- the
    sigma-scaled estimate only feeds the order-decrease decision, never the
    accept test itself. When ``kk > 2`` the order drops whenever the larger
    of the two lower-order estimates does not exceed the current-order
    estimate; at ``kk == 2`` the same comparison instead uses a halved
    current-order estimate; at ``kk == 1`` there is no order below to try,
    so the order never drops.

    All three per-order deltas are formed unconditionally over the fixed
    ``(MXORDP1, n)`` history so ``kk`` stays a traced integer with no
    data-dependent shape; dynamic indices are clamped into range and the
    inactive branches are masked out of both the returned
    ``err_km1``/``err_km2`` (zeroed) and the order decision.

    Args:
        ck: The step-size error constant from the coefficient recurrence.
        ee: The converged Newton correction, shape ``(n,)``.
        phi: The post-corrector divided-difference history (``phi*`` form),
            shape ``(MXORDP1, n)``.
        sigma: The order-dependent error-estimate scale factors, shape
            ``(MXORDP1,)``.
        kk: The integration order this attempt used (``int32``).
        ewt: The error-weight vector, shape ``(n,)``.
        id_mask: ``1`` for differential rows, ``0`` for algebraic rows,
            shape ``(n,)``.
        suppressalg: When set, algebraic rows are excluded from every error
            norm this test forms (never from Newton convergence norms).

    Returns:
        A 6-tuple ``(error_test_fail, knew, err_k, err_km1, err_km2,
        enorm_k)``: the reject verdict, the proposed next order, the three
        sigma-scaled per-order error estimates (the inactive ones zeroed),
        and the raw weighted-RMS norm the accept test used. Every entry is a
        step-control quantity and carries no gradient.
    """
    kk = jnp.asarray(kk)
    dtype = ee.dtype
    mask = id_mask if suppressalg else None

    kk_idx = jnp.clip(kk, 0, MXORDP1 - 1)
    km1_idx = jnp.clip(kk - 1, 0, MXORDP1 - 1)
    km2_idx = jnp.clip(kk - 2, 0, MXORDP1 - 1)

    enorm_k = wrms_norm(ee, ewt, mask=mask)
    err_k = sigma[kk_idx] * enorm_k
    terr_k = (kk + 1).astype(dtype) * err_k

    delta_km1 = phi[kk_idx] + ee
    enorm_km1 = wrms_norm(delta_km1, ewt, mask=mask)
    err_km1_raw = sigma[km1_idx] * enorm_km1
    terr_km1 = kk.astype(dtype) * err_km1_raw

    delta_km2 = phi[km1_idx] + delta_km1
    enorm_km2 = wrms_norm(delta_km2, ewt, mask=mask)
    err_km2_raw = sigma[km2_idx] * enorm_km2
    terr_km2 = (kk - 1).astype(dtype) * err_km2_raw

    kk_gt1 = kk > 1
    kk_gt2 = kk > 2

    knew_gt2 = jnp.where(jnp.maximum(terr_km1, terr_km2) <= terr_k, kk - 1, kk)
    knew_eq2 = jnp.where(terr_km1 <= _HALF * terr_k, kk - 1, kk)
    knew = jnp.where(kk_gt1, jnp.where(kk_gt2, knew_gt2, knew_eq2), kk).astype(jnp.int32)

    zero = jnp.zeros((), dtype=dtype)
    err_km1 = jnp.where(kk_gt1, err_km1_raw, zero)
    err_km2 = jnp.where(kk_gt2, err_km2_raw, zero)

    error_test_fail = ck * enorm_k > 1.0

    return jax.lax.stop_gradient((error_test_fail, knew, err_k, err_km1, err_km2, enorm_k))


__all__ = ["ida_test_error"]
