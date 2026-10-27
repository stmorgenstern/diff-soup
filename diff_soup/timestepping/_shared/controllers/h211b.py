"""Söderlind 2003 H211b predicted-step recurrence.

The recurrence returns predicted log step sizes per order. It does not
return a filtered error vector.

Reference:
  Söderlind, "Digital filters in adaptive time-stepping," ACM TOMS 29(1)
  (2003), §6.2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def h211b_log_h_hat(
    err_curr: jax.Array,
    err_prev: jax.Array,
    *,
    h: jax.Array,
    h_prev: jax.Array,
    tol: jax.Array,
    beta_scale: float,
    gamma: float,
    first_step: bool | jax.Array,
    err_floor: float = 1e-300,
    saturation_log_ratio: float = 2.0,
) -> jax.Array:
    """Return H211b-predicted log h for orders 1..k_max.

    Computes::

        r_n   = min(log(tol) - log(max(err_curr, err_floor)), s)
        r_nm1 = min(log(tol) - log(max(err_prev, err_floor)), s)
        log_h_hat_k = log(h) + β_k*r_n + β_k*r_nm1 + γ*(log(h)-log(h_prev))

    where β_k = beta_scale / (k + 1), k is the 1-indexed method order, and
    s = ``saturation_log_ratio``. If ``first_step`` is true, use
    ``err_prev = err_curr`` and ``h_prev = h``.

    ``first_step`` may be either a scalar bool (whole-step transient) or a
    bool array of shape ``(k_max,)`` (per-order transient — marks
    newly-eligible orders whose ``err_log_prev`` is stale).
    Broadcasting handles both cases automatically.

    Anti-windup: ``saturation_log_ratio`` caps each ``log(tol) - log(err)``
    informant so the integral term cannot exceed ``beta * s`` per step.
    Without this, deep under-tol regimes (``err/tol < 1e-3``) produce
    arbitrarily large ``log_h_hat`` that the per-step ``growth_max`` clip
    masks but cannot fundamentally bound — the resulting stair-step
    reject pattern on stiff systems is the classic Söderlind 2003 §6.3
    integral-controller windup. The default ``s=2.0`` permits up to
    ~e^2≈7.4× informant per step (still well within the practical range
    for smooth phases) while killing the runaway when ``err`` collapses
    far below ``tol``. Set ``s=jnp.inf`` to disable (recovers the bare
    H211b for ablation).
    """
    err_prev_eff = jnp.where(first_step, err_curr, err_prev)
    h_prev_eff = jnp.where(first_step, h, h_prev)
    dtype = err_curr.dtype
    orders = jnp.arange(1, err_curr.shape[0] + 1, dtype=dtype)
    beta = beta_scale / (orders + 1.0)
    log_tol = jnp.log(tol.astype(dtype))
    log_err = jnp.log(jnp.maximum(err_curr, jnp.asarray(err_floor, dtype=dtype)))
    log_err_prev = jnp.log(jnp.maximum(err_prev_eff, jnp.asarray(err_floor, dtype=dtype)))
    sat = jnp.asarray(saturation_log_ratio, dtype=dtype)
    informant = jnp.minimum(log_tol - log_err, sat)
    informant_prev = jnp.minimum(log_tol - log_err_prev, sat)
    log_h = jnp.log(h.astype(dtype))
    log_h_prev = jnp.log(h_prev_eff.astype(dtype))
    return log_h + beta * informant + beta * informant_prev + gamma * (log_h - log_h_prev)
