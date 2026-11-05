"""The corrector coefficient recurrence for the fully-implicit DAE engine.

[ida_set_coeffs][] advances the ``psi``, ``alpha``, ``beta``, ``sigma``,
and ``gamma`` coefficient arrays given the carried step-size and
divided-difference history, derives the leading coefficient ``cj`` and the
error constant ``ck``, and rescales the divided-difference history ``phi``
into the corrector-ready ``phi*`` that the predictor and the
accumulated-correction update consume.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup.utils.lanes import lane_cond

from .state import MAXORD, MXORDP1


def ida_set_coeffs(
    phi: jax.Array,
    psi: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    sigma: jax.Array,
    gamma: jax.Array,
    cj: jax.Array,
    kk: jax.Array,
    kused: jax.Array,
    hh: jax.Array,
    hused: jax.Array,
    ns: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Advance the corrector coefficients by one step attempt.

    The ``ns`` bookkeeping (the count of consecutive steps at constant step
    size and order, capped at ``kused + 2``) gates whether the
    ``psi``/``alpha``/``beta``/``sigma``/``gamma`` recurrences are
    recomputed this attempt: when the recent history is not current within
    ``kk + 1`` steps, the recurrences -- and therefore the ``phi`` ->
    ``phi*`` rescale -- are a no-op over the active order range. ``alphas``,
    ``alpha0``, the leading coefficient ``cj = -alphas / hh``, and the error
    constant ``ck`` are recomputed on every call regardless of that gate.

    These coefficients are step-control quantities and do not carry
    gradients: every argument except ``phi`` is stopped before use, so every
    returned array except ``phi_star`` is gradient-dead. ``phi_star`` is the
    one exception -- it keeps ``phi``'s tangent through the (gradient-dead)
    ``beta`` scale factors, because it feeds the predictor and therefore the
    accepted trajectory's gradient.

    Args:
        phi: The divided-difference history, shape ``(MXORDP1, n)``.
        psi: The step-size history, shape ``(MXORDP1,)``.
        alpha: Current-step-to-history step ratios, shape ``(MXORDP1,)``.
        beta: The ``phi`` -> ``phi*`` rescale factors, shape ``(MXORDP1,)``.
        sigma: Local-error scale factors, shape ``(MXORDP1,)``.
        gamma: Predictor-derivative coefficients, shape ``(MXORDP1,)``.
        cj: The leading coefficient from the previous step attempt.
        kk: The integration order for this attempt.
        kused: The order used on the last successful step.
        hh: The step size for this attempt.
        hused: The step size used on the last successful step.
        ns: The incoming count of consecutive constant-``(h, k)`` steps.

    Returns:
        A 12-tuple ``(phi_star, psi, alpha, beta, sigma, gamma, cjlast, cj,
        ck, ns, alphas, alpha0)``: the rescaled history and the updated
        coefficient arrays and scalars.
    """
    psi = jax.lax.stop_gradient(psi)
    alpha = jax.lax.stop_gradient(alpha)
    beta = jax.lax.stop_gradient(beta)
    sigma = jax.lax.stop_gradient(sigma)
    gamma = jax.lax.stop_gradient(gamma)
    cj = jax.lax.stop_gradient(cj)
    hh = jax.lax.stop_gradient(hh)
    hused = jax.lax.stop_gradient(hused)
    kk = jax.lax.stop_gradient(kk)
    kused = jax.lax.stop_gradient(kused)
    ns = jax.lax.stop_gradient(ns)

    dtype = psi.dtype
    one = jnp.asarray(1.0, dtype)
    zero = jnp.asarray(0.0, dtype)

    # ns bookkeeping: a change in step size or order restarts the count;
    # otherwise it grows by one, capped so it can never claim more history
    # than the last successful step's order supports.
    ns = jnp.where((hh != hused) | (kk != kused), jnp.zeros_like(ns), ns)
    ns = jnp.minimum(ns + 1, kused + 2)

    # The coefficient recurrences are only recomputed when the recent
    # history is still current within kk + 1 steps.
    recompute = kk + 1 >= ns

    # Each array is held as its scalar elements while the recurrence runs and
    # stacked at the end, so its updates compile to one fused computation.
    psi_i = [psi[i] for i in range(MXORDP1)]
    beta_i = [beta[i] for i in range(MXORDP1)]
    alpha_i = [alpha[i] for i in range(MXORDP1)]
    sigma_i = [sigma[i] for i in range(MXORDP1)]
    gamma_i = [gamma[i] for i in range(MXORDP1)]
    beta_i[0] = jnp.where(recompute, one, beta_i[0])
    alpha_i[0] = jnp.where(recompute, one, alpha_i[0])
    gamma_i[0] = jnp.where(recompute, zero, gamma_i[0])
    sigma_i[0] = jnp.where(recompute, one, sigma_i[0])

    # The 1..kk recurrence realized as a fixed 1..MAXORD Python unroll with a
    # per-i active mask, since kk is a traced int but every shape here is
    # compile-time fixed at MXORDP1 = MAXORD + 1.
    temp1 = hh
    psi_kk = psi[kk]
    for i in range(1, MAXORD + 1):
        active_i = recompute & (i <= kk)
        temp2 = psi_i[i - 1]
        psi_i[i - 1] = jnp.where(active_i, temp1, temp2)
        # Reads the just-shifted psi[i - 1] (load-bearing: after the write above).
        beta_i[i] = jnp.where(active_i, beta_i[i - 1] * psi_i[i - 1] / temp2, beta_i[i])
        temp1 = temp2 + hh
        alpha_i[i] = jnp.where(active_i, hh / temp1, alpha_i[i])
        sigma_i[i] = jnp.where(active_i, i * sigma_i[i - 1] * alpha_i[i], sigma_i[i])
        gamma_i[i] = jnp.where(active_i, gamma_i[i - 1] + alpha_i[i - 1] / hh, gamma_i[i])
        psi_kk = jnp.where(i == kk, temp1, psi_kk)
    psi = jnp.stack(psi_i)
    psi = psi.at[kk].set(jnp.where(recompute, psi_kk, psi[kk]))
    alpha, beta = jnp.stack(alpha_i), jnp.stack(beta_i)
    sigma, gamma = jnp.stack(sigma_i), jnp.stack(gamma_i)

    # alphas = -sum_{i=1..kk} 1/i ; alpha0 = -sum_{i=0..kk-1} alpha[i].
    # Always recomputed, independent of the recompute gate above.
    alphas = zero
    alpha0 = zero
    for i in range(MAXORD):
        active = i < kk
        alphas = alphas - jnp.where(active, one / (i + 1), zero)
        alpha0 = alpha0 - jnp.where(active, alpha[i], zero)

    cjlast = cj
    cj = -alphas / hh

    ck = jnp.abs(alpha[kk] + alphas - alpha0)
    ck = jnp.maximum(ck, alpha[kk])

    # phi -> phi*: scale only indices ns..kk; index 0 is never scaled. phi
    # itself is not stopped, so phi_star keeps phi's tangent through the
    # (already gradient-dead) beta scale factor.
    # A step that keeps its size and order past kk + 1 steps scales nothing,
    # so the history is left as it is.
    rows = jnp.arange(MXORDP1)
    scale = (rows >= ns) & (rows <= kk) & (rows > 0)
    factors = jnp.where(scale, jax.lax.stop_gradient(beta), one)[:, None]
    phi = lane_cond(ns <= kk, lambda p: factors * p, lambda p: p, phi, false_is_free=True)

    return phi, psi, alpha, beta, sigma, gamma, cjlast, cj, ck, ns, alphas, alpha0


__all__ = ["ida_set_coeffs"]
