"""The flash profile's accepted-step step-size controller.

The ``flash`` execution profile pairs a fresh Jacobian on every nonlinear
iteration with a proportional-integral step-size law. The source-tracking
controllers hold the step size inside deadbands and freeze it after
failures because every step-size change invalidates a reused Jacobian
factorization; with the factorization rebuilt every iteration those holds
protect nothing, so under flash the step is re-sized on **every** accepted
step. Everything else about an engine — its history representation,
coefficient recurrences, error estimators, order selection, and failure
ladders — is unchanged: the order moves only when the engine's own order
ladder identifies a better order on enough settled history.

The default law (``law="pi"``) is the proportional-integral controller

    ``factor = safety * err^(-ki/(k+1)) * (err/err_prev)^(-kp/(k+1))``

with gains ``kp = 0.4`` and ``ki = 0.3``, ``safety = 0.9``, and both
exponents normalized by the order ``k`` the error estimate was formed at.
``err`` is the engine's normalized local-error estimate with its accept
boundary at 1, and ``err_prev`` is the previous attempt's — the controller
memory each engine carries and refreshes on every attempt whose error test
ran. Two clamped step ratios come out of the one factor:

* accepted step: the factor clamped into ``[0.2, 5]``, additionally capped
  at 1 when the previous attempt was a rejection (the carried
  ``last_accept`` flag) so a step accepted right after a rejection may not
  grow;
* rejected step: the factor clamped into ``[0.2, 0.9]`` — the retry of a
  step that failed its error test never grows.

The Söderlind H211b digital filter remains available behind the same
adapter (``law="h211b"``) for step-law comparisons: ``log h_new = log h +
beta*(r + r_prev) + gamma*(log h - log h_prev)`` with ``beta =
1/(4*(order+1))`` and ``gamma = -1/4`` (Söderlind, "Digital filters in
adaptive time-stepping," ACM TOMS 29(1), 2003, §6.2), informants saturated
at ``+2`` in log space, a 0.9 safety factor, a growth cap of 10, and a
single-step shrink floor of 0.1; its rejection ratio is the same predicted
ratio clamped to at most 0.9.

Both laws respect the ``hmin`` floor and the ``hmax`` cap and preserve the
sign of ``h``, so backward-in-time integration works unchanged. Every
output is a step-control quantity and is gradient-dead (stopped), the same
discipline as the source controllers.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup.timestepping._shared.controllers.h211b import h211b_log_h_hat

FLASH_KP = 0.4
"""Proportional gain of the PI law, applied as ``kp/(order + 1)``."""

FLASH_KI = 0.3
"""Integral gain of the PI law, applied as ``ki/(order + 1)``."""

FLASH_SAFETY = 0.9
"""Safety factor multiplying the predicted step ratio (both laws)."""

FLASH_GROWTH_MIN = 0.2
"""Smallest step-size ratio an accepted step may apply under the PI law."""

FLASH_GROWTH_MAX = 5.0
"""Largest step-size ratio an accepted step may apply under the PI law."""

FLASH_REJECT_CAP = 0.9
"""Largest step-size ratio a rejected step's retry may apply: a rejected
step never grows."""

FLASH_ERR_TINY = 1e-30
"""Floor applied to both error informants before the power laws, guarding
the ``err = 0`` (exactly fitted step) singularity."""

FLASH_BETA_SCALE = 0.25
"""H211b error-informant gain scale: ``beta = FLASH_BETA_SCALE/(order + 1)``."""

FLASH_GAMMA = -0.25
"""H211b step-ratio damping coefficient (filter strength 4)."""

FLASH_SATURATION = 2.0
"""Log-space cap on each H211b error informant (the filter's anti-windup
limiter)."""

FLASH_ETA_SHRINK_MIN = 0.1
"""Smallest step-size ratio a single accepted step may apply under the
H211b law."""

FLASH_H211B_GROWTH_MAX = 10.0
"""Largest step-size ratio a single accepted step may apply under the
H211b law."""

FLASH_ERR_SEED = 1.0
"""Seed for the error memory before the first accepted step: the accept
boundary of the normalized error. With the seed at the boundary the first
step's proportional term reduces to a pure integral response, so the seed
never distorts the first decision."""


class FlashControlResult(eqx.Module):
    """The flash controller's step-size decision for one step attempt.

    Attributes:
        eta: The step-size ratio to apply if the attempt is accepted
            (positive; gradient-dead).
        h_new: The next step size on acceptance, ``h * eta`` (keeps the
            sign of ``h``; gradient-dead).
        eta_reject: The step-size ratio for the retry if the attempt fails
            its error test (positive, at most 0.9; gradient-dead).
        h_reject: The retry step size on rejection, ``h * eta_reject``
            (keeps the sign of ``h``; gradient-dead).
    """

    eta: jax.Array
    h_new: jax.Array
    eta_reject: jax.Array
    h_reject: jax.Array


def flash_step_control(
    err_norm: jax.Array,
    err_prev: jax.Array,
    *,
    h: jax.Array,
    h_prev: jax.Array,
    order: jax.Array,
    last_accept: jax.Array | bool,
    first_step: jax.Array | bool,
    hmax_inv: jax.Array,
    hmin: jax.Array,
    max_order: int = 5,
    law: str = "pi",
) -> FlashControlResult:
    """Size the next step from a normalized local-error estimate.

    ``err_norm`` is the engine's own normalized local-error estimate for
    this attempt, with the accept boundary at 1; ``err_prev`` is the
    previous attempt's (the controller memory each engine carries). ``h``
    is the step size this attempt used; its sign is ignored by both laws
    and preserved on the returned step sizes.

    Args:
        err_norm: Normalized local error of this attempt (boundary 1).
        err_prev: Normalized local error of the previous attempt whose
            error test ran.
        h: The step size this attempt used.
        h_prev: The step size of the previous accepted step (read only by
            the H211b law).
        order: The integration order the error estimate was formed at
            (traced ``int32``, clipped into ``1..max_order``).
        last_accept: Whether the previous attempt was accepted. A false
            value caps the accepted-step ratio at 1, suppressing growth on
            the step right after a rejection.
        first_step: True on the first step of a run, when no memory exists
            yet (read only by the H211b law, which then substitutes the
            current error and step for the missing previous ones; the PI
            law's boundary-valued seed already makes the memory inert).
        hmax_inv: Reciprocal of the maximum step size (``0`` for no bound).
        hmin: The minimum step size (``0`` for no bound).
        max_order: The engine's static maximum order.
        law: ``"pi"`` for the proportional-integral law (the default),
            ``"h211b"`` for the Söderlind digital filter.

    Returns:
        The gradient-dead [FlashControlResult][diff_soup.timestepping.vsvo.common.flash_controller.FlashControlResult] — the clamped
        accepted-step and rejected-step ratios and step sizes.
    """
    h = jnp.asarray(h)
    dtype = h.dtype
    one = jnp.ones((), dtype=dtype)
    habs = jnp.abs(h)
    tiny = jnp.asarray(FLASH_ERR_TINY, dtype=dtype)
    err = jnp.maximum(jnp.asarray(err_norm, dtype=dtype), tiny)
    prev = jnp.maximum(jnp.asarray(err_prev, dtype=dtype), tiny)
    last_accept = jnp.asarray(last_accept)

    if law == "pi":
        k_f = jnp.clip(jnp.asarray(order), 1, max_order).astype(dtype) + one
        ki_eff = jnp.asarray(FLASH_KI, dtype=dtype) / k_f
        kp_eff = jnp.asarray(FLASH_KP, dtype=dtype) / k_f
        factor = (
            jnp.asarray(FLASH_SAFETY, dtype=dtype)
            * jnp.power(err, -ki_eff)
            * jnp.power(err / prev, -kp_eff)
        )
        factor = jnp.where(last_accept, factor, jnp.minimum(one, factor))
        eta = jnp.clip(
            factor,
            jnp.asarray(FLASH_GROWTH_MIN, dtype=dtype),
            jnp.asarray(FLASH_GROWTH_MAX, dtype=dtype),
        )
        eta_reject = jnp.clip(
            factor,
            jnp.asarray(min(FLASH_GROWTH_MIN, FLASH_REJECT_CAP), dtype=dtype),
            jnp.asarray(FLASH_REJECT_CAP, dtype=dtype),
        )
    elif law == "h211b":
        habs_prev = jnp.abs(jnp.asarray(h_prev, dtype=dtype))
        # The shared filter evaluates per-order gains from vector position,
        # so evaluate it at every order 1..max_order with the same scalar
        # inputs and select the entry for the order the error estimate was
        # formed at.
        err_vec = jnp.full((max_order,), err, dtype=dtype)
        err_prev_vec = jnp.full((max_order,), prev, dtype=dtype)
        log_h_hat = h211b_log_h_hat(
            err_vec,
            err_prev_vec,
            h=habs,
            h_prev=habs_prev,
            tol=one,
            beta_scale=FLASH_BETA_SCALE,
            gamma=FLASH_GAMMA,
            first_step=jnp.asarray(first_step),
            saturation_log_ratio=FLASH_SATURATION,
        )
        idx = jnp.clip(jnp.asarray(order) - 1, 0, max_order - 1)
        eta_pred = jnp.asarray(FLASH_SAFETY, dtype=dtype) * jnp.exp(log_h_hat[idx]) / habs
        eta = jnp.clip(
            eta_pred,
            jnp.asarray(FLASH_ETA_SHRINK_MIN, dtype=dtype),
            jnp.asarray(FLASH_H211B_GROWTH_MAX, dtype=dtype),
        )
        eta_reject = jnp.clip(
            eta_pred,
            jnp.asarray(FLASH_ETA_SHRINK_MIN, dtype=dtype),
            jnp.asarray(FLASH_REJECT_CAP, dtype=dtype),
        )
    else:
        raise ValueError(f"unknown flash controller law {law!r}; expected 'pi' or 'h211b'.")

    # Bounds shared by both laws: the hmin floor on both ratios, the hard
    # hmax cap on the accepted-step ratio.
    hmin_ratio = jnp.asarray(hmin, dtype=dtype) / habs
    eta = jnp.maximum(eta, hmin_ratio)
    eta = eta / jnp.maximum(one, eta * habs * jnp.asarray(hmax_inv, dtype=dtype))
    eta_reject = jnp.maximum(eta_reject, hmin_ratio)

    return FlashControlResult(
        eta=jax.lax.stop_gradient(eta),
        h_new=jax.lax.stop_gradient(h * eta),
        eta_reject=jax.lax.stop_gradient(eta_reject),
        h_reject=jax.lax.stop_gradient(h * eta_reject),
    )


__all__ = [
    "FLASH_BETA_SCALE",
    "FLASH_ERR_SEED",
    "FLASH_ERR_TINY",
    "FLASH_ETA_SHRINK_MIN",
    "FLASH_GAMMA",
    "FLASH_GROWTH_MAX",
    "FLASH_GROWTH_MIN",
    "FLASH_H211B_GROWTH_MAX",
    "FLASH_KI",
    "FLASH_KP",
    "FLASH_REJECT_CAP",
    "FLASH_SAFETY",
    "FLASH_SATURATION",
    "FlashControlResult",
    "flash_step_control",
]
