"""The accepted-step order/step controller for the fully-implicit DAE engine.

[ida_complete_step][] runs once a step attempt has converged and passed
the local-error test: this routine advances the successful-step bookkeeping,
chooses the order and step size for the next attempt, and folds the converged
Newton correction into the divided-difference history.

The controller has two regimes. During the startup phase it raises the order
and doubles the step every step (after the first) until the order stops
climbing or a step fails; thereafter it runs the usual local-error control:
a LOWER/MAINTAIN/RAISE order decision followed by a step-ratio *deadband* that
only grows the step when the tentative ratio is at least 2, only shrinks it
when the ratio is at most 1, and otherwise holds the step exactly.

The final history update is order-sensitive. It saves the correction into the
next-higher history slot for a possible future order raise, then telescopes
the lower rows so that each row absorbs the *already-updated* row above it --
the standard scaled-divided-difference advance.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup.timestepping.vsvo.common.weights import wrms_norm

from .state import MAXORD, MXORDP1, IDAState

# Order-action sentinels for the LOWER/RAISE/MAINTAIN/UNSET decision.
# ACTION_DOUBLE is this engine's trace code for the startup
# order-raise-and-double branch, which does not set an ``action`` variable.
ACTION_UNSET = -1
ACTION_DOUBLE = 0
ACTION_LOWER = 1
ACTION_RAISE = 2
ACTION_MAINTAIN = 3

# Step-ratio controller knobs.
_ETA_MAX_FX = 2.0
"""Grow the step only when the tentative ratio is at least this."""
_ETA_MIN_FX = 1.0
"""Shrink the step only when the tentative ratio is at most this."""
_ETA_MAX = 2.0
"""Maximum growth factor."""
_ETA_MIN = 0.5
"""Minimum shrink factor on the success path."""
_ETA_LOW = 0.9
"""Upper bound on the shrink factor on the success path."""

_TWO = 2.0
"""Doubling factor for the startup step and the step-ratio bias."""
_HALF = 0.5
"""Order-1 raise threshold factor."""
_ERROR_RATIO_BIAS = 1e-4
"""Bias added inside the step-ratio power law."""


def _controller_action(kk, knew, kdiff, ns, err_k, err_km1, err_kp1, maxord):
    """The LOWER/MAINTAIN/RAISE order decision.

    Realized as a nested ``where`` chain with a fixed priority order: an
    already-decided reduction (``knew == kk - 1``) wins, then the
    maximum-order hold, then the raise gate (not enough constant-``(h, k)``
    history, or the order was just raised), then the truncation-error
    comparison at orders ``k - 1 / k / k + 1``.
    """
    dtype = err_k.dtype
    terr_k = (kk + 1).astype(dtype) * err_k
    terr_kp1 = (kk + 2).astype(dtype) * err_kp1
    terr_km1 = kk.astype(dtype) * err_km1

    action_kk1 = jnp.where(terr_kp1 >= _HALF * terr_k, ACTION_MAINTAIN, ACTION_RAISE)
    action_kk_gt1 = jnp.where(
        terr_km1 <= jnp.minimum(terr_k, terr_kp1),
        ACTION_LOWER,
        jnp.where(terr_kp1 >= terr_k, ACTION_MAINTAIN, ACTION_RAISE),
    )
    action_estimate = jnp.where(kk == 1, action_kk1, action_kk_gt1)

    raise_gate_closed = (kk + 1 >= ns) | (kdiff == 1)
    return jnp.where(
        knew == kk - 1,
        ACTION_LOWER,
        jnp.where(
            kk == maxord,
            ACTION_MAINTAIN,
            jnp.where(raise_gate_closed, ACTION_MAINTAIN, action_estimate),
        ),
    ).astype(jnp.int32)


def _step_ratio_eta(err_knew, kk_next, hh, hmax_inv, hmin):
    """The success-path step-ratio deadband.

    ``tmp = (2*err_knew + 1e-4)^(-1/(kk_next+1))``; grow to at most 2 (with a
    maximum-step clamp) when ``tmp >= 2``, shrink into ``[0.5, 0.9]`` (with a
    minimum-step floor) when ``tmp <= 1``, and hold ``eta == 1`` exactly in
    the open band in between.
    """
    dtype = err_knew.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    abs_hh = jnp.abs(hh)
    tmp = (_TWO * err_knew + _ERROR_RATIO_BIAS) ** (-one / (kk_next + 1).astype(dtype))

    eta_grow = jnp.minimum(tmp, _ETA_MAX)
    eta_grow = eta_grow / jnp.maximum(one, eta_grow * abs_hh * hmax_inv)

    eta_shrink = jnp.minimum(tmp, _ETA_LOW)
    eta_shrink = jnp.maximum(eta_shrink, _ETA_MIN)
    eta_shrink = jnp.maximum(eta_shrink, hmin / abs_hh)

    return jnp.where(tmp >= _ETA_MAX_FX, eta_grow, jnp.where(tmp <= _ETA_MIN_FX, eta_shrink, one))


def _advance_phi(phi, ee, kused, maxord):
    """Fold the converged correction into the divided-difference history.

    Saves ``ee`` into ``phi[kused+1]`` when the order allows (the slot a future
    order raise would read), then runs the sequential telescoping update
    ``phi[kused] += ee`` and, descending, ``phi[j] += phi[j+1]`` where each
    ``phi[j+1]`` is the value just written by the previous iteration -- the
    load-bearing sequencing of the fused vector-array sum.

    ``kused`` is the accepted order (the pre-controller ``kk``), so the save
    index and the telescope length are unaffected by any order raise/lower the
    controller just applied. The loop is a fixed descending unroll over the
    static history length masked by the traced ``kused``, so the shapes never
    depend on runtime values and the update stays linear in ``phi`` and ``ee``.
    """
    kp1_idx = jnp.clip(kused + 1, 0, MXORDP1 - 1)
    save_active = kused < maxord
    phi = phi.at[kp1_idx].set(jnp.where(save_active, ee, phi[kp1_idx]))

    for j in range(MXORDP1 - 1, -1, -1):
        higher = ee if j == MXORDP1 - 1 else phi[j + 1]
        addend = jnp.where(j == kused, ee, higher)
        phi = phi.at[j].set(jnp.where(j <= kused, phi[j] + addend, phi[j]))
    return phi


def ida_complete_step(
    state: IDAState,
    err_k: jax.Array,
    err_km1: jax.Array,
    knew: jax.Array,
    ewt: jax.Array,
    id_mask: jax.Array,
    suppressalg: bool,
    hmax_inv: jax.Array,
    hmin: jax.Array,
    maxord: int = MAXORD,
    ns_for_raise_gate: jax.Array | None = None,
) -> tuple[IDAState, jax.Array, jax.Array]:
    """Complete an accepted step: choose the next order/step and advance ``phi``.

    Runs the successful-step bookkeeping (increment the step count, record the
    used order/step), latches out of the startup phase once the order stops
    climbing, chooses the next order and step size (startup doubling or the
    normal LOWER/MAINTAIN/RAISE controller with its step-ratio deadband), and
    folds the converged correction ``state.ee`` into the divided-difference
    history.

    Args:
        state: The carried step state after a converged, error-test-passing
            attempt. Reads ``kk``, ``kused``, ``hh``, ``ns``, ``phase``,
            ``nst``, ``eta``, ``phi`` and ``ee``.
        err_k: The order-``kk`` local-error estimate from the error test.
        err_km1: The order-``kk-1`` local-error estimate from the error test.
        knew: The order the error test's decrease rule proposed (``int32``).
        ewt: The error-weight vector, shape ``(n,)``.
        id_mask: ``1`` for differential rows, ``0`` for algebraic rows, shape
            ``(n,)``.
        suppressalg: When set, algebraic rows are excluded from the order-``k+1``
            error norm.
        hmax_inv: The reciprocal maximum step size (``0`` for no bound).
        hmin: The minimum step size (``0`` for no bound).
        maxord: The maximum integration order (static).
        ns_for_raise_gate: Optional replacement for the history counter the
            order-raise gate reads. ``None`` (the default) reads the carried
            constant-``(h, k)`` counter ``ns``, matching the reference. A
            controller that re-sizes the step on every accepted step passes
            its own run-of-constant-order counter here — with the reference
            counter the raise gate would never open once the step size moves
            every step. Only the raise gate reads this value; the
            coefficient recurrences keep reading ``ns``.

    Returns:
        A 3-tuple ``(new_state, phase_action, err_knew)``: the updated state
        (``nst``, ``kused``, ``kk``, ``hused``, ``hh``, ``eta``, ``phase`` and
        the advanced ``phi``), the order-action code taken this step (one of
        the ``ACTION_*`` sentinels, for the trace), and the error estimate at
        the chosen next order (for the trace). Every control output is
        gradient-dead; only the advanced ``phi`` carries the accepted
        trajectory's gradient (it is linear in ``phi`` and ``ee``).
    """
    kk = state.kk
    kused_prev = state.kused
    hh = state.hh
    phi = state.phi
    ee = state.ee
    ns = state.ns
    phase = state.phase
    eta = state.eta
    dtype = ee.dtype

    # Successful-step bookkeeping. kdiff uses the previous kused; the new kused
    # and hused record the order and step this accepted step used, BEFORE the
    # controller below bumps kk/hh.
    nst_new = state.nst + 1
    kdiff = kk - kused_prev
    kused_new = kk
    hused_new = hh

    # Latch out of the startup phase once the order stops climbing, BEFORE the
    # phase branch reads it.
    phase_latched = jnp.where((knew == kk - 1) | (kk == maxord), 1, phase).astype(jnp.int32)

    # --- Startup arm: raise the order and double the step (after the first
    # step). eta is deliberately not touched here.
    do_double = nst_new > 1
    hnew = _TWO * hh
    hnew = hnew / jnp.maximum(jnp.asarray(1.0, dtype=dtype), jnp.abs(hnew) * hmax_inv)
    kk_startup = jnp.where(do_double, kk + 1, kk).astype(jnp.int32)
    hh_startup = jnp.where(do_double, hnew, hh)
    action_startup = jnp.where(do_double, ACTION_DOUBLE, ACTION_UNSET).astype(jnp.int32)

    # --- Normal arm: order decision then the step-ratio deadband. The
    # order-(k+1) estimate carries NO sigma factor.
    kp1_idx = jnp.clip(kk + 1, 0, MXORDP1 - 1)
    mask = id_mask if suppressalg else None
    enorm_kp1 = wrms_norm(ee - phi[kp1_idx], ewt, mask=mask)
    err_kp1 = enorm_kp1 / (kk + 2).astype(dtype)

    ns_gate = ns if ns_for_raise_gate is None else ns_for_raise_gate
    action = _controller_action(kk, knew, kdiff, ns_gate, err_k, err_km1, err_kp1, maxord)
    kk_normal = jnp.where(
        action == ACTION_RAISE, kk + 1, jnp.where(action == ACTION_LOWER, kk - 1, kk)
    ).astype(jnp.int32)
    err_knew_normal = jnp.where(
        action == ACTION_RAISE, err_kp1, jnp.where(action == ACTION_LOWER, err_km1, err_k)
    )
    eta_normal = _step_ratio_eta(err_knew_normal, kk_normal, hh, hmax_inv, hmin)
    hh_normal = hh * eta_normal

    # --- Select the arm on the (latched) phase.
    is_startup = phase_latched == 0
    kk_new = jnp.where(is_startup, kk_startup, kk_normal).astype(jnp.int32)
    hh_new = jnp.where(is_startup, hh_startup, hh_normal)
    eta_new = jnp.where(is_startup, eta, eta_normal)
    phase_action = jnp.where(is_startup, action_startup, action).astype(jnp.int32)
    err_knew = jnp.where(is_startup, err_k, err_knew_normal)

    # --- History advance uses the accepted (pre-controller) order kused_new.
    phi_new = _advance_phi(phi, ee, kused_new, maxord)

    new_state = eqx.tree_at(
        lambda s: (s.nst, s.kused, s.kk, s.hused, s.hh, s.eta, s.phase, s.phi),
        state,
        (
            jax.lax.stop_gradient(nst_new),
            jax.lax.stop_gradient(kused_new),
            jax.lax.stop_gradient(kk_new),
            jax.lax.stop_gradient(hused_new),
            jax.lax.stop_gradient(hh_new),
            jax.lax.stop_gradient(eta_new),
            jax.lax.stop_gradient(phase_latched),
            phi_new,
        ),
    )
    return new_state, jax.lax.stop_gradient(phase_action), jax.lax.stop_gradient(err_knew)


__all__ = [
    "ida_complete_step",
    "ACTION_UNSET",
    "ACTION_DOUBLE",
    "ACTION_LOWER",
    "ACTION_RAISE",
    "ACTION_MAINTAIN",
]
