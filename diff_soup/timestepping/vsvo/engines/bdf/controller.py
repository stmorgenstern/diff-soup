"""The step-size and order controller for the backward-differentiation-formula engine.

After an accepted step, the controller picks the order and step size for the
next attempt, and dispatches on two pieces of carried state:

* ``etamax == 1`` -- set whenever a convergence or error-test failure occurred
  on the way to this accepted step -- defers all changes: the order is held,
  the step size is unchanged, and the order-change wait counter is pushed out.
  ``etamax`` is a carried field: it starts large on the first step, resets to a
  moderate cap after each successful step (owned by the step assembly), and is
  forced to 1 by the failure ladders in
  [failure][diff_soup.timestepping.vsvo.engines.bdf.failure], persisting through
  the retry attempts so the eventual successful attempt sees it and defers.
* otherwise the order-``q`` step-size ratio ``etaq`` is formed; if the wait
  counter is still counting down only ``etaq`` is used (no order change),
  and if it has reached zero the order-``q-1`` and order-``q+1`` candidates
  are formed too and the largest wins ([bdf_choose_eta][]), with ties
  resolved same > lower > raise.

The chosen ratio is then clamped by [bdf_set_eta][]: a ratio inside the
deadband ``(0, 1.5)`` is snapped to 1 (the step size is retained), and a ratio
at or above the deadband's upper edge is capped by ``etamax`` and the optional
``hmax`` limit. With the deadband's lower edge at 0 every positive ratio below
1.5 lands in the deadband, so a *successful* step never shrinks the step size --
reductions come only from the failure ladders and the endpoint clamp.

Every scalar control the controller returns -- the step-size ratio, the next
order and step size, the wait counter, the candidate ratios, the chosen branch,
and the constant-step-run counter -- is gradient-dead and stopped. The one
exception is the returned Nordsieck history ``zn``: an order increase refreshes
its spare column with the accepted correction ``acor``, which is
part of the differentiable trajectory, so ``zn`` keeps its gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_soup.timestepping.vsvo.common.weights import wrms_norm

from .state import QMAX, BDFState

BIAS1 = 6.0
"""Bias in the order-decrease step-ratio candidate."""

BIAS2 = 6.0
"""Bias in the order-``q`` step-ratio candidate."""

BIAS3 = 10.0
"""Bias in the order-increase step-ratio candidate."""

ADDON = 1e-6
"""Safety addend in every step-ratio formula."""

ETA_MIN_FX = 0.0
"""Lower edge of the step-size-retention deadband."""

ETA_MAX_FX = 1.5
"""Upper edge of the step-size-retention deadband."""

ETA_MIN = 0.1
"""Floor applied on the step-size-decrease branch."""

# The maximum step-ratio integer power the order-increase candidate raises;
# BDF order tops out at QMAX so L = q+1 never exceeds QMAX+1.
_L_MAX = QMAX + 1


class ChosenBranch:
    """Sentinel integers naming which order/step branch the controller took.

    These feed the per-step trace so a run's step-by-step decisions can be
    inspected and compared.
    """

    HOLD = 0
    """The step size and order were both held (the deadband, or a deferred
    change after a failure)."""
    SAME = 1
    """The order was kept but the step size changed."""
    LOWER = 2
    """The order was decreased by one."""
    RAISE = 3
    """The order was increased by one."""


def _int_power(base: jax.Array, exponent: jax.Array, dtype: type) -> jax.Array:
    """Raise ``base`` to a traced non-negative integer ``exponent``.

    Uses a repeated multiplication rather than a floating-point ``pow``: the
    running product multiplies in ``base`` once for each step of a static
    ``1..L_MAX`` range where the index does not exceed ``exponent``.
    """
    prod = jnp.ones((), dtype=dtype)
    for j in range(1, _L_MAX + 1):
        prod = jnp.where(j <= exponent, prod * base, prod)
    return prod


def bdf_compute_eta_qm1(zn: jax.Array, q: jax.Array, tq: jax.Array, ewt: jax.Array) -> jax.Array:
    """The order-decrease step-size ratio candidate ``etaqm1``.

    Zero at order 1 (there is no order below to try); otherwise
    ``etaqm1 = 1/((BIAS1*ddn)^(1/q) + ADDON)`` with
    ``ddn = ||zn[q]||_wrms(ewt) * tq[1]``.
    """
    dtype = zn.dtype
    q_idx = jnp.clip(q, 0, zn.shape[0] - 1)
    q_f = q.astype(dtype)
    ddn = wrms_norm(zn[q_idx], ewt) * tq[1]
    val = 1.0 / (jnp.power(BIAS1 * ddn, 1.0 / q_f) + ADDON)
    return jnp.where(q > 1, val, jnp.zeros((), dtype=dtype))


def bdf_compute_eta_qp1(
    acor: jax.Array,
    zn: jax.Array,
    saved_tq5: jax.Array,
    tq: jax.Array,
    h: jax.Array,
    tau: jax.Array,
    L: jax.Array,  # noqa: N803 -- matches the reference symbol cv_L.
    q: jax.Array,
    ewt: jax.Array,
    dtype: type,
) -> jax.Array:
    """The order-increase step-size ratio candidate ``etaqp1``.

    Zero at the maximum order, and zero until a correction has been saved
    (``saved_tq5 == 0``). Otherwise it estimates the order-``q+2`` derivative
    from the difference between the current correction and the scaled saved one,
    with the scale ``cquot = (tq[5]/saved_tq5) * (h/tau[2])^L`` using the
    integer power ``(h/tau[2])^L``, and returns
    ``etaqp1 = 1/((BIAS3*dup)^(1/(L+1)) + ADDON)`` with
    ``dup = ||acor - cquot*zn[qmax]||_wrms(ewt) * tq[3]``.
    """
    active = (q != QMAX) & (saved_tq5 != 0)
    one = jnp.ones((), dtype=dtype)
    safe_saved = jnp.where(saved_tq5 != 0, saved_tq5, one)
    safe_tau2 = jnp.where(tau[2] != 0, tau[2], one)
    L_f = L.astype(dtype)

    cquot = (tq[5] / safe_saved) * _int_power(h / safe_tau2, L, dtype)
    tempv = acor - cquot * zn[QMAX]
    dup = wrms_norm(tempv, ewt) * tq[3]
    val = 1.0 / (jnp.power(BIAS3 * dup, 1.0 / (L_f + 1.0)) + ADDON)
    return jnp.where(active, val, jnp.zeros((), dtype=dtype))


def bdf_choose_eta(
    etaqm1: jax.Array,
    etaq: jax.Array,
    etaqp1: jax.Array,
    q: jax.Array,
    acor: jax.Array,
    zn: jax.Array,
    dtype: type,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pick the largest of the three step-ratio candidates and its order.

    Selects ``etam = max(etaqm1, etaq, etaqp1)``. If ``etam`` is inside the
    deadband ``(0, 1.5)`` the order is held and ``eta`` is snapped to 1.
    Otherwise the candidate equal to ``etam`` wins, with ties resolved in the
    order same > lower > raise (the ``==`` tests are evaluated in exactly that
    order). An order increase stores the correction into the spare Nordsieck
    column ``zn[qmax]`` for the coming order change.

    Returns:
        ``(eta, qprime, chosen_branch, zn)``: the chosen ratio, the next order,
        the [ChosenBranch][] sentinel, and the (possibly refreshed) history.
    """
    etam = jnp.maximum(etaqm1, jnp.maximum(etaq, etaqp1))
    deadband = (etam > ETA_MIN_FX) & (etam < ETA_MAX_FX)

    is_same = etam == etaq
    is_lower = (~is_same) & (etam == etaqm1)
    is_raise = (~is_same) & (~is_lower)  # etam == etaqp1

    eta = jnp.where(
        deadband,
        jnp.ones((), dtype=dtype),
        jnp.where(is_same, etaq, jnp.where(is_lower, etaqm1, etaqp1)),
    )
    qprime = jnp.where(
        deadband, q, jnp.where(is_same, q, jnp.where(is_lower, q - 1, q + 1))
    ).astype(jnp.int32)
    chosen_branch = jnp.where(
        deadband,
        ChosenBranch.HOLD,
        jnp.where(
            is_same, ChosenBranch.SAME, jnp.where(is_lower, ChosenBranch.LOWER, ChosenBranch.RAISE)
        ),
    ).astype(jnp.int32)

    do_raise = (~deadband) & is_raise
    zn_raise = zn.at[QMAX].set(acor)
    zn_out = jnp.where(do_raise, zn_raise, zn)

    return eta, qprime, chosen_branch, zn_out


def bdf_set_eta(
    eta: jax.Array,
    qprime: jax.Array,
    q: jax.Array,
    h: jax.Array,
    etamax: jax.Array,
    hmax_inv: jax.Array,
    hmin: jax.Array,
    nscon: jax.Array,
    dtype: type,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Clamp the chosen step-size ratio and set the next step size.

    A ratio inside the deadband ``(0, 1.5)`` is snapped to 1 and the step size
    is retained. At or above the deadband's upper edge the increase branch caps
    the ratio by ``etamax`` and the optional ``hmax`` limit; below the lower
    edge the decrease branch floors it -- unreachable while the deadband's lower
    edge stays at its default 0, since every candidate ratio is positive. An
    order decrease resets the constant-step-run counter ``nscon``.

    Returns:
        ``(eta, hprime, nscon)``: the clamped ratio, the next step size, and the
        (possibly reset) constant-step-run counter.
    """
    abs_h = jnp.abs(h)
    deadband = (eta > ETA_MIN_FX) & (eta < ETA_MAX_FX)

    # increase branch
    eta_inc = jnp.minimum(eta, etamax)
    eta_inc = eta_inc / jnp.maximum(jnp.ones((), dtype=dtype), abs_h * hmax_inv * eta_inc)
    # decrease branch
    eta_dec = jnp.maximum(eta, ETA_MIN)
    eta_dec = jnp.maximum(eta_dec, hmin / abs_h)

    eta_else = jnp.where(eta >= ETA_MAX_FX, eta_inc, eta_dec)
    hprime_else = h * eta_else
    nscon_else = jnp.where(qprime < q, jnp.zeros((), jnp.int32), nscon)

    eta_out = jnp.where(deadband, jnp.ones((), dtype=dtype), eta_else)
    hprime_out = jnp.where(deadband, h, hprime_else)
    nscon_out = jnp.where(deadband, nscon, nscon_else).astype(jnp.int32)
    return eta_out, hprime_out, nscon_out


def bdf_prepare_next_step(
    state: BDFState,
    acor: jax.Array,
    dsm: jax.Array,
    etamax: jax.Array,
    hmax_inv: jax.Array,
    hmin: jax.Array,
    ewt: jax.Array,
    dtype: type,
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
]:
    """Decide the next attempt's order and step size after an accepted step.

    Args:
        state: The carried step state (read for ``q``, ``L``, ``h``, ``qwait``,
            ``tq``, ``tau``, ``zn``, ``saved_tq5``, ``nscon``).
        acor: The accepted correction, shape ``(n,)``.
        dsm: The scaled local error from the error test.
        etamax: The step-growth cap carried into this step (1 defers all
            changes after a failure).
        hmax_inv: The reciprocal of the maximum step size (0 for no cap).
        hmin: The minimum step size.
        ewt: The error-weight vector, shape ``(n,)``.
        dtype: The floating-point dtype.

    Returns:
        ``(eta, qprime, hprime, qwait, etaq, etaqm1, etaqp1, chosen_branch,
        nscon, zn)``. Every entry but ``zn`` is a gradient-dead control; ``zn``
        keeps its gradient because an order increase refreshes it with ``acor``.
    """
    q = state.q
    L = state.L
    h = jnp.asarray(state.h, dtype=dtype)
    qwait = state.qwait
    nscon = state.nscon
    zn = state.zn
    etamax = jnp.asarray(etamax, dtype=dtype)
    hmax_inv = jnp.asarray(hmax_inv, dtype=dtype)
    hmin = jnp.asarray(hmin, dtype=dtype)
    dsm = jnp.asarray(dsm, dtype=dtype)
    acor = jnp.asarray(acor, dtype=dtype)

    L_f = L.astype(dtype)
    etaq = 1.0 / (jnp.power(BIAS2 * dsm, 1.0 / L_f) + ADDON)
    zero = jnp.zeros((), dtype=dtype)

    # --- no order change (qwait still counting down) ---
    eta_b, hprime_b, nscon_b = bdf_set_eta(etaq, q, q, h, etamax, hmax_inv, hmin, nscon, dtype)
    deadband_b = (etaq > ETA_MIN_FX) & (etaq < ETA_MAX_FX)
    branch_b = jnp.where(deadband_b, ChosenBranch.HOLD, ChosenBranch.SAME).astype(jnp.int32)

    # --- order change considered (qwait reached zero) ---
    etaqm1_c = bdf_compute_eta_qm1(zn, q, state.tq, ewt)
    etaqp1_c = bdf_compute_eta_qp1(
        acor, zn, state.saved_tq5, state.tq, h, state.tau, L, q, ewt, dtype
    )
    eta_c0, qprime_c, branch_c, zn_c = bdf_choose_eta(etaqm1_c, etaq, etaqp1_c, q, acor, zn, dtype)
    eta_c, hprime_c, nscon_c = bdf_set_eta(
        eta_c0, qprime_c, q, h, etamax, hmax_inv, hmin, nscon, dtype
    )

    # select the no-change vs order-change arm by the wait counter
    qnz = qwait != 0
    eta_bc = jnp.where(qnz, eta_b, eta_c)
    hprime_bc = jnp.where(qnz, hprime_b, hprime_c)
    nscon_bc = jnp.where(qnz, nscon_b, nscon_c)
    qprime_bc = jnp.where(qnz, q, qprime_c)
    branch_bc = jnp.where(qnz, branch_b, branch_c)
    zn_bc = jnp.where(qnz, zn, zn_c)
    etaqm1_bc = jnp.where(qnz, zero, etaqm1_c)
    etaqp1_bc = jnp.where(qnz, zero, etaqp1_c)
    qwait_bc = jnp.where(qnz, qwait, jnp.asarray(2, jnp.int32))

    # --- defer everything after a failure (etamax == 1) ---
    defer = etamax == 1
    one = jnp.ones((), dtype=dtype)

    eta_out = jnp.where(defer, one, eta_bc)
    qprime_out = jnp.where(defer, q, qprime_bc).astype(jnp.int32)
    hprime_out = jnp.where(defer, h, hprime_bc)
    qwait_out = jnp.where(defer, jnp.maximum(qwait, 2), qwait_bc).astype(jnp.int32)
    branch_out = jnp.where(defer, ChosenBranch.HOLD, branch_bc).astype(jnp.int32)
    nscon_out = jnp.where(defer, nscon, nscon_bc).astype(jnp.int32)
    zn_out = jnp.where(defer, zn, zn_bc)
    etaq_out = jnp.where(defer, zero, etaq)
    etaqm1_out = jnp.where(defer, zero, etaqm1_bc)
    etaqp1_out = jnp.where(defer, zero, etaqp1_bc)

    return (
        jax.lax.stop_gradient(eta_out),
        jax.lax.stop_gradient(qprime_out),
        jax.lax.stop_gradient(hprime_out),
        jax.lax.stop_gradient(qwait_out),
        jax.lax.stop_gradient(etaq_out),
        jax.lax.stop_gradient(etaqm1_out),
        jax.lax.stop_gradient(etaqp1_out),
        jax.lax.stop_gradient(branch_out),
        jax.lax.stop_gradient(nscon_out),
        zn_out,  # the order-raise history refresh keeps its gradient through acor
    )


__all__ = [
    "ChosenBranch",
    "bdf_compute_eta_qm1",
    "bdf_compute_eta_qp1",
    "bdf_choose_eta",
    "bdf_set_eta",
    "bdf_prepare_next_step",
]
