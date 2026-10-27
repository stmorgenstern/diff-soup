"""Step-size and order controllers for the NDF engine.

Both controllers implement the adaptive-BDF scheme of Alberdi Celaya, Anza
Aguirrezabala & Chatzipantelidis (2014), in the form adopted by
OrdinaryDiffEq.jl's QNDF (MIT license; see ``THIRD_PARTY_NOTICES.md``).

After an accepted step, three candidate orders (``k-1``, ``k``, ``k+1``)
each propose a step from their own error estimate through a banded factor
``z = target * est^(1/(k+delta))``; the largest proposal wins the order.
The proposal is then *realized* only outside two hold windows — a
constant-step preference window that applies until ``k + 2`` consecutive
steps have run at the current step size, and a steady-state window that
suppresses mild shrinks — which is what gives the method its
quasi-constant-step character. A successful step may therefore shrink the
step size; there is no grow-only rule.

After a rejected step, the ``k`` and ``k-1`` candidates propose shrunken
retries; a second consecutive rejection halves the step first, a third
forces the order down, and a fourth at order 1 restarts the history from
the current state (the step loop reseeds the difference table).

Every output is a step-control quantity and is gradient-dead.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

TARGET_SAME = 1.2
"""Error-target multiplier of the same-order candidate."""

TARGET_DOWN = 1.3
"""Error-target multiplier of the order-``k-1`` candidate."""

TARGET_UP = 1.4
"""Error-target multiplier of the order-``k+1`` candidate."""

GROWTH_BAND_FLOOR = 0.1
"""Factor values at or below this propose the maximum growth instead."""

GROWTH_CAP = 10.0
"""Maximum step growth per accepted step (also the zero-error growth)."""

CONST_WINDOW_LO = 0.6
CONST_WINDOW_HI = 1.2
"""The constant-step preference window on ``q = dt / dt_candidate``."""

QSTEADY_MIN = 1.0
QSTEADY_MAX = 2.0
"""The steady-state hold window on ``q`` (mild shrinks are suppressed)."""

REJECT_BAND_CAP = 10.0
"""Rejected-step factor values above this fall to the fixed-ratio floor."""

REJECT_FLOOR_RATIO = 0.1
"""The fixed shrink ratio once a rejected-step factor leaves its band."""


class AcceptControl(NamedTuple):
    """The accepted-step controller's decision."""

    dt_next: jax.Array
    k_next: jax.Array
    nconsteps_next: jax.Array
    consfailcnt_next: jax.Array
    held: jax.Array
    """Whether a hold window suppressed the scanned step proposal."""


class RejectControl(NamedTuple):
    """The rejected-step controller's decision."""

    dt_next: jax.Array
    k_next: jax.Array
    consfailcnt_next: jax.Array
    nconsteps_next: jax.Array
    restart_history: jax.Array
    """Whether the step loop must reseed the difference table at order 1."""


def _banded_proposal(dt, z, band_hi, dtype):
    """A candidate's step proposal from its factor ``z``.

    At or below the growth floor the candidate proposes the capped maximum
    growth; inside ``(floor, band_hi]`` it proposes ``dt / z``; above the
    band it proposes nothing (zero, which can never win the scan).
    """
    grow = jnp.asarray(GROWTH_CAP, dtype) * dt
    divide = dt / jnp.where(z > 0, z, 1.0)
    inside = (z > GROWTH_BAND_FLOOR) & (z <= band_hi)
    return jnp.where(z <= GROWTH_BAND_FLOOR, grow, jnp.where(inside, divide, 0.0))


def accept_step_control(
    *,
    dt: jax.Array,
    k: jax.Array,
    eest: jax.Array,
    eest1: jax.Array,
    eest2: jax.Array,
    nconsteps: jax.Array,
    max_order: jax.Array,
) -> AcceptControl:
    """Choose the next step size and order after an accepted step.

    Args:
        dt: The accepted step's (unsigned) size.
        k: The accepted step's order.
        eest: The accepted step's scaled error estimate (``<= 1``).
        eest1: The order-``k-1`` error estimate (read when ``k > 1``).
        eest2: The order-``k+1`` error estimate (read when ``k < max_order``).
        nconsteps: Consecutive steps at this step size, pre-increment.
        max_order: The order cap for this solve.

    Returns:
        The [AcceptControl][diff_soup.timestepping.vsvo.engines.ndf.controller.AcceptControl] decision. ``dt_next`` is unclamped; the
        step loop applies the ``[hmin, hmax]`` bounds and the endpoint clamp.
    """
    dtype = jnp.asarray(dt).dtype
    k_f = k.astype(dtype)
    nconsteps_next = (nconsteps + 1).astype(jnp.int32)
    prefer_const = nconsteps_next < (k + 2)

    z_same = TARGET_SAME * jnp.power(eest, 1.0 / (k_f + 1.0))
    # the same-order candidate has no upper band: any z above the growth
    # floor proposes dt / z
    h_same = jnp.where(
        z_same <= GROWTH_BAND_FLOOR,
        jnp.asarray(GROWTH_CAP, dtype) * dt,
        dt / jnp.where(z_same > 0, z_same, 1.0),
    )
    h_run, k_run = h_same, k

    active_dn = k > 1
    z_dn = TARGET_DOWN * jnp.power(eest1, 1.0 / k_f)
    h_dn = _banded_proposal(dt, z_dn, TARGET_DOWN, dtype)
    dn_wins = active_dn & (h_dn > h_run)
    h_run = jnp.where(dn_wins, h_dn, h_run)
    k_run = jnp.where(dn_wins, k - 1, k_run)

    active_up = k < max_order
    z_up = TARGET_UP * jnp.power(eest2, 1.0 / (k_f + 2.0))
    h_up = _banded_proposal(dt, z_up, TARGET_UP, dtype)
    up_wins = active_up & (h_up > h_run)
    h_run = jnp.where(up_wins, h_up, h_run)
    k_run = jnp.where(up_wins, k + 1, k_run)

    q = dt / h_run
    hold_const = prefer_const & (q > CONST_WINDOW_LO) & (q < CONST_WINDOW_HI)
    hold_steady = (q >= QSTEADY_MIN) & (q <= QSTEADY_MAX)
    held = hold_const | hold_steady
    dt_scan = jnp.where(held, dt, h_run)

    # zero-error fast path: maximum growth, order held
    is_zero = eest <= 0.0
    dt_next = jnp.where(is_zero, jnp.asarray(GROWTH_CAP, dtype) * dt, dt_scan)
    k_next = jnp.where(is_zero, k, k_run).astype(jnp.int32)
    held_out = jnp.where(is_zero, False, held)

    return AcceptControl(
        dt_next=jax.lax.stop_gradient(dt_next),
        k_next=k_next,
        nconsteps_next=nconsteps_next,
        consfailcnt_next=jnp.zeros((), jnp.int32),
        held=held_out,
    )


def reject_step_control(
    *,
    dt: jax.Array,
    k: jax.Array,
    eest: jax.Array,
    eest1: jax.Array,
    consfailcnt: jax.Array,
) -> RejectControl:
    """Choose the retry step size and order after an error-test rejection.

    Args:
        dt: The rejected step's (unsigned) size.
        k: The rejected step's order.
        eest: The rejected step's scaled error estimate (``> 1``).
        eest1: The order-``k-1`` error estimate (read when ``k > 1``).
        consfailcnt: Consecutive rejections of this step, pre-increment.

    Returns:
        The [RejectControl][diff_soup.timestepping.vsvo.engines.ndf.controller.RejectControl] decision. ``dt_next`` is unclamped; the
        step loop applies the ``hmin`` bound and its collapse abort.
    """
    dtype = jnp.asarray(dt).dtype
    k_f = k.astype(dtype)
    consfail_next = (consfailcnt + 1).astype(jnp.int32)

    # a second consecutive rejection halves the step before the candidates run
    h = jnp.where(consfail_next > 1, 0.5 * dt, dt)

    z_same = TARGET_SAME * jnp.power(eest, 1.0 / (k_f + 1.0))
    h_same = jnp.where(
        z_same <= REJECT_BAND_CAP,
        h / jnp.where(z_same > 0, z_same, 1.0),
        jnp.asarray(REJECT_FLOOR_RATIO, dtype) * h,
    )

    active_dn = k > 1
    z_dn = TARGET_DOWN * jnp.power(eest1, 1.0 / k_f)
    h_dn = jnp.where(
        z_dn <= REJECT_BAND_CAP,
        h / jnp.where(z_dn > 0, z_dn, 1.0),
        jnp.asarray(REJECT_FLOOR_RATIO, dtype) * h,
    )

    force_drop = consfail_next > 2
    dn = active_dn & (force_drop | (h_dn > h_same))
    # the order-drop retry never grows past the (possibly halved) step
    dt_next = jnp.where(dn, jnp.minimum(h, h_dn), h_same)
    k_next = jnp.where(dn, k - 1, k).astype(jnp.int32)

    restart = (k_next == 1) & (consfail_next > 3)

    return RejectControl(
        dt_next=jax.lax.stop_gradient(dt_next),
        k_next=k_next,
        consfailcnt_next=consfail_next,
        nconsteps_next=jnp.zeros((), jnp.int32),
        restart_history=restart,
    )


__all__ = [
    "AcceptControl",
    "RejectControl",
    "accept_step_control",
    "reject_step_control",
    "TARGET_SAME",
    "TARGET_DOWN",
    "TARGET_UP",
    "GROWTH_BAND_FLOOR",
    "GROWTH_CAP",
    "QSTEADY_MIN",
    "QSTEADY_MAX",
    "REJECT_BAND_CAP",
    "REJECT_FLOOR_RATIO",
]
