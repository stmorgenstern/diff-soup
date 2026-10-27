"""The runtime precision-rung policy behind ``Precision(solve_mode="adaptive")``.

An adaptive solve carries a traced rung — an index into the ladder of
concrete linear-solve modes (``fp32_ir`` then ``fp64``) — on the engine
carry, together with the hysteresis that keeps it stable:

* **Promotion** raises the rung when a failure is attributed to precision:
  the linear refinement escalated to its fp64 fallback, the Newton increments
  stagnated above tolerance, or the corrector converged by increment while
  its residual never dropped (false convergence). A promotion is sticky: it
  pins the higher rung for ``PIN_STEPS`` accepted steps.
* **Probing demotion** re-tries the lower rung when the pin expires; a probe
  that immediately re-promotes doubles the next pin, and a probe that
  survives an accepted step resets the doubling.
* **Diagnostics**: every promotion increments a per-cause counter, surfaced
  through the solver statistics.

Everything here is pure carry logic — traced scalars in, a new record out —
so the policy is testable without a solver in the loop.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# A Newton increment whose norm moved by less than this fraction between
# consecutive unconverged iterations reads as stagnant.
STAGNATION_BAND: float = 0.05

# Consecutive stagnant iterations before the detector latches.
STAGNATION_LATCH: int = 2

# Accepted steps a promotion pins the higher rung for.
PIN_STEPS: int = 20

# Promotion-cause indices in the promotion-counter vector.
CAUSE_GATE_MISS: int = 0  # reserved: a direct-fp32 bottom rung's residual gate
CAUSE_ESCALATION: int = 1  # the linear refinement fell back to fp64
CAUSE_STAGNATION: int = 2  # Newton increments stagnated above tolerance
CAUSE_FALSE_CONVERGENCE: int = 3  # converged by increment, residual never dropped
N_CAUSES: int = 4

# The terminal rung of the two-rung ladder (0 = fp32_ir, 1 = fp64).
TOP_RUNG: int = 1


class RungState(NamedTuple):
    """The adaptive policy's traced state, carried across step attempts.

    Attributes:
        rung: The active rung (int32; an index into the concrete-mode ladder).
        entered: Whether the previous transition changed the rung — the newly
            active rung's held factor must be rebuilt on the next attempt.
        pin_left: Accepted-step countdown holding an elevated rung.
        probe_boost: Pin multiplier; doubled by a failed probing demotion,
            reset by a probe that survives an accepted step.
        probing: Whether the current rung is a probing demotion.
        promo_counts: Per-cause promotion counters (int32, ``N_CAUSES``).
    """

    rung: jax.Array
    entered: jax.Array
    pin_left: jax.Array
    probe_boost: jax.Array
    probing: jax.Array
    promo_counts: jax.Array


def initial_rung_state() -> RungState:
    """The policy's starting state: the cheap rung, nothing pinned."""
    return RungState(
        rung=jnp.asarray(0, jnp.int32),
        entered=jnp.asarray(False),
        pin_left=jnp.asarray(0, jnp.int32),
        probe_boost=jnp.asarray(1, jnp.int32),
        probing=jnp.asarray(False),
        promo_counts=jnp.zeros((N_CAUSES,), jnp.int32),
    )


def stagnation_flag(
    dn: jax.Array, dn_prev: jax.Array, converged: jax.Array, band: float = STAGNATION_BAND
) -> jax.Array:
    """One iteration's stagnation predicate.

    True when the increment norm moved by at most ``band`` relative to the
    previous iteration while the test has not converged. An infinite previous
    norm (the first iteration) never reads as stagnant.

    Args:
        dn: This iteration's Newton-update norm.
        dn_prev: The previous iteration's update norm.
        converged: This iteration's convergence-test outcome.
        band: The relative freeze band.

    Returns:
        A traced boolean.
    """
    frozen = jnp.abs(dn - dn_prev) <= band * dn
    return frozen & jnp.isfinite(dn_prev) & jnp.logical_not(converged)


def advance_rung_state(state: RungState, *, cause: jax.Array, accepted: jax.Array) -> RungState:
    """One attempt-outcome transition of the rung policy.

    Args:
        state: The carried [RungState][diff_soup.timestepping.vsvo.common.adaptive.RungState].
        cause: Traced int32 promotion cause — one of the ``CAUSE_*`` indices,
            or ``-1`` for no promotion. Callers gate causes on
            ``rung < TOP_RUNG`` (the terminal rung never promotes).
        accepted: Whether the step attempt was accepted.

    Returns:
        The next [RungState][diff_soup.timestepping.vsvo.common.adaptive.RungState].
    """
    promote = cause >= 0
    hold = jnp.logical_not(promote)

    # Promotion: raise (clamped), pin for the boosted window, count the cause.
    # A promotion while probing means the probe failed: double the next pin.
    boost = jnp.where(promote & state.probing, state.probe_boost * 2, state.probe_boost)
    rung = jnp.where(promote, jnp.minimum(state.rung + 1, TOP_RUNG), state.rung)
    pin_left = jnp.where(promote, jnp.asarray(PIN_STEPS, jnp.int32) * boost, state.pin_left)
    probing = jnp.where(promote, jnp.asarray(False), state.probing)
    cause_ix = jnp.clip(cause, 0, N_CAUSES - 1)
    promo_counts = jnp.where(
        promote,
        state.promo_counts.at[cause_ix].add(1),
        state.promo_counts,
    )

    # An accepted step with no promotion: count the pin down; a surviving
    # probe ends and resets the doubling.
    tick = accepted & hold
    pin_left = jnp.where(tick & (rung > 0), jnp.maximum(pin_left - 1, 0), pin_left)
    boost = jnp.where(tick & probing, jnp.asarray(1, jnp.int32), boost)
    probing = jnp.where(tick, jnp.asarray(False), probing)

    # Pin expiry on an elevated rung: probe the rung below.
    demote = tick & (rung > 0) & (pin_left == 0)
    rung = jnp.where(demote, rung - 1, rung)
    probing = jnp.where(demote, jnp.asarray(True), probing)

    return RungState(
        rung=rung.astype(jnp.int32),
        entered=rung != state.rung,
        pin_left=pin_left.astype(jnp.int32),
        probe_boost=boost.astype(jnp.int32),
        probing=probing,
        promo_counts=promo_counts,
    )


__all__ = [
    "RungState",
    "initial_rung_state",
    "stagnation_flag",
    "advance_rung_state",
    "STAGNATION_BAND",
    "STAGNATION_LATCH",
    "PIN_STEPS",
    "CAUSE_GATE_MISS",
    "CAUSE_ESCALATION",
    "CAUSE_STAGNATION",
    "CAUSE_FALSE_CONVERGENCE",
    "N_CAUSES",
    "TOP_RUNG",
]
