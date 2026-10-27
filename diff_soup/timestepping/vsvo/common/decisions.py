"""Recorded step decisions for the checkpointed reverse replay.

A reverse-mode gradient through the checkpointed step loop replays segments of
the trajectory from saved loop states and differentiates each replayed step.
The replay runs the step body inside a different compiled program than the
forward sweep, so its floating-point results can differ in the last bits. The
adaptive step decisions -- accept or reject, the next step size, the next
order -- are thresholds on error norms and on the nonlinear solver's
convergence, and in single precision a last-bit difference in a norm can move
a decision. A replay that decides differently integrates a different trajectory
from the one the forward sweep took, and its gradient is silently wrong.

The forward sweep therefore records, for every attempt, the decision inputs the
engine branches on (the convergence flag and the error-test norms, see each
engine's ``decision_template``) together with the attempt's resulting time and
next step size. The replay reads the record back and uses the recorded
decision inputs in place of the recomputed ones, so every decision downstream
of them is the one the forward sweep made. Recorded floating inputs keep the
recomputed value's derivative, so the gradient is unchanged wherever the
decision is.

A step size that is not a function of recorded inputs is recorded itself: an
event affect restarts the engine, whose entry step comes from a search over the
affected state, so each attempt also records the entry step of a restart it
triggered and a replayed restart enters with that step instead of searching
again.

The record rides the checkpointed loop's write-only buffer channel: the forward
sweep writes one row per attempt, and the backward pass hands the replay the
completed buffers. A row that was written marks the attempt as a replay.
After a replayed attempt the driver compares its time and next step size with
the recorded ones and raises when they differ, so a replay that still departs
from the forward trajectory fails loudly instead of returning a wrong gradient.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "DecisionTape",
    "init_decision_tape",
    "replay_decisions",
    "REPLAY_DIVERGED_MESSAGE",
]

REPLAY_DIVERGED_MESSAGE = (
    "The reverse-mode replay of a checkpointed solve departed from the forward "
    "trajectory: a replayed step reached a different time or chose a different "
    "next step size than the forward sweep recorded, so the gradient would not "
    "be the gradient of the forward solution. The step engine's decision record "
    "does not capture everything its steps branch on; please report this with the "
    "solver, its tolerances and the problem size."
)


class DecisionTape(eqx.Module):
    """One row per step attempt of the forward sweep.

    Attributes:
        written: Whether the forward sweep has written the row.
        t_new: The attempt's resulting time.
        h_new: The step size the attempt chose for the next attempt.
        h_restart: The entry step of the engine restart an event affect
            triggered at the attempt; a replayed restart enters with it. On an
            attempt without a restart the row holds the step size of the carry
            handed on, a finite value no replay reads.
        values: The engine's decision inputs, a pytree of arrays with a
            leading attempt axis.
    """

    # Inside the checkpointed loop these fields are the loop's write-only
    # buffers rather than arrays, so they are not annotated as arrays.
    written: Any
    t_new: Any
    h_new: Any
    h_restart: Any
    values: Any


def _wide(dtype):
    """The storage dtype of a recorded scalar: floats widen to the default float."""
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.result_type(float)
    return dtype


def init_decision_tape(template: Any, max_steps: int) -> DecisionTape:
    """An empty tape of ``max_steps`` rows for decision inputs shaped like ``template``.

    Floating entries are stored at the default float width, so a float32
    decision input round-trips exactly.
    """
    wide = _wide(jnp.float32)
    return DecisionTape(
        written=jnp.zeros((max_steps,), jnp.bool_),
        t_new=jnp.zeros((max_steps,), wide),
        h_new=jnp.zeros((max_steps,), wide),
        h_restart=jnp.zeros((max_steps,), wide),
        values=jax.tree_util.tree_map(
            lambda leaf: jnp.zeros((max_steps,) + jnp.shape(leaf), _wide(jnp.asarray(leaf).dtype)),
            template,
        ),
    )


def replay_decisions(decision: Any, computed: Any) -> Any:
    """The decision inputs an attempt uses: recorded on a replay, computed otherwise.

    ``decision`` is ``None`` outside a recorded loop, else ``(replay,
    recorded)``. On a replay a floating input takes the recorded value with the
    recomputed input's derivative, and a boolean or integer input takes the
    recorded value.
    """
    if decision is None:
        return computed
    replay, recorded = decision

    def pick(value, stored):
        value = jnp.asarray(value)
        stored = jnp.asarray(stored).astype(value.dtype)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            # Exactly the recorded value; the derivative of the recomputed one.
            # The zero is masked so a non-finite recomputed value cannot turn
            # the recorded one into NaN.
            zero = value - jax.lax.stop_gradient(value)
            stored = stored + jnp.where(jnp.isfinite(value), zero, jnp.zeros_like(zero))
        return jnp.where(replay, stored, value)

    return jax.tree_util.tree_map(pick, computed, recorded)
