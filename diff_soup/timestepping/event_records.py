"""Bounded event and terminal records for the solver suite.

A solve reports the events it detected in an [EventLog][] — a
fixed-capacity, jit- and vmap-safe record of event times, triggering root
indices, event kinds, crossing directions, epoch ownership, and pre/post-event
states — and its exact stopping point in a [TerminalRecord][]. Both are
functional pytrees: appending or writing returns a new record.

When a requested save time coincides with an event time the recorded
sample follows one convention, expressed by
[collision_sample_source][]: a nonterminal event's save slot holds the
post-affect state (the pre-affect state remains available from the event
log), and a terminal event's save slot holds the exact terminal state.
"""

from __future__ import annotations

import enum

import equinox as eqx
import jax
import jax.numpy as jnp


class SampleSource(enum.Enum):
    """Which sample a save slot receives when times coincide."""

    INTERPOLANT = "interpolant"
    POST_AFFECT = "post_affect"
    TERMINAL = "terminal"


def collision_sample_source(*, save_equals_event: bool, event_is_terminal: bool) -> SampleSource:
    """The sample a save slot receives when it coincides with an event time.

    A save time equal to a nonterminal event time records the post-affect
    state; equal to a terminal event time it records the exact terminal
    state; otherwise the slot holds the step interpolant's sample.
    """
    if not save_equals_event:
        return SampleSource.INTERPOLANT
    return SampleSource.TERMINAL if event_is_terminal else SampleSource.POST_AFFECT


class EventLog(eqx.Module):
    """Fixed-capacity record of detected events.

    Attributes:
        ts: Event times, shape ``(max_events,)``; unwritten slots are NaN.
        indices: Triggering root-function index per event, ``(max_events,)``.
        kinds: Event kind per record (discrete or continuous),
            ``(max_events,)``.
        directions: Crossing direction per event (−1, 0, +1),
            ``(max_events,)``.
        ys: Post-event states, ``(max_events, n_state)``. A terminating event
            has the same state in ``pre_ys`` and ``ys``.
        pre_ys: Projected state immediately before the affect,
            ``(max_events, n_state)``. This is the event's left limit.
        pre_epochs: Control epoch owning ``pre_ys``, or zero without an
            epoch-bearing control channel.
        post_epochs: Control epoch owning ``ys`` after an affect, or zero
            without an epoch-bearing control channel.
        count: Number of events recorded so far.
        overflow: ``True`` once an event arrived with the log full; the
            stored records are preserved and the surplus event is dropped.
    """

    ts: jax.Array
    indices: jax.Array
    kinds: jax.Array
    directions: jax.Array
    ys: jax.Array
    pre_ys: jax.Array
    pre_epochs: jax.Array
    post_epochs: jax.Array
    count: jax.Array
    overflow: jax.Array

    @classmethod
    def empty(cls, max_events: int, n_state: int, *, dtype=jnp.float64) -> "EventLog":
        """A log with capacity ``max_events`` and no recorded events."""
        if max_events <= 0:
            raise ValueError(
                "event-log capacity must be positive so a detected event has a "
                f"bounded record slot; got {max_events}."
            )
        return cls(
            ts=jnp.full((max_events,), jnp.nan, dtype=dtype),
            indices=jnp.full((max_events,), -1, dtype=jnp.int32),
            kinds=jnp.full((max_events,), -1, dtype=jnp.int32),
            directions=jnp.full((max_events,), -1, dtype=jnp.int32),
            ys=jnp.full((max_events, n_state), jnp.nan, dtype=dtype),
            pre_ys=jnp.full((max_events, n_state), jnp.nan, dtype=dtype),
            pre_epochs=jnp.full((max_events,), -1, dtype=jnp.int32),
            post_epochs=jnp.full((max_events,), -1, dtype=jnp.int32),
            count=jnp.zeros((), dtype=jnp.int32),
            overflow=jnp.zeros((), dtype=bool),
        )

    def append(
        self,
        *,
        t,
        index,
        direction,
        y,
        kind=-1,
        pre_y=None,
        pre_epoch=-1,
        post_epoch=-1,
    ) -> "EventLog":
        """Record one event, or set [overflow][diff_soup.timestepping.event_records.EventLog.overflow] when the log is full.

        Only ``t``, ``index``, ``direction``, and ``y`` are required; the
        metadata arguments default to their unset sentinels. When ``pre_y`` is
        omitted, ``y`` is used for both limits.
        """
        capacity = self.ts.shape[0]
        has_room = self.count < capacity
        slot = jnp.minimum(self.count, capacity - 1)
        pre_y = y if pre_y is None else pre_y

        def write(buffer, value):
            return jnp.where(has_room, buffer.at[slot].set(value), buffer)

        return EventLog(
            ts=write(self.ts, jnp.asarray(t, dtype=self.ts.dtype)),
            indices=write(self.indices, jnp.asarray(index, dtype=jnp.int32)),
            kinds=write(self.kinds, jnp.asarray(kind, dtype=jnp.int32)),
            directions=write(self.directions, jnp.asarray(direction, dtype=jnp.int32)),
            ys=write(self.ys, jnp.asarray(y, dtype=self.ys.dtype)),
            pre_ys=write(self.pre_ys, jnp.asarray(pre_y, dtype=self.pre_ys.dtype)),
            pre_epochs=write(self.pre_epochs, jnp.asarray(pre_epoch, dtype=jnp.int32)),
            post_epochs=write(self.post_epochs, jnp.asarray(post_epoch, dtype=jnp.int32)),
            count=self.count + has_room.astype(jnp.int32),
            overflow=jnp.logical_or(self.overflow, jnp.logical_not(has_room)),
        )


class TerminalRecord(eqx.Module):
    """The exact stopping point of a solve.

    Attributes:
        t: The terminal time.
        y: The terminal state, shape ``(n_state,)``.
        lam: The terminal bordered-global values, shape ``(n_globals,)``.
        result_code: The solve's result code at termination.
        control_epoch: Control epoch owning the exact terminal state.
        event_index: Declaration-order event index, or ``-1`` when the solve
            did not terminate through an event.
        event_kind: Event kind code, or ``-1`` without a terminal event.
        event_direction: Crossing direction, or ``-1`` without a terminal
            event.
        valid: ``False`` until
            [write][diff_soup.timestepping.event_records.TerminalRecord.write]
            records a terminal point.
    """

    t: jax.Array
    y: jax.Array
    lam: jax.Array
    result_code: jax.Array
    control_epoch: jax.Array
    event_index: jax.Array
    event_kind: jax.Array
    event_direction: jax.Array
    valid: jax.Array

    @classmethod
    def empty(cls, n_state: int, n_globals: int, *, dtype=jnp.float64) -> "TerminalRecord":
        """An invalid record with NaN payloads, to be written at termination."""
        return cls(
            t=jnp.asarray(jnp.nan, dtype=dtype),
            y=jnp.full((n_state,), jnp.nan, dtype=dtype),
            lam=jnp.full((n_globals,), jnp.nan, dtype=dtype),
            result_code=jnp.asarray(-1, dtype=jnp.int32),
            control_epoch=jnp.asarray(-1, dtype=jnp.int32),
            event_index=jnp.asarray(-1, dtype=jnp.int32),
            event_kind=jnp.asarray(-1, dtype=jnp.int32),
            event_direction=jnp.asarray(-1, dtype=jnp.int32),
            valid=jnp.zeros((), dtype=bool),
        )

    def write(
        self,
        *,
        t,
        y,
        lam,
        result_code,
        control_epoch=-1,
        event_index=-1,
        event_kind=-1,
        event_direction=-1,
    ) -> "TerminalRecord":
        """A valid record holding the exact terminal time, state, and result."""
        return TerminalRecord(
            t=jnp.asarray(t, dtype=self.t.dtype),
            y=jnp.asarray(y, dtype=self.y.dtype),
            lam=jnp.asarray(lam, dtype=self.lam.dtype),
            result_code=jnp.asarray(result_code, dtype=jnp.int32),
            control_epoch=jnp.asarray(control_epoch, dtype=jnp.int32),
            event_index=jnp.asarray(event_index, dtype=jnp.int32),
            event_kind=jnp.asarray(event_kind, dtype=jnp.int32),
            event_direction=jnp.asarray(event_direction, dtype=jnp.int32),
            valid=jnp.ones((), dtype=bool),
        )
