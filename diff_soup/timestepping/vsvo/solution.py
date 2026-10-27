"""The solution surface shared across the scalar VSVO solver suite.

This module owns the single public [Solution][diff_soup.timestepping.vsvo.solution.Solution] and [SolverStats][diff_soup.timestepping.vsvo.solution.SolverStats]
records that every native core and engine adapter returns, plus the packagers
that build them.

* [SolverStats][diff_soup.timestepping.vsvo.solution.SolverStats] — one frozen counter record covering accept/reject, the
  Newton/linear/residual work, the box active-set bookkeeping, the time reached,
  and the terminal status.
* [Solution][diff_soup.timestepping.vsvo.solution.Solution] — an immutable output pytree carrying the trajectory
  (``ts``/``ys``/``lambdas``), the stats, the terminal [RESULT][diff_soup._solution.RESULT], dense
  output, and the diagnostic state. ``alg`` (the solver object) rides as static
  metadata and adds no array leaves; ``prob`` (the problem object) is a runtime
  pytree field because a lowered problem carries array leaves, which cannot live
  behind ``static=True``. ``ok`` is a traced array predicate, safe under
  ``jax.jit`` / ``jax.vmap``.
* [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory] — the
  integrator's state at the end of a solve, from which a later solve of the
  same problem continues its steps.
* [SavedOutputs][diff_soup.timestepping.vsvo.solution.SavedOutputs] — the
  outputs named by ``SaveAt(outputs=...)``, by name in request order.
* [build_solution][diff_soup.timestepping.vsvo.solution.build_solution] — a low-level keyword packager.

The single [RESULT][diff_soup._solution.RESULT] enum (success iff ``SUCCESS`` or ``EVENT_STOP``) is
reused from [`diff_soup._solution`][diff_soup._solution].
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from diff_soup._solution import RESULT, is_successful
from diff_soup.utils.lanes import lane_cond


class SolverStats(eqx.Module):
    """Counters and settings of one solve, carried on `Solution.stats`.

    Counters that an engine does not track stay zero.

    Attributes:
        accepts: Accepted steps.
        rejects: Rejected step attempts.
        trial_steps: Step attempts, accepted or not.
        newton_iters: Newton iterations.
        jac_refreshes: Jacobian evaluations.
        factorizations: Factorizations of the iteration matrix.
        linear_solves: Linear solves.
        rhs_evals: Right-hand-side evaluations.
        residual_evals: Residual evaluations.
        active_set_iters: Reserved for an active-set corrector, zero on the current engines.
        active_set_changes: Reserved for an active-set corrector, zero on the current engines.
        ic_alg_residual_norm: Algebraic residual left by consistent initialization, zero when none ran.
        t_reached: Time the solve reached.
        status: The `RESULT` of the solve.
        precision_promotions: Promotions of an adaptive solve to full precision, one counter per cause.
        newton_failures: Corrector failures.
        saveat_interpolations: Interpolant evaluations made to fill save times.
        ic_solves: 1 when consistent initialization was requested, else 0.
        event_solves: Affect events applied.
        ir_refinement_steps: Iterative-refinement steps of the reduced-precision solves.
        fp64_fallbacks: Fallbacks from a float32 factor to a float64 one.
        first_step: Size of the first step attempt.
        last_step: Size of the step the next attempt would take.
        linear_backend: Name of the linear backend the solve ran on, `"dense"`, `"klu"`, `"superlu"` or `"cudss"`, or `None`.
        adjoint: Class name of the differentiation strategy.
        adjoint_checkpoints: Configured checkpoint count of the strategy, or `None`.
        linear_solve_mode: Linear solve mode of the precision policy.
    """

    accepts: Array
    rejects: Array
    trial_steps: Array
    newton_iters: Array
    jac_refreshes: Array
    factorizations: Array
    linear_solves: Array
    rhs_evals: Array
    residual_evals: Array
    active_set_iters: Array
    active_set_changes: Array
    ic_alg_residual_norm: Array
    t_reached: Array
    status: RESULT
    precision_promotions: Array = eqx.field(default_factory=lambda: jnp.zeros((4,), jnp.int32))
    multirhs_solves: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    newton_failures: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    multirhs_columns: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    du_b_builds: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    schur_builds: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    saveat_interpolations: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    ic_solves: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    event_solves: Array = eqx.field(default_factory=lambda: jnp.asarray(0, jnp.int32))
    # fp64 scalars, not int32: forward-mode rules need an ordinary zero tangent
    # for counters that ride reduced-precision solve outputs.
    ir_refinement_steps: Array = eqx.field(default_factory=lambda: jnp.zeros((), jnp.float64))
    fp64_fallbacks: Array = eqx.field(default_factory=lambda: jnp.zeros((), jnp.float64))
    first_step: Array = eqx.field(default_factory=lambda: jnp.zeros((), jnp.float64))
    last_step: Array = eqx.field(default_factory=lambda: jnp.zeros((), jnp.float64))
    linear_backend: Optional[str] = eqx.field(static=True, default=None)
    adjoint: Optional[str] = eqx.field(static=True, default=None)
    adjoint_checkpoints: Optional[int] = eqx.field(static=True, default=None)
    linear_solve_mode: Optional[str] = eqx.field(static=True, default=None)

    @classmethod
    def zeros(cls) -> "SolverStats":
        """Build an all-zero record with an ``UNKNOWN`` terminal status.

        Integer counters start at ``0`` (int32), ``ic_alg_residual_norm`` and
        ``t_reached`` at ``0.0``, and ``status`` at `RESULT.UNKNOWN`.
        """
        z = jnp.asarray(0, dtype=jnp.int32)
        return cls(
            accepts=z,
            rejects=z,
            trial_steps=z,
            newton_iters=z,
            jac_refreshes=z,
            factorizations=z,
            linear_solves=z,
            rhs_evals=z,
            residual_evals=z,
            active_set_iters=z,
            active_set_changes=z,
            ic_alg_residual_norm=jnp.asarray(0.0),
            t_reached=jnp.asarray(0.0),
            status=RESULT.UNKNOWN,
        )


class SolveHistory(eqx.Module):
    """The integrator's state at the end of a solve, from which a later solve continues.

    A solve asked for it (``history=True``) returns it as ``Solution.history``.
    A later solve of the same problem by the same solver, started at this
    solve's end time from its end state, continues from it when it is passed
    as ``history=``: it takes the step size, the order, the history of past
    steps, the error and corrector memory and the iteration matrix in effect
    from the record instead of starting afresh. The two solves together then
    take the steps one solve over both spans would take, except that the
    first lands exactly on its end time::

        first = ds.pdesolve(problem, solver, (0.0, 1.0), y0, history=True)
        second = ds.pdesolve(
            problem, solver, (1.0, 2.0), first.terminal.y, history=first.history
        )

    The differential unknowns come from the later solve's initial state; the
    algebraic unknowns and everything else from the record. A later solve
    whose control changes what the algebraic rows demand at that state (a
    jump of a target, a new operating mode) meets a kink in the solution: it
    starts afresh at order one from the step the record carries instead.
    Derivatives flow through the record as they flow between two steps of one
    solve.

    Attributes:
        continues: Boolean scalar: whether a solve given this record continues
            from it. It is set on a solve that reached its end time; a solve
            given a record whose flag is not set starts as it does without one.
            It may be traced, so one compiled solve serves both cases.
        state: The record of the solver that produced it, opaque to every
            other solver.
    """

    continues: Array
    state: Any


class SavedOutputs(eqx.Module, Mapping):
    """The outputs a solve stored at its save times, by name, in request order.

    A read-only mapping (``outputs["voltage"]``, ``dict(outputs)``,
    iteration in the order the names were requested) and a pytree whose
    leaves are the stored arrays, so it passes through ``jax.jit`` and
    ``jax.vmap`` and keeps its order.

    Attributes:
        names: The output names, in request order.
        samples: One array per name; the leading axis is the save capacity
            (after a batch axis under ``jax.vmap``).
    """

    names: tuple[str, ...] = eqx.field(static=True)
    samples: tuple[Any, ...]

    def __getitem__(self, name: str) -> Any:
        """The stored samples of ``name``."""
        try:
            return self.samples[self.names.index(name)]
        except ValueError:
            raise KeyError(name) from None

    def __iter__(self):
        """Iterate over the names in request order."""
        return iter(self.names)

    def __len__(self) -> int:
        """The number of stored outputs."""
        return len(self.names)


class Solution(eqx.Module):
    """The record every solver returns, with the saved samples, the outcome, and the statistics.

    A solve never raises on non-convergence, so read `ok` or `result` before the arrays. Buffers have the size fixed at compile time, and rows past `valid_count` are unfilled.

    Attributes:
        ts: Save times, `NaN` in unfilled rows.
        ys: State at each save time, or `None` when `SaveAt(outputs=...)` leaves out `"state"`.
        lambdas: Global unknowns at each save time, or `None`.
        valid_mask: Boolean per row, `True` where the row was filled.
        valid_count: Number of filled rows.
        result: The `RESULT` of the solve.
        stats: The `SolverStats` of the solve.
        tolerances: The `ResolvedTolerances` of a solve given `tolerances=`, else `None`.
        event_log: Record of the events fired during the solve, or `None`.
        terminal: State where the solve stopped, with its time and owning control epoch.
        control_state: Final `ControlState` of a solve whose affect events update it, else `None`.
        control_epochs: Control epoch owning each saved row, `-1` in unfilled rows.
        interpolation: Dense interpolant from `SaveAt(dense=True)`, else `None`.
        outputs: Mapping from each name in `SaveAt(outputs=...)` to its samples, else `None`.
    """

    ts: Optional[Array] = None
    ys: Optional[Any] = None
    lambdas: Optional[Any] = None
    valid_mask: Optional[Array] = None
    valid_count: Optional[Array] = None
    control_epochs: Optional[Array] = None
    event_indices: Optional[Array] = None
    event_kinds: Optional[Array] = None
    event_directions: Optional[Array] = None
    result_codes: Optional[Array] = None
    stats: Optional[SolverStats] = None
    result: RESULT = eqx.field(default_factory=lambda: RESULT.UNKNOWN)
    info: Any = None
    interpolation: Any = None
    trace: Any = None
    control_state: Any = None
    event_mask: Any = None
    tolerances: Any = None
    event_log: Any = None
    terminal: Any = None
    alg: Optional[Any] = eqx.field(default=None, static=True)
    # Runtime field, NOT static: a lowered problem carries array leaves, and
    # arrays cannot be stored behind ``static=True``.
    prob: Optional[Any] = None
    history: Optional[SolveHistory] = None
    outputs: Optional[SavedOutputs] = None

    @property
    def ok(self) -> Array:
        """Whether the solve succeeded, meaning `result` is `SUCCESS` or `EVENT_STOP`."""
        return is_successful(self.result)

    def evaluate(self, t: Any) -> Any:
        """Evaluate the dense output, a piecewise cubic through the accepted steps, at `t`.

        It needs `SaveAt(dense=True)`. A time outside the solved span is clamped to its nearest end.

        Args:
            t: A time or a one-dimensional array of times.

        Returns:
            The state at `t`, with one row per time for an array.

        Raises:
            RuntimeError: If dense output was not requested.
        """
        if self.interpolation is None:
            raise RuntimeError("dense output was not requested for this solve")
        return self.interpolation.evaluate(t)

    def trimmed(self, lane: Any = None) -> "Solution":
        """Return a copy without the unfilled rows of the save buffers.

        It runs on the host only. Inside `jit` or `vmap`, use `valid_mask` and `valid_count` instead.

        Args:
            lane: Batch index to select from a batched solution, omitted for an unbatched one.

        Returns:
            A `Solution` whose buffers hold `valid_count` rows.

        Raises:
            RuntimeError: If called on traced values.
        """
        if self.valid_mask is None or self.valid_count is None:
            raise RuntimeError("this solution has no fixed-buffer validity metadata")
        leaves = jax.tree_util.tree_leaves((self.valid_mask, self.valid_count))
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            raise RuntimeError(
                "Solution.trimmed() is host-only; use valid_mask and valid_count "
                "inside jit or vmap."
            )
        mask_host = jnp.asarray(self.valid_mask)
        if mask_host.ndim == 1:
            if lane is not None:
                raise ValueError("lane must be omitted when trimming an unbatched solution")
            selected = self
        elif mask_host.ndim == 2:
            if lane is None:
                raise ValueError(
                    "a batched solution requires an explicit lane in trimmed(lane=...)"
                )
            try:
                lane_index = operator.index(lane)
            except TypeError as exc:
                raise ValueError(f"lane must be an integer batch index; got {lane!r}") from exc
            batch_size = int(mask_host.shape[0])
            if not 0 <= lane_index < batch_size:
                raise ValueError(f"lane must be in [0, {batch_size}); got {lane}")

            def take_lane(tree):
                return jax.tree_util.tree_map(
                    lambda leaf: (
                        leaf[lane_index]
                        if eqx.is_array(leaf) and leaf.ndim > 0 and int(leaf.shape[0]) == batch_size
                        else leaf
                    ),
                    tree,
                )

            selected = Solution(
                ts=take_lane(self.ts),
                ys=take_lane(self.ys),
                lambdas=take_lane(self.lambdas),
                valid_mask=take_lane(self.valid_mask),
                valid_count=take_lane(self.valid_count),
                control_epochs=take_lane(self.control_epochs),
                event_indices=take_lane(self.event_indices),
                event_kinds=take_lane(self.event_kinds),
                event_directions=take_lane(self.event_directions),
                result_codes=take_lane(self.result_codes),
                stats=take_lane(self.stats),
                result=take_lane(self.result),
                info=take_lane(self.info),
                interpolation=take_lane(self.interpolation),
                trace=take_lane(self.trace),
                control_state=take_lane(self.control_state),
                event_mask=take_lane(self.event_mask),
                tolerances=take_lane(self.tolerances),
                event_log=take_lane(self.event_log),
                terminal=take_lane(self.terminal),
                alg=self.alg,
                prob=take_lane(self.prob),
                history=take_lane(self.history),
                outputs=take_lane(self.outputs),
            )
        else:
            raise ValueError(
                "trimmed() supports unbatched or one-axis batched solutions; "
                f"valid_mask has rank {mask_host.ndim}"
            )

        count = int(jnp.asarray(selected.valid_count))
        capacity = int(jnp.asarray(selected.valid_mask).shape[0])

        def trim_aligned(tree):
            return jax.tree_util.tree_map(
                lambda leaf: (
                    leaf[:count]
                    if eqx.is_array(leaf) and leaf.ndim > 0 and int(leaf.shape[0]) == capacity
                    else leaf
                ),
                tree,
            )

        return Solution(
            ts=trim_aligned(selected.ts),
            ys=trim_aligned(selected.ys),
            lambdas=trim_aligned(selected.lambdas),
            valid_mask=jnp.ones((count,), dtype=bool),
            valid_count=jnp.asarray(count, dtype=jnp.int32),
            control_epochs=trim_aligned(selected.control_epochs),
            event_indices=trim_aligned(selected.event_indices),
            event_kinds=trim_aligned(selected.event_kinds),
            event_directions=trim_aligned(selected.event_directions),
            result_codes=trim_aligned(selected.result_codes),
            stats=selected.stats,
            result=selected.result,
            info=selected.info,
            interpolation=selected.interpolation,
            trace=selected.trace,
            control_state=selected.control_state,
            event_mask=selected.event_mask,
            tolerances=selected.tolerances,
            event_log=selected.event_log,
            terminal=selected.terminal,
            alg=selected.alg,
            prob=selected.prob,
            history=selected.history,
            outputs=trim_aligned(selected.outputs),
        )


def init_saveat_outputs(plan, ts_grid, u0, lam0, t0, *, capacity, emit):
    """Allocate the output buffers and the time buffer before the step loop.

    Every output of ``plan`` gets one buffer of shape ``(capacity + 1, *shape)``:
    one row per save slot plus a final discard row that masked writes target
    (writes to it never reach the solution). A query time at or before the
    start time holds the outputs of the initial state; every other row holds a
    NaN sentinel that a later sample overwrites. A sentinel that survives to
    the end marks a query time the solve never reached (for example beyond a
    terminating event, or past where the step budget ran out). With the
    default plan the two buffers are the state and bordered-global grids.

    Args:
        plan: The [OutputPlan][diff_soup.timestepping.saveat.OutputPlan].
        ts_grid: The requested save times, shape ``(n_save,)``.
        u0: The initial state, shape ``(n_u,)``.
        lam0: The initial bordered-global values, shape ``(n_lam,)``.
        t0: The integration start time.
        capacity: Fixed output capacity (``n_save``, plus one when a terminal
            reserve is planned).
        emit: ``emit(t, u, lam) -> tuple`` evaluating the outputs at one state;
            ``lam`` is ``None`` when no output reads the bordered globals.

    Returns:
        ``(ts_output, buffers)``: the time buffer, shape ``(capacity,)``, with
        the initial query times in its prefix and ``NaN`` after, and the tuple
        of output buffers in plan order.
    """
    n_query = int(ts_grid.shape[0])
    capacity = int(capacity)
    if capacity < n_query:
        raise ValueError(
            f"save-buffer capacity ({capacity}) cannot be smaller than query count ({n_query})"
        )
    initial_query = ts_grid <= t0
    initial = jnp.zeros((capacity + 1,), dtype=bool).at[:n_query].set(initial_query)
    rows = emit(t0, u0, lam0 if plan.needs_lam else None)
    buffers = []
    for spec, row in zip(plan.specs, rows):
        row = jnp.asarray(row, dtype=u0.dtype).reshape(spec.shape)
        mask = initial.reshape((capacity + 1,) + (1,) * len(spec.shape))
        sentinel = jnp.full((capacity + 1,) + tuple(spec.shape), jnp.nan, dtype=u0.dtype)
        buffers.append(jnp.where(mask, row[None], sentinel))
    ts_output = jnp.full((capacity,), jnp.nan, dtype=ts_grid.dtype)
    ts_output = ts_output.at[:n_query].set(jnp.where(initial_query, ts_grid, jnp.nan))
    return ts_output, tuple(buffers)


def fill_saveat_times(query_ts, output_ts, filled_count):
    """Materialize the reached requested-time prefix in an output buffer.

    Args:
        query_ts: The sorted, deduplicated requested grid, shape ``(N,)``.
        output_ts: Fixed-capacity time buffer, shape ``(M,)`` with ``M >= N``.
        filled_count: Number of requested queries reached so far.

    Returns:
        ``output_ts`` with ``query_ts[:filled_count]`` in its prefix and all
        untouched slots preserved. The optional terminal reserve is never
        populated by this helper.
    """
    n_query = int(query_ts.shape[0])
    capacity = int(output_ts.shape[0])
    if capacity < n_query:
        raise ValueError(
            f"save-buffer capacity ({capacity}) cannot be smaller than query count ({n_query})"
        )
    candidate = jnp.full((capacity,), jnp.nan, dtype=output_ts.dtype)
    candidate = candidate.at[:n_query].set(query_ts.astype(output_ts.dtype))
    reached = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(
        jnp.asarray(filled_count, dtype=jnp.int32), jnp.asarray(n_query, dtype=jnp.int32)
    )
    return jnp.where(reached, candidate, output_ts)


def terminal_sample_slot(output_ts, valid_count, t, root_tolerance):
    """Resolve the compact-prefix slot for an exact terminal sample.

    A terminal time within ``root_tolerance`` of the last valid requested
    sample reuses that sample's slot. Otherwise it appends at ``valid_count``.
    The returned ``has_room`` is false only when the caller failed to allocate
    the structural terminal reserve.

    Returns:
        ``(slot, collides, has_room)`` as traced scalar arrays.
    """
    capacity = int(output_ts.shape[0])
    count = jnp.asarray(valid_count, dtype=jnp.int32)
    previous = jnp.clip(count - 1, 0, max(capacity - 1, 0))
    previous_t = output_ts[previous]
    collides = jnp.logical_and(
        count > 0,
        jnp.abs(previous_t - jnp.asarray(t, dtype=output_ts.dtype))
        <= jnp.asarray(root_tolerance, dtype=output_ts.dtype),
    )
    slot = jnp.where(collides, previous, count)
    has_room = jnp.logical_or(collides, count < capacity)
    return slot.astype(jnp.int32), collides, has_room


# The output buffers may be write-only loop buffers (the checkpointed reverse
# loop keeps them out of its checkpoints), which allows every slot to be
# written at most once and never read inside the loop. The latest sample may
# still be replaced (an exact terminal sample at the time of the last
# requested one), so it is held back in a small ``pending`` record of the loop
# carry, ``(slot, rows)``, and written to its slot once a later sample exists
# or the loop has finished. ``pending=None`` writes every sample directly,
# for output buffers that ride the ordinary carry.


def hold_pending_sample(buffers, valid_count):
    """Move the last valid row of each buffer into a pending record.

    Used once, before the loop: the row at ``valid_count - 1`` (if any) is
    returned as the pending record and its buffer slot reset to the NaN
    sentinel, so the loop writes it at most once.

    Returns:
        ``(buffers, pending)`` with ``pending = (slot, rows)``; ``slot`` is
        ``-1`` when there is no valid row.
    """
    discard = int(buffers[0].shape[0]) - 1
    slot = jnp.asarray(valid_count, jnp.int32) - 1
    safe = jnp.clip(slot, 0, max(discard - 1, 0))
    rows = tuple(buffer[safe] for buffer in buffers)
    target = jnp.where(slot >= 0, slot, discard)
    buffers = tuple(buffer.at[target].set(jnp.nan) for buffer in buffers)
    return buffers, (slot, rows)


def commit_pending_sample(buffers, pending):
    """Write a pending record into its slot (after the loop)."""
    if pending is None:
        return buffers
    slot, rows = pending
    discard = int(buffers[0].shape[0]) - 1
    target = jnp.where(slot >= 0, slot, discard)
    return tuple(buffer.at[target].set(row) for buffer, row in zip(buffers, rows))


def place_saveat_sample(buffers, pending, slot, rows, write):
    """Write one sample's rows at ``slot`` when ``write`` holds.

    Without a pending record the rows go straight to the slot. With one, a
    sample at the pending slot replaces the pending rows, and a sample at a
    later slot writes the pending rows out and becomes the new pending record.

    Returns:
        ``(buffers, pending)``.
    """
    discard = int(buffers[0].shape[0]) - 1
    slot = jnp.asarray(slot, jnp.int32)
    if pending is None:
        target = jnp.where(write, slot, discard)
        return tuple(b.at[target].set(r) for b, r in zip(buffers, rows)), None
    pending_slot, pending_rows = pending
    flush = jnp.logical_and(jnp.logical_and(write, slot != pending_slot), pending_slot >= 0)
    target = jnp.where(flush, pending_slot, discard)
    buffers = tuple(b.at[target].set(r) for b, r in zip(buffers, pending_rows))
    pending_rows = tuple(
        jnp.where(write, jnp.asarray(r, p.dtype), p) for r, p in zip(rows, pending_rows)
    )
    return buffers, (jnp.where(write, slot, pending_slot), pending_rows)


def _write_saveat_rows(buffers, pending, idx, rows, covered, last, any_new):
    """Write the covered rows ``rows[covered]`` at slots ``idx[covered]``.

    ``idx`` is increasing and every covered slot lies after the pending slot;
    ``last`` is the largest covered slot. Uncovered rows target the discard
    row, so the write is one scatter per buffer.
    """
    discard = int(buffers[0].shape[0]) - 1
    if pending is None:
        target = jnp.where(covered, idx, discard)
        return tuple(b.at[target].set(r) for b, r in zip(buffers, rows)), None
    pending_slot, pending_rows = pending
    flush = jnp.logical_and(any_new, pending_slot >= 0)
    flush_target = jnp.where(flush, pending_slot, discard)
    buffers = tuple(b.at[flush_target].set(r) for b, r in zip(buffers, pending_rows))
    target = jnp.where(jnp.logical_and(covered, idx != last), idx, discard)
    buffers = tuple(b.at[target].set(r) for b, r in zip(buffers, rows))
    position = jnp.clip(last - idx[0], 0, int(idx.shape[0]) - 1)
    pending_rows = tuple(jnp.where(any_new, r[position], p) for r, p in zip(rows, pending_rows))
    return buffers, (jnp.where(any_new, last, pending_slot), pending_rows)


def _covered_samples(ts_grid, idx, cursor, hi, t_prev, t_acc, interp_u, emit, interp_lam):
    """Evaluate the outputs at the query times ``ts_grid[idx]`` inside the step."""
    covered = jnp.logical_and(idx >= cursor, idx < hi)
    # Clamp to the step so an out-of-step polynomial extrapolation cannot
    # overflow; out-of-step values are masked away, and their tangents with them.
    tq = jnp.clip(ts_grid[idx], t_prev, t_acc)

    def sample(t):
        u = interp_u(t)
        lam = None if interp_lam is None else interp_lam(t)
        return tuple(emit(t, u, lam))

    rows = jax.vmap(sample)(tq)
    return covered, rows


def fill_saveat_grid(
    ts_grid,
    buffers,
    *,
    cursor,
    t_prev,
    t_acc,
    accept,
    interp_u,
    emit,
    save_tol,
    interp_lam=None,
    pending=None,
):
    """Write the outputs at every grid time an accepted step covers.

    On an accepted step spanning ``(t_prev, t_acc]`` the unfilled requested
    save times ``tq <= t_acc + save_tol`` (the indices ``[cursor, hi)``)
    receive ``emit(tq, interp_u(tq), interp_lam(tq))`` — the outputs of the
    core's high-order one-step dense output (the same interpolant the
    continuous-event root finder uses). The upper bound carries a small
    ``save_tol`` so the final query, which lands on the endpoint up to a
    rounding error, is still captured; the grid is consumed front to back, so
    each query is claimed by exactly one step. ``t_prev`` / ``t_acc`` are
    passed gradient-dead so the only live derivative of a sample is through
    the interpolant (the Nordsieck / difference history), exactly like the
    event-state derivative.

    Args:
        ts_grid: The requested save times, strictly increasing, ``(n_save,)``.
        buffers: The output buffers to update, ``(capacity + 1, *shape)`` each.
        cursor: First unfilled grid index (``int32``).
        t_prev: The step's left endpoint (gradient-dead).
        t_acc: The step's right endpoint (gradient-dead).
        accept: Whether the step was accepted (no fill on a rejected step).
        interp_u: Callable ``tq -> u(tq)`` evaluating the one-step dense output.
        emit: ``emit(t, u, lam) -> tuple`` evaluating the outputs at a state.
        save_tol: Endpoint tolerance added to the upper bound.
        interp_lam: Callable ``tq -> lam(tq)`` evaluating every bordered row,
            or ``None`` when no output reads the bordered globals (``emit``
            then receives ``lam=None``).
        pending: The pending-sample record, or ``None`` to write directly.

    Returns:
        ``(buffers, hi, pending)``: the updated buffers, the advanced cursor
        and the pending record.
    """
    n_save = int(ts_grid.shape[0])
    if int(buffers[0].shape[0]) - 1 < n_save:
        raise ValueError("save output buffers must hold at least one row per requested time")
    hi_raw = jnp.searchsorted(ts_grid, t_acc + save_tol, side="right").astype(jnp.int32)
    hi = jnp.where(accept, jnp.maximum(hi_raw, cursor), cursor)
    idx = jnp.arange(n_save, dtype=jnp.int32)
    covered, rows = _covered_samples(
        ts_grid, idx, cursor, hi, t_prev, t_acc, interp_u, emit, interp_lam
    )
    buffers, pending = _write_saveat_rows(buffers, pending, idx, rows, covered, hi - 1, hi > cursor)
    return buffers, hi, pending


def fill_saveat_window(
    ts_grid,
    buffers,
    *,
    cursor,
    t_prev,
    t_acc,
    accept,
    interp_u,
    emit,
    save_tol,
    window,
    interp_lam=None,
    pending=None,
):
    """Incremental twin of [fill_saveat_grid][diff_soup.timestepping.vsvo.solution.fill_saveat_grid] over a bounded window.

    The save grid is strictly increasing and consumed once, front to back:
    ``cursor`` is the first unfilled index, and an accepted step covering
    ``(t_prev, t_acc]`` fills the covered indices ``[cursor, hi)`` where
    ``hi`` comes from one ``searchsorted``. At most ``window`` interpolant
    evaluations run on the fast path; a step covering more than ``window``
    save times falls back to the full-grid fill for that step, so no sample
    is ever dropped. An attempt that covers no save time (every rejected
    attempt among them) evaluates and writes nothing and leaves the cursor in
    place. Each covered slot is written once, with one scatter per buffer.

    Args:
        ts_grid: The requested save times, strictly increasing, ``(n_save,)``.
        buffers: The output buffers to update, ``(capacity + 1, *shape)`` each.
        cursor: First unfilled grid index (``int32``).
        t_prev: The step's left endpoint (gradient-dead).
        t_acc: The step's right endpoint (gradient-dead).
        accept: Whether the step was accepted.
        interp_u: Callable ``tq -> u(tq)`` evaluating the one-step dense output.
        emit: ``emit(t, u, lam) -> tuple`` evaluating the outputs at a state.
        save_tol: Endpoint tolerance added to the upper bound.
        window: Static window width (``1 <= window <= n_save``).
        interp_lam: Callable ``tq -> lam(tq)`` evaluating every bordered row,
            or ``None`` when no output reads the bordered globals.
        pending: The pending-sample record, or ``None`` to write directly.

    Returns:
        ``(buffers, cursor, evaluations, pending)`` — the updated buffers,
        the advanced cursor, the number of interpolant evaluations this
        attempt performed (``0`` when it covers no save time, ``window`` on
        the fast path, ``n_save`` on the overflow fallback), and the pending
        record.
    """
    n_save = int(ts_grid.shape[0])
    if int(buffers[0].shape[0]) - 1 < n_save:
        raise ValueError("save output buffers must hold at least one row per requested time")
    window = int(window)
    hi_raw = jnp.searchsorted(ts_grid, t_acc + save_tol, side="right").astype(jnp.int32)
    hi = jnp.where(accept, jnp.maximum(hi_raw, cursor), cursor)
    overflow = (hi - cursor) > window

    def _fill(operand, idx):
        buffers_in, pending_in = operand
        covered, rows = _covered_samples(
            ts_grid, idx, cursor, hi, t_prev, t_acc, interp_u, emit, interp_lam
        )
        return _write_saveat_rows(buffers_in, pending_in, idx, rows, covered, hi - 1, hi > cursor)

    def _full(operand):
        return _fill(operand, jnp.arange(n_save, dtype=jnp.int32))

    def _windowed(operand):
        start = jnp.clip(cursor, 0, n_save - window)
        return _fill(operand, start + jnp.arange(window, dtype=jnp.int32))

    def _cover(operand):
        return lane_cond(overflow, _full, _windowed, operand)

    # An attempt that covers no save time (every rejected attempt, most
    # accepted ones) evaluates nothing; under vmap the fill runs when some
    # lane covers a save time, and the full grid only when some lane overflows.
    covers = hi > cursor
    buffers, pending = lane_cond(
        covers, _cover, lambda operand: operand, (buffers, pending), false_is_free=True
    )
    evaluations = jnp.where(
        covers,
        jnp.where(overflow, jnp.asarray(n_save, jnp.int32), jnp.asarray(window, jnp.int32)),
        jnp.asarray(0, jnp.int32),
    )
    return buffers, hi, evaluations, pending


def _solution_capacity(ts, ys) -> Optional[int]:
    """Return the static leading output capacity from a trajectory payload."""
    if ts is not None:
        ts_arr = jnp.asarray(ts)
        if ts_arr.ndim != 1:
            raise ValueError(f"solution ts must be one-dimensional; got shape {ts_arr.shape}")
        return int(ts_arr.shape[0])
    for leaf in jax.tree_util.tree_leaves(ys):
        if eqx.is_array(leaf) and leaf.ndim > 0:
            return int(leaf.shape[0])
    return None


def _finite_rows(tree, capacity: int) -> Array:
    """Return rows whose aligned floating payload leaves are all finite."""
    valid = jnp.ones((capacity,), dtype=bool)
    for leaf in jax.tree_util.tree_leaves(tree):
        if not eqx.is_array(leaf) or leaf.ndim == 0 or int(leaf.shape[0]) != capacity:
            continue
        if not jnp.issubdtype(leaf.dtype, jnp.inexact):
            continue
        finite = jnp.isfinite(leaf)
        if leaf.ndim > 1:
            finite = jnp.all(finite, axis=tuple(range(1, leaf.ndim)))
        valid = jnp.logical_and(valid, finite)
    return valid


def _mask_aligned_payload(tree, valid_mask: Array, capacity: int):
    """Apply deterministic invalid-tail sentinels to aligned array leaves."""

    def mask_leaf(leaf):
        if not eqx.is_array(leaf) or leaf.ndim == 0 or int(leaf.shape[0]) != capacity:
            return leaf
        shape = (capacity,) + (1,) * (leaf.ndim - 1)
        keep = valid_mask.reshape(shape)
        if jnp.issubdtype(leaf.dtype, jnp.bool_):
            sentinel = jnp.zeros_like(leaf, dtype=bool)
        elif jnp.issubdtype(leaf.dtype, jnp.inexact):
            sentinel = jnp.full_like(leaf, jnp.nan)
        elif jnp.issubdtype(leaf.dtype, jnp.integer):
            sentinel = jnp.full_like(leaf, -1)
        else:
            return leaf
        return jnp.where(keep, leaf, sentinel)

    return jax.tree_util.tree_map(mask_leaf, tree)


def _aligned_int_metadata(value, valid_mask: Array, capacity: int, *, valid_default: int) -> Array:
    """Normalize one aligned integer metadata vector and its invalid tail."""
    if value is None:
        arr = jnp.full((capacity,), valid_default, dtype=jnp.int32)
    else:
        arr = jnp.asarray(value, dtype=jnp.int32)
        if arr.shape != (capacity,):
            raise ValueError(
                f"aligned solution metadata must have shape ({capacity},); got {arr.shape}"
            )
    return jnp.where(valid_mask, arr, jnp.asarray(-1, dtype=jnp.int32))


def build_solution(
    *,
    ts,
    ys,
    lambdas=None,
    stats,
    result,
    info=None,
    interpolation=None,
    trace=None,
    control_state=None,
    event_mask=None,
    valid_mask=None,
    valid_count=None,
    control_epochs=None,
    event_indices=None,
    event_kinds=None,
    event_directions=None,
    result_codes=None,
    tolerances=None,
    event_log=None,
    terminal=None,
    alg=None,
    prob=None,
    history=None,
    outputs=None,
) -> Solution:
    """Package the supplied fields into a [Solution][diff_soup.timestepping.vsvo.solution.Solution].

    The native cores and engine adapters share this keyword packager. It keeps
    the fixed leading capacity intact, derives a true-prefix validity mask when
    one is omitted, and normalizes every invalid tail of the aligned payload
    (``ts``, ``ys``, ``lambdas`` and the
    [SavedOutputs][diff_soup.timestepping.vsvo.solution.SavedOutputs]):
    floating leaves are ``NaN``, integer metadata is ``-1``, and booleans are
    false. The caller is
    responsible for having resolved ``result`` to a terminal
    [RESULT][diff_soup._solution.RESULT].
    """
    capacity = _solution_capacity(ts, (ys, outputs))
    if capacity is not None:
        if valid_mask is None:
            inferred = jnp.ones((capacity,), dtype=bool)
            if ts is not None:
                inferred = jnp.logical_and(inferred, jnp.isfinite(jnp.asarray(ts)))
            inferred = jnp.logical_and(inferred, _finite_rows(ys, capacity))
            inferred = jnp.logical_and(inferred, _finite_rows(lambdas, capacity))
            inferred = jnp.logical_and(inferred, _finite_rows(outputs, capacity))
        else:
            inferred = jnp.asarray(valid_mask, dtype=bool)
            if inferred.shape != (capacity,):
                raise ValueError(f"valid_mask must have shape ({capacity},); got {inferred.shape}")
        # Enforce a prefix in traced code as well: once a slot is invalid, no
        # later slot may become valid.
        valid_mask = jnp.cumprod(inferred.astype(jnp.int32)).astype(bool)
        derived_count = jnp.sum(valid_mask.astype(jnp.int32), dtype=jnp.int32)
        if valid_count is not None and not isinstance(valid_count, jax.core.Tracer):
            supplied = int(jnp.asarray(valid_count))
            actual = int(jnp.asarray(derived_count))
            if supplied != actual:
                raise ValueError(
                    "valid_count must equal sum(valid_mask); "
                    f"got valid_count={supplied} and sum={actual}"
                )
        valid_count = derived_count
        ts = _mask_aligned_payload(
            jnp.asarray(ts) if ts is not None else None, valid_mask, capacity
        )
        ys = _mask_aligned_payload(ys, valid_mask, capacity)
        lambdas = _mask_aligned_payload(lambdas, valid_mask, capacity)
        outputs = _mask_aligned_payload(outputs, valid_mask, capacity)
        control_epochs = _aligned_int_metadata(
            control_epochs, valid_mask, capacity, valid_default=0
        )
        event_indices = _aligned_int_metadata(event_indices, valid_mask, capacity, valid_default=-1)
        event_kinds = _aligned_int_metadata(event_kinds, valid_mask, capacity, valid_default=-1)
        event_directions = _aligned_int_metadata(
            event_directions, valid_mask, capacity, valid_default=-1
        )
        result_codes = _aligned_int_metadata(result_codes, valid_mask, capacity, valid_default=-1)

    return Solution(
        ts=ts,
        ys=ys,
        lambdas=lambdas,
        valid_mask=valid_mask,
        valid_count=valid_count,
        control_epochs=control_epochs,
        event_indices=event_indices,
        event_kinds=event_kinds,
        event_directions=event_directions,
        result_codes=result_codes,
        stats=stats,
        result=result,
        info=info,
        interpolation=interpolation,
        trace=trace,
        control_state=control_state,
        event_mask=event_mask,
        tolerances=tolerances,
        event_log=event_log,
        terminal=terminal,
        alg=alg,
        prob=prob,
        history=history,
        outputs=outputs,
    )


__all__ = [
    "Solution",
    "SolveHistory",
    "SolverStats",
    "SavedOutputs",
    "build_solution",
    "init_saveat_outputs",
    "fill_saveat_times",
    "terminal_sample_slot",
    "hold_pending_sample",
    "place_saveat_sample",
    "commit_pending_sample",
    "fill_saveat_grid",
    "fill_saveat_window",
]
