"""The driver's step loop: one attempted step, its events, its saves and its decisions."""

from __future__ import annotations

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup.timestepping._affect import apply_affect
from diff_soup.timestepping.events import (
    _events_of,
    resolve_in_loop,
)
from diff_soup.timestepping.saveat import output_emitter
from diff_soup.timestepping.vsvo.common._driver_state import (
    _CODE_ACCURACY_LIMIT,
    _CODE_EVENT_STOP,
    _CODE_INCONSISTENT_IC,
    _CODE_SUCCESS,
    _SAVE_TOL,
    _accuracy_guard_runs,
    _control_epoch,
    _DriverState,
    _event_projection_options,
    _output_plan,
    _tolerance_scale,
)
from diff_soup.timestepping.vsvo.common.engine import StepReport
from diff_soup.utils.lanes import _lane_where, lane_cond

from ..solution import (
    fill_saveat_times,
    fill_saveat_window,
    place_saveat_sample,
    terminal_sample_slot,
)
from .backends import initialization_policy
from .contract import (
    ic_gate_tolerance,
    pack_state,
    unpack_state,
)
from .decisions import REPLAY_DIVERGED_MESSAGE, DecisionTape
from .trace import (
    StepTraceHeader,
    buffer_at_set,
    write_trace,
    zero_record,
)

# Interpolant evaluations per attempt on the incremental save-fill fast path;
# a step covering more save times than this falls back to the full-grid fill.
_SAVE_WINDOW = 16


def _safe_ratio(num: jax.Array, den: jax.Array) -> jax.Array:
    """Return ``num / den`` with a unit fallback when ``den`` is zero."""
    ok = den != 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), jnp.ones_like(num))


def _freeze_finished(finished, leaf):
    """Freeze an inexact leaf on finished lanes, pass others through.

    Only inexact array leaves carry a tangent, so on a finished lane they are
    replaced by their stopped-gradient value (no cotangent can flow back through
    a lane that is only being kept alive by a slower sibling lane in a batched
    loop). Integer/bool control state is left as-is so the loop's own
    bookkeeping is untouched.
    """
    if eqx.is_inexact_array(leaf):
        return jnp.where(finished, jax.lax.stop_gradient(leaf), leaf)
    return leaf


def _buffer_fields(state):
    """The carry's write-only buffer subtrees: trace, dense nodes, decision record, outputs.

    The output buffers belong to the set only when they are written by slot
    alone (``state.outputs_buffered``).
    """
    fields = (state.trace, state.dense_nodes, state.decisions)
    return fields + (state.outputs,) if state.outputs_buffered else fields


def _strip_buffers(state):
    """``state`` with its write-only buffer subtrees replaced by ``None``."""
    width = len(_buffer_fields(state))
    return eqx.tree_at(_buffer_fields, state, (None,) * width, is_leaf=lambda x: x is None)


def _map_no_buffer(f, state):
    """Apply ``f`` to every leaf of ``state`` except the write-only buffers.

    The buffer subtrees (see ``_buffer_fields``) are the ones the
    checkpointed loop marks as write-only accumulation buffers; inside the
    loop body they support only ``.at[].set()``, so mapping over them would
    violate that contract. They are lifted out, left untouched, and
    restored. When absent they are ``None`` and both lifts are no-ops.
    """
    saved = _buffer_fields(state)
    mapped = jax.tree_util.tree_map(f, _strip_buffers(state))
    return eqx.tree_at(_buffer_fields, mapped, saved, is_leaf=lambda x: x is None)


def _select_no_buffer(pred, new, old):
    """Per-leaf ``where(pred, old_leaf, new_leaf)`` skipping both trees' write-only buffers.

    The buffers of ``new`` are kept as-is: the body's ``.at[].set()`` writes
    must survive; only the non-buffer leaves revert to ``old`` on lanes where
    ``pred`` is true. ``pred`` is a loop's finished flag, true only on a lane
    of a batch, so an unbatched body reads no ``old`` leaf here and the carry's
    large arrays can be updated in place.
    """
    saved = _buffer_fields(new)
    keep = jnp.logical_not(pred)
    picked = jax.tree_util.tree_map(
        lambda n, o: _lane_where(keep, n, o), _strip_buffers(new), _strip_buffers(old)
    )
    return eqx.tree_at(_buffer_fields, picked, saved, is_leaf=lambda x: x is None)


def _selected_event_tolerance(events, event_index, t_lo, t_hi, dtype):
    """Return the winning event's time-collision tolerance."""
    selected = jnp.asarray(_SAVE_TOL, dtype=dtype)
    width = jnp.abs(jnp.asarray(t_hi, dtype=dtype) - jnp.asarray(t_lo, dtype=dtype))
    for index, event in enumerate(events):
        root_finder = getattr(event, "root_finder", None)
        if root_finder is None:
            tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
        else:
            tolerance = jnp.maximum(
                jnp.asarray(float(getattr(root_finder, "atol", 0.0)), dtype=dtype),
                jnp.asarray(float(getattr(root_finder, "rtol", 0.0)), dtype=dtype) * width,
            )
        selected = jnp.where(event_index == index, tolerance, selected)
    return selected


def _check_replay(report: StepReport, tape: DecisionTape, slot, replay) -> StepReport:
    """Raise when a replayed attempt reaches a different time or next step.

    A replay uses the recorded decision inputs, so its time and next step size
    agree with the forward sweep's to rounding. A larger difference means the
    replay integrates a different trajectory; the solve raises instead of
    returning the gradient of that other trajectory.
    """
    eps = jnp.finfo(jnp.asarray(report.t_new).dtype).eps

    def close(value, recorded):
        value = jnp.asarray(value).astype(recorded.dtype)
        scale = jnp.maximum(jnp.abs(value), jnp.abs(recorded))
        return jnp.abs(value - recorded) <= 64.0 * eps * scale

    same = jnp.logical_and(
        close(report.t_new, tape.t_new[slot]), close(report.h_new, tape.h_new[slot])
    )
    diverged = jnp.logical_and(replay, jnp.logical_not(same))
    t_new = eqx.error_if(report.t_new, diverged, REPLAY_DIVERGED_MESSAGE)
    return eqx.tree_at(lambda r: r.t_new, report, t_new)


def _record_decisions(tape: DecisionTape, slot, report: StepReport, finished) -> DecisionTape:
    """Write one attempt's decision inputs, time and next step into its row.

    A lane that finished before the rest of a batched loop keeps its row as it
    was.
    """

    def write(buffer, value):
        value = jnp.asarray(value).astype(buffer.dtype)
        return buffer.at[slot].set(jnp.where(finished, buffer[slot], value))

    return DecisionTape(
        written=write(tape.written, True),
        t_new=write(tape.t_new, report.t_new),
        h_new=write(tape.h_new, report.h_new),
        h_restart=tape.h_restart,
        values=tuple(write(buffer, value) for buffer, value in zip(tape.values, report.decisions)),
    )


def _record_restart_step(tape: DecisionTape, slot, step, finished) -> DecisionTape:
    """Write the entry step of the carry an attempt hands on into its row.

    After an event restart this is the restarted engine's entry step, which a
    replay of the attempt reuses. A lane that finished before the rest of a
    batched loop keeps its row as it was.
    """
    buffer = tape.h_restart
    row = jnp.where(finished, buffer[slot], jnp.asarray(step).astype(buffer.dtype))
    return DecisionTape(
        written=tape.written,
        t_new=tape.t_new,
        h_new=tape.h_new,
        h_restart=buffer.at[slot].set(row),
        values=tape.values,
    )


def _write_collide_rows(buffers, collide, rows):
    """Overwrite the rows flagged by ``collide`` (``(capacity,)``) with one sample's rows.

    Used for the save time at a mutate-and-continue switch, whose buffers ride
    the ordinary carry.
    """
    flagged = jnp.concatenate([collide, jnp.zeros((1,), dtype=bool)])
    return tuple(
        jnp.where(flagged.reshape((-1,) + (1,) * row.ndim), row[None], buffer)
        for buffer, row in zip(buffers, rows)
    )


def _loop_fns(
    engine,
    problem,
    cfg,
    has_affect,
    control_folds_theta,
    ts_grid,
    t1,
    theta,
    control0,
    record_decisions=False,
):
    """Build the step loop's ``cond`` and ``body`` over the traced solve inputs.

    Every traced value the loop body reads (the save grid, the horizon, the
    parameters, the event control seed) arrives as an argument, so a caller
    can differentiate one step by differentiating through this factory's
    ``body``. Shared by the compiled solve core and by the composable forward
    rule, whose augmented loop differentiates ``body`` once per step.

    With ``record_decisions`` the carry holds a
    [DecisionTape][diff_soup.timestepping.vsvo.common.decisions.DecisionTape]:
    each attempt writes its decision inputs, time and next step to the row of
    its attempt index, and an attempt whose row is already written (the
    backward replay of a checkpointed loop) uses the recorded decision inputs
    and raises if it still reaches a different time or next step.
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionStatus, _project_algebraic_bidirectional

    _has_affect = has_affect
    _control_folds_theta = control_folds_theta
    _operational_controls = bool(getattr(problem, "has_operational_control", False))
    _recipe_controls = bool(getattr(problem, "has_recipe_controls", False))
    _recipe_controls = _recipe_controls or _operational_controls
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)
    dtype = ts_grid.dtype
    max_steps = int(cfg.max_steps)
    kind = getattr(engine, "trace_kind", "ndf")
    _events = _events_of(cfg.event)
    _has_projected_events = any(
        getattr(getattr(event, "state_policy", None), "mode", None) == "projected"
        for event in _events
    )
    _needs_projection = _has_projected_events or bool(cfg.epoch_projected_output)
    projection_options = _event_projection_options(engine) if _needs_projection else None
    output_plan = _output_plan(cfg, problem)
    needs_lam = output_plan.needs_lam

    # An engine whose ``attempt_step`` takes ``live`` is told which lanes the
    # loop keeps (see the engine protocol).
    takes_live = "live" in inspect.signature(type(engine).attempt_step).parameters

    def cond_fun(state: _DriverState) -> jax.Array:
        return jnp.logical_and(state.step < max_steps, jnp.logical_not(state.done))

    def _step_body(state: _DriverState, theta_local, live) -> _DriverState:
        lanes = {} if live is None else {"live": live}
        if record_decisions:
            tape = state.decisions
            slot = state.step
            replay = jnp.logical_and(tape.written[slot], jnp.logical_not(state.done))
            recorded = tuple(buffer[slot] for buffer in tape.values)
            new_carry, report = engine.attempt_step(
                state.engine_carry, decision=(replay, recorded), **lanes
            )
            report = _check_replay(report, tape, slot, replay)
            decisions = _record_decisions(tape, slot, report, state.done)
        else:
            new_carry, report = engine.attempt_step(state.engine_carry, **lanes)
            decisions = None
        accepted = jnp.asarray(report.accepted, jnp.bool_)
        t_new = report.t_new

        def interp_u(tq):
            y_t, _ = engine.dense(new_carry, tq)
            return y_t

        if n_lam > 0:

            def interp_lam(tq):
                y_full, _ = engine.dense_full(new_carry, tq)
                return y_full[n_u:]

        else:
            interp_lam = None
        emit = output_emitter(output_plan, problem, theta_local, dtype)
        # The bordered-row interpolant runs for the saved outputs only when
        # one of them reads the globals.
        if needs_lam:
            save_interp_lam = interp_lam
            if save_interp_lam is None:

                def save_interp_lam(tq):
                    return jnp.zeros((0,), dtype)

        else:
            save_interp_lam = None

        def output_rows(t, u, lam):
            return emit(t, u, lam if needs_lam else None)

        # Event resolution (skipped entirely on the default no-event path).
        # On an affect solve it runs BEFORE the grid fill so a mid-step switch
        # caps the grid claim at the crossing — the segment past the switch
        # belongs to the pre-switch dynamics and is integrated again by the
        # restarted engine.
        if cfg.event is not None:
            control = state.control_state if (_has_affect or _recipe_controls) else control0
            events = _events

            def control_at(tq):
                if _recipe_controls:
                    return problem.control_values(tq, theta_local, control)
                return control

            def projected_output_state(tq, active_control):
                """Project a sample before its control epoch leaves the loop carry."""
                interpolated, interpolated_dot = engine.dense_full(new_carry, tq)
                projected_control = (
                    problem.control_values(tq, theta_local, active_control)
                    if _recipe_controls
                    else active_control
                )
                return _project_algebraic_bidirectional(
                    problem,
                    t=tq,
                    state=interpolated,
                    state_dot=interpolated_dot,
                    seed=interpolated,
                    theta=theta_local,
                    control=projected_control,
                    options=projection_options,
                )

            def interp_full(tq):
                lam_tq = report.lam_new if n_lam == 0 else interp_lam(tq)
                return pack_state(interp_u(tq), lam_tq)

            def projected_event_state(event, tq):
                del event
                interpolated, interpolated_dot = engine.dense_full(new_carry, tq)
                return _project_algebraic_bidirectional(
                    problem,
                    t=tq,
                    state=interpolated,
                    state_dot=interpolated_dot,
                    seed=interpolated,
                    theta=theta_local,
                    control=control_at(tq),
                    options=projection_options,
                )

            def event_state_at(event, tq):
                policy = getattr(event, "state_policy", None)
                if policy is None or policy.mode == "fast":
                    return interp_full(tq)
                projected = projected_event_state(event, tq)
                succeeded = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
                return jnp.where(
                    succeeded, projected.state, jnp.full_like(projected.state, jnp.nan)
                )

            def event_state_status(event, tq):
                policy = getattr(event, "state_policy", None)
                if policy is None or policy.mode == "fast":
                    return jnp.asarray(True)
                return projected_event_state(event, tq).status == int(
                    AlgebraicProjectionStatus.SUCCESS
                )

            def selected_event_state(event_index, tq):
                selected = interp_full(tq)
                for declared_index, declared_event in enumerate(events):
                    candidate = event_state_at(declared_event, tq)
                    selected = jnp.where(event_index == declared_index, candidate, selected)
                return selected

            # Endpoint conditions read the EXACT accepted states — the loop
            # carry at ``t_prev`` and the just-accepted ``report`` at
            # ``t_new`` — never the one-step interpolant.  Accepted DAE states
            # are algebraically consistent, so endpoint bracketing needs no
            # projection even for a ``projected`` state policy. Mid-step root
            # candidates read every row from the engine interpolant; projection
            # is reserved for policies that require strict constraint
            # satisfaction. ``resolve_in_loop`` enters that arm only on an
            # accepted step whose endpoints bracket a crossing.
            def cond_lo_fn(cf):
                return jnp.asarray(
                    cf(
                        state.t_prev,
                        pack_state(state.y_last, state.lam_last),
                        theta_local,
                        control=control_at(state.t_prev),
                    )
                )

            def cond_hi_fn(cf):
                return jnp.asarray(
                    cf(
                        t_new,
                        pack_state(report.y_new, report.lam_new),
                        theta_local,
                        control=control_at(t_new),
                    )
                )

            fired, idx, t_star, ev_kind, ev_direction, event_evaluation_ok = resolve_in_loop(
                events,
                state.t_prev,
                t_new,
                cond_lo_fn,
                cond_hi_fn,
                interp_full,
                control=control,
                control_at=control_at if _recipe_controls else None,
                state_at=event_state_at if _has_projected_events else None,
                state_status=event_state_status if _has_projected_events else None,
                active=accepted,
                args=theta_local,
            )
            # Root-finder tolerances are static Python values and may be
            # strong float64. Keep the crossing on the solve's numeric lane
            # before it enters restart state or loop-carry selections.
            t_star = jnp.asarray(t_star, dtype=dtype)
            event_projection_failed = jnp.logical_and(accepted, ~event_evaluation_ok)
            fired = jnp.logical_and(jnp.logical_and(accepted, fired), event_evaluation_ok)

        pre_output_projection_failed = jnp.asarray(False)
        if _has_affect:
            # Apply the fired event's affect at the crossing state. A
            # terminate-only member reports ``terminate`` instead, keeping
            # the stop semantics; the affect outputs are used only under
            # ``do_affect``.
            #
            # Build the crossing state only on the firing step: under the
            # ``projected`` state policy ``selected_event_state`` is a dense
            # interpolant evaluation plus a full Newton algebraic projection,
            # and running it branchlessly on every attempted step is the same
            # per-step leak the located arm and the terminate arm carried
            # before they were gated.  The non-firing branch feeds the accepted
            # endpoint state -- algebraically consistent by construction and a
            # valid in-domain point, the guarded-inputs rule of the restart
            # below -- and every consumer is already masked by ``do_affect`` /
            # ``log_fire``.  ``apply_affect`` itself stays branchless: it is
            # the user's traced affect arithmetic, cheap and previously
            # branchless too, and its ``terminate`` flag is value-independent.
            def _crossing_state(_):
                return selected_event_state(idx, t_star)

            def _endpoint_state(_):
                return pack_state(report.y_new, report.lam_new)

            u_star = lane_cond(fired, _crossing_state, _endpoint_state, None, false_is_free=True)
            affect_control = control if _operational_controls else control_at(t_star)
            u_prime, control_prime, terminate = apply_affect(
                events,
                idx,
                t_star,
                u_star,
                theta_local,
                affect_control,
                normalize_control=(
                    problem.normalize_control
                    if _recipe_controls and not _operational_controls
                    else None
                ),
            )
            if _recipe_controls and not _operational_controls:
                control_prime = state.control_state.with_values(control_prime)
            u_p, lam_p = unpack_state(u_prime, n_u)
            pre_epoch = _control_epoch(state.control_state)
            post_epoch = _control_epoch(control_prime)
            # A switch within the endpoint tolerance of ``t1`` has nothing
            # left to integrate: let the solve complete instead of restarting
            # a zero-length interval.
            before_end = t_star < t1 - _SAVE_TOL
            do_affect = jnp.logical_and(
                jnp.logical_and(fired, jnp.logical_not(terminate)), before_end
            )
            if _operational_controls:
                # An event condition can remain on the newly active control
                # manifold after a switch (for example, a voltage cutoff that
                # becomes the voltage-hold row). Numerical noise may then
                # report the same crossing again. A no-op operating update
                # keeps its epoch, so it must not restart the engine or
                # invalidate a factor generation a second time.
                epoch_changed = control_prime.epoch != state.control_state.epoch
                state_changed = jnp.any(u_prime != u_star)
                do_affect = jnp.logical_and(
                    do_affect,
                    jnp.logical_or(epoch_changed, state_changed),
                )
            event_stop = jnp.logical_and(fired, terminate)
            t_fill = jnp.where(jnp.logical_or(do_affect, event_stop), t_star, t_new)
            # One bounded log records every fire: a mutate-and-continue fire
            # holds the post-affect state, a terminate-member fire holds the
            # crossing (stopping) state.
            log_fire = jnp.logical_or(do_affect, event_stop)
            log_y = jnp.where(do_affect, u_prime, u_star)
            if cfg.epoch_projected_output:

                def _project_pre_log(_):
                    safe_t = jnp.where(log_fire, t_star, state.t_prev)
                    projected = projected_output_state(safe_t, state.control_state)
                    return projected.state, projected.status

                pre_log_y, pre_log_status = lane_cond(
                    log_fire,
                    _project_pre_log,
                    lambda _: (
                        u_star,
                        jnp.asarray(int(AlgebraicProjectionStatus.SUCCESS), jnp.int32),
                    ),
                    None,
                    false_is_free=True,
                )
                pre_output_projection_failed = jnp.logical_and(
                    log_fire,
                    pre_log_status != int(AlgebraicProjectionStatus.SUCCESS),
                )
            else:
                pre_log_y = u_star
                pre_output_projection_failed = jnp.asarray(False)
            appended = state.event_log.append(
                t=t_star,
                index=idx,
                kind=ev_kind,
                direction=ev_direction,
                y=log_y,
                pre_y=pre_log_y,
                pre_epoch=pre_epoch,
                post_epoch=jnp.where(do_affect, post_epoch, pre_epoch),
            )
            event_log = jax.tree_util.tree_map(
                lambda new, old: jnp.where(log_fire, new, old),
                appended,
                state.event_log,
            )
        else:
            if cfg.event is not None:
                event_stop = fired
                # A terminating crossing stops the output at the crossing:
                # the remainder of the step belongs to a trajectory the solve
                # rejected by stopping (a discrete stop keeps the boundary,
                # where t_star equals the step end).
                t_fill = jnp.where(event_stop, t_star, t_new)
            else:
                t_fill = t_new

        # A step whose event-state projection failed cannot be accepted into
        # the returned trajectory: the solver does not know whether it crossed
        # a terminating surface. Stop at the prior accepted point instead.
        trajectory_accepted = (
            jnp.logical_and(accepted, jnp.logical_not(event_projection_failed))
            if cfg.event is not None
            else accepted
        )
        sample_control = state.control_state if (_has_affect or _recipe_controls) else control0
        sample_epoch = _control_epoch(sample_control)

        if cfg.event is not None:
            root_tolerance = _selected_event_tolerance(events, idx, state.t_prev, t_new, dtype)
            if not cfg.save_every:
                # Use the root tolerance for the single requested sample
                # allowed to collide with a terminal crossing. This prevents
                # a broader endpoint band from admitting post-termination
                # queries and then appending the earlier exact terminal time.
                next_query = jnp.searchsorted(ts_grid, t_star, side="right").astype(jnp.int32)
                safe_next_query = jnp.minimum(
                    next_query, jnp.asarray(int(ts_grid.shape[0]) - 1, jnp.int32)
                )
                next_query_collides = jnp.logical_and(
                    next_query < int(ts_grid.shape[0]),
                    jnp.abs(ts_grid[safe_next_query] - t_star) <= root_tolerance,
                )
                terminal_fill_bound = jnp.where(
                    next_query_collides, ts_grid[safe_next_query], t_star
                )
                t_fill = jnp.where(event_stop, terminal_fill_bound, t_fill)

        # A mutate-and-continue switch owns the pre-switch flow only up to
        # the crossing itself: the endpoint tolerance band must not claim
        # save times just beyond ``t*`` (they belong to the restarted
        # post-switch flow), so the crossing step's fill drops the band.
        if _has_affect:
            step_save_tol = jnp.where(
                jnp.logical_or(do_affect, event_stop), jnp.asarray(0.0, dtype), _SAVE_TOL
            )
        elif cfg.event is not None:
            step_save_tol = jnp.where(event_stop, jnp.asarray(0.0, dtype), _SAVE_TOL)
        else:
            step_save_tol = _SAVE_TOL
        capacity = int(state.ts_output.shape[0])
        slots = jnp.arange(capacity, dtype=jnp.int32)
        pending = state.pending
        if cfg.save_every:
            outputs = state.outputs
            save_cursor = state.save_cursor
            valid_count = state.valid_count
            ts_output = state.ts_output
            control_epochs = state.control_epochs
            fill_evals = jnp.asarray(0, jnp.int32)
            pre_owned = jnp.zeros((capacity,), dtype=bool)
            if _has_affect:
                collide = jnp.zeros((capacity,), dtype=bool)
        else:
            old_save_cursor = state.save_cursor
            outputs, save_cursor, fill_evals, pending = fill_saveat_window(
                ts_grid,
                state.outputs,
                cursor=state.save_cursor,
                t_prev=state.t_prev,
                t_acc=t_fill,
                accept=trajectory_accepted,
                interp_u=interp_u,
                emit=emit,
                save_tol=step_save_tol,
                window=min(int(ts_grid.shape[0]), _SAVE_WINDOW),
                interp_lam=save_interp_lam,
                pending=pending,
            )
            ts_output = fill_saveat_times(ts_grid, state.ts_output, save_cursor)
            valid_count = jnp.maximum(state.valid_count, save_cursor)
            newly_filled = jnp.logical_and(slots >= old_save_cursor, slots < save_cursor)
            control_epochs = jnp.where(newly_filled, sample_epoch, state.control_epochs)
            pre_owned = newly_filled
            if _has_affect:
                # A row at the switch is the right limit: only the restarted
                # epoch projects and writes it. EventLog retains the left limit.
                collide_query = jnp.logical_and(do_affect, jnp.abs(ts_grid - t_star) <= _SAVE_TOL)
                collide = (
                    jnp.zeros((capacity,), dtype=bool).at[: ts_grid.shape[0]].set(collide_query)
                )
                pre_owned = jnp.logical_and(pre_owned, jnp.logical_not(collide))
        sample_projection_failed = jnp.asarray(False)
        if cfg.epoch_projected_output:
            # Project only rows claimed by this accepted segment. A static
            # for-loop keeps reverse-mode support while its scalar conditional
            # avoids running a Newton solve for untouched grid slots.
            def _project_saved_row(slot, carried):
                buffers, failed = carried

                def _project(_):
                    safe_t = jnp.where(pre_owned[slot], ts_grid[slot], state.t_prev)
                    projected = projected_output_state(safe_t, state.control_state)
                    projected_y, projected_lam = unpack_state(projected.state, n_u)
                    succeeded = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
                    rows = output_rows(safe_t, projected_y, projected_lam)
                    return (
                        tuple(
                            jnp.where(succeeded, buffer.at[slot].set(row), buffer)
                            for buffer, row in zip(buffers, rows)
                        ),
                        jnp.logical_or(failed, jnp.logical_not(succeeded)),
                    )

                return lane_cond(
                    pre_owned[slot], _project, lambda _: carried, None, false_is_free=True
                )

            outputs, sample_projection_failed = jax.lax.fori_loop(
                0,
                int(ts_grid.shape[0]),
                _project_saved_row,
                (outputs, sample_projection_failed),
            )
        output_projection_failed = jnp.logical_or(
            pre_output_projection_failed, sample_projection_failed
        )
        if cfg.epoch_projected_output:
            # A failed projected row invalidates the whole accepted segment.
            # Restore every aligned output leaf now; the final aggregate below
            # also covers a failure discovered while constructing the right
            # limit after restart.
            outputs = jax.tree_util.tree_map(
                lambda old, new: jnp.where(output_projection_failed, old, new),
                state.outputs,
                outputs,
            )
            ts_output = jnp.where(output_projection_failed, state.ts_output, ts_output)
            save_cursor = jnp.where(output_projection_failed, state.save_cursor, save_cursor)
            valid_count = jnp.where(output_projection_failed, state.valid_count, valid_count)
            control_epochs = jnp.where(
                output_projection_failed, state.control_epochs, control_epochs
            )
            trajectory_accepted = jnp.logical_and(
                trajectory_accepted, jnp.logical_not(output_projection_failed)
            )
        event_indices = state.event_indices
        event_kinds = state.event_kinds
        event_directions = state.event_directions
        result_codes = state.result_codes
        if _has_affect:
            # A save time colliding with a nonterminal event time records the
            # post-affect state (the pre-affect state stays in the event log).
            control_epochs = jnp.where(collide, post_epoch, control_epochs)

        if cfg.dense:
            # One node per accepted step, at slot ``accepts + 1``. A rejected
            # attempt writes its (discarded) values into the same pending
            # slot, which the eventually accepted attempt overwrites; the
            # node count never includes a pending slot.
            slot = state.accepts + 1
            node_t = jnp.where(trajectory_accepted, t_new, jnp.asarray(jnp.inf, dtype))
            node_y = pack_state(report.y_new, report.lam_new)
            node_yd = pack_state(report.ydot_new, jnp.zeros_like(report.lam_new))
            dn_t, dn_y, dn_yd = state.dense_nodes
            dense_nodes = (
                buffer_at_set(dn_t, slot, node_t),
                buffer_at_set(dn_y, slot, node_y),
                buffer_at_set(dn_yd, slot, node_yd),
            )
        else:
            dense_nodes = state.dense_nodes

        # Termination: an engine-fatal attempt stops with its own code; an
        # accepted step reaching t1 succeeds; an event overrides both. A
        # mid-step switch rolls the step back to the crossing, so that step
        # no longer reaches the end.
        attempt_failed = jnp.logical_and(
            report.result != _CODE_SUCCESS, report.result != _CODE_EVENT_STOP
        )
        if cfg.event is not None:
            attempt_failed = jnp.logical_or(
                attempt_failed,
                jnp.logical_or(event_projection_failed, output_projection_failed),
            )
            attempt_result = jnp.where(
                jnp.logical_or(event_projection_failed, output_projection_failed),
                jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
                report.result,
            )
        else:
            attempt_result = report.result
        reached_end = jnp.logical_and(trajectory_accepted, t_new >= t1 - _SAVE_TOL)
        post_output_projection_failed = jnp.asarray(False)
        if _has_affect:
            reached_end = jnp.logical_and(reached_end, jnp.logical_not(do_affect))
        term_code = jnp.where(
            attempt_failed,
            attempt_result,
            jnp.where(reached_end, jnp.asarray(_CODE_SUCCESS, jnp.int32), state.term_code),
        )
        done = jnp.logical_or(attempt_failed, reached_end)
        t_reached = jnp.where(trajectory_accepted, t_new, state.t_reached)
        y_last = jnp.where(trajectory_accepted, report.y_new, state.y_last)
        lam_last = jnp.where(trajectory_accepted, report.lam_new, state.lam_last)
        t_prev = jnp.where(trajectory_accepted, t_new, state.t_prev)

        if cfg.event is not None:
            # A terminal fire stops the solve at the event time and records
            # the winning event (declaration order) for the solution mask.
            # The exact stopping point is the crossing state — the remainder
            # of the accepted step belongs to a trajectory the stop rejected.
            term_code = jnp.where(event_stop, jnp.asarray(_CODE_EVENT_STOP, jnp.int32), term_code)
            done = jnp.logical_or(done, event_stop)
            t_reached = jnp.where(event_stop, t_star, t_reached)
            event_idx = jnp.where(event_stop, idx, state.event_idx)
            event_kind = jnp.where(event_stop, ev_kind, state.event_kind)
            event_direction = jnp.where(event_stop, ev_direction, state.event_direction)
            # The crossing state is consumed only by the (at most one) step
            # whose terminal event fired, and constructing it evaluates the
            # dense interpolant — plus a full algebraic Newton projection for
            # a ``projected`` state policy — so it runs under ``lane_cond``
            # (under vmap, only when some lane's event fires): an always-on
            # build here cost one projection on EVERY attempted step (~10 s
            # of a 23 s full-scale cutoff solve). The skipped arm's value is
            # never read.
            y_stop, lam_stop = lane_cond(
                event_stop,
                lambda _: unpack_state(selected_event_state(idx, t_star), n_u),
                lambda _: (report.y_new, report.lam_new),
                None,
                false_is_free=True,
            )
            y_last = jnp.where(event_stop, y_stop, y_last)
            lam_last = jnp.where(event_stop, lam_stop, lam_last)

            terminal_slot, terminal_collides, terminal_room = terminal_sample_slot(
                ts_output,
                valid_count,
                t_star,
                root_tolerance,
            )
            safe_terminal_slot = jnp.minimum(terminal_slot, jnp.asarray(capacity - 1, jnp.int32))
            write_terminal = jnp.logical_and(event_stop, terminal_room)

            def _write_terminal(buffer, value):
                written = buffer.at[safe_terminal_slot].set(value)
                return jnp.where(write_terminal, written, buffer)

            ts_output = _write_terminal(ts_output, t_star)
            outputs, pending = place_saveat_sample(
                outputs,
                pending,
                safe_terminal_slot,
                output_rows(t_star, y_stop, lam_stop),
                write_terminal,
            )
            control_epochs = _write_terminal(control_epochs, sample_epoch)
            event_indices = _write_terminal(event_indices, idx)
            event_kinds = _write_terminal(event_kinds, ev_kind)
            event_directions = _write_terminal(event_directions, ev_direction)
            result_codes = _write_terminal(result_codes, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
            valid_count = jnp.where(
                write_terminal,
                valid_count + jnp.logical_not(terminal_collides).astype(jnp.int32),
                valid_count,
            )

            # Record every fire in the bounded log: a terminal fire holds the
            # crossing (stopping) state; a mutate-and-continue fire holds the
            # post-affect state and is appended below, where that state
            # exists.
            if not _has_affect:

                def _append_fire(_):
                    return state.event_log.append(
                        t=t_star,
                        index=idx,
                        kind=ev_kind,
                        direction=ev_direction,
                        y=pack_state(y_stop, lam_stop),
                        pre_y=pack_state(y_stop, lam_stop),
                        pre_epoch=sample_epoch,
                        post_epoch=sample_epoch,
                    )

                event_log = lane_cond(
                    event_stop, _append_fire, lambda _: state.event_log, None, false_is_free=True
                )
            # With an affect in play the log was already updated where the
            # post-affect state exists (the affect block above).
        else:
            event_idx = state.event_idx
            event_kind = state.event_kind
            event_direction = state.event_direction
            event_log = state.event_log

        if _has_affect:
            # The switch: the accepted endpoint rolls back to the crossing
            # and the engine restarts there from the affected state, through
            # the same entry that built the first carry at ``t0`` — so the
            # engine's own startup (fresh step size, first order, consistent
            # initialization) applies at the switch.
            theta_re = control_prime if _control_folds_theta else theta_local

            # On lanes where no event fires (``do_affect`` is False under a
            # batched solve, where the restart runs for the whole batch once
            # any lane fires and each lane keeps its own branch) feed the restart the lane's own last
            # accepted state -- a valid in-domain point -- so its forward stays
            # finite; the branch select below discards these values there.
            # Guarding the inputs (rather than stop-gradient-ing the discarded
            # outputs) is what removes the ``0 * inf`` the branch's own backward
            # would otherwise form at a degenerate restart point.
            t_star_g = jnp.where(do_affect, t_star, state.t_prev)
            u_p_g = jnp.where(do_affect, u_p, state.y_last)
            lam_p_g = jnp.where(do_affect, lam_p, state.lam_last)
            # The restarted engine evaluates the model with the affected
            # control values; under the theta fold the rewritten values ride
            # ``theta_re`` instead and the engine's control stays ``None``.
            if _control_folds_theta:
                control_re = None
            else:
                control_re = jax.tree_util.tree_map(
                    lambda nw, od: jnp.where(do_affect, nw, od),
                    control_prime,
                    state.control_state,
                )

            def _restart_carry(_):
                # The engine freezes its internal time grid (step-size
                # controls carry no derivative), so a parameter-dependent
                # switch time cannot reach the post-switch trajectory through
                # the restart time itself. Transfer it through the state:
                # restarting at the gradient-frozen time from
                # ``u - (t* - stop_gradient(t*)) * du/dt`` is primal-identical
                # (the offset is exactly zero) and carries the switch-time
                # derivative via the flow-map identity
                # ``d y(t1) / d t0 = -Phi(t1, t0) f(t0, y0)`` — which, with
                # the crossing state's own tangent, realizes the switching
                # (saltation) correction ``(f_pre - f_post) * dt*``.
                t_frozen = jax.lax.stop_gradient(t_star_g)
                if _operational_controls:
                    # The new operating point owns the restart seed for its
                    # algebraic current. Evaluate the policy only on guarded
                    # lane values so inactive vmapped lanes cannot poison the
                    # selected branch with an out-of-domain default.
                    lam_start = problem.operational_seed_multipliers(
                        t_star_g,
                        u_p_g,
                        lam_p_g,
                        theta_re,
                        control_re,
                    )
                else:
                    lam_start = lam_p_g
                restart_state = pack_state(u_p_g, lam_start)
                # The crossing slope realizes the switching correction and
                # therefore must be the true post-affect flow
                # derivative at the affected state.
                slope_layout, slope_precision = initialization_policy(
                    problem, getattr(engine, "precision", None), mode="forward"
                )
                _, slope_reverse_precision = initialization_policy(
                    problem, getattr(engine, "precision", None), mode="reverse"
                )
                ydot_p = problem.startup_slope(
                    t_star_g,
                    restart_state,
                    theta_re,
                    control_re,
                    tol=ic_gate_tolerance(engine.atol, getattr(engine, "ic_tol", None)),
                    backend=getattr(engine, "backend", "auto"),
                    layout=slope_layout,
                    precision=slope_precision,
                    reverse_precision=slope_reverse_precision,
                )[:n_u]
                u_start = u_p_g - (t_star_g - t_frozen) * ydot_p
                # The entry step is searched over the affected state, which a
                # replay recomputes and may round differently; a replay enters
                # with the step the forward sweep's restart recorded.
                entry = {"step": (replay, tape.h_restart[slot])} if record_decisions else {}
                return engine.init_carry(
                    problem,
                    t_frozen,
                    t1,
                    u_start,
                    ydot_p,
                    lam_start,
                    theta_re,
                    cfg,
                    control=control_re,
                    **entry,
                )

            def _keep_carry(_):
                return new_carry

            new_carry = lane_cond(do_affect, _restart_carry, _keep_carry, None, false_is_free=True)
            if record_decisions:
                decisions = _record_restart_step(
                    decisions, slot, engine.step_at_entry(new_carry), state.done
                )
            post_y = u_p
            post_lam = lam_p
            if cfg.epoch_projected_output:
                # The restart owns the right limit. Read its entry state and
                # slope, then project once more under the post-affect control so
                # this remains correct even when consistent initialization was
                # disabled for the integration itself.
                if hasattr(engine, "state_at_entry"):
                    post_seed = engine.state_at_entry(new_carry)
                else:
                    post_seed = pack_state(u_p_g, lam_p_g)
                if hasattr(engine, "slope_at_entry"):
                    post_slope = engine.slope_at_entry(new_carry)
                else:
                    post_slope = jnp.zeros_like(post_seed)
                theta_post = (
                    jax.tree_util.tree_map(
                        lambda new, old: jnp.where(do_affect, new, old), theta_re, theta_local
                    )
                    if _control_folds_theta
                    else theta_local
                )
                post_projection_control = (
                    problem.control_values(t_star_g, theta_post, control_re)
                    if _recipe_controls
                    else control_re
                )
                post_projection = _project_algebraic_bidirectional(
                    problem,
                    t=t_star_g,
                    state=post_seed,
                    state_dot=post_slope,
                    seed=post_seed,
                    theta=theta_post,
                    control=post_projection_control,
                    options=projection_options,
                )
                post_output_projection_failed = jnp.logical_and(
                    do_affect,
                    post_projection.status != int(AlgebraicProjectionStatus.SUCCESS),
                )
                post_y, post_lam = unpack_state(post_projection.state, n_u)

                # A requested sample at the switch is one right-continuous row:
                # the left limit remains solely in EventLog.pre_ys.
                outputs = _write_collide_rows(
                    outputs, collide, output_rows(t_star, post_y, post_lam)
                )

                log_slot = jnp.minimum(
                    state.event_log.count,
                    jnp.asarray(int(state.event_log.ts.shape[0]) - 1, jnp.int32),
                )
                log_has_room = state.event_log.count < int(state.event_log.ts.shape[0])
                log_post_ys = state.event_log.ys.at[log_slot].set(pack_state(post_y, post_lam))
                event_log = eqx.tree_at(
                    lambda log: log.ys,
                    event_log,
                    jnp.where(
                        jnp.logical_and(do_affect, log_has_room),
                        log_post_ys,
                        event_log.ys,
                    ),
                )
            else:
                outputs = _write_collide_rows(outputs, collide, output_rows(t_star, u_p, lam_p))
            t_prev = jnp.where(do_affect, t_star, t_prev)
            t_reached = jnp.where(do_affect, t_star, t_reached)
            y_last = jnp.where(do_affect, post_y, y_last)
            lam_last = jnp.where(do_affect, post_lam, lam_last)
            # A mutate-and-continue fire also records the fired event's
            # declaration index, so a switched solve that then runs to the
            # horizon surfaces the last-fired affect in the solution's event
            # mask (matching the other engine families).
            event_idx = jnp.where(do_affect, idx, event_idx)
            control_state = jax.tree_util.tree_map(
                lambda nw, od: jnp.where(do_affect, nw, od),
                control_prime,
                state.control_state,
            )
        else:
            control_state = state.control_state

        if cfg.epoch_projected_output:
            projection_failed = jnp.logical_or(
                output_projection_failed, post_output_projection_failed
            )
            # Projection is part of accepting an output-bearing segment. A
            # failed row, left limit, or right limit rolls the whole attempt
            # back to its prior accepted point and reports the same public
            # inconsistent-initialization result used by projected events.
            term_code = jnp.where(
                projection_failed,
                jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
                term_code,
            )
            done = jnp.logical_or(done, projection_failed)
            t_prev = jnp.where(projection_failed, state.t_prev, t_prev)
            t_reached = jnp.where(projection_failed, state.t_reached, t_reached)
            y_last = jnp.where(projection_failed, state.y_last, y_last)
            lam_last = jnp.where(projection_failed, state.lam_last, lam_last)
            ts_output = jnp.where(projection_failed, state.ts_output, ts_output)
            outputs = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new), state.outputs, outputs
            )
            save_cursor = jnp.where(projection_failed, state.save_cursor, save_cursor)
            valid_count = jnp.where(projection_failed, state.valid_count, valid_count)
            control_epochs = jnp.where(projection_failed, state.control_epochs, control_epochs)
            event_indices = jnp.where(projection_failed, state.event_indices, event_indices)
            event_kinds = jnp.where(projection_failed, state.event_kinds, event_kinds)
            event_directions = jnp.where(
                projection_failed, state.event_directions, event_directions
            )
            result_codes = jnp.where(projection_failed, state.result_codes, result_codes)
            event_idx = jnp.where(projection_failed, state.event_idx, event_idx)
            event_kind = jnp.where(projection_failed, state.event_kind, event_kind)
            event_direction = jnp.where(projection_failed, state.event_direction, event_direction)
            event_log = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new),
                state.event_log,
                event_log,
            )
            control_state = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new),
                state.control_state,
                control_state,
            )

        # Accuracy-limit guard: at the top of this attempt, form the
        # weighted-RMS norm of the last accepted solution against its
        # error weights and scale it by the unit roundoff -- the factor by
        # which the tolerances would have to grow for the requested accuracy
        # to be achievable at this magnitude (the guard fires when that
        # factor exceeds 1). The carry's ``ewt`` field -- the per-attempt
        # error-weight vector every engine in the suite carries under that
        # name -- duck-types whether this engine supports the guard, so no
        # engine file needs an edit. Every engine in the suite
        # refreshes ``state.engine_carry.ewt`` only on an accepted step, so it
        # always pairs with ``state.y_last``/``state.lam_last`` as the
        # last-accepted point's own weights. Gated on at least one prior
        # attempt so the very
        # first attempt always gets to run; this takes precedence over
        # whatever this attempt itself reports, matching the convention that
        # this accuracy check runs before a step is attempted. When the engine
        # carries a ``suppressalg`` policy (JIDA), the norm masks out
        # algebraic rows via the engine's resolved differential-row mask --
        # an explicit engine-level ``id`` override wins over the problem's
        # ``id_mask``, exactly as in the engine's own error test; JBDF/JNDF
        # never mask this norm, matching their own unmasked error tests.
        # Every input and the result are wrapped in ``stop_gradient``: this
        # is a termination diagnostic, not part of the differentiated
        # trajectory, and must add no traced leaf to the loop's tangent.
        #
        # The law ``ewt = 1/(rtol*|y| + atol)`` bounds every row of ``y*ewt`` by
        # ``1/rtol``, so the factor cannot exceed ``uround/rtol``: a solve whose
        # ``rtol`` is far above the roundoff forms this norm once, for the
        # report, on its final state (see ``_finalize``).
        if _accuracy_guard_runs(engine, dtype) and hasattr(state.engine_carry, "ewt"):
            tolsf_now = _tolerance_scale(
                engine, problem, state.y_last, state.lam_last, state.engine_carry.ewt, dtype
            )
            too_much_acc = jnp.logical_and(state.step > 0, tolsf_now > 1.0)
        else:
            tolsf_now = state.tolsf
            too_much_acc = jnp.asarray(False)

        done = jnp.logical_or(done, too_much_acc)
        term_code = jnp.where(too_much_acc, jnp.asarray(_CODE_ACCURACY_LIMIT, jnp.int32), term_code)

        one = jnp.asarray(1, jnp.int32)
        acc_i = accepted.astype(jnp.int32)

        if cfg.save_every:
            # Accepted-step output is a direct prefix write of the trajectory
            # state the driver retained. In particular, a switch writes the
            # exact post-affect right limit at the crossing, while a terminal
            # crossing has already used the structural reserve above. The
            # final accepted state is retained even when its accept count is
            # not a multiple of the requested stride (or a later rejected
            # attempt exhausts the budget).
            next_accepts = state.accepts + acc_i
            periodic = jnp.logical_and(
                trajectory_accepted,
                jnp.equal(jnp.mod(next_accepts, int(cfg.save_every)), 0),
            )
            event_stopped = event_stop if cfg.event is not None else jnp.asarray(False)
            leaving_loop = jnp.logical_or(done, state.step + one >= max_steps)
            final_non_event = jnp.logical_and(
                jnp.logical_and(leaving_loop, jnp.logical_not(event_stopped)),
                next_accepts > 0,
            )
            want_step_sample = jnp.logical_or(periodic, final_non_event)
            previous_slot = jnp.clip(valid_count - 1, 0, capacity - 1)
            collides_previous = jnp.logical_and(
                valid_count > 0,
                jnp.abs(ts_output[previous_slot] - t_reached) <= _SAVE_TOL,
            )
            step_slot = jnp.where(collides_previous, previous_slot, valid_count)
            step_room = jnp.logical_or(collides_previous, valid_count < capacity)
            write_step = jnp.logical_and(want_step_sample, step_room)
            safe_step_slot = jnp.minimum(step_slot, jnp.asarray(capacity - 1, jnp.int32))
            final_sample_epoch = _control_epoch(
                control_state if (_has_affect or _recipe_controls) else control0
            )

            def _write_step(buffer, value):
                written = buffer.at[safe_step_slot].set(value)
                return jnp.where(write_step, written, buffer)

            ts_output = _write_step(ts_output, t_reached)
            outputs, pending = place_saveat_sample(
                outputs,
                pending,
                safe_step_slot,
                output_rows(t_reached, y_last, lam_last),
                write_step,
            )
            control_epochs = _write_step(control_epochs, final_sample_epoch)
            valid_count = jnp.where(
                write_step,
                valid_count + jnp.logical_not(collides_previous).astype(jnp.int32),
                valid_count,
            )
            save_cursor = valid_count

        if cfg.trace:
            eta = _safe_ratio(report.h_new, state.h_prev)
            header = StepTraceHeader(
                nst=state.step,
                t=t_new,
                h=report.h_new,
                k=jnp.asarray(report.k_new, jnp.int32),
                accepted=accepted,
                result_code=jnp.asarray(report.result, jnp.int32),
                newton_iters=jnp.asarray(report.newton_iters, jnp.int32),
                err_estimate=report.err_estimate,
                eta=eta,
            )
            body_rec = (
                report.trace_record if report.trace_record is not None else zero_record(kind, dtype)
            )
            record = eqx.tree_at(lambda r: r.header, body_rec, header)
            trace = write_trace(state.trace, state.step, record)
        else:
            trace = None

        return _DriverState(
            engine_carry=new_carry,
            t_prev=t_prev,
            h_prev=report.h_new,
            y_last=y_last,
            lam_last=lam_last,
            control_state=control_state,
            event_log=event_log,
            ts_output=ts_output,
            outputs=outputs,
            pending=pending,
            save_cursor=save_cursor,
            valid_count=valid_count,
            control_epochs=control_epochs,
            event_indices=event_indices,
            event_kinds=event_kinds,
            event_directions=event_directions,
            result_codes=result_codes,
            dense_nodes=dense_nodes,
            trace=trace,
            step=state.step + one,
            done=done,
            term_code=term_code,
            event_idx=event_idx,
            event_kind=event_kind,
            event_direction=event_direction,
            t_reached=t_reached,
            tolsf=tolsf_now,
            accepts=state.accepts + acc_i,
            rejects=state.rejects + (one - acc_i),
            trial_steps=state.trial_steps + one,
            newton_iters=state.newton_iters + jnp.asarray(report.newton_iters, jnp.int32),
            jac_refreshes=state.jac_refreshes + jnp.asarray(report.jac_refreshes, jnp.int32),
            factorizations=state.factorizations + jnp.asarray(report.factorizations, jnp.int32),
            linear_solves=state.linear_solves + jnp.asarray(report.linear_solves, jnp.int32),
            rhs_evals=state.rhs_evals + jnp.asarray(report.rhs_evals, jnp.int32),
            residual_evals=state.residual_evals + jnp.asarray(report.residual_evals, jnp.int32),
            precision_promotions=(
                state.precision_promotions
                if report.precision_promotions is None
                else state.precision_promotions
                + jnp.asarray(report.precision_promotions, jnp.int32)
            ),
            ir_refinement_steps=(
                state.ir_refinement_steps
                if state.ir_refinement_steps is None or report.ir_refinement_steps is None
                else state.ir_refinement_steps
                + jnp.asarray(report.ir_refinement_steps, jnp.result_type(float))
            ),
            fp64_fallbacks=(
                state.fp64_fallbacks
                if state.fp64_fallbacks is None or report.fp64_fallbacks is None
                else state.fp64_fallbacks
                + jnp.asarray(report.fp64_fallbacks, jnp.result_type(float))
            ),
            multirhs_solves=(
                state.multirhs_solves
                if report.multirhs_solves is None
                else state.multirhs_solves + jnp.asarray(report.multirhs_solves, jnp.int32)
            ),
            newton_failures=(
                state.newton_failures
                if report.newton_failures is None
                else state.newton_failures + jnp.asarray(report.newton_failures, jnp.int32)
            ),
            multirhs_columns=(
                state.multirhs_columns
                if report.multirhs_columns is None
                else state.multirhs_columns + jnp.asarray(report.multirhs_columns, jnp.int32)
            ),
            du_b_builds=(
                state.du_b_builds
                if report.du_b_builds is None
                else state.du_b_builds + jnp.asarray(report.du_b_builds, jnp.int32)
            ),
            schur_builds=(
                state.schur_builds
                if report.schur_builds is None
                else state.schur_builds + jnp.asarray(report.schur_builds, jnp.int32)
            ),
            # The incremental fill evaluates one bounded window per attempt
            # (the full grid only on a window overflow); count the work
            # actually performed.
            saveat_interpolations=state.saveat_interpolations + fill_evals,
            event_solves=state.event_solves
            + (do_affect.astype(jnp.int32) if _has_affect else jnp.asarray(0, jnp.int32)),
            decisions=decisions,
            outputs_buffered=state.outputs_buffered,
        )

    def body_fun(state: _DriverState) -> _DriverState:
        # Freeze finished lanes through the loop body. ``state.done`` is a
        # per-lane scalar bool -- ``False`` on every entry of an unbatched solve
        # (``cond_fun`` guarantees it), so the freeze and the output-select are
        # exact identities there. Under a batched loop a lane kept alive by a
        # slower sibling has ``done == True``; freezing its inexact inputs (and
        # the closure ``theta`` the affect path reads) keeps its now-meaningless
        # step from manufacturing cotangents, and the output-select reverts its
        # state to the finish point.
        finished = state.done
        frozen = _map_no_buffer(lambda leaf: _freeze_finished(finished, leaf), state)
        theta_local = jax.tree_util.tree_map(lambda leaf: _freeze_finished(finished, leaf), theta)
        # A batched loop keeps the body's result only where its condition holds.
        live = cond_fun(state) if takes_live else None
        stepped = _step_body(frozen, theta_local, live)
        return _select_no_buffer(finished, stepped, state)

    return cond_fun, body_fun
