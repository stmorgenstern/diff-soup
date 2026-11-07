"""The driver's loop state, its exit codes, and the state it starts from."""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.timestepping.event_records import EventLog
from diff_soup.timestepping.events import (
    RESOLVE_KIND_CONTINUOUS,
    RESOLVE_KIND_DISCRETE,
    _events_of,
)
from diff_soup.timestepping.saveat import default_output_plan, output_emitter
from diff_soup.utils.weights import wrms_norm

from ..solution import (
    fill_saveat_times,
    hold_pending_sample,
    init_saveat_outputs,
    place_saveat_sample,
    terminal_sample_slot,
)
from .contract import (
    ic_gate_tolerance,
    pack_state,
    unpack_state,
)
from .trace import (
    alloc_trace,
)

# Integer termination codes in the space result_from_code maps (see
# diff_soup._solution.result_from_code): 0 unknown, 1 success, 2 step collapse,
# 3 nonlinear divergence, 4 max steps, 5 event stop, 6 inconsistent IC,
# 7 accuracy limit.
_CODE_UNKNOWN = 0


_CODE_SUCCESS = 1


_CODE_MAX_STEPS = 4


_CODE_EVENT_STOP = 5


_CODE_INCONSISTENT_IC = 6


_CODE_ACCURACY_LIMIT = 7


# Endpoint tolerance so the final save time, reached up to a rounding error, is
# still claimed by the last step, and so an exact landing on t1 terminates.
_SAVE_TOL = 1e-10


# ---------------------------------------------------------------------------
# Driver-owned loop state
# ---------------------------------------------------------------------------
class _DriverState(eqx.Module):
    """The bounded step loop's carry (all fixed-shape leaves)."""

    engine_carry: Any
    t_prev: jax.Array
    h_prev: jax.Array
    y_last: jax.Array
    lam_last: jax.Array
    control_state: Any
    event_log: Any
    ts_output: jax.Array
    # One buffer per stored output, ``(capacity + 1, *shape)`` (the last row
    # discards masked writes), and the held-back latest sample ``(slot, rows)``
    # when the buffers are write-only (``None`` otherwise).
    outputs: Any
    pending: Any
    save_cursor: jax.Array
    valid_count: jax.Array
    control_epochs: jax.Array
    event_indices: jax.Array
    event_kinds: jax.Array
    event_directions: jax.Array
    result_codes: jax.Array
    dense_nodes: Any
    trace: Any
    step: jax.Array
    done: jax.Array
    term_code: jax.Array
    event_idx: jax.Array
    event_kind: jax.Array
    event_direction: jax.Array
    t_reached: jax.Array
    tolsf: jax.Array
    accepts: jax.Array
    rejects: jax.Array
    trial_steps: jax.Array
    newton_iters: jax.Array
    jac_refreshes: jax.Array
    factorizations: jax.Array
    linear_solves: jax.Array
    rhs_evals: jax.Array
    residual_evals: jax.Array
    precision_promotions: jax.Array
    # Reduced-precision observability accumulators (fp64 scalars), or ``None``
    # when the engine's configuration reports no such counters — keeping the
    # default fp64 solve's compiled program unchanged.
    ir_refinement_steps: Any
    fp64_fallbacks: Any
    multirhs_solves: jax.Array
    newton_failures: jax.Array
    multirhs_columns: jax.Array
    du_b_builds: jax.Array
    schur_builds: jax.Array
    saveat_interpolations: jax.Array
    event_solves: jax.Array
    # The per-attempt decision record of a checkpointed reverse solve (a
    # write-only loop buffer), or ``None``.
    decisions: Any = None
    # Whether ``outputs`` is written only by slot (each slot once, never read
    # in the loop), so it rides the loop's write-only buffer channel.
    outputs_buffered: bool = eqx.field(static=True, default=False)


def _accuracy_guard_runs(engine, dtype) -> bool:
    """Whether the accuracy-limit guard can fire, so that each attempt must form its norm.

    With weights ``1/(rtol |y| + atol)`` at the same ``y``, every row of the
    weighted state is at most ``1/rtol``, so the factor ``u ‖y‖_wrms`` is at most
    ``u (1 + δ) / rtol``, ``δ < 1`` the rounding of the norm. It can pass 1, the
    guard's threshold, only when ``rtol < 2 u``.
    """
    rtol = getattr(engine, "rtol", None)
    return not isinstance(rtol, float) or rtol < 2.0 * float(jnp.finfo(dtype).eps)


def _tolerance_scale(engine, problem, y, lam, ewt, dtype) -> jax.Array:
    """The factor the tolerances would have to grow by for the accuracy to be achievable."""
    uround = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    prior_y = jax.lax.stop_gradient(pack_state(y, lam))
    ewt = jax.lax.stop_gradient(ewt)
    if getattr(engine, "suppressalg", False):
        resolve_mask = getattr(engine, "_id_mask", None)
        raw_mask = (
            resolve_mask(problem, dtype)
            if resolve_mask is not None
            else jnp.asarray(problem.id_mask, dtype=dtype)
        )
        guard_mask = jax.lax.stop_gradient(raw_mask)
    else:
        guard_mask = None
    nrm = jax.lax.stop_gradient(wrms_norm(prior_y, ewt, mask=guard_mask))
    return jax.lax.stop_gradient(uround * nrm)


def _output_plan(cfg, problem):
    """The solve's output layout: ``cfg.outputs``, or the default state and globals."""
    if cfg.outputs is not None:
        return cfg.outputs
    return default_output_plan(int(problem.n_u), int(problem.n_lam))


def _outputs_buffered(cfg, has_affect) -> bool:
    """Whether the output buffers are written by slot alone, each slot once.

    A mutate-and-continue event rewrites the row at its switch time with the
    post-affect state, and per-epoch projected output reverts a failed
    segment's rows; both read the buffers back, so those solves keep the
    buffers on the ordinary carry.
    """
    return not has_affect and not bool(cfg.epoch_projected_output)


def _control_epoch(control: Any) -> jax.Array:
    """Return a control's traced epoch, or zero without an epoch-bearing channel."""
    if control is None:
        epoch = 0
    elif isinstance(control, dict):
        epoch = control.get("epoch", 0)
    else:
        epoch = getattr(control, "epoch", 0)
    return jnp.asarray(epoch, dtype=jnp.int32)


def _event_projection_options(engine: Any) -> Any:
    """Build the shared algebraic-projection policy for event candidates.

    Returns an [AlgebraicProjectionOptions][diff_soup.projection.AlgebraicProjectionOptions].
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionOptions

    projection_atol = ic_gate_tolerance(engine.atol, getattr(engine, "ic_tol", None))
    projection_backend = getattr(getattr(engine, "backend", None), "backend_name", "auto")
    if projection_backend not in ("dense", "klu", "cudss"):
        projection_backend = "auto"
    return AlgebraicProjectionOptions(
        rtol=float(getattr(engine, "rtol", 1e-8)),
        atol=projection_atol,
        max_steps=30,
        linear_solver=projection_backend,
        # Root-time IFT needs a time-direction JVP even when the surrounding
        # solve is transposed. This internal projection rule is transpose-safe.
        ad_mode="forward",
    )


def _loop_init(
    engine,
    problem,
    cfg,
    has_affect,
    control_folds_theta,
    ts_grid,
    ts_output,
    t0,
    t1,
    x0,
    theta,
    control0,
    atol_rows=None,
    dt0_value=None,
    history_in=None,
):
    """Build the initial driver-loop state (everything before the first attempt).

    ``atol_rows`` is the per-row absolute-tolerance vector of a
    layout-resolved solve, or ``None``. It arrives as a separate traced
    argument (the stripped configuration is static and cannot hold an
    array) and is reinstated as ``cfg.atol`` here, inside the trace, so the
    engine reads one configuration surface.

    ``history_in`` is the [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
    the solve continues from when its flag is set, or ``None``; the engine
    builds its first carry from it.

    Runs the engine's ``init_carry`` (including any consistent-initialization
    projection), seeds the save grid from the state the solve actually starts
    from, and packs the full `_DriverState`. Shared by the compiled
    solve core and by the composable forward rule, which derives the initial
    state's tangent by differentiating this function directly.
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionStatus, _project_algebraic_bidirectional

    if atol_rows is not None:
        cfg = _dc_replace(cfg, atol=atol_rows)
    if dt0_value is not None:
        cfg = _dc_replace(cfg, dt0=jax.lax.stop_gradient(jnp.asarray(dt0_value)))
    _has_affect = has_affect
    n_u = int(problem.n_u)
    dtype = x0.dtype
    u0, lam0 = unpack_state(x0, n_u)
    # The engine obtains the physical startup slope from the view's
    # residual/operator route. This value is only a shape-correct Newton seed.
    ydot0 = jnp.zeros((n_u,), dtype=dtype)
    control_channel0 = (
        control0
        if (
            _has_affect
            or problem.has_recipe_controls
            or getattr(problem, "has_operational_control", False)
        )
        else None
    )
    # The engine evaluates the model with the explicitly supplied control values;
    # with the control channel folded into the parameters (an affect solve
    # with no explicit ``control_state0``) the dynamics read them from theta
    # and the engine's control stays ``None``.
    engine_control0 = None if control_folds_theta else control0

    # Statically size the trace to the step budget, only when a trace is asked
    # for (zero cost on the default path).
    max_steps = int(cfg.max_steps)
    kind = getattr(engine, "trace_kind", "ndf")
    trace0 = alloc_trace(kind, max_steps, dtype) if cfg.trace else None

    continuing = {} if history_in is None else {"history": history_in}
    engine_carry0 = engine.init_carry(
        problem, t0, t1, u0, ydot0, lam0, theta, cfg, control=engine_control0, **continuing
    )

    # Seed the output grid from the state the solve actually starts from: the
    # engine's consistent-initialization runs inside ``init_carry``, so the
    # ``t0`` sample must read the projected state off the carry, not the raw
    # input (for a DAE the two differ on every algebraic row).
    if hasattr(engine, "state_at_entry"):
        u0_entry, lam0_entry = unpack_state(engine.state_at_entry(engine_carry0), n_u)
    else:
        u0_entry, lam0_entry = u0, lam0
    entry_state = pack_state(u0_entry, lam0_entry)
    if hasattr(engine, "slope_at_entry"):
        entry_slope = jnp.asarray(engine.slope_at_entry(engine_carry0), dtype)
    else:
        entry_slope = jnp.concatenate([ydot0, jnp.zeros((lam0.shape[0],), dtype)])
    if history_in is not None and cfg.dense:
        # A continued history starts on the interpolant of the earlier steps.
        continued_slope = jnp.asarray(engine.dense_full(engine_carry0, t0)[1], dtype)
        entry_slope = jnp.where(history_in.continues, continued_slope, entry_slope)
    capacity = int(ts_output.shape[0])
    output_plan = _output_plan(cfg, problem)
    outputs_buffered = _outputs_buffered(cfg, _has_affect)
    emit = output_emitter(output_plan, problem, theta, dtype)
    needs_lam = output_plan.needs_lam
    ts_output0, outputs0 = init_saveat_outputs(
        output_plan,
        ts_grid,
        u0_entry,
        lam0_entry,
        t0,
        capacity=capacity,
        emit=emit,
    )

    h0 = jnp.asarray(cfg.dt0, dtype)
    if history_in is not None and hasattr(engine, "step_at_entry"):
        # A continued history enters with the step its controller chose.
        carried_step = jax.lax.stop_gradient(
            jnp.asarray(engine.step_at_entry(engine_carry0), dtype)
        )
        h0 = jnp.where(history_in.continues, carried_step, h0)
    z = jnp.asarray(0, jnp.int32)
    # The fired-event index starts at -1 ("nothing fired"): a non-negative
    # value marks the declaration index of the last event that fired, whether
    # it terminated the solve or switched it and continued.
    no_event = jnp.asarray(-1, jnp.int32)
    event_log0 = (
        EventLog.empty(int(cfg.max_events), int(x0.shape[0]), dtype=dtype)
        if cfg.event is not None
        else None
    )
    # Dense-output node buffers: one (time, state, derivative) node per
    # accepted step plus the entry node. Sized by the step budget; slots
    # beyond the recorded count hold +inf times so segment lookup skips them.
    if cfg.dense:
        n_full = int(x0.shape[0])
        cap = max_steps + 1
        dense_nodes0 = (
            jnp.full((cap,), jnp.inf, dtype).at[0].set(t0),
            jnp.zeros((cap, n_full), dtype).at[0].set(entry_state),
            jnp.zeros((cap, n_full), dtype).at[0].set(entry_slope),
        )
    else:
        dense_nodes0 = None

    # Reduced-precision counter channel: engaged exactly when the engine's
    # corrector solves report in-band counters (a reduced or adaptive solve
    # mode); ``None`` otherwise so the default fp64 solve carries no counters.
    _prec = getattr(engine, "precision", None)
    _reports_reduced = getattr(_prec, "solve_mode", "fp64") in ("fp32", "fp32_ir", "adaptive")
    reduced_zero = jnp.zeros((), jnp.result_type(float)) if _reports_reduced else None

    # The explicit grid is consumed once, front to back. Accepted-step output
    # instead owns a direct-write prefix and may seed its first slot with the
    # exact entry state; it never runs the interpolation fill machinery.
    if cfg.save_every:
        save_cursor0 = jnp.asarray(int(cfg.save_initial), jnp.int32)
        valid_count0 = save_cursor0
        if cfg.save_initial:
            ts_output0 = ts_output0.at[0].set(t0)
            outputs0, _ = place_saveat_sample(
                outputs0,
                None,
                0,
                emit(t0, u0_entry, lam0_entry if needs_lam else None),
                jnp.asarray(True),
            )
    else:
        save_cursor0 = jnp.searchsorted(ts_grid, t0, side="right").astype(jnp.int32)
        valid_count0 = save_cursor0
        ts_output0 = fill_saveat_times(ts_grid, ts_output0, save_cursor0)
    sample_slots = jnp.arange(capacity, dtype=jnp.int32)
    epoch0 = _control_epoch(control_channel0 if control_channel0 is not None else control0)
    control_epochs0 = jnp.where(sample_slots < valid_count0, epoch0, no_event)
    event_indices0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    event_kinds0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    event_directions0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    result_codes0 = jnp.full((capacity,), -1, dtype=jnp.int32)

    # A terminating event already satisfied at the integration start owns an
    # exact terminal row without taking a numerical step. Continuous events
    # use their projected state policy here as they do during localization.
    initial_fired = jnp.asarray(False)
    initial_projection_failed = jnp.logical_not(
        jnp.all(jnp.isfinite(entry_state)) & jnp.all(jnp.isfinite(entry_slope))
    )
    initial_event_idx = no_event
    initial_event_kind = no_event
    initial_event_direction = no_event
    initial_state = entry_state
    initial_root_tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
    events = _events_of(cfg.event)
    recipe_controls = bool(getattr(problem, "has_recipe_controls", False)) or bool(
        getattr(problem, "has_operational_control", False)
    )
    declared_control0 = control_channel0 if (_has_affect or recipe_controls) else control0
    event_control0 = (
        problem.control_values(t0, theta, declared_control0)
        if recipe_controls
        else declared_control0
    )
    projected_initial = any(
        getattr(getattr(event, "state_policy", None), "mode", None) == "projected"
        and getattr(event, "affect", None) is None
        for event in events
    )
    projection_options = (
        _event_projection_options(engine)
        if projected_initial or cfg.epoch_projected_output
        else None
    )
    if cfg.epoch_projected_output:
        has_initial_output = save_cursor0 > 0

        def _project_entry(_):
            projected = _project_algebraic_bidirectional(
                problem,
                t=t0,
                state=entry_state,
                state_dot=entry_slope,
                seed=entry_state,
                theta=theta,
                control=event_control0,
                options=projection_options,
            )
            return projected.state, projected.status

        projected_entry_state, projected_entry_status = jax.lax.cond(
            has_initial_output,
            _project_entry,
            lambda _: (
                entry_state,
                jnp.asarray(int(AlgebraicProjectionStatus.SUCCESS), jnp.int32),
            ),
            None,
        )
        initial_output_projection_failed = jnp.logical_and(
            has_initial_output,
            projected_entry_status != int(AlgebraicProjectionStatus.SUCCESS),
        )
        projected_entry_y, projected_entry_lam = unpack_state(projected_entry_state, n_u)
        projected_rows = emit(t0, projected_entry_y, projected_entry_lam if needs_lam else None)
        keep_projected = jnp.logical_and(
            jnp.arange(capacity + 1, dtype=jnp.int32) < save_cursor0,
            ~initial_output_projection_failed,
        )
        outputs0 = tuple(
            jnp.where(keep_projected.reshape((-1,) + (1,) * row.ndim), row[None], buffer)
            for buffer, row in zip(outputs0, projected_rows)
        )
        initial_projection_failed = jnp.logical_or(
            initial_projection_failed, initial_output_projection_failed
        )
        save_cursor0 = jnp.where(initial_output_projection_failed, z, save_cursor0)
        valid_count0 = jnp.where(initial_output_projection_failed, z, valid_count0)
        ts_output0 = jnp.where(
            initial_output_projection_failed,
            jnp.full_like(ts_output0, jnp.nan),
            ts_output0,
        )
        outputs0 = tuple(
            jnp.where(initial_output_projection_failed, jnp.full_like(buffer, jnp.nan), buffer)
            for buffer in outputs0
        )
        control_epochs0 = jnp.where(
            initial_output_projection_failed,
            jnp.full_like(control_epochs0, -1),
            control_epochs0,
        )
    for index, event in enumerate(events):
        if getattr(event, "affect", None) is not None:
            continue
        active = jnp.logical_not(jnp.logical_or(initial_fired, initial_projection_failed))
        policy = getattr(event, "state_policy", None)
        if policy is not None and policy.mode == "projected":
            projected = _project_algebraic_bidirectional(
                problem,
                t=t0,
                state=entry_state,
                state_dot=entry_slope,
                seed=entry_state,
                theta=theta,
                control=event_control0,
                options=projection_options,
            )
            evaluation_ok = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
            event_state = jnp.where(evaluation_ok, projected.state, entry_state)
            initial_projection_failed = jnp.logical_or(
                initial_projection_failed,
                jnp.logical_and(active, jnp.logical_not(evaluation_ok)),
            )
        else:
            evaluation_ok = jnp.asarray(True)
            event_state = entry_state
        value = jnp.asarray(event.cond_fn(t0, event_state, theta, control=event_control0)).reshape(
            ()
        )
        root_finder = getattr(event, "root_finder", None)
        if root_finder is None:
            satisfied = jnp.asarray(value, dtype=jnp.bool_)
            kind = RESOLVE_KIND_DISCRETE
            crossing_direction = jnp.asarray(0, jnp.int32)
            tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
        else:
            value_tolerance = max(
                float(getattr(root_finder, "atol", 0.0)),
                10.0 * float(np.finfo(dtype).eps),
            )

            def _condition_along_entry(tq):
                delta = tq - t0
                state_seed = entry_state + delta * entry_slope
                control_tq = (
                    problem.control_values(tq, theta, declared_control0)
                    if recipe_controls
                    else declared_control0
                )
                if policy is not None and policy.mode == "projected":
                    projected_tq = _project_algebraic_bidirectional(
                        problem,
                        t=tq,
                        state=state_seed,
                        state_dot=entry_slope,
                        seed=state_seed,
                        theta=theta,
                        control=control_tq,
                        options=projection_options,
                    )
                    state_tq = projected_tq.state
                else:
                    state_tq = state_seed
                return jnp.asarray(event.cond_fn(tq, state_tq, theta, control=control_tq)).reshape(
                    ()
                )

            _, outgoing_slope = jax.jvp(
                _condition_along_entry,
                (jnp.asarray(t0, dtype=dtype),),
                (jnp.asarray(1.0, dtype=dtype),),
            )
            crossing_direction = jnp.where(
                outgoing_slope > 0,
                jnp.asarray(1, jnp.int32),
                jnp.where(
                    outgoing_slope < 0,
                    jnp.asarray(-1, jnp.int32),
                    jnp.asarray(0, jnp.int32),
                ),
            )
            declared_direction = getattr(event, "direction", "any")
            if declared_direction == "up":
                direction_matches = crossing_direction > 0
            elif declared_direction == "down":
                direction_matches = crossing_direction < 0
            else:
                direction_matches = jnp.asarray(True)
            satisfied = jnp.logical_and(
                jnp.abs(value) <= jnp.asarray(value_tolerance, dtype=value.dtype),
                direction_matches,
            )
            kind = RESOLVE_KIND_CONTINUOUS
            tolerance = jnp.maximum(
                jnp.asarray(float(getattr(root_finder, "atol", 0.0)), dtype=dtype),
                jnp.asarray(float(getattr(root_finder, "rtol", 0.0)), dtype=dtype),
            )
        take = jnp.logical_and(
            active,
            jnp.logical_and(evaluation_ok, satisfied),
        )
        initial_fired = jnp.logical_or(initial_fired, take)
        initial_event_idx = jnp.where(take, jnp.asarray(index, jnp.int32), initial_event_idx)
        initial_event_kind = jnp.where(take, jnp.asarray(kind, jnp.int32), initial_event_kind)
        initial_event_direction = jnp.where(take, crossing_direction, initial_event_direction)
        initial_state = jnp.where(take, event_state, initial_state)
        initial_root_tolerance = jnp.where(take, tolerance, initial_root_tolerance)

    initial_y, initial_lam = unpack_state(initial_state, n_u)
    terminal_slot0, terminal_collides0, terminal_room0 = terminal_sample_slot(
        ts_output0,
        valid_count0,
        t0,
        initial_root_tolerance,
    )
    safe_terminal_slot0 = jnp.minimum(terminal_slot0, jnp.asarray(capacity - 1, jnp.int32))
    write_initial_terminal = jnp.logical_and(initial_fired, terminal_room0)

    def _write_initial(buffer, value):
        written = buffer.at[safe_terminal_slot0].set(value)
        return jnp.where(write_initial_terminal, written, buffer)

    ts_output0 = _write_initial(ts_output0, jnp.asarray(t0, dtype=dtype))
    outputs0, _ = place_saveat_sample(
        outputs0,
        None,
        safe_terminal_slot0,
        emit(t0, initial_y, initial_lam if needs_lam else None),
        write_initial_terminal,
    )
    control_epochs0 = _write_initial(control_epochs0, epoch0)
    event_indices0 = _write_initial(event_indices0, initial_event_idx)
    event_kinds0 = _write_initial(event_kinds0, initial_event_kind)
    event_directions0 = _write_initial(event_directions0, initial_event_direction)
    result_codes0 = _write_initial(result_codes0, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
    valid_count0 = jnp.where(
        write_initial_terminal,
        valid_count0 + jnp.logical_not(terminal_collides0).astype(jnp.int32),
        valid_count0,
    )
    if cfg.event is not None:
        appended0 = event_log0.append(
            t=t0,
            index=initial_event_idx,
            kind=initial_event_kind,
            direction=initial_event_direction,
            y=initial_state,
            pre_y=initial_state,
            pre_epoch=epoch0,
            post_epoch=epoch0,
        )
        event_log0 = jax.tree_util.tree_map(
            lambda new, old: jnp.where(initial_fired, new, old),
            appended0,
            event_log0,
        )
    # Write-only buffers take each slot once: hold the latest sample back in
    # the carry, where an exact terminal sample at its time may replace it.
    pending0 = None
    if outputs_buffered:
        outputs0, pending0 = hold_pending_sample(outputs0, valid_count0)
    initial_done = jnp.logical_or(initial_fired, initial_projection_failed)
    initial_term_code = jnp.where(
        initial_fired,
        jnp.asarray(_CODE_EVENT_STOP, jnp.int32),
        jnp.where(
            initial_projection_failed,
            jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
            jnp.asarray(_CODE_UNKNOWN, jnp.int32),
        ),
    )
    init_state = _DriverState(
        engine_carry=engine_carry0,
        t_prev=t0,
        h_prev=h0,
        y_last=jnp.where(initial_fired, initial_y, u0_entry),
        lam_last=jnp.where(initial_fired, initial_lam, lam0_entry),
        control_state=control_channel0,
        event_log=event_log0,
        ts_output=ts_output0,
        save_cursor=save_cursor0,
        valid_count=valid_count0,
        control_epochs=control_epochs0,
        event_indices=event_indices0,
        event_kinds=event_kinds0,
        event_directions=event_directions0,
        result_codes=result_codes0,
        dense_nodes=dense_nodes0,
        outputs=outputs0,
        pending=pending0,
        trace=trace0,
        step=z,
        done=initial_done,
        term_code=initial_term_code,
        event_idx=jnp.where(initial_fired, initial_event_idx, no_event),
        event_kind=jnp.where(initial_fired, initial_event_kind, no_event),
        event_direction=jnp.where(initial_fired, initial_event_direction, no_event),
        t_reached=t0,
        tolsf=jnp.asarray(0.0, dtype),
        accepts=z,
        rejects=z,
        trial_steps=z,
        newton_iters=z,
        jac_refreshes=z,
        factorizations=z,
        linear_solves=z,
        rhs_evals=z,
        residual_evals=z,
        precision_promotions=jnp.zeros((4,), jnp.int32),
        ir_refinement_steps=reduced_zero,
        fp64_fallbacks=reduced_zero,
        multirhs_solves=z,
        newton_failures=z,
        multirhs_columns=z,
        du_b_builds=z,
        schur_builds=z,
        saveat_interpolations=z,
        event_solves=z,
        outputs_buffered=outputs_buffered,
    )
    return init_state
