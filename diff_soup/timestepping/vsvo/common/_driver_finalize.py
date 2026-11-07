"""The driver's final report: its status, its counts and its saved outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._solution import result_from_code
from diff_soup.timestepping.event_records import TerminalRecord
from diff_soup.timestepping.events import (
    _events_of,
)
from diff_soup.timestepping.vsvo.common._driver_state import (
    _CODE_EVENT_STOP,
    _CODE_MAX_STEPS,
    _CODE_SUCCESS,
    _accuracy_guard_runs,
    _control_epoch,
    _output_plan,
    _tolerance_scale,
)

from ..interpolation import CubicHermiteInterpolation
from ..solution import (
    SavedOutputs,
    SolveHistory,
    SolverStats,
    build_solution,
    commit_pending_sample,
)


def _finalize(
    engine, problem, cfg, adjoint, has_affect, control_folds_theta, ts_grid, init_state, final
):
    """Package the loop's final state into the shared solution record."""
    _has_affect = has_affect
    _control_folds_theta = control_folds_theta
    dtype = ts_grid.dtype
    has_ewt = hasattr(init_state.engine_carry, "ewt")
    engine_carry0 = init_state.engine_carry

    # The loop exits either because an attempt terminated it (use that code) or
    # because the step budget ran out (max steps before t1).
    term_code = jnp.where(final.done, final.term_code, jnp.asarray(_CODE_MAX_STEPS, jnp.int32))
    result = result_from_code(term_code)

    # The realized linear backend (post-``auto`` routing), the linear solve
    # mode, and the loop strategy actually in effect, recorded as static
    # metadata so a consumer can prove which backend and precision a solve
    # ran against.
    _backend = getattr(engine, "backend", None)
    _linear_name = getattr(type(_backend), "backend_name", None) if _backend is not None else None
    _precision = getattr(engine, "precision", None)
    _solve_mode = getattr(_precision, "solve_mode", None)
    if _solve_mode == "fp32" and np.dtype(_precision.compute_dtype).itemsize > 4:
        # Float32 factors under float64 compute: the fp32_factor policy, not end-to-end fp32.
        _solve_mode = "fp32_factor"
    _adjoint_ckpts = getattr(adjoint, "checkpoints", None)
    _zero_f64 = jnp.zeros((), jnp.result_type(float))

    stats = SolverStats(
        accepts=final.accepts,
        rejects=final.rejects,
        trial_steps=final.trial_steps,
        newton_iters=final.newton_iters,
        jac_refreshes=final.jac_refreshes,
        factorizations=final.factorizations,
        linear_solves=final.linear_solves,
        rhs_evals=final.rhs_evals,
        residual_evals=final.residual_evals,
        active_set_iters=jnp.asarray(0, jnp.int32),
        active_set_changes=jnp.asarray(0, jnp.int32),
        # An engine that runs a consistent-initialization pre-pass stashes the
        # achieved algebraic/global-constraint residual on its initial carry;
        # an engine that does not leaves this at zero.
        ic_alg_residual_norm=jnp.asarray(
            getattr(engine_carry0, "ic_alg_residual_norm", 0.0), dtype
        ),
        t_reached=final.t_reached,
        status=result,
        precision_promotions=final.precision_promotions,
        multirhs_solves=final.multirhs_solves,
        newton_failures=final.newton_failures,
        multirhs_columns=final.multirhs_columns,
        du_b_builds=final.du_b_builds,
        schur_builds=final.schur_builds,
        saveat_interpolations=final.saveat_interpolations,
        ic_solves=jnp.asarray(int(bool(cfg.consistent_ic)), jnp.int32),
        event_solves=final.event_solves,
        ir_refinement_steps=(
            _zero_f64 if final.ir_refinement_steps is None else final.ir_refinement_steps
        ),
        fp64_fallbacks=(_zero_f64 if final.fp64_fallbacks is None else final.fp64_fallbacks),
        # The step the first attempt takes, read on the initial carry: the
        # engine's estimate or the dt0 override, after its clamps.
        first_step=(
            jax.lax.stop_gradient(jnp.asarray(engine.step_at_entry(engine_carry0), _zero_f64.dtype))
            if hasattr(engine, "step_at_entry")
            else _zero_f64
        ),
        # The step the next attempt would take, read on the final carry.
        last_step=(
            jax.lax.stop_gradient(
                jnp.asarray(engine.step_at_entry(final.engine_carry), _zero_f64.dtype)
            )
            if hasattr(engine, "step_at_entry")
            else _zero_f64
        ),
        linear_backend=_linear_name,
        adjoint=type(adjoint).__name__,
        adjoint_checkpoints=None if _adjoint_ckpts is None else int(_adjoint_ckpts),
        linear_solve_mode=_solve_mode,
    )

    # The held-back latest sample goes to its slot, and the discard row of
    # every buffer is dropped.
    output_plan = _output_plan(cfg, problem)
    capacity = int(final.ts_output.shape[0])
    stored = tuple(
        buffer[:capacity] for buffer in commit_pending_sample(final.outputs, final.pending)
    )
    by_name = dict(zip(output_plan.names, stored))
    ys = by_name.get("state")
    lambdas = by_name.get("lambdas")
    if lambdas is not None and lambdas.shape[-1] == 0:
        lambdas = None
    saved_outputs = SavedOutputs(output_plan.names, stored) if output_plan.requested else None

    # Fixed-shape buffers leave the compiled core untrimmed; run_engine slices
    # them to the recorded attempt count where the count is concrete.
    trace_out = final.trace if cfg.trace else None

    # One flag per event (declaration order): the event that terminated the
    # solve, or — on a mutate-and-continue solve that ran to the horizon — the
    # last affect that fired (all-False when nothing fired before the end).
    # Built from the in-loop winner index with traced ops, so it works inside
    # jit and under both differentiation modes.
    event_mask = None
    if cfg.event is not None:
        stopped = jnp.equal(term_code, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
        completed = jnp.equal(term_code, jnp.asarray(_CODE_SUCCESS, jnp.int32))
        fired_any = jnp.logical_and(final.event_idx >= 0, jnp.logical_or(stopped, completed))
        event_mask = tuple(
            jnp.logical_and(fired_any, final.event_idx == i)
            for i in range(len(_events_of(cfg.event)))
        )

    # The final operating protocol is surfaced only for an affect solve seeded
    # with an explicit control pytree (matching the other engine families); a
    # control channel folded from the parameters stays internal.
    control_state = final.control_state if (_has_affect and not _control_folds_theta) else None

    # The achieved tolerance scale factor, for an engine whose carry supports
    # the accuracy-limit guard; ``None`` (no info payload at all, matching
    # today's every-solve behavior) for one that does not.
    if has_ewt and not _accuracy_guard_runs(engine, dtype):
        tolsf = _tolerance_scale(
            engine, problem, final.y_last, final.lam_last, final.engine_carry.ewt, dtype
        )
    else:
        tolsf = final.tolsf
    info = {"tolsf": tolsf} if has_ewt else None

    # The exact stopping point, independent of the requested output grid: an
    # event stop holds the crossing state, a completed solve the endpoint
    # state, a failed solve the last accepted point.
    terminal = TerminalRecord.empty(
        int(final.y_last.shape[0]), int(final.lam_last.shape[0]), dtype=dtype
    ).write(
        t=final.t_reached,
        y=final.y_last,
        lam=final.lam_last,
        result_code=term_code,
        control_epoch=_control_epoch(final.control_state),
        event_index=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_idx,
            jnp.asarray(-1, jnp.int32),
        ),
        event_kind=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_kind,
            jnp.asarray(-1, jnp.int32),
        ),
        event_direction=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_direction,
            jnp.asarray(-1, jnp.int32),
        ),
    )

    interpolation = None
    if cfg.dense and final.dense_nodes is not None:
        dn_t, dn_y, dn_yd = final.dense_nodes
        interpolation = CubicHermiteInterpolation(
            ts=dn_t, ys=dn_y, ydots=dn_yd, count=final.accepts + 1
        )

    # The record a later solve continues from: the final carry's, which a
    # later solve may continue only when this one reached its end time.
    history = None
    if cfg.keep_history:
        history = SolveHistory(
            continues=jnp.equal(term_code, jnp.asarray(_CODE_SUCCESS, jnp.int32)),
            state=engine.history_at_exit(final.engine_carry),
        )

    return build_solution(
        ts=final.ts_output,
        ys=ys,
        lambdas=lambdas,
        valid_mask=(jnp.arange(final.ts_output.shape[0], dtype=jnp.int32) < final.valid_count),
        valid_count=final.valid_count,
        control_epochs=final.control_epochs,
        event_indices=final.event_indices,
        event_kinds=final.event_kinds,
        event_directions=final.event_directions,
        result_codes=final.result_codes,
        stats=stats,
        result=result,
        info=info,
        interpolation=interpolation,
        trace=trace_out,
        control_state=control_state,
        event_mask=event_mask,
        event_log=final.event_log,
        terminal=terminal,
        alg=engine,
        prob=problem,
        history=history,
        outputs=saved_outputs,
    )
