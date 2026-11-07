"""One solve over a schedule's segments, as a scan with carried state and masked idle segments."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._schedule.motion import _field_units, _segment_solve
from diff_soup._schedule.records import _ARRAY_FIELDS, ScheduleArrays, SegmentEnds, _as_real
from diff_soup._solution import RESULT
from diff_soup.problem.implicit import solver_view
from diff_soup.timestepping import ControlState
from diff_soup.timestepping.event_records import TerminalRecord
from diff_soup.timestepping.vsvo.solution import Solution, SolveHistory

# A segment that does not run still executes one solve, because under ``vmap``
# both sides of a choice execute. That solve spans this fraction of the whole
# solve, starting where the last segment that ran ended, and its outputs are
# discarded.
_IDLE_FRACTION = 1e-6


class _Samples(NamedTuple):
    """The merged per-sample buffers of a scheduled solve."""

    ts: Any
    ys: Any
    lambdas: Any
    valid: Any
    epochs: Any
    event_indices: Any
    event_kinds: Any
    event_directions: Any
    result_codes: Any


class _Carry(NamedTuple):
    """What one segment hands to the next."""

    y: Any
    end: Any
    last_mode: Any
    last_target: Any
    last_end: Any
    failed: Any
    result: Any
    samples: _Samples
    step: Any
    last_terms: Any
    last_start: Any
    history: Any = None


def _may_carry(schedule: ScheduleArrays) -> bool:
    """Whether a segment of ``schedule`` may continue where the one before it ended.

    ``False`` only when the ``carry_step`` leaf is concrete and every entry is
    ``False``; a traced leaf may hold anything.
    """
    flags = schedule.carry_step
    if isinstance(flags, jax.core.Tracer):
        return True
    return bool(np.any(np.asarray(flags)))


def _segment_count(schedule: ScheduleArrays) -> int:
    """The capacity of ``schedule``; refuses leaves of different or empty lengths."""
    shapes = {name: tuple(np.shape(getattr(schedule, name))) for name in _ARRAY_FIELDS}
    first = shapes["mode"]
    if len(first) != 1 or first[0] < 1 or any(shape != first for shape in shapes.values()):
        raise ValueError(
            f"the schedule arrays {_ARRAY_FIELDS} must be one-dimensional and share one "
            f"nonzero length; got shapes {shapes}. Build them with schedule_arrays."
        )
    if schedule.terms is not None:
        terms = tuple(np.shape(schedule.terms))
        if len(terms) != 2 or terms[0] != first[0] or terms[1] < 1:
            raise ValueError(
                f"the target terms of a schedule have shape (capacity, degree) with capacity "
                f"{first[0]} and degree at least 1; got shape {terms}. Build them with "
                "schedule_arrays."
            )
    return first[0]


def _refuse_stop_conditions(schedule: ScheduleArrays) -> None:
    """Refuse a concrete nonzero ``stop_kind``: segments end on their duration only."""
    kinds = schedule.stop_kind
    if isinstance(kinds, jax.core.Tracer):
        return
    nonzero = np.flatnonzero(np.asarray(kinds).reshape(-1) != 0)
    if nonzero.size:
        raise ConfigurationError(
            f"stop_kind is nonzero at segment {int(nonzero[0])}; a segment stops when its "
            "duration has elapsed, and no other stop condition is supported. Set every "
            "stop_kind entry to 0."
        )


def _refuse_unknown_modes(schedule: ScheduleArrays, modes: tuple) -> None:
    """Refuse a concrete active mode index outside the model's mode tuple."""
    if isinstance(schedule.mode, jax.core.Tracer) or isinstance(schedule.active, jax.core.Tracer):
        return
    codes = np.asarray(schedule.mode).reshape(-1)
    active = np.asarray(schedule.active, dtype=bool).reshape(-1)
    outside = np.flatnonzero(active & ((codes < 0) | (codes >= len(modes))))
    if outside.size:
        index = int(outside[0])
        raise ConfigurationError(
            f"segment {index} selects mode index {int(codes[index])}, and the model's "
            f"operational table has the modes {modes!r}, indices 0 to {len(modes) - 1}; "
            "lower the schedule against this model with "
            "schedule_arrays(schedule, problem.operational_modes())."
        )


def _refuse_empty_span(t0: Any, t_end: Any) -> None:
    """Refuse concrete ends of the solve with ``t_end`` not after ``t0``."""
    if isinstance(t0, jax.core.Tracer) or isinstance(t_end, jax.core.Tracer):
        return
    start, stop = np.asarray(t0), np.asarray(t_end)
    if start.shape == () and stop.shape == () and not float(stop) > float(start):
        raise ValueError(
            f"t_end ({float(stop)!r}) must be after t0 ({float(start)!r}): a schedule is "
            "solved over the times after t0 up to t_end."
        )


def _full_state(view: Any, y0: Any) -> Any:
    """``y0`` as the full state, field unknowns then bordered globals.

    A state of the field unknowns alone takes the globals ``pdesolve`` would
    give it: the bordered part of the solver view's own translation of that
    state, which holds the values the model declares.
    """
    n_u, n_lam = int(view.n_u), int(view.n_lam)
    full = n_u + n_lam
    if y0.ndim != 1 or int(y0.shape[0]) not in (n_u, full):
        if n_lam:
            accepted = (
                f"of length {full} (the field unknowns, then the model's {n_lam} bordered "
                f"globals) or of length {n_u} (the field unknowns alone; the globals then "
                "start from the values the model declares)"
            )
        else:
            accepted = f"of length {full}"
        raise ValueError(
            f"the initial state has shape {tuple(y0.shape)}; a schedule starts from a "
            f"one-dimensional state {accepted}."
        )
    if int(y0.shape[0]) == full:
        return y0
    translated = jnp.asarray(view.to_internal_state(y0))
    if translated.shape != (full,):
        raise ValueError(
            f"this model does not give its {n_lam} bordered globals a start of their own; "
            f"pass the initial state of length {full}, the field unknowns then the globals."
        )
    return jnp.concatenate([y0, translated[n_u:].astype(y0.dtype)])


def _configured_step(solver: Any) -> float:
    """The first step ``solver`` is configured with (its ``dt0`` attribute), ``0`` without one.

    Raises:
        TypeError: If the attribute is not a real number.
    """
    value = getattr(solver, "dt0", None)
    return 0.0 if value is None else _as_real(value, "the dt0 of the solver")


def _next_step(solution: Solution, dtype: Any) -> Any:
    """The step ``solution``'s solve would have taken next, ``0`` when its solver reports none."""
    step = getattr(solution.stats, "last_step", None)
    if step is None:
        return jnp.zeros((), dtype=dtype)
    return jax.lax.stop_gradient(jnp.asarray(step).astype(dtype))


def _history_template(solve: Any) -> Optional[SolveHistory]:
    """A history that does not continue, shaped as the one ``solve()`` returns.

    ``solve`` is the first segment's solve asked for its history. It is
    traced for the shapes only, never compiled or run. ``None`` when its
    solution carries no history.
    """
    shapes = jax.eval_shape(lambda: solve().history)
    if shapes is None:
        return None
    zeros = jax.tree_util.tree_map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes)
    return SolveHistory(continues=jnp.zeros((), dtype=bool), state=zeros.state)


def _merge(merged: Any, part: Any, take: Any, count: int) -> Any:
    """``merged`` with the rows where ``take`` holds replaced by ``part``'s."""
    if merged is None or part is None:
        return merged
    part = jnp.asarray(part)[:count]
    mask = jnp.reshape(take, take.shape + (1,) * (part.ndim - 1))
    return jnp.where(mask, part.astype(merged.dtype), merged)


def solve_schedule(
    problem: Any,
    solver: Any,
    schedule: ScheduleArrays,
    y0: Any,
    *,
    t0: Any,
    t_end: Any,
    times: Any,
    theta: Any = None,
    adjoint: Any = None,
    dt0: Any = None,
    max_steps: int = 10_000,
) -> tuple[Solution, SegmentEnds]:
    """Solve the segments of a schedule one after another and merge their samples.

    Segment ``k`` runs over ``(a_k, b_k]``, where ``a_0 = t0``, each later
    segment starts where the one before it ends, and ``b_k`` is its end cut at
    ``t_end``. A segment runs when it is active, starts before ``t_end`` and no
    earlier segment failed. It is one ``pdesolve`` from the carried state, with
    ``ControlState.operating(mode=..., target=...)`` as its control state, the
    solver's default consistent initialization, and ``times`` as ordered save
    times. The next segment starts from the state the solve ended at,
    ``Solution.terminal``, handed over in the form ``pdesolve`` takes: the solver
    keeps a field declared with a ``scale`` or ``offset`` as
    ``(value - offset) / scale``, so the terminal field unknowns are mapped back
    with ``offset + scale * value``; the bordered globals are taken as they
    stand, as ``pdesolve`` takes them.

    Sample ``i`` belongs to the running segment with ``a_k < times[i] <= b_k``;
    the first segment also owns ``times[i] == t0``. The merged solution takes
    each sample from its segment, in the solver's units as ``pdesolve`` saves
    it; a sample without one, or not valid there, is not valid (``NaN``, epoch
    ``-1``). ``control_epochs`` holds each sample's
    segment index; ``result`` is the result of the segment that failed, success
    otherwise; the counters of ``stats`` add up the segments that ran,
    ``t_reached``, ``terminal`` and ``last_step`` come from the last one,
    ``ic_alg_residual_norm`` is the largest, and ``first_step``, where the
    solver reports it, is the first step of the first segment that ran.

    A segment whose ``carry_step`` entry is set continues the integrator where
    the last segment that ran ended: its solve is given that segment's
    ``Solution.history`` (``history=``), from which it takes the step size,
    the order and the history of past steps. Every other segment starts
    afresh from ``dt0``. The history's shape comes from one trace of the
    first segment's solve, which is not compiled or run. When the solver keeps
    no history, a segment whose entry is set starts instead from the step the
    last segment would have taken next (``SolverStats.last_step``, held out of
    differentiation), and a solve that reports no step (``last_step`` of
    ``0``) leaves nothing to carry. When no segment can continue (a concrete
    ``carry_step`` leaf of ``False`` only), each segment's solve is called as
    without the option.

    A segment with target terms holds
    ``target + terms[0] tau + terms[1] tau**2 + ...`` with ``tau`` the time
    since its start, through ``ControlState.operating(target_terms=...,
    target_origin=<its start>)``; its consistent initialization reads the
    target at ``tau = 0``.

    With more than one segment the segments run inside ``jax.lax.scan``, so the
    program holds one solve whatever the capacity. A segment that does not run
    still executes a short solve whose outputs are discarded and whose work is
    not counted. Inside the scan the solve cannot see a differentiation applied
    around it: reverse mode then differentiates through JAX's transposition of
    the forward rule. Every number may be traced; the function is valid under
    ``jax.jit``, ``jax.vmap``, ``jax.jacfwd``, and ``jax.grad`` with a reverse
    adjoint such as ``CheckpointAdjoint``.

    The adaptive mesh stays fixed under differentiation, and a segment's ends
    move along the flow: moving the start by ``s`` starts the solve from
    ``y - s f + s**2 / 2 f'`` and moving the end by ``e`` moves the terminal
    state by ``e f + e**2 / 2 f'``, with ``f`` and ``f'`` the first and second
    time derivatives of the consistent state there. First and second
    derivatives in the segment ends (the durations, ``t0``, ``t_end``) are
    exact, in any composition with derivatives in the targets and the
    parameters; a third raises ``NotImplementedError``.
    A sample exactly at a switch belongs to the preceding segment; its
    derivative is that segment's smooth endpoint limit. A fixed query crossing
    a discontinuous operating switch has no two-sided derivative.

    Args:
        problem: What ``pdesolve`` accepts; its model must declare an
            operational control table.
        solver: The solver, as for ``pdesolve``.
        schedule: The [ScheduleArrays][diff_soup._schedule.ScheduleArrays],
            lowered against this model's mode names.
        y0: The initial state: the field unknowns then the bordered globals, or
            the field unknowns alone, the globals then starting from the values
            the model declares, as ``pdesolve`` treats them.
        t0: The time of the initial state.
        t_end: The last sample time; no segment runs past it.
        times: The sample times, strictly increasing. Entries after ``t_end``
            pad and are not valid.
        theta: The parameters, as for ``pdesolve``.
        adjoint: The differentiation strategy, as for ``pdesolve``.
        max_steps: Step-attempt budget for each segment, as for ``pdesolve``.
            The default is 10,000, including rejected attempts.
        dt0: The first step a segment's solve starts from when it carries no
            step, passed to its ``pdesolve`` as ``dt0=``; ``0`` lets the
            integrator estimate it, and it may be traced. ``None`` passes no
            first step, so such a segment starts as the solver does by default;
            when a segment may carry a step, ``None`` passes the first step
            the solver is configured with (its ``dt0`` attribute), or ``0``
            when it names none.

    Returns:
        ``(solution, ends)``: the merged ``Solution`` and the
        [SegmentEnds][diff_soup._schedule.SegmentEnds].

    Raises:
        TypeError: If ``schedule`` is not a ``ScheduleArrays``.
        ValueError: If the arrays differ in length, the initial state has
            neither accepted length, or a concrete ``t_end`` is not after ``t0``.
        ConfigurationError: If the model declares no operational control table,
            a concrete mode index is outside the model's modes, a concrete
            ``stop_kind`` entry is nonzero, or the problem translates its initial
            state in a way the carried state cannot be mapped back through.
    """
    if not isinstance(schedule, ScheduleArrays):
        raise TypeError(
            f"schedule must be a ScheduleArrays; got {type(schedule).__name__}. Lower a "
            "Schedule with schedule_arrays(schedule, problem.operational_modes())."
        )
    count = _segment_count(schedule)
    _refuse_stop_conditions(schedule)
    view = solver_view(problem)
    table = getattr(view, "operational_table", None)
    if table is None:
        raise ConfigurationError(
            "a schedule selects rows of the model's operational control table, and this "
            "model declares none; declare one with spec.control.operational(...)."
        )
    _refuse_unknown_modes(schedule, tuple(table.modes))
    _refuse_empty_span(t0, t_end)
    units = _field_units(view)

    y0 = jnp.asarray(y0)
    # Wide enough to hold the solve's own precision exactly: the carried state
    # and the merged samples round-trip through it without loss.
    dtype = jnp.promote_types(y0.dtype, jax.dtypes.canonicalize_dtype(jnp.float64))
    start_state = _full_state(view, y0.astype(dtype))
    t0 = jnp.asarray(t0, dtype=dtype)
    t_end = jnp.asarray(t_end, dtype=dtype)
    times = jnp.asarray(times, dtype=dtype)
    samples_count = int(times.shape[0])
    n_u, n_lam = int(view.n_u), int(view.n_lam)
    idle = _IDLE_FRACTION * (t_end - t0)
    carrying = _may_carry(schedule)
    if carrying:
        base_step = jnp.asarray(_configured_step(solver) if dt0 is None else dt0, dtype=dtype)
    with_terms = schedule.terms is not None
    segments = (
        jnp.arange(count, dtype=jnp.int32),
        jnp.asarray(schedule.mode, dtype=jnp.int32),
        jnp.asarray(schedule.target, dtype=dtype),
        jnp.asarray(schedule.duration, dtype=dtype),
        jnp.asarray(schedule.active, dtype=bool),
        jnp.asarray(schedule.carry_step, dtype=bool) if carrying else None,
        jnp.asarray(schedule.terms, dtype=dtype) if with_terms else None,
    )

    def segment_solve(carry, segment, **extra):
        """Segment ``segment``'s solve from ``carry``, and where it runs."""
        index, mode, target, duration, active, _, terms = segment
        start = carry.end
        end = start + jnp.where(active, duration, jnp.zeros_like(duration))
        stop = jnp.minimum(end, t_end)
        run = active & (start < t_end) & jnp.logical_not(carry.failed)
        varying = {}
        if with_terms:
            # A segment that does not run keeps the last terms and their origin.
            varying = {
                "target_terms": jnp.where(run, terms, carry.last_terms),
                "target_origin": jnp.where(run, start, carry.last_start),
            }
        # Preserve the authored array aliases used by ThetaTable. Filtering
        # a LoweredProblem through custom_jvp rebuilds those arrays separately.
        solution = _segment_solve(
            view,
            solver,
            (jnp.where(run, start, carry.last_end), jnp.where(run, stop, carry.last_end + idle)),
            carry.y,
            times,
            theta,
            ControlState.operating(
                mode=jnp.where(run, mode, carry.last_mode),
                target=jnp.where(run, target, carry.last_target),
                **varying,
            ),
            dict(adjoint=adjoint, max_steps=max_steps, **extra),
        )
        return solution, start, end, stop, run

    def step(carry, segment):
        index, mode, target, _, _, carries, terms = segment
        if with_history:
            # A segment that continues takes the integrator's state from the
            # history the last segment that ran ended with.
            continues = carries & carry.history.continues
            extra = {
                "dt0": base_step,
                "history": SolveHistory(continues=continues, state=carry.history.state),
            }
        elif carrying:
            # A segment that carries a step starts from the last one's next step.
            continues = jnp.zeros((), dtype=bool)
            carried = carries & (carry.step != 0)
            extra = {"dt0": jnp.where(carried, carry.step, base_step)}
        else:
            continues = jnp.zeros((), dtype=bool)
            extra = {} if dt0 is None else {"dt0": dt0}
        solution, start, end, stop, run = segment_solve(carry, segment, **extra)
        solved = jnp.asarray(solution.ok, dtype=bool)
        failed_here = run & jnp.logical_not(solved)
        terminal = solution.terminal
        fields = terminal.y.astype(dtype)
        if units is not None:
            fields = units[1].astype(dtype) + units[0].astype(dtype) * fields
        reached = jnp.concatenate([fields, terminal.lam.astype(dtype)])
        owned = ((start < times) & (times <= stop)) | ((index == 0) & (times == t0))
        valid_there = jnp.asarray(solution.valid_mask, dtype=bool)[:samples_count]
        take = run & owned & valid_there
        merged = carry.samples
        samples = _Samples(
            ts=_merge(merged.ts, solution.ts, take, samples_count),
            ys=_merge(merged.ys, solution.ys, take, samples_count),
            lambdas=_merge(merged.lambdas, solution.lambdas, take, samples_count),
            valid=merged.valid | take,
            epochs=jnp.where(take, index, merged.epochs),
            event_indices=_merge(merged.event_indices, solution.event_indices, take, samples_count),
            event_kinds=_merge(merged.event_kinds, solution.event_kinds, take, samples_count),
            event_directions=_merge(
                merged.event_directions, solution.event_directions, take, samples_count
            ),
            result_codes=_merge(merged.result_codes, solution.result_codes, take, samples_count),
        )
        following = _Carry(
            y=jnp.where(run & solved, reached, carry.y),
            end=end,
            last_mode=jnp.where(run, mode, carry.last_mode),
            last_target=jnp.where(run, target, carry.last_target),
            last_end=jnp.where(run, stop, carry.last_end),
            failed=carry.failed | failed_here,
            result=RESULT.where(failed_here, solution.result, carry.result),
            samples=samples,
            step=(
                jnp.where(run, _next_step(solution, dtype), carry.step)
                if carrying and not with_history
                else None
            ),
            last_terms=jnp.where(run, terms, carry.last_terms) if with_terms else None,
            last_start=jnp.where(run, start, carry.last_start) if with_terms else None,
            history=(
                jax.tree_util.tree_map(
                    lambda new, old: jnp.where(run, new, old), solution.history, carry.history
                )
                if with_history
                else None
            ),
        )
        # The work of a segment that did not run is not counted.
        stats = jax.tree_util.tree_map(
            lambda leaf: jnp.where(run, leaf, jnp.zeros_like(leaf)), solution.stats
        )
        return following, (run, stats, terminal, run & continues)

    def codes():
        return jnp.full((samples_count,), -1, dtype=jnp.int32)

    initial = _Carry(
        y=start_state,
        end=t0,
        last_mode=segments[1][0],
        last_target=segments[2][0],
        last_end=t0,
        failed=jnp.zeros((), dtype=bool),
        result=RESULT.SUCCESS,
        step=jnp.zeros((), dtype=dtype) if carrying else None,
        last_terms=segments[6][0] if with_terms else None,
        last_start=t0 if with_terms else None,
        samples=_Samples(
            ts=jnp.full((samples_count,), jnp.nan, dtype=dtype),
            ys=jnp.full((samples_count, n_u), jnp.nan, dtype=dtype),
            lambdas=None if n_lam == 0 else jnp.full((samples_count, n_lam), jnp.nan, dtype=dtype),
            valid=jnp.zeros((samples_count,), dtype=bool),
            epochs=codes(),
            event_indices=codes(),
            event_kinds=codes(),
            event_directions=codes(),
            result_codes=codes(),
        ),
    )
    # A history to continue needs a later segment that may continue and a
    # solver that keeps one; its shape is the first segment's.
    first_segment = tuple(None if values is None else values[0] for values in segments)
    template = None
    if carrying and count > 1:
        template = _history_template(
            lambda: segment_solve(initial, first_segment, dt0=base_step, history=True)[0]
        )
    with_history = template is not None
    if with_history:
        initial = initial._replace(history=template, step=None)
    if count == 1:
        final, per_segment = step(initial, first_segment)
        per_segment = jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, 0), per_segment)
    else:
        final, per_segment = jax.lax.scan(step, initial, segments)
    ran, stats, terminals, continued = per_segment

    last = jnp.max(jnp.where(ran, jnp.arange(count, dtype=jnp.int32), -1))
    any_ran = last >= 0
    at = jnp.maximum(last, 0)
    picked = jax.tree_util.tree_map(lambda leaf: leaf[at], terminals)
    unset = TerminalRecord.empty(
        int(picked.y.shape[0]), int(picked.lam.shape[0]), dtype=picked.t.dtype
    )
    terminal = jax.tree_util.tree_map(
        lambda kept, empty: jnp.where(any_ran, kept, empty), picked, unset
    )
    totals = jax.tree_util.tree_map(lambda leaf: jnp.sum(leaf, axis=0, dtype=leaf.dtype), stats)
    totals = eqx.tree_at(
        lambda record: (record.t_reached, record.ic_alg_residual_norm, record.status),
        totals,
        (
            jnp.where(any_ran, stats.t_reached[at], t0.astype(stats.t_reached.dtype)),
            jnp.max(stats.ic_alg_residual_norm, axis=0),
            final.result,
        ),
    )
    if hasattr(totals, "first_step"):
        # A first step is a size, not a count: the first segment that ran gives it.
        first = jnp.argmax(ran)
        totals = eqx.tree_at(lambda record: record.first_step, totals, stats.first_step[first])
    if hasattr(totals, "last_step"):
        # So is the next step: the last segment that ran gives it.
        totals = eqx.tree_at(
            lambda record: record.last_step,
            totals,
            jnp.where(any_ran, stats.last_step[at], jnp.zeros_like(stats.last_step[at])),
        )

    merged = final.samples
    valid = merged.valid
    out_dtype = picked.y.dtype

    def states(values):
        if values is None:
            return None
        mask = jnp.reshape(valid, valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values, jnp.nan).astype(out_dtype)

    def labels(values):
        return jnp.where(valid, values, -1).astype(jnp.int32)

    solution = Solution(
        ts=states(merged.ts),
        ys=states(merged.ys),
        lambdas=states(merged.lambdas),
        valid_mask=valid,
        valid_count=jnp.sum(valid, dtype=jnp.int32),
        control_epochs=labels(merged.epochs),
        event_indices=labels(merged.event_indices),
        event_kinds=labels(merged.event_kinds),
        event_directions=labels(merged.event_directions),
        result_codes=labels(merged.result_codes),
        stats=totals,
        result=final.result,
        terminal=terminal,
    )

    def per_segment(name):
        values = getattr(stats, name, None)
        dtype = terminals.t.dtype
        if values is None:
            return jnp.full(ran.shape, jnp.nan, dtype=dtype)
        return jnp.where(ran, jnp.asarray(values, dtype=dtype), jnp.nan)

    ends = SegmentEnds(
        t_end=jnp.where(ran, terminals.t, jnp.nan),
        ran=ran,
        result_code=jnp.where(ran, terminals.result_code, -1).astype(jnp.int32),
        first_step=per_segment("first_step"),
        last_step=per_segment("last_step"),
        continued=continued,
    )
    return solution, ends
