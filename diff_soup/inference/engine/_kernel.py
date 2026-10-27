"""The lane kernel: one span of one experiment solved at one parameter point.

A lane is an experiment solved in one shoot over its span, or one window of an
experiment fitted in windows that start from observed states: a shoot from
the measured state at the window's start, scored at its end. A lane receives
its parameter point in narrow coordinates, the coordinates its residual
depends on (the shared parameters, those of its owner, its own), and
everything else it needs as numbers: the parameters that are not fitted, the
initial state, the span, the sample times, the targets, the weights, the masks
and the leaves of each stream's likelihood object. The compiled program of a
group of lanes therefore depends on the model, the solver, the likelihood
families and the shapes, and on no value. Lanes are batched with
``jax.vmap``; every lane keeps its own step controller.

A stream's rows are its weighted residual, or the ``least_squares_rows`` of
its likelihood object at the targets and the predictions; either is zero on a
frame that is not observed. A lane's cost is half the sum of its squared rows.

What a lane returns has a size set by its narrow width and its sample count,
never by the number of unknowns of its model:

==============  ==============================================================
mode            returns
==============  ==============================================================
``value``       the cost, half the squared whitened residual
``blocks``      the cost, ``J^T J`` and ``J^T r`` in narrow coordinates
``gradient``    the cost and ``J^T r`` in narrow coordinates
``observe``     the predicted observables at the sample times
``hessian``     the cost, its exact Hessian and its gradient in narrow
                coordinates, the gradient by reverse mode and the Hessian by
                forward mode over it
==============  ==============================================================

with the row count, the status and the work counters of the solve in every
mode. A lane that fails returns zeros and says so in its status.

A lane starts its solve from the first step it carries, ``dt0``, a number of
the lane like every other: ``0`` lets the integrator estimate it. The work
counters report the first step the solve took, so a lane that failed can be
solved again from a fraction of it through the same compiled program.

A lane may follow a schedule of operating segments
([ScheduleArrays][diff_soup._schedule.ScheduleArrays]). Its segments are then
solved one after another from the carried state by
[solve_schedule][diff_soup._schedule.solve_schedule], whose merged solution
the lane reads exactly as it reads a single solve, and its result also holds
the time at which each segment ended. The modes, targets and durations of the
segments are numbers of the lane, so lanes whose schedules have one length
share one compiled program.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._schedule import ScheduleArrays, SegmentEnds, solve_schedule
from diff_soup.inference.engine._status import LaneStatus, lane_status

Array = jax.Array

MODES: tuple[str, ...] = ("value", "blocks", "gradient", "observe", "hessian")
_COUNTERS: tuple[str, ...] = (
    "accepts",
    "rejects",
    "trial_steps",
    "newton_iters",
    "jac_refreshes",
    "factorizations",
)


@dataclasses.dataclass(frozen=True, eq=False)
class LaneProgram:
    """What the lanes of one compiled program share. Holds no lane's numbers.

    A host record, not a pytree: the kernels close over it, and two programs
    are the same program only when they are the same object.

    Attributes:
        problem: The solver view every lane of the program is solved on.
        solver: The solver configuration.
        unflatten: Maps the flat fitted-parameter vector of one experiment to
            the fitted part of its parameter pytree.
        observes: Per stream, ``observe(states, times, theta)`` mapping the
            saved states of the stream's frames to observable space.
        width: The narrow width: the number of coordinates a lane is
            differentiated against.
        point_width: Length of the flat fitted-parameter vector of one
            experiment.
        reverse: Whether the solver carries a reverse adjoint. The
            ``gradient`` mode then differentiates in reverse mode, and the
            ``blocks`` mode is unavailable.
        reverse_solver: The solver with the reverse adjoint whose gradient
            the ``hessian`` mode differentiates forward: ``solver`` itself
            when it carries one, else ``solver`` with the reverse adjoint its
            experiment resolves for reverse mode. ``None`` when that gradient
            cannot be differentiated forward, and then the program has no
            ``hessian`` mode.
    """

    problem: Any
    solver: Any
    unflatten: Callable
    observes: tuple
    width: int
    point_width: int
    reverse: bool = False
    reverse_solver: Any = None


class StreamInputs(eqx.Module):
    """The numbers of one observation stream of one lane.

    Attributes:
        frame: Per frame, the index of its sample time in the lane's
            ``times``.
        keep: Per frame, whether the frame is observed. A padded frame is not.
        target: The observed values, one leading entry per frame.
        weight: The whitening weight of every observed value; ones for a
            stream with a likelihood object.
        likelihood: The stream's likelihood object, whose
            ``least_squares_rows(target, predicted)`` are the stream's rows, with
            its leaves as numbers of the lane: a leaf with one entry per frame
            is padded with copies of its last frame, and the factor of a
            ``CorrelatedGaussian`` over every frame with an identity block.
            ``None`` for a stream whose rows are its weighted residual. Its
            type and static fields are part of the program.
    """

    frame: Array
    keep: Array
    target: Array
    weight: Array
    likelihood: Any = None


class LaneInputs(eqx.Module):
    """The numbers of one lane. A batch of lanes stacks every leaf.

    Attributes:
        columns: Per narrow coordinate, its position in the flat
            fitted-parameter vector of the lane's experiment.
        frozen: The parameters of the experiment that are not fitted.
        y0: The initial state.
        t0: The time of the initial state.
        t1: The end of the span, the last sample time.
        times: The sample times, strictly increasing. Entries after ``t1`` pad
            a lane that records fewer samples than its program has slots.
        streams: One [StreamInputs][diff_soup.inference.engine._kernel.StreamInputs]
            per stream.
        dt0: The first step the lane's solve starts from, a scalar; ``0``
            lets the integrator estimate it. Lowering gives each lane the
            first step its experiment's solver is configured with (the
            solver's ``dt0`` attribute), else ``0``. A unit solved again at
            rung ``r`` after a failure carries ``retry_first_step_factors[r -
            1]`` times the first step its rung-0 solve took, for that call
            only; a padded copy of a unit carries ``0``. For a lane that
            follows a schedule, every segment starts from it.
        schedule: The segments the lane follows, as
            [ScheduleArrays][diff_soup._schedule.ScheduleArrays] padded to the
            segment capacity of its program, or ``None`` for a lane solved over
            its span in one solve. The lanes of one batch either all carry a
            schedule of one capacity or none does.
    """

    columns: Array
    frozen: Any
    y0: Array
    t0: Array
    t1: Array
    times: Array
    streams: tuple
    dt0: Array
    schedule: Optional[ScheduleArrays] = None


class LaneWork(eqx.Module):
    """The work counters of one lane's solve, and the first step it took.

    The counters are the solver's ``stats`` of the same names, ``0`` where the
    solver reports none.

    Attributes:
        first_step: The size of the first trial step of the solve, as the
            integrator took it: its own estimate, or the ``dt0`` the lane
            gave it. ``0`` when the solver does not report it; for a lane that
            follows a schedule, the first step of the first segment that ran.
            ``None`` only in a record built without it.
    """

    accepts: Array
    rejects: Array
    trial_steps: Array
    newton_iters: Array
    jac_refreshes: Array
    factorizations: Array
    first_step: Optional[Array] = None


class LaneResult(eqx.Module):
    """What one lane returns. Fields a mode does not compute are ``None``.

    Attributes:
        status: How the solve ended.
        cost: Half the squared whitened residual; zero for a failed lane.
        rows: The number of residual rows of the lane.
        normal: ``J^T J`` (``blocks``), or the exact Hessian of the cost
            (``hessian``), in narrow coordinates.
        gradient: ``J^T r`` in narrow coordinates (``blocks``, ``gradient``,
            ``hessian``).
        observables: Per stream, the predicted observables at the stream's
            frames; padded frames hold zeros (``observe``).
        work: The work counters of the solve, added up over the segments that
            ran for a lane that follows a schedule.
        ends: For a lane that follows a schedule, one entry per segment of its
            capacity: the time the segment ended, and for a segment that did
            not run the end of the last one before it that did; always finite,
            and not differentiated (every mode). ``None`` for a lane without a
            schedule.
        end_sensitivity: For a lane that follows a schedule, the derivative of
            each entry of ``ends`` with respect to the narrow coordinates, one
            row per segment. A segment ends when its duration has elapsed, at a
            time no parameter moves, so every entry is zero (``blocks``,
            ``gradient``). ``None`` for a lane without a schedule.
    """

    status: LaneStatus
    cost: Array
    rows: Array
    normal: Optional[Array]
    gradient: Optional[Array]
    observables: Optional[tuple]
    work: LaneWork
    ends: Optional[Array] = None
    end_sensitivity: Optional[Array] = None


def _work(solution: Any, dtype: Any) -> LaneWork:
    """The work counters of a solve, and its first step in ``dtype``."""
    stats = getattr(solution, "stats", None)

    def counter(name: str) -> Array:
        value = getattr(stats, name, None)
        if value is None:
            return jnp.zeros((), dtype=jnp.int32)
        return jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.int32))

    first = getattr(stats, "first_step", None)
    first = (
        jnp.zeros((), dtype=dtype)
        if first is None
        else jax.lax.stop_gradient(jnp.asarray(first, dtype=dtype))
    )
    return LaneWork(**{name: counter(name) for name in _COUNTERS}, first_step=first)


def _frame_mask(keep: Array, like: Array) -> Array:
    """A per-frame mask broadcast to the shape of per-frame values."""
    return jnp.reshape(keep, keep.shape + (1,) * (like.ndim - 1))


def _segment_ends(segments: SegmentEnds, t0: Array) -> Array:
    """The end time of every segment, finite and not differentiated.

    A segment that ran gives the time it reached. A segment that did not run,
    or whose end is not finite, gives the end of the last segment before it
    that ran with a finite end, and ``t0`` when there is none.
    """
    t_end = jax.lax.stop_gradient(jnp.asarray(segments.t_end))
    usable = jnp.asarray(segments.ran, dtype=bool) & jnp.isfinite(t_end)
    last = jax.lax.stop_gradient(jnp.asarray(t0, dtype=t_end.dtype))
    ends = []
    for index in range(int(t_end.shape[0])):
        last = jnp.where(usable[index], t_end[index], last)
        ends.append(last)
    return jnp.stack(ends)


def _end_sensitivity(ends: Optional[Array], point: Array) -> Optional[Array]:
    """Zero derivatives of the segment ends in narrow coordinates, or ``None`` without ends."""
    if ends is None:
        return None
    return jnp.zeros(ends.shape + point.shape, dtype=point.dtype)


def _solve(program: LaneProgram, point: Array, lane: LaneInputs) -> tuple:
    """Solve one lane and return its residual rows and what describes the solve.

    A lane without a schedule is one solve over ``(t0, t1)``; a lane with one
    runs its segments in order through
    [solve_schedule][diff_soup._schedule.solve_schedule], and the rest reads the
    merged solution as it reads a single solve. Either starts from the lane's
    ``dt0``. A later segment of a schedule whose ``carry_step`` entry is set
    continues the integrator where the one before it ended (step size, order
    and history of past steps); across a switch that changes what the row
    demands it restarts at order one from the carried step. Every other
    segment starts afresh from the lane's ``dt0``.

    Returns ``(rows, aux)`` with ``rows`` the flat whitened residual (per
    stream its weighted residual or the least-squares rows of its likelihood
    object), zero where a frame is not observed and zero everywhere for a
    failed lane, and
    ``aux = (row_count, status, work, predictions, ends)``, with ``ends`` the
    end time of each segment, or ``None`` for a lane without a schedule.
    """
    from diff_soup.pdesolve import pdesolve
    from diff_soup.timestepping.saveat import SaveAt

    flat = jnp.zeros((program.point_width,), dtype=point.dtype).at[lane.columns].set(point)
    theta = eqx.combine(program.unflatten(flat), lane.frozen)
    if lane.schedule is None:
        solution = pdesolve(
            program.problem,
            program.solver,
            (lane.t0, lane.t1),
            lane.y0,
            saveat=SaveAt(ts=lane.times, ordered=True),
            theta=theta,
            adjoint=getattr(program.solver, "adjoint", None),
            dt0=lane.dt0,
        )
        ends = None
    else:
        solution, segments = solve_schedule(
            program.problem,
            program.solver,
            lane.schedule,
            lane.y0,
            t0=lane.t0,
            t_end=lane.t1,
            times=lane.times,
            theta=theta,
            adjoint=getattr(program.solver, "adjoint", None),
            dt0=lane.dt0,
        )
        ends = _segment_ends(segments, lane.t0)
    states = solution.ys
    valid = solution.valid_mask
    if valid is None:
        valid = jnp.all(jnp.isfinite(states), axis=tuple(range(1, states.ndim)))
    # A slot the solve never reached holds no state. Give it the initial state,
    # so an observation operator sees finite numbers there; the rows of such
    # a slot are masked below, and a requested one fails the lane. The saved
    # state is the leading block of the initial state when that one also
    # carries the bordered unknowns.
    initial = jnp.reshape(jnp.asarray(lane.y0, states.dtype), (-1,))[: states[0].size]
    filled = jnp.where(_frame_mask(valid, states), states, jnp.reshape(initial, states.shape[1:]))

    rows = []
    predictions = []
    missing = jnp.zeros((), dtype=bool)
    count = jnp.zeros((), dtype=jnp.int32)
    for observe, stream in zip(program.observes, lane.streams):
        times = lane.times[stream.frame]
        predicted = observe(filled[stream.frame], times, theta)
        keep = _frame_mask(stream.keep, predicted)
        if stream.likelihood is None:
            residual = jnp.where(keep, stream.weight * (predicted - stream.target), 0.0)
        else:
            # A frame that is not observed gives zero rows, and its prediction
            # reaches no derivative through them.
            held = jnp.where(keep, predicted, jax.lax.stop_gradient(predicted))
            residual = jnp.where(
                keep, stream.likelihood.least_squares_rows(stream.target, held), 0.0
            )
        rows.append(jnp.reshape(residual, (-1,)))
        predictions.append(jnp.where(keep, predicted, 0.0))
        missing = missing | jnp.any(stream.keep & jnp.logical_not(valid[stream.frame]))
        per_frame = int(residual.size // max(int(residual.shape[0]), 1))
        count = count + per_frame * jnp.sum(stream.keep.astype(jnp.int32))
    flat_rows = jnp.concatenate(rows) if rows else jnp.zeros((0,), dtype=point.dtype)

    not_finite = jnp.logical_not(jnp.all(jnp.isfinite(jax.lax.stop_gradient(flat_rows))))
    status = lane_status(solution, sample_missing=missing, not_finite=not_finite)
    flat_rows = jnp.where(status.failed, 0.0, flat_rows)
    predictions = tuple(
        jax.lax.stop_gradient(jnp.where(status.failed, 0.0, value)) for value in predictions
    )
    return flat_rows, (count, status, _work(solution, lane.dt0.dtype), predictions, ends)


def _zero_where_failed(status: LaneStatus, value: Array) -> Array:
    return jnp.where(status.failed, jnp.zeros((), dtype=value.dtype), value)


def lane_value(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost of one lane."""
    rows, (count, status, work, _predictions, ends) = _solve(program, point, lane)
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=None,
        gradient=None,
        observables=None,
        work=work,
        ends=ends,
    )


def lane_blocks(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost and the normal blocks of one lane, by forward mode.

    The residual rides the forward pass of its Jacobian, so the lane is solved
    once.
    """
    if program.reverse:
        raise ValueError(
            "the blocks of a lane are computed in forward mode, and this solver carries a "
            "reverse adjoint, which refuses forward-mode differentiation; use a solver "
            "without a reverse adjoint, or ask for the gradient mode."
        )
    jacobian, (rows, aux) = jax.jacfwd(
        lambda q: (lambda out: (out[0], out))(_solve(program, q, lane)), has_aux=True
    )(point)
    count, status, work, _predictions, ends = aux
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=_zero_where_failed(status, jacobian.T @ jacobian),
        gradient=_zero_where_failed(status, jacobian.T @ rows),
        observables=None,
        work=work,
        ends=ends,
        end_sensitivity=_end_sensitivity(ends, point),
    )


def lane_gradient(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost of one lane and its gradient ``J^T r``.

    In reverse mode when the solver carries a reverse adjoint, which is the
    cheaper way once the narrow width is large; otherwise from the forward
    Jacobian, which gives the same numbers.
    """
    if not program.reverse:
        blocks = lane_blocks(program, point, lane)
        return LaneResult(
            status=blocks.status,
            cost=blocks.cost,
            rows=blocks.rows,
            normal=None,
            gradient=blocks.gradient,
            observables=None,
            work=blocks.work,
            ends=blocks.ends,
            end_sensitivity=blocks.end_sensitivity,
        )

    def cost(q):
        rows, aux = _solve(program, q, lane)
        return 0.5 * rows @ rows, aux

    (value, (count, status, work, _predictions, ends)), gradient = jax.value_and_grad(
        cost, has_aux=True
    )(point)
    return LaneResult(
        status=status,
        cost=value,
        rows=count,
        normal=None,
        gradient=_zero_where_failed(status, gradient),
        observables=None,
        work=work,
        ends=ends,
        end_sensitivity=_end_sensitivity(ends, point),
    )


def lane_hessian(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost of one lane, its gradient and its exact Hessian, forward over reverse.

    The gradient is taken in reverse mode through the solver with a reverse
    adjoint, ``program.reverse_solver``, and that gradient is differentiated
    again in forward mode. The Hessian is returned in ``normal``, in narrow
    coordinates, so the lanes' Hessians are summed over the tree as their
    ``blocks`` are.
    """
    if program.reverse_solver is None:
        raise ValueError(
            "the hessian mode differentiates a lane's reverse gradient forward, and the "
            "adjoints of this solver offer no such composition; ask for the blocks or the "
            "gradient mode."
        )
    reverse = dataclasses.replace(program, solver=program.reverse_solver, reverse=True)

    def cost(q):
        rows, aux = _solve(reverse, q, lane)
        return 0.5 * rows @ rows, aux

    # Not compiled apart: the solve has to see the forward push over its reverse gradient to
    # bind the derivative entries that push differentiates.
    def gradient(q):
        (value, aux), slope = jax.value_and_grad(cost, has_aux=True)(q)
        return slope, (value, slope, aux)

    hessian, (value, slope, aux) = jax.jacfwd(gradient, has_aux=True)(point)
    count, status, work, _predictions, ends = aux
    return LaneResult(
        status=status,
        cost=value,
        rows=count,
        normal=_zero_where_failed(status, hessian),
        gradient=_zero_where_failed(status, slope),
        observables=None,
        work=work,
        ends=ends,
        end_sensitivity=_end_sensitivity(ends, point),
    )


def lane_observe(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The predicted observables of one lane at its sample times."""
    rows, (count, status, work, predictions, ends) = _solve(program, point, lane)
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=None,
        gradient=None,
        observables=predictions,
        work=work,
        ends=ends,
    )


_LANE_FUNCTIONS = {
    "value": lane_value,
    "blocks": lane_blocks,
    "gradient": lane_gradient,
    "observe": lane_observe,
    "hessian": lane_hessian,
}


def batched_kernel(program: LaneProgram, mode: str) -> Callable[[Array, LaneInputs], LaneResult]:
    """The compiled kernel of one program in one mode, over a batch of lanes.

    Args:
        program: What the lanes share.
        mode: One of [MODES][diff_soup.inference.engine._kernel.MODES].

    Returns:
        ``kernel(points, lanes)``: ``points`` holds one narrow point per lane,
        ``lanes`` one [LaneInputs][diff_soup.inference.engine._kernel.LaneInputs]
        whose leaves lead with the lane axis. The result leads with the lane
        axis too. The program is compiled once per batch width.
    """
    if mode not in MODES:
        raise ValueError(f"a lane kernel runs in one of the modes {MODES}; got {mode!r}.")
    function = _LANE_FUNCTIONS[mode]
    return jax.jit(jax.vmap(lambda point, lane: function(program, point, lane)))


__all__ = [
    "MODES",
    "LaneInputs",
    "LaneProgram",
    "LaneResult",
    "LaneWork",
    "StreamInputs",
    "batched_kernel",
    "lane_blocks",
    "lane_gradient",
    "lane_hessian",
    "lane_observe",
    "lane_value",
]
