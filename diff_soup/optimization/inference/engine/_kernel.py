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
``frozen``      the cost and ``J^T r`` in narrow coordinates, ``r`` from a
                solve without derivatives and ``J`` the Jacobian of the lane's
                last ``blocks`` evaluation, frozen
==============  ==============================================================

with the row count, the status and the work counters of the solve in every
mode. A lane that fails returns zeros and says so in its status.

The ``frozen`` mode takes a third argument per lane: the Jacobian ``J`` of its
rows in narrow coordinates, of shape (rows, narrow width). The ``blocks``
program of a study that prepared the ``frozen`` mode (``KEEPING_BLOCKS``)
returns that Jacobian with its blocks, and the batched backend keeps it on the
device for the frozen requests that follow; a ``blocks`` program of any other
study returns none.

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

A lane may produce outputs for other lanes, and a lane may be driven by them
([Link][diff_soup.optimization.inference.Link]). A source lane also returns its outputs,
flat, in every mode but ``gradient`` in reverse mode and ``hessian``; in the ``blocks``
mode, and the ``gradient`` mode in forward mode, it returns their derivative in
its narrow coordinates from the same forward pass. A driven lane takes, per
link, the values of the source's output and their derivative as numbers, and
sets the values of its signal to ``values + tangent (x - x0)``, ``x`` the
entries of its narrow point at the source's coordinates and ``x0`` their value
at the point: the value is the solve on the source's output, and the one
``jax.jacfwd`` of the ``blocks`` mode carries the source's coordinates by the
chain rule.

A lane whose experiment computes its initial state from its parameters
([ComputedInitialState][diff_soup.optimization.inference.ComputedInitialState]) starts
from ``start(theta)``, formed from the same parameter values as the solve, links
included, in place of its ``y0``; every mode differentiates through it as it
differentiates through the solve.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from numpy.typing import DTypeLike

from diff_soup._schedule import ScheduleArrays, SegmentEnds, solve_schedule
from diff_soup.optimization.inference.engine._status import LaneStatus, lane_status
from diff_soup.timestepping.vsvo.solution import Solution

Array = jax.Array

MODES: tuple[str, ...] = ("value", "blocks", "gradient", "observe", "hessian", "frozen")
KEEPING_BLOCKS: str = "blocks, keeping J"
"""The program of the ``blocks`` mode that also returns each lane's Jacobian.

A study that prepared the ``frozen`` mode runs its ``blocks`` requests
through it. It is not a mode of a request; its records say ``blocks``.
"""
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
        outputs: Per output a source lane produces for other lanes,
            ``observe(state, t=, theta=)`` reading one saved state; empty for
            a lane that drives none.
        links: Per link of a driven lane, the handle of the signal function
            whose values the link sets; empty for a lane without links.
        link_coordinates: The trailing coordinates of a driven lane's narrow
            point that only its links read: its source's coordinates its model
            does not declare. Their place in the parameter vector is one past
            its end, and they are dropped there.
        start: For an experiment whose initial state is computed from its
            parameters, ``start(theta)`` returning that state, in the shape of
            the lane's ``y0``; ``None`` for a lane that starts from its ``y0``.
    """

    problem: Any
    solver: Any
    unflatten: Callable
    observes: tuple
    width: int
    point_width: int
    reverse: bool = False
    reverse_solver: Any = None
    outputs: tuple = ()
    links: tuple = ()
    link_coordinates: int = 0
    start: Optional[Callable] = None


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


class LinkInputs(eqx.Module):
    """The numbers of one link of a driven lane.

    Attributes:
        values: The source's output at the point, in the shape of the signal's
            ``values`` array.
        tangent: Its derivative in the source's narrow coordinates, of shape
            ``values.shape + (source width,)``; zeros in a mode that forms no
            derivative.
        columns: Per narrow coordinate of the source, its position in this
            lane's narrow point; a coordinate this lane does not read has a
            zero column of ``tangent``.
    """

    values: Array
    tangent: Array
    columns: Array


class LaneInputs(eqx.Module):
    """The numbers of one lane. A batch of lanes stacks every leaf.

    Attributes:
        columns: Per narrow coordinate, its position in the flat
            fitted-parameter vector of the lane's experiment.
        frozen: The parameters of the experiment that are not fitted.
        y0: The initial state. For a lane whose program computes its start
            from the parameters (``LaneProgram.start``), the state at the
            experiment's starting parameters, which gives the shape and fills
            a sample slot the solve did not reach.
        t0: The time of the initial state.
        t1: The end of the span, the last sample time.
        times: The sample times, strictly increasing. Entries after ``t1`` pad
            a lane that records fewer samples than its program has slots.
        streams: One [StreamInputs][diff_soup.optimization.inference.engine._kernel.StreamInputs]
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
        outputs: Per output of a source lane, the index of each of its times
            in ``times``; empty for a lane that drives none.
        links: Per link of a driven lane, its
            [LinkInputs][diff_soup.optimization.inference.engine._kernel.LinkInputs];
            empty for a lane without links.
    """

    columns: Array
    frozen: PyTree
    y0: Array
    t0: Array
    t1: Array
    times: Array
    streams: tuple
    dt0: Array
    schedule: Optional[ScheduleArrays] = None
    outputs: tuple = ()
    links: tuple = ()


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
            ``hessian``, ``frozen``).
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
            ``gradient``, ``frozen``). ``None`` for a lane without a schedule.
        jacobian: The Jacobian of the lane's rows in narrow coordinates, of
            shape (rows, narrow width), zero for a failed lane
            (``KEEPING_BLOCKS`` only). It stays on the device: the batched
            backend keeps it and never returns it.
        outputs: For a source lane, its outputs one after another, flat; zero
            for a failed lane. ``None`` for any other lane, and in the
            ``hessian`` mode and the ``gradient`` mode in reverse mode.
        output_tangents: The derivative of ``outputs`` in narrow coordinates,
            of shape (outputs, narrow width), zero for a failed lane
            (``blocks``, and ``gradient`` in forward mode). ``None`` otherwise.
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
    jacobian: Optional[Array] = None
    outputs: Optional[Array] = None
    output_tangents: Optional[Array] = None


def public_mode(mode: str) -> str:
    """The mode of a request a program serves: ``blocks`` for ``KEEPING_BLOCKS``."""
    return "blocks" if mode == KEEPING_BLOCKS else mode


def _work(solution: Solution, dtype: DTypeLike) -> LaneWork:
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


def _with_links(program: LaneProgram, point: Array, lane: LaneInputs, theta: PyTree) -> PyTree:
    """``theta`` with the values of every linked signal set from the lane's links.

    The values are ``values + tangent (x - x0)``, ``x`` the lane's narrow point
    at the source's coordinates and ``x0`` the same entries held constant: equal
    to ``values`` at the point, with the derivative ``tangent`` in ``x``.
    """
    from diff_soup.problem._registry import theta_replace

    replaced = {}
    for handle, link in zip(program.links, lane.links):
        moved = point[link.columns]
        step = moved - jax.lax.stop_gradient(moved)
        replaced[handle] = link.values + jnp.tensordot(link.tangent, step, axes=1)
    return theta_replace(theta, replaced, caller="Link")


def _computed_start(program: LaneProgram, theta: PyTree, like: Array) -> Array:
    """The initial state ``program.start`` computes from ``theta``, in the shape and dtype of ``like``."""
    value = jnp.asarray(program.start(theta), dtype=like.dtype)
    if value.shape != like.shape:
        raise ValueError(
            f"the initial state computed from the parameters has shape {value.shape}, and "
            f"at the experiment's starting parameters it had shape {like.shape}; the start "
            "has to keep one shape whatever the parameters."
        )
    return value


def _solve(program: LaneProgram, point: Array, lane: LaneInputs) -> tuple:
    """Solve one lane and return its residual rows and what describes the solve.

    As [_solve_lane][diff_soup.optimization.inference.engine._kernel._solve_lane]
    without the lane's outputs.
    """
    rows, _outputs, aux = _solve_lane(program, point, lane)
    return rows, aux


def _solve_lane(program: LaneProgram, point: Array, lane: LaneInputs) -> tuple:
    """Solve one lane and return its residual rows, its outputs and what describes the solve.

    A lane without a schedule is one solve over ``(t0, t1)``; a lane with one
    runs its segments in order through
    [solve_schedule][diff_soup._schedule.solve_schedule], and the rest reads the
    merged solution as it reads a single solve. Either starts from the lane's
    ``dt0``. A later segment of a schedule whose ``carry_step`` entry is set
    continues the integrator where the one before it ended (step size, order
    and history of past steps); across a switch that changes what the row
    demands it restarts at order one from the carried step. Every other
    segment starts afresh from the lane's ``dt0``.

    A driven lane's linked signals take their values from its links
    (``_with_links``) before the solve. A source lane reads each output at its
    times from the same solution as its streams; an output time the solve did
    not reach, or an output that is not finite, fails the lane.

    Returns ``(rows, outputs, aux)`` with ``rows`` the flat whitened residual
    (per stream its weighted residual or the least-squares rows of its
    likelihood object), zero where a frame is not observed and zero everywhere
    for a failed lane, ``outputs`` the lane's outputs one after another, flat
    and zero for a failed lane (``None`` for a lane that drives none), and
    ``aux = (row_count, status, work, predictions, ends)``, with ``ends`` the
    end time of each segment, or ``None`` for a lane without a schedule.
    """
    from diff_soup.pdesolve import pdesolve
    from diff_soup.timestepping.saveat import SaveAt

    if program.link_coordinates:
        # The coordinates only the links read land in a slot past the parameters, dropped.
        flat = jnp.zeros((program.point_width + 1,), dtype=point.dtype)
        flat = flat.at[lane.columns].set(point)[: program.point_width]
    else:
        flat = jnp.zeros((program.point_width,), dtype=point.dtype).at[lane.columns].set(point)
    theta = eqx.combine(program.unflatten(flat), lane.frozen)
    if program.links:
        theta = _with_links(program, point, lane, theta)
    y0 = lane.y0 if program.start is None else _computed_start(program, theta, lane.y0)
    if lane.schedule is None:
        solution = pdesolve(
            program.problem,
            program.solver,
            (lane.t0, lane.t1),
            y0,
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
            y0,
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
    # carries the bordered unknowns. A computed start fills them without its derivative.
    held = lane.y0 if program.start is None else jax.lax.stop_gradient(y0)
    initial = jnp.reshape(jnp.asarray(held, states.dtype), (-1,))[: states[0].size]
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
    outputs = None
    if program.outputs:
        pieces = []
        for observe, frame in zip(program.outputs, lane.outputs):
            values = jax.vmap(lambda state, t, observe=observe: observe(state, t=t, theta=theta))(
                filled[frame], lane.times[frame]
            )
            pieces.append(jnp.reshape(values, (-1,)))
            missing = missing | jnp.any(jnp.logical_not(valid[frame]))
        outputs = jnp.concatenate(pieces)
        not_finite = not_finite | jnp.logical_not(
            jnp.all(jnp.isfinite(jax.lax.stop_gradient(outputs)))
        )
    status = lane_status(solution, sample_missing=missing, not_finite=not_finite)
    flat_rows = jnp.where(status.failed, 0.0, flat_rows)
    if outputs is not None:
        outputs = jnp.where(status.failed, 0.0, outputs)
    predictions = tuple(
        jax.lax.stop_gradient(jnp.where(status.failed, 0.0, value)) for value in predictions
    )
    return flat_rows, outputs, (count, status, _work(solution, lane.dt0.dtype), predictions, ends)


def _zero_where_failed(status: LaneStatus, value: Array) -> Array:
    return jnp.where(status.failed, jnp.zeros((), dtype=value.dtype), value)


def lane_value(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost of one lane."""
    rows, outputs, (count, status, work, _predictions, ends) = _solve_lane(program, point, lane)
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=None,
        gradient=None,
        observables=None,
        work=work,
        ends=ends,
        outputs=outputs,
    )


def _blocks(program: LaneProgram, point: Array, lane: LaneInputs, *, keep: bool) -> LaneResult:
    """The cost and the normal blocks of one lane, by forward mode; with ``keep``, its Jacobian."""
    if program.reverse:
        raise ValueError(
            "the blocks of a lane are computed in forward mode, and this solver carries a "
            "reverse adjoint, which refuses forward-mode differentiation; use a solver "
            "without a reverse adjoint, or ask for the gradient mode."
        )
    outputs = tangents = None
    if program.outputs:
        # The outputs ride the same forward pass as the rows, with their derivative.
        def evaluate(q):
            rows, outputs, aux = _solve_lane(program, q, lane)
            return (rows, outputs), (rows, outputs, aux)

        (jacobian, tangents), (rows, outputs, aux) = jax.jacfwd(evaluate, has_aux=True)(point)
    else:
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
        jacobian=_zero_where_failed(status, jacobian) if keep else None,
        outputs=outputs,
        output_tangents=None if tangents is None else _zero_where_failed(status, tangents),
    )


def lane_blocks(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """The cost and the normal blocks of one lane, by forward mode.

    The residual rides the forward pass of its Jacobian, so the lane is solved
    once.
    """
    return _blocks(program, point, lane, keep=False)


def lane_keeping_blocks(program: LaneProgram, point: Array, lane: LaneInputs) -> LaneResult:
    """``lane_blocks``, and the Jacobian of the lane's rows in ``jacobian``."""
    return _blocks(program, point, lane, keep=True)


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
            outputs=blocks.outputs,
            output_tangents=blocks.output_tangents,
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
    rows, outputs, (count, status, work, predictions, ends) = _solve_lane(program, point, lane)
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=None,
        gradient=None,
        observables=predictions,
        work=work,
        ends=ends,
        outputs=outputs,
    )


def lane_frozen(
    program: LaneProgram, point: Array, lane: LaneInputs, jacobian: Array
) -> LaneResult:
    """The cost of one lane from a solve without derivatives, and ``J^T r`` on a given ``J``.

    ``jacobian`` is the Jacobian of the lane's rows in narrow coordinates at
    another point, of shape (rows, narrow width): the one its last ``blocks``
    evaluation returned. At that point the gradient is the ``blocks`` mode's.
    """
    rows, outputs, (count, status, work, _predictions, ends) = _solve_lane(program, point, lane)
    return LaneResult(
        status=status,
        cost=0.5 * rows @ rows,
        rows=count,
        normal=None,
        gradient=_zero_where_failed(status, jacobian.T @ rows),
        observables=None,
        work=work,
        ends=ends,
        end_sensitivity=_end_sensitivity(ends, point),
        outputs=outputs,
    )


_LANE_FUNCTIONS = {
    "value": lane_value,
    "blocks": lane_blocks,
    KEEPING_BLOCKS: lane_keeping_blocks,
    "gradient": lane_gradient,
    "observe": lane_observe,
    "hessian": lane_hessian,
    "frozen": lane_frozen,
}


def batched_kernel(program: LaneProgram, mode: str) -> Callable[[Array, LaneInputs], LaneResult]:
    """The compiled kernel of one program in one mode, over a batch of lanes.

    Args:
        program: What the lanes share.
        mode: One of [MODES][diff_soup.optimization.inference.engine._kernel.MODES],
            or ``KEEPING_BLOCKS``.

    Returns:
        ``kernel(points, lanes)``: ``points`` holds one narrow point per lane,
        ``lanes`` one [LaneInputs][diff_soup.optimization.inference.engine._kernel.LaneInputs]
        whose leaves lead with the lane axis. The result leads with the lane
        axis too. The ``frozen`` kernel is ``kernel(points, lanes, jacobians)``,
        one Jacobian per lane. The program is compiled once per batch width.
    """
    if mode not in _LANE_FUNCTIONS:
        raise ValueError(f"a lane kernel runs in one of the modes {MODES}; got {mode!r}.")
    function = _LANE_FUNCTIONS[mode]
    return jax.jit(jax.vmap(lambda point, lane, *kept: function(program, point, lane, *kept)))


__all__ = [
    "KEEPING_BLOCKS",
    "MODES",
    "LaneInputs",
    "LaneProgram",
    "LaneResult",
    "LaneWork",
    "LinkInputs",
    "StreamInputs",
    "batched_kernel",
    "lane_blocks",
    "lane_frozen",
    "lane_gradient",
    "lane_hessian",
    "lane_keeping_blocks",
    "lane_observe",
    "lane_value",
    "public_mode",
]
