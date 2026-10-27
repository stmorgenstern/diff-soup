"""The lanes of a lowered study, as the numbers their kernel takes.

A joint problem holds one fitted problem per experiment and the layout of the
joint parameter vector, which
[lane_layout][diff_soup.inference.engine._layout.lane_layout] reads; a study
of one experiment has that experiment's fitted problem. An experiment solved
in one shoot is one lane; an experiment fitted in windows that start from
observed states is one lane per window, a shoot from the measured state at
the window's start, scored at the window's end. This module turns the lanes
that can share a compiled program into that program's
[LaneProgram][diff_soup.inference.engine._kernel.LaneProgram], one stacked
[LaneInputs][diff_soup.inference.engine._kernel.LaneInputs], the table that
gathers every lane's narrow point from the joint vector, and the tables of the
parameter tree.

Experiments share a program when their identities are equal
([lane_identity][diff_soup.inference.engine._identity.lane_identity]) and so
is their derivative width. They may differ in every number: parameter values,
initial state, span, sample times, number of frames, targets, weights, masks
and the leaves of their likelihood objects. A lane with fewer sample times or
frames than the slots of its frame class
([frame_classes][diff_soup.inference.engine._capacity.frame_classes]) is
padded, and the padding is masked. Experiments that follow a schedule carry its
segment arrays as numbers too; a group pads them with inactive segments to one
segment capacity, and an inactive segment does not run. Each lane also carries
the first step its solve starts from: the one its experiment's solver is
configured with, or ``0`` for the integrator's own estimate. The numbers are
stacked with numpy and moved to the device with ``jax.device_put``, so building
a group compiles nothing.
"""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.engine._capacity import Frames, lane_frames, smallest_frames
from diff_soup.inference.engine._identity import (
    _lane_streams,
    _likelihood_leaves,
    _window_streams,
    lane_identity,
    lane_refusal,
)
from diff_soup.inference.engine._kernel import LaneInputs, LaneProgram, StreamInputs
from diff_soup.inference.engine._layout import (
    LaneLayout,
    _window_count,
    lane_layout,
    lane_problems,
)
from diff_soup.inference.engine._reducers import TreeTables, tree_tables

Array = jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class LaneGroup:
    """The lanes that run through one compiled program, in a fixed order.

    Attributes:
        program: What the lanes share.
        lanes: The numbers of the lanes, every leaf leading with the lane axis.
        gather: Per lane and narrow coordinate, the column of the joint
            parameter vector it is read from.
        lane_ids: The identifier of each lane, which fixes the order.
        children: Per lane, its index in the lane layout of the joint problem.
        owners: Per lane, the index of its owner in the joint problem.
        local: Per lane and local narrow coordinate, its position in the
            local block of the lane's owner.
    """

    program: LaneProgram
    lanes: LaneInputs
    gather: Array
    lane_ids: tuple
    children: tuple
    owners: np.ndarray
    local: np.ndarray

    @property
    def n_lanes(self) -> int:
        """The number of lanes."""
        return len(self.lane_ids)

    def points(self, x: Any) -> Array:
        """The narrow point of every lane at the joint parameter vector ``x``."""
        return jnp.asarray(x)[self.gather]


def _check_same_program(children: tuple, lane_ids: tuple) -> None:
    """Refuse a group whose experiments need different programs, naming what differs.

    Every experiment must be one the batched lanes can take, and every
    experiment must have the identity of the first; a refusal names the
    components of the identity that differ. The lanes of one experiment share
    its fitted problem, which is checked once.
    """
    distinct: dict = {}
    for child, lane in zip(children, lane_ids):
        distinct.setdefault(id(child), (child, lane))
    children, lane_ids = zip(*distinct.values())
    for child, lane in zip(children, lane_ids):
        reason = lane_refusal(child)
        if reason is not None:
            raise ValueError(f"experiment {lane!r} cannot run in the batched lanes: {reason}.")
    first = lane_identity(children[0])
    for child, lane in zip(children[1:], lane_ids[1:]):
        identity = lane_identity(child)
        if identity != first:
            raise ValueError(
                f"experiments {lane_ids[0]!r} and {lane!r} cannot share a compiled program: "
                f"they differ in {', '.join(first.differs(identity))}. A group takes only "
                "experiments whose identities are equal."
            )


def _lane_schedules(problems: tuple, lane_ids: tuple, segments: Any) -> Any:
    """Each lane's schedule arrays padded to one capacity, or ``None`` when no lane has one.

    Reads every experiment's lowered schedule on the host and refuses what is
    a property of the group, naming an experiment: a group that mixes
    experiments with and without a schedule, a capacity smaller than a
    schedule, and a capacity given for experiments without one. The target
    terms are padded with zero coefficients to the largest target degree of
    the group. What one experiment alone cannot run with is refused by
    [lane_refusal][diff_soup.inference.engine._identity.lane_refusal].
    """
    arrays = tuple(getattr(child, "_schedule_arrays", None) for child in problems)
    scheduled = tuple(item is not None for item in arrays)
    if any(scheduled) and not all(scheduled):
        with_one = lane_ids[scheduled.index(True)]
        without = lane_ids[scheduled.index(False)]
        raise ValueError(
            f"experiment {with_one!r} follows a schedule and experiment {without!r} does not; "
            "the lanes of one group run one program, which follows a schedule in every lane "
            "or in none. Build separate groups for the experiments with a schedule and those "
            "without."
        )
    if not any(scheduled):
        if segments is not None:
            raise ValueError(
                f"segments={segments!r} gives the segment capacity of the lanes' schedules, and "
                "no experiment of this group follows a schedule; leave segments=None."
            )
        return None
    counts = tuple(int(np.shape(items.mode)[0]) for items in arrays)
    largest = max(counts)
    if segments is None:
        capacity = largest
    elif isinstance(segments, (bool, np.bool_)) or not isinstance(segments, numbers.Integral):
        raise TypeError(f"segments= is a whole number of segments; got {segments!r}.")
    else:
        capacity = int(segments)
        if capacity < largest:
            raise ValueError(
                f"experiment {lane_ids[counts.index(largest)]!r} follows {largest} segments, "
                f"more than segments={capacity} holds; give segments= at least {largest}."
            )
    degree = max(items.degree for items in arrays)
    return tuple(items.padded(capacity, degree=degree) for items in arrays)


def _configured_first_step(solver: Any, lane: Any, dtype: Any) -> np.ndarray:
    """The first step ``solver`` starts its solves from, ``0`` when it names none.

    The solver names one with a ``dt0`` attribute; ``None`` or no attribute
    leaves the estimate to the integrator.
    """
    value = getattr(solver, "dt0", None)
    if value is None:
        value = 0.0
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        first = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"the solver of experiment {lane!r} has dt0={value!r}; a lane starts its solve "
            "from one first step, a real number, 0 to let the integrator estimate it."
        ) from None
    return np.asarray(first, dtype=dtype)


def _padded_times(times: np.ndarray, capacity: int) -> np.ndarray:
    """``times`` followed by increasing entries beyond its last one."""
    t0, t1 = float(times[0]), float(times[-1])
    stride = max(t1 - t0, 1.0)
    extra = t1 + stride * np.arange(1, capacity - times.shape[0] + 1, dtype=times.dtype)
    return np.concatenate([times, extra])


def _likelihood_numbers(series: Any, capacity: int, frames: Any = None) -> Any:
    """The stream's likelihood object with its leaves as host numbers for ``capacity`` frames.

    ``frames`` are the frames of the stream the lane holds, every frame when
    ``None``. A leaf with one entry per frame is broadcast to the stream's
    shape, cut to those frames and padded with copies of its last frame. The
    factor of a correlated Gaussian whose one event holds every frame becomes
    the factor of the covariance of those frames, padded with an identity
    block: the padded frames come after the lane's own, so its own rows are
    whitened as before and each padded frame on its own. Every other leaf is
    kept. ``None`` for a stream without a likelihood object.
    """
    if getattr(series, "likelihood", None) is None:
        return None
    shape = tuple(int(size) for size in np.shape(series.values))
    numbers = []
    for _path, leaf, span in _likelihood_leaves(series):
        leaf = np.asarray(leaf)
        if span == "frame":
            leaf = np.broadcast_to(leaf, shape)
            if frames is not None:
                leaf = leaf[np.asarray(frames)]
            pad = ((0, capacity - leaf.shape[0]),) + ((0, 0),) * (len(shape) - 1)
            leaf = np.pad(leaf, pad, mode="edge")
        elif span == "event":
            if frames is not None:
                covariance = leaf @ leaf.T
                leaf = np.linalg.cholesky(covariance[np.ix_(frames, frames)])
            factor = np.eye(capacity, dtype=leaf.dtype)
            factor[: leaf.shape[0], : leaf.shape[0]] = leaf
            leaf = factor
        numbers.append(leaf)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(series.likelihood), numbers)


def _stream_arrays(
    series: Any, times: np.ndarray, capacity: int, dtype: Any, frames: Any = None
) -> tuple:
    """The host arrays of one stream of one lane, padded to ``capacity`` frames.

    ``frames`` are the frames of the stream the lane holds, every frame when
    ``None``. Returns ``(frame, keep, target, weight, likelihood)``, the fields
    of [StreamInputs][diff_soup.inference.engine._kernel.StreamInputs].
    """
    held = slice(None) if frames is None else np.asarray(frames, dtype=np.int64)
    stamps = np.asarray(series.ts, dtype=times.dtype).reshape(-1)[held]
    frame = np.searchsorted(times, stamps)
    off_grid = (frame >= times.shape[0]) | ~np.isclose(
        times[np.minimum(frame, times.shape[0] - 1)], stamps
    )
    if np.any(off_grid):
        frame = np.asarray([int(np.argmin(np.abs(times - stamp))) for stamp in stamps])
    n_frames = stamps.shape[0]
    keep = np.ones((n_frames,), dtype=bool)
    if series.frame_mask is not None:
        keep = np.asarray(series.frame_mask, dtype=bool).reshape(-1)[held]
    values = np.asarray(series.values, dtype=dtype)
    weight = (
        np.ones_like(values)
        if series.weight is None
        else np.broadcast_to(np.asarray(series.weight, dtype=dtype), values.shape)
    )
    values, weight = values[held], weight[held]
    pad = capacity - n_frames
    trailing = ((0, 0),) * (values.ndim - 1)
    return (
        np.pad(frame, (0, pad)).astype(np.int32),
        np.pad(keep, (0, pad)),
        np.pad(values, ((0, pad),) + trailing),
        np.pad(weight, ((0, pad),) + trailing),
        _likelihood_numbers(series, capacity, frames),
    )


def _lane_spans(child: Any, dtype: Any) -> tuple:
    """Per lane of one experiment, ``(y0, times, streams)``: its start, its times, what it scores.

    ``y0`` is the lane's initial state, ``times`` its sample times from the
    time of that state, and ``streams`` holds per stream the lane scores
    ``(series, frames)``, ``frames`` the frames of the stream the lane holds
    (``None`` for all of them). An experiment solved in one shoot has one
    lane, from its initial state over its sample times, holding every frame.
    An experiment fitted in observed-start windows has one lane per window, in
    time order: a window starts from the measured state of its first frame at
    that frame's time, and holds the one frame of the scored stream at its end.
    """
    from diff_soup.inference.fitting._forward_seam import observed_windows
    from diff_soup.inference.fitting.joint import _resolve_child_y0

    if not _window_count(child):
        times = np.asarray(child._union_ts, dtype=dtype).reshape(-1)
        streams = tuple((series, None) for series in child.data)
        return ((np.asarray(_resolve_child_y0(child)), times, streams),)
    seed, scored = _window_streams(child)
    windows = observed_windows(seed, scored=scored)
    return tuple(
        (np.asarray(start), np.asarray([t0, t1], dtype=dtype), ((scored, (frame,)),))
        for start, t0, t1, frame in zip(windows.starts, windows.t0, windows.t1, windows.frames)
    )


def _stack(trees: list) -> Any:
    """Stack the host leaves of equal trees on a leading lane axis, in numpy."""

    def stack(*leaves):
        if all(leaf is None for leaf in leaves):
            return None
        if any(leaf is None for leaf in leaves):
            raise ValueError("the lanes of one group disagree on which leaves they carry.")
        return np.stack([np.asarray(leaf) for leaf in leaves])

    return jax.tree_util.tree_map(stack, *trees, is_leaf=lambda leaf: leaf is None)


def _slots(frames: Any, own: tuple, lane_ids: tuple) -> Frames:
    """The slots ``frames=`` names, checked against the counts of every lane."""
    try:
        union, streams = frames
        slots = Frames(int(union), tuple(int(count) for count in streams))
    except (TypeError, ValueError):
        raise ValueError(
            "frames= gives the slots of a frame class as (union sample times, (frames per "
            f"stream, ...)); got {frames!r}."
        ) from None
    needed = smallest_frames(own)
    if len(slots.streams) != len(needed.streams):
        raise ValueError(
            f"frames= gives frame slots for {len(slots.streams)} streams, and the experiments "
            f"record {len(needed.streams)} streams; give one count per stream."
        )
    for lane, counts in zip(lane_ids, own):
        if counts.union > slots.union or any(
            count > slot for count, slot in zip(counts.streams, slots.streams)
        ):
            raise ValueError(
                f"experiment {lane!r} records {counts.union} sample times and "
                f"{counts.streams} frames per stream, more than frames=({slots.union}, "
                f"{slots.streams}) holds; give frames= at least ({needed.union}, "
                f"{needed.streams})."
            )
    return slots


def lane_group(
    joint: Any, children: Any = None, *, frames: Any = None, segments: Any = None
) -> LaneGroup:
    """The lanes of a group of a joint problem's lanes, or of one experiment's.

    An experiment solved in one shoot has one lane; one fitted in
    observed-start windows has one lane per window, in time order, each
    solved from the measured state at its start and scored at its end. Every
    lane is padded to the same sample slots: the union of its sample
    times to ``frames[0]`` entries and each stream to its entry of
    ``frames[1]``. A padded sample time lies beyond the lane's last one and a
    padded frame is masked, so padding adds no residual row. When the
    experiments follow schedules, each lane carries its own schedule's arrays,
    padded with inactive segments to ``segments`` entries and with zero target
    terms to the group's largest target degree; an inactive segment does not
    run. Every lane starts its solve from the first step its
    experiment's solver is configured with (its ``dt0`` attribute), else from
    the integrator's own estimate. The lanes' numbers are stacked on the host
    and moved to the device in one transfer.

    Args:
        joint: A joint problem, or the fitted problem of one experiment.
        children: The indices of the lanes of the group in the problem's
            [lane_layout][diff_soup.inference.engine._layout.lane_layout], in
            any order; every lane when omitted. When every experiment has one
            lane, lane ``i`` is experiment ``i``. The lanes are ordered by
            index.
        frames: The slots of the group's frame class,
            ``(union sample times, (frames per stream, ...))``, the time of the
            initial state counted among the sample times; see
            [frame_classes][diff_soup.inference.engine._capacity.frame_classes].
            When omitted, the smallest slots that hold every lane of the group.
        segments: The segment capacity of the group's schedules: the number of
            segment entries every lane carries. When omitted, the largest
            segment count of the group. Only for experiments that follow a
            schedule.

    Returns:
        The [LaneGroup][diff_soup.inference.engine._inputs.LaneGroup].

    Raises:
        ValueError: If the experiments cannot share a compiled program (the
            message names what differs), a lane records more samples than
            ``frames`` holds, the group mixes experiments with and without a
            schedule, an experiment's model declares an operational control
            table and the experiment follows no schedule, a schedule has a
            nonzero ``stop_kind``, a stream's ``CorrelatedGaussian`` couples
            frames a lane cannot pad or mask, a schedule has more segments than
            ``segments``, ``segments`` is given for experiments without a
            schedule, or a solver's ``dt0`` is not a real number.
        TypeError: If ``segments`` is not a whole number.
    """
    return _lane_group(
        lane_layout(joint), lane_problems(joint), children, frames=frames, segments=segments
    )


def _reverse_solver(child: Any) -> Any:
    """The solver ``child`` takes reverse-mode gradients with, or ``None`` without a Hessian.

    ``None`` when the experiment's adjoints cannot differentiate that gradient
    forward over reverse.
    """
    if "forward-over-reverse" not in getattr(child, "supported_compositions", frozenset()):
        return None
    return child._resolved_solver("reverse")


def _lane_group(
    layout: LaneLayout,
    every: tuple,
    children: Any,
    *,
    frames: Any,
    segments: Any,
    device: Any = None,
) -> LaneGroup:
    """``lane_group`` of the lanes ``layout`` places, whose fitted problems are ``every``.

    The lanes' numbers are moved to ``device``, the default device for ``None``.
    """
    n_children = len(every)
    selected = tuple(range(n_children)) if children is None else tuple(sorted(set(children)))
    if not selected or selected[0] < 0 or selected[-1] >= n_children:
        raise ValueError(f"the lanes of a group are indices in 0 .. {n_children - 1}.")
    problems = tuple(every[index] for index in selected)
    lane_ids = tuple(layout.lanes[index] for index in selected)
    _check_same_program(problems, lane_ids)
    schedules = _lane_schedules(problems, lane_ids, segments)

    n_shared = layout.shared
    width = len(layout.columns[selected[0]])
    first = problems[0]
    dtype = np.asarray(first._seam.x0).dtype
    point_width = layout.widths[selected[0]]

    own = tuple(lane_frames(child) for child in problems)
    slots = smallest_frames(own) if frames is None else _slots(frames, own, lane_ids)

    # A lane's place among the lanes of its experiment picks its span of that experiment.
    first_lane: dict = {}
    for lane_index, experiment in enumerate(layout.experiments):
        first_lane.setdefault(experiment, lane_index)
    spans: dict = {}
    lanes, gather, owner_of_lane, local = [], [], [], []
    for position, (index, child, lane) in enumerate(zip(selected, problems, lane_ids)):
        columns = layout.columns[index]
        if len(columns) != width or layout.widths[index] != point_width:
            raise ValueError(
                f"experiments {lane_ids[0]!r} and {lane!r} cannot share a compiled program: "
                f"they differ in derivative width ({width} and {len(columns)} "
                f"coordinates, over {point_width} and {layout.widths[index]} fitted values). "
                "A group takes only experiments of one derivative width."
            )
        if not np.array_equal(columns[:n_shared], np.arange(n_shared)):
            raise ValueError(
                f"experiment {lane!r} does not read the shared parameters as the leading "
                "coordinates of its narrow point."
            )
        experiment = layout.experiments[index]
        if experiment not in spans:
            spans[experiment] = _lane_spans(child, dtype)
        y0, times, streams = spans[experiment][index - first_lane[experiment]]
        if times.shape[0] < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(
                f"the sample times of experiment {lane!r} have to increase strictly from the "
                "time of its initial state."
            )
        lanes.append(
            (
                np.asarray(layout.targets[index], dtype=np.int32),
                child._seam.theta_frozen,
                y0,
                times[0],
                times[-1],
                _padded_times(times, slots.union),
                tuple(
                    _stream_arrays(series, times, capacity, dtype, held)
                    for (series, held), capacity in zip(streams, slots.streams)
                ),
                _configured_first_step(child.solver, lane, dtype),
                None if schedules is None else schedules[position],
            )
        )
        gather.append(columns)
        owner_of_lane.append(layout.nodes[index])
        local.append(np.asarray(layout.local[index], dtype=np.int32))

    program = LaneProgram(
        problem=first._seam.problem,
        solver=first.solver,
        unflatten=first._seam.unflatten,
        observes=tuple(series.observe for series in _lane_streams(first)),
        width=int(width),
        point_width=point_width,
        reverse=not bool(getattr(getattr(first.solver, "adjoint", None), "is_forward_only", True)),
        reverse_solver=_reverse_solver(first),
    )
    columns, frozen, y0, t0, t1, times, streams, dt0, schedule = jax.device_put(
        _stack(lanes), device
    )
    return LaneGroup(
        program=program,
        lanes=LaneInputs(
            columns=columns,
            frozen=frozen,
            y0=y0,
            t0=t0,
            t1=t1,
            times=times,
            streams=tuple(
                StreamInputs(
                    frame=frame, keep=keep, target=target, weight=weight, likelihood=likelihood
                )
                for frame, keep, target, weight, likelihood in streams
            ),
            dt0=dt0,
            schedule=schedule,
        ),
        gather=jax.device_put(np.stack(gather).astype(np.int32), device),
        lane_ids=lane_ids,
        children=selected,
        owners=np.asarray(owner_of_lane, dtype=np.int64),
        local=np.stack(local),
    )


def tree_of(joint: Any, groups: Any) -> tuple[TreeTables, tuple]:
    """The tables of the parameter tree for the lanes of several groups.

    The lanes are taken group after group, each group in its own order; lane
    results have to be concatenated in that order before they are reduced.

    Args:
        joint: The joint problem, or the fitted problem of one experiment, the
            groups were built from.
        groups: The [LaneGroup][diff_soup.inference.engine._inputs.LaneGroup]
            objects, in the order of the concatenation.

    Returns:
        The tables, with one block per owner of the joint in the joint's owner
        order (one block of no coordinates of its own that holds every lane
        when it has no owner tier), and the identifiers of the lanes in the
        order of the concatenation.

    Raises:
        ValueError: If the groups differ in narrow width.
    """
    groups = tuple(groups)
    widths = {group.program.width for group in groups}
    if len(widths) != 1:
        raise ValueError(
            f"the groups have the narrow widths {sorted(widths)}; their lane results can be "
            "concatenated only at one width."
        )
    layout = lane_layout(joint)
    tables = tree_tables(
        np.concatenate([group.owners for group in groups]),
        np.concatenate([group.local for group in groups], axis=0),
        layout.node_widths,
        layout.shared,
    )
    return tables, tuple(lane for group in groups for lane in group.lane_ids)


def concatenate_results(results: Any) -> Any:
    """Concatenate the lane results of several groups along the lane axis."""
    results = tuple(results)

    def join(*leaves):
        if all(leaf is None for leaf in leaves):
            return None
        return jnp.concatenate([jnp.asarray(leaf) for leaf in leaves], axis=0)

    return jax.tree_util.tree_map(join, *results, is_leaf=lambda leaf: leaf is None)


def lanes_at(lanes: LaneInputs, index: Any) -> LaneInputs:
    """The lanes at the given positions of a stacked batch."""
    index = jnp.asarray(index, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda leaf: leaf[index], lanes)


__all__ = ["LaneGroup", "concatenate_results", "lane_group", "lanes_at", "tree_of"]
