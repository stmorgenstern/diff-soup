"""Lanes driven by another lane's solved series: the links of a plan and their evaluation.

An experiment may declare outputs ([Output][diff_soup.optimization.inference.Output]),
series it produces at given times, and another experiment may be driven by one of
them ([Link][diff_soup.optimization.inference.Link]): the output's values become the
values of a fixed signal function of the driven model. The lane of a source
produces its linked outputs with its rows; the lane of a driven experiment takes
them as numbers ([LinkInputs][diff_soup.optimization.inference.engine._kernel.LinkInputs]).

[build_link_plan][diff_soup.optimization.inference.engine._links.build_link_plan] records,
from a study and its lane layout, which lanes produce which outputs, where each
output sits in its lane's flat outputs, and per link which positions of the driven
lane's narrow point hold the source's narrow coordinates. The plan carries it
([Plan.links][diff_soup.optimization.inference.engine._plan.Plan]).

[evaluate][diff_soup.optimization.inference.engine._links.evaluate] runs a request of a
study with links in two phases, each an ordinary evaluation of the batched lanes
in one process or in several:

1. the requested lanes that no link drives, the sources among them; a source a
   requested driven lane needs and the request leaves out is evaluated in an
   evaluation of its own, whose results only give its outputs;
2. the driven lanes, each unit with its sources' outputs at its member's point
   and, in the ``blocks`` and ``gradient`` modes, their derivative.

The results of the two phases are written to one
[Results][diff_soup.optimization.inference.engine.Results] in canonical order. A driven
unit whose source failed at its point is not solved; it takes the kind of the
source's failure. Lanes without links, and studies without links, run as they
would without this module.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np

from diff_soup.optimization.inference.engine._capacity import Frames
from diff_soup.optimization.inference.engine._order import canonical_lanes
from diff_soup.optimization.inference.engine._request import Request
from diff_soup.optimization.inference.engine._result import Batch, Results, Status, Work
from diff_soup.optimization.inference.engine._status import ABSENT

#: The modes whose driven lanes carry the derivative of their sources' outputs.
CARRIED: tuple[str, ...] = ("blocks", "gradient")
#: The modes whose driven lanes take their sources' outputs as fixed values.
HELD: tuple[str, ...] = ("value", "observe", "frozen")


class SignalHandle(NamedTuple):
    """The signal function a link fills, as ``theta_replace`` reads a handle.

    Attributes:
        name: The name of the function in the driven model.
        kind: Always ``"function"``.
        identity: Unused; present because a handle has one.
    """

    name: str
    kind: str = "function"
    identity: tuple = ()


@dataclasses.dataclass(frozen=True, eq=False)
class OutputSlot:
    """One output a source lane produces for the lanes it drives.

    Attributes:
        name: The output's name.
        times: Its times, on the source's clock.
        observe: ``observe(state, t=, theta=)``, reading one saved state.
        offset: Where it starts in the lane's flat outputs.
        shape: Its shape, one leading entry per time: the shape of the values
            of every signal it fills.
    """

    name: str
    times: np.ndarray
    observe: Callable
    offset: int
    shape: tuple

    @property
    def size(self) -> int:
        """The number of values of the output."""
        return int(np.prod(self.shape, dtype=np.int64))


@dataclasses.dataclass(frozen=True, eq=False)
class LinkSlot:
    """One link of a driven lane.

    Attributes:
        source: The id of the source lane.
        output: The index of the output among the source's linked outputs.
        into: The handle of the signal function the link fills.
        shape: The shape of the signal's values.
        columns: Per narrow coordinate of the source, its position in the
            driven lane's narrow point; ``0`` where the driven lane does not
            read it.
        reads: Per narrow coordinate of the source, whether the driven lane
            reads it.
    """

    source: Hashable
    output: int
    into: SignalHandle
    shape: tuple
    columns: np.ndarray
    reads: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class LinkPlan:
    """Which lanes of a study produce outputs and which are driven by them.

    Attributes:
        outputs: Per source lane, by its index in the lane layout, its linked
            outputs in the order of its experiment's declaration.
        links: Per driven lane, by its index in the lane layout, its links.
        lanes: The id of every lane, by its index in the lane layout.
    """

    outputs: Mapping
    links: Mapping
    lanes: tuple

    @functools.cached_property
    def index(self) -> dict:
        """Every lane id to its index in the lane layout."""
        return {lane: index for index, lane in enumerate(self.lanes)}

    def links_of(self, lane: Hashable) -> tuple:
        """The links of the lane ``lane`` (an id); empty for a lane no link drives."""
        return self.links.get(self.index.get(lane), ())

    def outputs_of(self, lane: Hashable) -> tuple:
        """The linked outputs of the lane ``lane`` (an id); empty for a lane that drives none."""
        return self.outputs.get(self.index.get(lane), ())

    def record(self, index: int) -> tuple | None:
        """The part of a lane's identity its outputs and links make; ``None`` without either.

        A source's record names each output, its shape and its readout; a driven
        lane's names each signal, the shape of its values and the source's
        narrow width.
        """
        parts = ()
        if index in self.outputs:
            parts += (
                ("outputs", tuple((s.name, s.shape, s.observe) for s in self.outputs[index])),
            )
        if index in self.links:
            parts += (
                (
                    "links",
                    tuple(
                        (s.into.name, s.shape, int(s.columns.shape[0])) for s in self.links[index]
                    ),
                ),
            )
        return parts or None

    def times(self, index: int, times: Any) -> np.ndarray:
        """A lane's sample times with the times of its outputs added, in its dtype."""
        slots = self.outputs.get(index)
        if not slots:
            return times
        extra = np.concatenate([np.asarray(slot.times) for slot in slots]).astype(times.dtype)
        return np.union1d(times, extra)

    def frames(self, index: int, frames: Frames, union: Any) -> Frames:
        """A lane's sample counts with the times of its outputs among its union of sample times.

        ``union`` is the lane's union of sample times without them.
        """
        if index not in self.outputs:
            return frames
        return frames._replace(union=int(self.times(index, np.asarray(union)).shape[0]))


def _signal_shape(experiment: Any, into: str) -> tuple:
    """The shape of the values of the signal function ``into`` of an experiment's model."""
    from diff_soup.optimization.inference.statistics.measurement import _signal_values

    return tuple(int(size) for size in _signal_values(experiment._declarations[into]).shape)


def build_link_plan(study: Any, layout: Any, names: Sequence) -> LinkPlan | None:
    """The links of a study's lanes, or ``None`` for a study without links.

    Args:
        study: The [Study][diff_soup.optimization.inference.Study].
        layout: Its [LaneLayout][diff_soup.optimization.inference.engine._layout.LaneLayout].
        names: The id of every lane, by its index in the lane layout, as the
            plan names them.

    Returns:
        The [LinkPlan][diff_soup.optimization.inference.engine._links.LinkPlan].

    Raises:
        ValueError: If two links read one output into signals of different
            shapes.
    """
    experiments = tuple(study.experiments)
    if not any(experiment.links for experiment in experiments):
        return None
    ids = tuple(study.experiment_ids)
    position = {experiment_id: index for index, experiment_id in enumerate(ids)}
    lane_of: dict = {}
    for lane, experiment in enumerate(layout.experiments):
        lane_of.setdefault(int(experiment), lane)
    shapes: dict = {}
    for index, experiment in enumerate(experiments):
        for link in experiment.links:
            key = (position[link.source], link.output)
            shape = _signal_shape(experiment, link.into)
            if shapes.setdefault(key, shape) != shape:
                raise ValueError(
                    f"the output {link.output!r} of experiment {link.source!r} fills signals of "
                    f"the shapes {shapes[key]} and {shape} (experiment {ids[index]!r}, signal "
                    f"{link.into!r}); every signal one output fills has the output's shape."
                )
    outputs = {}
    for index, experiment in enumerate(experiments):
        slots, offset = [], 0
        for output in experiment.outputs:
            shape = shapes.get((index, output.name))
            if shape is None:
                continue
            slots.append(
                OutputSlot(output.name, np.asarray(output.times), output.observe, offset, shape)
            )
            offset += slots[-1].size
        if slots:
            outputs[lane_of[index]] = tuple(slots)
    links = {}
    for index, experiment in enumerate(experiments):
        if not experiment.links:
            continue
        driven = lane_of[index]
        reads = {int(column): place for place, column in enumerate(layout.columns[driven])}
        slots = []
        for link in experiment.links:
            source = lane_of[position[link.source]]
            produced = [slot.name for slot in outputs[source]]
            columns = np.asarray(layout.columns[source], dtype=np.int64)
            slots.append(
                LinkSlot(
                    source=names[source],
                    output=produced.index(link.output),
                    into=SignalHandle(link.into),
                    shape=_signal_shape(experiment, link.into),
                    columns=np.asarray([reads.get(int(c), 0) for c in columns], dtype=np.int32),
                    reads=np.asarray([int(c) in reads for c in columns], dtype=bool),
                )
            )
        links[driven] = tuple(slots)
    return LinkPlan(
        outputs=types.MappingProxyType(outputs),
        links=types.MappingProxyType(links),
        lanes=tuple(names),
    )


def link_inputs(links: LinkPlan, index: int, dtype: Any) -> tuple:
    """The host numbers of a driven lane's links before an evaluation gives them their values.

    Args:
        links: The plan's links.
        index: The lane's index in the lane layout.
        dtype: The lane's floating-point dtype.

    Returns:
        Per link ``(values, tangent, columns)``: zero values and tangent, and
        the positions of the source's coordinates in the lane's narrow point.
    """
    return tuple(
        (
            np.zeros(slot.shape, dtype=dtype),
            np.zeros(slot.shape + (int(slot.columns.shape[0]),), dtype=dtype),
            np.asarray(slot.columns, dtype=np.int32),
        )
        for slot in links.links.get(index, ())
    )


def output_frames(links: LinkPlan, index: int, times: Any) -> tuple:
    """Per linked output of a source lane, the index of each of its times in ``times``."""
    return tuple(
        np.searchsorted(times, np.asarray(slot.times).astype(times.dtype)).astype(np.int32)
        for slot in links.outputs.get(index, ())
    )


# The evaluation of a request ---------------------------------------------------------------


def _part(request: Request, chosen: Sequence, columns: Sequence, units: Any) -> Request:
    """The request of the lanes ``columns`` of ``chosen`` (canonical positions), units ``units``."""
    lanes = tuple(chosen[column] for column in columns)
    held = set(lanes)
    order = tuple(lane for lane in (request.order or ()) if lane in held) or None

    def part(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else np.asarray(array)[:, columns]

    return Request(
        mode=request.mode,
        points=request.points,
        active=request.active,
        lanes=lanes,
        order=order,
        rungs=part(request.rungs),
        first_steps=part(request.first_steps),
        units=None if units is None or np.all(units) else units,
    )


def _refuse_mode(lanes: Any, mode: str, sources: Sequence) -> None:
    """Refuse a mode in which the driven lanes cannot be evaluated."""
    if mode not in CARRIED + HELD:
        raise ValueError(
            f"the {mode} mode differentiates every lane twice, and a link carries the first "
            "derivative of its source's output only; evaluate a study with links in the modes "
            f"{CARRIED + HELD}."
        )
    if mode == "gradient":
        for source in sources:
            bucket = lanes.buckets[lanes.where[source][0]].bucket
            if "blocks" not in bucket.chunk:
                raise ValueError(
                    f"the gradient mode carries the derivative of the output of experiment "
                    f"{source!r} forward, and its solver carries a reverse adjoint; give the "
                    "sources of a study with links solvers without a reverse adjoint."
                )


def _source_numbers(mode: str, source: Hashable, covered: np.ndarray, parts: Sequence) -> tuple:
    """A source's outputs, their derivative and its status per member, from the phase-one results.

    ``parts`` is ``(first, extra)``: the results of the request's own lanes
    and those of the evaluation of the sources it leaves out (either
    ``None``); a member whose unit of the source the request evaluates
    (``covered``) reads ``first``, any other ``extra``.
    """
    values = tangents = kinds = None
    for results, rows in zip(parts, (covered, ~covered)):
        if results is None or not np.any(rows):
            continue
        column = results.lane_ids.index(source)
        held = (results.outputs or {}).get(source)
        if held is None:
            raise RuntimeError(f"the evaluation of experiment {source!r} returned no outputs.")
        given, slopes = held
        if values is None:
            values = np.zeros_like(given)
            kinds = np.full(rows.shape, ABSENT, dtype=np.asarray(results.status.kind).dtype)
            if slopes is not None:
                tangents = np.zeros_like(slopes)
        values[rows] = given[rows]
        kinds[rows] = np.asarray(results.status.kind)[rows, column]
        if tangents is not None and slopes is not None:
            tangents[rows] = slopes[rows]
    if mode in CARRIED and tangents is None:
        raise RuntimeError(f"the evaluation of experiment {source!r} returned no derivative.")
    return values, tangents, kinds


def evaluate(
    links: LinkPlan,
    lanes: Any,
    request: Request,
    run: Callable[[Request, Mapping | None], Results],
    ready: Callable[[str], None],
) -> Results:
    """Evaluate a request of a study with links: the sources first, then the lanes they drive.

    Args:
        links: The plan's links.
        lanes: The study's lanes (on process 0 of a run of several processes,
            its lanes without inputs).
        request: The [Request][diff_soup.optimization.inference.engine.Request].
        run: ``run(request, link_values)`` evaluates a request through the
            batched lanes, in one process or in several, with each driven
            lane's link values: per link ``(values, tangents)`` with one row
            per member that takes part, ``tangents`` ``None`` in a mode that
            forms no derivative. It returns their results, with the requested
            sources' outputs in ``Results.outputs``.
        ready: ``ready(mode)`` compiles ``mode`` for every bucket that lacks
            it, for the evaluation of the sources a request leaves out.

    Returns:
        The [Results][diff_soup.optimization.inference.engine.Results] of the request, in
        canonical order, as one evaluation would give them.

    Raises:
        ValueError: As an evaluation of the batched lanes; for the ``hessian``
            mode, and the ``gradient`` mode with a source whose solver carries
            a reverse adjoint, when a driven lane is requested; and when a
            source's output moves with a coordinate the lane it drives does not
            read (a parameter of the source that is not shared).
    """
    chosen = canonical_lanes(
        request.lanes,
        request.order,
        study_ids=lanes.lane_ids,
        planned=lanes.where,
        refused=lanes.refused,
    )
    members = request.members
    if not chosen or not members.shape[0]:
        return run(request, None)
    present = request.unit_mask(len(chosen))
    shape = (int(members.shape[0]), len(chosen))
    mask = np.ones(shape, dtype=bool) if present is None else np.asarray(present, dtype=bool)
    driven = [column for column, lane in enumerate(chosen) if links.links_of(lane)]
    if not driven or not np.any(mask[:, driven]):
        return run(request, None)
    mode = request.mode
    position = {lane: column for column, lane in enumerate(chosen)}
    needed: dict = {}
    for column in driven:
        for slot in links.links_of(chosen[column]):
            wanted = needed.setdefault(slot.source, np.zeros(shape[0], dtype=bool))
            wanted |= mask[:, column]
    order = {lane: index for index, lane in enumerate(lanes.lane_ids)}
    sources = sorted(needed, key=order.__getitem__)
    _refuse_mode(lanes, mode, sources)

    # Phase one: the requested lanes no link drives, and the sources the request leaves out.
    others = [column for column in range(shape[1]) if column not in set(driven)]
    first = None
    if others and np.any(mask[:, others]):
        first = run(
            _part(request, chosen, others, None if present is None else mask[:, others]), None
        )
    covered = {
        source: (
            mask[:, position[source]]
            if source in position and first is not None
            else np.zeros(shape[0], dtype=bool)
        )
        for source in sources
    }
    missing = [source for source in sources if np.any(needed[source] & ~covered[source])]
    extra = None
    if missing:
        extra_mode = mode if mode in CARRIED else "value"
        if extra_mode != mode:
            ready(extra_mode)
        extra = run(
            Request(
                mode=extra_mode,
                points=request.points,
                active=request.active,
                lanes=tuple(missing),
                units=np.stack([needed[s] & ~covered[s] for s in missing], axis=1),
            ),
            None,
        )

    # Phase two: the driven lanes, each with its sources' outputs at its member's point.
    numbers = {
        source: _source_numbers(mode, source, covered[source], (first, extra)) for source in sources
    }
    failed = np.zeros(shape, dtype=np.int64)
    values: dict = {}
    for column in driven:
        lane = chosen[column]
        given = []
        for slot in links.links_of(lane):
            output = links.outputs_of(slot.source)[slot.output]
            outputs, slopes, kinds = numbers[slot.source]
            span = slice(output.offset, output.offset + output.size)
            value = outputs[:, span].reshape((shape[0],) + slot.shape)
            tangent = None
            if mode in CARRIED:
                tangent = np.array(slopes[:, span, :], copy=True)
                solved = mask[:, column] & (kinds == 0)
                unread = tangent[solved][..., ~slot.reads]
                if np.any(unread != 0.0):
                    raise ValueError(
                        f"the output {output.name!r} of experiment {slot.source!r} moves with "
                        f"coordinates that experiment {lane!r} does not read, so its derivative "
                        "cannot reach them; a link carries the source's shared parameters: give "
                        "the parameters that move a source's output to the study as shared "
                        "parameters, declared by every experiment."
                    )
                tangent[..., ~slot.reads] = 0.0
                tangent = tangent.reshape((shape[0],) + slot.shape + (tangent.shape[-1],))
            given.append((value, tangent))
            broken = mask[:, column] & (kinds != 0) & (failed[:, column] == 0)
            failed[broken, column] = kinds[broken]
        values[lane] = tuple(given)
    solve = mask[:, driven] & (failed[:, driven] == 0)
    second = None
    if np.any(solve):
        second = run(_part(request, chosen, driven, solve), values)
    parts = tuple(results for results in (first, second) if results is not None)
    return _merged(mode, chosen, members, mask, lanes, parts, failed, extra)


def _merged(
    mode: str,
    chosen: Sequence,
    members: np.ndarray,
    mask: np.ndarray,
    lanes: Any,
    parts: Sequence,
    failed: np.ndarray,
    extra: Results | None,
) -> Results:
    """The results of the phases written to the request's units, in canonical order.

    A unit ``failed`` names (a driven unit whose source failed) takes that kind
    and zeros; a unit the request leaves out is ``absent``.
    """
    shape = (int(members.shape[0]), len(chosen))
    position = {lane: column for column, lane in enumerate(chosen)}
    template = (parts + ((extra,) if extra is not None else ()))[0]
    columns = [
        np.asarray([position[lane] for lane in part.lane_ids], dtype=np.int64) for part in parts
    ]

    def field(read: Callable, fill: Any = 0) -> np.ndarray | None:
        given = [read(part) for part in parts]
        reference = next((value for value in given if value is not None), None)
        if reference is None:
            reference = read(template)
            if reference is None:
                return None
        reference = np.asarray(reference)
        out = np.full(shape + reference.shape[2:], fill, dtype=reference.dtype)
        for value, where in zip(given, columns):
            if value is not None:
                out[:, where] = np.asarray(value)
        return out

    kind = field(lambda part: part.status.kind, ABSENT)
    t_reached = field(lambda part: part.status.t_reached, np.nan)
    rung = field(lambda part: part.status.rung)
    cost = field(lambda part: part.cost)
    rows = field(lambda part: part.rows)
    normal = field(lambda part: part.normal)
    gradient = field(lambda part: part.gradient)
    names = tuple(entry.name for entry in dataclasses.fields(template.work))
    work = {name: field(lambda part, name=name: getattr(part.work, name)) for name in names}
    lane_seconds = field(lambda part: part.lane_seconds)
    diagonal = field(lambda part: part.normal_diagonal)
    broken = (failed != 0) & mask
    kind[broken] = failed[broken]
    t_reached[broken] = np.nan
    kind[~mask] = ABSENT

    observables = None
    if mode == "observe":
        grid = [[None] * shape[1] for _ in range(shape[0])]
        for part, where in zip(parts, columns):
            for row in range(shape[0]):
                for index, column in enumerate(where):
                    grid[row][int(column)] = part.observables[row][index]
        observables = tuple(tuple(row) for row in grid)

    held = [part.levels for part in parts if part.levels is not None]
    levels = None
    if held:
        levels = held[0]
        for other in held[1:]:
            levels = jax.tree_util.tree_map(np.add, levels, other)
    elif extra is not None and extra.levels is not None and extra.mode == mode:
        levels = jax.tree_util.tree_map(np.zeros_like, extra.levels)

    seconds: dict = {}
    batches: dict = {}
    retries = []
    outputs: dict = {}
    for part in parts:
        for bucket, wall in part.seconds.items():
            seconds[int(bucket)] = seconds.get(int(bucket), 0.0) + float(wall)
        for bucket, batch in part.batches.items():
            have = batches.get(int(bucket))
            batches[int(bucket)] = (
                batch
                if have is None
                else Batch(have.units + batch.units, have.width, have.calls + batch.calls)
            )
        retries.extend(part.retries)
        outputs.update(part.outputs or {})
    retries.sort(key=lambda entry: (int(entry["member"]), position[entry["lane"]]))
    return Results(
        mode=mode,
        lane_ids=tuple(chosen),
        members=np.asarray(members, dtype=np.int64),
        bucket=np.asarray([lanes.where[lane][0] for lane in chosen], dtype=np.int64),
        status=Status(kind=kind, t_reached=t_reached, rung=rung),
        cost=cost,
        rows=rows,
        normal=normal,
        gradient=gradient,
        observables=observables,
        work=Work(**work),
        seconds=types.MappingProxyType(dict(sorted(seconds.items()))),
        retries=tuple(retries),
        batches=types.MappingProxyType(dict(sorted(batches.items()))),
        levels=levels,
        lane_seconds=lane_seconds,
        normal_diagonal=diagonal,
        outputs=types.MappingProxyType(outputs) if outputs else None,
    )


__all__ = [
    "CARRIED",
    "HELD",
    "LinkPlan",
    "LinkSlot",
    "OutputSlot",
    "SignalHandle",
    "build_link_plan",
    "evaluate",
    "link_inputs",
    "output_frames",
]
