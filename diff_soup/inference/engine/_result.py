"""The results of one evaluation of the batched lanes, in canonical order, on the host.

The units of a request are evaluated in calls of compiled programs, in an order
chosen for speed, and the last call of a bucket is filled up with copies of a
unit. [assemble][diff_soup.inference.engine._result.assemble] keeps the
results of each call's own units, drops the copies, and writes every unit to
its place in canonical order: members ascending, then lanes experiment by
experiment, the windows of one experiment in time order. The predictions of
the ``observe`` mode are cut to each lane's own frames. A failed unit keeps
what its lane returned: zeros and a typed status.

A unit that failed with a kind of the study's ``retry_kinds`` is solved again,
from a smaller first step, in later calls of the same programs.
[merge_calls][diff_soup.inference.engine._result.merge_calls] replaces such a
unit's result by the one of its later call, and ``assemble`` then records the
rung that produced each unit's result and the record of every unit that was
solved again. Per bucket the evaluation touched, a
[Batch][diff_soup.inference.engine._result.Batch] records how its units were
cut into calls.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import numpy as np

from diff_soup.inference.engine._kernel import LaneResult, LaneWork
from diff_soup.inference.engine._status import ABSENT, KINDS, LaneStatus


class Status(eqx.Module):
    """How the solve of every unit ended: numpy arrays with leading axes (members, lanes).

    Attributes:
        kind: Index into [KINDS][diff_soup.inference.engine._status.KINDS];
            zero means the unit solved.
        t_reached: The time the solve reached.
        rung: The rung that produced the result: 0 for the solve from the
            lane's own first step, ``r`` for the solve from
            ``retry_first_step_factors[r - 1]`` times that first step.
    """

    kind: np.ndarray
    t_reached: np.ndarray
    rung: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        """Whether each unit solved."""
        return self.kind == 0

    @property
    def failed(self) -> np.ndarray:
        """Whether each unit failed; a unit that was not evaluated (``absent``) did not."""
        return (self.kind != 0) & (self.kind != ABSENT)

    @property
    def absent(self) -> np.ndarray:
        """Whether each unit was left out of its request (``Request(units=...)``)."""
        return self.kind == ABSENT


class Work(eqx.Module):
    """The work counters of every unit's solve: numpy arrays with leading axes (members, lanes).

    The fields are those of [LaneWork][diff_soup.inference.engine._kernel.LaneWork],
    from the solve that produced each unit's result.

    Attributes:
        first_step: The size of the first trial step of each unit's solve,
            ``0`` where the solver does not report it. At rung 0 it is the
            unit's own first step, the base its rungs scale.
    """

    accepts: np.ndarray
    rejects: np.ndarray
    trial_steps: np.ndarray
    newton_iters: np.ndarray
    jac_refreshes: np.ndarray
    factorizations: np.ndarray
    first_step: np.ndarray


class Batch(NamedTuple):
    """How one evaluation ran the units of one bucket: their count, the width, the calls made.

    It records the evaluation's first pass over the bucket, the one that runs
    every requested unit; a pass that solves failed units again is recorded in
    ``Results.retries``.

    Attributes:
        units: The units the pass ran: the members that take part times the
            requested lanes of the bucket.
        width: ``W``, the units one call of the bucket's program evaluates.
        calls: The calls of the program the pass made,
            ``ceil(units / width)``.
    """

    units: int
    width: int
    calls: int


@dataclasses.dataclass(frozen=True, eq=False)
class Results:
    """What one evaluation of the batched lanes returns, one entry per requested unit.

    Every array is a numpy array whose leading axes are (members, lanes), in
    canonical order. Padded copies of a unit appear nowhere.

    Attributes:
        mode: The mode the lanes ran in.
        lane_ids: The ids of the lanes, in canonical order: an experiment's id
            for its one lane, a window's own id for each lane of an experiment
            fitted in observed-start windows.
        members: The population members, ascending, shape ``(P,)``.
        bucket: Per lane, the index of its bucket in the plan, shape ``(L,)``.
        status: The [Status][diff_soup.inference.engine._result.Status] of
            every unit.
        cost: Half the squared whitened residual; zero for a failed unit.
        rows: The number of residual rows of every unit.
        normal: ``J^T J`` in narrow coordinates, shape ``(P, L, w, w)``, in
            the ``blocks`` mode, and the exact Hessian of the cost in the
            ``hessian`` mode; ``None`` otherwise.
        gradient: ``J^T r`` in narrow coordinates, shape ``(P, L, w)``, in
            the ``blocks``, ``gradient`` and ``hessian`` modes; ``None``
            otherwise.
        observables: In the ``observe`` mode, ``observables[p][l][s]`` holds
            the predictions of stream ``s`` of lane ``l`` for member ``p`` at
            the lane's own frames; ``None`` otherwise.
        work: The [Work][diff_soup.inference.engine._result.Work] counters of
            every unit, from the solve that produced its result.
        seconds: Per bucket index, the wall seconds its calls took, from the
            first call to the results on the host, summed over the passes
            that solved its units again. Reported, not a measurement to
            compare against.
        retries: One record per unit that was solved again, in canonical
            order. A unit that fails with a kind of the lowered study's
            ``retry_kinds`` is solved again from a smaller first step, each
            factor of ``retry_first_step_factors`` one rung, until it solves
            or the factors run out; ``status.rung`` says which rung produced
            its result. A record is a dict with ``member``, ``lane``,
            ``rungs`` (the rungs tried, in order), ``kinds`` (the kind each of
            them ended with, ``"ok"`` last when the unit solved),
            ``first_step`` (the unit's rung-0 first step, which the factors
            scale) and ``seconds`` (per rung tried after the first, the wall
            seconds of the calls of the unit's bucket that tried it).
        batches: Per index of a bucket the evaluation touched, and no other,
            the [Batch][diff_soup.inference.engine._result.Batch] of its
            first pass: the units it ran, the width of the bucket's program
            and the calls it made. A request of ``P`` points runs ``P`` times
            a bucket's lanes in calls of that width, so a bucket's ``calls``
            show whether the points were evaluated together. Empty when the
            results were not made by an evaluation.
        levels: In a lowered study whose lanes run in several processes, per
            member of ``members`` (the leading axis), the sums of the lanes'
            blocks (``blocks`` and ``hessian`` modes, a
            [LevelBlocks][diff_soup.inference.engine.LevelBlocks]) or of
            their gradients (``gradient`` mode, a
            [LevelGradients][diff_soup.inference.engine.LevelGradients]) over
            the level tables of the study's fits, every lane weighing one; the
            lanes' own matrices stay in the processes that hold them, so
            ``normal`` and ``gradient`` are then ``None``. ``None`` in one
            process, and in the ``value`` mode.
        lane_seconds: Per unit, shape ``(P, L)``, the wall seconds of its
            bucket's calls shared among the bucket's units in proportion to
            their trial steps: a measure of what each lane costs, reported, not
            a measurement to compare against. ``None`` when not recorded.
        normal_diagonal: In a lowered study whose lanes run in several
            processes, in the ``blocks`` and ``hessian`` modes, per unit the
            diagonal of its lane's matrix in narrow coordinates, shape
            ``(P, L, w)``, gathered from the processes that hold the lanes; a
            unit left out has zeros. ``None`` in one process (the diagonal is
            then read from ``normal``), and in the other modes.

    A unit the request left out (``Request(units=...)``) has the status kind
    ``absent``, zero cost, rows, derivatives and work, and lies outside every
    sum; it is not a failure.
    """

    mode: str
    lane_ids: tuple
    members: np.ndarray
    bucket: np.ndarray
    status: Status
    cost: np.ndarray
    rows: np.ndarray
    normal: Optional[np.ndarray]
    gradient: Optional[np.ndarray]
    observables: Optional[tuple]
    work: Work
    seconds: Mapping
    retries: tuple = ()
    batches: Mapping = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))
    levels: Any = None
    lane_seconds: Optional[np.ndarray] = None
    normal_diagonal: Optional[np.ndarray] = None

    def failures(self) -> tuple[dict, ...]:
        """The failed units, in canonical order.

        Returns:
            One record per failed unit: ``member``, ``lane``, ``kind`` (a name
            of ``KINDS``), ``t_reached``, ``rung`` (the last rung tried). A
            unit that was not evaluated (``absent``) has no record.
        """
        kinds = np.asarray(self.status.kind)
        return tuple(
            {
                "member": int(self.members[row]),
                "lane": self.lane_ids[column],
                "kind": KINDS[int(kinds[row, column])],
                "t_reached": float(self.status.t_reached[row, column]),
                "rung": int(self.status.rung[row, column]),
            }
            for row, column in np.argwhere((kinds != 0) & (kinds != ABSENT))
        )

    def lane_result(self, member: Any, device: Any = None) -> LaneResult:
        """The results of one population member as a lane result for the reducers.

        Args:
            member: A population member of ``members``.
            device: The JAX device of the lane result; ``None`` for the
                default device.

        Returns:
            A [LaneResult][diff_soup.inference.engine._kernel.LaneResult]
            whose leaves lead with the lane axis, in the order of
            ``lane_ids``, on ``device``. It carries no predictions.

        Raises:
            ValueError: If ``member`` is not one of ``members``.
        """
        member = operator.index(member)
        rows = np.flatnonzero(self.members == member)
        if rows.shape[0] != 1:
            raise ValueError(
                f"member {member} is not among the members of these results "
                f"{self.members.tolist()}; ask for one of them."
            )
        row = int(rows[0])

        def take(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if value is None else value[row]

        return LaneResult(
            status=LaneStatus(
                kind=jax.device_put(self.status.kind[row], device),
                t_reached=jax.device_put(self.status.t_reached[row], device),
                rung=jax.device_put(self.status.rung[row], device),
            ),
            cost=jax.device_put(self.cost[row], device),
            rows=jax.device_put(self.rows[row], device),
            normal=jax.device_put(take(self.normal), device),
            gradient=jax.device_put(take(self.gradient), device),
            observables=None,
            work=LaneWork(
                **{
                    entry.name: jax.device_put(getattr(self.work, entry.name)[row], device)
                    for entry in dataclasses.fields(self.work)
                }
            ),
        )


class CallOutput(NamedTuple):
    """What one call of a compiled program returned, on the host.

    Attributes:
        rows: Per own unit of the call, its position in the results' members.
        columns: Per own unit, its position in the results' lanes.
        output: The call's [LaneResult][diff_soup.inference.engine._kernel.LaneResult]
            as numpy arrays, leading with the call's width; its first
            ``len(rows)`` entries are the call's own units, the rest padded
            copies.
        frames: Per own unit, per stream, the number of frames the lane
            records.
    """

    rows: np.ndarray
    columns: np.ndarray
    output: Any
    frames: tuple


def unit_values(calls: Sequence, shape: tuple, read: Any) -> Optional[np.ndarray]:
    """One field of every call's own units, written to canonical positions.

    ``read(output)`` gives the field of a call's output, leading with the
    call's width, or ``None`` when the mode does not compute it. A position
    no call fills holds zero.
    """
    values = [read(call.output) for call in calls]
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("the calls of one evaluation disagree on which fields they return.")
    trailing = {tuple(np.shape(value)[1:]) for value in values}
    if len(trailing) != 1:
        raise RuntimeError(
            f"the calls of one evaluation return one field in the shapes {sorted(trailing)}."
        )
    dtype = np.result_type(*(np.asarray(value).dtype for value in values))
    out = np.zeros(shape + trailing.pop(), dtype=dtype)
    for call, value in zip(calls, values):
        out[call.rows, call.columns] = np.asarray(value)[: call.rows.shape[0]]
    return out


def _own_units(call: CallOutput, keep: np.ndarray) -> Optional[CallOutput]:
    """``call`` with only the own units where ``keep`` holds, or ``None`` when it keeps none."""
    if np.all(keep):
        return call
    if not np.any(keep):
        return None
    index = np.flatnonzero(keep)
    return CallOutput(
        rows=call.rows[index],
        columns=call.columns[index],
        output=jax.tree_util.tree_map(lambda leaf: np.asarray(leaf)[index], call.output),
        frames=tuple(call.frames[unit] for unit in index),
    )


def merge_calls(calls: Sequence, again: Sequence, replaced: np.ndarray) -> tuple:
    """The call outputs of an evaluation after some of its units were solved again.

    Args:
        calls: The call outputs so far, one result for every requested unit.
        again: The call outputs of the units solved again. Their ``rows`` and
            ``columns`` are positions in the evaluation's members and lanes;
            they may hold units that were run only to fill their calls.
        replaced: Booleans of shape (members, lanes): the units whose result
            ``again`` replaces.

    Returns:
        The call outputs with the results of the ``replaced`` units taken from
        ``again`` and every other unit's from ``calls``, so that each unit has
        exactly one result.
    """
    replaced = np.asarray(replaced, dtype=bool)
    merged = []
    for call in calls:
        part = _own_units(call, ~replaced[call.rows, call.columns])
        if part is not None:
            merged.append(part)
    for call in again:
        part = _own_units(call, replaced[call.rows, call.columns])
        if part is not None:
            merged.append(part)
    return tuple(merged)


def assemble(
    mode: str,
    lane_ids: tuple,
    members: np.ndarray,
    bucket: np.ndarray,
    calls: Sequence,
    seconds: Mapping,
    *,
    rungs: Optional[np.ndarray] = None,
    retries: tuple = (),
    batches: Optional[Mapping] = None,
    present: Optional[np.ndarray] = None,
) -> Results:
    """The results of one evaluation from the outputs of its calls.

    Args:
        mode: The mode the calls ran in.
        lane_ids: The requested lane ids, in canonical order.
        members: The requested population members, ascending.
        bucket: Per lane of ``lane_ids``, its bucket index.
        calls: One [CallOutput][diff_soup.inference.engine._result.CallOutput]
            per call of a program; at least one.
        seconds: Per bucket index, the wall seconds of its calls.
        rungs: Per unit, shape (members, lanes), the rung whose solve
            produced its result; ``None`` keeps the rung the calls report.
        retries: The records of the units solved again, for
            ``Results.retries``.
        batches: Per bucket index, the
            [Batch][diff_soup.inference.engine._result.Batch] of its first
            pass, for ``Results.batches``; ``None`` records none.
        present: Per unit, shape (members, lanes), whether it was evaluated;
            ``None`` when every unit was. A unit not evaluated gets the kind
            ``absent`` and zeros everywhere else.

    Returns:
        The [Results][diff_soup.inference.engine._result.Results].

    Raises:
        RuntimeError: If the calls do not give exactly one result for every
            evaluated unit, or give one for a unit that was not evaluated.
    """
    calls = tuple(calls)
    shape = (int(members.shape[0]), len(lane_ids))
    count = np.zeros(shape, dtype=np.int64)
    for call in calls:
        np.add.at(count, (call.rows, call.columns), 1)
    expected = np.ones(shape, dtype=np.int64) if present is None else present.astype(np.int64)
    if not calls or np.any(count != expected):
        missing = int(np.count_nonzero(count < expected))
        repeated = int(np.count_nonzero(count > expected))
        raise RuntimeError(
            f"an evaluation assembled {missing} unit(s) without a result and {repeated} with "
            "more than one, or a result for a unit not evaluated; every evaluated unit needs "
            "exactly one result."
        )

    field = functools.partial(unit_values, calls, shape)
    rung = field(lambda output: output.status.rung)
    if rungs is not None:
        rung = np.asarray(rungs).astype(rung.dtype)
    kind = field(lambda output: output.status.kind)
    t_reached = field(lambda output: output.status.t_reached)
    if present is not None:
        left_out = ~np.asarray(present, dtype=bool)
        kind[left_out] = ABSENT
        t_reached[left_out] = np.nan
        rung[left_out] = 0
    status = Status(kind=kind, t_reached=t_reached, rung=rung)
    counters = tuple(entry.name for entry in dataclasses.fields(calls[0].output.work))
    work = Work(
        **{name: field(lambda output, name=name: getattr(output.work, name)) for name in counters}
    )
    observables = None
    if mode == "observe":
        grid = [[None] * shape[1] for _ in range(shape[0])]
        for call in calls:
            streams = call.output.observables
            for unit, (row, column) in enumerate(zip(call.rows, call.columns)):
                grid[int(row)][int(column)] = tuple(
                    np.array(stream[unit, :own]) for stream, own in zip(streams, call.frames[unit])
                )
        observables = tuple(tuple(row) for row in grid)
    return Results(
        mode=mode,
        lane_ids=tuple(lane_ids),
        members=np.asarray(members, dtype=np.int64),
        bucket=np.asarray(bucket, dtype=np.int64),
        status=status,
        cost=field(lambda output: output.cost),
        rows=field(lambda output: output.rows),
        normal=field(lambda output: output.normal),
        gradient=field(lambda output: output.gradient),
        observables=observables,
        work=work,
        seconds=seconds,
        retries=tuple(retries),
        batches=types.MappingProxyType(dict(sorted((batches or {}).items()))),
    )


__all__ = [
    "Batch",
    "CallOutput",
    "Results",
    "Status",
    "Work",
    "assemble",
    "merge_calls",
    "unit_values",
]
