"""The batched backend: compiled programs per bucket and mode, and the evaluation of a request.

A planned study ([build_plan][diff_soup.optimization.inference.engine._plan.build_plan])
sorts its lanes into buckets. [batched_lanes][diff_soup.optimization.inference.engine._batched.batched_lanes]
builds the numbers of every bucket's lanes on the host.
[prepare][diff_soup.optimization.inference.engine._batched.prepare] holds one program per
bucket and mode, compiling each that no study compiled before with the
function it is given ([compile_program][diff_soup.optimization.inference.engine._prepare.compile_program]
traces the bucket's kernel with the arrays of the bucket's first call, lowers
it and compiles it). [evaluate][diff_soup.optimization.inference.engine._batched.evaluate]
then runs a [Request][diff_soup.optimization.inference.engine._request.Request] through those
compiled programs only.

A program is known by its key: the bucket's identity, the mode and the chunk
width ``W``. The process-wide registry ``_PROGRAMS`` maps a key to its compiled
program, so equal keys compile once per process, across buckets, studies and
repeated lowering; lanes of a study that reuses a program run on the problem of
the study that compiled it, which the identity makes equivalent. The registry
holds its programs weakly: the lanes of a study hold the programs they use, and
a program no study holds is freed.

An evaluation compiles and traces nothing. Host work is numpy; parameter points
and lane inputs reach the device by ``jax.device_put``, and the device works
only inside the compiled programs. The lane inputs of a call stay on the device
while the call's lanes are unchanged from the last evaluation. On a plan with a
mesh, every array of a call is split along its lane axis over the devices of
the mesh's ``data`` axis, the compiled program runs each device's share, and
the units of a call are dealt to the devices by their last step counts
([device_order][diff_soup.optimization.inference.engine._balance.device_order]).

A unit that fails with a kind the study retries is solved again within the same
evaluation, through the same programs, from a smaller first step: rung ``r``
starts it from ``retry_first_step_factors[r - 1]`` times the first step its
rung-0 solve took. Each pass runs the members and lanes of the failed units, and
keeps the new result of the failed units only; the climb ends when no unit
qualifies. The evaluation keeps nothing of it for the next request: a request
names the rung of each unit, and ``Results`` reports the rung that produced each
result and the record of every unit solved again.

A study that prepared the ``frozen`` mode runs its ``blocks`` requests through
the program that also returns each lane's Jacobian, and keeps those Jacobians
on the device, call by call ([KeptJacobians][diff_soup.optimization.inference.engine._kept.KeptJacobians]):
each unit's from its last ``blocks`` result. A ``frozen`` request runs the
kept calls that hold its units, in their own layout, each with its kept
Jacobians; a unit without one is refused. A study that did not prepare the
``frozen`` mode keeps nothing.

``Results.batches`` records, per bucket, the units of the first pass, the width
of the program and the calls made. A call of a program that raises (a device
out of memory, the sparse direct solver failing inside the program) raises a
``BackendError`` that names the bucket, the mode and the width, chained to the
original error; it is never turned into the status of a lane.
"""

from __future__ import annotations

import dataclasses
import time
import types
import weakref
from collections.abc import Callable, Hashable, Iterable, Mapping
from numbers import Integral, Real
from typing import Any, NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.stages import Compiled
from jax.tree_util import PyTreeDef

from diff_soup._errors import BackendError
from diff_soup.optimization.inference.engine._balance import device_order
from diff_soup.optimization.inference.engine._capacity import lane_frames
from diff_soup.optimization.inference.engine._identity import LaneIdentity
from diff_soup.optimization.inference.engine._inputs import _lane_group
from diff_soup.optimization.inference.engine._kept import KeptCall, KeptJacobians
from diff_soup.optimization.inference.engine._kernel import (
    KEEPING_BLOCKS,
    MODES,
    LaneProgram,
    LaneResult,
)
from diff_soup.optimization.inference.engine._layout import lane_layout, lane_problems
from diff_soup.optimization.inference.engine._levels import LevelTables
from diff_soup.optimization.inference.engine._order import (
    StepMemory,
    canonical_lanes,
    cut,
    unit_order,
)
from diff_soup.optimization.inference.engine._plan import Bucket, Plan
from diff_soup.optimization.inference.engine._reducers import TreeTables, tree_tables
from diff_soup.optimization.inference.engine._request import Request
from diff_soup.optimization.inference.engine._result import (
    Batch,
    CallOutput,
    Results,
    assemble,
    merge_calls,
    unit_values,
)
from diff_soup.optimization.inference.engine._status import KINDS
from diff_soup.optimization.inference.statistics.hierarchy import Hierarchy


class ProgramKey(NamedTuple):
    """What one compiled program is: equal keys are one program.

    Attributes:
        identity: The bucket's [identity][diff_soup.optimization.inference.engine._identity.LaneIdentity];
            ``str()`` gives its first twelve hexadecimal characters.
        mode: The mode of the lane kernel.
        chunk: ``W``, the units one call evaluates.
        mesh: The mesh whose data axis every call is split over (the plan's
            [mesh][diff_soup.optimization.inference.engine._plan.Plan]); ``None`` for a
            program that runs every call on one device.
    """

    identity: LaneIdentity
    mode: str
    chunk: int
    mesh: Mesh | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class CompiledProgram:
    """One entry of the registry: a program compiled for one key.

    Attributes:
        key: The [ProgramKey][diff_soup.optimization.inference.engine._batched.ProgramKey].
        program: The [LaneProgram][diff_soup.optimization.inference.engine._kernel.LaneProgram]
            of the bucket that compiled it.
        kernel: The kernel of one call of
            [batched_kernel][diff_soup.optimization.inference.engine._kernel.batched_kernel].
        compiled: The compiled executable; called as
            ``compiled(points, lanes)``.
        lanes_tree: The tree structure of the lane inputs it was traced with.
        lane_leaves: Per leaf of the lane inputs, its shape without the lane
            axis and its dtype.
        record: The [ProgramRecord][diff_soup.optimization.inference.engine._prepare.ProgramRecord]
            of its compile.
    """

    key: ProgramKey
    program: LaneProgram
    kernel: Callable[..., LaneResult]
    compiled: Compiled
    lanes_tree: PyTreeDef
    lane_leaves: tuple
    record: Any


# Every compiled program of the process that a study's lanes hold, by its ProgramKey.
_PROGRAMS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class PreparedProgram(NamedTuple):
    """The program of one bucket and mode after ``prepare``.

    Attributes:
        bucket: The bucket's index in the plan.
        mode: The mode.
        program: The [CompiledProgram][diff_soup.optimization.inference.engine._batched.CompiledProgram].
        reused: Whether the program was compiled before this call, for this
            study or another one.
    """

    bucket: int
    mode: str
    program: CompiledProgram
    reused: bool


@dataclasses.dataclass(frozen=True, eq=False)
class BucketLanes:
    """The numbers of one bucket's lanes, on the host.

    Attributes:
        bucket: The planned [Bucket][diff_soup.optimization.inference.engine._plan.Bucket].
        program: The bucket's [LaneProgram][diff_soup.optimization.inference.engine._kernel.LaneProgram].
        leaves: The leaves of its stacked lane inputs, as numpy arrays leading
            with the lane axis, in the order of ``bucket.lanes``.
        tree: The tree structure of its lane inputs.
        gather: Per lane and narrow coordinate, the column of the joint
            parameter vector it is read from.
        owners: Per lane, the index of its owner in the joint problem.
        local: Per lane and local narrow coordinate, its position in the
            local block of its owner.
        frames: Per lane, its own sample counts.
        dtype: The dtype the lanes' parameter points are cast to.
        dt0_leaf: The position in ``leaves`` of the lanes' first steps,
            ``LaneInputs.dt0``.
        state_sizes: Per lane, the size of its initial state; empty when
            not recorded.

    The numbers of the plan alone ([lane_metadata][diff_soup.optimization.inference.engine._batched.lane_metadata])
    carry no inputs: ``program`` and ``tree`` are ``None``, ``leaves`` is
    empty and ``dt0_leaf`` is ``-1``.
    """

    bucket: Bucket
    program: LaneProgram
    leaves: tuple
    tree: PyTreeDef | None
    gather: np.ndarray
    owners: np.ndarray
    local: np.ndarray
    frames: tuple
    dtype: np.dtype
    dt0_leaf: int
    state_sizes: tuple = ()


@dataclasses.dataclass(frozen=True, eq=False)
class BatchedLanes:
    """The lanes of one planned study, the programs they hold, and what evaluations learn.

    Attributes:
        plan: The study's [Plan][diff_soup.optimization.inference.engine._plan.Plan].
        buckets: One [BucketLanes][diff_soup.optimization.inference.engine._batched.BucketLanes]
            per bucket of the plan, in the plan's order.
        lane_ids: Every experiment id of the study that has a lane or was
            refused one, by experiment index.
        where: Experiment id to ``(bucket index, position in the bucket)``.
        refused: Experiment id to the reason it has no lane.
        x0: The joint parameter vector at the start point.
        local_widths: Per owner of the joint problem, the width of its local
            block; without an owner tier, one block that holds every lane.
        n_shared: The number of shared coordinates.
        programs: ``(bucket index, mode)`` to the
            [CompiledProgram][diff_soup.optimization.inference.engine._batched.CompiledProgram]
            the study holds; ``prepare`` adds to it.
        memory: The trial steps every unit took when it was last evaluated,
            which order the units of the next evaluation.
        resident: Per bucket index, the device inputs of the calls of its last
            evaluation, keyed by their lanes.
        kept: Per bucket index, the
            [KeptJacobians][diff_soup.optimization.inference.engine._kept.KeptJacobians]
            of its units' last ``blocks`` results, once the ``frozen`` mode is
            prepared.

    ``memory``, ``resident`` and ``kept`` are what one series of evaluations
    learns; [with_own_state][diff_soup.optimization.inference.engine._batched.with_own_state]
    gives lanes that share everything else and start them afresh, as every fit
    of a lowered study does.
    """

    plan: Plan
    buckets: tuple
    lane_ids: tuple
    where: Mapping
    refused: Mapping
    x0: np.ndarray
    local_widths: tuple
    n_shared: int
    programs: dict = dataclasses.field(default_factory=dict, repr=False)
    memory: StepMemory = dataclasses.field(default_factory=StepMemory, repr=False)
    resident: dict = dataclasses.field(default_factory=dict, repr=False)
    kept: dict = dataclasses.field(default_factory=dict, repr=False)


def batched_lanes(plan: Plan, joint: Any) -> BatchedLanes:
    """The lanes of a planned study, built on the host; nothing is compiled.

    Each bucket's lanes are padded to its frame class and, when they follow
    schedules, to its segment class.

    Args:
        plan: The plan of the study, from
            [build_plan][diff_soup.optimization.inference.engine._plan.build_plan].
        joint: The fitted problem the plan was built from.

    Returns:
        The [BatchedLanes][diff_soup.optimization.inference.engine._batched.BatchedLanes],
        with no program yet.
    """
    layout = lane_layout(joint)
    problems = lane_problems(joint)
    buckets, where, index = [], {}, {}
    for bucket in plan.buckets:
        group = _lane_group(
            layout,
            problems,
            bucket.children,
            frames=bucket.frames,
            segments=bucket.segments,
            device=plan.device,
        )
        if tuple(group.children) != tuple(bucket.children):
            raise RuntimeError(
                f"the lanes of bucket {bucket.index} came back for the experiments "
                f"{tuple(group.children)}, and the plan names {tuple(bucket.children)}."
            )
        leaves, tree = jax.tree_util.tree_flatten(group.lanes)
        dt0_leaf = next(
            (position for position, leaf in enumerate(leaves) if leaf is group.lanes.dt0), None
        )
        if dt0_leaf is None:
            raise RuntimeError(
                f"the lane inputs of bucket {bucket.index} hold no leaf for their first steps."
            )
        buckets.append(
            BucketLanes(
                bucket=bucket,
                program=group.program,
                leaves=tuple(np.asarray(leaf) for leaf in leaves),
                tree=tree,
                gather=np.asarray(group.gather, dtype=np.int64),
                owners=np.asarray(group.owners, dtype=np.int64),
                local=np.asarray(group.local, dtype=np.int32),
                frames=tuple(lane_frames(problems[child]) for child in bucket.children),
                dtype=np.dtype(group.lanes.t0.dtype),
                dt0_leaf=int(dt0_leaf),
                state_sizes=(int(np.asarray(group.lanes.y0)[0].size),) * len(bucket.children),
            )
        )
        for position, (lane, child) in enumerate(zip(bucket.lanes, bucket.children)):
            where[lane] = (bucket.index, position)
            index[lane] = int(child)
    refused = {entry.lane: entry.reason for entry in plan.refused}
    planned = sorted(where, key=index.__getitem__)
    return BatchedLanes(
        plan=plan,
        buckets=tuple(buckets),
        lane_ids=tuple(planned) + tuple(lane for lane in refused if lane not in where),
        where=types.MappingProxyType(where),
        refused=types.MappingProxyType(refused),
        x0=np.asarray(layout.start),
        local_widths=layout.node_widths,
        n_shared=layout.shared,
    )


def replanned_lanes(lanes: BatchedLanes, plan: Plan) -> BatchedLanes:
    """The lanes of a study under another plan of the same buckets, built from its own numbers.

    ``plan`` holds the buckets of ``lanes.plan``, in the same order with the
    same lanes and modes, and cuts their units into calls of other widths, as
    [population_plan][diff_soup.optimization.inference.engine._plan.population_plan] does.
    The host numbers of the lanes are shared, not built again; the programs,
    the step memory and the inputs and Jacobians kept on the device start
    empty, so nothing of ``lanes`` changes.

    Args:
        lanes: The lanes of the study.
        plan: The other plan.

    Returns:
        The [BatchedLanes][diff_soup.optimization.inference.engine._batched.BatchedLanes]
        of ``plan``, with no program yet.

    Raises:
        ValueError: If ``plan`` does not hold the buckets of ``lanes.plan``.
    """
    old = lanes.plan.buckets
    if len(plan.buckets) != len(old) or any(
        new.index != entry.index
        or tuple(new.lanes) != tuple(entry.lanes)
        or tuple(new.chunk) != tuple(entry.chunk)
        for new, entry in zip(plan.buckets, old)
    ):
        raise ValueError(
            "the plan does not hold the buckets of the study's plan: the same buckets, lanes "
            "and modes in the same order are needed to share the study's lanes."
        )
    return dataclasses.replace(
        lanes,
        plan=plan,
        buckets=tuple(
            dataclasses.replace(entry, bucket=bucket)
            for entry, bucket in zip(lanes.buckets, plan.buckets)
        ),
        programs={},
        memory=StepMemory(),
        resident={},
        kept={},
    )


def with_own_state(lanes: BatchedLanes) -> BatchedLanes:
    """The lanes of a study with a step memory and device inputs of their own.

    The plan, the host numbers of the lanes and the programs are those of
    ``lanes``, shared: a program ``prepare`` compiles through either is held
    by both. The step memory starts empty, so the first evaluation runs its
    units in canonical order, and no input or Jacobian is kept on the device
    yet; what evaluations through the returned lanes learn never reaches
    ``lanes``, and the other way round.

    Args:
        lanes: The lanes of the study.

    Returns:
        The [BatchedLanes][diff_soup.optimization.inference.engine._batched.BatchedLanes]
        sharing the programs of ``lanes``.
    """
    return dataclasses.replace(lanes, memory=StepMemory(), resident={}, kept={})


def lane_metadata(plan: Plan, joint: Any) -> BatchedLanes:
    """The lanes of a planned study without their inputs: where every lane reads, nothing else.

    The process that coordinates a run of several processes answers every
    question about the lanes (which columns a lane reads, its owner, its
    frames) for lanes it does not evaluate, without building their inputs.
    Nothing is built on a device; the returned lanes evaluate nothing.

    Args:
        plan: The plan of the study.
        joint: The fitted problem the plan was built from.

    Returns:
        The [BatchedLanes][diff_soup.optimization.inference.engine._batched.BatchedLanes]
        whose buckets carry no inputs.
    """
    from diff_soup.optimization.inference.fitting._joint_residual import _resolve_child_y0

    layout = lane_layout(joint)
    problems = lane_problems(joint)
    buckets, where, index = [], {}, {}
    for bucket in plan.buckets:
        children = tuple(bucket.children)
        first = problems[children[0]]
        size = int(np.size(_resolve_child_y0(first)))
        buckets.append(
            BucketLanes(
                bucket=bucket,
                program=None,
                leaves=(),
                tree=None,
                gather=np.stack([np.asarray(layout.columns[c]) for c in children]).astype(np.int64),
                owners=np.asarray([layout.nodes[c] for c in children], dtype=np.int64),
                local=np.stack([np.asarray(layout.local[c], dtype=np.int32) for c in children]),
                frames=tuple(lane_frames(problems[child]) for child in children),
                dtype=np.dtype(np.asarray(first._seam.x0).dtype),
                dt0_leaf=-1,
                state_sizes=(size,) * len(children),
            )
        )
        for position, (lane, child) in enumerate(zip(bucket.lanes, children)):
            where[lane] = (bucket.index, position)
            index[lane] = int(child)
    refused = {entry.lane: entry.reason for entry in plan.refused}
    planned = sorted(where, key=index.__getitem__)
    return BatchedLanes(
        plan=plan,
        buckets=tuple(buckets),
        lane_ids=tuple(planned) + tuple(lane for lane in refused if lane not in where),
        where=types.MappingProxyType(where),
        refused=types.MappingProxyType(refused),
        x0=np.asarray(layout.start),
        local_widths=layout.node_widths,
        n_shared=layout.shared,
    )


def restricted_plan(plan: Plan, lanes: Iterable[Hashable]) -> tuple:
    """The plan of a subset of a study's lanes, each bucket's programs as wide as in ``plan``.

    A bucket keeps the lanes of ``lanes`` in its own order; a bucket left
    with none is dropped. The kept buckets are numbered from 0 in the plan's
    order. Each kept bucket's chunk width per mode is the one ``plan`` gives
    it, so its programs are the programs of ``plan`` whatever the subset; the
    calls and the copies that fill the last one count the kept lanes.

    Args:
        plan: The study's plan (or its cut for a population).
        lanes: The lane ids to keep.

    Returns:
        ``(restricted, origin)``: the restricted
        [Plan][diff_soup.optimization.inference.engine.Plan] and, per kept bucket, its
        index in ``plan``.
    """
    from diff_soup.optimization.inference.engine._plan import Chunk

    keep = set(lanes)
    buckets, origin = [], []
    for bucket in plan.buckets:
        positions = [p for p, lane in enumerate(bucket.lanes) if lane in keep]
        if not positions:
            continue
        chunk = {}
        for mode, entry in bucket.chunk.items():
            # The units of the whole bucket are its members times its lanes.
            members = max((entry.width * entry.chunks - entry.padded) // len(bucket.lanes), 1)
            units = members * len(positions)
            count = -(-units // entry.width)
            chunk[mode] = Chunk(width=entry.width, chunks=count, padded=count * entry.width - units)
        buckets.append(
            dataclasses.replace(
                bucket,
                index=len(buckets),
                lanes=tuple(bucket.lanes[p] for p in positions),
                owners=tuple(bucket.owners[p] for p in positions),
                children=tuple(bucket.children[p] for p in positions),
                chunk=types.MappingProxyType(chunk),
            )
        )
        origin.append(int(bucket.index))
    return dataclasses.replace(plan, buckets=tuple(buckets)), tuple(origin)


def _without(mode: str) -> str:
    """Why a bucket's program has no ``mode``."""
    if mode == "hessian":
        return (
            "the adjoints of its solver cannot differentiate its lanes' reverse gradient "
            "forward, which that mode does"
        )
    return (
        "its solver carries a reverse adjoint, which refuses the forward-mode "
        "differentiation of that mode"
    )


def _check_modes(lanes: BatchedLanes, modes: Any) -> tuple:
    """``modes`` as a tuple, each a mode every bucket's program can run in.

    The ``frozen`` mode steps on the Jacobians ``blocks`` evaluations keep, so
    it brings the ``blocks`` mode, before it.
    """
    if isinstance(modes, str):
        modes = (modes,)
    modes = tuple(modes)
    if "frozen" in modes:
        modes = tuple(mode for mode in modes if mode != "blocks")
        at = modes.index("frozen")
        modes = modes[:at] + ("blocks",) + modes[at:]
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"prepare compiles the modes {MODES}; got {mode!r}.")
        for entry in lanes.buckets:
            bucket = entry.bucket
            if mode not in bucket.chunk:
                raise ValueError(
                    f"bucket {bucket.index} (identity {bucket.identity}) has no {mode} mode: "
                    f"{_without(mode)}; prepare the modes {tuple(bucket.chunk)} for it."
                )
    return tuple(dict.fromkeys(modes))


def placement(plan: Plan) -> jax.Device | NamedSharding | None:
    """Where the arrays of one call go: the plan's device, or the lane axis split over its mesh.

    Every array of a call (the points and every leaf of the lane inputs) leads
    with the lane axis; on a mesh that axis is split over the devices of its
    data axis.
    """
    if plan.mesh is None:
        return plan.device
    return NamedSharding(plan.mesh, PartitionSpec("data"))


def _first_call(lanes: BatchedLanes, entry: BucketLanes, width: int) -> tuple:
    """The points and lane inputs of a bucket's first call at the start point, on the device."""
    positions, _own = cut(len(entry.bucket.lanes), width)[0]
    points = lanes.x0[entry.gather[positions]].astype(entry.dtype)
    target = placement(lanes.plan)
    inputs = jax.tree_util.tree_unflatten(
        entry.tree, jax.device_put([leaf[positions] for leaf in entry.leaves], target)
    )
    return jax.device_put(points, target), inputs


def _lane_leaves(entry: BucketLanes) -> tuple:
    return tuple((leaf.shape[1:], leaf.dtype.str) for leaf in entry.leaves)


def prepare(
    lanes: BatchedLanes,
    *,
    modes: Any,
    compile_program: Callable,
    buckets: Iterable[Integral] | None = None,
    keeping: bool = False,
) -> tuple:
    """Compile the program of every bucket (or of the buckets ``buckets``) in every mode of ``modes``.

    A program the study holds already is not looked up again; one the
    registry holds for an equal key is reused; any other is compiled by
    ``compile_program`` and registered. Nothing is solved.

    The ``blocks`` program of a bucket that has, or is given, the ``frozen``
    mode is the one that keeps its lanes' Jacobians
    ([KEEPING_BLOCKS][diff_soup.optimization.inference.engine._kernel.KEEPING_BLOCKS]);
    it replaces a ``blocks`` program held before, and the ``frozen`` mode
    brings the ``blocks`` mode.

    Args:
        lanes: The lanes of the study.
        modes: The modes to compile, among ``MODES``.
        compile_program: ``compile_program(lanes, entry, key)`` returns the
            [CompiledProgram][diff_soup.optimization.inference.engine._batched.CompiledProgram]
            of ``key`` for the bucket ``entry``; a program it raises for is
            not registered.
        buckets: The indices of the buckets to prepare; ``None`` prepares
            every bucket.
        keeping: Whether the ``blocks`` program keeps the Jacobians although
            ``modes`` holds no ``frozen``: a process that prepares one mode at
            a time of a request that holds it.

    Returns:
        One [PreparedProgram][diff_soup.optimization.inference.engine._batched.PreparedProgram]
        per bucket and mode, buckets in the plan's order.

    Raises:
        ValueError: If a mode is not one of ``MODES``, or a bucket's solver
            cannot run it.
    """
    modes = _check_modes(lanes, modes)
    keeping = keeping or "frozen" in modes
    prepared = []
    chosen = None if buckets is None else {int(index) for index in buckets}
    for entry in lanes.buckets:
        bucket = entry.bucket
        if chosen is not None and int(bucket.index) not in chosen:
            continue
        keep = keeping or (bucket.index, "frozen") in lanes.programs
        for mode in modes:
            program_mode = KEEPING_BLOCKS if keep and mode == "blocks" else mode
            held = lanes.programs.get((bucket.index, mode))
            if held is not None and held.key.mode == program_mode:
                prepared.append(PreparedProgram(bucket.index, mode, held, True))
                continue
            key = ProgramKey(
                bucket.full_identity, program_mode, int(bucket.chunk[mode].width), lanes.plan.mesh
            )
            program = _PROGRAMS.get(key)
            reused = program is not None
            if program is None:
                program = compile_program(lanes, entry, key)
                _PROGRAMS[key] = program
            elif program.lane_leaves != _lane_leaves(entry):
                raise ValueError(
                    f"bucket {bucket.index} has the identity {bucket.identity} of a compiled "
                    "program, and its lane inputs have other shapes than that program's; lanes "
                    "of one identity have to carry inputs of equal shapes."
                )
            lanes.programs[(bucket.index, mode)] = program
            prepared.append(PreparedProgram(bucket.index, mode, program, reused))
    return tuple(prepared)


def _resolve(lanes: BatchedLanes, lane_ids: Any) -> tuple:
    """``(bucket index, position)`` of each of ``lane_ids``, every id checked."""
    ids = tuple(lane_ids)
    if not ids:
        raise ValueError("the tables of the tree are built for at least one lane; got none.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"the lanes {ids} name an experiment more than once; name each once.")
    canonical_lanes(ids, None, study_ids=lanes.lane_ids, planned=lanes.where, refused=lanes.refused)
    return tuple(lanes.where[lane] for lane in ids)


def _one_width(lanes: BatchedLanes, ids: tuple) -> tuple:
    """``_resolve`` of ``ids``, refusing lanes of more than one narrow width."""
    places = _resolve(lanes, ids)
    widths = {lanes.buckets[bucket].bucket.width for bucket, _ in places}
    if len(widths) != 1:
        raise ValueError(
            f"the lanes {ids} have the narrow widths {sorted(widths)}; their results are reduced "
            "only at one width, so build tables for the lanes of each width apart."
        )
    return places


def _refuse_nodes_without_lanes(
    lanes: BatchedLanes, hierarchy: Hierarchy, experiments: Iterable
) -> None:
    """Refuse a hierarchy study with a node every experiment below which the plan refused.

    The level tables hold every node of the hierarchy; a node with no planned
    lane below it has no lane result to sum. The outermost such node is named,
    with the plan's reason for each of its experiments.
    """
    if not lanes.refused:
        return
    from diff_soup.optimization.inference.statistics.hierarchy import _tree

    tree = _tree(hierarchy, experiments)
    for position, level in enumerate(tree.names):
        column = tree.nodes[:, position]
        for node, member in enumerate(tree.members[position]):
            below = tuple(tree.ids[index] for index in np.flatnonzero(column == node))
            if not all(lane in lanes.refused for lane in below):
                continue
            reasons = "; ".join(f"{lane}: {lanes.refused[lane]}" for lane in below[:8])
            if len(below) > 8:
                reasons += f"; and {len(below) - 8} more"
            raise ValueError(
                f"node {member!r} of level {level!r} has no lane: the plan refused one to every "
                f"experiment below it ({reasons}). The level tables hold every node of the "
                "hierarchy, so every node needs an experiment with a lane; LoweredStudy.plan() "
                "lists the reasons."
            )


def level_tables_of(
    lanes: BatchedLanes, lane_ids: Any, hierarchy: Hierarchy, experiments: Iterable
) -> LevelTables:
    """The level tables of a hierarchy study for lane results in the order of ``lane_ids``.

    Args:
        lanes: The lanes of the study.
        lane_ids: Experiment ids with a lane, in the order of the lane
            results the tables will reduce, as ``Results.lane_ids`` gives it.
        hierarchy: The study's hierarchy.
        experiments: Every experiment of the study, in study order.

    Returns:
        The [LevelTables][diff_soup.optimization.inference.engine.LevelTables], on the
        device of the plan.

    Raises:
        ValueError: If an id has no lane, an id repeats, the lanes differ in
            narrow width, or the plan refused a lane to every experiment below
            a node of the hierarchy.
    """
    from diff_soup.optimization.inference.engine._levels import level_tables

    ids = tuple(lane_ids)
    _one_width(lanes, ids)
    _refuse_nodes_without_lanes(lanes, hierarchy, experiments)
    return level_tables(hierarchy, experiments, lanes=ids, device=lanes.plan.device)


def tables(lanes: BatchedLanes, lane_ids: Any) -> TreeTables:
    """The tables of the parameter tree for lane results in the order of ``lane_ids``.

    Args:
        lanes: The lanes of the study.
        lane_ids: Experiment ids with a lane, in the order of the lane
            results the tables will reduce, as ``Results.lane_ids`` gives it.

    Returns:
        The [TreeTables][diff_soup.optimization.inference.engine._reducers.TreeTables], on the
        device of the plan.

    Raises:
        ValueError: If an id has no lane, an id repeats, or the lanes differ
            in narrow width; lanes of different widths are reduced apart.
    """
    ids = tuple(lane_ids)
    places = _one_width(lanes, ids)
    return tree_tables(
        np.asarray([lanes.buckets[bucket].owners[position] for bucket, position in places]),
        np.stack([lanes.buckets[bucket].local[position] for bucket, position in places]),
        lanes.local_widths,
        lanes.n_shared,
        device=lanes.plan.device,
    )


def _points(lanes: BatchedLanes, request: Request) -> np.ndarray:
    """The joint parameter vectors of the members that take part, checked against the study."""
    points = request.points
    width = int(lanes.x0.shape[0])
    if points.shape[1] != width:
        raise ValueError(
            f"points= gives {points.shape[1]} entries per member, and this study's joint "
            f"parameter vector has {width}; give points of shape ({width},) or (P, {width})."
        )
    return points[request.members]


def _check_programs(lanes: BatchedLanes, mode: str, touched: list) -> None:
    """Refuse a request whose buckets have no compiled program in ``mode``."""
    for index in touched:
        bucket = lanes.buckets[index].bucket
        if mode not in bucket.chunk:
            raise ValueError(
                f"bucket {bucket.index} (identity {bucket.identity}) has no {mode} mode: "
                f"{_without(mode)}; request one of {tuple(bucket.chunk)}."
            )
        if (index, mode) not in lanes.programs:
            raise ValueError(
                f"the {mode} mode of bucket {bucket.index} (identity {bucket.identity}) is not "
                f"compiled, and an evaluation compiles nothing; call prepare(modes=({mode!r},)) "
                "before evaluating it."
            )
    if mode in ("blocks", "gradient", "hessian", "frozen"):
        widths = {lanes.buckets[index].bucket.width for index in touched}
        if len(widths) > 1:
            raise ValueError(
                f"the requested lanes have the narrow widths {sorted(widths)}, and the {mode} "
                "mode returns derivatives of one width; request the lanes of each width apart."
            )


def _run_bucket(
    lanes: BatchedLanes,
    index: int,
    mode: str,
    points: np.ndarray,
    columns: np.ndarray,
    positions: np.ndarray,
    priority: np.ndarray,
    members: np.ndarray,
    starts: np.ndarray,
    *,
    keep_inputs: bool = True,
    keep: np.ndarray | None = None,
    keys: np.ndarray | None = None,
) -> tuple:
    """Evaluate the units of one bucket; its call outputs, its wall seconds and its batch.

    ``columns`` and ``positions`` give, per requested lane of the bucket in
    canonical order, its column in the results and its position in the
    bucket; ``priority`` its rank in the request's ``order``. ``points`` and
    ``starts`` have one row per entry of ``members``; ``starts`` gives, per
    member and lane of the bucket, the first step the unit's solve starts
    from. A padded copy of a unit starts from ``0``, the integrator's
    estimate. A call whose first steps are its lanes' own takes the lane
    inputs kept on the device; any other call sends its first steps along.
    ``keep_inputs=False`` leaves the inputs kept from the last evaluation in
    place, for a pass that solves a few units again. ``keep``, of shape
    (members, lanes of the bucket), names the units to run when not every
    unit of that rectangle is evaluated; ``None`` runs every one. The
    [Batch][diff_soup.optimization.inference.engine._result.Batch] counts the units, the
    width and the calls this pass made. Through the program that keeps the
    Jacobians, each call is also returned as a
    [KeptCall][diff_soup.optimization.inference.engine._kept.KeptCall] with its count of
    own units, its Jacobians left on the device, its units named by ``keys``
    (one per entry of ``members``; ``None`` takes ``members``).

    Returns:
        ``(outputs, seconds, batch, kept)``: ``kept`` holds ``(call, own)`` per
        call, empty for any other program.

    Raises:
        BackendError: If a call of the program, or the transfer of its
            results to the host, raises; it names the bucket, the mode and
            the width, and is raised from the original error.
    """
    entry = lanes.buckets[index]
    program = lanes.programs[(index, mode)]
    keeping = program.key.mode == KEEPING_BLOCKS
    device = placement(lanes.plan)
    width = int(program.key.chunk)
    # Canonical units: members ascending, then lanes by experiment index.
    unit_row = np.repeat(np.arange(members.shape[0], dtype=np.int64), positions.shape[0])
    unit_lane = np.tile(np.arange(positions.shape[0], dtype=np.int64), members.shape[0])
    if keep is not None:
        kept = np.asarray(keep, dtype=bool)[unit_row, unit_lane]
        unit_row, unit_lane = unit_row[kept], unit_lane[kept]
    count = int(unit_row.shape[0])
    steps = lanes.memory.steps(index, members[unit_row], positions[unit_lane])
    order = unit_order(priority[unit_lane], steps)
    # On a mesh, each call's units are dealt to its devices by their last step counts.
    order = order[device_order(steps[order], width, lanes.plan.devices)]
    row, lane = unit_row[order], unit_lane[order]
    narrow = points.astype(entry.dtype)[row[:, None], entry.gather[positions[lane]]]
    configured = entry.leaves[entry.dt0_leaf]
    first = np.asarray(starts).astype(configured.dtype)[row, lane]

    previous = lanes.resident.get(index, {})
    resident, pending, kept = {}, [], []
    start = time.perf_counter()
    calls = cut(count, width)
    try:
        for call, own in calls:
            chosen = positions[lane[call]]
            held = chosen.tobytes()
            inputs = resident.get(held)
            if inputs is None:
                inputs = previous.get(held)
            if inputs is None:
                inputs = jax.device_put([leaf[chosen] for leaf in entry.leaves], device)
            resident[held] = inputs
            call_first = first[call]
            call_first[own:] = 0
            if call_first.tobytes() != configured[chosen].tobytes():
                inputs = list(inputs)
                inputs[entry.dt0_leaf] = jax.device_put(call_first, device)
            output = program.compiled(
                jax.device_put(narrow[call], device),
                jax.tree_util.tree_unflatten(program.lanes_tree, inputs),
            )
            if keeping:
                # The Jacobians stay on the device, with what solves the call again.
                kept.append(
                    (
                        KeptCall(
                            jacobian=output.jacobian,
                            members=(members if keys is None else keys)[row[call]],
                            positions=chosen,
                            narrow=narrow[call],
                            first=call_first,
                            inputs=list(inputs),
                        ),
                        own,
                    )
                )
                output = dataclasses.replace(output, jacobian=None)
            pending.append((call[:own], output))
        fetched = jax.device_get([output for _, output in pending])
    except (jax.errors.JaxRuntimeError, ValueError) as error:
        # A program that fails at run time raises JaxRuntimeError, or ValueError once it
        # has run before (JAX's fast dispatch path); both carry the runtime's message.
        raise _call_failed(entry.bucket, mode, count, width, len(calls), error) from error
    seconds = time.perf_counter() - start
    if keep_inputs:
        lanes.resident[index] = resident

    outputs = []
    for (units, _), output in zip(pending, fetched):
        own = int(units.shape[0])
        outputs.append(
            CallOutput(
                rows=row[units],
                columns=columns[lane[units]],
                output=output,
                frames=tuple(entry.frames[p].streams for p in positions[lane[units]]),
            )
        )
        lanes.memory.record(
            index,
            len(entry.bucket.lanes),
            members[row[units]],
            positions[lane[units]],
            np.asarray(output.work.trial_steps)[:own],
        )
    return tuple(outputs), seconds, Batch(units=count, width=width, calls=len(pending)), kept


def _run_frozen(
    lanes: BatchedLanes,
    index: int,
    points: np.ndarray,
    columns: np.ndarray,
    positions: np.ndarray,
    members: np.ndarray,
    starts: np.ndarray,
    *,
    keep: np.ndarray | None = None,
    keys: np.ndarray | None = None,
) -> tuple:
    """Evaluate the units of one bucket in the ``frozen`` mode, each on its kept Jacobian.

    The arguments are those of ``_run_bucket``; a unit's Jacobian is found by
    its member's entry of ``keys`` (``None`` takes ``members``). Every kept
    call that holds a requested unit runs once, in its own layout and with its
    own Jacobians:
    each requested unit at its point and first step, every other slot at the
    point and first step it was kept with, its result dropped. The order of
    the units and the step memory play no part, and the inputs kept for other
    modes are left as they are.

    Raises:
        ValueError: If a requested unit has no kept Jacobian.
        BackendError: As ``_run_bucket``.
    """
    entry = lanes.buckets[index]
    program = lanes.programs[(index, "frozen")]
    device = placement(lanes.plan)
    width = int(program.key.chunk)
    unit_row = np.repeat(np.arange(members.shape[0], dtype=np.int64), positions.shape[0])
    unit_lane = np.tile(np.arange(positions.shape[0], dtype=np.int64), members.shape[0])
    if keep is not None:
        kept_units = np.asarray(keep, dtype=bool)[unit_row, unit_lane]
        unit_row, unit_lane = unit_row[kept_units], unit_lane[kept_units]
    count = int(unit_row.shape[0])
    kept = lanes.kept.get(index, KeptJacobians())
    keys = members if keys is None else keys
    groups: dict = {}
    for unit, (row, lane) in enumerate(zip(unit_row, unit_lane)):
        place = kept.where.get((int(keys[row]), int(positions[lane])))
        if place is None:
            raise ValueError(
                f"lane {entry.bucket.lanes[int(positions[lane])]!r} of member {int(members[row])} "
                "has no kept Jacobian: the frozen mode steps on the Jacobian of a unit's last "
                "blocks evaluation after prepare(modes=(..., 'frozen')); evaluate the blocks "
                "mode for it first."
            )
        groups.setdefault(place[0], []).append((unit, place[1]))
    narrow = points.astype(entry.dtype)[unit_row[:, None], entry.gather[positions[unit_lane]]]
    first = np.asarray(starts).astype(entry.leaves[entry.dt0_leaf].dtype)[unit_row, unit_lane]

    pending = []
    start = time.perf_counter()
    try:
        for number in sorted(groups):
            call = kept.calls[number]
            units = np.asarray([unit for unit, _ in groups[number]], dtype=np.int64)
            slots = np.asarray([slot for _, slot in groups[number]], dtype=np.int64)
            call_narrow = call.narrow.copy()
            call_narrow[slots] = narrow[units]
            call_first = call.first.copy()
            call_first[slots] = first[units]
            inputs = call.inputs
            if call_first.tobytes() != call.first.tobytes():
                inputs = list(inputs)
                inputs[entry.dt0_leaf] = jax.device_put(call_first, device)
            output = program.compiled(
                jax.device_put(call_narrow, device),
                jax.tree_util.tree_unflatten(program.lanes_tree, inputs),
                call.jacobian,
            )
            pending.append((units, slots, output))
        fetched = jax.device_get([output for _, _, output in pending])
    except (jax.errors.JaxRuntimeError, ValueError) as error:
        raise _call_failed(entry.bucket, "frozen", count, width, len(pending), error) from error
    seconds = time.perf_counter() - start

    outputs = []
    for (units, slots, _), output in zip(pending, fetched):
        own = jax.tree_util.tree_map(lambda leaf, slots=slots: np.asarray(leaf)[slots], output)
        lane = unit_lane[units]
        outputs.append(
            CallOutput(
                rows=unit_row[units],
                columns=columns[lane],
                output=own,
                frames=tuple(entry.frames[p].streams for p in positions[lane]),
            )
        )
        lanes.memory.record(
            index,
            len(entry.bucket.lanes),
            members[unit_row[units]],
            positions[lane],
            np.asarray(own.work.trial_steps),
        )
    return tuple(outputs), seconds, Batch(units=count, width=width, calls=len(pending))


def _keep(lanes: BatchedLanes, kept: list, replaced: set | None) -> None:
    """Keep the Jacobians of ``kept``, ``(bucket index, (call, own))`` per call.

    Each call's own units point at it; with ``replaced``, a set of
    ``(member, bucket index, position)``, only those of its own units do: a
    pass that solved some units again replaces those units' results only.
    """
    for index, (call, own) in kept:
        slots = range(own)
        if replaced is not None:
            slots = [
                slot
                for slot in slots
                if (int(call.members[slot]), index, int(call.positions[slot])) in replaced
            ]
        lanes.kept.setdefault(index, KeptJacobians()).keep(call, slots)


def _call_failed(
    bucket: Bucket, mode: str, units: int, width: int, calls: int, error: Exception
) -> BackendError:
    """The error of a bucket whose program raised: what ran, what failed, what satisfies it."""
    return BackendError(
        f"the {mode} program of bucket {bucket.index} (identity {bucket.identity}) raised while "
        f"it evaluated {units} unit(s) in {calls} call(s) of width W = {width}: "
        f"{type(error).__name__}: {error}. When device memory ran out, a smaller chunk= "
        "(ds.lower(study, chunk=...), or LoweredStudy.for_population(members, chunk=...)) needs "
        "less of it per call; memory that the sparse direct solver takes outside JAX's pool is "
        "left free by XLA_PYTHON_CLIENT_PREALLOCATE=false or a smaller "
        "XLA_PYTHON_CLIENT_MEM_FRACTION, set before JAX starts."
    )


def _requested_rungs(request: Request, shape: tuple, factors: np.ndarray) -> tuple:
    """Per unit, the rung the request asks for and its rung-0 first step, checked."""
    rungs = np.zeros(shape, dtype=np.int64)
    bases = np.zeros(shape, dtype=np.float64)
    if request.rungs is not None:
        if request.rungs.shape != shape:
            raise ValueError(
                f"rungs= has the shape {request.rungs.shape}, and the request selects "
                f"{shape[0]} member(s) and {shape[1]} lane(s); give rungs= of the shape {shape}, "
                "members and lanes in canonical order."
            )
        highest = int(request.rungs.max())
        if highest > factors.shape[0]:
            raise ValueError(
                f"rungs= asks for rung {highest}, and this study retries from "
                f"{factors.shape[0]} smaller first step(s) (retry_first_step_factors="
                f"{tuple(float(f) for f in factors)}); give rungs from 0 to {factors.shape[0]}."
            )
        rungs = np.array(request.rungs, dtype=np.int64)
    if request.first_steps is not None:
        if request.first_steps.shape != shape:
            raise ValueError(
                f"first_steps= has the shape {request.first_steps.shape}, and the request selects "
                f"{shape[0]} member(s) and {shape[1]} lane(s); give first_steps= of the shape "
                f"{shape}."
            )
        bases = np.array(request.first_steps, dtype=np.float64)
    return rungs, bases


def evaluate(
    lanes: BatchedLanes,
    request: Request,
    *,
    retry_kinds: Iterable[str] = (),
    retry_first_step_factors: Iterable[Real] = (),
    member_keys: Any = None,
) -> Results:
    """Evaluate a request through the compiled programs of a study.

    Every requested unit returns exactly one result, ``ok`` or failed; a lane
    that fails does not raise. The results come back in canonical order,
    whatever the calls, their order, ``order`` or the selection.

    A ``blocks`` request of a study that prepared the ``frozen`` mode keeps
    the Jacobian of every unit's result on the device, replacing the one kept
    before; a ``frozen`` request runs each unit on its kept Jacobian.

    Every unit first runs at the rung the request names (0 when it names
    none). Then, while a unit has failed with a kind of ``retry_kinds``, is
    below the last rung and has a nonzero rung-0 first step, those units are
    run again at their next rung: rung ``r`` starts the solve from
    ``retry_first_step_factors[r - 1]`` times the unit's rung-0 first step,
    the one its rung-0 result reports when it ran at rung 0 here, else the
    request's ``first_steps``. One such pass runs the members and lanes of
    those units through the same programs; a unit of that selection that did
    not fail runs at its own rung and keeps its earlier result. Each unit's
    result is the one of the last rung it ran at.

    Args:
        lanes: The lanes of the study, with the programs of the requested
            mode prepared.
        request: The [Request][diff_soup.optimization.inference.engine._request.Request].
        retry_kinds: The names of ``KINDS`` whose failures are solved again;
            empty solves nothing again.
        retry_first_step_factors: The factors of the rungs, in the order
            they are tried; empty solves nothing again.
        member_keys: Per member that takes part, the number its kept
            Jacobians are known by; ``None`` takes the members' own. A process
            that evaluates one group of a request's members gives their
            numbers in the request.

    Returns:
        The [Results][diff_soup.optimization.inference.engine._result.Results], with the
        rung that produced each unit's result in ``status.rung``, one record
        per unit solved again in ``retries``, and per bucket touched the
        units, width and calls of its first pass in ``batches``.

    Raises:
        ValueError: If the points do not fit the study, a lane is not an
            experiment of the study or has no lane, ``order`` names a lane the
            request does not ask for, the request selects no unit, a bucket's
            program for the mode is not compiled, the lanes of a derivative
            mode differ in narrow width, ``rungs`` or ``first_steps`` do not
            have one entry per requested unit or ask for a rung above the
            number of ``retry_first_step_factors``, or a unit of a ``frozen``
            request has no kept Jacobian.
        BackendError: If a call of a bucket's program raises; the error
            names the bucket, the mode and the width and is raised from the
            original error.
    """
    if not isinstance(request, Request):
        raise ValueError(f"evaluate takes a Request; got {type(request).__name__}.")
    points = _points(lanes, request)
    chosen = canonical_lanes(
        request.lanes,
        request.order,
        study_ids=lanes.lane_ids,
        planned=lanes.where,
        refused=lanes.refused,
    )
    members = request.members
    if not chosen or not members.shape[0]:
        raise ValueError(
            f"the request selects no unit: {len(chosen)} lane(s) and {members.shape[0]} active "
            "member(s); select at least one lane and one member."
        )
    present = request.unit_mask(len(chosen))
    if present is not None and not np.any(present):
        raise ValueError(
            "the request selects no unit: units= leaves out every unit of its members and "
            "lanes; set at least one entry of units= to True."
        )
    bucket_of = np.asarray([lanes.where[lane][0] for lane in chosen], dtype=np.int64)
    rank = {lane: position for position, lane in enumerate(request.order or ())}
    priority = np.asarray([rank.get(lane, len(rank)) for lane in chosen], dtype=np.int64)
    if present is None:
        touched = sorted(set(bucket_of.tolist()))
    else:
        # A bucket none of whose units is evaluated is not touched.
        touched = sorted(set(bucket_of[np.any(present, axis=0)].tolist()))
    _check_programs(lanes, request.mode, touched)
    # A bucket that holds a priority lane runs first.
    touched.sort(key=lambda index: (int(priority[bucket_of == index].min()), index))

    shape = (int(members.shape[0]), len(chosen))
    keys = members if member_keys is None else np.asarray(member_keys, dtype=np.int64)
    if keys.shape != members.shape:
        raise ValueError(
            f"member_keys= gives {keys.shape[0]} numbers for {members.shape[0]} member(s); give "
            "one per member that takes part."
        )
    factors = np.asarray(tuple(float(f) for f in retry_first_step_factors), dtype=np.float64)
    codes = np.asarray([KINDS.index(kind) for kind in retry_kinds], dtype=np.int64)
    rung, base = _requested_rungs(request, shape, factors)
    position_of = np.asarray([lanes.where[lane][1] for lane in chosen], dtype=np.int64)
    configured = np.asarray(
        [
            lanes.buckets[bucket].leaves[lanes.buckets[bucket].dt0_leaf][position]
            for bucket, position in zip(bucket_of, position_of)
        ],
        dtype=np.float64,
    )

    def starts(at: np.ndarray) -> np.ndarray:
        """Per unit, the first step its solve starts from at the rungs ``at``."""
        first = np.broadcast_to(configured, shape).copy()
        climbed = at > 0
        first[climbed] = factors[at[climbed] - 1] * base[climbed]
        return first

    def run(rows: np.ndarray, columns_of: np.ndarray, at: np.ndarray, again: bool) -> tuple:
        """The calls, walls, batches and kept calls of the units ``rows`` x ``columns_of``."""
        first = starts(at)
        outputs, walls, batches, kept = [], {}, {}, []
        for index in touched:
            columns = columns_of[bucket_of[columns_of] == index]
            if not columns.size:
                continue
            keep = None if present is None else present[np.ix_(rows, columns)]
            if keep is not None and not np.any(keep):
                continue
            if request.mode == "frozen":
                out, walls[index], batches[index] = _run_frozen(
                    lanes,
                    index,
                    points[rows],
                    columns,
                    position_of[columns],
                    members[rows],
                    first[np.ix_(rows, columns)],
                    keep=keep,
                    keys=keys[rows],
                )
            else:
                out, walls[index], batches[index], held = _run_bucket(
                    lanes,
                    index,
                    request.mode,
                    points[rows],
                    columns,
                    position_of[columns],
                    priority[columns],
                    members[rows],
                    first[np.ix_(rows, columns)],
                    keep_inputs=not again,
                    keep=keep,
                    keys=keys[rows],
                )
                kept.extend((index, call) for call in held)
            outputs.extend(call._replace(rows=rows[call.rows]) for call in out)
        return outputs, walls, batches, kept

    every_row = np.arange(shape[0], dtype=np.int64)
    calls, walls, batches, kept = run(every_row, np.arange(shape[1], dtype=np.int64), rung, False)
    _keep(lanes, kept, None)
    seconds = dict(walls)
    kind = unit_values(calls, shape, lambda output: output.status.kind)
    made = unit_values(calls, shape, lambda output: output.work.first_step)
    if made is not None:
        # The base of a unit that ran at rung 0 is the first step its solve took.
        base = np.where(rung == 0, np.asarray(made, dtype=np.float64), base)

    tried: dict = {}
    while True:
        due = np.isin(kind, codes) & (rung < factors.shape[0]) & (base != 0.0)
        if not np.any(due):
            break
        for row, column in np.argwhere(due):
            unit = (int(row), int(column))
            if unit not in tried:
                tried[unit] = ([int(rung[unit])], [KINDS[int(kind[unit])]], [])
        rung = np.where(due, rung + 1, rung)
        again, walls, _, kept = run(
            np.flatnonzero(due.any(axis=1)), np.flatnonzero(due.any(axis=0)), rung, True
        )
        _keep(
            lanes,
            kept,
            {
                (int(keys[row]), int(bucket_of[column]), int(position_of[column]))
                for row, column in np.argwhere(due)
            },
        )
        for index, wall in walls.items():
            seconds[index] += wall
        calls = merge_calls(calls, again, due)
        kind = unit_values(calls, shape, lambda output: output.status.kind)
        for row, column in np.argwhere(due):
            unit = (int(row), int(column))
            rungs_tried, kinds, walls_tried = tried[unit]
            rungs_tried.append(int(rung[unit]))
            kinds.append(KINDS[int(kind[unit])])
            walls_tried.append(float(walls[int(bucket_of[column])]))

    retries = tuple(
        {
            "member": int(members[row]),
            "lane": chosen[column],
            "rungs": tuple(rungs_tried),
            "kinds": tuple(kinds),
            "first_step": float(base[row, column]),
            "seconds": tuple(walls_tried),
        }
        for (row, column), (rungs_tried, kinds, walls_tried) in sorted(tried.items())
    )
    return assemble(
        request.mode,
        chosen,
        members,
        bucket_of,
        calls,
        types.MappingProxyType(seconds),
        rungs=rung,
        retries=retries,
        batches=batches,
        present=present,
    )


__all__ = [
    "BatchedLanes",
    "BucketLanes",
    "CompiledProgram",
    "PreparedProgram",
    "ProgramKey",
    "batched_lanes",
    "evaluate",
    "lane_metadata",
    "placement",
    "prepare",
    "replanned_lanes",
    "restricted_plan",
    "tables",
    "with_own_state",
]
