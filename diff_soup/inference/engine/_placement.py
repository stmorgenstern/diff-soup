"""Which process evaluates which lanes: the placement of a study's lanes over the processes of a run.

A run of several processes (one per GPU, launched as one MPI world) splits the
lanes of a lowered study among its processes before anything is compiled.
[place][diff_soup.inference.engine._placement.place] is a pure function of the
plan, the number of processes and the cost of every lane; the same inputs give
the same placement in every process and in every run. Its rules, in order:

1. every lane of one experiment (the windows of an experiment fitted in
   windows) goes to one process;
2. the lanes of one bucket go to one process when the balance allows it: when
   the most loaded process then carries at most ``1 + bucket_slack`` times
   its load without this rule (``BUCKET_SLACK``, 0.1, by default), and at
   most the mean load plus the costliest experiment. Buckets are tried in the
   plan's order, each kept whole when the bound holds with it and the
   buckets kept before it. Every process that holds a lane of a bucket
   compiles the bucket's program and runs it at its full width, the lanes it
   does not hold filled with copies, so a bucket split over two processes
   costs one call of its program more than on one process; the costs of the
   lanes, counted once each, do not see that call;
3. the experiments of one member of the top level of the parameter tree go
   to the process that already holds that member's experiments when that does
   not make the most loaded process more loaded than the balance rule would;
4. balance: the experiments (and the buckets kept whole) are placed longest
   first, each on the least loaded process; process 0 coordinates the run and
   evaluates lanes like every other process, with no discount;
5. ties are broken by the lanes' order in the study, then by the lower process;
6. the lanes of a bucket that rule 2 does not keep whole may sit on several
   processes: a bucket is what one program is compiled for, not what one
   process holds.

A lane's cost is modelled as its state size times its frame count, or taken
from the seconds the lanes measured in an earlier run (the ledger a checkpoint
holds). The placement is recorded in the run's report, never in a checkpoint:
a run resumed on another number of processes places its lanes afresh.

This is the process placement, one of the two data placements of the lane
engine that a [Layout][diff_soup.Layout] selects, one at a time. The other,
the device placement, runs inside one process: a layout whose mesh has a
``data`` axis of several devices splits every call of every program over
those devices (the plan's ``mesh``), and
[deal][diff_soup.inference.engine._balance.deal] hands each call's units to
them by their last step counts. It needs no record of its own: every lane is
evaluated in the one process, and the plan says which devices share its calls.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

BUCKET_SLACK: float = 0.1
"""How much more the most loaded process may carry so that a bucket's lanes stay on one process.

A bucket whose lanes belong to several experiments goes whole to one process
when the most loaded process then carries at most ``1 + BUCKET_SLACK`` times
its load without that rule, and at most the mean load plus the costliest
experiment. ``place(bucket_slack=)`` takes another fraction; 0 keeps a bucket
whole only when that leaves the most loaded process no more loaded.
"""


@dataclasses.dataclass(frozen=True, eq=False)
class Placement:
    """The process of every lane of a study, and what it was computed from.

    This is the process placement of a run of several processes; a study
    placed over the devices of a mesh in one process has none.

    Attributes:
        world_size: The number of processes.
        rank_of: Lane id to the process that evaluates it.
        per_rank: Per process, ``((bucket index, lane ids), ...)``: the
            buckets it holds lanes of, in the plan's order, each with its lanes
            in the bucket's order.
        loads: Per process, the summed cost of its lanes.
        costs: Lane id to the cost the placement balanced.
        source: ``"modelled"`` when every cost is the model's,
            ``"measured"`` when every cost is a measured one, ``"mixed"``
            when the model fills the lanes without a measurement, scaled to
            the measured ones.
        whole_buckets: The indices of the buckets whose lanes belong to
            several experiments and sit on one process because the balance
            allowed it (the bucket rule of
            [place][diff_soup.inference.engine._placement.place]), ascending.
    """

    world_size: int
    rank_of: Mapping
    per_rank: tuple
    loads: tuple
    costs: Mapping
    source: str
    whole_buckets: tuple = ()

    def lanes_of(self, rank: int) -> tuple:
        """The lanes process ``rank`` evaluates, in the plan's bucket order."""
        return tuple(lane for _, lanes in self.per_rank[int(rank)] for lane in lanes)

    def buckets_of(self, rank: int) -> tuple:
        """The bucket indices process ``rank`` holds lanes of, ascending."""
        return tuple(bucket for bucket, _ in self.per_rank[int(rank)])

    def compiling_rank(self, bucket: int) -> int:
        """The lowest process that holds a lane of ``bucket``: the one that writes its programs."""
        for rank, entries in enumerate(self.per_rank):
            if any(index == int(bucket) for index, _ in entries):
                return rank
        raise ValueError(f"no process holds a lane of bucket {bucket}.")

    def __str__(self) -> str:
        kept = len(self.whole_buckets)
        whole = f", {kept} bucket(s) kept on one process" if kept else ""
        lines = [f"placement over {self.world_size} process(es), costs {self.source}{whole}:"]
        for rank, entries in enumerate(self.per_rank):
            count = sum(len(lanes) for _, lanes in entries)
            buckets = ", ".join(str(index) for index, _ in entries) or "none"
            lines.append(
                f"  rank {rank}: {count} lane(s), load {self.loads[rank]:.6g}, buckets {buckets}"
            )
        return "\n".join(lines)


def _world_size(world_size: Any) -> int:
    if isinstance(world_size, bool) or not isinstance(world_size, (int, np.integer)):
        raise ValueError(f"world_size= is a positive whole number; got {world_size!r}.")
    if int(world_size) < 1:
        raise ValueError(f"world_size= is a positive whole number; got {world_size!r}.")
    return int(world_size)


_WINDOW = re.compile(r"^(?:(?P<experiment>.*) )?window \d+$")


def experiment_of(lane: Any) -> Any:
    """The experiment a lane belongs to, read from its id.

    An experiment solved in one shoot has one lane of its own id; one fitted in
    observed-start windows has one lane per window, named ``"<experiment>
    window k"`` (``"window k"`` in a study of one experiment), as
    [LaneLayout][diff_soup.inference.engine._layout.LaneLayout] names them.
    """
    match = _WINDOW.match(str(lane)) if isinstance(lane, str) else None
    if match is None:
        return lane
    return match.group("experiment") or ""


def plan_costs(plan: Any) -> dict:
    """Lane id to the cost the plan alone models: the frame slots of the lane's bucket.

    The plan knows the frames of every bucket's class and not the size of a
    lane's state; a lowered study places by ``modelled_costs``, state size
    times the lane's own frames.
    """
    costs = {}
    for bucket in plan.buckets:
        slots = float(max(int(sum(bucket.frames.streams)), 1))
        for lane in bucket.lanes:
            costs[lane] = slots
    return costs


def _cost_table(plan: Any, lanes: tuple, costs: Any, measured: Any) -> tuple:
    """Per lane, the cost to balance, and the source of those costs."""
    if costs is None:
        costs = plan_costs(plan)
    if not isinstance(costs, Mapping):
        raise ValueError(
            "costs= maps every lane id of the plan to its modelled cost, a positive number; got "
            f"{type(costs).__name__}."
        )
    model = {}
    for lane in lanes:
        if lane not in costs:
            raise ValueError(
                f"costs= gives no cost for lane {lane!r}; give one positive number per lane."
            )
        value = float(costs[lane])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"costs= gives lane {lane!r} the cost {value!r}; a cost is a positive finite "
                "number."
            )
        model[lane] = value
    if measured is None:
        return model, "modelled"
    if not isinstance(measured, Mapping):
        measured = getattr(measured, "seconds_per_evaluation", None)
        if callable(measured):
            measured = measured()
    if not isinstance(measured, Mapping):
        raise ValueError(
            "measured= maps lane ids to the seconds one evaluation of the lane took, or is a "
            "Ledger; got another object."
        )
    seen = {
        lane: float(measured[lane])
        for lane in lanes
        if lane in measured and np.isfinite(float(measured[lane])) and float(measured[lane]) > 0
    }
    if not seen:
        return model, "modelled"
    if len(seen) == len(lanes):
        return seen, "measured"
    # The model fills the lanes without a measurement, scaled to the measured ones.
    scale = sum(seen.values()) / sum(model[lane] for lane in seen)
    return {lane: seen.get(lane, model[lane] * scale) for lane in lanes}, "mixed"


def _slack(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"bucket_slack= is a non-negative finite number, a fraction of the most loaded "
            f"process's load; got {value!r}."
        )
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(
            f"bucket_slack= is a non-negative finite number, a fraction of the most loaded "
            f"process's load; got {value!r}."
        )
    return float(value)


def _assign(units: list, table: Mapping, order: Mapping, members: Mapping, size: int) -> tuple:
    """Every unit (lanes kept together) on a process: longest first onto the least loaded.

    A unit whose first lane's member already has a process goes there when
    that leaves the most loaded process no more loaded than the least loaded
    choice would. Returns ``(rank_of, loads)``.
    """
    ranked = [(sum(table[lane] for lane in unit), order[unit[0]], unit) for unit in units]
    # Longest first; ties by the first lane's place in the study.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    loads = [0.0] * size
    home: dict = {}
    rank_of: dict = {}
    for cost, _, entry in ranked:
        least = min(range(size), key=lambda rank: (loads[rank], rank))
        chosen = least
        member = members.get(entry[0])
        if member is not None and member in home:
            held = home[member]
            # The member's own process, when it leaves the most loaded process no more loaded
            # than the least loaded choice would.
            if max(max(loads), loads[held] + cost) <= max(max(loads), loads[least] + cost):
                chosen = held
        loads[chosen] += cost
        if member is not None:
            home.setdefault(member, chosen)
        for lane in entry:
            rank_of[lane] = chosen
    return rank_of, loads


def _joined(blocks: list, block_of: Mapping, buckets: tuple, order: Mapping) -> list:
    """The units placed: the experiments, those sharing a bucket of ``buckets`` joined into one."""
    parent = list(range(len(blocks)))

    def root(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    for bucket in buckets:
        roots = sorted({root(block_of[lane]) for lane in bucket.lanes})
        for other in roots[1:]:
            parent[other] = roots[0]
    units: dict = {}
    for index, block in enumerate(blocks):
        units.setdefault(root(index), []).extend(block)
    return [tuple(sorted(unit, key=order.__getitem__)) for unit in units.values()]


def place(
    plan: Any,
    world_size: int,
    costs: Mapping | None = None,
    measured: Any = None,
    *,
    experiments: Mapping | None = None,
    members: Mapping | None = None,
    bucket_slack: float = BUCKET_SLACK,
) -> Placement:
    """Place the lanes of a plan on ``world_size`` processes.

    Args:
        plan: The [Plan][diff_soup.inference.engine.Plan] of the study.
        world_size: The number of processes, a positive whole number.
        costs: Lane id to its modelled cost, a positive number for every
            planned lane (``modelled_costs`` gives state size times frames);
            ``None`` models each lane by the frame slots of its bucket
            (``plan_costs``), all the plan alone knows.
        measured: ``None``, a mapping of lane id to the seconds one
            evaluation of the lane took in an earlier run, or a ledger that
            gives that mapping (``Ledger.seconds_per_evaluation()``). A lane
            with a measurement is balanced by it; the others by the model,
            scaled to the measured lanes.
        experiments: Lane id to the experiment it belongs to; the lanes of one
            experiment go to one process. ``None`` reads the experiment from
            the lane's id (``experiment_of``: a window ``"<experiment> window
            k"`` belongs to ``<experiment>``).
        members: Lane id to its member of the top level of the parameter tree;
            the experiments of one member go to one process when that costs
            nothing. ``None`` keeps no such affinity.
        bucket_slack: The bucket rule's bound, a non-negative fraction: the
            lanes of a bucket go to one process when the most loaded process
            then carries at most ``1 + bucket_slack`` times its load without
            the rule, and at most the mean load plus the costliest
            experiment. ``BUCKET_SLACK`` (0.1) by default; 0 keeps a bucket
            whole only when that leaves the most loaded process no more
            loaded.

    Returns:
        The [Placement][diff_soup.inference.engine._placement.Placement].

    Raises:
        ValueError: If ``world_size`` is not a positive whole number, a cost
            or measurement is missing or not a positive finite number, or
            ``bucket_slack`` is not a non-negative finite number.
    """
    size = _world_size(world_size)
    slack = _slack(bucket_slack)
    order = {}
    for bucket in plan.buckets:
        for lane, child in zip(bucket.lanes, bucket.children):
            order[lane] = int(child)
    lanes = tuple(sorted(order, key=order.__getitem__))
    table, source = _cost_table(plan, lanes, costs, measured)
    experiments = {} if experiments is None else dict(experiments)
    members = {} if members is None else dict(members)

    # Every experiment's lanes together, in study order.
    groups: dict = {}
    for lane in lanes:
        key = experiments[lane] if lane in experiments else ("experiment", experiment_of(lane))
        groups.setdefault(key, []).append(lane)
    blocks = [tuple(entry) for entry in groups.values()]
    rank_of, loads = _assign(blocks, table, order, members, size)

    # A bucket of several experiments whole on one process, while the bound holds.
    whole: list = []
    if size > 1 and blocks:
        block_of = {lane: index for index, block in enumerate(blocks) for lane in block}
        largest = max(sum(table[lane] for lane in block) for block in blocks)
        bound = min((1.0 + slack) * max(loads), sum(loads) / size + largest)
        units = blocks
        for bucket in plan.buckets:
            if len({block_of[lane] for lane in bucket.lanes}) < 2:
                continue
            trial = _joined(blocks, block_of, (*whole, bucket), order)
            if len(trial) == len(units):
                whole.append(bucket)  # already whole through the buckets kept before it
                continue
            trial_rank_of, trial_loads = _assign(trial, table, order, members, size)
            if max(trial_loads) <= bound:
                whole.append(bucket)
                units, rank_of, loads = trial, trial_rank_of, trial_loads

    per_rank = []
    for rank in range(size):
        entries = []
        for bucket in plan.buckets:
            mine = tuple(lane for lane in bucket.lanes if rank_of.get(lane) == rank)
            if mine:
                entries.append((int(bucket.index), mine))
        per_rank.append(tuple(entries))
    return Placement(
        world_size=size,
        rank_of=dict(rank_of),
        per_rank=tuple(per_rank),
        loads=tuple(float(load) for load in loads),
        costs=dict(table),
        source=source,
        whole_buckets=tuple(sorted(int(bucket.index) for bucket in whole)),
    )


def modelled_costs(lanes: Any) -> dict:
    """Lane id to its modelled cost: the size of its state times the frames it records.

    Args:
        lanes: The study's [BatchedLanes][diff_soup.inference.engine._batched.BatchedLanes]
            (host numbers, or the numbers of the plan alone), whose buckets
            give every lane's frames, and its initial state's size through
            ``state_sizes``.
    """
    costs = {}
    for entry in lanes.buckets:
        for position, lane in enumerate(entry.bucket.lanes):
            frames = int(sum(entry.frames[position].streams))
            size = int(entry.state_sizes[position]) if entry.state_sizes else 1
            costs[lane] = float(max(size, 1) * max(frames, 1))
    return costs


__all__ = ["BUCKET_SLACK", "Placement", "experiment_of", "modelled_costs", "place", "plan_costs"]
