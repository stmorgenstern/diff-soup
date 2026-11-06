"""Where the lanes of a study read their coordinates in the fitted vector.

A lane is one span of one experiment solved at one point of the study's fitted
vector: the experiment's whole span when it is solved in one shoot, or one of
its windows when it is fitted in windows that start from observed states. It
reads some coordinates of that point, its narrow point: the shared coordinates
first, then the coordinates of its node in the tier between the shared
coordinates and the lanes, then its experiment's own; the lanes of one
experiment read the same coordinates. The plan, the numbers of the lanes and
the tables of the tree are built from this record, not from the problem it is
read from.

A study whose parameters vary between owners has that tier, one node per owner.
A study whose experiments share every coordinate, and a study of one
experiment, have none: there, one node of no coordinates of its own holds every
lane.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True, eq=False)
class LaneLayout:
    """Where every lane reads its coordinates in the fitted vector, and the experiment of each.

    Attributes:
        lanes: The identifier of each lane. The lane of an experiment solved in
            one shoot has the experiment's identifier (``"experiment i"`` when
            the problem names none); window ``k`` of an experiment fitted in
            observed-start windows is
            ``"<experiment> window k"``, or ``"window k"`` when the problem
            holds one experiment. The lanes of one experiment follow each
            other, its windows in time order.
        columns: Per lane, the coordinates of the fitted vector its narrow
            point reads, in the order of that point, the shared ones first.
        targets: Per lane, the position each coordinate of its narrow point
            takes in the lane's own parameter vector.
        widths: Per lane, the length of its own parameter vector, its
            experiment's.
        shared: The shared count: the leading coordinates of the fitted
            vector, which every lane reads first.
        nodes: Per lane, the index of its node in the tier between the shared
            coordinates and the lanes; ``0`` for every lane without that tier.
        labels: The label of each node of that tier; empty without it.
        node_widths: Per node, the width of its block: its own coordinates and
            those of its lanes beyond the shared ones.
        tier_width: The coordinates a node holds itself, before those of its
            lanes (those of the first node); ``0`` without the tier.
        local: Per lane, the position of each of its coordinates beyond the
            shared ones in the block of its node.
        start: The start point, the fitted vector the problem was built with.
        experiments: Per lane, the index of its experiment among the
            problem's experiments, in their order (``joint.problems``, the
            study's order).
    """

    lanes: tuple
    columns: tuple
    targets: tuple
    widths: tuple
    shared: int
    nodes: tuple
    labels: tuple
    node_widths: tuple
    tier_width: int
    local: tuple
    start: np.ndarray
    experiments: tuple


def _experiment_problems(problem: Any) -> tuple:
    """The fitted problem of every experiment: a joint problem's, or the one experiment's."""
    return tuple(problem.problems) if hasattr(problem, "problems") else (problem,)


def _window_count(child: Any) -> int:
    """The observed-start windows of one experiment's fitted problem; ``0`` when it has none."""
    from diff_soup.inference.fitting.joint import _child_mode

    return int(child.n_segments) if _child_mode(child) == "observed_ic" else 0


def _lane_ids(names: tuple, problems: tuple) -> tuple:
    """``(lanes, experiments)``: the identifier of every lane and the index of its experiment.

    ``names`` are the identifiers of the experiments of ``problems``. An
    experiment solved in one shoot has one lane, of its identifier; one fitted
    in observed-start windows has one lane per window, in time order, named as
    [LaneLayout][diff_soup.inference.engine._layout.LaneLayout] says.
    """
    lanes, experiments = [], []
    for index, (name, child) in enumerate(zip(names, problems)):
        windows = _window_count(child)
        prefix = "" if len(problems) == 1 else f"{name} "
        lanes += [f"{prefix}window {window}" for window in range(windows)] or [name]
        experiments += [index] * max(windows, 1)
    return tuple(lanes), tuple(experiments)


def lane_problems(problem: Any) -> tuple:
    """The fitted problem of every lane: its experiment's, once per lane of that experiment."""
    return tuple(
        child
        for child in _experiment_problems(problem)
        for _ in range(max(_window_count(child), 1))
    )


def lane_layout(problem: Any) -> LaneLayout:
    """The lane layout of a joint problem, or of the fitted problem of one experiment.

    Reads the layout of the problem; compiles and solves nothing.

    Args:
        problem: A joint problem, whose experiments hold the lanes in their
            order, or the fitted problem of one experiment, whose lanes read
            every coordinate of the fitted vector as a shared one.

    Returns:
        The [LaneLayout][diff_soup.inference.engine._layout.LaneLayout].
    """
    start = np.asarray(problem.x0)
    every = _experiment_problems(problem)
    if not hasattr(problem, "problems"):
        everything = np.arange(start.size, dtype=np.int64)
        lanes, experiments = _lane_ids(("experiment 0",), every)
        return LaneLayout(
            lanes=lanes,
            columns=(everything,) * len(lanes),
            targets=(everything,) * len(lanes),
            widths=(int(start.size),) * len(lanes),
            shared=int(start.size),
            nodes=(0,) * len(lanes),
            labels=(),
            node_widths=(0,),
            tier_width=0,
            local=(np.zeros((0,), dtype=np.int64),) * len(lanes),
            start=start,
            experiments=experiments,
        )
    count = len(problem.problems)
    names = tuple(getattr(problem, "experiment_ids", None) or ())
    lanes = tuple(
        names[index] if len(names) == count else f"experiment {index}" for index in range(count)
    )
    layout = problem.ownership_layout
    shared = int(layout.n_shared)
    columns, targets, widths, nodes, local = ([None] * count for _ in range(5))
    if getattr(problem, "owner_ids", ()):
        plans = problem.owner_point_plans
        for owner in plans:
            indices = np.asarray(owner.indices, dtype=np.int64)
            for plan in owner.child_plans:
                child = int(plan.child_index)
                columns[child] = indices[np.asarray(plan.source_positions, dtype=np.int64)]
                targets[child] = np.asarray(plan.target_positions, dtype=np.int64)
                widths[child] = int(plan.child_width)
                nodes[child] = int(owner.owner_index)
                local[child] = np.asarray(plan.local_positions, dtype=np.int64)
        labels = tuple(problem.owner_ids)
        node_widths = tuple(int(owner.local_width) for owner in plans)
        first, stop = layout.owner_slices[0]
        tier_width = int(stop - first)
    else:
        # One node of no coordinates of its own holds every lane and their own coordinates.
        offset = 0
        for child in range(count):
            local_start, local_stop = layout.local_slices[child]
            state_start, state_stop = layout.state_slices[child]
            own = tuple(range(local_start, local_stop)) + tuple(range(state_start, state_stop))
            theta = len(layout.shared_cols[child]) + len(layout.local_cols[child])
            columns[child] = np.asarray(tuple(range(shared)) + own, dtype=np.int64)
            targets[child] = np.asarray(
                tuple(layout.shared_cols[child])
                + tuple(layout.local_cols[child])
                + tuple(range(theta, theta + state_stop - state_start)),
                dtype=np.int64,
            )
            widths[child] = int(np.asarray(problem.problems[child].x0).size)
            nodes[child] = 0
            local[child] = np.arange(offset, offset + len(own), dtype=np.int64)
            offset += len(own)
        labels, node_widths, tier_width = (), (offset,), 0
    # The lanes of one experiment read its coordinates.
    lanes, experiments = _lane_ids(lanes, every)
    return LaneLayout(
        lanes=lanes,
        columns=tuple(columns[index] for index in experiments),
        targets=tuple(targets[index] for index in experiments),
        widths=tuple(widths[index] for index in experiments),
        shared=shared,
        nodes=tuple(nodes[index] for index in experiments),
        labels=labels,
        node_widths=node_widths,
        tier_width=tier_width,
        local=tuple(local[index] for index in experiments),
        start=start,
        experiments=experiments,
    )


__all__ = ["LaneLayout", "lane_layout", "lane_problems"]
