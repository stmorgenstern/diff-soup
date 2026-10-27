"""Which units a request asks for, and the order in which they fill the calls of a program.

A unit is one pair (population member, lane). Results always come back in
canonical order: members ascending, then lanes by experiment index in the
study. The order in which units run is a different matter. The ``n`` units of
a bucket are cut into ``k = ceil(n / W)`` calls of its compiled program, each
of ``W`` units; the last call is filled up with copies of its own first unit,
whose results are dropped. Units are dealt to the calls in this order:

1. the lanes the request names in ``order``, in that order;
2. then units whose step count is not known yet, in canonical order;
3. then the other units by the number of trial steps they took when they were
   last evaluated, most first.

The lanes of one call advance in lockstep, so a call lasts as long as its
slowest lane; units of similar step counts in one call waste the least. The
first evaluation knows no step count and runs in canonical order. No call is
made smaller to reduce the spread.

The step counts are kept in a [StepMemory][diff_soup.inference.engine._order.StepMemory]
per series of evaluations: every fit of a lowered study keeps its own, so a
fit's first evaluation runs in canonical order whatever the study evaluated
before, and two fits from the same start order their units alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

UNKNOWN_STEPS: int = -1
"""The step count recorded for a unit that has not been evaluated."""


def canonical_lanes(
    requested: Any, order: Any, *, study_ids: tuple, planned: Mapping, refused: Mapping
) -> tuple:
    """The requested lanes in canonical order, every one checked.

    Args:
        requested: The experiment ids a request names, or ``None`` for every
            planned lane.
        order: The experiment ids the request evaluates first, or ``None``.
        study_ids: Every experiment id of the study, by experiment index.
        planned: The experiment ids that have a lane.
        refused: Experiment id to the reason it has no lane.

    Returns:
        The requested experiment ids, by experiment index.

    Raises:
        ValueError: If an id is not an experiment of the study, names an
            experiment without a lane, or ``order`` names a lane the request
            does not ask for.
    """
    index = {lane: position for position, lane in enumerate(study_ids)}
    if requested is None:
        chosen = tuple(lane for lane in study_ids if lane in planned)
    else:
        chosen = tuple(requested)
    for lane in chosen:
        if lane in refused:
            raise ValueError(
                f"experiment {lane!r} has no lane in the batched backend: {refused[lane]}. "
                "Leave it out of lanes=."
            )
        if lane not in planned:
            known = ", ".join(repr(name) for name in study_ids[:4])
            more = ", ..." if len(study_ids) > 4 else ""
            raise ValueError(
                f"the study has no experiment {lane!r}; lanes= names experiment ids of the "
                f"study ({known}{more})."
            )
    chosen = tuple(sorted(chosen, key=index.__getitem__))
    for lane in order or ():
        if lane not in chosen:
            raise ValueError(
                f"order= names experiment {lane!r}, which the request does not evaluate; "
                "order= names only lanes the request asks for."
            )
    return chosen


def unit_order(priority: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """The order in which the canonical units of one bucket fill the calls of its program.

    Args:
        priority: Per unit in canonical order, the position of its lane in
            the request's ``order``; any value not below the length of
            ``order`` for a lane it does not name.
        steps: Per unit in canonical order, its trial steps when it was last
            evaluated, or ``UNKNOWN_STEPS``.

    Returns:
        The permutation of the canonical units: priority lanes first, then
        units of unknown step count, then the others by descending step count;
        ties in canonical order.
    """
    priority = np.asarray(priority, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    canonical = np.arange(priority.shape[0], dtype=np.int64)
    known = (steps != UNKNOWN_STEPS).astype(np.int64)
    # np.lexsort sorts by its last key first.
    return np.lexsort((canonical, -steps, known, priority))


def cut(count: int, width: int) -> tuple:
    """The calls of a program that evaluate ``count`` units, ``width`` at a time.

    Args:
        count: The units to evaluate, in the order they fill the calls.
        width: ``W``, the units one call evaluates.

    Returns:
        Per call, ``(positions, own)``: the positions of its ``width`` units in
        that order, and ``own``, how many of them are its own units. The first
        ``own`` positions are the call's own units; the rest are copies of its
        first unit, whose results are dropped. Only the last call can have
        ``own < width``.
    """
    if width < 1:
        raise ValueError(f"a call of a program evaluates at least one unit; got the width {width}.")
    calls = []
    for start in range(0, count, width):
        own = np.arange(start, min(start + width, count), dtype=np.int64)
        filler = np.full((width - own.shape[0],), own[0], dtype=np.int64)
        calls.append((np.concatenate([own, filler]), int(own.shape[0])))
    return tuple(calls)


class StepMemory:
    """The trial steps every unit of every bucket took when it was last evaluated.

    Units are keyed by population member and by the lane's position in its
    bucket. A unit never evaluated reports ``UNKNOWN_STEPS``.
    """

    def __init__(self) -> None:
        """Start with no step count known."""
        self._steps: dict[int, np.ndarray] = {}

    def steps(self, bucket: int, members: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """The last step counts of the units ``(members[i], positions[i])`` of ``bucket``."""
        table = self._steps.get(bucket)
        members = np.asarray(members, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        found = np.full(members.shape, UNKNOWN_STEPS, dtype=np.int64)
        if table is None:
            return found
        inside = (members < table.shape[0]) & (positions < table.shape[1])
        found[inside] = table[members[inside], positions[inside]]
        return found

    def record(
        self, bucket: int, lanes: int, members: np.ndarray, positions: np.ndarray, steps: Any
    ) -> None:
        """Keep the step counts of units of ``bucket``, a bucket of ``lanes`` lanes."""
        members = np.asarray(members, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        if not members.size:
            return
        table = self._steps.get(bucket)
        rows = int(members.max()) + 1
        if table is None or table.shape[0] < rows:
            grown = np.full((rows, lanes), UNKNOWN_STEPS, dtype=np.int64)
            if table is not None:
                grown[: table.shape[0]] = table
            table = self._steps[bucket] = grown
        table[members, positions] = np.asarray(steps, dtype=np.int64)


__all__ = ["UNKNOWN_STEPS", "StepMemory", "canonical_lanes", "cut", "unit_order"]
