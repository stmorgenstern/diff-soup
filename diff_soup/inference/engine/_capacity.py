"""Frame classes: the sample slots a compiled program gives each of its lanes.

A compiled program has a fixed number of slots for the union of a lane's
sample times, the time of its initial state included, and per stream a fixed
number of frame slots. A lane that records fewer is padded up to the slots and
its padding is masked, so a padded slot adds no residual row; it still costs
the time of solving through it. Lanes of one identity are therefore cut into
frame classes, so that a lane with few samples does not run in the slots of a
lane with many:

1. sort the lanes by their count of union sample times, largest first;
2. a class opens at the largest remaining count ``m`` and takes every lane
   whose count is at least ``(1 - b) m``, with ``b`` the waste bound (0.5 by
   default): no lane of the class wastes more than that fraction of its slots;
3. the class gives every lane ``m`` union slots and, per stream, the largest
   frame count of the class.

Lanes of different derivative width are never in one class: a program is
compiled for the exact width of its lanes.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np

from diff_soup.inference.engine._layout import _window_count

WASTE_BOUND: float = 0.5
"""The default waste bound ``b`` of a frame class."""


class Frames(NamedTuple):
    """Sample slots: the union of the sample times, and the frames of each stream.

    Attributes:
        union: Slots for the union of the sample times, the time of the
            initial state included.
        streams: Frame slots per stream, in the order of the streams the
            lanes score.
    """

    union: int
    streams: tuple[int, ...]


def lane_frames(child: Any) -> Frames:
    """The sample counts of a lane of one experiment, unpadded.

    Args:
        child: One experiment's fitted problem, an entry of ``joint.problems``.

    Returns:
        The count of union sample times of a lane of the experiment and its
        frame count per stream it scores. A window of an experiment fitted in
        observed-start windows counts the time it starts at and the one frame
        it is scored at.
    """
    if _window_count(child):
        return Frames(union=2, streams=(1,))
    return Frames(
        union=int(np.shape(child._union_ts)[0]),
        streams=tuple(int(np.shape(series.ts)[0]) for series in child.data),
    )


def smallest_frames(lanes: Sequence[Frames]) -> Frames:
    """The smallest slots that hold every one of ``lanes``.

    Args:
        lanes: The sample counts of lanes with the same streams.

    Returns:
        The largest union count and, per stream, the largest frame count.

    Raises:
        ValueError: If ``lanes`` is empty or the lanes have different numbers
            of streams.
    """
    lanes = tuple(lanes)
    if not lanes:
        raise ValueError("the slots of a frame class are taken from at least one lane; got none.")
    counts = {len(lane.streams) for lane in lanes}
    if len(counts) != 1:
        raise ValueError(
            f"the lanes of one frame class record the same streams; they have {sorted(counts)} "
            "streams."
        )
    return Frames(
        union=max(lane.union for lane in lanes),
        streams=tuple(max(column) for column in zip(*(lane.streams for lane in lanes))),
    )


def frame_classes(
    lanes: Sequence[Frames], *, waste_bound: float = WASTE_BOUND
) -> tuple[tuple[Frames, tuple[int, ...]], ...]:
    """Cut lanes of one identity and one derivative width into frame classes.

    Args:
        lanes: The unpadded sample counts of the lanes, from
            [lane_frames][diff_soup.inference.engine._capacity.lane_frames].
        waste_bound: The largest fraction ``b`` of its union slots a lane may
            leave unused; ``0 <= b < 1``.

    Returns:
        One ``(slots, positions)`` pair per class, the class of the largest
        count first: the slots of the class and the ascending positions in
        ``lanes`` of its lanes.

    Raises:
        ValueError: If ``waste_bound`` is outside ``[0, 1)``, or the lanes have
            different numbers of streams.
    """
    lanes = tuple(lanes)
    if not 0.0 <= float(waste_bound) < 1.0:
        raise ValueError(
            f"the waste bound of a frame class is a fraction in [0, 1); got {waste_bound!r}."
        )
    if not lanes:
        return ()
    smallest_frames(lanes)
    # The bound as the decimal it was written as, so that a count equal to (1 - b) m joins.
    keep = 1 - Fraction(repr(float(waste_bound)))
    remaining = sorted(range(len(lanes)), key=lambda position: (-lanes[position].union, position))
    classes = []
    while remaining:
        largest = lanes[remaining[0]].union
        members = [position for position in remaining if lanes[position].union >= keep * largest]
        remaining = [position for position in remaining if lanes[position].union < keep * largest]
        slots = smallest_frames(tuple(lanes[position] for position in members))
        classes.append((slots, tuple(sorted(members))))
    return tuple(classes)


__all__ = [
    "WASTE_BOUND",
    "Frames",
    "frame_classes",
    "lane_frames",
    "smallest_frames",
]
