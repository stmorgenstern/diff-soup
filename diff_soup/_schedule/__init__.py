"""Protocols of operating segments, and the solve that runs them in order.

A model that declares an operational control table
(``spec.control.operational(...)``) closes one algebraic unknown with one of the
rows ``current``, ``voltage``, ``power`` or ``resistance``, chosen by a mode index and
a target. A ``Schedule`` lists ``Segment`` entries, each of which holds one of
those rows at a target for a duration: a charge at constant current, then a
hold at constant voltage, and so on. A target is a constant, or a polynomial in
the time since the segment started.

This module holds the two public types, their lowering to arrays of one static
length against one model's mode names (``schedule_arrays``), and the solve that
runs the segments one after another from the carried state
(``solve_schedule``). Every number of a lowered schedule enters a compiled
program as an argument, so schedules of one length and one target degree share
one program.
"""

from __future__ import annotations

from diff_soup._schedule.records import (
    Schedule,
    ScheduleArrays,
    Segment,
    SegmentEnds,
    schedule_arrays,
)
from diff_soup._schedule.solve import solve_schedule

__all__ = [
    "Schedule",
    "ScheduleArrays",
    "Segment",
    "SegmentEnds",
    "schedule_arrays",
    "solve_schedule",
]
