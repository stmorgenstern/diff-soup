"""A request to the batched lanes: which units to evaluate, in which mode, at which points.

A unit is one pair (population member, lane). A request names a mode of the
lane kernel, the joint parameter vector of every member, which members take
part and which lanes run. Every requested unit returns exactly one result, in
canonical order: members ascending, then lanes by experiment index in the
study, whatever order the request names them in.

A request may also name the rung each unit runs at. A unit at rung 0 starts
its solve from its lane's own first step; a unit at rung ``r`` starts from
``retry_first_step_factors[r - 1]`` times its rung-0 first step, which the
request then gives. A fit that solved a lane at a higher rung asks for that
rung again; the batched lanes keep no rung from one request to the next.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import numpy as np

from diff_soup.inference.engine._kernel import MODES


def _ids(value: Any, name: str) -> tuple | None:
    """``value`` as a tuple of experiment ids without repeats, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name}= is a tuple of experiment ids; got the single string {value!r}. Write "
            f"{name}=({value!r},)."
        )
    try:
        ids = tuple(value)
    except TypeError:
        raise ValueError(f"{name}= is a tuple of experiment ids; got {value!r}.") from None
    repeated = sorted(str(lane) for lane, count in collections.Counter(ids).items() if count > 1)
    if repeated:
        raise ValueError(
            f"{name}= names the experiments {repeated} more than once; name each experiment once."
        )
    return ids


def _rungs(rungs: Any, first_steps: Any, members: int) -> tuple:
    """``rungs`` and ``first_steps`` as read-only arrays, checked against each other."""
    if rungs is not None:
        given = rungs
        rungs = np.array(rungs, copy=True)
        if rungs.dtype == bool or not np.issubdtype(rungs.dtype, np.integer):
            raise ValueError(
                f"rungs= gives each unit the rung it runs at, whole numbers; got an array of "
                f"dtype {rungs.dtype}."
            )
        if rungs.ndim != 2 or rungs.shape[0] != members:
            raise ValueError(
                f"rungs= has one row per member that takes part ({members}) and one column per "
                f"requested lane, shape ({members}, L); got the shape {np.shape(given)}."
            )
        if np.any(rungs < 0):
            raise ValueError(
                f"rungs= gives a unit the rung {int(rungs.min())}; a rung is a whole number "
                "from 0, the solve from the lane's own first step, up to the number of "
                "retry_first_step_factors of the lowered study."
            )
        rungs = rungs.astype(np.int64)
        rungs.setflags(write=False)
    if first_steps is not None:
        given = first_steps
        first_steps = np.array(first_steps, dtype=np.float64, copy=True)
        expected = None if rungs is None else rungs.shape
        if first_steps.ndim != 2 or (expected is not None and first_steps.shape != expected):
            wanted = f"the shape of rungs= {expected}" if expected else "the shape (P, L)"
            raise ValueError(
                f"first_steps= gives each unit the first step its rung-0 solve took, in "
                f"{wanted}; got the shape {np.shape(given)}."
            )
        first_steps.setflags(write=False)
    if rungs is not None and np.any(rungs > 0):
        if first_steps is None:
            raise ValueError(
                "rungs= puts units above rung 0, and rung r starts a unit's solve from "
                "retry_first_step_factors[r - 1] times the first step its rung-0 solve took; "
                "give those first steps as first_steps=, of the shape of rungs= "
                "(Results.work.first_step, or the first_step of the unit's entry in "
                "Results.retries)."
            )
        above = rungs > 0
        bad = above & ~(np.isfinite(first_steps) & (first_steps != 0.0))
        if np.any(bad):
            row, column = (int(index) for index in np.argwhere(bad)[0])
            raise ValueError(
                f"first_steps= gives the unit at row {row}, column {column}, which runs at rung "
                f"{int(rungs[row, column])}, the first step {float(first_steps[row, column])!r}; "
                "a unit above rung 0 needs the finite, nonzero first step of its rung-0 solve."
            )
    return rungs, first_steps


@dataclasses.dataclass(frozen=True, eq=False)
class Request:
    """The units one evaluation of the batched lanes returns.

    Attributes:
        mode: One of [MODES][diff_soup.inference.engine._kernel.MODES]:
            ``value``, ``blocks``, ``gradient``, ``observe`` or ``hessian``.
        points: The joint parameter vector, shape ``(n,)``, or one per
            population member, shape ``(P, n)``. Stored as a read-only numpy
            array; a single vector is stored as one member, ``(1, n)``.
        active: Per population member, whether it takes part, shape ``(P,)``;
            every member when ``None``. Stored as a read-only numpy array.
        lanes: The experiment ids of the lanes to run, in any order; every
            planned lane when ``None``.
        order: Experiment ids whose lanes are evaluated first, most urgent
            first. Every id must be among the requested lanes. The order
            changes when a lane runs, never where its result appears.
        directions: Tangent directions of a product mode. The batched lanes
            evaluate no product mode, so this must be ``None``.
        rungs: Per requested unit, the rung it runs at: whole numbers of
            shape ``(P, L)``, one row per member that takes part and one
            column per requested lane, both in canonical order. Rung 0
            starts the solve from the lane's own first step, rung ``r`` from
            ``retry_first_step_factors[r - 1]`` times the unit's rung-0 first
            step; a rung lies in ``0 .. len(retry_first_step_factors)`` of the
            lowered study. ``None`` runs every unit at rung 0. A unit that
            fails with a kind of ``retry_kinds`` still climbs from its rung.
            Stored as a read-only numpy array.
        first_steps: Per requested unit, the first step its rung-0 solve took
            (``Results.work.first_step``, or the ``first_step`` of its entry
            in ``Results.retries``), shape ``(P, L)`` as ``rungs``. Required
            when a rung is above 0, and then finite and nonzero at every unit
            above rung 0; units at rung 0 ignore it. Stored as a read-only
            numpy array.
        units: Which units are evaluated: booleans of shape ``(P, L)``, one
            row per population member (the rows of ``points``) or one row per
            member that takes part (as ``rungs``), and one column per
            requested lane in canonical order. ``None`` evaluates every unit of
            the members that take part and the requested lanes. A unit left
            out is not solved: its result has the status kind ``absent``, zero
            cost, rows, derivatives and work, and lies outside every sum. A
            reduction that would give such a unit a nonzero weight refuses it.
            Stored as a read-only numpy array.

    Raises:
        ValueError: If ``directions`` is given, the mode is not one of
            ``MODES``, ``points`` is not a vector or a matrix of real
            numbers, ``active`` does not give one flag per member, ``lanes``
            or ``order`` repeat an experiment, ``rungs`` is not a matrix of
            whole numbers of at least 0 with one row per member that takes
            part, or ``first_steps`` is missing where a rung is above 0, is
            not of the shape of ``rungs``, or is not finite and nonzero at a
            unit above rung 0, or ``units`` is not a matrix of booleans.
    """

    mode: str
    points: Any
    active: Any = None
    lanes: Any = None
    order: Any = None
    directions: Any = None
    rungs: Any = None
    first_steps: Any = None
    units: Any = None

    def __post_init__(self) -> None:
        """Check the request on its own; the study checks the rest when it evaluates it."""
        if self.directions is not None:
            raise ValueError(
                "directions= gives the tangent directions of a product mode, and the batched "
                f"lanes evaluate the modes {MODES} only; leave directions= as None."
            )
        if self.mode not in MODES:
            raise ValueError(f"a request's mode is one of {MODES}; got {self.mode!r}.")
        points = np.array(self.points, copy=True)
        if not (
            np.issubdtype(points.dtype, np.floating) or np.issubdtype(points.dtype, np.integer)
        ):
            raise ValueError(f"points= holds real numbers; got an array of dtype {points.dtype}.")
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2:
            raise ValueError(
                f"points= has the shape (n,) of one joint parameter vector or (P, n) of one per "
                f"population member; got the shape {np.shape(self.points)}."
            )
        points.setflags(write=False)
        object.__setattr__(self, "points", points)
        if self.active is not None:
            active = np.array(self.active, copy=True)
            if active.dtype != bool or active.shape != (points.shape[0],):
                raise ValueError(
                    f"active= flags which of the {points.shape[0]} population members take part: "
                    f"booleans of shape ({points.shape[0]},); got {active.dtype} of shape "
                    f"{active.shape}."
                )
            active.setflags(write=False)
            object.__setattr__(self, "active", active)
        object.__setattr__(self, "lanes", _ids(self.lanes, "lanes"))
        object.__setattr__(self, "order", _ids(self.order, "order"))
        rungs, first_steps = _rungs(self.rungs, self.first_steps, int(self.members.shape[0]))
        object.__setattr__(self, "rungs", rungs)
        object.__setattr__(self, "first_steps", first_steps)
        if self.units is not None:
            units = np.array(self.units, copy=True)
            if units.dtype != bool or units.ndim != 2:
                raise ValueError(
                    "units= flags which units (member, lane) are evaluated: booleans of shape "
                    f"(P, L), one row per member and one column per requested lane; got "
                    f"{units.dtype} of shape {units.shape}."
                )
            units.setflags(write=False)
            object.__setattr__(self, "units", units)

    @property
    def members(self) -> np.ndarray:
        """The population members that take part, ascending."""
        if self.active is None:
            return np.arange(self.points.shape[0], dtype=np.int64)
        return np.flatnonzero(self.active).astype(np.int64)

    def unit_mask(self, lanes: int) -> np.ndarray | None:
        """``units`` as one row per member that takes part, checked against ``lanes`` columns.

        Args:
            lanes: The number of requested lanes.

        Returns:
            Booleans of shape ``(members that take part, lanes)``, or ``None``
            when the request evaluates every unit.

        Raises:
            ValueError: If ``units`` has neither one row per population member
                nor one per member that takes part, or not ``lanes`` columns.
        """
        if self.units is None:
            return None
        members = self.members
        rows, columns = self.units.shape
        if columns == lanes and rows == members.shape[0]:
            return self.units
        if columns == lanes and rows == self.points.shape[0]:
            return self.units[members]
        raise ValueError(
            f"units= has the shape {self.units.shape}, and the request has "
            f"{self.points.shape[0]} member(s), {members.shape[0]} of them taking part, and "
            f"{lanes} requested lane(s); give units= of the shape ({self.points.shape[0]}, "
            f"{lanes}) or ({members.shape[0]}, {lanes}), lanes in canonical order."
        )


__all__ = ["Request"]
