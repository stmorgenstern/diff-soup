"""The per-lane record of a trust-region fit's attempts and iterations."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np

from diff_soup.optimization.inference.hierarchical._lane_route import _Failure, _LaneRoute
from diff_soup.optimization.solvers._loop import (
    _saved_count,
)


def _strings(values: Any) -> np.ndarray:
    """Labels as a host array of strings (no object array)."""
    return np.asarray([str(value) for value in values], dtype=str)


def _string_rows(rows: Any, width: int) -> np.ndarray:
    """Rows of labels as a ``(len(rows), width)`` host array of strings."""
    rows = [[str(value) for value in row] for row in rows]
    return np.asarray(rows, dtype=str).reshape(len(rows), width)


def _padded_rows(rows: Any, fill: Any, dtype: Any) -> np.ndarray:
    """Rows of different lengths as one host array, each row padded with ``fill``."""
    rows = [list(row) for row in rows]
    width = max((len(row) for row in rows), default=0)
    padded = [row + [fill] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=dtype).reshape(len(rows), width)


def _unpadded(row: Any) -> tuple:
    """A row of rungs written by ``_padded_rows`` with the fill -1, without its padding."""
    return tuple(int(value) for value in np.ravel(row) if int(value) >= 0)


# The nested mode's counts kept per member of the cut level.
_MEMBER_COUNTS = (
    "inner_iterations",
    "inner_rejections",
    "inner_unconverged",
    "inner_start_iterations",
    "inner_retreats",
)


def _hold_rungs(rungs: np.ndarray, first_steps: np.ndarray, results: Any, position: dict) -> None:
    """Raise the rung of every (member, lane) that ``results`` solved higher; never lower one.

    ``rungs`` and ``first_steps`` are ``(members, lanes)``, changed in place;
    a row is a population member, ``position`` gives a lane's column. A
    raised rung's base is the first step of the rung-0 solve its rungs
    scaled (its retry record's ``first_step``); a unit that solved at rung 0
    while its lane holds rung 0 keeps the first step its solve took. Results
    that report no rungs leave both as they are.
    """
    for record in getattr(results, "retries", ()):
        member = int(record["member"])
        column = position[str(record["lane"])]
        tried = tuple(int(rung) for rung in record["rungs"])
        if tuple(record["kinds"])[-1] == "ok" and tried[-1] > rungs[member, column]:
            rungs[member, column] = tried[-1]
            first_steps[member, column] = float(record["first_step"])
    status = getattr(results, "status", None)
    first = getattr(getattr(results, "work", None), "first_step", None)
    if status is None or first is None:
        return
    members = np.asarray(results.members, dtype=np.int64)
    columns = np.asarray([position[str(lane)] for lane in results.lane_ids], dtype=np.int64)
    kinds = np.asarray(status.kind)
    produced = np.asarray(status.rung)
    first = np.asarray(first, dtype=np.float64)
    keep = (kinds == 0) & (produced == 0) & (rungs[np.ix_(members, columns)] == 0)
    row, column = np.nonzero(keep)
    first_steps[members[row], columns[column]] = first[row, column]


class _Work:
    """The counts, the failure log and the rungs of one fit, reported as ``OptResult.work``.

    Every lane holds a rung, 0 at the start of a fresh fit: a lane that a
    request solved at a higher rung than it asked for is asked for at that rung
    from then on, with the first step of the rung-0 solve that rung scales as
    its base; a lane that solved at rung 0 keeps the first step its solve took
    as its base. A rung is never lowered.
    """

    COUNTS = (
        "attempts",
        "accepted",
        "rejected",
        "blocks",
        "value",
        "resume_blocks",
        "predicate_rejections",
        "invalid_trials",
        "solves",
    )

    def __init__(self, route: _LaneRoute):
        for name in self.COUNTS:
            setattr(self, name, 0)
        self.route = route
        self.lane_failures = np.zeros((len(route.ids),), dtype=np.int64)
        self.held: tuple = ()
        self.log: list = []
        # The rung each lane is asked for, the first step of its rung-0 solve, the number
        # of evaluations that solved it again, and the consistency re-solves made.
        self.lane_rungs = np.zeros((len(route.ids),), dtype=np.int64)
        self.lane_first_steps = np.zeros((len(route.ids),), dtype=np.float64)
        self.retries = np.zeros((len(route.ids),), dtype=np.int64)
        self.rung_switch_resolves = 0
        # The retry records and the rung switches since the last attempt record.
        self.attempt_retries: list = []
        self.attempt_switches: list = []
        # The nested mode's counts per member of the cut level, and the number of attempts
        # made with the polish switched off; None in the joint mode.
        self.inner_members: Any = None
        self.inner_iterations: Any = None
        self.inner_rejections: Any = None
        self.inner_unconverged: Any = None
        self.inner_start_iterations: Any = None
        self.inner_retreats: Any = None
        self.polish_skipped: Any = None

    def nest(self, members: tuple) -> None:
        """Count the inner iterations of every member of the cut level from here on."""
        self.inner_members = tuple(members)
        for name in _MEMBER_COUNTS:
            setattr(self, name, np.zeros((len(members),), dtype=np.int64))
        self.polish_skipped = 0

    def restore(self, state: Any) -> None:
        """Continue the counts and the failure log of the saved state of the same fit."""
        for name in self.COUNTS:
            setattr(self, name, int(np.asarray(state[f"work.{name}"])))
        saved = dict(
            zip(
                (str(lane) for lane in np.ravel(state["work.lane_ids"])),
                (int(count) for count in np.ravel(state["work.lane_failures"])),
            )
        )
        self.lane_failures = np.asarray(
            [saved.get(lane, 0) for lane in self.route.ids], dtype=np.int64
        )
        attempts = np.ravel(state["work.failure_attempt"])
        # A state saved without the rungs of its failures logs none.
        try:
            tried = [_unpadded(row) for row in np.asarray(state["work.failure_rungs"])]
        except (KeyError, TypeError):
            tried = [()] * attempts.size
        # A state saved without the ignored marks logs every failure as counted.
        try:
            ignored = [bool(value) for value in np.ravel(state["work.failure_ignored"])]
        except (KeyError, TypeError):
            ignored = [False] * attempts.size
        self.log = [
            (
                int(attempt),
                _Failure(
                    str(lane),
                    str(kind),
                    float(reached),
                    tuple(str(node) for node in nodes),
                    rungs,
                    skipped,
                ),
                bool(inner),
            )
            for attempt, lane, kind, reached, nodes, inner, rungs, skipped in zip(
                attempts,
                np.ravel(state["work.failure_lane"]),
                np.ravel(state["work.failure_kind"]),
                np.ravel(state["work.failure_t"]),
                np.asarray(state["work.failure_nodes"]),
                np.ravel(state["work.failure_inner"]),
                tried,
                ignored,
            )
        ]
        # A state saved without the rungs resumes with every lane at rung 0.
        ids = [str(lane) for lane in np.ravel(state["work.lane_ids"])]
        for name, dtype in (
            ("lane_rungs", np.int64),
            ("lane_first_steps", np.float64),
            ("retries", np.int64),
        ):
            try:
                values = dict(zip(ids, np.ravel(np.asarray(state[f"work.{name}"]))))
            except (KeyError, TypeError):
                values = {}
            setattr(
                self,
                name,
                np.asarray([values.get(lane, 0) for lane in self.route.ids], dtype=dtype),
            )
        self.rung_switch_resolves = _saved_count(state, "work.rung_switch_resolves")
        if self.inner_members is not None:
            for name in ("inner_iterations", "inner_rejections", "inner_unconverged"):
                setattr(self, name, np.array(state[f"work.{name}"], dtype=np.int64))
            # A state saved without the start's iterations, the retreats or the skipped
            # polishes resumes with none.
            for name in ("inner_start_iterations", "inner_retreats"):
                try:
                    saved = np.array(state[f"work.{name}"], dtype=np.int64)
                except (KeyError, TypeError):
                    saved = np.zeros((len(self.inner_members),), dtype=np.int64)
                setattr(self, name, saved)
            self.polish_skipped = _saved_count(state, "work.polish_skipped")

    def record(self, attempt: int, failures: tuple, *, inner: bool) -> None:
        """Log failures the run survives; a failure marked ``ignored`` is logged as such."""
        for failure in failures:
            self.lane_failures[self.route.position[failure.lane]] += 1
            self.log.append((attempt, failure, inner))

    def attempted(self, accepted: bool) -> None:
        """Count an attempt, its trial accepted or rejected."""
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        self.attempts += 1

    def request(self, lanes: Any) -> dict:
        """The keywords of a request for ``lanes`` (``None``: every lane) at the rungs held.

        A request whose lanes all hold rung 0 names no rung.
        """
        route = self.route
        keywords: dict = {}
        if lanes is None:
            positions = np.arange(len(route.ids))
        else:
            positions = np.asarray([route.position[lane] for lane in lanes], dtype=np.int64)
            keywords["lanes"] = tuple(lanes)
        rungs = self.lane_rungs[positions]
        if np.any(rungs > 0):
            keywords["rungs"] = rungs[None, :].copy()
            keywords["first_steps"] = self.lane_first_steps[positions][None, :].copy()
        return keywords

    def hold(self, results: Any) -> None:
        """Hold the rungs ``results`` raised (``_hold_rungs``, the fit's one member 0).

        Every retry record counts for its lane and joins the records of the
        attempt.
        """
        for record in getattr(results, "retries", ()):
            self.retries[self.route.position[str(record["lane"])]] += 1
            self.attempt_retries.append(record)
        _hold_rungs(
            self.lane_rungs[None, :], self.lane_first_steps[None, :], results, self.route.position
        )

    def attempt_records(self) -> dict:
        """The retry records and rung switches since the last attempt record, then none."""
        records = {
            "retries": tuple(self.attempt_retries),
            "switches": tuple(self.attempt_switches),
        }
        self.attempt_retries, self.attempt_switches = [], []
        return records

    def mapping(self) -> MappingProxyType:
        """The work so far as a read-only mapping of host arrays."""
        route = self.route
        entries = {name: np.asarray(getattr(self, name), dtype=np.int64) for name in self.COUNTS}
        entries["lane_ids"] = _strings(route.ids)
        entries["lane_failures"] = self.lane_failures.copy()
        entries["held"] = _string_rows(
            [(route.names[level - 1], route.members[level - 1][node]) for level, node in self.held],
            2,
        )
        entries["failure_attempt"] = np.asarray([entry[0] for entry in self.log], dtype=np.int64)
        entries["failure_lane"] = _strings(entry[1].lane for entry in self.log)
        entries["failure_kind"] = _strings(entry[1].kind for entry in self.log)
        entries["failure_t"] = np.asarray(
            [entry[1].t_reached for entry in self.log], dtype=np.float64
        )
        entries["failure_nodes"] = _string_rows([entry[1].nodes for entry in self.log], route.depth)
        entries["failure_inner"] = np.asarray([entry[2] for entry in self.log], dtype=bool)
        entries["failure_ignored"] = np.asarray(
            [entry[1].ignored for entry in self.log], dtype=bool
        )
        entries["failure_rungs"] = _padded_rows(
            [entry[1].rungs for entry in self.log], -1, np.int64
        )
        entries["lane_rungs"] = self.lane_rungs.copy()
        entries["lane_first_steps"] = self.lane_first_steps.copy()
        entries["retries"] = self.retries.copy()
        entries["rung_switch_resolves"] = np.asarray(self.rung_switch_resolves, dtype=np.int64)
        if self.inner_members is not None:
            entries["inner_members"] = _strings(self.inner_members)
            for name in _MEMBER_COUNTS:
                entries[name] = getattr(self, name).copy()
            entries["polish_skipped"] = np.asarray(self.polish_skipped, dtype=np.int64)
        return MappingProxyType(entries)


def _record(
    rho: float,
    lam: float,
    accepted: bool,
    trial: np.ndarray,
    failures: tuple,
    depth: int,
    *,
    retries: tuple = (),
    switches: tuple = (),
):
    """The attempt just made, as the ``record.*`` entries of the state.

    ``retries`` are the engine's records of the lanes its requests solved
    again; ``switches`` the consistency re-solves, one row per lane: the lane,
    its old and new rung, and its cost at the current point before and after.
    """
    return {
        "rho": np.asarray(rho, dtype=np.float64),
        "lam": np.asarray(lam, dtype=np.float64),
        "accepted": np.asarray(bool(accepted)),
        "trial": np.array(trial, dtype=np.float64),
        "failed_lanes": _strings(failure.lane for failure in failures),
        "failed_kinds": _strings(failure.kind for failure in failures),
        "failed_nodes": _string_rows([failure.nodes for failure in failures], depth),
        "rungs_tried": _padded_rows([failure.rungs for failure in failures], -1, np.int64),
        "rung_switches": _string_rows(
            [
                (lane, old, new, repr(float(before)), repr(float(after)))
                for lane, old, new, before, after in switches
            ],
            5,
        ),
        "retry_lanes": _strings(record["lane"] for record in retries),
        "retry_rungs": _padded_rows([record["rungs"] for record in retries], -1, np.int64),
        "retry_kinds": _padded_rows([record["kinds"] for record in retries], "", str),
        "retry_first_steps": np.asarray(
            [record["first_step"] for record in retries], dtype=np.float64
        ),
        "retry_seconds": _padded_rows(
            [record["seconds"] for record in retries], np.nan, np.float64
        ),
    }
