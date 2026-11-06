"""What every lane of a study cost so far: evaluations, trial steps, seconds and last statuses.

A lowered study records every evaluation's results in its
[Ledger][diff_soup.inference.engine._ledger.Ledger]. A fit's checkpoint saves it
after every attempt, and a later run can balance its lanes over its processes
by the seconds the lanes measured instead of the cost model
([place][diff_soup.inference.engine._placement.place], ``measured=``). Seconds
are the walls of a bucket's calls shared among the bucket's units in
proportion to their trial steps: a measure of cost, reported, never a
measurement to compare against.

When a study's lanes are split over the processes of an MPI world
(``ds.distributed.run``), the ledger is process 0's: process 0 requests every
evaluation and records its results, and only process 0 runs the fit that
saves the ledger in its checkpoint. The other processes evaluate their lanes
and record nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from diff_soup.inference.engine._status import ABSENT

KEEP: int = 8
"""The last statuses the ledger keeps per lane."""

_PREFIX = "ledger."


def attributed_seconds(results: Any) -> np.ndarray:
    """Per unit of ``results``, ``(P, L)``, its share of its bucket's wall seconds.

    A bucket's seconds (``Results.seconds``) are shared among its evaluated
    units in proportion to their trial steps, equally when none took a step;
    a unit not evaluated gets zero.
    """
    if getattr(results, "lane_seconds", None) is not None:
        return np.asarray(results.lane_seconds, dtype=np.float64)
    kinds = np.asarray(results.status.kind)
    shape = kinds.shape
    out = np.zeros(shape, dtype=np.float64)
    walls = dict(results.seconds)
    if not walls:
        return out
    steps = np.asarray(results.work.trial_steps, dtype=np.float64).reshape(shape)
    evaluated = kinds != ABSENT
    buckets = np.asarray(results.bucket, dtype=np.int64)
    # Every column's bucket as a position in ``walls``; a bucket without a wall gets none.
    keys = np.asarray([int(bucket) for bucket in walls], dtype=np.int64)
    wall = np.asarray([float(value) for value in walls.values()], dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    found = np.searchsorted(keys[order], buckets)
    found = np.minimum(found, keys.shape[0] - 1)
    slot = np.where(keys[order][found] == buckets, order[found], -1)
    counted = evaluated & (slot >= 0)[None, :]
    slots = np.broadcast_to(slot[None, :], shape)[counted]
    count = np.zeros(wall.shape, dtype=np.float64)
    total = np.zeros(wall.shape, dtype=np.float64)
    np.add.at(count, slots, 1.0)
    np.add.at(total, slots, steps[counted])
    share = np.where(
        total[slots] > 0.0,
        wall[slots] * steps[counted] / np.where(total[slots] > 0.0, total[slots], 1.0),
        wall[slots] / np.maximum(count[slots], 1.0),
    )
    out[counted] = share
    return out


class Ledger:
    """The lanes' evaluations, trial steps, seconds and last statuses, by lane id.

    Attributes:
        lane_ids: The lanes, in the study's order.
        evaluations: Per lane, the units of it evaluated so far.
        steps: Per lane, the trial steps those evaluations took.
        seconds: Per lane, the seconds attributed to those evaluations.
        kinds: Per lane, the codes of the last ``KEEP`` statuses, oldest
            first, ``-1`` where fewer were recorded.
    """

    def __init__(self, lane_ids: Any):
        """Start an empty ledger for the lanes ``lane_ids``."""
        self.lane_ids = tuple(lane_ids)
        self.position = {lane: index for index, lane in enumerate(self.lane_ids)}
        count = len(self.lane_ids)
        self.evaluations = np.zeros((count,), dtype=np.int64)
        self.steps = np.zeros((count,), dtype=np.int64)
        self.seconds = np.zeros((count,), dtype=np.float64)
        self.kinds = np.full((count, KEEP), -1, dtype=np.int64)

    def record(self, results: Any) -> None:
        """Add the evaluated units of ``results`` to their lanes."""
        kinds = np.asarray(results.status.kind)
        if kinds.ndim != 2:
            return
        steps = np.asarray(results.work.trial_steps, dtype=np.int64).reshape(kinds.shape)
        seconds = attributed_seconds(results)
        index = np.asarray(
            [self.position.get(lane, -1) for lane in results.lane_ids], dtype=np.int64
        )
        evaluated = (kinds != ABSENT) & (index >= 0)[None, :]
        if not np.any(evaluated):
            return
        rows, columns = np.nonzero(evaluated)
        lanes = index[columns]
        np.add.at(self.evaluations, lanes, 1)
        np.add.at(self.steps, lanes, steps[rows, columns])
        np.add.at(self.seconds, lanes, seconds[rows, columns])
        # The last KEEP statuses of every lane, members in order: the statuses kept so far
        # shifted left by the new ones, which fill the right end, oldest first.
        columns = np.flatnonzero(np.any(evaluated, axis=0))
        lanes = index[columns]
        flags = evaluated[:, columns]
        count = np.sum(flags, axis=0)
        from_end = count[None, :] - np.cumsum(flags, axis=0)
        recent = np.full((columns.shape[0], KEEP), -1, dtype=np.int64)
        rows, slots = np.nonzero(flags & (from_end < KEEP))
        recent[slots, KEEP - 1 - from_end[rows, slots]] = kinds[rows, columns[slots]]
        shift = np.minimum(count, KEEP)[:, None]
        place = np.arange(KEEP)[None, :]
        kept = np.take_along_axis(self.kinds[lanes], np.minimum(place + shift, KEEP - 1), axis=1)
        self.kinds[lanes] = np.where(place < KEEP - shift, kept, recent)

    def seconds_per_evaluation(self) -> dict:
        """Lane id to the mean seconds of one evaluation, for the lanes evaluated so far."""
        return {
            lane: float(self.seconds[index] / self.evaluations[index])
            for lane, index in self.position.items()
            if self.evaluations[index] > 0
        }

    def longest(self) -> float:
        """The longest mean seconds of one evaluation of a lane; 0 before any evaluation."""
        values = self.seconds_per_evaluation().values()
        return max(values, default=0.0)

    def entries(self) -> dict:
        """The saved form: host arrays named ``ledger.<field>``."""
        return {
            f"{_PREFIX}lane_ids": np.asarray([str(lane) for lane in self.lane_ids]),
            f"{_PREFIX}evaluations": self.evaluations.copy(),
            f"{_PREFIX}steps": self.steps.copy(),
            f"{_PREFIX}seconds": self.seconds.copy(),
            f"{_PREFIX}kinds": self.kinds.copy(),
        }

    def restore(self, entries: Mapping) -> bool:
        """Take the counts of a saved form for the lanes both name; whether one was found."""
        if f"{_PREFIX}lane_ids" not in entries:
            return False
        saved = [str(lane) for lane in np.asarray(entries[f"{_PREFIX}lane_ids"]).reshape(-1)]
        by_text = {str(lane): index for lane, index in self.position.items()}
        for row, lane in enumerate(saved):
            index = by_text.get(lane)
            if index is None:
                continue
            self.evaluations[index] = int(np.asarray(entries[f"{_PREFIX}evaluations"])[row])
            self.steps[index] = int(np.asarray(entries[f"{_PREFIX}steps"])[row])
            self.seconds[index] = float(np.asarray(entries[f"{_PREFIX}seconds"])[row])
            self.kinds[index] = np.asarray(entries[f"{_PREFIX}kinds"])[row]
        return True

    @classmethod
    def load(cls, path: Any) -> Ledger:
        """The ledger a checkpoint saved: an ``attempt-`` or ``population-`` file, or its directory.

        A directory gives its newest such file.

        Raises:
            ValueError: If no file is found or it holds no ledger.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("attempt-*.npz")) or sorted(path.glob("population-*.npz"))
            if not files:
                raise ValueError(f"{path} holds no attempt- or population- checkpoint file.")
            path = files[-1]
        with np.load(path, allow_pickle=False) as data:
            entries = {}
            for name in data.files:
                for prefix in ("", "state.", "state:"):
                    if name.startswith(prefix + _PREFIX):
                        entries[name[len(prefix) :]] = np.asarray(data[name])
        if f"{_PREFIX}lane_ids" not in entries:
            raise ValueError(f"{path} holds no ledger of the lanes.")
        ledger = cls([str(lane) for lane in entries[f"{_PREFIX}lane_ids"]])
        ledger.restore(entries)
        return ledger


__all__ = ["KEEP", "Ledger", "attributed_seconds"]
