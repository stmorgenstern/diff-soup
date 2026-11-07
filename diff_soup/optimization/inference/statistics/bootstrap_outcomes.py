"""The outcome of each bootstrap draw's fit."""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np

# The statuses of a replicate whose estimate enters the intervals.
_KEPT = ("converged", "steps_reached", "stalled")


class BootstrapOutcome(NamedTuple):
    """How one replicate, or one leave-one-member-out member, ended.

    The fields are those of [BootstrapOutcomes][diff_soup.optimization.inference.statistics.bootstrap_outcomes.BootstrapOutcomes]
    for one population member; ``point`` is its row of ``points``.
    """

    kind: str
    index: int
    status: str
    reason: Any
    stop: Any
    error: Any
    attempts: int
    accepted: int
    requests: int
    invalid_trials: int
    stationarity: float
    radius: float
    active_bounds: tuple
    failed_lanes: tuple
    point: np.ndarray
    final_cost: float
    data_cost: np.ndarray
    data_rows: np.ndarray
    training_rmse: float


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapOutcomes:
    """How every replicate, then every leave-one-member-out member, ended: one entry each.

    Population member ``q`` is replicate ``q`` for ``q < replicates``; member
    ``replicates + m`` leaves member ``m`` of the top level out (weight zero
    on it, one elsewhere) and serves the jackknife. Every attribute but
    ``replicates`` has one entry per population member, in that order;
    ``outcomes[q]`` gives member ``q``'s entries as one
    [BootstrapOutcome][diff_soup.optimization.inference.statistics.bootstrap_outcomes.BootstrapOutcome],
    and iterating gives them all.

    Attributes:
        replicates: ``R``, the replicates requested.
        kind: ``"replicate"`` or ``"jackknife"``.
        index: The replicate's number, or the number of the member left out.
        status: ``"converged"`` (the fit's converged tests, or the test of
            the next attempt passed at the last step), ``"steps_reached"``
            (it made its accepted steps), ``"stalled"`` (it moved, then its
            radius collapsed, the evaluation noise ended it, invalid trials
            ended it, or its attempts ran out) or ``"failed"``.
            ``converged``, ``steps_reached`` and ``stalled`` members are kept:
            their points are estimates; ``failed`` ones are left out.
        reason: ``None`` for ``converged`` and ``steps_reached``. For
            ``stalled``: ``"radius_collapsed"``, ``"evaluation_noise"``,
            ``"invalid_trials"`` or ``"attempts"``. For ``failed``:
            ``"no_step"`` (it ended without converging and without an
            accepted step, at the fit's point, which is not a draw),
            ``"system"`` (its level system at a point, or its full step at the
            fit, was not finite), ``"gauge"`` (every member a gauge averages
            over weighs zero) or ``"error"`` (its host work raised; ``error``
            holds the message).
        stop: The event that ended it: ``"stationarity"``,
            ``"gradient_tolerance"``, ``"step_tolerance"``,
            ``"cost_tolerance"``, ``"evaluation_noise"``,
            ``"invalid_trials"``, ``"steps"`` (its accepted steps were
            reached), ``"attempts"`` (its attempts ran out), or the failure's
            reason.
        error: The type and message of the exception that ended it, or
            ``None``.
        attempts: The attempts it made.
        accepted: Its accepted steps.
        requests: The requests it took part in. The evaluation at the fit,
            one request of the whole bootstrap at one point, is counted for
            replicate 0 (in a sharded run, by the part that holds it); the
            ``blocks`` request that formed the running replicates' systems
            when a run resumed from a checkpoint is counted for each of them.
        invalid_trials: Its invalid trials, withdrawn acceptances included.
        stationarity: The stationarity number at its end point (the decrease
            its model predicts, relative to ``max(|cost|, 1)``).
        radius: Its trust-region radius at the end.
        active_bounds: The sides active at its end point,
            ``((index, -1 | 1), ...)``.
        failed_lanes: Every lane that failed at one of its trials or at a
            point it accepted, each a
            [ReplicateLaneFailure][diff_soup.optimization.inference.hierarchical._replicates.ReplicateLaneFailure]
            with ``lane``, ``kind``, ``t_reached``, ``rungs``, ``attempt``,
            ``at`` (``"trial"`` or ``"accepted"``) and ``ignored`` (a lane of
            weight zero).
        points: ``(Q, n)``, the point each member ended at.
        final_cost: ``(Q,)``, its weighted cost there, regularizer rows
            included; NaN for a member that failed before its first attempt.
        data_cost: ``(Q, M)``: per member of the top level, the unweighted
            data cost of its lanes there (half the squared whitened residual,
            no regularizer row); NaN for a member one of whose lanes, of
            weight zero, failed there.
        data_rows: ``(Q, M)``: per member of the top level, the residual rows
            of its lanes (NaN as ``data_cost``).
        training_rmse: ``(Q,)``, ``sqrt(sum_m w_m 2 C_m / sum_m w_m N_m)``
            over the members of nonzero weight: the root mean square whitened
            residual of the resampled data set, each member counted ``w_m``
            times.
    """

    replicates: int
    kind: tuple
    index: np.ndarray
    status: tuple
    reason: tuple
    stop: tuple
    error: tuple
    attempts: np.ndarray
    accepted: np.ndarray
    requests: np.ndarray
    invalid_trials: np.ndarray
    stationarity: np.ndarray
    radius: np.ndarray
    active_bounds: tuple
    failed_lanes: tuple
    points: np.ndarray
    final_cost: np.ndarray
    data_cost: np.ndarray
    data_rows: np.ndarray
    training_rmse: np.ndarray

    @property
    def kept(self) -> np.ndarray:
        """``(Q,)`` booleans: the members whose points are estimates (not ``failed``)."""
        return np.asarray([status in _KEPT for status in self.status], dtype=bool)

    def __len__(self) -> int:
        """``Q``, the replicates and the leave-one-member-out members."""
        return len(self.status)

    def __getitem__(self, member: int) -> BootstrapOutcome:
        """Population member ``member``'s entries."""
        q = operator.index(member)
        if not -len(self) <= q < len(self):
            raise IndexError(
                f"the outcomes hold {len(self)} population members; got member {member}."
            )
        return BootstrapOutcome(
            kind=self.kind[q],
            index=int(self.index[q]),
            status=self.status[q],
            reason=self.reason[q],
            stop=self.stop[q],
            error=self.error[q],
            attempts=int(self.attempts[q]),
            accepted=int(self.accepted[q]),
            requests=int(self.requests[q]),
            invalid_trials=int(self.invalid_trials[q]),
            stationarity=float(self.stationarity[q]),
            radius=float(self.radius[q]),
            active_bounds=self.active_bounds[q],
            failed_lanes=self.failed_lanes[q],
            point=self.points[q],
            final_cost=float(self.final_cost[q]),
            data_cost=self.data_cost[q],
            data_rows=self.data_rows[q],
            training_rmse=float(self.training_rmse[q]),
        )

    def __iter__(self) -> Iterator[BootstrapOutcome]:
        """Every population member's entries, in order."""
        return (self[q] for q in range(len(self)))

    def __repr__(self) -> str:
        """A one-line summary."""
        return (
            f"BootstrapOutcomes(replicates={self.replicates}, members={len(self)}, "
            f"kept={int(np.count_nonzero(self.kept[: self.replicates]))})"
        )


__all__ = ["BootstrapOutcome", "BootstrapOutcomes"]
