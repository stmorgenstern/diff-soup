"""The nested trust-region fit: the joint step, then each member's own parameters polished with the outer ones held."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.fitting.result import OptResult, StepInfo
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _inequality as inequality
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._joint_fit import _JointFit
from diff_soup.optimization.inference.hierarchical._lane_route import (
    _LABEL,
    _Failure,
    _LaneRoute,
)
from diff_soup.optimization.inference.hierarchical._level_inner import model_split
from diff_soup.optimization.inference.hierarchical._level_system import nonfinite_members
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _noise_trial_limit,
    _scale_candidate,
    _subproblem,
    _trial_valid,
)
from diff_soup.optimization.inference.hierarchical._verdicts import (
    NONFINITE_BLOCKS,
    _failures_text,
    _stationarity_limit,
    _within,
)
from diff_soup.optimization.solvers._loop import (
    _NOISE_RUN,
    LoopState,
    _saved_count,
    cost_small,
    cost_threshold,
    loop_tolerances,
    step_small,
)
from diff_soup.optimization.solvers._loop_judge import (
    _TINY,
    _gain_ratio,
    _median,
    _next_radius,
    _noise_dominated,
    cost_scale,
)

if TYPE_CHECKING:
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion


# ---------------------------------------------------------------------------
# The polish
# ---------------------------------------------------------------------------


# Why a member's polish ended at an attempt: the codes of the record's ``inner_stop``.
_NOT_POLISHED = 0  # no polish: the outer trial was not accepted, or the member holds no data
_ACCURACY = 1  # its predicted decrease fell to inner_fraction of the outer step's
_AT_MAX_STEPS = 2  # it made the polish's largest number of iterations
_STEP_TOLERANCE = 3  # its step test
_COST_TOLERANCE = 4  # its cost test
_GRADIENT_TOLERANCE = 5  # its gradient test
_STATIONARITY = 6  # its stationarity number at most the inner stationarity_tol
_EVALUATION_NOISE = 7  # three noise-dominated rejected trials in a row
_RADIUS = 8  # its radius fell to the smallest positive float
_SKIPPED = 9  # the polish was switched off at this attempt
_NO_FINITE_STEP = 10  # no finite step inside its radius: its subtree's model is not finite
_UNSOLVED = 11  # its lanes failed at its polished point after every retreat

# The polish's iterations per member when the nested solver is given no inner solver.
_DEFAULT_POLISH = 2
# A polish whose gain is below this fraction of the outer step's predicted decrease on
# _SMALL_GAINS accepted attempts in a row is switched off until an outer rejection.
_POLISH_GAIN = 1e-3
_SMALL_GAINS = 2


class _Point(NamedTuple):
    """A point of the fit where every lane solved: the point, its system and its lane costs."""

    point: np.ndarray
    system: elimination.LevelSystem
    lane_costs: np.ndarray


class _Polish(NamedTuple):
    """A polish from one point: where it ended, the members' state there, and its record.

    ``point``, ``system`` and ``lane_costs`` are where it ended, a point
    where every lane solved; ``scale`` the scale it read; ``radii``,
    ``locks`` and ``thresholds`` every member's radius, growth lock and
    gradient threshold after it; ``iterations``, ``codes`` and ``retreats``
    every member's iterations, why they ended (the codes of ``inner_stop``)
    and its retreats; ``numbers`` every member's stationarity number at
    ``point``; ``first_cost`` the cost where it started.
    """

    point: np.ndarray
    system: elimination.LevelSystem
    lane_costs: np.ndarray
    scale: np.ndarray
    radii: np.ndarray
    locks: np.ndarray
    thresholds: np.ndarray
    iterations: np.ndarray
    codes: np.ndarray
    retreats: np.ndarray
    numbers: np.ndarray
    first_cost: float


def _spliced(
    system: elimination.LevelSystem,
    other: elimination.LevelSystem,
    cut: int,
    members: np.ndarray,
    point: np.ndarray,
    cost: float,
) -> elimination.LevelSystem:
    """``system`` with the subtrees of ``members`` of level ``cut`` taken from ``other``, at ``point``.

    Every node of level ``cut`` and below whose member (one flag per member)
    is set takes its diagonal block, its couplings and its gradient from
    ``other``; the levels above ``cut``, the held and fixed coordinates are
    ``system``'s, and the cost is ``cost``.
    """
    take = np.asarray(members, dtype=bool)
    diagonal, gradient = list(system.diagonal), list(system.gradient)
    cross = [None] + [list(row) for row in system.cross[1:]]
    for at in range(cut, system.depth + 1):
        if not system.widths[at]:
            continue
        own = take if at == cut else take[np.asarray(system.ancestors[at][cut])]
        pick = jax.numpy.asarray(own)
        diagonal[at] = jax.numpy.where(pick[:, None, None], other.diagonal[at], system.diagonal[at])
        gradient[at] = jax.numpy.where(pick[:, None], other.gradient[at], system.gradient[at])
        for above in range(at):
            if cross[at][above] is not None:
                cross[at][above] = jax.numpy.where(
                    pick[:, None, None], other.cross[at][above], system.cross[at][above]
                )
    return dataclasses.replace(
        system,
        diagonal=tuple(diagonal),
        cross=(None,) + tuple(tuple(row) for row in cross[1:]),
        gradient=tuple(gradient),
        point=jax.numpy.asarray(point, dtype=system.root.dtype),
        cost=jax.numpy.asarray(cost, dtype=system.root.dtype),
    )


def _member_numbers(decreases: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Every member's predicted decrease over its cost scale ``scales``; ``nan`` stays ``nan``."""
    with np.errstate(all="ignore"):
        return np.asarray(decreases, dtype=np.float64) / np.asarray(scales, dtype=np.float64)


class _NestedFit(_JointFit):
    """The joint fit, its members polished at one level after every accepted step.

    Every attempt is the joint mode's: its scale, subproblem, trial, gain
    ratio, radius, stop rules and verdict, the acceptance read on the trial's
    cost. While the polish is on, the trial's request is a frozen pass: each
    lane's cost at the trial and its ``J^T r`` there on the Jacobian of its
    last ``blocks`` request. After an accepted trial every member of the cut
    level iterates on its own subtree from there, the outer coordinates held,
    on the matrix of the point the attempt stepped from with the trial's
    gradient; all members in lockstep, each with its own radius, growth lock
    and acceptance, for at most the polish's number of iterations. An
    iteration's request is a frozen pass too, whose ``J^T r`` the moved
    members step from next on the kept matrix. A member done leaves the
    polish: an iteration's requests cover the lanes of the members still in
    it. The polish ends with a ``blocks`` request of every lane where it
    ended, the attempt's one, whose point, cost and system are the next
    attempt's. A member whose lanes fail at its polished point retreats along
    its polish step. The members' radii and locks carry from one polish to
    the next. With no polish iterations, or the polish switched off, the
    trial's request is ``value`` and the fit makes the joint mode's attempts;
    a polish in which no member iterates does too.
    """

    # Whether the polish's request at its trial is a frozen pass when the lanes run that
    # mode; False makes it a value request (for measurements).
    frozen_polish = True

    def __init__(self, solver: TrustRegion, route: _LaneRoute, callbacks: tuple):
        super().__init__(solver, route, callbacks)
        self.inner = solver if solver.inner is None else solver.inner
        # The polish's largest number of iterations per member and attempt.
        self.polish_steps = _DEFAULT_POLISH if solver.inner is None else int(solver.inner.max_steps)
        # The mode of the polish's request at its trial: a frozen pass, whose gradients on
        # the Jacobians the last blocks request kept the moved members take, when the polish
        # makes an iteration and the lanes run that mode; else value, whose moved members a
        # blocks request then reads.
        polishing = self.polish_steps > 0 or int(solver.inner_start_iterations) > 0
        frozen = self.frozen_polish and polishing and route.problem._offers("frozen")
        self.polish_pass = "frozen" if frozen else "value"
        if frozen:
            # Every blocks request keeps its lanes' Jacobians, which the frozen pass reads;
            # both programs are compiled with the others before the fit's first request.
            self.modes = ("value", "blocks", "frozen")
        self.cut = route.names.index(solver.nested_at) + 1
        self.members = tuple(route.members[self.cut - 1])
        self.lane_member = np.asarray(route.nodes[:, self.cut - 1], dtype=np.int64)
        # The member of the cut level of every joint column; set by the fit's first ``blocks``.
        self.owners: np.ndarray | None = None
        regularizer = route.view.regularization
        rows = regularizer.residual(jax.device_put(route.problem.x0, route.device))
        self.regularized = int(np.shape(rows)[0]) > 0
        self.work.nest(self.members)
        # The point the attempt steps from and the one before it, the attempt's polish,
        # the outer step's predicted decrease, the lane costs of the latest value request
        # of every lane, and the members' stationarity numbers at the current point and
        # at the point the attempt stepped from.
        self.current: _Point | None = None
        self.previous: _Point | None = None
        self.polished: _Polish | None = None
        self.outer_predicted = float("nan")
        self.trial_lanes: tuple | None = None
        self.member_numbers: np.ndarray | None = None
        self.stepped_from: np.ndarray | None = None

    def run(self, state: Mapping | None = None) -> OptResult:
        """Fit from the start point, or continue the fit whose ``state`` is given.

        The members' radii, locks and gradient thresholds and the polish's
        switch start afresh, or are those the saved state carries.
        """
        inner = self.inner
        dtype = np.asarray(self.route.problem.x0).dtype
        count = len(self.members)
        # The polish's tolerances: the outer ones when no inner solver is given.
        self.inner_tolerances = loop_tolerances(
            dtype,
            grad_tol=inner.grad_tol,
            step_tol=inner.step_tol,
            cost_tol=inner.cost_tol,
            cost_atol=inner.cost_atol,
            stationarity_tol=inner.stationarity_tol,
            label=_LABEL,
        )
        if state is None:
            self.radii = np.full((count,), float(inner.radius_init), dtype=np.float64)
            self.locks = np.zeros((count,), dtype=np.int64)
            self.thresholds = np.full((count,), np.nan)
            self.polish_off, self.small_gains = False, 0
            self.resumed_scale = None
        else:
            self.radii = np.array(state["inner_radius"], dtype=np.float64)
            self.locks = np.array(state["inner_lock"], dtype=np.int64)
            self.thresholds = np.array(state["inner_grad_threshold"], dtype=np.float64)
            # A state saved without the switch resumes with the polish on.
            self.polish_off = bool(_saved_count(state, "polish_off"))
            self.small_gains = _saved_count(state, "polish_small_gains")
            self.resumed_scale = np.array(state["scale"], dtype=dtype)
        return super().run(state)

    def nested_entries(self) -> dict:
        """The mode, the cut level, every member's inner radius, lock and threshold, the switch.

        ``polish_off`` is 1 while the polish is switched off, and
        ``polish_small_gains`` the number of accepted attempts in a row whose
        polish made an iteration and gained too little.
        """
        return {
            "mode": np.asarray("nested"),
            "nested_at": np.asarray(str(self.solver.nested_at)),
            "inner_radius": np.array(self.radii, dtype=np.float64),
            "inner_lock": np.array(self.locks, dtype=np.int64),
            "inner_grad_threshold": np.array(self.thresholds, dtype=np.float64),
            "polish_off": np.asarray(int(self.polish_off), dtype=np.int64),
            "polish_small_gains": np.asarray(self.small_gains, dtype=np.int64),
        }

    # -- the joint loop's hooks ------------------------------------------------

    def evaluate(self, mode: str, point: np.ndarray, lanes: Iterable[str] | None = None) -> Any:
        """The joint fit's request; a ``value`` or ``frozen`` request of every lane keeps its lane costs.

        The record of a rejected trial reads them (``owner_split``).
        """
        results = super().evaluate(mode, point, lanes)
        if mode in ("value", "frozen") and lanes is None:
            self.trial_lanes = (
                np.array(point, copy=True),
                np.array(results.cost, dtype=np.float64)[0],
            )
        return results

    def resolve(
        self, point: np.ndarray, raised: np.ndarray, before: np.ndarray, lane_costs: np.ndarray
    ) -> tuple:
        """The joint fit's consistency re-solve; at the current point its lane costs are kept."""
        replaced, change = super().resolve(point, raised, before, lane_costs)
        current = self.current
        if current is not None and np.array_equal(point, current.point):
            self.current = current._replace(lane_costs=replaced)
        return replaced, change

    def blocks(self, point: np.ndarray, *, start: bool) -> tuple:
        """The joint fit's ``blocks``; the first one also sets ``owners``, once.

        ``owners`` is the member of the cut level whose subtree holds each
        joint column (``-1`` above it). The fit's first request is this one,
        at the start or the restored point, before every method that reads it.
        A member whose blocks or gradient are not finite there, its lanes at a
        finite cost, fails as a lane that does not solve there.
        """
        system, lane_costs = super().blocks(point, start=start)
        if self.owners is None:
            self.owners = elimination.node_of_columns(system, self.cut)
        broken = nonfinite_members(system, self.cut)
        if broken.any():
            failures = tuple(
                _Failure(
                    str(self.route.ids[position]),
                    NONFINITE_BLOCKS,
                    float("nan"),
                    (),
                    (int(self.work.lane_rungs[position]),),
                )
                for position in np.flatnonzero(np.isin(self.lane_member, np.flatnonzero(broken)))
            )
            where = "the start point" if start else "an accepted point"
            raise ValueError(
                f"lane(s) {_failures_text(failures)} of {self.solver.nested_at} member(s) "
                f"{[self.members[int(member)] for member in np.flatnonzero(broken)]} failed at "
                f"{where} in the blocks mode; every lane has to solve at {where}."
            )
        return system, lane_costs

    def at_start(
        self, point: np.ndarray, system: elimination.LevelSystem, lane_costs: np.ndarray, *, fresh
    ) -> tuple:
        """The polish at a ``fresh`` start, at most ``inner_start_iterations``; the inner programs.

        A resumed fit's members' numbers are read at its point in its saved scale.
        """
        elimination.inner_programs(system, self.cut)
        self.logged = len(self.work.log)
        limit = int(self.solver.inner_start_iterations) if fresh else 0
        polished = self.lockstep(
            point,
            system,
            lane_costs,
            scale=self.resumed_scale,
            limit=limit,
            attempt=-1,
            start=True,
        )
        if limit == 0:
            polished = polished._replace(codes=np.full_like(polished.codes, _NOT_POLISHED))
        self.keep(polished)
        self.polished = polished if fresh else None
        return polished.point, polished.system, polished.lane_costs

    def trial_pass(self) -> str:
        """``frozen`` while the polish is on and iterates on frozen passes; else ``value``.

        The polish's switch moves only in ``record``, after the attempt.
        """
        if self.polish_pass == "frozen" and self.polish_steps > 0 and not self.polish_off:
            return "frozen"
        return "value"

    def accept(
        self,
        loop: LoopState,
        system: elimination.LevelSystem,
        results: Any,
        *,
        attempt: int,
        predicted: float,
        level: Real | None,
    ) -> tuple:
        """The polish from the accepted trial ``loop.point``; the system and lane costs where it ended.

        After a frozen pass at the trial (``trial_pass``) the members polish
        from the trial on ``system``'s matrix, that of the point the attempt
        stepped from, with the gradient the pass gave at the trial
        (``frozen_system``); the ``blocks`` request where the polish ends is
        the attempt's one. After a ``value`` request the joint mode's
        ``blocks`` request at the trial comes first. ``loop.point``,
        ``loop.cost`` and ``loop.scale`` move to where the polish ended.
        """
        if self.trial_pass() == "frozen":
            lane_costs = np.array(results.cost, dtype=np.float64)[0]
            kept = self.route.frozen_system(system, results, loop.point)
            return self.polish(
                loop, kept, lane_costs, attempt=attempt, predicted=predicted, level=level, kept=True
            )
        system, lane_costs = super().accept(
            loop, system, results, attempt=attempt, predicted=predicted, level=level
        )
        return self.polish(
            loop, system, lane_costs, attempt=attempt, predicted=predicted, level=level
        )

    def polish(
        self,
        loop: LoopState,
        system: elimination.LevelSystem,
        lane_costs: np.ndarray,
        *,
        attempt: int,
        predicted: float,
        level: Real | None,
        kept: bool = False,
    ) -> tuple:
        """The members' polish at the accepted point, the outer values held.

        At most the polish's number of iterations, none while the polish is
        switched off; every member done at ``inner_fraction`` of the outer
        step's predicted decrease ``predicted``, by its stationarity number
        (raised to its gradient floor, ``level`` times its full step, once the
        floor is measured) and by its own tests (``lockstep``). With ``kept``
        ``system`` is a frozen system at the point (``lockstep``).
        ``loop.point``, ``loop.cost`` and ``loop.scale`` move to where it
        ended.
        """
        self.stepped_from = self.member_numbers
        limit = self.polish_steps
        if self.polish_off:
            self.work.polish_skipped += 1
            limit = 0
        before = self.current
        polished = self.lockstep(
            loop.point,
            system,
            lane_costs,
            scale=loop.scale,
            limit=limit,
            attempt=attempt,
            start=False,
            outer_predicted=predicted,
            noise=_median(loop.noise_dominated_cost_change),
            level=level,
            kept=kept,
        )
        self.keep(polished)
        self.previous, self.polished, self.outer_predicted = before, polished, float(predicted)
        loop.cost = float(polished.system.cost)
        if np.any(polished.iterations):
            loop.point, loop.scale = polished.point, polished.scale
        return polished.system, polished.lane_costs

    def keep(self, polished: _Polish) -> None:
        """Make the point where ``polished`` ended the current one, with the members' state there."""
        self.radii, self.locks = polished.radii, polished.locks
        self.thresholds = polished.thresholds
        self.member_numbers = polished.numbers
        self.current = _Point(polished.point, polished.system, polished.lane_costs)
        self.work.held = polished.system.held
        # The lanes' terms of the gradient floor's bound move with the point.
        self.floor.current = self.floor.latest

    def record(self, rho: float, lam: float, accepted: bool, trial: np.ndarray, failures: tuple):
        """The attempt just made (or the start), with every member's polish and the cost split.

        Per member: its polish iterations (``inner_iterations``), the code of
        why its polish ended (``inner_stop``) and its retreats
        (``inner_retreats``), all zero for an attempt without a polish; and
        the split of the cost by member (``owner_split``). It also moves the
        polish's switch: an accepted attempt whose polish made an iteration
        and gained less than ``_POLISH_GAIN`` of the outer step's predicted
        decrease counts toward switching it off (``_SMALL_GAINS`` in a row),
        one that gained more resets the count, and an outer rejection or a
        failed lane turns it on.
        """
        entries = super().record(rho, lam, accepted, trial, failures)
        count = len(self.members)
        polished = self.polished
        if polished is None:
            entries["inner_iterations"] = np.zeros((count,), dtype=np.int64)
            entries["inner_stop"] = np.full((count,), _NOT_POLISHED, dtype=np.int64)
            entries["inner_retreats"] = np.zeros((count,), dtype=np.int64)
        else:
            entries["inner_iterations"] = np.array(polished.iterations, dtype=np.int64)
            entries["inner_stop"] = np.array(polished.codes, dtype=np.int64)
            entries["inner_retreats"] = np.array(polished.retreats, dtype=np.int64)
        entries.update(self.owner_split(accepted, trial, failures))
        if not accepted or len(self.work.log) > self.logged:
            self.polish_off, self.small_gains = False, 0
        elif polished is not None and int(np.max(polished.iterations, initial=0)) > 0:
            gain = polished.first_cost - float(polished.system.cost)
            small = gain < _POLISH_GAIN * self.outer_predicted
            self.small_gains = self.small_gains + 1 if small else 0
            if self.small_gains >= _SMALL_GAINS:
                self.polish_off, self.small_gains = True, 0
        self.logged = len(self.work.log)
        self.polished = None
        return entries

    def step_info(self, loop: LoopState, attempt: int, **fields: Any) -> StepInfo:
        """The joint fit's ``StepInfo``, every member's number at the point the attempt stepped from."""
        numbers = self.stepped_from if fields.get("accepted") else self.member_numbers
        return super().step_info(loop, attempt, member_stationarity=numbers, **fields)

    def outcome(self, loop: LoopState, stop: str, **values: Any) -> OptResult:
        """The joint fit's result, every member's number at the end point reported beside it."""
        return super().outcome(loop, stop, members=self.member_numbers, **values)

    # -- the cost by member ------------------------------------------------------

    def member_costs(self, lane_costs: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Each member's cost: its lanes' weighted costs and the regularizer rows of its subtree.

        Every regularizer block counts with the route's weight of that block.
        """
        lanes = self.route.omega * np.asarray(lane_costs, dtype=np.float64)
        costs = np.bincount(self.lane_member, weights=lanes, minlength=len(self.members))
        if not self.regularized:
            return costs
        regularizer = self.route.view.regularization
        weights = self.route.block_weights
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.route.device)))
        if weights is not None and len(declared) != len(weights):
            raise ValueError(
                f"the regularizer's block_rows(x) declares {len(declared)} blocks at this point "
                f"and {len(weights)} where its weights were attributed; a regularizer declares "
                "the same blocks at every point."
            )
        for index, (columns, residual, _) in enumerate(declared):
            members = self.owners[np.asarray(columns, dtype=np.int64)]
            below = members[members >= 0]
            if below.size:
                weight = None if weights is None else float(weights[index])
                if weight == 0.0:
                    continue
                residual = np.ravel(np.asarray(residual, dtype=np.float64))
                if weight is None:
                    costs[int(below[0])] += 0.5 * float(residual @ residual)
                else:
                    costs[int(below[0])] += weight * (0.5 * float(residual @ residual))
        return costs

    def shared_cost(self, point: np.ndarray) -> float:
        """The regularizer rows of no member: the blocks on coordinates above the cut level only."""
        if not self.regularized:
            return 0.0
        regularizer = self.route.view.regularization
        weights = self.route.block_weights
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.route.device)))
        total = 0.0
        for index, (columns, residual, _) in enumerate(declared):
            members = self.owners[np.asarray(columns, dtype=np.int64)]
            if members[members >= 0].size:
                continue
            residual = np.ravel(np.asarray(residual, dtype=np.float64))
            weight = 1.0 if weights is None else float(weights[index])
            total += weight * (0.5 * float(residual @ residual))
        return total

    def owner_split(self, accepted: bool, trial: np.ndarray, failures: tuple) -> dict:
        """The attempt's cost split by member of the cut level, read off without changing the fit.

        ``owner_costs`` holds every member's cost at the point the attempt
        stepped from (``member_costs``) and ``owner_shared_cost`` the rest of
        the cost there (``shared_cost``); ``owner_trial_costs`` and
        ``owner_trial_shared_cost`` the same where the attempt ended: after
        the polish for an accepted trial, at the trial for a rejected one
        (``nan`` when the trial was not evaluated, and for a member whose
        lanes failed there); ``owner_predicted`` every member's part of the
        decrease the Gauss-Newton model there predicts for the whole move
        (its subtree's terms and their coupling to the outer step,
        ``model_split``) and ``owner_shared_predicted`` the outer
        coordinates' own terms; together they sum to the model's decrease for
        that move.
        """
        count = len(self.members)
        point, system, lane_costs = self.previous if accepted else self.current
        entries = {
            "owner_costs": np.asarray(self.member_costs(lane_costs, point), dtype=np.float64),
            "owner_shared_cost": np.asarray(self.shared_cost(point), dtype=np.float64),
            "owner_trial_costs": np.full((count,), np.nan),
            "owner_trial_shared_cost": np.asarray(np.nan),
        }
        if accepted:
            reached, costs = self.current.point, self.current.lane_costs
        else:
            reached, costs = np.asarray(trial), None
            kept = self.trial_lanes
            if kept is not None and np.array_equal(kept[0], reached):
                costs = kept[1]
        delta = np.asarray(reached, dtype=np.float64) - np.asarray(point, dtype=np.float64)
        members, rest = model_split(system, delta, self.cut)
        entries["owner_predicted"] = np.asarray(members, dtype=np.float64)
        entries["owner_shared_predicted"] = np.asarray(rest, dtype=np.float64)
        if costs is not None:
            split = np.asarray(self.member_costs(costs, reached), dtype=np.float64)
            split[self.failed_members(failures)] = np.nan
            entries["owner_trial_costs"] = split
            entries["owner_trial_shared_cost"] = np.asarray(
                self.shared_cost(reached), dtype=np.float64
            )
        return entries

    # -- the members' models -----------------------------------------------------

    def member_decrease(
        self,
        system: elimination.LevelSystem,
        scale: ArrayLike,
        zero: tuple,
        addition: ArrayLike | None = None,
    ) -> np.ndarray:
        """The decrease every member's model predicts for its full step at zero damping.

        ``zero`` is ``inner_solve_scaled(system, scale, cut, addition=addition)``;
        member ``n``'s full step is its part of ``z / scale``, and its decrease
        is the one its own model (the scaled matrix carrying ``addition`` on its
        diagonal) predicts for that step: ``nan`` for a member whose solve is
        not finite.
        """
        z, q_squared = np.asarray(zero[0]), np.asarray(zero[1], dtype=np.float64)
        with np.errstate(all="ignore"):
            delta = z / np.asarray(scale, dtype=z.dtype)
        owners = self.owners
        below = owners >= 0
        broken = np.bincount(
            owners[below],
            weights=(~np.isfinite(z[below])).astype(np.float64),
            minlength=len(self.members),
        )
        predicted = elimination.inner_predicted(system, delta, self.cut)
        if addition is not None:
            predicted -= 0.5 * elimination.inner_added(system, addition, z, self.cut)
        predicted[(broken > 0) | ~np.isfinite(q_squared)] = np.nan
        return predicted

    def member_scales(self, costs: ArrayLike) -> float | np.ndarray:
        """Every member's cost scale ``max(|cost_n|, cost_atol / cost_tol)``, inner tolerances."""
        tolerances = self.inner_tolerances
        return cost_scale(costs, tolerances.cost_tol, tolerances.cost_atol)

    def member_full_steps(self, zero: tuple, costs: np.ndarray) -> np.ndarray:
        """Every member's full-step norm over its cost scale (``member_scales``).

        ``zero`` is every member's solve at zero damping
        (``inner_solve_scaled``), in the scaled coordinates its radius is
        measured in. The gradient floor times this number is the member's
        floor in its stationarity number's units; ``nan`` for a member whose
        solve is not finite.
        """
        z = np.asarray(zero[0], dtype=np.float64)
        owners = self.owners
        below = owners >= 0
        with np.errstate(all="ignore"):
            norms = np.sqrt(
                np.bincount(owners[below], weights=z[below] ** 2, minlength=len(self.members))
            )
        return norms / self.member_scales(costs)

    def member_quadratic(
        self, system: elimination.LevelSystem, coordinates: box.Scaling, member: int
    ) -> Callable[..., float]:
        """``quadratic_form`` of one member's scaled subtree model, its diagonal addition in it.

        It takes a vector over the member's columns in the coordinates
        ``z = coordinates.scale * s`` and gives ``z^T (H_z + diag(addition)) z``.
        """
        columns = np.flatnonzero(self.owners == member)
        with np.errstate(all="ignore"):
            d = 1.0 / coordinates.scale
        addition = coordinates.addition[columns]

        def quadratic_form(vector: np.ndarray) -> float:
            full = np.zeros_like(d)
            with np.errstate(all="ignore"):
                full[columns] = d[columns] * vector
                quadratic = elimination.inner_quadratic(system, full, self.cut)[member]
                return float(quadratic) + float(addition @ (vector * vector))

        return quadratic_form

    def choose_members(
        self,
        system: elimination.LevelSystem,
        x: np.ndarray,
        gradient: np.ndarray,
        coordinates: box.Scaling,
        step: np.ndarray,
        radii: np.ndarray,
        active: np.ndarray,
        predicted: np.ndarray,
    ) -> tuple:
        """Every active member's trial step under the box, from its subproblem's solution.

        A member whose ``x + step`` stays in the box keeps its step and its
        predicted decrease; every other one takes ``_bounds.select_step`` on its
        scaled subtree model with its radius. Returns ``(step, scaled_step,
        predicted)`` in joint columns and per member.
        """
        lower, upper = self.route.bounds
        with np.errstate(all="ignore"):
            d = 1.0 / coordinates.scale
            scaled = coordinates.scale * step
        g_h = d * gradient
        chosen, chosen_scaled = step.copy(), scaled.copy()
        predicted = np.array(predicted, dtype=np.float64, copy=True)
        for member in np.flatnonzero(active):
            columns = np.flatnonzero(self.owners == member)
            if box.in_bounds(x[columns] + step[columns], lower[columns], upper[columns]):
                continue
            reach = max(float(radii[member]), float(np.linalg.norm(scaled[columns])))
            member_step, member_scaled, value = box.select_step(
                x[columns],
                self.member_quadratic(system, coordinates, int(member)),
                g_h[columns],
                step[columns],
                scaled[columns],
                d[columns],
                reach,
                lower[columns],
                upper[columns],
                box.step_back_ratio(gradient[columns], coordinates.v[columns]),
            )
            chosen[columns] = member_step
            chosen_scaled[columns] = member_scaled
            predicted[member] = float(value)
        return chosen, chosen_scaled, predicted

    # -- the polish's requests ---------------------------------------------------

    def valid(self, point: np.ndarray) -> bool:
        """The trial predicate at ``point``; a predicate that raises rejects it."""
        return _trial_valid(self.route.view, point)

    def nonfinite_block_failures(self, results: Any, system: elimination.LevelSystem) -> tuple:
        """The lanes of every member whose Gauss-Newton blocks or gradient are not finite at a finite cost.

        Each of the member's lanes is a failure of kind ``NONFINITE_BLOCKS``: its solve reached
        the end at a finite cost and its sensitivities are not finite there. A member none of
        whose lanes ``results`` holds is not named.
        """
        broken = nonfinite_members(system, self.cut)
        if not broken.any():
            return ()
        route = self.route
        costs = self.lane_costs_of(results, np.full((len(route.ids),), np.nan))
        records = []
        for member in np.flatnonzero(broken):
            positions = np.flatnonzero(self.lane_member == member)
            if not np.all(np.isfinite(costs[positions])):
                continue
            for position in positions:
                row = route.nodes[position]
                nodes = tuple(
                    str(route.members[level][int(row[level])]) for level in range(route.depth)
                )
                rungs = (int(self.work.lane_rungs[position]),)
                records.append(
                    _Failure(str(route.ids[position]), NONFINITE_BLOCKS, float("nan"), nodes, rungs)
                )
        return tuple(records)

    def failed_members(self, failures: tuple) -> np.ndarray:
        """The members of the cut level that the failed lanes belong to, each once, in order."""
        position = self.route.position
        return np.unique(
            np.asarray(
                [self.lane_member[position[failure.lane]] for failure in failures], dtype=np.int64
            )
        )

    def member_lanes(self, members: np.ndarray) -> tuple:
        """The ids of the lanes of ``members`` (one flag per member of the cut level), in lane order."""
        chosen = np.asarray(members, dtype=bool)[self.lane_member]
        return tuple(self.route.ids[position] for position in np.flatnonzero(chosen))

    def lane_costs_of(self, results: Any, costs: np.ndarray) -> np.ndarray:
        """``costs``, one per planned lane, with those of the lanes ``results`` holds replaced."""
        replaced = np.array(costs, dtype=np.float64, copy=True)
        positions = [self.route.position[str(lane)] for lane in results.lane_ids]
        replaced[positions] = np.asarray(results.cost, dtype=np.float64)[0]
        return replaced

    def request_blocks(
        self,
        point: np.ndarray,
        attempt: int,
        base: _Point | None = None,
        moved: np.ndarray | None = None,
    ) -> tuple:
        """One ``blocks`` request at ``point``: every lane, or the lanes of the members ``moved``.

        With ``moved`` the system is ``base.system`` with the subtrees of
        those members taken from the request (``_spliced``), and the lane
        costs ``base.lane_costs`` with theirs replaced; the outer coordinates
        are held, so every other member's nodes are those at ``point``. A
        member whose lanes solved at a finite cost but whose Gauss-Newton
        blocks are not finite fails like a member whose lanes did not solve.
        A failed lane of weight zero is logged as ignored.

        Returns ``(lane_costs, system, failures)``: ``failures`` (not logged)
        and ``None`` for the others when a lane failed.
        """
        route, work = self.route, self.work
        lanes = None if moved is None else self.member_lanes(moved)
        results = self.evaluate("blocks", point, lanes)
        failures, ignored = route.weigh(route.failures(results))
        work.record(attempt, ignored, inner=True)
        if failures:
            return None, None, failures
        system = route.system(results, point)
        if moved is None:
            lane_costs = np.array(results.cost, dtype=np.float64)[0]
        else:
            lane_costs = self.lane_costs_of(results, base.lane_costs)
            cost = self.cost_at(lane_costs, point)
            system = _spliced(base.system, system, self.cut, moved, point, cost)
        failures, ignored = route.weigh(self.nonfinite_block_failures(results, system))
        work.record(attempt, ignored, inner=True)
        if failures:
            return None, None, failures
        if moved is None:
            # The lanes' terms of the gradient floor's bound at this point.
            self.floor.record(results)
        return lane_costs, system, ()

    def cost_at(self, lane_costs: np.ndarray, point: np.ndarray) -> float:
        """The cost at ``point`` from every lane's cost there: the weighted lanes and the regularizer."""
        blocks = self.route.regularizer_at(point)
        regularizer = 0.0 if blocks is None else float(blocks[1])
        return (
            float(np.sum(self.route.omega * np.asarray(lane_costs, dtype=np.float64))) + regularizer
        )

    def retreating_blocks(
        self,
        point: np.ndarray,
        base: _Point,
        moved: np.ndarray | None,
        radii: np.ndarray,
        retreats: np.ndarray,
        attempt: int,
    ) -> tuple:
        """The ``blocks`` request at a polished point, the members retreating.

        ``base`` is the point before the polish's last iteration, where every
        lane solved; ``moved`` the members whose step that iteration accepted,
        whose lanes the request covers (``request_blocks``), or ``None`` for
        every lane. A member with a lane that no rung solves at ``point``
        retreats along its
        move from ``anchor``: at its ``j``-th retreat its subtree's coordinates
        are ``anchor + (point - anchor) / 2**j``, its radius is quartered, and
        only its lanes are solved again there, in one ``value`` request at
        the rungs they hold with every other coordinate as it was, up to the
        solver's ``inner_retreats`` times. When the lanes of every member that
        retreated solve, the ``blocks`` request is made again at the
        retreated point; a member whose lane fails there again goes on with
        the retreats it has left. A retreated point that the trial predicate
        rejects counts as a retreat at which the member's lanes failed.
        ``radii`` and ``retreats`` (the retreats of the attempt, per member)
        are updated in place, and every failure a retreat answers is logged
        as the member's. A failed lane of weight zero is logged as ignored and
        makes no member retreat.

        Returns ``(point, lane_costs, system, failures)``: the point where
        every lane solved, the lane costs and system there, and ``()``; or,
        when a member cannot be solved at any retreat, the last point tried,
        ``None``, ``None`` and the failures of the last request, logged.
        """
        route, work = self.route, self.work
        allowed = int(self.solver.inner_retreats)
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        anchor = np.asarray(base.point)
        failing = np.array(point, copy=True)
        point = np.array(point, copy=True)
        tried = np.zeros((len(self.members),), dtype=np.int64)
        while True:
            lane_costs, system, failures = self.request_blocks(point, attempt, base, moved)
            if not failures:
                return point, lane_costs, system, ()
            pending = self.failed_members(failures)
            unlogged = failures
            while pending.size:
                if np.any(tried[pending] >= allowed):
                    work.record(attempt, unlogged, inner=True)
                    return point, None, None, failures
                if unlogged:
                    work.record(attempt, unlogged, inner=True)
                    unlogged = ()
                tried[pending] += 1
                retreats[pending] += 1
                work.inner_retreats[pending] += 1
                radii[pending] = np.maximum(radii[pending] / 4.0, _TINY)
                moving = below & np.isin(member_of, pending)
                halving = 0.5 ** tried[member_of[moving]]
                point[moving] = anchor[moving] + (failing[moving] - anchor[moving]) * halving
                lanes = self.member_lanes(np.isin(np.arange(len(self.members)), pending))
                if not self.valid(point):
                    work.predicate_rejections += 1
                    continue
                retried = self.evaluate("value", point, lanes)
                failures, ignored = route.weigh(route.failures(retried))
                work.record(attempt, ignored, inner=True)
                unlogged = failures
                pending = self.failed_members(failures) if failures else pending[:0]

    def settle(
        self,
        point: np.ndarray,
        base: _Point,
        moved: np.ndarray,
        every: bool,
        radii: np.ndarray,
        retreats: np.ndarray,
        attempt: int,
        codes: np.ndarray,
        active: np.ndarray,
    ) -> tuple:
        """The system at ``point``, where the members ``moved`` left ``base``, a point where every lane solved.

        One ``blocks`` request at ``point``: of every lane with ``every``,
        else of the lanes of ``moved`` (``retreating_blocks``, whose member
        that fails retreats toward ``base``). A member that no retreat solves
        goes back to its coordinates in ``base`` and leaves the polish (its
        code ``_UNSOLVED`` in ``codes``, cleared in ``active``), and the
        request is made again for the others; with none left, it is ``base``.

        Returns ``(point, lane_costs, system, partial)``: ``partial`` whether
        the system holds nodes of a request of some lanes only, ``None`` when
        it is ``base``'s.
        """
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        moved = np.array(moved, dtype=bool, copy=True)
        point = np.array(point, copy=True)
        while True:
            reached, lane_costs, system, failures = self.retreating_blocks(
                point, base, None if every else moved, radii, retreats, attempt
            )
            if not failures:
                return reached, lane_costs, system, not every
            unsolved = self.failed_members(failures)
            codes[unsolved] = _UNSOLVED
            active[unsolved] = False
            progress = bool(moved[unsolved].any())
            moved[unsolved] = False
            if not progress or not moved.any():
                return np.array(base.point, copy=True), base.lane_costs, base.system, None
            point = reached
            back = below & np.isin(member_of, unsolved)
            point[back] = np.asarray(base.point)[back]

    def polish_request(self, point: np.ndarray, members: np.ndarray) -> Any:
        """The polish's request at its trial ``point`` over the lanes of ``members``.

        In ``polish_pass`` mode: ``value``, or ``frozen``, which also gives
        each lane's ``J^T r`` there on the Jacobian of its last ``blocks``
        request, the moved members' gradient at their new point.
        """
        return self.evaluate(self.polish_pass, point, self.member_lanes(members))

    def lockstep(
        self,
        point: np.ndarray,
        system: elimination.LevelSystem,
        lane_costs: np.ndarray,
        *,
        scale: ArrayLike | None,
        limit: int,
        attempt: int,
        start: bool,
        outer_predicted: Real | None = None,
        noise: float = 0.0,
        level: Real | None = None,
        kept: bool = False,
    ) -> _Polish:
        """Every member of the cut level iterates on its subtree from ``point``, the outer values held.

        ``system`` and ``lane_costs`` are those at ``point``; with ``kept``
        ``system`` is a frozen system there (``frozen_system``: the matrix of
        an earlier point, the gradient at ``point``), and the polish ends with
        a ``blocks`` request of every lane however it went: the joint
        mode's at ``point`` when no member moved from it. The members start
        with the radii, locks and gradient thresholds the fit carries and make
        at most ``limit`` lockstep iterations; ``limit=0`` reads ``point``
        alone. An iteration makes one request at the members' combined trial
        over the lanes of the members it moves (``polish_request``). After a
        frozen pass the members that moved take its gradient on the kept
        matrix; after a ``value`` request (or a frozen pass with a failed lane)
        one ``blocks`` request at their new point over the lanes of the members
        that moved updates their nodes of the system (``request_blocks``): of
        every lane when they hold every lane or it is the last iteration. A
        polish that ends on a system of some lanes' requests makes the
        ``blocks`` request of every lane where it ended, a member whose lanes
        fail there retreating toward where the polish started. ``start`` marks the
        start's polish, whose iterations count apart. A member whose lane fails
        at a ``blocks`` request after an iteration retreats along its last
        step (``retreating_blocks``); one that no retreat solves goes back to
        its point before that iteration and leaves the polish. A member is done when
        its stationarity number is at most the inner ``stationarity_tol``,
        when its gradient is within the inner ``grad_tol`` (when that is set),
        when the decrease its model predicts for its full step is at most
        ``inner_fraction`` times ``outer_predicted`` (the decrease the outer
        model predicted for the outer step; ``None`` tests nothing), by its
        step and cost tests, when three of its rejected trials in a row
        within these iterations are dominated by the noise of the cost's
        evaluation, or when its radius falls to the smallest positive float.
        The tolerance's default reads ``noise``, the evaluation noise the
        outer trials measured, and so do the members' gain ratios under the
        inner ``noise_relaxed_ratio``. With ``level``, the gradient floor the
        outer trials measured (``None`` before the first sample), a member is
        also done when its stationarity number is at most its own floor,
        ``level`` times its full step's scaled norm over its cost scale
        ``max(|cost_n|, cost_atol / cost_tol)`` under the inner tolerances.
        """
        solver, inner, route, work = self.solver, self.inner, self.route, self.work
        tolerances = self.inner_tolerances
        count = len(self.members)
        x = np.array(point, copy=True)
        lane_costs = np.array(lane_costs, dtype=np.float64, copy=True)
        scale = None if scale is None else np.array(scale, copy=True)
        radii = np.array(self.radii, dtype=np.float64, copy=True)
        locks = np.array(self.locks, dtype=np.int64, copy=True)
        thresholds = np.array(self.thresholds, dtype=np.float64, copy=True)
        iterations = np.zeros((count,), dtype=np.int64)
        retreats = np.zeros((count,), dtype=np.int64)
        bounds = route.bounds
        fixed = np.asarray(route.fixed, dtype=np.int64)
        tiny = float(np.finfo(float).tiny)
        member_limit = _stationarity_limit(tolerances.stationarity_tol, tolerances.cost_tol, noise)
        noise_limit = _noise_trial_limit(inner, route)
        accuracy = (
            None if outer_predicted is None else solver.inner_fraction * float(outer_predicted)
        )
        first_cost = float(system.cost)
        # Where the polish started, and whether the system holds nodes of a request of
        # some lanes only.
        start = _Point(np.array(point, copy=True), system, lane_costs)
        spliced = kept
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        held = np.array(system.held_mask[self.cut], dtype=bool)
        active = ~held
        floored = np.zeros((count,), dtype=bool)
        codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
        # Per member: its noise-dominated rejected trials in a row, and the radius of its
        # last accepted trial in these iterations (nan before the first).
        noise_runs = np.zeros((count,), dtype=np.int64)
        accepted_radii = np.full((count,), np.nan)

        def per_member(values: np.ndarray) -> np.ndarray:
            return np.sqrt(np.bincount(owners[below], weights=values[below] ** 2, minlength=count))

        def finish(done: np.ndarray, code: int) -> None:
            codes[done] = code
            active[done] = False

        k = 0
        while True:
            # (2) the scale; (3) every member's gradient, its threshold, who is done
            gradient, diagonal = self.route.vectors(system)
            candidate = _scale_candidate(system, solver.x_scale, diagonal)
            scale = candidate if scale is None else np.maximum(scale, candidate)
            costs = self.member_costs(lane_costs, x)
            # Every member's subproblem coordinates; under bounds shaped by the
            # distance to the sides its gradient points at.
            coordinates = _subproblem(x, gradient, scale, bounds)
            # A member's gradient test, when the inner grad_tol is set; without it the
            # members' stationarity numbers below are their test.
            if tolerances.grad_tolerance is not None:
                norms = per_member(_bounded_gradient(gradient, coordinates, bounds))
                for member in np.flatnonzero(np.isnan(thresholds) & ~held):
                    thresholds[member] = tolerances.grad_tolerance.threshold(norms[member])
                finish(active & (norms <= thresholds), _GRADIENT_TOLERANCE)
            # Every member's solve at zero damping: its predicted decrease and its
            # stationarity number at the current point, and the first solve of the
            # active members' searches.
            # Under inequality rows on the members' own values, every member's solve holds
            # those its full step would violate; ``first`` is the solve holding none.
            own = None
            if self.rows is not None:
                own = inequality.member_rows(self.rows, system, x, self.cut)
            if own is None:
                zero = first = elimination.inner_solve_scaled(
                    system, coordinates.scale, self.cut, addition=coordinates.addition
                )
            else:
                zero, first, solves = inequality.member_zero(own, system, x, coordinates, self.cut)
                work.solves += solves - 1
            decreases = self.member_decrease(system, coordinates.scale, zero, coordinates.addition)
            stationarity = _member_numbers(decreases, self.member_scales(costs))
            full_steps = self.member_full_steps(zero, costs)
            # A member whose model predicts no further decrease the evaluation can
            # show is done, as one whose gradient is within its threshold, and as
            # one that has less to gain than inner_fraction of the outer step; once
            # the gradient floor is measured, so is one whose model predicts no more
            # than its derivatives resolve.
            limits = member_limit if level is None else np.fmax(member_limit, level * full_steps)
            finish(active & _within(stationarity, limits), _STATIONARITY)
            if accuracy is not None:
                finish(active & _within(decreases, accuracy), _ACCURACY)
            # (4) the exit always follows a blocks request at the current point
            if not active.any() or k == int(limit):
                # the exit makes no step; its one solve is the one above
                work.solves += 1
                break
            # (5) every active member's step; (6) bounds, fixed coordinates, predicate
            if own is None:
                step, predicted, multipliers, solves = elimination.inner_secular_step(
                    system,
                    coordinates.scale,
                    radii,
                    self.cut,
                    active,
                    first=zero,
                    addition=coordinates.addition,
                    hold=True,
                )
            else:
                step, predicted, multipliers, solves, held_rows = inequality.member_step(
                    own, system, x, gradient, coordinates, radii, self.cut, active, first, bounds
                )
            work.solves += solves
            # A member with no finite step inside its radius (its subtree's model is not
            # finite) ends its polish at its current point; the others go on.
            finish(active & np.isnan(multipliers), _NO_FINITE_STEP)
            if not active.any():
                break
            if fixed.size:
                step[fixed] = 0.0
            moving = below & active[member_of]
            trial = x.copy()
            if bounds is None:
                trial[moving] = x[moving] + step[moving]
                scaled = coordinates.scale * step
            else:
                # Every member whose step leaves the box takes the best of its step,
                # the step reflected at the sides it crosses and the Cauchy step, by
                # its model's value; its trial is kept strictly inside the box.
                step, scaled, predicted = self.choose_members(
                    system, x, gradient, coordinates, step, radii, active, predicted
                )
                trial[moving] = box.make_strictly_feasible(
                    x[moving] + step[moving], bounds[0][moving], bounds[1][moving], rstep=0
                )
            if own is not None:
                # A member whose trial crosses an own row it does not hold is scaled back.
                trial, scaled, predicted = inequality.member_back(
                    self.rows, own, system, x, trial, coordinates, predicted, held_rows, self.cut
                )
            step = trial - x
            step_norms = per_member(step)
            scaled_norms = per_member(scaled)
            passing = active.copy()
            for member in np.flatnonzero(active):
                single = x.copy()
                columns = owners == member
                single[columns] = trial[columns]
                if not self.valid(single):
                    passing[member] = False
                    work.predicate_rejections += 1
            combined = x.copy()
            chosen = below & passing[member_of]
            combined[chosen] = trial[chosen]
            if passing.any() and not self.valid(combined):
                work.predicate_rejections += int(passing.sum())
                passing[:] = False
            # (7) one request at the combined trial over the lanes of the members it
            # moves; a failed lane rejects its member
            rho = np.full((count,), -np.inf)
            failed = np.zeros((count,), dtype=bool)
            trial_costs = np.full((count,), np.inf)
            evaluated = bool(passing.any())
            if evaluated:
                before = work.lane_rungs.copy()
                trial_results = self.polish_request(combined, passing)
                # A failed lane of weight zero rejects no member's step: logged, ignored.
                trial_failures, ignored = route.weigh(route.failures(trial_results))
                work.record(attempt, ignored, inner=True)
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung at the inner trial is solved again
                    # at the current inner point at its new rung, so that its member's gain
                    # ratio compares costs of the same rung.
                    lane_costs, _ = self.resolve(x, raised, before, lane_costs)
                    costs = self.member_costs(lane_costs, x)
                for failure in trial_failures:
                    failed[self.lane_member[route.position[failure.lane]]] = True
                work.record(attempt, trial_failures, inner=True)
                trial_costs = self.member_costs(
                    self.lane_costs_of(trial_results, lane_costs), combined
                )
                gains = passing & ~failed & (predicted > 0.0)
                # Every member's gain ratio; relaxed by the noise the outer trials
                # measured when the inner noise_relaxed_ratio is set.
                rho[gains] = _gain_ratio(
                    costs[gains] - trial_costs[gains],
                    predicted[gains],
                    noise=noise,
                    relative_to=self.member_scales(costs[gains]),
                    relaxed=inner.noise_relaxed_ratio,
                )
            # (8) every member's acceptance, radius and lock, as the joint attempt's
            accepted = active & (rho >= float(inner.eta))
            step_radii = radii.copy()
            for member in np.flatnonzero(active):
                radii[member] = _next_radius(
                    float(radii[member]),
                    float(rho[member]),
                    float(scaled_norms[member]),
                    lock=int(locks[member]),
                    radius_max=inner.radius_max,
                    computed=bool(evaluated and passing[member] and not failed[member]),
                )
                if accepted[member]:
                    if locks[member]:
                        locks[member] -= 1
                elif failed[member]:
                    locks[member] = 2
            iterations[active] += 1
            work.inner_iterations[active] += 1
            if start:
                work.inner_start_iterations[active] += 1
            work.inner_rejections[active & ~accepted] += 1
            taken = below & accepted[member_of]
            previous = x.copy()
            x[taken] = trial[taken]
            # A member's step and cost tests read its trial whenever the trial's cost
            # was computed, accepted or rejected: the step within the step tolerance,
            # or the cost change and the decrease its model predicted both within the
            # cost tolerance (for an accepted trial the cost change alone). A trial the
            # predicate rejected, or whose lanes did not all solve, tests neither. Its
            # run of noise-dominated rejected trials, as the outer loop's: a trial whose
            # cost was computed extends or ends it, a failed one ends it, one the
            # predicate rejected leaves it; three in a row end its iterations.
            for member in np.flatnonzero(active):
                computed = bool(
                    evaluated
                    and passing[member]
                    and not failed[member]
                    and np.isfinite(trial_costs[member])
                )
                last = accepted_radii[member]
                noisy = _noise_dominated(
                    evaluated=computed,
                    accepted=bool(accepted[member]),
                    step_radius=float(step_radii[member]),
                    accepted_radius=None if np.isnan(last) else float(last),
                    cost=float(costs[member]),
                    trial_cost=float(trial_costs[member]),
                    predicted=float(predicted[member]),
                    relative_to=float(self.member_scales(costs[member])),
                    limit=noise_limit,
                )
                if computed:
                    noise_runs[member] = noise_runs[member] + 1 if noisy else 0
                elif failed[member]:
                    noise_runs[member] = 0
                if accepted[member]:
                    accepted_radii[member] = step_radii[member]
                # The loop's own step and cost tests, at the member's point after the
                # trial and at the member's cost threshold.
                if step_small(
                    float(step_norms[member]),
                    x[owners == member],
                    tolerances.step_tol,
                    tested=computed,
                ):
                    codes[member] = _STEP_TOLERANCE
                    active[member] = False
                elif cost_small(
                    accepted=bool(accepted[member]),
                    evaluated=computed,
                    cost_change=float(abs(trial_costs[member] - costs[member])),
                    predicted=float(predicted[member]),
                    threshold=cost_threshold(
                        costs[member], tolerances.cost_tol, tolerances.cost_atol
                    ),
                ):
                    codes[member] = _COST_TOLERANCE
                    active[member] = False
                elif noise_runs[member] >= _NOISE_RUN:
                    codes[member] = _EVALUATION_NOISE
                    active[member] = False
            floor = active & (radii <= tiny)
            floored |= floor
            finish(floor, _RADIUS)
            k += 1
            if not taken.any():
                continue
            if self.polish_pass == "frozen" and not trial_failures:
                # The frozen pass gave the moved members' gradient at their new point on
                # the matrix their last blocks request kept: their nodes take it, and the
                # exit's request of every lane gives the system where the polish ends.
                at_trial = self.lane_costs_of(trial_results, lane_costs)
                lane_costs = np.where(accepted[self.lane_member], at_trial, lane_costs)
                frozen = route.frozen_system(system, trial_results, combined)
                cost = self.cost_at(lane_costs, x)
                system = _spliced(system, frozen, self.cut, accepted, x, cost)
                spliced = True
                continue
            # (1) of the next iteration: the system at the members' new point. The last
            # iteration requests every lane there, and so does one whose moved members
            # hold every lane; an earlier one the lanes of the members that moved.
            every = k == int(limit) or not active.any() or bool(np.all(accepted[self.lane_member]))
            base = _Point(previous, system, lane_costs)
            x, lane_costs, system, partial = self.settle(
                x, base, accepted, every, radii, retreats, attempt, codes, active
            )
            if partial is not None:
                spliced = partial
        if spliced:
            # The exit's request of every lane where the polish ended, whose root block and
            # levels above the cut the next attempt reads; a member whose lanes fail there
            # retreats toward where the polish started.
            changed = np.bincount(
                member_of[below], weights=(x != start.point)[below], minlength=count
            )
            settled = None
            if not (kept and np.all(changed == 0)):
                x, lane_costs, system, settled = self.settle(
                    x, start, changed > 0, True, radii, retreats, attempt, codes, active
                )
            if kept and settled is None:
                # The polish ends where it started, whose system holds an earlier point's
                # matrix: the joint mode's blocks request there.
                system, lane_costs = self.blocks(x, start=False)
        if int(limit) == 0:
            codes[~held] = _SKIPPED
        else:
            codes[active] = _AT_MAX_STEPS
            work.inner_unconverged[active | floored] += 1
        return _Polish(
            x,
            system,
            lane_costs,
            scale,
            radii,
            locks,
            thresholds,
            iterations,
            codes,
            retreats,
            stationarity,
            first_cost,
        )
