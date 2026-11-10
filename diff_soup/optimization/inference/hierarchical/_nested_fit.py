"""The nested trust-region fit: the joint step, then each member's own parameters polished with the outer ones held."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.engine._batched import cut_by_member
from diff_soup.optimization.inference.fitting.result import OptResult, StepInfo
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical import _polish_program as polish
from diff_soup.optimization.inference.hierarchical._joint_fit import _JointFit
from diff_soup.optimization.inference.hierarchical._lane_route import (
    _LABEL,
    _Failure,
    _LaneRoute,
)
from diff_soup.optimization.inference.hierarchical._level_inner import model_split
from diff_soup.optimization.inference.hierarchical._level_system import nonfinite_members
from diff_soup.optimization.inference.hierarchical._step_program import as_host, coordinates
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _noise_trial_limit,
    _trial_valid,
    _trials_valid,
)
from diff_soup.optimization.inference.hierarchical._verdicts import (
    NONFINITE_BLOCKS,
    _failures_text,
    _stationarity_limit,
)
from diff_soup.optimization.solvers._loop import LoopState, _saved_count, loop_tolerances
from diff_soup.optimization.solvers._loop_judge import _TINY, _median

if TYPE_CHECKING:
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion


# ---------------------------------------------------------------------------
# The polish
# ---------------------------------------------------------------------------


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
        frozen = polishing and route.problem._offers("frozen")
        self.polish_pass = "frozen" if frozen else "value"
        if frozen:
            # Every blocks request keeps its lanes' Jacobians, which the frozen pass reads;
            # both programs are compiled with the others before the fit's first request.
            self.modes = ("value", "blocks", "frozen")
        self.cut = route.names.index(solver.nested_at) + 1
        self.members = tuple(route.members[self.cut - 1])
        self.lane_member = np.asarray(route.nodes[:, self.cut - 1], dtype=np.int64)
        if frozen and getattr(route.problem, "_lanes", None) is not None:
            # On a CPU every call of the blocks and frozen programs holds one member's lanes,
            # so a frozen request of the polish runs the calls of the members still in it only.
            cut_by_member(route.problem._lanes, dict(zip(route.ids, self.lane_member.tolist())))
        # The member of the cut level of every joint column; set by the fit's first ``blocks``.
        self.owners: np.ndarray | None = None
        # The member of the cut level of every regularizer block; set at its first cost.
        self.block_members: np.ndarray | None = None
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
        # The polish's settings, layout and proposing program, made at its first iteration.
        self.polish_settings: polish.PolishSettings | None = None
        self.polish_layout: polish.Layout | None = None
        self.polish_program: Any = None

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
            polished = polished._replace(codes=np.full_like(polished.codes, polish.NOT_POLISHED))
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
            noise=float(_median(loop.noise_dominated_cost_change)),
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
            entries["inner_stop"] = np.full((count,), polish.NOT_POLISHED, dtype=np.int64)
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
        member, blocks = self.block_split(point)
        weights = self.route.block_weights
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            blocks = np.where(weights == 0.0, 0.0, weights * blocks)
        owned = member >= 0
        return costs + np.bincount(member[owned], weights=blocks[owned], minlength=costs.size)

    def shared_cost(self, point: np.ndarray) -> float:
        """The regularizer rows of no member: the blocks on coordinates above the cut level only."""
        if not self.regularized:
            return 0.0
        member, blocks = self.block_split(point)
        weights = self.route.block_weights
        if weights is not None:
            blocks = np.asarray(weights, dtype=np.float64) * blocks
        return float(np.sum(blocks[member < 0]))

    def block_split(self, point: np.ndarray) -> tuple:
        """``(member, costs)``: every regularizer block's member of the cut level and its cost.

        A block's member is that of the first of its columns below the cut
        level, ``-1`` for a block on coordinates above it only; its cost the
        unweighted ``0.5 |r_b|^2`` at ``point`` (``_LaneRoute.block_costs``).
        """
        columns, costs = self.route.block_costs(point)
        if self.block_members is None or len(self.block_members) != len(columns):
            members = [self.owners[np.asarray(item, dtype=np.int64)] for item in columns]
            self.block_members = np.asarray(
                [int(item[item >= 0][0]) if np.any(item >= 0) else -1 for item in members],
                dtype=np.int64,
            )
        return self.block_members, np.asarray(costs, dtype=np.float64)

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
        the retreats it has left. A member keeps its retreated point only when
        its cost there is at or below its cost at ``anchor``; otherwise it goes
        back to its coordinates in ``anchor``, where the ``blocks`` request
        evaluates its lanes again. A retreated point that the trial predicate
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
        count = len(self.members)
        tried = np.zeros((count,), dtype=np.int64)
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
                failed = self.failed_members(failures) if failures else pending[:0]
                # A member whose lanes solve at its retreated point keeps it only at a cost
                # there at or below its cost at the anchor; else it goes back to the anchor.
                solved = np.isin(np.arange(count), pending) & ~np.isin(np.arange(count), failed)
                if solved.any():
                    here = self.member_costs(self.lane_costs_of(retried, base.lane_costs), point)
                    there = self.member_costs(base.lane_costs, anchor)
                    worse = solved & ~(here <= there)
                    back = below & worse[member_of]
                    point[back] = anchor[back]
                pending = failed

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
        code ``polish.UNSOLVED`` in ``codes``, cleared in ``active``), and the
        request is made again for the others and for it, so that its lanes are
        evaluated again where it went back to and it keeps no Jacobian of the
        point it left; with no member that moved left, that request covers the
        members that went back alone, at ``base``.

        Returns ``(point, lane_costs, system, partial)``: ``partial`` whether
        the system holds nodes of a request of some lanes only, ``None`` when
        it is ``base``'s.
        """
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        moved = np.array(moved, dtype=bool, copy=True)
        returned = np.zeros_like(moved)
        point = np.array(point, copy=True)
        while True:
            reached, lane_costs, system, failures = self.retreating_blocks(
                point, base, None if every else moved | returned, radii, retreats, attempt
            )
            if not failures:
                return reached, lane_costs, system, not every
            unsolved = self.failed_members(failures)
            codes[unsolved] = polish.UNSOLVED
            active[unsolved] = False
            progress = bool(moved[unsolved].any())
            moved[unsolved] = False
            returned[unsolved] = True
            if not progress or not moved.any():
                start = np.array(base.point, copy=True)
                lane_costs, system, again = self.request_blocks(start, attempt, base, returned)
                if again:
                    self.work.record(attempt, again, inner=True)
                    return start, base.lane_costs, base.system, None
                return start, lane_costs, system, True
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

    def polish_setup(self) -> None:
        """The polish's settings, the members' layout and its proposing program, once per fit.

        Made at the first polish iteration, after the fit's first ``blocks`` request has set
        ``owners``. Without inequality rows the proposing program is shared by every fit;
        with them it holds the rows and is kept on them (``RowSet.program``), shared by the
        fits of one study under the same rows.
        """
        if self.polish_settings is not None:
            return
        inner, tolerances, route = self.inner, self.inner_tolerances, self.route
        dtype = np.dtype(np.asarray(route.problem.x0).dtype)
        self.polish_settings = polish.PolishSettings(
            level=self.cut,
            count=len(self.members),
            fixed=tuple(int(index) for index in route.fixed),
            grad_test=tolerances.grad_tolerance is not None,
            step_tol=float(tolerances.step_tol),
            cost_tol=float(tolerances.cost_tol),
            cost_atol=float(tolerances.cost_atol),
            eta=float(inner.eta),
            radius_max=float(inner.radius_max),
            relaxed=bool(inner.noise_relaxed_ratio),
            noise_limit=_noise_trial_limit(inner, route),
            eps=float(np.finfo(dtype).eps),
            tiny=float(np.finfo(float).tiny),
        )
        owners = np.asarray(self.owners, dtype=np.int64)
        self.polish_layout = jax.device_put(polish.layout(owners, len(self.members)), route.device)
        if self.rows is None:
            self.polish_program = polish.propose_program
        else:
            rows = self.rows
            self.polish_program = rows.program(
                ("polish", self.cut, owners.tobytes()),
                lambda: jax.jit(
                    functools.partial(polish.propose, rows=rows, owners=owners),
                    static_argnames=("settings",),
                ),
            )

    def member_thresholds(
        self,
        system: elimination.LevelSystem,
        x: np.ndarray,
        scale: np.ndarray,
        thresholds: np.ndarray,
        held: np.ndarray,
    ) -> None:
        """Set the gradient threshold of every member not held that has none, in place.

        From its bounded gradient's norm at ``x`` in the iteration's scale, as the joint
        fit sets its threshold from its first attempt's gradient.
        """
        gradient, _, subproblem = coordinates(
            system, self.solver.x_scale, x, scale, self.route.bounds
        )
        bounded = _bounded_gradient(gradient, subproblem, self.route.bounds)
        owners = self.owners
        below = owners >= 0
        norms = np.sqrt(
            np.bincount(owners[below], weights=bounded[below] ** 2, minlength=len(self.members))
        )
        grad_tolerance = self.inner_tolerances.grad_tolerance
        for member in np.flatnonzero(np.isnan(thresholds) & ~held):
            thresholds[member] = grad_tolerance.threshold(norms[member])

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
        alone. An iteration is two compiled programs (``_polish_program``):
        one proposes every member's tests and trial, one judges the trials;
        between them one request at the members' combined trial over the lanes
        of the members it moves (``polish_request``). After a frozen pass the
        members that moved take its gradient on the kept matrix; after a
        ``value`` request (or a frozen pass with a failed lane) one ``blocks``
        request at their new point over the lanes of the members that moved
        updates their nodes of the system (``request_blocks``): of every lane
        when they hold every lane or it is the last iteration. A polish that
        ends on a system of some lanes' requests makes the ``blocks`` request
        of every lane where it ended, a member whose lanes fail there
        retreating toward where the polish started. ``start`` marks the
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
        solver, route, work = self.solver, self.route, self.work
        tolerances = self.inner_tolerances
        self.polish_setup()
        settings, where = self.polish_settings, self.polish_layout
        count = len(self.members)
        x = np.array(point, copy=True)
        lane_costs = np.array(lane_costs, dtype=np.float64, copy=True)
        if scale is None:
            scale = np.full(x.shape, np.nan, dtype=x.dtype)
        scale = np.array(scale, dtype=x.dtype, copy=True)
        thresholds = np.array(self.thresholds, dtype=np.float64, copy=True)
        iterations = np.zeros((count,), dtype=np.int64)
        retreats = np.zeros((count,), dtype=np.int64)
        member_limit = _stationarity_limit(tolerances.stationarity_tol, tolerances.cost_tol, noise)
        accuracy = (
            np.nan if outer_predicted is None else solver.inner_fraction * float(outer_predicted)
        )
        floor_level = np.nan if level is None else float(level)
        first_cost = float(system.cost)
        # Where the polish started, and whether the system holds nodes of a request of
        # some lanes only.
        start_point = _Point(np.array(point, copy=True), system, lane_costs)
        spliced = kept
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        held = np.array(system.held_mask[self.cut], dtype=bool)
        active = ~held
        floored = np.zeros((count,), dtype=bool)
        state = polish.MemberState(
            radii=np.array(self.radii, dtype=np.float64, copy=True),
            locks=np.array(self.locks, dtype=np.int64, copy=True),
            noise_runs=np.zeros((count,), dtype=np.int64),
            accepted_radii=np.full((count,), np.nan),
            codes=np.full((count,), polish.NOT_POLISHED, dtype=np.int64),
        )
        k = 0
        while True:
            costs = self.member_costs(lane_costs, x)
            if tolerances.grad_tolerance is not None and np.any(np.isnan(thresholds) & ~held):
                self.member_thresholds(system, x, scale, thresholds, held)
            before = polish.MemberBefore(
                x,
                scale,
                costs,
                state.radii,
                thresholds,
                active,
                state.codes,
                np.asarray(floor_level, dtype=np.float64),
                np.asarray(accuracy, dtype=np.float64),
                np.asarray(member_limit, dtype=np.float64),
                np.asarray(k < int(limit)),
            )
            proposal = _writable(
                self.polish_program(
                    system, solver.x_scale, route.bounds, where, before, settings=settings
                )
            )
            scale, active = proposal.scale, proposal.active
            state = state._replace(codes=proposal.codes)
            stationarity = proposal.stationarity
            work.solves += int(proposal.zero_solves)
            if not bool(proposal.stepped):
                # the exit makes no step; its one solve is the one at zero damping
                work.solves += 1
                break
            work.solves += int(proposal.solves)
            if not active.any():
                break
            trial = proposal.trial
            # The predicate at every member's own trial, the others' values held, and then at
            # the combined trial of the members it passed.
            own = below[None, :] & (member_of[None, :] == np.arange(count)[:, None])
            valid = _trials_valid(route.view, np.where(own, trial[None, :], x[None, :]))
            passing = active & valid
            work.predicate_rejections += int(np.sum(active & ~valid))
            combined = x.copy()
            chosen = below & passing[member_of]
            combined[chosen] = trial[chosen]
            if passing.any() and not self.valid(combined):
                work.predicate_rejections += int(passing.sum())
                passing[:] = False
            # One request at the combined trial over the lanes of the members it moves; a
            # failed lane rejects its member.
            failed = np.zeros((count,), dtype=bool)
            trial_costs = np.full((count,), np.inf)
            evaluated = bool(passing.any())
            trial_failures: tuple = ()
            if evaluated:
                before_rungs = work.lane_rungs.copy()
                trial_results = self.polish_request(combined, passing)
                # A failed lane of weight zero rejects no member's step: logged, ignored.
                trial_failures, ignored = route.weigh(route.failures(trial_results))
                work.record(attempt, ignored, inner=True)
                raised = np.flatnonzero(work.lane_rungs > before_rungs)
                if raised.size:
                    # A lane that moved to a higher rung at the inner trial is solved again
                    # at the current inner point at its new rung, so that its member's gain
                    # ratio compares costs of the same rung.
                    lane_costs, _ = self.resolve(x, raised, before_rungs, lane_costs)
                    costs = self.member_costs(lane_costs, x)
                if trial_failures:
                    failed[self.failed_members(trial_failures)] = True
                work.record(attempt, trial_failures, inner=True)
                trial_costs = self.member_costs(
                    self.lane_costs_of(trial_results, lane_costs), combined
                )
            outcome = polish.MemberOutcome(
                active,
                passing,
                failed,
                np.asarray(evaluated),
                np.asarray(costs, dtype=np.float64),
                np.asarray(trial_costs, dtype=np.float64),
                proposal.predicted,
                proposal.scaled_norms,
                proposal.step_norms,
                x,
                trial,
                np.asarray(noise, dtype=np.float64),
            )
            accepted, x, state, after, floor = _writable(
                polish.judge_program(where, outcome, state, settings=settings)
            )
            iterations[active] += 1
            work.inner_iterations[active] += 1
            if start:
                work.inner_start_iterations[active] += 1
            work.inner_rejections[active & ~accepted] += 1
            previous = outcome.point
            active = after
            floored |= floor
            k += 1
            if not np.any(below & accepted[member_of]):
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
                x, base, accepted, every, state.radii, retreats, attempt, state.codes, active
            )
            if partial is not None:
                spliced = partial
        if spliced:
            # The exit's request of every lane where the polish ended, whose root block and
            # levels above the cut the next attempt reads; a member whose lanes fail there
            # retreats toward where the polish started.
            changed = np.bincount(
                member_of[below], weights=(x != start_point.point)[below], minlength=count
            )
            settled = None
            if not (kept and np.all(changed == 0)):
                x, lane_costs, system, settled = self.settle(
                    x,
                    start_point,
                    changed > 0,
                    True,
                    state.radii,
                    retreats,
                    attempt,
                    state.codes,
                    active,
                )
            if kept and settled is None:
                # The polish ends where it started, whose system holds an earlier point's
                # matrix: the joint mode's blocks request there.
                system, lane_costs = self.blocks(x, start=False)
        codes = state.codes
        if int(limit) == 0:
            codes[~held] = polish.SKIPPED
        else:
            codes[active] = polish.AT_MAX_STEPS
            work.inner_unconverged[active | floored] += 1
        return _Polish(
            x,
            system,
            lane_costs,
            scale,
            state.radii,
            state.locks,
            thresholds,
            iterations,
            codes,
            retreats,
            stationarity,
            first_cost,
        )


def _writable(value: Any) -> Any:
    """A program's output as host values the polish may change in place: every leaf a copy."""
    return jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), as_host(value))
