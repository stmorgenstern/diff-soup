"""The nested trust-region fit: an outer step on the shared parameters, each member's own parameters polished inside it."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np

from diff_soup._distributed import assert_decision
from diff_soup.optimization.inference.fitting._trial import trial_is_valid
from diff_soup.optimization.inference.fitting.result import (
    OptResult,
)
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._joint_fit import _JointFit
from diff_soup.optimization.inference.hierarchical._lane_route import (
    _LABEL,
    _Failure,
    _LaneRoute,
    _listed,
)
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _full_step,
    _noise_trial_limit,
    _radius_floor,
    _scale_candidate,
    _subproblem,
    _trial_point,
    _trial_valid,
)
from diff_soup.optimization.inference.hierarchical._verdicts import (
    NONFINITE_BLOCKS,
    _stationarity_limit,
)
from diff_soup.optimization.inference.hierarchical._work import _record
from diff_soup.optimization.solvers._loop import (
    _NOISE_RUN,
    _TINY,
    LoopState,
    _after_attempt,
    _floor_level,
    _floor_sigma,
    _gain_ratio,
    _judge,
    _median,
    _next_radius,
    _noise_dominated,
    _saved_count,
    _stop_after,
    cost_scale,
    cost_small,
    cost_threshold,
    loop_tolerances,
    step_small,
)

if TYPE_CHECKING:
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion


# ---------------------------------------------------------------------------
# The nested attempt
# ---------------------------------------------------------------------------


# Why a member's polish ended at an attempt: the codes of the record's ``inner_stop``.
_NOT_POLISHED = 0  # no polish: the outer trial was not evaluated, or the member holds no data
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

# The polish's iterations per member when the nested solver is given no inner solver.
_DEFAULT_POLISH = 2
# A polish whose gain is below this fraction of the outer step's predicted decrease on
# _SMALL_GAINS accepted attempts in a row is switched off until an outer rejection.
_POLISH_GAIN = 1e-3
_SMALL_GAINS = 2


class _InnerResult(NamedTuple):
    """An inner solve at one outer point: where it ended, its system, and the inner state.

    ``failures`` holds the failed lanes of a member that no retreat solved at
    an outer trial, whose member cannot be solved there; ``system`` is then
    ``None`` and ``point`` the last point tried. ``stationarity`` holds every
    member's stationarity number at ``point``, ``decreases`` the decrease its
    model predicts for its full step there, ``lane_costs`` the lane costs
    there, ``codes`` why every member's iterations ended (the codes of
    ``inner_stop``) and ``first_cost`` the cost at the first request (all
    ``None`` when the solve ended at a failure). ``retreats`` holds every
    member's retreats in the solve. ``full_steps`` holds every member's
    full-step norm over its cost at ``point`` (``member_full_steps``) and
    ``floor_terms`` the lanes' terms of the gradient floor's bound there
    (both ``None`` when the solve ended at a failure).
    """

    point: np.ndarray
    system: Any
    scale: Any
    radii: np.ndarray
    locks: np.ndarray
    thresholds: np.ndarray
    iterations: np.ndarray
    failures: tuple
    stationarity: Any = None
    lane_costs: Any = None
    decreases: Any = None
    codes: Any = None
    first_cost: Any = None
    retreats: Any = None
    full_steps: Any = None
    floor_terms: Any = None


def _member_numbers(decreases: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Every member's predicted decrease over its cost scale ``scales``; ``nan`` stays ``nan``."""
    with np.errstate(all="ignore"):
        return np.asarray(decreases, dtype=np.float64) / np.asarray(scales, dtype=np.float64)


def _finite_blocks(system: Any) -> bool:
    """Whether every block of a system's matrix and its gradient are finite."""
    arrays = [system.root] + list(system.diagonal[1:])
    arrays += [entry for row in system.cross[1:] for entry in row]
    arrays += [system.root_gradient] + list(system.gradient[1:])
    return all(bool(np.all(np.isfinite(np.asarray(entry)))) for entry in arrays)


class _NestedFit(_JointFit):
    """Two loops cut at one level: the levels above it, and every member's subtree below.

    The outer trust region steps on the Schur complement of the inner blocks
    into the outer blocks, with the outer gradient corrected for the gradient
    the members leave, the inner gradients folded through the inner factors;
    its gradient test (with ``grad_tol`` set) reads that corrected gradient.
    At an outer trial the members move by their response to the outer step,
    each bounded by its radius, then polish: every member of the cut level
    iterates on its own subtree with the outer coordinates held, all members
    in lockstep (one ``blocks`` and one ``value`` request over every lane per
    iteration), each with its own radius, growth lock and acceptance, for at
    most the polish's number of iterations. A member whose lane fails at its
    responded or polished point retreats toward its last point where its
    lanes solved. The trial cost is the cost after the polish. The members'
    radii and locks carry from one attempt to the next.
    """

    subproblem = "level-nested"

    def __init__(self, solver: TrustRegion, route: _LaneRoute, callbacks: tuple):
        super().__init__(solver, route, callbacks)
        self.inner = solver if solver.inner is None else solver.inner
        # The polish's largest number of iterations per member and attempt.
        self.polish_steps = _DEFAULT_POLISH if solver.inner is None else int(solver.inner.max_steps)
        self.cut = route.names.index(solver.nested_at) + 1
        counts, widths = route.tables.counts, route.tables.widths
        self.prefix = int(widths[0]) + sum(
            int(counts[level - 1]) * int(widths[level]) for level in range(1, self.cut)
        )
        self.members = tuple(route.members[self.cut - 1])
        self.lane_member = np.asarray(route.nodes[:, self.cut - 1], dtype=np.int64)
        # The member of the cut level of every joint column; set by the fit's first ``blocks``.
        self.owners: Any = None
        regularizer = route.view.regularization
        rows = regularizer.residual(jax.device_put(route.problem.x0, route.device))
        self.regularized = int(np.shape(rows)[0]) > 0
        self.work.nest(self.members)

    def nested_entries(self, values: dict) -> dict:
        """The mode, the cut level, every member's inner radius, lock and threshold, the switch.

        ``polish_off`` is 1 while the polish is switched off, and
        ``polish_small_gains`` the number of accepted attempts in a row whose
        polish made an iteration and gained too little.
        """
        return {
            "mode": np.asarray("nested"),
            "nested_at": np.asarray(str(self.solver.nested_at)),
            "inner_radius": np.array(values["inner_radius"], dtype=np.float64),
            "inner_lock": np.array(values["inner_lock"], dtype=np.int64),
            "inner_grad_threshold": np.array(values["inner_grad_threshold"], dtype=np.float64),
            "polish_off": np.asarray(int(values["polish_off"]), dtype=np.int64),
            "polish_small_gains": np.asarray(values["polish_small_gains"], dtype=np.int64),
        }

    def member_above(
        self, members: Any, noise: float, tolerance: Any = None, floors: Any = None
    ) -> Any:
        """The member whose model predicts the most beyond the members' limit, or ``None``.

        The limit is ``tolerance``, or for ``None`` the members' yardstick
        ``max(inner cost_tol, 3 * noise)``, ``noise`` the evaluation noise the
        outer trials measured; ``floors``, every member's gradient floor in
        its stationarity number's units, raises each member's limit to its
        floor (``None`` raises none). A member whose number is not finite is
        beyond it. Returns ``(level, label, number, limit)``.
        """
        if members is None:
            return None
        numbers = np.asarray(members, dtype=np.float64)
        limit = _stationarity_limit(tolerance, self.inner_tolerances.cost_tol, noise)
        limits = limit if floors is None else np.fmax(limit, np.asarray(floors, dtype=np.float64))
        beyond = np.flatnonzero(~(numbers <= limits))
        if not beyond.size:
            return None
        worst = beyond[np.argmax(np.where(np.isfinite(numbers[beyond]), numbers[beyond], np.inf))]
        if floors is not None:
            limit = float(limits[worst])
        return (self.solver.nested_at, self.members[int(worst)], float(numbers[worst]), limit)

    def members_stationary(self, members: Any, noise: float, floors: Any = None) -> bool:
        """Whether every member's stationarity number is within the inner stationarity limit.

        The limit is the inner ``stationarity_tol``, or for ``None`` the
        members' yardstick ``max(inner cost_tol, 3 * noise)``: the limit each
        member's own stationarity test reads in the polish; ``floors`` raises
        each member's limit to its gradient floor, as ``member_above``.
        """
        return (
            self.member_above(members, noise, self.inner_tolerances.stationarity_tol, floors)
            is None
        )

    def record(
        self,
        rho,
        lam,
        accepted,
        trial,
        failures,
        iterations,
        codes,
        *,
        retreats=None,
        capped=None,
        norms=None,
        owners=None,
        **attempt,
    ) -> dict:
        """The attempt just made, with every member's polish, retreats and response in it.

        Per member: its polish iterations, the code of why its polish ended,
        its retreats (``retreats``), whether its response was bounded by its
        radius (``capped``) and its response's scaled norm before the bound
        (``norms``); the last three are 0, ``False`` and ``nan`` when not
        given (the start's record). ``owners`` adds the split of the cost by
        member (``owner_split``) when given.
        """
        count = len(self.members)
        entries = _record(rho, lam, accepted, trial, failures, self.route.depth, **attempt)
        if owners is not None:
            entries.update(owners)
        entries["inner_iterations"] = np.array(iterations, dtype=np.int64)
        entries["inner_stop"] = np.array(codes, dtype=np.int64)
        entries["inner_retreats"] = (
            np.zeros((count,), dtype=np.int64)
            if retreats is None
            else np.array(retreats, dtype=np.int64)
        )
        entries["response_capped"] = (
            np.zeros((count,), dtype=bool) if capped is None else np.array(capped, dtype=bool)
        )
        entries["response_norms"] = (
            np.full((count,), np.nan) if norms is None else np.array(norms, dtype=np.float64)
        )
        return entries

    def valid(self, point: np.ndarray) -> bool:
        """The trial predicate at ``point``; a predicate that raises rejects it."""
        return _trial_valid(self.route.view, point)

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

    def owner_split(
        self, system: Any, point: np.ndarray, lane_costs: np.ndarray, solved: Any
    ) -> dict:
        """The attempt's cost split by member of the cut level, read off without changing the fit.

        ``owner_costs`` holds every member's cost at the current ``point``
        (``member_costs``) and ``owner_shared_cost`` the rest of the cost there
        (``shared_cost``); ``owner_trial_costs`` and
        ``owner_trial_shared_cost`` the same at the trial after its polish
        (``nan`` when the trial was not evaluated or a member's lanes did not
        solve there); ``owner_predicted`` every member's part of the decrease
        the Gauss-Newton model at ``point`` predicts for the whole move to the
        trial (its subtree's terms and their coupling to the outer step,
        ``model_split``) and ``owner_shared_predicted`` the outer coordinates'
        own terms; together they sum to the model's decrease for that move.
        """
        count = len(self.members)
        entries = {
            "owner_costs": np.asarray(self.member_costs(lane_costs, point), dtype=np.float64),
            "owner_shared_cost": np.asarray(self.shared_cost(point), dtype=np.float64),
            "owner_trial_costs": np.full((count,), np.nan),
            "owner_trial_shared_cost": np.asarray(np.nan),
            "owner_predicted": np.full((count,), np.nan),
            "owner_shared_predicted": np.asarray(np.nan),
        }
        if solved is None:
            return entries
        delta = np.asarray(solved.point, dtype=np.float64) - np.asarray(point, dtype=np.float64)
        members, rest = elimination.model_split(system, delta, self.cut)
        entries["owner_predicted"] = np.asarray(members, dtype=np.float64)
        entries["owner_shared_predicted"] = np.asarray(rest, dtype=np.float64)
        if not solved.failures and solved.lane_costs is not None:
            entries["owner_trial_costs"] = np.asarray(
                self.member_costs(solved.lane_costs, solved.point), dtype=np.float64
            )
            entries["owner_trial_shared_cost"] = np.asarray(
                self.shared_cost(solved.point), dtype=np.float64
            )
        return entries

    def member_decrease(
        self, system: Any, scale: Any, zero: tuple, addition: Any = None
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

    def member_stationarity(
        self, system: Any, scale: Any, zero: tuple, costs: np.ndarray, addition: Any = None
    ) -> np.ndarray:
        """Every member's stationarity number: ``member_decrease`` over its cost scale."""
        return _member_numbers(
            self.member_decrease(system, scale, zero, addition), self.member_scales(costs)
        )

    def member_scales(self, costs: Any) -> Any:
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

    def members_at(self, system: Any, scale: Any, point: np.ndarray, lane_costs: Any) -> tuple:
        """Every member's predicted decrease, stationarity number and full step at ``point``.

        From one solve at zero damping of every subtree, in the subproblem's
        coordinates at ``point`` with ``scale``. Returns ``(decreases,
        numbers, full_steps)``, ``full_steps`` as ``member_full_steps`` gives
        them.
        """
        gradient, _ = self.route.vectors(system)
        coordinates = _subproblem(point, gradient, scale, self.route.bounds)
        zero = elimination.inner_solve_scaled(
            system, coordinates.scale, self.cut, addition=coordinates.addition
        )
        decreases = self.member_decrease(system, coordinates.scale, zero, coordinates.addition)
        costs = self.member_costs(lane_costs, point)
        numbers = _member_numbers(decreases, self.member_scales(costs))
        return decreases, numbers, self.member_full_steps(zero, costs)

    def full_stationarity(
        self,
        system: Any,
        reduced: Any,
        point: np.ndarray,
        scale: np.ndarray,
        relative_to: float,
        decreases: np.ndarray,
        *,
        outer: Any = None,
        zero: Any = None,
    ) -> tuple:
        """The stationarity number of the full system at ``point``, the solves, the outer step.

        The decrease the full Gauss-Newton model predicts for its full step is
        the reduced model's for its full step plus every member's for its own
        (``decreases``, the outer coordinates held), since the reduced system
        is the Schur complement of the members' blocks; the number is that sum
        over ``relative_to``, the cost scale at the point (``cost_scale``), the
        joint mode's number at the same point.
        Under bounds the outer coordinates take the Coleman-Li coordinates of
        the outer part of the full gradient, as the joint mode's do. ``outer``
        and ``zero`` are the outer subproblem's coordinates and its solve at
        zero damping, already made: they are used when the coordinates agree.
        Returns ``(number, solves, step)``, ``step`` the reduced model's full
        step (``_FullStep``), which the gradient floor reads.
        """
        prefix = self.prefix
        bounds = self.route.bounds
        outer_gradient, _ = elimination.level_vectors(reduced)
        if bounds is None:
            coordinates = (
                _subproblem(point[:prefix], outer_gradient, scale[:prefix], None)
                if outer is None
                else outer
            )
        else:
            gradient, _ = elimination.level_vectors(system)
            coordinates = _subproblem(
                point[:prefix],
                gradient[:prefix],
                scale[:prefix],
                (bounds[0][:prefix], bounds[1][:prefix]),
            )
        solves = 0
        agree = (
            outer is not None
            and zero is not None
            and np.array_equal(coordinates.scale, outer.scale)
            and (
                (coordinates.addition is None and outer.addition is None)
                or (
                    coordinates.addition is not None
                    and outer.addition is not None
                    and np.array_equal(coordinates.addition, outer.addition)
                )
            )
        )
        if not agree:
            zero = elimination.solve_scaled(reduced, coordinates.scale, 0.0, coordinates.addition)
            solves = 1
        step = _full_step(reduced, outer_gradient, coordinates.scale, zero, coordinates.addition)
        total = step.decrease + float(np.sum(np.asarray(decreases, dtype=np.float64)))
        return total / relative_to, solves, step

    def member_quadratic(self, system: Any, coordinates: Any, member: int) -> Any:
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
        system: Any,
        x: np.ndarray,
        gradient: np.ndarray,
        coordinates: Any,
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

    def respond(
        self,
        system: Any,
        x: np.ndarray,
        scale: np.ndarray,
        multiplier: float,
        outer_step: Any,
        radii: np.ndarray,
    ) -> tuple:
        """The trial of an outer step: the inner coordinates moved by their response to it.

        The response is the inner part of the minimizer of the damped model
        with the outer step held, at the outer step's multiplier, bounded per
        member by its radius in ``radii``: a member whose response's scaled
        length exceeds its radius takes its own trust-region step of that
        radius on its subtree's model at the gradient the response answers
        (``_level_elimination.inner_bounded_response``). Under bounds it is
        solved in the Coleman-Li coordinates of the inner coordinates at
        ``x``, with their diagonal addition; every member whose bounded
        response leaves the box takes ``_bounds.select_step``'s choice on its
        scaled subtree model with the outer step held (its gradient the inner
        gradient that the response answers, its reach the bounded response's
        scaled length), and the trial is kept strictly inside the box.

        Returns ``(trial, norms, capped, solves)``: the trial point, every
        member's response norm before the bound, whether it was bounded, and
        the damped solves the bound made.
        """
        bounds = self.route.bounds
        if bounds is None:
            step, _, _, norms, capped, solves = elimination.inner_bounded_response(
                system, scale, multiplier, outer_step, radii
            )
            return x + step, norms, capped, solves
        lower, upper = bounds
        prefix = self.prefix
        gradient, _ = elimination.level_vectors(system)
        coordinates = box.coleman_li_scaling(x, gradient, lower, upper, scale)
        inner_scale = np.array(coordinates.scale, copy=True)
        inner_scale[:prefix] = scale[:prefix]
        addition = np.array(coordinates.addition, copy=True)
        addition[:prefix] = 0.0
        coordinates = coordinates._replace(scale=inner_scale, addition=addition)
        response, z, shifted, norms, capped, solves = elimination.inner_bounded_response(
            system, inner_scale, multiplier, outer_step, radii, addition=addition
        )
        with np.errstate(all="ignore"):
            d = 1.0 / inner_scale
        trial = x + response
        for member in range(len(self.members)):
            columns = np.flatnonzero(self.owners == member)
            if box.in_bounds(trial[columns], lower[columns], upper[columns]):
                continue
            member_step, _, _ = box.select_step(
                x[columns],
                self.member_quadratic(system, coordinates, member),
                shifted[columns],
                response[columns],
                z[columns],
                d[columns],
                float(np.linalg.norm(z[columns])),
                lower[columns],
                upper[columns],
                box.step_back_ratio(gradient[columns], coordinates.v[columns]),
            )
            trial[columns] = x[columns] + member_step
        return box.make_strictly_feasible(trial, lower, upper, rstep=0), norms, capped, solves

    def reduce(self, system: Any) -> Any:
        """The outer system: the inner blocks eliminated; a singular inner problem is refused.

        Under bounds the inner blocks carry the diagonal ``|g_i| / v_i`` of the
        Coleman-Li model wherever the gradient points at a finite side, so the
        outer model sees an inner coordinate held at its side as held.
        """
        bounds = self.route.bounds
        source = system
        if bounds is not None:
            gradient, _ = elimination.level_vectors(system)
            point = np.asarray(system.point)
            v, dv = box.CL_scaling_vector(point, gradient, bounds[0], bounds[1])
            inner = np.arange(point.shape[0]) >= self.prefix
            with np.errstate(all="ignore"):
                term = np.where(inner & (dv != 0), gradient * dv / v, 0.0)
            source = elimination.scaled_system(system, np.ones_like(point), term)
        reduced = elimination.reduce_to(source, self.cut)
        if _finite_blocks(reduced):
            return reduced
        singular = np.flatnonzero(elimination.inner_singular(system, self.cut))
        level = self.solver.nested_at
        if singular.size:
            member = self.members[int(singular[0])]
            raise ValueError(
                "the nested mode needs the Gauss-Newton matrix of every inner problem to be "
                f"nonsingular, and member {member!r} of level {level!r} has a singular one at "
                "this point; use nested_at=None."
            )
        raise ValueError(
            f"the Gauss-Newton matrix of the levels above {level!r}, with the inner problems "
            "eliminated, is not finite at this point; use nested_at=None."
        )

    def blocks(self, point: np.ndarray, *, start: bool) -> tuple:
        """The joint fit's ``blocks``; the first one also sets ``owners``, once.

        ``owners`` is the member of the cut level whose subtree holds each
        joint column (``-1`` above it). The fit's first request is this one,
        at the start or the restored point, before every method that reads it.
        """
        system, lane_costs = super().blocks(point, start=start)
        if self.owners is None:
            self.owners = elimination.node_of_columns(system, self.cut)
        return system, lane_costs

    def nonfinite_block_failures(self, results: Any, system: Any) -> tuple:
        """The lanes of every member whose Gauss-Newton blocks or gradient are not finite at a finite cost.

        Each of the member's lanes is a failure of kind ``NONFINITE_BLOCKS``: its solve reached
        the end at a finite cost and its sensitivities are not finite there.
        """
        broken = elimination.nonfinite_members(system, self.cut)
        if not broken.any():
            return ()
        route = self.route
        costs = np.asarray(results.cost, dtype=np.float64)[0]
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

    def retreating_blocks(
        self,
        point: np.ndarray,
        anchor: np.ndarray,
        radii: np.ndarray,
        retreats: np.ndarray,
        attempt: int,
    ) -> tuple:
        """The ``blocks`` request at an inner point of an outer trial, the members retreating.

        ``point`` is the responded point or a point the polish reached, and
        ``anchor`` holds every member's coordinates at its last point where
        its lanes solved: the current point for the responded point, the
        point before the polish's last iteration otherwise. A member with a
        lane that no rung solves at ``point`` retreats along its move from
        ``anchor``: at its ``j``-th retreat its subtree's coordinates are
        ``anchor + (point - anchor) / 2**j``, its radius is quartered, and
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

        Returns ``(point, results, system, failures)``: the point where every
        lane solved, the results and system there, and ``()``; or, when a
        member cannot be solved at any retreat, the last point tried,
        ``None``, ``None`` and the failures of the last request.
        """
        route, work = self.route, self.work
        allowed = int(self.solver.inner_retreats)
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        anchor = np.asarray(anchor)
        failing = np.array(point, copy=True)
        point = np.array(point, copy=True)
        tried = np.zeros((len(self.members),), dtype=np.int64)
        while True:
            results = self.evaluate("blocks", point)
            failures, ignored = route.weigh(route.failures(results))
            work.record(attempt, ignored, inner=True)
            if not failures:
                system = route.system(results, point)
                # A member whose lanes solved at a finite cost but whose Gauss-Newton blocks
                # are not finite fails like a member whose lanes did not solve.
                failures, ignored = route.weigh(self.nonfinite_block_failures(results, system))
                work.record(attempt, ignored, inner=True)
            if not failures:
                work.held = system.held
                # The lanes' terms of the gradient floor's bound at this point.
                self.floor.record(results)
                return point, results, system, ()
            pending = self.failed_members(failures)
            unlogged = failures
            while pending.size:
                if np.any(tried[pending] >= allowed):
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
                lanes = tuple(
                    route.ids[position]
                    for position in np.flatnonzero(np.isin(self.lane_member, pending))
                )
                if not self.valid(point):
                    work.predicate_rejections += 1
                    continue
                retried = self.evaluate("value", point, lanes)
                failures, ignored = route.weigh(route.failures(retried))
                work.record(attempt, ignored, inner=True)
                unlogged = failures
                pending = self.failed_members(failures) if failures else pending[:0]

    def inner_solve(
        self,
        point: np.ndarray,
        *,
        origin: Any = None,
        scale: Any,
        radii: np.ndarray,
        locks: np.ndarray,
        thresholds: np.ndarray,
        attempt: int,
        start: bool,
        limit: int,
        outer_predicted: Any = None,
        noise: float = 0.0,
        level: Any = None,
    ) -> _InnerResult:
        """Every member of the cut level iterates on its subtree at the outer point of ``point``.

        The members start from their coordinates in ``point`` with the radii
        ``radii`` and the growth locks ``locks`` they carry, and make at most
        ``limit`` lockstep iterations; ``limit=0`` evaluates ``point`` alone.
        ``start`` marks the start point of a fresh fit, where a lane that fails
        at any ``blocks`` request is an error. At an outer trial a member whose
        lane fails at the first request retreats along its response toward its
        coordinates in ``origin`` (the current point; ``None`` takes
        ``point``), and one whose lane fails at the ``blocks`` request after
        an iteration retreats along its last step (``retreating_blocks``); a
        member that no retreat solves ends the solve with its failures. A
        member is done when its
        stationarity number is at most the inner ``stationarity_tol``, when its
        gradient is within the inner ``grad_tol`` (when that is set), when the
        decrease its model predicts for its full step is at most
        ``inner_fraction`` times ``outer_predicted`` (the decrease the outer
        model predicted for the outer step; ``None`` tests nothing), by its
        step and cost tests, when three of its rejected trials in a row within
        these iterations are dominated by the noise of the cost's evaluation,
        or when its radius falls to the smallest positive float. The
        tolerance's default reads ``noise``, the evaluation noise the outer
        trials measured, and so do the members' gain ratios under the inner
        ``noise_relaxed_ratio``. With ``level``, the gradient floor the outer
        trials measured (``None`` before the first sample), a member is also
        done when its stationarity number is at most its own floor, ``level``
        times its full step's scaled norm over its cost scale
        ``max(|cost_n|, cost_atol / cost_tol)`` under the inner tolerances.
        """
        solver, inner, route, work = self.solver, self.inner, self.route, self.work
        tolerances = self.inner_tolerances
        count = len(self.members)
        x = np.array(point, copy=True)
        scale = None if scale is None else np.array(scale, copy=True)
        radii = np.array(radii, dtype=np.float64, copy=True)
        locks = np.array(locks, dtype=np.int64, copy=True)
        thresholds = np.array(thresholds, dtype=np.float64, copy=True)
        iterations = np.zeros((count,), dtype=np.int64)
        bounds = route.bounds
        fixed = np.asarray(route.fixed, dtype=np.int64)
        tiny = float(np.finfo(float).tiny)
        member_limit = _stationarity_limit(tolerances.stationarity_tol, tolerances.cost_tol, noise)
        noise_limit = _noise_trial_limit(inner, route)
        accuracy = (
            None if outer_predicted is None else solver.inner_fraction * float(outer_predicted)
        )

        # Every member's retreats in this solve.
        retreats = np.zeros((count,), dtype=np.int64)
        # The lane costs at the current inner point, with those of a consistency re-solve.
        if start:
            system, lane_costs = self.blocks(x, start=True)
        else:
            # The responded point: a member whose lane fails there retreats along its
            # response toward its coordinates at the current point.
            x, results, system, failures = self.retreating_blocks(
                x, x if origin is None else origin, radii, retreats, attempt
            )
            if failures:
                return _InnerResult(
                    x,
                    None,
                    scale,
                    radii,
                    locks,
                    thresholds,
                    iterations,
                    failures,
                    retreats=retreats,
                )
            lane_costs = np.array(results.cost, dtype=np.float64)[0]
        first_cost = float(system.cost)
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
            zero = elimination.inner_solve_scaled(
                system, coordinates.scale, self.cut, addition=coordinates.addition
            )
            decreases = self.member_decrease(system, coordinates.scale, zero, coordinates.addition)
            stationarity = _member_numbers(decreases, self.member_scales(costs))
            full_steps = self.member_full_steps(zero, costs)
            # A member whose model predicts no further decrease the evaluation can
            # show is done, as one whose gradient is within its threshold, and as
            # one that has less to gain than inner_fraction of the outer step; once
            # the gradient floor is measured, so is one whose model predicts no more
            # than its derivatives resolve.
            limits = member_limit if level is None else np.fmax(member_limit, level * full_steps)
            finish(active & (stationarity <= limits), _STATIONARITY)
            if accuracy is not None:
                finish(active & (decreases <= accuracy), _ACCURACY)
            # (4) the exit always follows a blocks request at the current point
            if not active.any() or k == int(limit):
                # the exit makes no step; its one solve is the one above
                work.solves += 1
                break
            # (5) every active member's step; (6) bounds, fixed coordinates, predicate
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
            # (7) one value request at the combined trial; a failed lane rejects its member
            rho = np.full((count,), -np.inf)
            failed = np.zeros((count,), dtype=bool)
            trial_costs = np.full((count,), np.inf)
            evaluated = bool(passing.any())
            if evaluated:
                before = work.lane_rungs.copy()
                trial_results = self.evaluate("value", combined)
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
                    np.asarray(trial_results.cost, dtype=np.float64)[0], combined
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
            if evaluated:
                # (1) of the next iteration: the members' new point, every lane solved
                # there. At an outer trial a member whose lane fails there retreats along
                # its last step; at the start point the failure is an error.
                if start:
                    system, lane_costs = self.blocks(x, start=False)
                else:
                    x, results, system, failures = self.retreating_blocks(
                        x, previous, radii, retreats, attempt
                    )
                    if failures:
                        return _InnerResult(
                            x,
                            None,
                            scale,
                            radii,
                            locks,
                            thresholds,
                            iterations,
                            failures,
                            retreats=retreats,
                        )
                    lane_costs = np.array(results.cost, dtype=np.float64)[0]
        if int(limit) == 0:
            codes[~held] = _SKIPPED
        else:
            codes[active] = _AT_MAX_STEPS
            work.inner_unconverged[active | floored] += 1
        return _InnerResult(
            x,
            system,
            scale,
            radii,
            locks,
            thresholds,
            iterations,
            (),
            stationarity,
            lane_costs,
            decreases,
            codes,
            first_cost,
            retreats,
            full_steps,
            self.floor.latest,
        )

    def run(self, state: Any = None) -> OptResult:
        """Fit from the start point, or continue the fit whose ``state`` is given."""
        solver, inner, route, work = self.solver, self.inner, self.route, self.work
        dtype = np.asarray(route.problem.x0).dtype
        x0 = np.array(route.problem.x0, dtype=dtype)
        bounds = route.bounds
        count = len(self.members)
        prefix = self.prefix
        # grad_tol=None tests no gradient, in the outer loop and in the inner fits it sets.
        tolerances = loop_tolerances(
            dtype,
            grad_tol=solver.grad_tol,
            step_tol=solver.step_tol,
            cost_tol=solver.cost_tol,
            cost_atol=solver.cost_atol,
            stationarity_tol=solver.stationarity_tol,
            label=_LABEL,
        )
        grad_tolerance, step_tol, cost_tol, cost_atol, _ = tolerances
        # The inner fits' tolerances, read by the polish and the members' tests: the
        # outer ones when no inner solver is given.
        self.inner_tolerances = (
            tolerances
            if solver.inner is None
            else loop_tolerances(
                dtype,
                grad_tol=inner.grad_tol,
                step_tol=inner.step_tol,
                cost_tol=inner.cost_tol,
                cost_atol=inner.cost_atol,
                stationarity_tol=inner.stationarity_tol,
                label=_LABEL,
            )
        )
        outer_fixed = np.asarray([index for index in route.fixed if index < prefix], dtype=np.int64)
        outer_bounds = None if bounds is None else (bounds[0][:prefix], bounds[1][:prefix])
        n_constraints = int(route.constraint_matrix.shape[0])
        route.problem.prepare(modes=("value", "blocks"))
        start_iterations = np.zeros((count,), dtype=np.int64)
        start_codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
        # The polish's switch: off after two accepted attempts in a row whose polish
        # gained too little, on again after an outer rejection or a failed lane.
        polish_off, small_gains = False, 0
        if state is None:
            x = x0.copy()
            if bounds is not None:
                # Iterates stay strictly inside the box: a start on or beyond a side
                # moves to the next float inside it.
                x = box.make_strictly_feasible(x, bounds[0], bounds[1], rstep=0)
            if not trial_is_valid(route.view, x):
                raise ValueError("TrustRegion initial point failed the trial predicate.")
            radii = np.full((count,), float(inner.radius_init), dtype=np.float64)
            locks = np.zeros((count,), dtype=np.int64)
            thresholds = np.full((count,), np.nan)
            if solver.inner_start_iterations > 0:
                # At most inner_start_iterations lockstep iterations at the start point,
                # within the members' radii, before the first outer attempt.
                solved = self.inner_solve(
                    x,
                    scale=None,
                    radii=radii,
                    locks=locks,
                    thresholds=thresholds,
                    attempt=-1,
                    start=True,
                    limit=solver.inner_start_iterations,
                )
                x, system, scale = solved.point, solved.system, solved.scale
                radii, locks, thresholds = solved.radii, solved.locks, solved.thresholds
                start_iterations, start_codes = solved.iterations, solved.codes
                member_numbers, member_decreases = solved.stationarity, solved.decreases
                member_steps = solved.full_steps
                lane_costs = solved.lane_costs
            else:
                # No inner fit at the start point: the first outer attempt reads the
                # full system there.
                system, lane_costs = self.blocks(x, start=True)
                scale = _scale_candidate(
                    system, solver.x_scale, elimination.level_vectors(system)[1]
                )
                member_decreases, member_numbers, member_steps = self.members_at(
                    system, scale, x, lane_costs
                )
                work.solves += 1
        else:
            # The saved point is the start or an accepted point, after its polish; its
            # blocks are evaluated again. The members' radii and locks, and the
            # polish's switch, are those the saved state carries.
            x = np.array(state["point"], dtype=dtype)
            work.restore(state)
            work.resume_blocks += 1
            system, lane_costs = self.blocks(x, start=False)
            scale = np.array(state["scale"], dtype=dtype)
            radii = np.array(state["inner_radius"], dtype=np.float64)
            locks = np.array(state["inner_lock"], dtype=np.int64)
            thresholds = np.array(state["inner_grad_threshold"], dtype=np.float64)
            # A state saved without the switch resumes with the polish on.
            polish_off = bool(_saved_count(state, "polish_off"))
            small_gains = _saved_count(state, "polish_small_gains")
        cost = float(system.cost)
        reduced = self.reduce(system)
        # Every program an attempt runs is compiled here, before the loop: the outer
        # solve, quadratic and vectors of the reduced system, the inner response to
        # an outer step, and the inner solves.
        elimination.level_vectors(reduced)
        elimination.solve_scaled(reduced, np.ones((prefix,), dtype=dtype), 0.0)
        elimination.level_quadratic(reduced, np.zeros((prefix,), dtype=dtype))
        elimination.inner_response(
            system, scale, 0.0, np.zeros((prefix,), dtype=dtype), addition=np.zeros_like(scale)
        )
        elimination.inner_programs(system, self.cut)
        if state is not None:
            # The members' predicted decreases and stationarity numbers at the
            # restored point, from the restored scale, as its polish ended there.
            member_decreases, member_numbers, member_steps = self.members_at(
                system, scale, x, lane_costs
            )

        for callback in self.callbacks:
            callback.on_start(route.n, route.view.n_residuals, solver.max_steps)
        self.started = time.perf_counter()
        if state is None:
            # A fresh fit's first record holds the retries of the start's requests.
            record = self.record(
                float("nan"),
                float("nan"),
                False,
                x,
                (),
                start_iterations,
                start_codes,
                **work.attempt_records(),
            )
            loop = LoopState.start(
                point=x, cost=cost, scale=scale, radius=float(solver.radius_init), record=record
            )
        else:
            # A resumed fit's first record holds the retries of its first attempt.
            loop = LoopState.restore(state, point=x, cost=cost, scale=scale)
        # The largest relative predicted decrease of a trial counted as noise-dominated;
        # None bounds none.
        noise_limit = _noise_trial_limit(solver, route)
        start_attempt = loop.attempt
        attempt = start_attempt
        for attempt in range(start_attempt, int(solver.max_steps)):
            # The outer gradient with the inner gradients folded in: the gradient of
            # the model the outer step minimizes, and to first order the gradient of
            # the cost after the inner fits.
            outer_gradient, _ = elimination.level_vectors(reduced)
            x_outer = loop.point[:prefix]
            grad_norm = float(np.linalg.norm(outer_gradient))
            # The outer subproblem's coordinates, in the outer part of the scale.
            outer = _subproblem(x_outer, outer_gradient, loop.scale[:prefix], outer_bounds)
            stop_gradient = float(
                np.linalg.norm(_bounded_gradient(outer_gradient, outer, outer_bounds))
            )
            loop.cost_history.append(loop.cost)
            loop.grad_history.append(grad_norm)
            if grad_tolerance is not None:
                if loop.grad_threshold is None:
                    loop.grad_threshold = grad_tolerance.threshold(stop_gradient)
                if not n_constraints and stop_gradient <= loop.grad_threshold:
                    stop = "gradient_tolerance"
                    break

            # The outer step on the reduced system, in the outer part of the scale. Its
            # solve at lambda = 0 is the first solve of the step's multiplier search;
            # with every member's decrease at zero damping it gives the stationarity
            # number of the full system at the current point.
            zero = elimination.solve_scaled(reduced, outer.scale, 0.0, outer.addition)
            # Every relative number of the attempt is relative to the cost scale here.
            relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
            stationarity, solves, full_step = self.full_stationarity(
                system,
                reduced,
                loop.point,
                loop.scale,
                relative_to,
                member_decreases,
                outer=outer,
                zero=zero,
            )
            work.solves += solves
            # The gradient's accuracy floor here, as in the joint mode; the bound reads
            # every coordinate, the outer ones in the outer step's coordinates. Every
            # member's floor is the fit's floor along the member's own full step.
            bound = self.floor.bound(np.concatenate((full_step.scale, loop.scale[prefix:])))
            level = _floor_level(loop.gradient_floor_samples, bound)
            floors = None if level is None else level * member_steps
            # The full model predicts no further decrease the evaluation can show here,
            # and neither does any member's model, against the members' yardstick.
            if stationarity <= _stationarity_limit(
                solver.stationarity_tol, cost_tol, _median(loop.noise_dominated_cost_change)
            ) and self.members_stationary(
                member_numbers, _median(loop.noise_dominated_cost_change), floors
            ):
                work.solves += 1
                stop = "stationarity"
                break
            # The full model predicts no further decrease its derivatives resolve here,
            # and neither does any member's model, against its floor.
            if (
                level is not None
                and stationarity <= _floor_sigma(level, full_step.norm, relative_to)
                and self.member_above(
                    member_numbers, _median(loop.noise_dominated_cost_change), floors=floors
                )
                is None
            ):
                work.solves += 1
                stop = "derivative_accuracy"
                break
            radius_floor = _radius_floor(level, full_step.norm, outer_gradient, outer.scale)
            current_members = np.array(member_numbers, dtype=np.float64)
            step_radius = loop.radius
            trial_outer, scaled_step_norm, predicted, multiplier, solves = _trial_point(
                reduced,
                x_outer,
                outer_gradient,
                outer,
                loop.radius,
                first=zero,
                fixed=outer_fixed,
                bounds=outer_bounds,
                equality=n_constraints > 0,
            )
            work.solves += solves
            step = trial_outer - x_outer
            # The trial's inner coordinates are the current ones moved by their
            # response to the outer step: the inner part of the minimizer of the
            # damped model with the outer step held, at the outer step's multiplier,
            # each member's bounded by its carried radius. The polish at the trial
            # starts there.
            trial, response_norms, response_capped, solves = self.respond(
                system, loop.point, loop.scale, multiplier, step, radii
            )
            work.solves += solves
            trial[:prefix] = trial_outer
            predicted_value = float(predicted)
            # The step test reads the trial's whole move, the outer step and the
            # members' response together, as the joint mode's reads its whole step.
            step_norm = float(np.linalg.norm(trial - loop.point))

            failures: tuple = ()
            failed_owners = None
            rejection_reason = None
            invalid = False
            solved = None
            iterations = np.zeros((count,), dtype=np.int64)
            codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
            retreats = np.zeros((count,), dtype=np.int64)
            logged = len(work.log)
            trial_ok = self.valid(trial)
            # A trial move within the step tolerance ends the fit at the current point
            # before the trial is evaluated.
            step_stop = step_small(step_norm, loop.point, step_tol, tested=trial_ok)
            if not trial_ok:
                work.predicate_rejections += 1
                trial_cost = float("inf")
                rejection_reason = "the trial predicate rejected the point"
            elif step_stop:
                trial_cost = float("inf")
                rejection_reason = "the step is within the step tolerance"
            else:
                # The polish at the trial, from the responded inner coordinates, with
                # the members' carried radii and locks; switched off, the responded
                # point alone is evaluated.
                before = work.lane_rungs.copy()
                if polish_off:
                    work.polish_skipped += 1
                solved = self.inner_solve(
                    trial,
                    origin=loop.point,
                    scale=loop.scale,
                    radii=radii,
                    locks=locks,
                    thresholds=thresholds,
                    attempt=attempt,
                    start=False,
                    limit=0 if polish_off else self.polish_steps,
                    outer_predicted=predicted_value,
                    noise=_median(loop.noise_dominated_cost_change),
                    level=level,
                )
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung in the polish at the trial is
                    # solved again at the current point at its new rung, so that the outer
                    # gain ratio compares costs of the same rung; the outer model there
                    # stays the earlier rung's.
                    lane_costs, change = self.resolve(loop.point, raised, before, lane_costs)
                    loop.cost += change
                iterations = solved.iterations
                trial = solved.point
                retreats = solved.retreats
                if solved.codes is not None:
                    codes = solved.codes
                if solved.failures:
                    invalid = True
                    failures = solved.failures
                    work.invalid_trials += 1
                    work.record(attempt, failures, inner=False)
                    trial_cost = float("inf")
                    failed_owners = route.owners(failures)
                    unsolvable = []
                    for failure in failures:
                        member = self.members[self.lane_member[route.position[failure.lane]]]
                        if member not in unsolvable:
                            unsolvable.append(member)
                    rejection_reason = (
                        f"the inner problem of {solver.nested_at} member(s) "
                        f"{_listed(unsolvable)} has no point where its lanes solve at the trial"
                    )
                    if solver.inner_retreats:
                        rejection_reason += (
                            f", after {solver.inner_retreats} retreat(s) toward the last point "
                            "where its lanes solved"
                        )
                else:
                    trial_cost = float(solved.system.cost)

            # The cost split by member at the current point and at the trial: read
            # off the current system and the polish's result, nothing of the fit changes.
            owners = self.owner_split(system, loop.point, lane_costs, solved)
            # A step within the step tolerance leaves the radius as it is.
            rho, accepted, loop.radius, reason = _judge(
                cost=loop.cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                valid=trial_ok,
                invalid=invalid,
                scaled_norm=scaled_step_norm,
                radius=loop.radius,
                lock=loop.growth_lock,
                noise_window=loop.noise_dominated_cost_change,
                eta=float(solver.eta),
                radius_max=solver.radius_max,
                relaxed=solver.noise_relaxed_ratio,
                relative_to=cost_scale(loop.cost, cost_tol, cost_atol),
                hold=step_stop,
                radius_floor=radius_floor,
            )
            assert_decision(accepted)
            rejection_reason = rejection_reason or reason

            previous_cost = loop.cost
            if accepted:
                # The point, its system and the members' state become the trial's; a
                # rejected trial leaves the members' radii and locks as they were
                # before its polish.
                loop.point, system = solved.point, solved.system
                loop.scale, radii, locks, thresholds = (
                    solved.scale,
                    solved.radii,
                    solved.locks,
                    solved.thresholds,
                )
                loop.cost = float(system.cost)
                lane_costs = solved.lane_costs
                reduced = self.reduce(system)
                member_numbers, member_decreases = solved.stationarity, solved.decreases
                member_steps = solved.full_steps
                # The lanes' terms of the gradient floor's bound move with the point.
                self.floor.current = solved.floor_terms
                loop.accepted += 1
            # The polish's switch: an accepted attempt whose polish made an iteration and
            # gained less than _POLISH_GAIN of the outer step's predicted decrease (the
            # cost at the responded point minus the cost after the polish) counts toward
            # switching it off, one that gained more resets the count, and one whose
            # polish made no iteration does neither; an outer rejection or a lane that
            # no rung solves turns it on.
            if not accepted or len(work.log) > logged:
                polish_off, small_gains = False, 0
            elif not polish_off and int(np.max(solved.iterations, initial=0)) > 0:
                gain = float(solved.first_cost) - float(solved.system.cost)
                small_gains = small_gains + 1 if gain < _POLISH_GAIN * predicted_value else 0
                if small_gains >= _SMALL_GAINS:
                    polish_off, small_gains = True, 0
            work.attempted(accepted)
            loop.radius_history.append(loop.radius)
            # An outer trial is evaluated when its cost was computed after its polish (a
            # step within the step tolerance is not); the outer trials alone measure the
            # evaluation noise and make the runs, the polish's trials do not enter.
            after = _after_attempt(
                loop,
                accepted=accepted,
                valid=trial_ok,
                invalid=invalid,
                cost=previous_cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                step_radius=step_radius,
                cost_tol=cost_tol,
                cost_atol=cost_atol,
                noise_limit=noise_limit,
                scaled_norm=scaled_step_norm,
                floor_bound=bound,
            )
            loop = dataclasses.replace(loop, **after.trials._asdict())
            loop.record = self.record(
                rho,
                float(multiplier),
                accepted,
                trial,
                failures,
                iterations,
                codes,
                retreats=retreats,
                capped=response_capped,
                norms=response_norms,
                owners=owners,
                **work.attempt_records(),
            )
            loop.attempt = attempt + 1
            info = self.step_info(
                loop,
                attempt,
                grad_norm=grad_norm,
                accepted=accepted,
                delta_norm=step_norm,
                rho=rho,
                lam=float(multiplier),
                rejection_reason=rejection_reason,
                failed_owners=failed_owners,
                state=self.state(
                    loop,
                    inner_radius=radii,
                    inner_lock=locks,
                    inner_grad_threshold=thresholds,
                    polish_off=polish_off,
                    polish_small_gains=small_gains,
                ),
                stationarity=stationarity,
                member_stationarity=current_members,
            )
            stop = _stop_after(
                after,
                requested=self.notify(info),
                max_invalid=int(solver.max_consecutive_invalid_trials),
                # The step test read the current point before the trial was evaluated.
                step_small=step_stop,
                accepted_count=loop.accepted,
                target_steps=solver.target_accepted_steps,
                # The budget counts the consistency re-solves and the inner fits' too.
                evaluations=work.blocks + work.value,
                max_evaluations=solver.max_evaluations,
            )
            if stop is not None:
                break
        else:
            attempt = max(int(solver.max_steps), start_attempt) - 1
            stop = "max_steps"

        gradient, _ = elimination.level_vectors(system)
        grad_norm = float(np.linalg.norm(gradient))
        projected_grad_norm = float(
            np.linalg.norm(
                _bounded_gradient(
                    gradient, _subproblem(loop.point, gradient, loop.scale, bounds), bounds
                )
            )
        )
        # The full system's stationarity number at the end point, from the members'
        # decreases there (those of the polish that ended there) and the outer part.
        end_relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
        end_stationarity, _, end_step = self.full_stationarity(
            system, reduced, loop.point, loop.scale, end_relative_to, member_decreases
        )
        end_floor = self.floor_end(
            loop,
            end_step,
            np.concatenate((end_step.scale, loop.scale[prefix:])),
            cost_tol,
            end_relative_to,
            member_steps,
            system=system,
        )
        return self.outcome(
            loop,
            stop,
            attempt=attempt,
            x0=x0,
            cost_tol=cost_tol,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            stationarity=end_stationarity,
            members=member_numbers,
            floor=end_floor,
            inner_radius=radii,
            inner_lock=locks,
            inner_grad_threshold=thresholds,
            polish_off=polish_off,
            polish_small_gains=small_gains,
        )
