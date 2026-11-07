"""The joint trust-region fit: one step on every parameter of a lowered study."""

from __future__ import annotations

import dataclasses
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from diff_soup._distributed import any_process, assert_decision, is_primary
from diff_soup.optimization.inference.fitting._trial import trial_is_valid
from diff_soup.optimization.inference.fitting.callbacks import StopFit
from diff_soup.optimization.inference.fitting.result import (
    EvalCounts,
    OptResult,
    StepInfo,
    TerminationStatus,
    TrustRegionReport,
)
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._lane_route import _LABEL, _LaneRoute, _listed
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _FloorEnd,
    _full_step,
    _FullStep,
    _GradientFloor,
    _noise_ceiling,
    _noise_trial_limit,
    _radius_floor,
    _scale_candidate,
    _stationarity,
    _subproblem,
    _trial_point,
    _trial_valid,
)
from diff_soup.optimization.inference.hierarchical._verdicts import (
    _failures_text,
    _last_invalid,
    _stalled_message,
    _stationarity_limit,
    _stop_report,
    _verdict,
)
from diff_soup.optimization.inference.hierarchical._work import _record, _Work
from diff_soup.optimization.solvers._loop import (
    LoopState,
    _after_attempt,
    _floor_level,
    _floor_parameter_radius,
    _floor_sigma,
    _judge,
    _jump_level,
    _median,
    _stop_after,
    cost_scale,
    loop_tolerances,
    step_small,
)

if TYPE_CHECKING:
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion


# ---------------------------------------------------------------------------
# The joint attempt
# ---------------------------------------------------------------------------


class _JointFit:
    """One trust region over every coordinate of the levels."""

    # The subproblem the result's ``info`` names.
    subproblem = "level-exact"

    def __init__(self, solver: TrustRegion, route: _LaneRoute, callbacks: tuple):
        self.solver = solver
        self.route = route
        self.callbacks = tuple(solver.callbacks) + callbacks if is_primary() else ()
        self.work = _Work(route)
        # The mode whose matrix the attempts step on: the Gauss-Newton blocks, or the
        # exact Hessian under hessian="true".
        self.matrix = "hessian" if solver.hessian == "true" else "blocks"
        # The bound of the gradient's error at the current point, from the lanes'
        # tolerances and blocks.
        self.floor = _GradientFloor(solver, route)

    def evaluate(self, mode: str, point: np.ndarray, lanes: Any = None) -> Any:
        """One request at ``point``, every lane (or ``lanes``) at the rung it holds.

        A lane the evaluation solved at a higher rung holds that rung from then on.
        A request of the matrix's mode (``blocks``, or ``hessian`` under
        ``hessian="true"``) counts in the work's ``blocks``, any other in ``value``.
        """
        results = self.route.evaluate(mode, point, **self.work.request(lanes))
        self.work.hold(results)
        if mode == self.matrix:
            self.work.blocks += 1
        else:
            self.work.value += 1
        return results

    def resolve(
        self, point: np.ndarray, raised: np.ndarray, before: np.ndarray, lane_costs: np.ndarray
    ) -> tuple:
        """The consistency re-solve after the lanes ``raised`` moved to a higher rung at a trial.

        One ``value`` request at the current point ``point`` over those lanes,
        each at its new rung. Returns the current lane costs with theirs
        replaced by the re-solved ones, and the change of the weighted cost;
        a lane that does not solve there keeps its cost. Each lane's switch is
        recorded with its old rung ``before``, its new rung, and its cost at the
        current point before and after.
        """
        route, work = self.route, self.work
        positions = np.sort(np.asarray(raised, dtype=np.int64))
        lanes = tuple(route.ids[position] for position in positions)
        results = self.evaluate("value", point, lanes)
        work.rung_switch_resolves += 1
        failed = {failure.lane for failure in route.failures(results)}
        costs = np.asarray(results.cost, dtype=np.float64)[0]
        replaced = np.array(lane_costs, dtype=np.float64, copy=True)
        change = 0.0
        for column, position in enumerate(positions):
            old = float(replaced[position])
            new = float("nan") if lanes[column] in failed else float(costs[column])
            if np.isfinite(new):
                replaced[position] = new
                change += float(route.omega[position]) * (new - old)
            work.attempt_switches.append(
                (lanes[column], int(before[position]), int(work.lane_rungs[position]), old, new)
            )
        return replaced, change

    def blocks(self, point: np.ndarray, *, start: bool) -> tuple:
        """The system at an accepted point (or the start), every lane solved there.

        Every lane of nonzero weight, that is: a failed lane of weight zero is
        logged as ignored. Returns the system and the lane costs at ``point``.
        """
        route = self.route
        results = self.evaluate(self.matrix, point)
        failures, ignored = route.weigh(route.failures(results))
        if failures and start:
            raise ValueError(
                f"the initial solve failed for lane(s) {_failures_text(failures)}; every lane "
                "has to solve at the start point."
            )
        if failures:
            raise ValueError(
                f"lane(s) {_failures_text(failures)} failed at an accepted point in the "
                f"{self.matrix} mode; every lane has to solve at an accepted point."
            )
        self.work.record(self.work.attempts, ignored, inner=False)
        system = route.system(results, point)
        self.work.held = system.held
        # The point becomes the current one: the gradient floor's bound reads its lanes.
        self.floor.current = self.floor.record(results)
        return system, np.array(results.cost, dtype=np.float64)[0]

    def state(self, loop: LoopState, **nested: Any) -> MappingProxyType:
        """The state after an attempt, read-only host arrays; ``nested``: the nested mode's own."""
        route = self.route
        entries = {
            "solver": np.asarray(_LABEL),
            "mode": np.asarray("joint"),
            "nested_at": np.asarray(""),
            "fingerprint": np.asarray(route.fingerprint),
            "n": np.asarray(route.n, dtype=np.int64),
            **loop.entries(),
        }
        entries.update(self.nested_entries(nested))
        entries.update({f"work.{key}": value for key, value in self.work.mapping().items()})
        # What the lanes cost so far, and the plan they ran under; never where they ran.
        problem = route.problem
        ledger = getattr(problem, "_ledger", None)
        if ledger is not None:
            entries.update(ledger.entries())
        fingerprint = getattr(problem, "_plan_fingerprint", None)
        value = fingerprint() if callable(fingerprint) else None
        if value is not None:
            entries["ledger.plan_fingerprint"] = np.asarray(value)
        return MappingProxyType(entries)

    def nested_entries(self, values: dict) -> dict:
        """The entries of the state that only the nested mode has (none here)."""
        return {}

    def member_above(
        self, members: Any, noise: float, tolerance: Any = None, floors: Any = None
    ) -> Any:
        """The member whose model predicts the most beyond the members' yardstick (none here)."""
        return None

    def verdict(
        self,
        stop: str,
        *,
        sigma: float,
        cost_tol: float,
        lock: int,
        invalid_run: int,
        radius: float,
        windows: tuple,
        members: Any = None,
        floor: Any = None,
    ) -> tuple:
        """The status, the message and the ``info`` entries of the fit's end.

        ``stop`` is the event that ended the loop and ``sigma`` the stationarity
        number at the end point; ``windows`` holds the noise-dominated trials'
        cost changes and the rejected trials' cost changes and predicted
        decreases; ``members`` the members' stationarity numbers at the end
        point in the nested mode; ``floor`` the gradient floor there
        (``_FloorEnd``). A fit that converged to the accuracy of its
        derivatives says so in its message, with its numbers, and so does a
        fit that converged at the jumps of its computed cost.
        """
        noise_changes, rejected_changes, rejected_decreases = windows
        noise = _median(noise_changes)
        floors = None if floor is None else floor.members
        above = self.member_above(members, noise, floors=floors)
        floor_sigma = 0.0 if floor is None else floor.sigma
        # With a gradient floor, the jumps of the computed cost that every rejected trial
        # met, when they all did, are what the verdict reads beside the measured noise.
        ceiling = _noise_ceiling(self.route)
        jumps = None
        if self.floor.coefficients is not None:
            jumps = _jump_level(rejected_changes, rejected_decreases, ceiling)
        jump_yardstick = None if jumps is None else max(cost_tol, 3.0 * jumps)
        jump_above = None if jumps is None else self.member_above(members, jumps, floors=floors)
        yardstick = max(cost_tol, 3.0 * noise)
        status, stall_reason = _verdict(
            stop,
            sigma=sigma,
            yardstick=yardstick,
            lock=lock,
            members=above is None,
            floor_sigma=floor_sigma,
            floor_known=floor is not None and floor.level is not None,
            jump_yardstick=jump_yardstick,
            jump_members=jump_above is None,
        )
        at_jumps = (
            jump_yardstick is not None
            and status in (TerminationStatus.STEP_TOL, TerminationStatus.COST_TOL)
            and not (sigma <= yardstick and above is None)
        )
        failures = _last_invalid(self.work.log)
        stalled = None
        if status is TerminationStatus.STALLED:
            stalled = _stalled_message(
                stall_reason,
                stop,
                consecutive=invalid_run,
                failures=failures,
                radius=radius,
                sigma=sigma,
                noise=noise,
                ceiling=_noise_ceiling(self.route),
                member=above,
            )
        budget = None
        if stop == "max_evaluations":
            budget = (
                f"{self.work.blocks + self.work.value} evaluations reached "
                f"max_evaluations={self.solver.max_evaluations}"
            )
        message, report = _stop_report(
            status, rejected_changes, rejected_decreases, stalled, budget
        )
        if status is TerminationStatus.DERIVATIVE_ACCURACY:
            message = self.floor.message(sigma, floor_sigma)
        if at_jumps:
            message = (
                f"{status.value}: every rejected trial met a jump of the computed cost (median "
                f"{jumps:.2g}, within the noise {ceiling:.2g} the solver tolerance allows), and "
                f"the model's predicted decrease ({sigma:.2g}) is within {jump_yardstick:.2g}, "
                "the larger of cost_tol and three times that jump, all relative to the cost "
                "scale max(|cost|, cost_atol / cost_tol); no trial can show a decrease that "
                "small"
            )
            report = {"evaluation_jumps": jumps, "cost_floor": jump_yardstick}
        entries = {
            "stop": stop,
            "stall_reason": stall_reason,
            "consecutive_invalid_trials": int(invalid_run),
            "failed_lanes": tuple(
                (
                    failure.lane,
                    failure.kind,
                    float(failure.t_reached),
                    tuple(int(rung) for rung in failure.rungs),
                )
                for failure in failures
            ),
            **report,
        }
        return status, message, entries

    def step_info(self, loop: LoopState, attempt: int, **fields: Any) -> StepInfo:
        """The ``StepInfo`` of attempt ``attempt``: the loop's numbers, the work, and ``fields``."""
        work = self.work
        return StepInfo(
            step=attempt,
            cost=loop.cost,
            params=np.array(loop.point),
            wall_time=time.perf_counter() - self.started,
            radius=float(loop.radius),
            accepted_count=loop.accepted,
            attempt_count=attempt + 1,
            residual_evals=work.blocks + work.value,
            jacobian_evals=work.blocks,
            trial_cost_evals=work.value,
            statistics_evals=work.blocks,
            trial_predicate_rejections=work.predicate_rejections,
            invalid_pde_trials=work.invalid_trials,
            block_solves=work.solves,
            **fields,
        )

    def notify(self, info: StepInfo) -> bool:
        """Hand ``info`` to every callback; whether one asked, in any process, to stop the fit."""
        requested = False
        for callback in self.callbacks:
            try:
                callback.on_step(info)
            except StopFit:
                requested = True
        return any_process(requested)

    def outcome(
        self,
        loop: LoopState,
        stop: str,
        *,
        attempt: int,
        x0: np.ndarray,
        cost_tol: float,
        grad_norm: float,
        projected_grad_norm: float,
        stationarity: float,
        members: Any = None,
        floor: Any = None,
        **nested: Any,
    ) -> OptResult:
        """The fit's result, ended by ``stop`` after attempt ``attempt``, handed to the callbacks.

        ``stationarity`` is the stationarity number at the end point and
        ``members`` the members' numbers there in the nested mode; ``floor``
        the gradient floor there (``_FloorEnd``); ``nested`` holds the nested
        mode's own entries of the final state.
        """
        work, route = self.work, self.route
        loop.attempt = work.attempts
        state = self.state(loop, **nested)
        status, message, report = self.verdict(
            stop,
            sigma=stationarity,
            cost_tol=cost_tol,
            lock=loop.growth_lock,
            invalid_run=loop.consecutive_invalid_trials,
            radius=loop.radius,
            windows=(
                loop.noise_dominated_cost_change,
                loop.rejected_cost_change,
                loop.rejected_predicted_decrease,
            ),
            members=members,
            floor=floor,
        )
        evaluations = work.blocks + work.value
        result = OptResult(
            params=np.array(loop.point),
            initial_params=x0,
            cost=loop.cost,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            status=status,
            message=message,
            n_steps=attempt + 1,
            max_steps=int(self.solver.max_steps),
            n_accepted=loop.accepted,
            eval_counts=EvalCounts(
                residual_evals=evaluations,
                jacobian_evals=work.blocks,
                pde_solves=evaluations * len(route.ids),
                trial_cost_evals=work.value,
                statistics_evals=work.blocks,
                trial_predicate_rejections=work.predicate_rejections,
                invalid_pde_trials=work.invalid_trials,
                block_solves=work.solves,
            ),
            cost_history=np.asarray(loop.cost_history, dtype=np.float64),
            grad_history=np.asarray(loop.grad_history, dtype=np.float64),
            wall_time=time.perf_counter() - self.started,
            info={
                "failure_sentinel_absent": True,
                "hessian": "gauss-newton" if self.matrix == "blocks" else "true",
                "hessian_composition": (
                    None if self.matrix == "blocks" else "forward-over-reverse"
                ),
                "linear_solver": "cholesky",
                "subproblem": self.subproblem,
                "active_bounds": box.active_bounds(loop.point, route.bounds),
            },
            work=work.mapping(),
            state=state,
            trust_region=TrustRegionReport(
                **{
                    "radius_history": tuple(loop.radius_history),
                    "stationarity": stationarity,
                    "member_stationarity": (
                        None if members is None else tuple(float(value) for value in members)
                    ),
                    "evaluation_noise": _median(loop.noise_dominated_cost_change),
                    "evaluation_noise_trials": len(loop.noise_dominated_cost_change),
                    "evaluation_noise_ceiling": _noise_ceiling(route),
                    **({} if floor is None else floor.entries),
                    **report,
                }
            ),
        )
        for callback in self.callbacks:
            callback.on_end(result)
        return result

    def floor_end(
        self,
        loop: LoopState,
        step: _FullStep,
        scale: Any,
        cost_tol: float,
        relative_to: float,
        member_steps: Any = None,
        system: Any = None,
    ) -> _FloorEnd:
        """The gradient floor at the end point: what the verdict reads, and the ``info`` entries.

        ``step`` is the model's full step at the end point and ``scale`` the
        coordinates the bound reads there; ``relative_to`` is the cost scale
        there (``cost_scale``); ``member_steps`` holds every member's
        full-step norm over its cost scale in the nested mode; ``system``,
        when given, is the full system at the end point, whose dense matrix
        gives the floor's reach along its softest direction. Before the first
        floor sample the bound stands in for the floor in the entries, and no
        verdict reads it.
        """
        samples = loop.gradient_floor_samples
        bound = self.floor.bound(scale)
        level = _floor_level(samples, bound)
        reported = bound if level is None else level
        source = None
        if bound is not None:
            source = "a_posteriori" if samples and _median(samples) <= bound else "a_priori"
        radius = _floor_parameter_radius(reported, step.norm, step.length, step.decrease)
        box = None
        if reported is not None and system is not None:
            box = float("nan")
            if step.norm > 0.0 and step.length > 0.0:
                # The floor per unit parameter length along the full step, over the
                # smallest eigenvalue of the model's matrix: how far the minimizer can
                # move along the softest direction within the gradient's accuracy.
                matrix, _ = elimination.level_dense(system)
                smallest = float(np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))[0])
                if smallest > 0.0:
                    box = reported * step.norm / step.length / smallest
        entries = {
            "gradient_floor": (
                None if reported is None else float(_floor_sigma(reported, step.norm, relative_to))
            ),
            "gradient_floor_bound": (
                None if bound is None else float(_floor_sigma(bound, step.norm, relative_to))
            ),
            "gradient_floor_trials": len(samples),
            "cost_floor": max(float(cost_tol), 3.0 * _median(loop.noise_dominated_cost_change)),
            "floor_parameter_radius": None if radius is None else float(radius),
            "floor_parameter_box": None if box is None else float(box),
            "floor_source": source,
        }
        members = (
            None
            if level is None or member_steps is None
            else level * np.asarray(member_steps, dtype=np.float64)
        )
        return _FloorEnd(
            level, float(_floor_sigma(level, step.norm, relative_to)), members, entries
        )

    def run(self, state: Any = None) -> OptResult:
        """Fit from the start point, or continue the fit whose ``state`` is given."""
        solver, route, work = self.solver, self.route, self.work
        dtype = np.asarray(route.problem.x0).dtype
        x0 = np.array(route.problem.x0, dtype=dtype)
        bounds = route.bounds
        if state is None:
            x = x0.copy()
            if bounds is not None:
                # Iterates stay strictly inside the box: a start on or beyond a side
                # moves to the next float inside it.
                x = box.make_strictly_feasible(x, bounds[0], bounds[1], rstep=0)
            if not trial_is_valid(route.view, x):
                raise ValueError("TrustRegion initial point failed the trial predicate.")
        else:
            # The saved point is the start or an accepted point: every lane solved
            # there and the predicate passed. Its blocks are evaluated again below.
            x = np.array(state["point"], dtype=dtype)
            work.restore(state)
            work.resume_blocks += 1
        # grad_tol=None tests no gradient: the stationarity number is the test there.
        grad_tolerance, step_tol, cost_tol, cost_atol, _ = loop_tolerances(
            dtype,
            grad_tol=solver.grad_tol,
            step_tol=solver.step_tol,
            cost_tol=solver.cost_tol,
            cost_atol=solver.cost_atol,
            stationarity_tol=solver.stationarity_tol,
            label=_LABEL,
        )
        fixed = np.asarray(route.fixed, dtype=np.int64)
        n_constraints = int(route.constraint_matrix.shape[0])
        route.problem.prepare(modes=("value", self.matrix))
        system, lane_costs = self.blocks(x, start=state is None)
        cost = float(system.cost)
        # Every program the fit runs is compiled here, before the loop: the scale, the
        # attempt's program (the solve at lambda = 0 and the step of the initial radius,
        # kept as the first attempt's of a fresh fit), the quadratic, and the solve at
        # lambda = 0 alone that the end point's numbers take. The regularizer was
        # evaluated at the point by the system's blocks.
        gradient, diagonal = route.vectors(system)
        start = _subproblem(x, gradient, _scale_candidate(system, solver.x_scale, diagonal), bounds)
        first = route.solve_attempt(system, start, float(solver.radius_init))
        elimination.level_quadratic(system, np.zeros_like(x))
        elimination.solve_scaled(system, start.scale, 0.0, start.addition)

        for callback in self.callbacks:
            callback.on_start(route.n, route.view.n_residuals, solver.max_steps)
        self.started = time.perf_counter()
        if state is None:
            # A fresh fit's first record holds the retries of the start's request.
            record = _record(
                float("nan"), float("nan"), False, x, (), route.depth, **work.attempt_records()
            )
            loop = LoopState.start(
                point=x, cost=cost, scale=None, radius=float(solver.radius_init), record=record
            )
        else:
            # A resumed fit's first record holds the retries of its first attempt, which
            # makes its own lambda = 0 solve in the restored scale, as an uninterrupted fit.
            first = None
            loop = LoopState.restore(
                state, point=x, cost=cost, scale=np.array(state["scale"], dtype=dtype)
            )
        # The largest relative predicted decrease of a trial counted as noise-dominated;
        # None bounds none.
        noise_limit = _noise_trial_limit(solver, route)
        start_attempt = loop.attempt
        attempt = start_attempt
        for attempt in range(start_attempt, int(solver.max_steps)):
            gradient, diagonal = route.vectors(system)
            candidate = _scale_candidate(system, solver.x_scale, diagonal)
            loop.scale = candidate if loop.scale is None else np.maximum(loop.scale, candidate)
            grad_norm = float(np.linalg.norm(gradient))
            # The subproblem's coordinates: the scale, under bounds shaped by the
            # distance to the sides the gradient points at.
            coordinates = _subproblem(loop.point, gradient, loop.scale, bounds)
            stop_gradient = float(np.linalg.norm(_bounded_gradient(gradient, coordinates, bounds)))
            loop.cost_history.append(loop.cost)
            loop.grad_history.append(grad_norm)
            if grad_tolerance is not None:
                if loop.grad_threshold is None:
                    loop.grad_threshold = grad_tolerance.threshold(stop_gradient)
                if not n_constraints and stop_gradient <= loop.grad_threshold:
                    stop = "gradient_tolerance"
                    break

            # The solve at lambda = 0 in this attempt's scale gives the stationarity
            # number of the current point and is the first solve of the step's
            # multiplier search; one program makes both, with the step of this radius
            # (a fresh fit's first attempt has them from above).
            if (
                attempt == start_attempt
                and first is not None
                and first.answers(coordinates.scale, loop.radius, coordinates.addition)
            ):
                zero = first
            else:
                zero = route.solve_attempt(system, coordinates, loop.radius)
            # Every relative number of the attempt is relative to the cost scale here.
            relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
            stationarity = _stationarity(
                system, gradient, coordinates.scale, zero, relative_to, coordinates.addition
            )
            # The model predicts no further decrease the evaluation can show here.
            if stationarity <= _stationarity_limit(
                solver.stationarity_tol, cost_tol, _median(loop.noise_dominated_cost_change)
            ):
                work.solves += 1
                stop = "stationarity"
                break
            # The gradient's accuracy floor here: the median of the floor samples, at
            # most the bound from the lanes' tolerances and blocks; None before the
            # first sample, when no rule reads it.
            bound = self.floor.bound(coordinates.scale)
            level = _floor_level(loop.gradient_floor_samples, bound)
            full_norm = float(np.linalg.norm(np.asarray(zero[0])))
            # The model predicts no further decrease its derivatives resolve here.
            if level is not None and stationarity <= _floor_sigma(level, full_norm, relative_to):
                work.solves += 1
                stop = "derivative_accuracy"
                break
            radius_floor = _radius_floor(level, full_norm, gradient, coordinates.scale)
            step_radius = loop.radius
            trial, scaled_step_norm, predicted, multiplier, solves = _trial_point(
                system,
                loop.point,
                gradient,
                coordinates,
                loop.radius,
                first=zero,
                fixed=fixed,
                bounds=bounds,
                equality=n_constraints > 0,
            )
            work.solves += solves
            step = trial - loop.point
            step_norm = float(np.linalg.norm(step))

            trial_ok = _trial_valid(route.view, trial)
            failures: tuple = ()
            failed_owners = None
            rejection_reason = None
            invalid = False
            if not trial_ok:
                work.predicate_rejections += 1
                trial_cost = float("inf")
                rejection_reason = "the trial predicate rejected the point"
            else:
                before = work.lane_rungs.copy()
                results = self.evaluate("value", trial)
                # A failed lane of weight zero is outside the objective: logged, ignored.
                failures, ignored = route.weigh(route.failures(results))
                work.record(attempt, ignored, inner=False)
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung at the trial is solved again at
                    # the current point at its new rung, so that the gain ratio compares
                    # costs of the same rung; the model there stays the earlier rung's.
                    lane_costs, change = self.resolve(loop.point, raised, before, lane_costs)
                    loop.cost += change
                if failures:
                    invalid = True
                    work.invalid_trials += 1
                    work.record(attempt, failures, inner=False)
                    trial_cost = float("inf")
                    failed_owners = route.owners(failures)
                    rejection_reason = (
                        "the lane solve failed at the trial for experiment(s) "
                        f"{_listed(failure.lane for failure in failures)} "
                        f"({_listed(failure.kind for failure in failures)})"
                    )
                else:
                    trial_cost = route.trial_cost(results, trial)

            predicted_value = float(predicted)
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
                radius_floor=radius_floor,
            )
            assert_decision(accepted)
            rejection_reason = rejection_reason or reason

            previous_cost = loop.cost
            if accepted:
                loop.point = trial
                system, lane_costs = self.blocks(loop.point, start=False)
                loop.cost = float(system.cost)
                loop.accepted += 1
            work.attempted(accepted)
            loop.radius_history.append(loop.radius)
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
            loop.record = _record(
                rho,
                float(multiplier),
                accepted,
                trial,
                failures,
                route.depth,
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
                state=self.state(loop),
                stationarity=stationarity,
            )
            stop = _stop_after(
                after,
                requested=self.notify(info),
                max_invalid=int(solver.max_consecutive_invalid_trials),
                # The step test reads every trial whose cost was computed, at the point
                # where the acceptance left the fit.
                step_small=step_small(step_norm, loop.point, step_tol, tested=after.evaluated),
                accepted_count=loop.accepted,
                target_steps=solver.target_accepted_steps,
                # The budget counts the consistency re-solves too.
                evaluations=work.blocks + work.value,
                max_evaluations=solver.max_evaluations,
            )
            if stop is not None:
                break
        else:
            attempt = max(int(solver.max_steps), start_attempt) - 1
            stop = "max_steps"

        gradient, diagonal = elimination.level_vectors(system)
        grad_norm = float(np.linalg.norm(gradient))
        # The stationarity number at the end point, in the coordinates the next
        # attempt would take there.
        candidate = _scale_candidate(system, solver.x_scale, diagonal)
        end_scale = candidate if loop.scale is None else np.maximum(loop.scale, candidate)
        end = _subproblem(loop.point, gradient, end_scale, bounds)
        projected_grad_norm = float(np.linalg.norm(_bounded_gradient(gradient, end, bounds)))
        end_step = _full_step(
            system,
            gradient,
            end.scale,
            elimination.solve_scaled(system, end.scale, 0.0, end.addition),
            end.addition,
        )
        end_relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
        end_stationarity = end_step.decrease / end_relative_to
        return self.outcome(
            loop,
            stop,
            attempt=attempt,
            x0=x0,
            cost_tol=cost_tol,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            stationarity=end_stationarity,
            floor=self.floor_end(
                loop, end_step, end.scale, cost_tol, end_relative_to, system=system
            ),
        )
