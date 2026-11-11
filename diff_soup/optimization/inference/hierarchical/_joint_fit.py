"""The joint trust-region fit: one step on every parameter of a lowered study."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

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
from diff_soup.optimization.inference.hierarchical import _active_set as active_set
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _inequality as inequality
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._constraints import start_on_rows
from diff_soup.optimization.inference.hierarchical._lane_route import _LABEL, _LaneRoute, _listed
from diff_soup.optimization.inference.hierarchical._level_solve import (
    _attempt,
    _quadratic,
    _scaled_multipliers,
)
from diff_soup.optimization.inference.hierarchical._step_program import (
    PRE_STOPS,
    Before,
    Outcome,
    Previous,
    Proposal,
    StepSettings,
    Trial,
    Zero,
    as_host,
    coordinates,
    judge_program,
    no_previous,
    propose,
    propose_program,
)
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _FloorEnd,
    _FullStep,
    _GradientFloor,
    _noise_ceiling,
    _noise_trial_limit,
    _trial_from,
    _trial_valid,
)
from diff_soup.optimization.inference.hierarchical._verdicts import (
    _failures_text,
    _last_invalid,
    _stalled_message,
    _stop_report,
    _verdict,
)
from diff_soup.optimization.inference.hierarchical._work import _record, _Work
from diff_soup.optimization.solvers._loop import (
    LoopState,
    loop_tolerances,
    stop_name,
)
from diff_soup.optimization.solvers._loop_judge import (
    Window,
    _floor_level,
    _floor_parameter_radius,
    _floor_sigma,
    _jump_level,
    _median,
    cost_scale,
    rejection_text,
    window_entries,
)

if TYPE_CHECKING:
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion


# ---------------------------------------------------------------------------
# The joint attempt
# ---------------------------------------------------------------------------


class _ZeroAnswer(NamedTuple):
    """An active-set round's solve at zero damping: its step, ``z``, decrease and solves."""

    step: jax.Array
    z: jax.Array
    decrease: jax.Array
    solves: jax.Array
    failed: jax.Array


class _TrialAnswer(NamedTuple):
    """An active-set round's trial: its step, scaled norm, predicted decrease, multiplier, solves."""

    step: jax.Array
    norm: jax.Array
    predicted: jax.Array
    multiplier: jax.Array
    solves: jax.Array
    failed: jax.Array


def _rows_solve(rows, system, gradient, coordinates, radius, point, settings, bounds) -> tuple:
    """``propose``'s ``solve`` under the inequality rows ``rows`` (a ``RowSet``), traced.

    The rows are evaluated and linearized at ``point`` and placed in the system
    once, at a fixed shape (``place_rows``); every round of the active-set method
    (``_active_set.active_set``) flips which of them the system holds. The solve
    at zero damping holds the rows its full step would violate; when that set
    does not settle, ``z`` is the method's last iterate and the decrease ``nan``.
    The trial is the active set's step on the attempt program, with a step that
    holds a row taking the shortened candidate under the box; a step that does
    not settle its set is the method's last iterate, its predicted decrease the
    model's. A trial that crosses another row beyond its slack, as the rows are
    nonlinear, is scaled back along its step to that row's boundary
    (``traced_step_back``), and then moved back onto the rows it violates
    (``traced_onto_rows``) and kept strictly inside the box.
    """
    c, C = inequality.traced_rows(rows, point)
    frozen = inequality.frozen_columns(system)
    C = jnp.where(frozen[None, :], 0.0, C)
    placed = inequality.place_rows(rows, system, point, c, C)
    scale = coordinates.scale
    addition = jnp.zeros_like(scale) if coordinates.addition is None else coordinates.addition
    radius = jnp.asarray(radius, dtype=scale.dtype)
    none = jnp.zeros((rows.count,), dtype=bool)
    shape, columns = (1, rows.count), np.zeros((point.shape[0],), dtype=np.int64)
    held_equality = int(np.shape(system.constraint_matrix)[0]) > 0

    def attempt(active):
        held = inequality.hold_rows(rows, placed, active)
        return _attempt(held, scale, addition, radius, eps=settings.eps, tiny=settings.tiny), held

    def multipliers(damping: Callable) -> Callable:
        def at(active, answer):
            held = inequality.hold_rows(rows, placed, active)
            flat = _scaled_multipliers(held, scale, addition, damping(answer))
            return inequality.rows_multipliers(rows, system, placed, flat)

        return at

    def zero(active) -> _ZeroAnswer:
        out, _ = attempt(active)
        return _ZeroAnswer(out[0] / scale, out[0], out[2], out[6], jnp.asarray(False))

    found = active_set.active_set(
        zero,
        multipliers(lambda answer: jnp.zeros((), scale.dtype)),
        c,
        C,
        point,
        zero(none),
        shape=shape,
        columns=columns,
    )
    settled = jnp.all(found.settled)
    z = jnp.where(settled, found.answer.z, scale * found.step)
    decrease = jnp.where(settled, found.answer.decrease, jnp.nan)

    def trial(active) -> _TrialAnswer:
        out, held = attempt(active)
        point_, norm, predicted = _trial_from(
            held,
            point,
            gradient,
            coordinates,
            radius,
            out[3],
            out[4],
            fixed=settings.fixed,
            bounds=bounds,
            equality=held_equality | jnp.any(active & placed.keep),
        )
        return _TrialAnswer(point_ - point, norm, predicted, out[5], out[6], out[7])

    def make() -> Trial:
        first = trial(none)
        path = active_set.active_set(
            trial,
            multipliers(lambda answer: answer.multiplier),
            c,
            C,
            point,
            first,
            shape=shape,
            columns=columns,
        )
        settled = jnp.all(path.settled)
        step = jnp.where(settled, path.answer.step, path.step)
        norm = jnp.where(settled, path.answer.norm, jnp.linalg.norm(scale * step))
        model = _model_decrease(system, gradient, coordinates, step)
        predicted = jnp.where(settled, path.answer.predicted, model)
        fraction = inequality.traced_step_back(rows, point, step, path.active)
        shorter = fraction < 1.0
        step = jnp.where(shorter, fraction * step, step)
        norm = jnp.where(shorter, fraction * norm, norm)
        predicted = jnp.where(
            shorter, _model_decrease(system, gradient, coordinates, step), predicted
        )
        moved = inequality.traced_onto_rows(rows, point + step, scale, ~frozen, bounds)
        if bounds is not None:
            moved = box.make_strictly_feasible(moved, bounds[0], bounds[1], rstep=0)
        return Trial(
            moved,
            norm,
            predicted,
            path.answer.multiplier,
            found.solves + first.solves + path.solves,
            first.failed | path.answer.failed,
        )

    return Zero(z, decrease), make


def _model_decrease(system, gradient, coordinates, step) -> jax.Array:
    """The decrease the attempt's scaled model predicts for ``step`` (traced)."""
    scaled = coordinates.scale * step
    quadratic = _quadratic(system, step)
    if coordinates.addition is not None:
        quadratic = quadratic + coordinates.addition @ (scaled * scaled)
    return -(gradient @ step + 0.5 * quadratic)


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
        # The modes the fit evaluates, compiled before its first request.
        self.modes = ("value", self.matrix)
        # The bound of the gradient's error at the current point, from the lanes'
        # tolerances and blocks.
        self.floor = _GradientFloor(solver, route)
        # The inequality rows kept at or below zero, or None.
        cut = None if solver.nested_at is None else route.names.index(solver.nested_at) + 1
        self.rows = (
            inequality.row_set(solver.inequalities, route, cut) if solver.inequalities else None
        )

    def evaluate(self, mode: str, point: np.ndarray, lanes: Iterable[str] | None = None) -> Any:
        """One request at ``point``, every lane (or ``lanes``) at the rung it holds.

        A lane the evaluation solved at a higher rung holds that rung from then on.
        A request of the matrix's mode (``blocks``, or ``hessian`` under
        ``hessian="true"``) counts in the work's ``blocks``, a ``frozen`` one in
        ``frozen``, any other in ``value``.
        """
        results = self.route.evaluate(mode, point, **self.work.request(lanes))
        self.work.hold(results)
        if mode == self.matrix:
            self.work.blocks += 1
        elif mode == "frozen":
            self.work.frozen += 1
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

    def state(self, loop: LoopState) -> MappingProxyType:
        """The state after an attempt, read-only host arrays, the nested mode's own entries in it."""
        route = self.route
        entries = {
            "solver": np.asarray(_LABEL),
            "mode": np.asarray("joint"),
            "nested_at": np.asarray(""),
            "fingerprint": np.asarray(route.fingerprint),
            "n": np.asarray(route.n, dtype=np.int64),
            **loop.entries(),
        }
        entries.update(self.nested_entries())
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

    def nested_entries(self) -> dict:
        """The entries of the state that only the nested mode has (none here)."""
        return {}

    def record(self, rho: float, lam: float, accepted: bool, trial: np.ndarray, failures: tuple):
        """The attempt just made (or the start), as the ``record.*`` entries of the state."""
        return _record(
            rho, lam, accepted, trial, failures, self.route.depth, **self.work.attempt_records()
        )

    def at_start(
        self, point: np.ndarray, system: elimination.LevelSystem, lane_costs: np.ndarray, *, fresh
    ) -> tuple:
        """The nested mode's start (its polish at a ``fresh`` start); here ``point``, ``system`` and ``lane_costs``."""
        return point, system, lane_costs

    def trial_pass(self) -> str:
        """The mode of the request at an attempt's trial: ``value``."""
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
        """The system and lane costs at the accepted trial ``loop.point``; ``loop.cost`` is set there.

        One request of the matrix's mode at the trial. ``system`` is the
        system the attempt stepped on and ``results`` the trial's request;
        the nested mode polishes from them and moves ``loop.point``,
        ``loop.cost`` and ``loop.scale`` to where its polish ended.
        """
        system, lane_costs = self.blocks(loop.point, start=False)
        loop.cost = float(system.cost)
        return system, lane_costs

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
        floor: _FloorEnd | None = None,
    ) -> tuple:
        """The status, the message and the ``info`` entries of the fit's end.

        ``stop`` is the event that ended the loop and ``sigma`` the stationarity
        number at the end point; ``windows`` holds the noise-dominated trials'
        cost changes and the rejected trials' cost changes and predicted
        decreases; ``floor`` the gradient floor there (``_FloorEnd``). A fit
        that converged to the accuracy of its derivatives says so in its
        message, with its numbers, and so does a fit that converged at the
        jumps of its computed cost.
        """
        noise_changes, rejected_changes, rejected_decreases = (
            window_entries(window).tolist() if isinstance(window, Window) else list(window)
            for window in windows
        )
        noise = float(_median(noise_changes))
        floor_sigma = 0.0 if floor is None else floor.sigma
        # With a gradient floor, the jumps of the computed cost that every rejected trial
        # met, when they all did, are what the verdict reads beside the measured noise.
        ceiling = _noise_ceiling(self.route)
        jumps = None
        if self.floor.coefficients is not None:
            jumps = _jump_level(rejected_changes, rejected_decreases, ceiling)
        jump_yardstick = None if jumps is None else max(cost_tol, 3.0 * jumps)
        yardstick = max(cost_tol, 3.0 * noise)
        status, stall_reason = _verdict(
            stop,
            sigma=sigma,
            yardstick=yardstick,
            lock=lock,
            floor_sigma=floor_sigma,
            floor_known=floor is not None and floor.level is not None,
            jump_yardstick=jump_yardstick,
        )
        at_jumps = (
            jump_yardstick is not None
            and status in (TerminationStatus.STEP_TOL, TerminationStatus.COST_TOL)
            and not sigma <= yardstick
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
            )
        budget = None
        if stop == "max_evaluations":
            budget = (
                f"{self.work.evaluations()} evaluations reached "
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
            residual_evals=work.evaluations(),
            jacobian_evals=work.blocks,
            trial_cost_evals=work.value + work.frozen,
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
        members: ArrayLike | None = None,
        floor: _FloorEnd | None = None,
    ) -> OptResult:
        """The fit's result, ended by ``stop`` after attempt ``attempt``, handed to the callbacks.

        ``stationarity`` is the stationarity number at the end point and
        ``members`` the members' numbers there in the nested mode; ``floor``
        the gradient floor there (``_FloorEnd``).
        """
        work, route = self.work, self.route
        loop.attempt = work.attempts
        state = self.state(loop)
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
            floor=floor,
        )
        evaluations = work.evaluations()
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
                trial_cost_evals=work.value + work.frozen,
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
                "inequality_rows": 0 if self.rows is None else int(self.rows.count),
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
                    "evaluation_noise": float(_median(loop.noise_dominated_cost_change)),
                    "evaluation_noise_trials": int(loop.noise_dominated_cost_change.count),
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
        scale: ArrayLike,
        cost_tol: float,
        relative_to: float,
        system: elimination.LevelSystem | None = None,
    ) -> _FloorEnd:
        """The gradient floor at the end point: what the verdict reads, and the ``info`` entries.

        ``step`` is the model's full step at the end point and ``scale`` the
        coordinates the bound reads there; ``relative_to`` is the cost scale
        there (``cost_scale``); ``system``, when given, is the full system at the end point, whose dense matrix
        gives the floor's reach along its softest direction. Before the first
        floor sample the bound stands in for the floor in the entries, and no
        verdict reads it.
        """
        samples = loop.gradient_floor_samples
        count = int(samples.count)
        bound = self.floor.bound(scale)
        level = float(_floor_level(samples, bound))
        level = None if np.isnan(level) else level
        reported = bound if level is None else level
        source = None
        if bound is not None:
            source = "a_posteriori" if count and _median(samples) <= bound else "a_priori"
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
            "gradient_floor_trials": count,
            "cost_floor": max(
                float(cost_tol), 3.0 * float(_median(loop.noise_dominated_cost_change))
            ),
            "floor_parameter_radius": None if radius is None else float(radius),
            "floor_parameter_box": None if box is None else float(box),
            "floor_source": source,
        }
        return _FloorEnd(level, float(_floor_sigma(level, step.norm, relative_to)), entries)

    def settings(self, tolerances: tuple, n_constraints: int) -> StepSettings:
        """The attempt programs' settings for this fit: the solver's options and the tolerances."""
        solver, route = self.solver, self.route
        grad_tolerance, step_tol, cost_tol, cost_atol, _ = tolerances
        dtype = np.dtype(np.asarray(route.problem.x0).dtype)
        target = solver.target_accepted_steps
        return StepSettings(
            fixed=tuple(int(index) for index in route.fixed),
            grad_test=grad_tolerance is not None and not n_constraints,
            equality=n_constraints > 0,
            stationarity_tol=solver.stationarity_tol,
            cost_tol=float(cost_tol),
            cost_atol=float(cost_atol),
            step_tol=float(step_tol),
            eta=float(solver.eta),
            radius_max=float(solver.radius_max),
            relaxed=bool(solver.noise_relaxed_ratio),
            noise_limit=_noise_trial_limit(solver, route),
            max_invalid=int(solver.max_consecutive_invalid_trials),
            target_steps=None if target is None else int(target),
            max_evaluations=solver.max_evaluations,
            eps=float(np.finfo(dtype).eps),
            tiny=float(np.finfo(float).tiny),
        )

    def before(self, loop: LoopState) -> Before:
        """The loop at the start of an attempt as the proposing program takes it."""
        point = np.asarray(loop.point)
        scale = np.full_like(point, np.nan) if loop.scale is None else np.asarray(loop.scale)
        threshold = np.nan if loop.grad_threshold is None else loop.grad_threshold
        return Before(
            point,
            scale.astype(point.dtype),
            np.asarray(loop.cost, dtype=np.float64),
            np.asarray(loop.radius, dtype=np.float64),
            np.asarray(threshold, dtype=np.float64),
            loop.noise_dominated_cost_change,
            loop.gradient_floor_samples,
            self.floor.current_terms(),
        )

    def propose(self, system, loop: LoopState, previous: Previous) -> Proposal:
        """The attempt's proposal at ``loop.point`` (``_step_program.propose``), as host values.

        One compiled program; under inequality rows it holds the active set's
        solves (``_rows_solve``), kept on the rows (``RowSet.program``).
        """
        route, settings = self.route, self.step_settings
        floor = self.floor.terms()
        if self.rows is None:
            proposal = propose_program(
                system,
                self.solver.x_scale,
                route.bounds,
                floor,
                self.before(loop),
                previous,
                settings=settings,
            )
        else:
            rows = self.rows
            program = rows.program(
                "joint",
                lambda: jax.jit(
                    functools.partial(propose, solve=functools.partial(_rows_solve, rows)),
                    static_argnames=("settings",),
                ),
            )
            proposal = program(
                system,
                self.solver.x_scale,
                route.bounds,
                floor,
                self.before(loop),
                previous,
                settings=settings,
            )
        return as_host(proposal)

    def judge(self, loop: LoopState, cost: float, outcome: Outcome) -> tuple:
        """The trial's judgment (``_step_program.judge``), one compiled program, as host values."""
        return as_host(
            judge_program(
                loop.trials(),
                np.asarray(cost, dtype=np.float64),
                outcome,
                settings=self.step_settings,
            )
        )

    def run(self, state: Mapping | None = None) -> OptResult:
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
            # The steps keep the equality rows from a point on them: a start off them moves
            # onto them first.
            x = start_on_rows(x, route.constraint_matrix, route.constraint_rhs, route.fixed, bounds)
            if not trial_is_valid(route.view, x):
                raise ValueError("TrustRegion initial point failed the trial predicate.")
            if self.rows is not None:
                c = self.rows.values(x)
                if np.any(c > active_set.slack(c, np.zeros((c.size, 1)), np.zeros((1,)))):
                    raise ValueError(
                        "TrustRegion keeps its inequality rows at or below zero from a start where "
                        f"they hold, and rows {np.flatnonzero(c > 0).tolist()} are above zero at "
                        f"the start (largest {float(np.max(c)):.3g}); start inside them."
                    )
        else:
            # The saved point is the start or an accepted point: every lane solved
            # there and the predicate passed. Its blocks are evaluated again below.
            x = np.array(state["point"], dtype=dtype)
            work.restore(state)
            work.resume_blocks += 1
        # grad_tol=None tests no gradient: the stationarity number is the test there.
        tolerances = loop_tolerances(
            dtype,
            grad_tol=solver.grad_tol,
            step_tol=solver.step_tol,
            cost_tol=solver.cost_tol,
            cost_atol=solver.cost_atol,
            stationarity_tol=solver.stationarity_tol,
            label=_LABEL,
        )
        grad_tolerance, _, cost_tol, _, _ = tolerances
        # With equality or inequality rows the gradient test is off: the projected
        # stationarity number ends the fit.
        n_constraints = int(route.constraint_matrix.shape[0]) + (
            0 if self.rows is None else self.rows.count
        )
        self.step_settings = self.settings(tolerances, n_constraints)
        route.problem.prepare(modes=self.modes)
        system, lane_costs = self.blocks(x, start=state is None)
        x, system, lane_costs = self.at_start(x, system, lane_costs, fresh=state is None)
        cost = float(system.cost)
        if state is None:
            # A fresh fit's first record holds the retries of the start's request.
            record = self.record(float("nan"), float("nan"), False, x, ())
            loop = LoopState.start(
                point=x, cost=cost, scale=None, radius=float(solver.radius_init), record=record
            )
        else:
            # A resumed fit's first record holds the retries of its first attempt, which
            # makes its own solve at zero damping in the restored scale, as an uninterrupted fit.
            loop = LoopState.restore(
                state, point=x, cost=cost, scale=np.array(state["scale"], dtype=dtype)
            )
        if grad_tolerance is not None and loop.grad_threshold is None:
            # The gradient test's threshold, from the first attempt's gradient in its scale.
            gradient, _, subproblem = coordinates(
                system, solver.x_scale, np.asarray(loop.point), self.before(loop).scale, bounds
            )
            loop.grad_threshold = grad_tolerance.threshold(
                float(np.linalg.norm(_bounded_gradient(gradient, subproblem, bounds)))
            )
        # Both programs are compiled here, before the loop: the first attempt's proposal is
        # made, and the judgment once on it, its answer dropped.
        previous = no_previous(dtype)
        proposal = self.propose(system, loop, previous)
        self.judge(loop, loop.cost, self.outcome_of(proposal, loop.radius, False, False, np.inf))

        for callback in self.callbacks:
            callback.on_start(route.n, route.view.n_residuals, solver.max_steps)
        self.started = time.perf_counter()
        start_attempt = loop.attempt
        attempt = start_attempt - 1
        made = False
        stop = None
        for attempt in range(start_attempt, int(solver.max_steps)):
            if made:
                proposal = self.propose(system, loop, previous)
            if int(proposal.previous_stop):
                # The attempt before ended the loop, at the point it left.
                stop = stop_name(proposal.previous_stop)
                attempt -= 1
                break
            loop.scale = np.asarray(proposal.scale)
            loop.cost_history.append(loop.cost)
            loop.grad_history.append(float(proposal.grad_norm))
            if int(proposal.pre_stop):
                stop = PRE_STOPS[int(proposal.pre_stop) - 1]
                if stop != "gradient_tolerance":
                    work.solves += 1
                break
            if bool(proposal.failed):
                raise ValueError(
                    "TrustRegion could not bracket a finite Gauss-Newton trust-region step (or "
                    "the solve under its inequality rows did not refine to rounding); improve "
                    "parameter scaling."
                )
            work.solves += int(proposal.solves)
            trial = np.asarray(proposal.trial)

            trial_ok = _trial_valid(route.view, trial)
            failures: tuple = ()
            failed_owners = None
            rejection_reason = None
            invalid = False
            results = None
            if not trial_ok:
                work.predicate_rejections += 1
                trial_cost = float("inf")
                rejection_reason = "the trial predicate rejected the point"
            else:
                before = work.lane_rungs.copy()
                results = self.evaluate(self.trial_pass(), trial)
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

            predicted = float(proposal.predicted)
            rho, accepted, radius, reason, after = self.judge(
                loop,
                loop.cost,
                self.outcome_of(proposal, loop.radius, trial_ok, invalid, trial_cost),
            )
            rho, accepted = float(rho), bool(accepted)
            loop.radius = float(radius)
            assert_decision(accepted)
            rejection_reason = rejection_reason or rejection_text(reason, rho, solver.eta)

            level = float(proposal.level)
            if accepted:
                loop.point = trial
                loop.accepted += 1
                system, lane_costs = self.accept(
                    loop,
                    system,
                    results,
                    attempt=attempt,
                    predicted=predicted,
                    level=None if np.isnan(level) else level,
                )
            work.attempted(accepted)
            loop.radius_history.append(loop.radius)
            loop.keep(after.trials)
            loop.record = self.record(rho, float(proposal.multiplier), accepted, trial, failures)
            loop.attempt = attempt + 1
            info = self.step_info(
                loop,
                attempt,
                grad_norm=float(proposal.grad_norm),
                accepted=accepted,
                delta_norm=float(proposal.step_norm),
                rho=rho,
                lam=float(proposal.multiplier),
                rejection_reason=rejection_reason,
                failed_owners=failed_owners,
                state=self.state(loop),
                stationarity=float(proposal.stationarity),
            )
            # The stop tests of this attempt run at the start of the next proposal, at the
            # point it left, with the callbacks' request and the evaluations made; the
            # budget counts the consistency re-solves and the frozen passes too.
            previous = Previous(
                np.True_,
                after,
                np.asarray(proposal.step_norm, dtype=after.predicted.dtype),
                np.asarray(self.notify(info)),
                np.asarray(work.evaluations(), dtype=np.int64),
                np.asarray(loop.accepted, dtype=np.int64),
            )
            made = True
        else:
            attempt = max(int(solver.max_steps), start_attempt) - 1
            stop = "max_steps"
            proposal = self.propose(system, loop, previous)
            if made and int(proposal.previous_stop):
                stop = stop_name(proposal.previous_stop)

        # The numbers at the end point, in the coordinates the next attempt would take
        # there: the last proposal's.
        gradient_norm = float(proposal.grad_norm)
        end_step = _FullStep(
            float(proposal.full_norm),
            float(proposal.full_length),
            float(proposal.decrease),
            np.asarray(proposal.scale),
        )
        end_stationarity = float(proposal.stationarity)
        end_relative_to = cost_scale(loop.cost, cost_tol, self.step_settings.cost_atol)
        return self.outcome(
            loop,
            stop,
            attempt=attempt,
            x0=x0,
            cost_tol=cost_tol,
            grad_norm=gradient_norm,
            projected_grad_norm=float(proposal.stop_gradient),
            stationarity=end_stationarity,
            floor=self.floor_end(
                loop, end_step, end_step.scale, cost_tol, end_relative_to, system=system
            ),
        )

    def outcome_of(
        self, proposal: Proposal, radius: float, valid: bool, invalid: bool, trial_cost: float
    ) -> Outcome:
        """The judgment's ``Outcome`` of a trial from its proposal, at the step's radius ``radius``."""
        return Outcome(
            np.asarray(valid),
            np.asarray(invalid),
            np.asarray(trial_cost, dtype=np.float64),
            np.asarray(proposal.predicted, dtype=np.float64),
            np.asarray(proposal.scaled_norm, dtype=np.float64),
            np.asarray(radius, dtype=np.float64),
            np.asarray(proposal.radius_floor, dtype=np.float64),
            np.asarray(proposal.bound, dtype=np.float64),
        )
