"""Optimization result types.

[OptResult][diff_soup.inference.fitting.result.OptResult] holds the solution,
an enum termination status, iteration and evaluation counts, cost and
gradient histories, the trust-region record
([TrustRegionReport][diff_soup.inference.fitting.result.TrustRegionReport])
and conditioning diagnostics of one fit. Both are pytrees: the arrays and the
numbers of the fit are leaves, everything else is static.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, TypedDict

import equinox as eqx
import numpy as np


class TerminationStatus(enum.Enum):
    """How an optimizer run ended, at `result.status`.

    `status.converged` is `True` for `SUCCESS`, `GRAD_TOL`, `STEP_TOL`, `COST_TOL` and
    `DERIVATIVE_ACCURACY`.

    Attributes:
        SUCCESS: Convergence without a named test, which no built-in solver reports.
        GRAD_TOL: The gradient norm fell below its tolerance.
        STEP_TOL: The step fell below its tolerance.
        COST_TOL: The change in the objective fell below its tolerance.
        ACCEPTED_STEPS: `TrustRegion` reached its `target_accepted_steps`.
        MAX_STEPS: The step or evaluation budget ran out.
        SOLVER_FAILED: A time-stepping solve at the solution did not reach the end of its span.
        DIVERGED: `BFGS` met an objective or gradient that is not finite.
        STOPPED: A callback raised `StopFit`.
        UNKNOWN: The status before any run.
        STALLED: `TrustRegion` stopped without converging, after failed trials in a row or while its model still predicted a decrease.
        DERIVATIVE_ACCURACY: `TrustRegion` converged to the accuracy of its derivatives, which a tighter solver `rtol` improves.
    """

    SUCCESS = "success"
    GRAD_TOL = "gradient_tolerance"
    STEP_TOL = "step_tolerance"
    COST_TOL = "cost_tolerance"
    ACCEPTED_STEPS = "accepted_steps_reached"
    MAX_STEPS = "max_steps_reached"
    SOLVER_FAILED = "pde_solver_failed"
    DIVERGED = "optimizer_diverged"
    STOPPED = "stopped_by_callback"
    UNKNOWN = "unknown"
    STALLED = "stalled"
    DERIVATIVE_ACCURACY = "derivative_accuracy"

    @property
    def converged(self) -> bool:
        """True iff this status corresponds to a converged optimizer state."""
        return self in (
            TerminationStatus.SUCCESS,
            TerminationStatus.GRAD_TOL,
            TerminationStatus.STEP_TOL,
            TerminationStatus.COST_TOL,
            TerminationStatus.DERIVATIVE_ACCURACY,
        )


@dataclass(frozen=True)
class StepInfo:
    """Per-step diagnostic information, passed to callbacks.

    ``stationarity`` is the decrease the Gauss-Newton model predicts for its
    full step (the step at zero damping, with no trust-region limit) at the
    point the attempt steps from, relative to the cost scale
    ``max(|cost|, cost_atol / cost_tol)``: the cost itself, never below the
    cost at which ``cost_tol`` reaches ``cost_atol``. It does not depend on
    the scale of the cost or the units of the parameters, and it is near zero
    at a stationary point; ``nan`` when the model's matrix is singular there.
    ``member_stationarity`` holds the same number for every member of the cut
    level of a nested fit, each from its own subtree and relative to its own
    cost scale, in member order. Both are ``None`` for solvers
    that do not report them. For ``TrustRegion`` on the lanes of a lowered
    study, ``residual_evals`` is the number of ``blocks`` and ``value``
    requests so far, the count its ``max_evaluations`` budgets.
    """

    step: int
    cost: float
    grad_norm: float
    params: np.ndarray
    accepted: bool
    delta_norm: float
    wall_time: float
    # Damping (Levenberg-Marquardt) or trust-region state; None for other methods
    rho: Optional[float] = None
    lam: Optional[float] = None
    radius: Optional[float] = None
    # Per-segment timing (None if not available)
    segment_times: Optional[list[float]] = None
    # Restartable trust-region state (None for non-trust-region methods).
    nu: Optional[float] = None
    accepted_count: Optional[int] = None
    attempt_count: Optional[int] = None
    residual_evals: Optional[int] = None
    jacobian_evals: Optional[int] = None
    trial_cost_evals: Optional[int] = None
    statistics_evals: Optional[int] = None
    trial_predicate_rejections: Optional[int] = None
    invalid_pde_trials: Optional[int] = None
    damping_only_resolves: Optional[int] = None
    block_solves: Optional[int] = None
    statistics_identity: Optional[str] = None
    # Why a rejected step was rejected, and the owners whose PDE solves failed
    # at the trial (None for an accepted step or a solver that does not report them).
    rejection_reason: Optional[str] = None
    failed_owners: Optional[tuple] = None
    # Everything a solver that resumes needs to make the next attempt as the
    # uninterrupted run would: a read-only mapping from names to host arrays,
    # taken after this attempt. None for solvers that do not resume a state.
    state: Any = None
    # The relative decrease the Gauss-Newton model predicts for its full step at
    # the point the attempt steps from, and in the nested mode every member's.
    stationarity: Optional[float] = None
    member_stationarity: Optional[np.ndarray] = None


@dataclass(frozen=True)
class EvalCounts:
    """Cumulative evaluation counts."""

    residual_evals: int = 0
    jacobian_evals: int = 0
    gradient_evals: int = 0
    hessian_evals: int = 0
    hessian_vector_products: int = 0
    pde_solves: int = 0
    trial_cost_evals: int = 0
    statistics_evals: int = 0
    trial_predicate_rejections: int = 0
    invalid_pde_trials: int = 0
    damping_only_resolves: int = 0
    block_solves: int = 0


@dataclass(frozen=True)
class ChildFit:
    """One child's reconstruction inside a joint fit result. Frozen.

    A [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem] fit
    returns one of
    these per child (in input order) instead of a single ``opt_params``.

    Attributes:
        opt_params: The child's fitted parameter pytree (the shared groups
            carry the tied joint value), or ``None`` if the child exposed no
            reconstruction.
        local: Mapping from this child's local group name to its fitted value.
        window_states: The child's final decision-variable / learned-initial-state
            window states, or ``None`` for a fixed-IC child.
    """

    opt_params: Any = None
    local: dict = field(default_factory=dict)
    window_states: Any = None


@dataclass(frozen=True)
class ExperimentFit:
    """Fitted parameters for one experiment, indexed by Recipe handle."""

    opt_params: Any = None
    parameter_schema: Any = None
    params: Any = None
    owners: dict = field(default_factory=dict)
    window_states: Any = None

    def __getitem__(self, handle):
        """Return the parameter value used in this experiment."""
        value = self.parameter_schema.value(handle, self.params)
        return value[self.owners[handle.name]] if handle.name in self.owners else value


def _read_only(value: Any) -> Any:
    """A mapping as a read-only view (``None`` stays ``None``).

    The view is one opaque leaf, so the arrays a mapping holds are never
    walked as leaves of the static field that holds it.
    """
    if value is None or isinstance(value, MappingProxyType):
        return value
    return MappingProxyType(dict(value))


def _host_float(value: Any) -> Any:
    """``value`` as a Python float, ``None`` kept."""
    return None if value is None else float(value)


def _host_int(value: Any) -> Any:
    """``value`` as a Python int, ``None`` kept."""
    return None if value is None else int(value)


def _host_floats(value: Any) -> Any:
    """``value`` as a tuple of Python floats, ``None`` kept."""
    return None if value is None else tuple(float(item) for item in value)


def _float_vector(value: Any) -> Any:
    """``value`` as a 1-D float64 array (``()`` gives the empty one)."""
    return np.asarray(value, dtype=np.float64).reshape(-1)


class TrustRegionReport(eqx.Module):
    """What a trust-region fit measured at its end, beside its status. Frozen pytree.

    ``TrustRegion`` on the lanes of a lowered study fills every field;
    ``NewtonTrustRegion`` and a checkpoint of a running fit fill
    ``radius_history`` and leave the rest at their defaults. A relative
    number below is relative to the cost scale
    ``c = max(|cost|, cost_atol / cost_tol)`` at the point it was measured
    at: the cost itself, never below the cost at which ``cost_tol`` reaches
    ``cost_atol``.

    ``radius_history`` is the one dynamic leaf. Every other field is static:
    the strings, the counts, the tuples, ``None`` and the end-point floats,
    which are host numbers the fit computed once at its end (no array of the
    fit depends on them), so ``jax.tree.map`` over a result maps the radius
    history and leaves the verdict alone.

    Attributes:
        stop: The event that ended the loop: ``"gradient_tolerance"``,
            ``"stationarity"``, ``"derivative_accuracy"``,
            ``"step_tolerance"``, ``"cost_tolerance"``,
            ``"evaluation_noise"``, ``"invalid_trials"``, ``"max_steps"``,
            ``"max_evaluations"``, ``"accepted_steps"`` or ``"callback"``;
            ``None`` for a solver that does not name it. The status follows
            from it and the verdict below.
        stall_reason: Why a stalled fit stalled: ``"invalid_trials"`` (a run
            of invalid trials ended the loop or an invalid trial shrank the
            radius), ``"evaluation_noise"`` (the noise of the cost's
            evaluation hides the decrease the model still predicts) or
            ``"radius_collapsed"`` (the step or the cost change fell within
            its tolerance while the model still predicts a decrease);
            ``None`` for a fit that did not stall.
        consecutive_invalid_trials: The run of invalid trials at the end.
        failed_lanes: ``(lane, kind, t_reached, rungs_tried)`` of every lane
            that failed at the last invalid trial, ``rungs_tried`` the rungs
            its solve was tried at, in order; empty when no trial was invalid.
        radius_history: The trust radius after every attempt, the initial
            radius first, as a 1-D float64 array.
        stationarity: The stationarity number at the end point: the decrease
            the Gauss-Newton model predicts for its full step (the step at
            zero damping, with no trust-region limit), over ``c``; near zero
            at a stationary point, ``nan`` when the model's matrix is
            singular there. In the nested mode it is the full system's
            number. ``None`` when the solver does not report it.
        member_stationarity: The same number for every member of the cut
            level of a nested fit, each from its own subtree and relative to
            its own cost scale, in member order; ``None`` in the joint mode.
        evaluation_noise: The noise of the cost's evaluation (not the
            data's) measured from the fit's own rejected trials: the median,
            relative to ``c``, of the cost changes of up to ten of the latest
            rejected trials solved at a radius below that of the last
            accepted step whose cost rose or changed by more than the model
            predicted; 0 when there was no such trial.
        evaluation_noise_trials: The number of trials ``evaluation_noise`` is
            the median of (at most ten).
        evaluation_noise_ceiling: The relative tolerance the lane solvers were
            asked for (``LoweredStudy.evaluation_noise``), about the largest
            the noise should be; ``None`` when it is unknown.
        gradient_floor: The cost equivalent of the gradient floor at the end
            point, relative to ``c``: the floor (the accuracy, in cost per
            unit of scaled step length, to which the lanes compute the
            model's gradient, the median of up to ten samples from rejected
            trials) times the scaled norm of the model's full step, over
            ``c``. Before the first sample the bound stands in for it.
            ``None`` when the fit has no floor.
        gradient_floor_bound: The a priori bound's cost equivalent, in the
            same units; ``None`` when the fit has no floor.
        gradient_floor_trials: The number of samples the floor is the median
            of (at most ten).
        cost_floor: The cost's yardstick at the end point,
            ``max(cost_tol, 3 * evaluation_noise)``; for a fit that a step or
            a cost test ended converged because every rejected trial met a
            jump of the computed cost, ``max(cost_tol, 3 * evaluation_jumps)``.
        floor_parameter_radius: How far, in parameter units, the floor moves
            the minimizer along the model's full step at the end point (the
            step's length times the floor times its scaled norm, over twice
            the decrease the model predicts for it); ``nan`` where the model
            predicts no decrease, ``None`` without a floor.
        floor_parameter_box: How far the floor can move the minimizer along
            the softest direction of the model's matrix at the end point (the
            floor per unit parameter length along the full step, over that
            matrix's smallest eigenvalue); ``nan`` when the eigenvalue is not
            positive, ``None`` without a floor.
        floor_source: ``"a_posteriori"`` when the measured floor is at most
            the bound, ``"a_priori"`` when the bound capped it or stands in
            for it; ``None`` without a floor.
        evaluation_jumps: The median jump of the computed cost that every
            rejected trial of the window met, relative to ``c``, for a fit
            that a step or a cost test ended converged at those jumps;
            ``None`` otherwise.
        last_rejected_trials: For a fit that ended ``max_steps_reached`` or
            stalled, the number of rejected trials in its window (the last
            ten); ``None`` otherwise.
        median_rejected_cost_change: The median size of those trials' cost
            changes, each relative to the cost scale at its attempt; ``None``
            when there are none or it is not finite.
        median_rejected_predicted_decrease: The median of those trials'
            predicted decreases, in the same units; ``None`` when there are
            none or it is not finite.
    """

    stop: Optional[str] = eqx.field(static=True, default=None)
    stall_reason: Optional[str] = eqx.field(static=True, default=None)
    consecutive_invalid_trials: int = eqx.field(static=True, default=0, converter=_host_int)
    failed_lanes: tuple = eqx.field(static=True, default=())
    radius_history: Any = eqx.field(default=(), converter=_float_vector)
    stationarity: Optional[float] = eqx.field(static=True, default=None, converter=_host_float)
    member_stationarity: Optional[tuple] = eqx.field(
        static=True, default=None, converter=_host_floats
    )
    evaluation_noise: Optional[float] = eqx.field(static=True, default=None, converter=_host_float)
    evaluation_noise_trials: int = eqx.field(static=True, default=0, converter=_host_int)
    evaluation_noise_ceiling: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )
    gradient_floor: Optional[float] = eqx.field(static=True, default=None, converter=_host_float)
    gradient_floor_bound: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )
    gradient_floor_trials: int = eqx.field(static=True, default=0, converter=_host_int)
    cost_floor: Optional[float] = eqx.field(static=True, default=None, converter=_host_float)
    floor_parameter_radius: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )
    floor_parameter_box: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )
    floor_source: Optional[str] = eqx.field(static=True, default=None)
    evaluation_jumps: Optional[float] = eqx.field(static=True, default=None, converter=_host_float)
    last_rejected_trials: Optional[int] = eqx.field(static=True, default=None, converter=_host_int)
    median_rejected_cost_change: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )
    median_rejected_predicted_decrease: Optional[float] = eqx.field(
        static=True, default=None, converter=_host_float
    )


# The fields of a TrustRegionReport that ``OptResult.save`` writes as arrays; the
# others go into its ``trust_region_json`` entry.
_REPORT_ARRAYS = ("radius_history",)


class OptInfo(TypedDict, total=False):
    """The keys a result's ``info`` may hold, and what each is.

    ``info`` is solver-specific detail; a key is present only when the solver
    or the end-of-fit step that writes it ran. The trust-region numbers are
    not in it: they are the result's ``trust_region``
    ([TrustRegionReport][diff_soup.inference.fitting.result.TrustRegionReport]).

    Attributes:
        hessian: ``"true"`` or ``"gauss-newton"``, the model matrix a Newton
            trust-region solver used.
        hessian_composition: The differentiation composition of the true
            Hessian (``"forward-over-reverse"``, ...), ``None`` for the
            Gauss-Newton matrix.
        subproblem: The trust-region subproblem method.
        linear_solver: The linear solver of the step equations.
        linear_solver_fallbacks: Factorizations that fell back to the
            regularized route (``NewtonTrustRegion``).
        linear_factorizations: Factorizations made (``NewtonTrustRegion``).
        ad_mode: The differentiation mode of the objective's gradient.
        final_opt_state: The optax optimizer state after the last step.
        final_residual: The residual rows (data rows, then regularization
            rows) at the returned point, when their cost reproduces the cost.
        final_jacobian: Their Jacobian at the returned point.
        final_data_residual: The data rows of ``final_residual``.
        final_data_jacobian: The data rows of ``final_jacobian``.
        failure_sentinel_absent: True when the objective carries no failure
            sentinel, so no segment failure is inferred from the residual.
        trial_predicate_rejections: Trials the problem's trial predicate
            rejected.
        active_bounds: ``((index, -1 | 1), ...)``, the bounds active at the
            returned point.
        restricted_initialization_complete: True when a restricted first
            stage ran before the main fit.
        restricted_initialization_wall_time_s: That stage's wall time.
        restricted_initialization_steps: Its steps.
        restricted_initialization_accepted: Its accepted steps.
        restricted_initialization_cost: Its final cost.
        restricted_initialization_status: Its termination status.
        main_wall_time_s: The main fit's wall time.
        uq: The statistical provenance of the fit (the study's fingerprint,
            the objective, the coordinate and the likelihood).
        schema_fingerprint: The parameter schema's fingerprint a loaded
            result was saved with.
        checkpoint_step: The attempt or step a checkpoint file was written
            after.
        accepted_count: The accepted steps at that checkpoint.
        attempt_count: The attempts at that checkpoint.
        lam: The damping multiplier at that checkpoint.
    """

    hessian: str
    hessian_composition: Optional[str]
    subproblem: str
    linear_solver: str
    linear_solver_fallbacks: int
    linear_factorizations: int
    ad_mode: Optional[str]
    final_opt_state: Any
    final_residual: np.ndarray
    final_jacobian: np.ndarray
    final_data_residual: np.ndarray
    final_data_jacobian: np.ndarray
    failure_sentinel_absent: bool
    trial_predicate_rejections: int
    active_bounds: tuple
    restricted_initialization_complete: bool
    restricted_initialization_wall_time_s: float
    restricted_initialization_steps: int
    restricted_initialization_accepted: int
    restricted_initialization_cost: float
    restricted_initialization_status: str
    main_wall_time_s: float
    uq: dict
    schema_fingerprint: str
    checkpoint_step: int
    accepted_count: int
    attempt_count: int
    lam: float


_SCALARS = (str, int, float, bool)


def _persisted(value: Any) -> bool:
    """Whether ``OptResult.save`` keeps an ``info`` value.

    Kept: a JSON scalar, a sequence of numbers, or a sequence of rows whose
    items are JSON scalars or sequences of JSON scalars.
    """
    if value is None or isinstance(value, _SCALARS):
        return True
    if not isinstance(value, (tuple, list)):
        return False

    def plain(item: Any) -> bool:
        return item is None or isinstance(item, _SCALARS)

    return all(isinstance(item, (int, float)) for item in value) or all(
        isinstance(row, (tuple, list))
        and all(
            plain(item) or (isinstance(item, (tuple, list)) and all(plain(sub) for sub in item))
            for item in row
        )
        for row in value
    )


def _tupled(value: Any) -> Any:
    """A JSON value with every list turned into a tuple, rows included."""
    if isinstance(value, list):
        return tuple(_tupled(item) for item in value)
    return value


def _report_fields() -> tuple:
    """The field names of a ``TrustRegionReport``, in declaration order."""
    import dataclasses

    return tuple(item.name for item in dataclasses.fields(TrustRegionReport))


def _jsonable_report(value: Any) -> Any:
    """A static field of a ``TrustRegionReport`` as JSON: tuples as lists, numbers as floats."""
    if isinstance(value, (tuple, list)):
        return [_jsonable_report(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _uq_text(uq: Any) -> str:
    """``info["uq"]`` as JSON when it is a mapping of JSON scalars, else ``""``."""
    if not isinstance(uq, dict) or not all(
        isinstance(key, str) and (value is None or isinstance(value, _SCALARS))
        for key, value in uq.items()
    ):
        return ""
    import json

    return json.dumps(uq)


def _saved_outcome(data: Any, files: set) -> dict:
    """The failed segments, log likelihood and log prior a saved file holds, as fields."""
    fields: dict = {}
    if "n_failed_segments" in files:
        fields["n_failed_segments"] = int(data["n_failed_segments"])
        fields["failed_segment_indices"] = tuple(
            int(index) for index in np.asarray(data["failed_segment_indices"]).reshape(-1)
        )
    for name in ("log_likelihood", "log_prior"):
        if name in files and np.isfinite(float(data[name])):
            fields[name] = float(data[name])
    return fields


def _attached(result: Any, problem: Any, recorded_uq: Any, file: str) -> Any:
    """``result`` with ``problem``'s schema, objective and ``info["uq"]``, after the checks."""
    import dataclasses

    from diff_soup.inference.fitting._optsolve import _recorded_fingerprint

    schema = getattr(problem, "parameter_schema", None)
    if schema is None:
        raise TypeError(
            f"OptResult.load(problem=...) needs the problem the fit was made on, with a "
            f"parameter schema; {type(problem).__name__} has none."
        )
    where = f"OptResult.load({file!r}, problem=...)"
    recorded = _recorded_fingerprint(result)
    if not recorded:
        raise ValueError(
            f"{where}: the file records no parameter schema fingerprint, so it cannot be "
            f"checked against the problem's {str(schema.fingerprint)!r}; load it without "
            "problem=."
        )
    if str(recorded) != str(schema.fingerprint):
        raise ValueError(
            f"{where}: the file holds a fit of another problem: its parameter schema "
            f"fingerprint is {str(recorded)!r}, the problem's is {str(schema.fingerprint)!r}. "
            "Load it with the problem it was fitted on."
        )
    # A statistical problem (an MLE or MAP objective over a study) finalizes its fits with
    # ``finalize_opt_result``; any other problem only lends its schema.
    if not callable(getattr(problem, "finalize_opt_result", None)):
        return dataclasses.replace(result, parameter_schema=schema)
    kind = type(problem.objective).__name__
    uq = dict(recorded_uq) if recorded_uq else None
    if uq is not None:
        if uq.get("study_fingerprint") != problem.study.fingerprint:
            raise ValueError(
                f"{where}: the file holds a fit of another study: its study fingerprint is "
                f"{uq.get('study_fingerprint')!r}, the problem's is "
                f"{problem.study.fingerprint!r}. Load it with the problem it was fitted on."
            )
        if uq.get("objective") != kind:
            raise ValueError(
                f"{where}: the file holds a {uq.get('objective')!r} fit and the problem's "
                f"objective is {kind!r}. Load it with a problem of the objective it was "
                "fitted with."
            )
    else:
        uq = {
            "study_fingerprint": problem.study.fingerprint,
            "objective": kind,
            "coordinate": "physical",
            "likelihood": (
                "normalized measurement likelihood"
                if getattr(problem, "_log_normalizer", None) is not None
                else "weighted least squares; no measurement declares a noise model"
            ),
        }
    return dataclasses.replace(
        result,
        parameter_schema=schema,
        objective=problem.objective,
        info={**result.info, "uq": uq},
    )


class OptResult(eqx.Module):
    """The result of `ds.inference.optsolve`.

    `result[handle]` returns the fitted value of a parameter handle. A field the solver does
    not fill keeps its default.

    Attributes:
        params: The fitted values as one flat array.
        cost: The final value of the objective.
        status: The [TerminationStatus][diff_soup.inference.TerminationStatus] of the run.
        message: How the run ended, as text.
        n_steps: The number of steps the optimizer took.
        diagnostics: The [OptDiagnostics][diff_soup.inference.OptDiagnostics] of the fit, or `None` with `diagnostics=False`.
        log_likelihood: The log-likelihood at the solution, or `None` when no measurement declares a likelihood.
        log_prior: The log-prior at the solution under a `MAP` objective, or `None`.
        opt_params: The fitted parameter values in the form `ds.pdesolve` takes through `theta=`.
        initial_params: The starting values.
        grad_norm: The gradient norm at the solution.
        projected_grad_norm: The gradient norm projected onto the bounds, zero on an active bound.
        cost_history: The objective after every step.
        wall_time: The run time in seconds.
        experiments: The fitted values of each experiment, by its `id`.
        window_states: The fitted window starts of a multiple-shooting fit, or `None`.
        info: Further records of the solver's settings and work.
    """

    # The fitted point and the numbers of the fit: dynamic leaves.
    opt_params: Any = None
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    initial_params: np.ndarray = field(default_factory=lambda: np.array([]))
    cost: float = float("inf")
    grad_norm: float = float("inf")

    # Termination
    status: TerminationStatus = eqx.field(static=True, default=TerminationStatus.UNKNOWN)
    message: str = eqx.field(static=True, default="")

    # Iteration counts
    n_steps: int = eqx.field(static=True, default=0)
    max_steps: int = eqx.field(static=True, default=0)
    n_accepted: int = eqx.field(static=True, default=0)
    n_failed_segments: int = eqx.field(static=True, default=0)
    failed_segment_indices: tuple[int, ...] = eqx.field(static=True, default=())
    eval_counts: EvalCounts = eqx.field(static=True, default_factory=EvalCounts)

    # History
    cost_history: np.ndarray = field(default_factory=lambda: np.array([]))
    grad_history: np.ndarray = field(default_factory=lambda: np.array([]))

    # Timing
    wall_time: float = eqx.field(static=True, default=0.0)

    # Conditioning and identifiability evidence at the solution, and the final
    # decision-variable window-start states (``None`` outside that mode).
    diagnostics: Any = eqx.field(static=True, default=None)
    window_states: Any = None

    experiments: Mapping = eqx.field(static=True, default_factory=dict, converter=_read_only)
    parameter_schema: Any = eqx.field(static=True, default=None)
    objective: Any = eqx.field(static=True, default=None)
    log_likelihood: Optional[float] = eqx.field(static=True, default=None)
    log_prior: Optional[float] = eqx.field(static=True, default=None)

    info: Mapping = eqx.field(static=True, default_factory=dict, converter=_read_only)

    # Appended so positional construction keeps the field order above.
    projected_grad_norm: Optional[float] = None
    work: Any = eqx.field(static=True, default=None, converter=_read_only)
    state: Any = eqx.field(static=True, default=None, converter=_read_only)
    trust_region: Any = None

    @property
    def converged(self) -> bool:
        """Whether `status` counts as convergence."""
        return self.status.converged

    @property
    def evals(self) -> EvalCounts:
        """Cumulative objective and derivative evaluation counts."""
        return self.eval_counts

    def save(self, path: Any) -> None:
        """Save the result to an `.npz` file that `OptResult.load` reads.

        `opt_params`, `diagnostics`, `experiments` and `wall_time` are not saved.

        Args:
            path: The file path, with `.npz` appended when missing.
        """
        import json

        payload = {
            "params": np.asarray(self.params, dtype=float),
            "initial_params": np.asarray(self.initial_params, dtype=float),
            "cost": np.asarray(float(self.cost)),
            "grad_norm": np.asarray(float(self.grad_norm)),
            "n_steps": np.asarray(int(self.n_steps)),
            "max_steps": np.asarray(int(self.max_steps)),
            "n_accepted": np.asarray(int(self.n_accepted)),
            "status": np.asarray(str(self.status.value)),
            "message": np.asarray(str(self.message)),
            "cost_history": np.asarray(self.cost_history, dtype=float),
            "grad_history": np.asarray(self.grad_history, dtype=float),
            "schema_fingerprint": np.asarray(
                "" if self.parameter_schema is None else str(self.parameter_schema.fingerprint)
            ),
            "n_failed_segments": np.asarray(int(self.n_failed_segments)),
            "failed_segment_indices": np.asarray(
                tuple(int(index) for index in self.failed_segment_indices), dtype=np.int64
            ),
            "log_likelihood": np.asarray(
                np.nan if self.log_likelihood is None else float(self.log_likelihood)
            ),
            "log_prior": np.asarray(np.nan if self.log_prior is None else float(self.log_prior)),
            "uq_json": np.asarray(_uq_text(self.info.get("uq"))),
            "window_states": np.asarray(
                [] if self.window_states is None else np.asarray(self.window_states, dtype=float)
            ),
            "info_json": np.asarray(
                json.dumps(
                    {
                        key: list(value) if isinstance(value, (tuple, list)) else value
                        for key, value in self.info.items()
                        if _persisted(value)
                    }
                )
            ),
        }
        if self.trust_region is not None:
            report = self.trust_region
            payload["trust_region_json"] = np.asarray(
                json.dumps(
                    {
                        name: _jsonable_report(getattr(report, name))
                        for name in _report_fields()
                        if name not in _REPORT_ARRAYS
                    }
                )
            )
            for name in _REPORT_ARRAYS:
                payload[f"trust_region.{name}"] = np.asarray(getattr(report, name))
        for prefix, mapping in (("state", self.state), ("work", self.work)):
            if mapping is not None:
                for key, value in dict(mapping).items():
                    payload[f"{prefix}.{key}"] = np.asarray(value)
        np.savez(str(path) if str(path).endswith(".npz") else f"{path}.npz", **payload)

    @classmethod
    def load(cls, path: Any, *, problem: Any = None) -> "OptResult":
        """Load a result saved with `OptResult.save`.

        Args:
            path: The file path, with `.npz` appended when missing.
            problem: The problem the fit was made on, used to restore `result[handle]` after checking that the file holds a fit of it.

        Returns:
            The loaded `OptResult`.

        Raises:
            ValueError: If `problem` is given and the file holds a fit of another problem.
        """
        import json

        file = str(path) if str(path).endswith(".npz") else f"{path}.npz"
        with np.load(file) as data:
            info = {
                key: _tupled(value) for key, value in json.loads(str(data["info_json"])).items()
            }
            info["schema_fingerprint"] = str(data["schema_fingerprint"])
            window_states = np.asarray(data["window_states"])
            mappings = {}
            for prefix in ("state", "work"):
                entries = {
                    name[len(prefix) + 1 :]: np.asarray(data[name])
                    for name in data.files
                    if name.startswith(f"{prefix}.")
                }
                mappings[prefix] = MappingProxyType(entries) if entries else None
            files = set(data.files)
            recorded_uq = (
                json.loads(str(data["uq_json"]))
                if "uq_json" in files and str(data["uq_json"])
                else None
            )
            report = None
            if "trust_region_json" in files:
                report = TrustRegionReport(
                    **{
                        key: _tupled(value)
                        for key, value in json.loads(str(data["trust_region_json"])).items()
                    },
                    **{
                        name: np.asarray(data[f"trust_region.{name}"])
                        for name in _REPORT_ARRAYS
                        if f"trust_region.{name}" in files
                    },
                )
            result = cls(
                params=np.asarray(data["params"]),
                initial_params=np.asarray(data["initial_params"]),
                cost=float(data["cost"]),
                grad_norm=float(data["grad_norm"]),
                status=TerminationStatus(str(data["status"])),
                message=str(data["message"]),
                n_steps=int(data["n_steps"]),
                max_steps=int(data["max_steps"]),
                n_accepted=int(data["n_accepted"]),
                cost_history=np.asarray(data["cost_history"]),
                grad_history=np.asarray(data["grad_history"]),
                window_states=None if window_states.size == 0 else window_states,
                info=info,
                work=mappings["work"],
                state=mappings["state"],
                trust_region=report,
                **_saved_outcome(data, files),
            )
        return result if problem is None else _attached(result, problem, recorded_uq, file)

    def __getitem__(self, handle: Any) -> Any:
        """Reconstruct a fitted physical value by its Recipe handle."""
        if self.parameter_schema is None:
            raise TypeError(
                "this OptResult has no handle-based parameter schema; handle lookup "
                "is available on results from a handle-based inference problem."
            )
        return self.parameter_schema.value(handle, self.params)
