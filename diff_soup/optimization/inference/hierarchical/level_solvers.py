"""Trust-region fits of a lowered study on its lanes, along the tree of its parameters."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar

import equinox as eqx
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization._abc import RESUME, OptSolver
from diff_soup.optimization.inference.fitting.result import (
    OptResult,
)
from diff_soup.optimization.inference.fitting.scales import (
    ConstantScale,
    HessianScale,
    JacScale,
    XScale,
)
from diff_soup.optimization.inference.hierarchical._joint_fit import _JointFit
from diff_soup.optimization.inference.hierarchical._lane_route import _LABEL, _LaneRoute, _lowered
from diff_soup.optimization.inference.hierarchical._nested_fit import _NestedFit
from diff_soup.optimization.solvers._loop import (
    _saved_value,
)
from diff_soup.optimization.solvers._termination import step_budget


def _refuse_state(state: Mapping, *, solver: str, mode: str, nested_at: Any, n: int) -> None:
    """Refuse a saved state this solver cannot continue, naming both settings."""
    saved_nested = _saved_value(state, "nested_at")
    raise ValueError(
        f"the saved state is of {_saved_value(state, 'solver')} in {_saved_value(state, 'mode')} "
        f"mode (nested_at={saved_nested or None!r}, {_saved_value(state, 'n')} coordinates) and "
        f"this solver is {solver} in {mode} mode (nested_at={nested_at!r}, {n}); resume with the "
        "same settings, or pass the point only: resume=dataclasses.replace(result, state=None)."
    )


def _check_state(state: Mapping, solver: TrustRegion, route: _LaneRoute) -> None:
    """Refuse a saved state of another solver, mode, cut level, size or problem."""
    mode = "joint" if solver.nested_at is None else "nested"
    expected = {
        "solver": _LABEL,
        "mode": mode,
        "nested_at": "" if solver.nested_at is None else str(solver.nested_at),
        "n": route.n,
        "fingerprint": route.fingerprint,
    }
    if any(_saved_value(state, key) != value for key, value in expected.items()):
        _refuse_state(state, solver=_LABEL, mode=mode, nested_at=solver.nested_at, n=route.n)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


class TrustRegion(OptSolver):
    """The trust-region Gauss-Newton method for a hierarchical study, with each step solved cluster by cluster.

    It needs a study lowered with `ds.lower(study)`. Box bounds are kept. Pass `checkpoint=` to
    `ds.optimization.optsolve` to save the fit and resume it later.

    Args:
        hessian: `"gauss-newton"` steps on the Gauss-Newton curvature, `"true"` on the exact Hessian.
        nested_at: A level whose members polish their own parameters, the others held, after every accepted step.
        inner: A `TrustRegion` without `nested_at` whose budget and tolerances the polish uses, or `None` for two iterations per member; with `max_steps=0` the fit makes the steps of `nested_at=None`.
        radius_init: The initial trust radius, in scaled parameter units.
        radius_max: The largest trust radius.
        eta: The ratio of achieved to predicted decrease a step must reach to be accepted.
        grad_tol: The gradient-norm tolerance, or `None` to test no gradient.
        step_tol: The step-norm tolerance relative to the parameter norm, or `None` for the dtype's default.
        cost_tol: The tolerance on the change in the objective relative to the objective, or `None` for the dtype's default.
        cost_atol: An absolute floor under the cost-change test, zero when `None`.
        stationarity_tol: The relative predicted decrease below which the fit ends, or `None` for the larger of `cost_tol` and three times the measured noise.
        max_consecutive_invalid_trials: The number of trials in a row with failed solves that ends the fit as `STALLED`.
        noise_trial_limit: A rejected trial counts as noise only when its predicted decrease is within this multiple of the study's evaluation noise.
        noise_relaxed_ratio: If `True`, the ratio of achieved to predicted decrease allows for the measured evaluation noise.
        gradient_floor: The relative accuracy of the derivatives, below which the fit ends `DERIVATIVE_ACCURACY`, `"auto"` to derive it from the solvers' `rtol`, or `None`.
        max_steps: The step budget, counted across resumes.
        max_evaluations: The budget of objective evaluations, which ends the fit as `MAX_STEPS`.
        inner_start_iterations: The number of polish iterations at the start point, before the first step.
        inner_fraction: A member's polish ends when its predicted decrease falls to this fraction of the accepted step's.
        inner_retreats: How many times a member whose solve fails at its polished point halves its polish step before its polish ends where its lanes solved.
        target_accepted_steps: Ends the fit as `ACCEPTED_STEPS` after this many accepted steps.
        x_scale: The parameter scaling, `HessianScale()` when `None`, `JacScale()` or a `ConstantScale`, with `JacScale` refused under `hessian="true"`.
        weights: One weight per member of the first level, or a `(level, weights)` pair, multiplying each member's share of the objective.
        weight_priors: If `True`, the weights also multiply each member's prior and regularization terms.
        constraints: Linear equality constraints on the joint parameter vector, as `LinearConstraint` objects.
        inequalities: Rows kept at or below zero, as [InequalityRows][diff_soup.optimization.inference.InequalityRows].
        gauges: [Gauge][diff_soup.optimization.inference.Gauge] constraints that pin directions the data leave free.
        fixed_indices: The indices of the joint parameter vector held at their starting values.
        callbacks: Callbacks run together with those passed to `ds.optimization.optsolve`.
    """

    requires: ClassVar[frozenset] = frozenset({"batched_lanes", RESUME})

    hessian: str = eqx.field(static=True)
    nested_at: Any = eqx.field(static=True)
    inner: TrustRegion | None
    radius_init: Real = eqx.field(static=True)
    radius_max: Real = eqx.field(static=True)
    eta: Real = eqx.field(static=True)
    grad_tol: Real | None = eqx.field(static=True)
    step_tol: Real | None = eqx.field(static=True)
    cost_tol: Real | None = eqx.field(static=True)
    cost_atol: Real | None = eqx.field(static=True)
    stationarity_tol: Real | None = eqx.field(static=True)
    max_consecutive_invalid_trials: int = eqx.field(static=True)
    noise_trial_limit: float | None = eqx.field(static=True)
    noise_relaxed_ratio: bool = eqx.field(static=True)
    gradient_floor: str | float | None = eqx.field(static=True)
    max_steps: Real = eqx.field(static=True)
    max_evaluations: int | None = eqx.field(static=True)
    inner_start_iterations: int = eqx.field(static=True)
    inner_fraction: float = eqx.field(static=True)
    inner_retreats: int = eqx.field(static=True)
    target_accepted_steps: Real | None = eqx.field(static=True)
    x_scale: XScale
    weights: Any
    weight_priors: bool = eqx.field(static=True)
    constraints: tuple
    inequalities: tuple
    gauges: tuple
    fixed_indices: tuple = eqx.field(static=True)
    callbacks: tuple = eqx.field(static=True)

    def __init__(
        self,
        *,
        hessian: Any = "gauss-newton",
        nested_at: Any = None,
        inner: Any = None,
        radius_init: Any = 1.0,
        radius_max: Any = 1e3,
        eta: Any = 1e-4,
        grad_tol: Any = None,
        step_tol: Any = None,
        cost_tol: Any = None,
        cost_atol: Any = None,
        stationarity_tol: Any = None,
        max_consecutive_invalid_trials: Any = 5,
        noise_trial_limit: Any = None,
        noise_relaxed_ratio: Any = False,
        gradient_floor: Any = "auto",
        max_steps: Any = 30,
        max_evaluations: Any = None,
        inner_start_iterations: Any = 0,
        inner_fraction: Any = 1e-2,
        inner_retreats: Any = 3,
        target_accepted_steps: Any = None,
        x_scale: Any = None,
        weights: Any = None,
        weight_priors: Any = True,
        constraints: Any = (),
        inequalities: Any = (),
        gauges: Any = (),
        fixed_indices: Any = (),
        callbacks: Any = (),
    ):
        """Check the settings that do not depend on the problem, then store them.

        Raises:
            ConfigurationError: If ``hessian``, ``inner``, ``x_scale``,
                ``max_steps``, ``max_evaluations``, ``target_accepted_steps``,
                the radii, ``eta``, ``stationarity_tol``,
                ``max_consecutive_invalid_trials``, ``noise_trial_limit``,
                ``noise_relaxed_ratio``, ``gradient_floor``, ``weight_priors``,
                ``inner_start_iterations``, ``inner_fraction`` or
                ``inner_retreats`` are out of range.
        """
        if hessian not in ("gauss-newton", "true"):
            raise ConfigurationError(
                f"TrustRegion hessian must be 'gauss-newton' or 'true'; got {hessian!r}."
            )
        if inner is not None and (
            not isinstance(inner, TrustRegion) or inner.nested_at is not None
        ):
            raise ConfigurationError(
                f"TrustRegion inner= takes a TrustRegion without nested_at=; got {inner!r}."
            )
        x_scale = HessianScale() if x_scale is None else x_scale
        if not isinstance(x_scale, (HessianScale, JacScale, ConstantScale)):
            raise ConfigurationError(
                "TrustRegion x_scale must be ds.optimization.inference.HessianScale, ds.optimization.inference.JacScale, "
                "or ds.optimization.inference.ConstantScale."
            )
        if hessian == "true" and isinstance(x_scale, JacScale):
            raise ConfigurationError(
                "TrustRegion cannot use JacScale with hessian='true'. Use "
                "ds.optimization.inference.HessianScale() or ds.optimization.inference.ConstantScale(...)."
            )
        step_budget(max_steps, label=_LABEL)
        if target_accepted_steps is not None and int(target_accepted_steps) < 0:
            raise ConfigurationError(
                "TrustRegion target_accepted_steps must be nonnegative or None."
            )
        if float(radius_init) <= 0.0 or float(radius_max) < float(radius_init):
            raise ConfigurationError("TrustRegion requires 0 < radius_init <= radius_max.")
        if not 0.0 <= float(eta) < 1.0:
            raise ConfigurationError("TrustRegion eta must satisfy 0 <= eta < 1.")
        try:
            tolerance = 0.0 if stationarity_tol is None else float(stationarity_tol)
        except (TypeError, ValueError):
            tolerance = float("nan")
        if not tolerance >= 0.0:
            raise ConfigurationError(
                "TrustRegion stationarity_tol must be None or a number of at least 0; got "
                f"{stationarity_tol!r}."
            )
        if (
            isinstance(max_consecutive_invalid_trials, bool)
            or not isinstance(max_consecutive_invalid_trials, (int, np.integer))
            or max_consecutive_invalid_trials < 1
        ):
            raise ConfigurationError(
                "TrustRegion max_consecutive_invalid_trials must be an integer of at least 1; "
                f"got {max_consecutive_invalid_trials!r}."
            )
        if noise_trial_limit is not None and (
            isinstance(noise_trial_limit, bool)
            or not isinstance(noise_trial_limit, (int, float, np.integer, np.floating))
            or not float(noise_trial_limit) > 0.0
        ):
            raise ConfigurationError(
                "TrustRegion noise_trial_limit must be None or a positive number; got "
                f"{noise_trial_limit!r}."
            )
        if not isinstance(noise_relaxed_ratio, (bool, np.bool_)):
            raise ConfigurationError(
                "TrustRegion noise_relaxed_ratio must be True or False; got "
                f"{noise_relaxed_ratio!r}."
            )
        if not (
            gradient_floor is None
            or (isinstance(gradient_floor, str) and gradient_floor == "auto")
            or (
                not isinstance(gradient_floor, (bool, np.bool_))
                and isinstance(gradient_floor, (int, float, np.integer, np.floating))
                and bool(np.isfinite(gradient_floor))
                and float(gradient_floor) > 0.0
            )
        ):
            raise ConfigurationError(
                "TrustRegion gradient_floor must be 'auto', None or a positive number, the "
                f"relative accuracy of the lanes' derivatives; got {gradient_floor!r}."
            )
        if not isinstance(weight_priors, (bool, np.bool_)):
            raise ConfigurationError(
                f"TrustRegion weight_priors must be True or False; got {weight_priors!r}."
            )
        if max_evaluations is not None and (
            isinstance(max_evaluations, (bool, np.bool_))
            or not isinstance(max_evaluations, (int, np.integer))
            or max_evaluations < 1
        ):
            raise ConfigurationError(
                "TrustRegion max_evaluations must be None or an integer of at least 1; got "
                f"{max_evaluations!r}."
            )
        if (
            isinstance(inner_start_iterations, (bool, np.bool_))
            or not isinstance(inner_start_iterations, (int, np.integer))
            or inner_start_iterations < 0
        ):
            raise ConfigurationError(
                "TrustRegion inner_start_iterations must be an integer of at least 0; got "
                f"{inner_start_iterations!r}."
            )
        if isinstance(inner_fraction, (bool, np.bool_)) or not isinstance(
            inner_fraction, (int, float, np.integer, np.floating)
        ):
            fraction = float("nan")
        else:
            fraction = float(inner_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ConfigurationError(
                "TrustRegion inner_fraction must be a number of at least 0 and below 1; got "
                f"{inner_fraction!r}."
            )
        if (
            isinstance(inner_retreats, (bool, np.bool_))
            or not isinstance(inner_retreats, (int, np.integer))
            or inner_retreats < 0
        ):
            raise ConfigurationError(
                "TrustRegion inner_retreats must be an integer of at least 0; got "
                f"{inner_retreats!r}."
            )
        self.hessian = hessian
        self.nested_at = nested_at
        self.inner = inner
        self.radius_init = radius_init
        self.radius_max = radius_max
        self.eta = eta
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.cost_tol = cost_tol
        self.cost_atol = cost_atol
        self.stationarity_tol = stationarity_tol
        self.max_consecutive_invalid_trials = int(max_consecutive_invalid_trials)
        self.noise_trial_limit = None if noise_trial_limit is None else float(noise_trial_limit)
        self.noise_relaxed_ratio = bool(noise_relaxed_ratio)
        self.gradient_floor = (
            gradient_floor
            if gradient_floor is None or isinstance(gradient_floor, str)
            else float(gradient_floor)
        )
        self.max_steps = max_steps
        self.max_evaluations = None if max_evaluations is None else int(max_evaluations)
        self.inner_start_iterations = int(inner_start_iterations)
        self.inner_fraction = fraction
        self.inner_retreats = int(inner_retreats)
        self.target_accepted_steps = target_accepted_steps
        self.x_scale = x_scale
        self.weights = weights
        self.weight_priors = bool(weight_priors)
        self.constraints = tuple(constraints)
        self.inequalities = tuple(inequalities)
        self.gauges = tuple(gauges)
        self.fixed_indices = tuple(fixed_indices)
        self.callbacks = tuple(callbacks)

    def solve(self, problem: Any, *, callbacks: tuple = (), state: Any = None) -> OptResult:
        """Fit ``problem``.

        Args:
            problem: The optimization problem of a lowered study, whose
                structure holds ``"batched_lanes"`` (what ``optsolve`` passes),
                or the lowered study itself (``ds.lower(study)``), which is
                lowered here.
            callbacks: Callbacks of this fit, after the solver's own.
            state: The state a step record or a saved result carries, to
                continue that fit: the point, the radius, the growth lock,
                the scale, the gradient threshold, the counters and the
                histories (in the nested mode also the members' radii, locks
                and gradient thresholds and the polish's switch) are
                restored, the blocks are evaluated once at the
                point, and the next attempt is the one the uninterrupted fit
                would make; ``max_steps`` counts the attempts of the whole
                fit. ``None`` starts afresh.

        Returns:
            The [OptResult][diff_soup.optimization.OptResult]; on the lanes it
            carries ``work`` and ``state``.

        Raises:
            ConfigurationError: If the settings do not fit the problem, or
                ``hessian="true"`` is asked of a study whose plan does not
                offer the lanes' ``hessian`` mode.
            ConfigurationError: If ``problem`` is not evaluated on the lanes
                of a lowered study (its structure lacks ``"batched_lanes"``).
            TypeError: If ``problem`` is neither an optimization problem nor
                a lowered study.
            ValueError: If a lane that no rung solves fails at the start
                point or at an accepted point (a resumed point included), the
                start point fails the trial predicate, or ``state`` was saved
                by another solver, mode, cut level, problem or number of
                coordinates.
        """
        callbacks = tuple(callbacks)
        problem = _lowered(self, problem)
        # The modes the problem's lanes evaluate decide the method: the blocks give
        # Gauss-Newton steps, the hessian mode the exact Hessian's.
        offers = problem.lanes._offers
        if self.hessian == "true":
            if self.nested_at is not None:
                raise ConfigurationError(
                    "TrustRegion nested_at= runs on the lanes of a lowered study with "
                    "hessian='gauss-newton', and this solver has hessian='true'; lower the "
                    "study with ds.lower(study) and use hessian='gauss-newton'."
                )
            if offers("value") and not offers("hessian"):
                raise ConfigurationError(
                    "TrustRegion hessian='true' steps on the exact Hessian of the lanes' "
                    "hessian mode, which this lowered study does not offer: the adjoints of an "
                    "experiment's solver cannot differentiate its lanes' reverse gradient "
                    "forward; use hessian='gauss-newton'."
                )
        route = _LaneRoute(self, problem)
        if self.nested_at is not None:
            route.refuse_inner_constraints(self.nested_at)
        if state is not None:
            _check_state(state, self, route)
            # The lanes' measured costs continue from the saved ledger, in the fit's own.
            ledger = getattr(route.problem, "_ledger", None)
            if ledger is not None:
                ledger.restore(state)
        if self.nested_at is not None:
            return _NestedFit(self, route, callbacks).run(state)
        return _JointFit(self, route, callbacks).run(state)


__all__ = ["TrustRegion"]
