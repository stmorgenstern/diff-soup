"""The replicates of a bootstrap: the evaluation at the fit, and every replicate's trust region.

The replicates are the members of a population of points
([PopulationRoute][diff_soup.inference.hierarchical._population.PopulationRoute]
built at the fit): replicate ``q`` is population member ``q`` of every request,
and the level system and the trial cost of replicate ``q`` weigh its lanes,
its regularizer blocks and its gauge rows by its own weights, through the same
definition a weighted fit uses (``_LaneRoute.weigh_by``).

``FitProducts`` is what one ``blocks`` evaluation at the fit serves with no
further evaluation: every weighting's level system at the fit, its one-step
estimate (the full Gauss-Newton step from the fit, kept inside the box), the
leave-one-member-out one-steps, the Laplace covariance, the members' scores and
influence vectors, and the cluster sandwich. Every system is built through the
route's weighting, and every solve is a call of the fit's own compiled
elimination programs with the fit's shapes.

``ReplicateStepper`` runs the fit's joint trust region for every member of the
population at once: one attempt of every member still running per lockstep
iteration, with at most three requests per iteration (one ``value`` request at
the trials, one at the current points when a trial raised a lane's rung, and
one ``blocks`` request at the points accepted). It calls the helpers a fit's
attempt calls, in the same order, and ends every member with an outcome; a
lane that fails or host work that raises ends or redirects that member only.
"""

from __future__ import annotations

import dataclasses
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple

import jax
import numpy as np

from diff_soup.inference.hierarchical import _bounds as box
from diff_soup.inference.hierarchical import _level_elimination as elimination
from diff_soup.inference.hierarchical._population import (
    Lockstep,
    PopulationRoute,
    _held_from,
    _NotFinite,
)
from diff_soup.inference.hierarchical.level_solvers import (
    _LABEL,
    _bounded_gradient,
    _choose,
    _failures_text,
    _noise_trial_limit,
    _scale_candidate,
    _stationarity,
    _stationarity_limit,
    _subproblem,
    _trial_point,
    _trial_valid,
    _verdict,
)
from diff_soup.inference.solvers._loop import (
    _WINDOW,
    Trials,
    _after_attempt,
    _judge,
    _median,
    _next_radius,
    _saved_window,
    _stop_after,
    cost_scale,
    loop_tolerances,
    step_small,
)

# ---------------------------------------------------------------------------------------
# The evaluation at the fit, and what it serves with no further evaluation
# ---------------------------------------------------------------------------------------


class OneStepFailure(NamedTuple):
    """A one-step estimate that could not be computed; its row of estimates is NaN.

    Attributes:
        kind: ``"replicate"`` for a replicate's one-step, ``"jackknife"`` for
            the one-step that leaves a member out.
        index: The replicate's number, or the number of the member left out.
        reason: ``"gauge"``: every member a gauge averages over weighs zero,
            so its row constrains nothing; ``"system"``: the level system at
            the fit, or its full step, is not finite (its matrix is not
            positive definite on the coordinates the step moves); ``"error"``:
            any other exception of its host work.
        message: The exception's type and text.
    """

    kind: str
    index: int
    reason: str
    message: str


class OneStep(NamedTuple):
    """The full Gauss-Newton step from a point, kept inside the box.

    Attributes:
        point: ``(n,)``, the point the step reaches.
        scaled_length: The step's length in the coordinates of its subproblem,
            ``||z||`` with ``z`` the solve at zero damping: the smallest
            trust-region radius that leaves the full step uncut.
    """

    point: np.ndarray
    scaled_length: float


class _ZeroGauge(ValueError):
    """Weights under which every member a gauge averages over weighs zero."""


def _finite(gradient: np.ndarray, diagonal: np.ndarray, system: Any = None) -> bool:
    """Whether a level system's gradient, diagonal and (given ``system``) cost are finite."""
    return bool(
        np.all(np.isfinite(gradient))
        and np.all(np.isfinite(diagonal))
        and (system is None or np.isfinite(float(system.cost)))
    )


def one_step(system: Any, point: Any, *, x_scale: Any, bounds: Any, fixed: Any) -> OneStep:
    """The step a trust-region fit takes from ``point`` when its radius does not cut the full step.

    The scale is ``x_scale``'s, from the diagonal of the system's matrix; the
    subproblem's coordinates take the system's own gradient (under bounds the
    Coleman-Li coordinates and their diagonal addition); the step is the solve
    at zero damping under the system's equality rows. Fixed coordinates, and
    every coordinate of a held node, do not move. Without bounds the point is
    ``point + step``. With bounds the step is the best, by the model's value,
    of the step shortened to stay inside the box, the step reflected at the
    sides it crosses and the Cauchy step, all at the step's own scaled length
    (with equality rows the shortened step alone), and the point is kept
    strictly inside the box.

    Args:
        system: The level system at ``point``.
        point: ``(n,)``, a point strictly inside the box.
        x_scale: The scale strategy of the solver (``HessianScale``,
            ``JacScale`` or ``ConstantScale``).
        bounds: ``(lower, upper)`` or ``None``.
        fixed: The fixed coordinates.

    Returns:
        The [OneStep][diff_soup.inference.hierarchical._replicates.OneStep].

    Raises:
        ArithmeticError: If the system's gradient or diagonal, or the solve at
            zero damping, is not finite.
    """
    x = np.asarray(point)
    gradient, diagonal = elimination.level_vectors(system)
    if not _finite(gradient, diagonal):
        raise _NotFinite("the level system at the point is not finite")
    scale = _scale_candidate(system, x_scale, diagonal)
    coordinates = _subproblem(x, gradient, scale, bounds)
    zero, _ = elimination.solve_scaled(system, coordinates.scale, 0.0, coordinates.addition)
    if not np.all(np.isfinite(zero)):
        raise _NotFinite(
            "the full step is not finite: the matrix is not positive definite on the coordinates "
            "the step moves"
        )
    with np.errstate(all="ignore"):
        step = (1.0 / coordinates.scale) * zero
    fixed = np.asarray(fixed, dtype=np.int64)
    if fixed.size:
        step[fixed] = 0.0
    length = float(np.linalg.norm(zero))
    if bounds is None:
        return OneStep(x + step, length)
    chosen, _, _ = _choose(
        system,
        x,
        gradient,
        coordinates,
        step,
        length,
        bounds,
        equality=int(system.constraint_matrix.shape[0]) > 0,
    )
    return OneStep(box.make_strictly_feasible(x + chosen, bounds[0], bounds[1], rstep=0), length)


def _kept(system: Any) -> np.ndarray:
    """``(n,)`` booleans: the coordinates a solve moves, neither fixed nor in a held node."""
    keep = ~np.asarray(system.fixed_mask, dtype=bool)
    offsets = elimination._offsets(system.counts, system.widths)
    for level, node in system.held:
        width = int(system.widths[level])
        start = offsets[level] + node * width
        keep[start : start + width] = False
    return keep


def _against(system: Any, vector: np.ndarray) -> Any:
    """``system`` with ``vector`` as its gradient and a zero constraint target.

    Its solve at zero damping is ``-H^-1 vector`` on the manifold of its
    equality rows: the solve of the rows' system with a right-hand side of
    zero on the rows. Every array keeps its shape, so the solve is a call of
    the program the system's own solves compiled; the new arrays are copied
    to the device as they are, placed as the system's point is, which
    compiles nothing.
    """
    dtype = np.dtype(system.root.dtype)
    placed = system.point
    device = placed.sharding if placed.committed else None
    parts = elimination._split(system, np.asarray(vector, dtype=dtype))
    return dataclasses.replace(
        system,
        root_gradient=jax.device_put(np.ascontiguousarray(parts[0]), device),
        gradient=(None,)
        + tuple(jax.device_put(np.ascontiguousarray(part), device) for part in parts[1:]),
        point=jax.device_put(np.zeros(np.shape(system.point), dtype=dtype), device),
        constraint_rhs=jax.device_put(
            np.zeros(np.shape(system.constraint_rhs), dtype=dtype), device
        ),
    )


def _described(error: Exception) -> str:
    """``"<type>: <message>"`` of an exception, its type named by its first public class.

    A record that a user reads names ``ArithmeticError`` for a system that is
    not finite and ``ValueError`` for a gauge of weight zero, not the private
    classes that carry them here.
    """
    kind = next(cls for cls in type(error).__mro__ if not cls.__name__.startswith("_"))
    return f"{kind.__name__}: {error}"


def _failure(kind: str, index: int, error: Exception) -> OneStepFailure:
    """The record of a one-step whose host work raised ``error``."""
    if isinstance(error, _ZeroGauge):
        reason = "gauge"
    elif isinstance(error, _NotFinite):
        reason = "system"
    else:
        reason = "error"
    return OneStepFailure(kind, int(index), reason, _described(error))


class FitProducts:
    """What one ``blocks`` evaluation at a fit serves with no further evaluation.

    One ``blocks`` request at the fit's point, every lane at the rung the fit
    holds, through the route (with ``chunk=None`` the fit's own study and
    programs); every lane has to solve there. From that one result, through
    the route's weighting (a member's lanes and its own regularizer blocks
    take its weight, a member of weight zero is held):

    - per replicate, its level system at the fit under its weights and its
      one-step estimate (``one_step``);
    - per member ``m``, the one-step under weight zero on ``m`` and one
      elsewhere: the leave-one-member-out estimate;
    - the level system ``H`` of the fit's own objective (every weight one);
      per member its score ``s_m``, the gradient its lanes and its own
      regularizer blocks add at the fit (the gradient of the fit's objective
      less that of the objective without the member, both with the member's
      regularizer blocks weighed by its weight), its influence vector
      ``u_m = H^-1 s_m``, the Laplace covariance ``H^-1`` of the shared
      coordinates and the cluster sandwich ``sum_m u_m u_m^T`` (no
      small-sample factor) of the shared coordinates. With equality rows the
      inverse is the one on the rows' manifold. Fixed coordinates and held
      nodes have zero rows and columns.

    Every system is assembled by the fit's compiled programs and every solve
    is a call of its compiled elimination with the fit's shapes.

    Attributes:
        point: ``(n,)``, the fit's point.
        results: The ``blocks`` result at the point.
        system: The level system of the fit's own objective at the point.
        estimates: ``(R, n)``, the replicates' one-step estimates (NaN rows
            for the failures).
        scaled_lengths: ``(R,)``, each replicate's full step length in its
            own scaled coordinates (NaN for the failures).
        jackknife: ``(M, n)``, the leave-one-member-out one-step estimates.
        scores: ``(M, n)``, the members' scores at the point.
        influence: ``(M, n)``, the members' influence vectors.
        laplace: ``(g, g)``, the Laplace covariance of the shared coordinates.
        sandwich: ``(g, g)``, the cluster sandwich of the shared coordinates.
        score_balance: ``||sum_m s_m|| / max_m ||s_m||``: zero at a fit that
            is stationary and has no regularizer row on the shared
            coordinates.
        active_bounds: The sides active at the point, ``((index, -1 | 1), ...)``.
        failures: The [OneStepFailure][diff_soup.inference.hierarchical._replicates.OneStepFailure]
            records, replicates first.
        solves: The solves of the compiled elimination, by product.
        seconds: The wall seconds of the ``blocks`` request (``request``),
            the host seconds of each product, and their ``total``.
        requested: The population members that took part in the ``blocks``
            request.
        systems: With ``keep``, per replicate its level system at the point
            (``None`` for the failures); else empty.
        jackknife_systems: With ``keep``, per member the level system that
            leaves it out, under the route's weighting (``None`` for the
            failures); else empty.
        jackknife_lengths: ``(M,)``, the full step length in scaled
            coordinates of each leave-one-member-out one-step (NaN for the
            failures).
    """

    def __init__(
        self,
        route: PopulationRoute,
        fit: Any,
        weights: Any,
        *,
        x_scale: Any,
        keep: bool = False,
        numbers: Any = None,
    ):
        """Evaluate the blocks at the fit's point and compute every product from them.

        Args:
            route: The population route of the study the fit was made on.
            fit: The result of ``TrustRegion`` on that study.
            weights: ``(R, M)``, one weight per member of the top level for
                every replicate.
            x_scale: The scale strategy of the one-steps.
            keep: Keep every replicate's and every leave-one-member-out
                system, from which their trust regions take their first
                attempt.
            numbers: ``(R,)``, the number of the replicate each row of
                ``weights`` is, as its failure records name it; ``None``
                numbers the rows ``0 .. R - 1``.

        Raises:
            ValueError: If the fit's point is not finite or not of the study's
                coordinates, ``weights`` has not one column per member, a lane
                fails at the fit's point after its rungs, or the system of the
                fit's objective there is not finite or its matrix not positive
                definite on the fitted coordinates.
        """
        started = time.perf_counter()
        base = route._base
        x = np.array(np.asarray(fit.params), dtype=base.dtype)
        if x.shape != (route.n,) or not np.all(np.isfinite(x)):
            raise ValueError(
                f"fit.params is the fit's point, {route.n} finite coordinates of this study; got "
                f"shape {x.shape}{'' if np.all(np.isfinite(x)) else ' with non-finite entries'}."
            )
        count = int(route.tables.counts[0])
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[1] != count:
            raise ValueError(
                f"weights= holds one row per replicate and one column per member of the level "
                f"{route.over!r}, which has {count}; got shape {weights.shape}."
            )
        points = np.tile(x, (route.population, 1))
        active = None if route.population == 1 else np.arange(route.population) == 0
        results = route.evaluate("blocks", points, active=active)
        self.requested = np.asarray(results.members, dtype=np.int64)
        seconds = {"request": time.perf_counter() - started}
        mark = time.perf_counter()
        ones = np.ones((count,))
        failed, _ = route.failures(results, member=0, weights=ones)
        if failed:
            raise ValueError(
                f"the blocks evaluation at the fit's point failed for lane(s) "
                f"{_failures_text(failed)}; every lane has to solve at the point the fit "
                "accepted. Give the result of the fit made on this lowered study."
            )
        self.route = route
        self.point = x
        self.results = results
        self._x_scale = x_scale
        self._fixed = np.asarray(base.fixed, dtype=np.int64)
        shared = int(route.tables.widths[0])
        self.system = route.system(results, x, member=0, weights=ones)
        gradient, diagonal = elimination.level_vectors(self.system)
        if not _finite(gradient, diagonal):
            raise ValueError(
                "the level system of the fit's objective at the fit's point is not finite; the "
                "bootstrap starts from a point where every lane's blocks are finite."
            )
        self._keep = _kept(self.system)
        self._columns: dict = {}
        seconds["system"] = time.perf_counter() - mark
        failures: list = []

        mark = time.perf_counter()
        replicates = weights.shape[0]
        numbers = np.arange(replicates) if numbers is None else np.asarray(numbers, dtype=np.int64)
        self.estimates = np.full((replicates, route.n), np.nan)
        self.scaled_lengths = np.full((replicates,), np.nan)
        self.systems: list = [None] * replicates if keep else []
        for replicate, values in enumerate(weights):
            try:
                system = self._system_of(values)
                step = self._one_step(system)
            except Exception as error:  # noqa: BLE001 -- recorded as that replicate's failure
                failures.append(_failure("replicate", int(numbers[replicate]), error))
                continue
            self.estimates[replicate] = step.point
            self.scaled_lengths[replicate] = step.scaled_length
            if keep:
                self.systems[replicate] = system
        seconds["one_step"] = time.perf_counter() - mark

        mark = time.perf_counter()
        self.jackknife = np.full((count, route.n), np.nan)
        self.jackknife_lengths = np.full((count,), np.nan)
        self.jackknife_systems: list = [None] * count if keep else []
        self.scores = np.zeros((count, route.n))
        for member in range(count):
            values = np.ones((count,))
            values[member] = 0.0
            # The member's score: what its lanes and its own regularizer blocks add to the
            # gradient at the fit, the gradient of the fit's objective less that of the
            # objective without the member.
            without = self._system_of(values, weight_priors=True)
            self.scores[member] = (gradient - elimination.level_vectors(without)[0]) * self._keep
            try:
                system = without if route.weight_priors else self._system_of(values)
                step = self._one_step(system)
            except Exception as error:  # noqa: BLE001 -- recorded as that member's failure
                failures.append(_failure("jackknife", member, error))
                continue
            self.jackknife[member] = step.point
            self.jackknife_lengths[member] = step.scaled_length
            if keep:
                self.jackknife_systems[member] = system
        seconds["jackknife"] = time.perf_counter() - mark

        mark = time.perf_counter()
        self.influence = np.stack([self._inverse_times(score) for score in self.scores])
        seconds["influence"] = time.perf_counter() - mark

        mark = time.perf_counter()
        columns = [self._column(column) for column in range(shared)]
        self.laplace = np.stack(columns, axis=1)[:shared] if shared else np.zeros((0, 0))
        seconds["laplace"] = time.perf_counter() - mark

        lead = self.influence[:, :shared]
        self.sandwich = lead.T @ lead
        largest = float(np.max(np.linalg.norm(self.scores, axis=1))) if count else 0.0
        total = float(np.linalg.norm(np.sum(self.scores, axis=0)))
        self.score_balance = total / largest if largest > 0.0 else 0.0
        self.active_bounds = box.active_bounds(x, route.bounds)
        self.failures = tuple(failures)
        self.solves = MappingProxyType(
            {
                "one_step": int(replicates),
                "jackknife": int(count),
                "influence": int(count),
                "laplace": int(np.count_nonzero(self._keep[:shared])),
            }
        )
        seconds["total"] = time.perf_counter() - started
        self.seconds = MappingProxyType(seconds)

    def _system_of(self, weights: Any, weight_priors: Any = None) -> Any:
        """The level system at the fit under ``weights``, through the route's weighting.

        The system of the fit's objective at unit weights was formed first, so
        a weighting that fails here fails on its weights; with a gauge, the
        only such failure is a gauge whose members all weigh zero.
        """
        route = self.route
        try:
            weighed = route._weighted(weights, weight_priors)
        except ValueError as error:
            if route._base.gauges:
                raise _ZeroGauge(str(error)) from error
            raise
        return weighed.system(self.results, self.point, member=0)

    def _one_step(self, system: Any) -> OneStep:
        """The one-step from the fit's point under ``system``."""
        return one_step(
            system, self.point, x_scale=self._x_scale, bounds=self.route.bounds, fixed=self._fixed
        )

    def _inverse_times(self, vector: Any) -> np.ndarray:
        """``H^-1 vector`` on the manifold of the equality rows; zero on fixed and held coordinates."""
        right = np.asarray(vector, dtype=np.float64) * self._keep
        if not np.any(right):
            return np.zeros_like(right)
        step, _ = elimination.solve_damped(_against(self.system, right), 0.0)
        solved = -np.asarray(step, dtype=np.float64)
        if not np.all(np.isfinite(solved)):
            raise ValueError(
                "the Gauss-Newton matrix of the fit's objective at the fit's point is not "
                "positive definite on the fitted coordinates, so it has no inverse there; fix or "
                "regularize the coordinates the data do not determine."
            )
        return solved

    def _column(self, column: int) -> np.ndarray:
        """Column ``column`` of ``H^-1``, solved once."""
        column = int(column)
        if column not in self._columns:
            unit = np.zeros((self.route.n,))
            unit[column] = 1.0
            self._columns[column] = self._inverse_times(unit)
        return self._columns[column]

    @property
    def inverse_columns(self) -> Any:
        """The columns of ``H^-1`` solved so far, ``(n,)`` each, by coordinate."""
        return MappingProxyType(dict(self._columns))

    def inverse_block(self, columns: Any) -> np.ndarray:
        """``H^-1`` on ``columns`` x ``columns``: ``(k, k)``, a solve per column not solved yet."""
        columns = [int(column) for column in np.ravel(np.asarray(columns, dtype=np.int64))]
        if not columns:
            return np.zeros((0, 0))
        return np.stack([self._column(column)[columns] for column in columns], axis=1)


# ---------------------------------------------------------------------------------------
# The replicate stepper: the fit's joint trust region, run for every population member
# ---------------------------------------------------------------------------------------

# A member's equality rows hold at a point when every row's residual is within this
# fraction of the row's own scale, max(1, |b_i|, sum_j |a_ij x_j|).
_ON_ROWS = 1e-10
# A member's whole numbers and windows, as a checkpoint keeps them.
_COUNTERS = (
    "lock",
    "accepted",
    "attempts",
    "requests",
    "invalid_trials",
    "predicate_rejections",
    "solves",
    "invalid_run",
    "noise_run",
)
_WINDOWS = ("noise_changes", "rejected_changes", "rejected_decreases")


class ReplicateLaneFailure(NamedTuple):
    """One failed lane of a population member, as the member's record logs it.

    Attributes:
        lane: The experiment id of the lane.
        kind: The kind of the failure.
        t_reached: The time the lane's solve reached.
        rungs: The rungs the solve was tried at, in order.
        attempt: The member's attempt, counted from 0, at which the lane failed.
        at: ``"trial"``: at the member's trial point, which then counted as
            an invalid trial; ``"accepted"``: at the point the member had just
            accepted, whose acceptance was then withdrawn.
        ignored: The lane weighs zero: it lies outside the member's
            objective, so its failure is logged and decides nothing.
    """

    lane: str
    kind: str
    t_reached: float
    rungs: tuple
    attempt: int
    at: str
    ignored: bool


class _Settings(NamedTuple):
    """What every member's trust region takes from the solver and from the study."""

    view: Any
    bounds: Any
    fixed: np.ndarray
    x_scale: Any
    grad_tolerance: Any
    step_tol: float
    cost_tol: float
    cost_atol: float
    stationarity_tol: Any
    eta: float
    radius_init: float
    radius_max: float
    max_invalid: int
    relaxed: bool
    noise_limit: Any
    steps: Any
    max_attempts: int


def inherited_noise(fit: Any) -> list:
    """The window of noise-dominated cost changes a replicate starts with: the fit's own.

    The window the fit's state saved, when its state holds one; else a window
    of the one evaluation noise the fit reports, when that is above 0; else
    an empty window.
    """
    state = getattr(fit, "state", None)
    if state is not None and "noise_dominated_cost_change" in state:
        return _saved_window(state, "noise_dominated_cost_change")
    noise = dict(getattr(fit, "info", None) or {}).get("evaluation_noise")
    if noise is not None and np.isfinite(float(noise)) and float(noise) > 0.0:
        return [float(noise)]
    return []


def _label(status: str, reason: Any) -> str:
    """A status and its reason as one label: ``"stalled: attempts"``, or the status alone."""
    return status if reason is None else f"{status}: {reason}"


class _Member:
    """One population member's trust region: the fit's joint attempt, split into its stages.

    ``propose`` is the attempt's host work up to its trial point;
    ``correct`` the consistency re-solve of the lanes it raised at its trial;
    ``judge`` the trial's cost, gain ratio, acceptance and radius;
    ``settle`` the ``blocks`` evaluation at a point it accepted; ``conclude``
    the windows, the runs and the stop tests, in the fit's order. Each reads
    the same helpers, in the same order, as an attempt of a fit.
    """

    def __init__(self, index: int, kind: str, number: int, weights: Any, noise: list):
        self.index = int(index)
        self.kind = kind
        self.number = int(number)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.route: Any = None
        self.point: Any = None
        self.system: Any = None
        self.cost = float("nan")
        self.lane_costs: Any = None
        # The lanes' costs, rows and failures of the blocks result at the member's point.
        self.held: Any = None
        self.scale: Any = None
        self.radius = float("nan")
        self.lock = 0
        self.accepted = 0
        self.attempts = 0
        self.requests = 0
        self.invalid_trials = 0
        self.predicate_rejections = 0
        self.solves = 0
        self.invalid_run = 0
        self.noise_run = 0
        self.accepted_radius: Any = None
        self.grad_threshold: Any = None
        self.noise_changes = list(noise)
        self.rejected_changes: list = []
        self.rejected_decreases: list = []
        self.log: list = []
        self.status: Any = None
        self.reason: Any = None
        self.stop: Any = None
        self.error: Any = None
        self.stationarity = float("nan")
        self.seconds = 0.0
        self.attempt: Any = None
        # The weighted cost of the system at the member's point, kept for a member restored
        # from a checkpoint without its system.
        self.final_cost: Any = None
        # Whether the member starts at a converged fit: one that needs no step there is kept.
        self.start_converged = False

    @property
    def running(self) -> bool:
        """Whether the member has not ended."""
        return self.status is None

    def final(self) -> float:
        """The weighted cost of the member's system at its point; NaN when it never had one."""
        if self.system is not None:
            return float(self.system.cost)
        return float("nan") if self.final_cost is None else float(self.final_cost)

    def rebuild(self, results: Any, failures: tuple) -> None:
        """The member's system at its point, from a ``blocks`` request there.

        Raises:
            ValueError: If a lane of nonzero weight failed in ``results``.
            ArithmeticError: If the system is not finite.
        """
        counted, ignored = self.route.weigh(failures)
        self._log(ignored, "accepted")
        if counted:
            self._log(counted, "accepted")
            raise ValueError(
                f"lane(s) {_failures_text(counted)} failed at the blocks evaluation that formed "
                "this member's system again at its point when the run resumed; every lane of "
                "nonzero weight has to solve at a member's point"
            )
        system = self.route.system(results, self.point, member=self.index)
        if not _finite(*elimination.level_vectors(system), system):
            raise _NotFinite("the level system at the member's point is not finite")
        self.system = system

    def on_rows(self) -> bool:
        """Whether the member's point satisfies its own equality rows."""
        matrix = self.route.constraint_matrix
        if matrix.shape[0] == 0:
            return True
        x = np.asarray(self.point, dtype=np.float64)
        rhs = np.asarray(self.route.constraint_rhs, dtype=np.float64)
        residual = np.abs(matrix @ x - rhs)
        scale = np.maximum(1.0, np.maximum(np.abs(rhs), np.abs(matrix) @ np.abs(x)))
        return bool(np.all(residual <= _ON_ROWS * scale))

    def _log(self, failures: tuple, at: str) -> None:
        for failure in failures:
            self.log.append(
                ReplicateLaneFailure(
                    failure.lane,
                    failure.kind,
                    float(failure.t_reached),
                    tuple(int(rung) for rung in failure.rungs),
                    self.attempts,
                    at,
                    bool(failure.ignored),
                )
            )

    def propose(self, s: _Settings) -> bool:
        """The stationarity test, then the step at the member's radius and its trial point.

        Returns whether the trial goes to the ``value`` request. A member that
        the test ends makes no trial; a trial the predicate rejects is
        rejected unevaluated. A member whose point is off its own equality
        rows is not tested: its step goes to the rows.
        """
        system, x = self.system, self.point
        gradient, diagonal = elimination.level_vectors(system)
        candidate = _scale_candidate(system, s.x_scale, diagonal)
        self.scale = candidate if self.scale is None else np.maximum(self.scale, candidate)
        coordinates = _subproblem(x, gradient, self.scale, s.bounds)
        rows = int(self.route.constraint_matrix.shape[0])
        if s.grad_tolerance is not None:
            stop_gradient = float(
                np.linalg.norm(_bounded_gradient(gradient, coordinates, s.bounds))
            )
            if self.grad_threshold is None:
                self.grad_threshold = s.grad_tolerance.threshold(stop_gradient)
            if not rows and stop_gradient <= self.grad_threshold:
                self.end(s, "gradient_tolerance")
                return False
        zero = elimination.solve_scaled(system, coordinates.scale, 0.0, coordinates.addition)
        stationarity = _stationarity(
            system,
            gradient,
            coordinates.scale,
            zero,
            cost_scale(self.cost, s.cost_tol, s.cost_atol),
            coordinates.addition,
        )
        limit = _stationarity_limit(s.stationarity_tol, s.cost_tol, _median(self.noise_changes))
        if stationarity <= limit and self.on_rows():
            self.solves += 1
            self.end(s, "stationarity", sigma=stationarity)
            return False
        self.attempt = SimpleNamespace(
            trial=None,
            step_radius=self.radius,
            predicted=float("nan"),
            scaled_norm=float("nan"),
            step_norm=float("nan"),
            valid=False,
            raised=np.zeros((0,), dtype=np.int64),
            invalid=False,
            trial_cost=float("inf"),
            accepting=False,
            radius=self.radius,
            system=None,
            held=None,
        )
        trial, scaled_norm, predicted, _, solves = _trial_point(
            system,
            x,
            gradient,
            coordinates,
            self.radius,
            first=zero,
            fixed=s.fixed,
            bounds=s.bounds,
            equality=rows > 0,
        )
        self.solves += solves
        valid = _trial_valid(s.view, trial)
        if not valid:
            self.predicate_rejections += 1
        attempt = self.attempt
        attempt.trial = trial
        attempt.predicted = float(predicted)
        attempt.scaled_norm = scaled_norm
        attempt.step_norm = float(np.linalg.norm(trial - x))
        attempt.valid = valid
        return valid

    def correct(self, results: Any, failed: set) -> None:
        """Replace the current costs of the lanes this member raised at its trial by their re-solve.

        ``results`` is the ``value`` request at the members' current points over
        the lanes raised; a lane that does not solve there keeps its cost. The
        member's current cost changes by the weighted difference.
        """
        row = int(np.flatnonzero(np.asarray(results.members) == self.index)[0])
        costs = np.asarray(results.cost, dtype=np.float64)[row]
        columns = {str(lane): column for column, lane in enumerate(results.lane_ids)}
        change = 0.0
        for position in self.attempt.raised:
            lane = str(self.route.ids[int(position)])
            old = float(self.lane_costs[position])
            new = float("nan") if lane in failed else float(costs[columns[lane]])
            if np.isfinite(new):
                self.lane_costs[position] = new
                change += float(self.route.omega[position]) * (new - old)
        self.cost += change

    def judge(self, s: _Settings, results: Any, failures: tuple) -> None:
        """The trial's cost, its gain ratio, whether it is accepted, and the next radius."""
        attempt = self.attempt
        if attempt.valid:
            # A failed lane of weight zero is outside the objective: logged, ignored.
            counted, ignored = self.route.weigh(failures)
            self._log(ignored, "trial")
            if counted:
                attempt.invalid = True
                self.invalid_trials += 1
                self._log(counted, "trial")
            else:
                attempt.trial_cost = float(
                    self.route.trial_cost(results, attempt.trial, member=self.index)
                )
        _, attempt.accepting, attempt.radius, _ = _judge(
            cost=self.cost,
            trial_cost=attempt.trial_cost,
            predicted=attempt.predicted,
            valid=attempt.valid,
            invalid=attempt.invalid,
            scaled_norm=attempt.scaled_norm,
            radius=self.radius,
            lock=self.lock,
            noise_window=self.noise_changes,
            eta=s.eta,
            radius_max=s.radius_max,
            relaxed=s.relaxed,
            relative_to=cost_scale(self.cost, s.cost_tol, s.cost_atol),
        )

    def settle(self, s: _Settings, results: Any, failures: tuple) -> None:
        """The ``blocks`` evaluation at the point the member accepted.

        A lane of nonzero weight that fails there withdraws the acceptance: the
        member stays at its point with its system, and the trial counts as an
        invalid one (its radius a quarter of the attempt's, the growth lock 2,
        the run of invalid trials extended); the failure is logged at
        ``"accepted"``. Otherwise the member's system there is formed.

        Raises:
            ArithmeticError: If the system at the accepted point is not finite.
        """
        attempt = self.attempt
        counted, ignored = self.route.weigh(failures)
        self._log(ignored, "accepted")
        if counted:
            self._log(counted, "accepted")
            attempt.accepting = False
            attempt.invalid = True
            attempt.trial_cost = float("inf")
            self.invalid_trials += 1
            attempt.radius = _next_radius(
                self.radius,
                float("-inf"),
                attempt.scaled_norm,
                lock=self.lock,
                radius_max=s.radius_max,
                computed=False,
            )
            return
        system = self.route.system(results, attempt.trial, member=self.index)
        if not _finite(*elimination.level_vectors(system), system):
            raise _NotFinite("the level system at the point the member accepted is not finite")
        attempt.system = system
        attempt.held = _held_from(results, self.index)

    def conclude(self, s: _Settings) -> None:
        """The attempt's end: the point, the windows, the runs, then the stop tests in order."""
        attempt = self.attempt
        previous_cost = self.cost
        accepted = bool(attempt.accepting)
        if accepted:
            self.point = attempt.trial
            self.system = attempt.system
            self.cost = float(attempt.system.cost)
            self.held = attempt.held
            self.lane_costs = np.array(attempt.held[0], copy=True)
            self.accepted += 1
        self.radius = attempt.radius
        self.attempts += 1
        self.attempt = None
        # A trial is evaluated when the predicate passed, every lane of nonzero weight
        # solved and the cost is finite.
        after = _after_attempt(
            Trials(
                growth_lock=self.lock,
                rejected_cost_change=self.rejected_changes,
                rejected_predicted_decrease=self.rejected_decreases,
                noise_dominated_cost_change=self.noise_changes,
                accepted_step_radius=self.accepted_radius,
                consecutive_invalid_trials=self.invalid_run,
                consecutive_noise_dominated_trials=self.noise_run,
            ),
            accepted=accepted,
            valid=attempt.valid,
            invalid=attempt.invalid,
            cost=previous_cost,
            trial_cost=attempt.trial_cost,
            predicted=attempt.predicted,
            step_radius=attempt.step_radius,
            cost_tol=s.cost_tol,
            cost_atol=s.cost_atol,
            noise_limit=s.noise_limit,
        )
        trials = after.trials
        self.lock = trials.growth_lock
        self.rejected_changes = trials.rejected_cost_change
        self.rejected_decreases = trials.rejected_predicted_decrease
        self.noise_changes = trials.noise_dominated_cost_change
        self.accepted_radius = trials.accepted_step_radius
        self.invalid_run = trials.consecutive_invalid_trials
        self.noise_run = trials.consecutive_noise_dominated_trials
        # No callback or evaluation budget: the accepted steps end a member at ``steps``,
        # the attempts at ``max_attempts``.
        stop = _stop_after(
            after,
            max_invalid=s.max_invalid,
            step_small=step_small(
                attempt.step_norm, self.point, s.step_tol, tested=after.evaluated
            ),
            accepted_count=self.accepted,
            target_steps=s.steps,
            target_stop="steps",
        )
        if stop is None and self.attempts >= s.max_attempts:
            stop = "attempts"
        if stop is not None:
            self.end(s, stop)

    def _end_numbers(self, s: _Settings) -> tuple:
        """The stationarity number and the gradient the next attempt would test, at the point."""
        system, x = self.system, self.point
        gradient, diagonal = elimination.level_vectors(system)
        candidate = _scale_candidate(system, s.x_scale, diagonal)
        scale = candidate if self.scale is None else np.maximum(self.scale, candidate)
        coordinates = _subproblem(x, gradient, scale, s.bounds)
        zero = elimination.solve_scaled(system, coordinates.scale, 0.0, coordinates.addition)
        sigma = _stationarity(
            system,
            gradient,
            coordinates.scale,
            zero,
            cost_scale(self.cost, s.cost_tol, s.cost_atol),
            coordinates.addition,
        )
        stop_gradient = float(np.linalg.norm(_bounded_gradient(gradient, coordinates, s.bounds)))
        return sigma, stop_gradient

    def end(self, s: _Settings, stop: str, *, sigma: Any = None) -> None:
        """End the member on the event ``stop``, with the status and reason it gives.

        The stationarity number at the end point (``sigma``, else computed
        there as the fit computes it after its loop) labels a step, cost or
        noise stop through the fit's verdict. At the ``steps``-th accepted step
        the member converged when the tests its next attempt would make first
        pass there, else it reached its steps. A member that ends stalled, or
        at its attempt cap, without an accepted step needed none when it
        starts at a converged fit and its weights are the original sample's
        (the fit is its own answer) or its point passes the same tests, with
        the noise its rejected trials measured: it is kept as converged.
        Otherwise it failed: its point is the fit's, which is not a draw.
        """
        stop_gradient = None
        if sigma is None:
            sigma, stop_gradient = self._end_numbers(s)
        self.stationarity = float(sigma)
        self.stop = stop
        if stop in ("stationarity", "gradient_tolerance"):
            status, reason = "converged", None
        elif stop == "steps":
            converged = self.passes_converged_test(s, sigma, stop_gradient)
            status, reason = ("converged", None) if converged else ("steps_reached", None)
        elif stop == "attempts":
            if self.accepted:
                status, reason = "stalled", "attempts"
            else:
                status, reason = (
                    ("converged", None)
                    if self.needs_no_step(s, sigma, stop_gradient)
                    else ("failed", "no_step")
                )
        else:
            verdict, stall = _verdict(
                stop,
                sigma=float(sigma),
                yardstick=max(s.cost_tol, 3.0 * _median(self.noise_changes)),
                lock=self.lock,
            )
            if verdict.converged:
                status, reason = "converged", None
            elif self.accepted:
                status, reason = "stalled", stall
            elif self.needs_no_step(s, sigma, stop_gradient):
                status, reason = "converged", None
            else:
                status, reason = "failed", "no_step"
        self.status, self.reason = status, reason

    def needs_no_step(self, s: _Settings, sigma: float, stop_gradient: Any) -> bool:
        """Whether a member without an accepted step starts at a converged fit that is its answer.

        It is when the fit converged and the member's weights are the original sample's (all
        one), or its point passes the converged tests (``passes_converged_test``).
        """
        if not self.start_converged:
            return False
        if np.all(np.asarray(self.weights, dtype=np.float64) == 1.0):
            return True
        return self.passes_converged_test(s, sigma, stop_gradient, compute=True)

    def passes_converged_test(
        self, s: _Settings, sigma: float, stop_gradient: Any, *, compute: bool = False
    ) -> bool:
        """The tests the member's next attempt would make first: its gradient test, or its stationarity test.

        The stationarity limit takes the noise its rejected trials measured. With ``compute``, a
        gradient not given is computed at the member's point when the gradient test is set.
        """
        limit = _stationarity_limit(s.stationarity_tol, s.cost_tol, _median(self.noise_changes))
        rows = int(self.route.constraint_matrix.shape[0])
        testable = s.grad_tolerance is not None and not rows and self.grad_threshold is not None
        if testable and stop_gradient is None and compute:
            stop_gradient = self._end_numbers(s)[1]
        gradient_test = (
            testable and stop_gradient is not None and stop_gradient <= self.grad_threshold
        )
        return bool(gradient_test or (float(sigma) <= limit and self.on_rows()))

    def fail(self, reason: str, error: Any = None) -> None:
        """End the member failed: its host work raised (``error``), or its gauge or system."""
        if self.attempt is not None:
            self.attempts += 1
            self.attempt = None
        self.status, self.reason, self.stop = "failed", reason, reason
        if error is not None:
            self.error = _described(error)


class ReplicateStepper(Lockstep):
    """The fit's joint trust region for every member of a population at once, in lockstep.

    The population is the replicates, then one member per member of the top
    level that leaves that member out (weight zero on it, one elsewhere).
    Population member ``q`` is replicate ``q`` in every request. Each member
    starts at the fit's point with the level system that the evaluation at the
    fit gave its weights, the rungs the fit holds, the fit's window of
    noise-dominated trials, and a first radius of
    ``min(max(radius_init, |z|), radius_max)``, ``|z|`` its full step's length
    in its scaled coordinates, so that its first trial is its one-step
    estimate. Everything else starts fresh.

    One lockstep iteration is one attempt of every member still running:

    1. per member, on the host: the stationarity test at its point (a member
       off its own equality rows is not tested), else its step at its radius
       and its trial point;
    2. one ``value`` request at the trials;
    3. when a member's trial raised the rung of a lane: one ``value`` request
       at those members' points over the lanes raised, each member's current
       cost corrected for its own raised lanes;
    4. per member: the trial's cost, its gain ratio, its acceptance (every
       lane of nonzero weight solved) and its next radius;
    5. one ``blocks`` request at the points accepted; a lane of nonzero
       weight that fails there withdraws that member's acceptance;
    6. per member: its system at the point accepted, its windows and runs,
       and the fit's stop tests in the fit's order, then the count of accepted
       steps and the cap on attempts.

    A member ends ``converged``, ``steps_reached``, ``stalled`` (for its
    radius, the evaluation noise, invalid trials, or the cap on its attempts)
    or ``failed`` (``no_step``: it ended without converging and without an
    accepted step; ``system``: its system at a point was not finite; ``gauge``:
    every member its gauge averages over weighs zero; ``error``: its host work
    raised, with the message kept). Finished members are masked, never
    removed. Each request's points are the full ``(Q, n)`` array; nothing is
    compiled: every request runs the route's prepared programs and every host
    step the fit's compiled elimination at the fit's shapes.

    Attributes:
        members: The population's members, in population order.
        events: One read-only mapping per lockstep iteration: ``iteration``,
            ``active`` (the members that made an attempt), ``requests`` (per
            request its ``mode``, ``members``, ``lanes``, ``calls`` and
            ``seconds``), ``seconds`` (the iteration's wall seconds),
            ``host_seconds`` (those outside the requests) and ``counts`` (the
            replicates' outcomes so far, ``running`` for those still running).
        seconds: The wall seconds of the loop.
    """

    def __init__(
        self,
        route: PopulationRoute,
        home: PopulationRoute,
        products: Any,
        weights: Any,
        *,
        solver: Any,
        steps: Any,
        max_attempts: int,
        noise: list,
        numbers: Any = None,
        jackknife: Any = None,
        state: Any = None,
        fit_converged: bool = False,
    ):
        """Start every member from the evaluation at the fit, or from a checkpoint's state.

        Args:
            route: The population route every request of the loop goes through.
            home: The population route of the evaluation at the fit, whose
                weighting every member's objective takes.
            products: The evaluation at the fit, its systems kept; ``None``
                with ``state``.
            weights: ``(R, M)``, the replicates' member weights.
            solver: The ``TrustRegion`` whose settings every member takes.
            steps: The accepted steps a replicate takes, or ``"refit"`` for no
                count.
            max_attempts: The attempts a member may make.
            noise: The window of noise-dominated cost changes every member
                starts with.
            fit_converged: Whether the fit the replicates start from
                converged: a replicate that needs no step there is kept.
            numbers: ``(R,)``, the number of the replicate each row of
                ``weights`` is; ``None`` numbers them ``0 .. R - 1``.
            jackknife: The members of the top level whose leave-one-out
                member the population holds after the replicates, ascending;
                ``None`` holds one for every member.
            state: The state a checkpoint saved at the end of an iteration
                (``PopulationCheckpoint.read``): every member continues from
                it, and ``rebuild`` forms the running members' systems.
        """
        weights = np.asarray(weights, dtype=np.float64)
        replicates, count = weights.shape
        if tuple(route.ids) != tuple(home.ids):
            raise ValueError(
                "the population route and the route of the evaluation at the fit hold different "
                "lanes; build both on the same lowered study."
            )
        dtype = np.dtype(home._base.dtype)
        self.settings = _Settings(
            view=home.view,
            bounds=home.bounds,
            fixed=np.asarray(home.fixed, dtype=np.int64),
            x_scale=solver.x_scale,
            **loop_tolerances(
                dtype,
                grad_tol=solver.grad_tol,
                step_tol=solver.step_tol,
                cost_tol=solver.cost_tol,
                cost_atol=solver.cost_atol,
                stationarity_tol=solver.stationarity_tol,
                label=_LABEL,
            )._asdict(),
            eta=float(solver.eta),
            radius_init=float(solver.radius_init),
            radius_max=float(solver.radius_max),
            max_invalid=int(solver.max_consecutive_invalid_trials),
            relaxed=bool(solver.noise_relaxed_ratio),
            noise_limit=_noise_trial_limit(solver, home),
            steps=None if steps == "refit" else int(steps),
            max_attempts=int(max_attempts),
        )
        self.route = route
        self.home = home
        self.replicates = int(replicates)
        self.count = int(count)
        self.events: list = []
        self.seconds = 0.0
        self.iteration = 0
        self.numbers = (
            np.arange(replicates, dtype=np.int64)
            if numbers is None
            else np.asarray(numbers, dtype=np.int64)
        )
        self.jackknife = (
            np.arange(count, dtype=np.int64)
            if jackknife is None
            else np.asarray(jackknife, dtype=np.int64)
        )
        # The level-1 member of every lane, in the order of the route's lanes.
        self.lane_members = np.asarray(home.nodes, dtype=np.int64)[:, 0]
        rows = np.vstack([weights, 1.0 - np.eye(count)[self.jackknife]])
        if state is not None:
            self._restore(state, rows)
            return
        failures = {(failure.kind, failure.index): failure for failure in products.failures}
        at_fit = _held_from(products.results, int(products.requested[0]))
        self.fit_held = at_fit
        settings = self.settings
        members = []
        for index, values in enumerate(rows):
            kind, number = self._identity(index)
            if kind == "replicate":
                system = products.systems[index] if products.systems else None
                length = float(products.scaled_lengths[index])
            else:
                system = products.jackknife_systems[number] if products.jackknife_systems else None
                length = float(products.jackknife_lengths[number])
            member = _Member(index, kind, number, values, noise)
            member.start_converged = bool(fit_converged)
            member.point = np.array(products.point, copy=True)
            member.held = tuple(np.array(part, copy=True) for part in at_fit)
            failure = failures.get((kind, number))
            if failure is not None:
                member.status, member.reason, member.stop = "failed", failure.reason, failure.reason
                member.error = failure.message
            elif system is None:
                raise ValueError(
                    "the evaluation at the fit kept no level system for population member "
                    f"{index}; make it with keep=True."
                )
            else:
                member.route = home._weighted(values)
                member.system = system
                member.cost = float(system.cost)
                member.lane_costs = np.array(at_fit[0], copy=True)
                member.radius = min(max(settings.radius_init, length), settings.radius_max)
            members.append(member)
        # The evaluation at the fit is one request of the whole bootstrap, at one point: it is
        # counted for replicate 0, which carries it when every replicate runs in one population,
        # and for no member of a population without replicate 0 (a shard), so that the counts of
        # the shards' populations put together are those of the one population.
        for index in np.asarray(products.requested, dtype=np.int64):
            member = members[int(index)]
            if member.kind == "replicate" and member.number == 0:
                member.requests += 1
        self.members = members

    def _units(self, positions: Any = None) -> np.ndarray:
        """``(Q, L)``: the units every request evaluates, the lanes of nonzero weight per member.

        A lane whose member of the top level weighs zero in a replicate lies
        outside that replicate's objective, so it is not evaluated for it
        (status ``absent``); ``positions`` keeps the columns of the lanes a
        request names.
        """
        weights = np.stack(
            [np.asarray(member.weights, dtype=np.float64) for member in self.members]
        )
        units = weights[:, self.lane_members] != 0.0
        return units if positions is None else units[:, np.asarray(positions, dtype=np.int64)]

    def _identity(self, index: int) -> tuple:
        """``(kind, number)`` of population member ``index``: a replicate, or the member it leaves out."""
        if index < self.replicates:
            return "replicate", int(self.numbers[index])
        return "jackknife", int(self.jackknife[index - self.replicates])

    def _restore(self, state: Any, rows: np.ndarray) -> None:
        """Every member as a checkpoint saved it at the end of an iteration; no system yet."""
        arrays, texts = state["arrays"], state["texts"]
        saved = np.asarray(arrays["weights"], dtype=np.float64)
        if saved.shape != rows.shape or not np.array_equal(saved, rows):
            raise ValueError(
                "the checkpoint holds another population than this bootstrap's: its members' "
                "weights differ. Resume with the settings the checkpoint was written with."
            )
        self.fit_held = None
        self.iteration = int(state["iteration"])
        members = []
        for index, values in enumerate(rows):
            kind, number = self._identity(index)
            member = _Member(index, kind, number, values, [])
            member.point = np.array(arrays["point"][index], dtype=np.float64, copy=True)
            member.cost = float(arrays["cost"][index])
            member.final_cost = float(arrays["final_cost"][index])
            if bool(arrays["has_lane_costs"][index]):
                member.lane_costs = np.array(arrays["lane_costs"][index], copy=True)
            member.held = (
                np.array(arrays["held_costs"][index], dtype=np.float64, copy=True),
                np.array(arrays["held_rows"][index], dtype=np.float64, copy=True),
                np.array(arrays["held_failed"][index], dtype=bool, copy=True),
            )
            if bool(arrays["has_scale"][index]):
                member.scale = np.array(arrays["scale"][index], copy=True)
            member.radius = float(arrays["radius"][index])
            for name in _COUNTERS:
                setattr(member, name, int(arrays[name][index]))
            for name in ("accepted_radius", "grad_threshold"):
                value = float(arrays[name][index])
                setattr(member, name, None if np.isnan(value) else value)
            lengths = arrays["window_lengths"][index]
            for position, name in enumerate(_WINDOWS):
                window = arrays["windows"][index, position, : int(lengths[position])]
                setattr(member, name, [float(value) for value in window])
            member.log = [
                ReplicateLaneFailure(
                    lane=str(entry["lane"]),
                    kind=str(entry["kind"]),
                    t_reached=float(entry["t_reached"]),
                    rungs=tuple(int(rung) for rung in entry["rungs"]),
                    attempt=int(entry["attempt"]),
                    at=str(entry["at"]),
                    ignored=bool(entry["ignored"]),
                )
                for entry in texts["log"][index]
            ]
            member.status = texts["status"][index]
            member.reason = texts["reason"][index]
            member.stop = texts["stop"][index]
            member.error = texts["error"][index]
            member.stationarity = float(arrays["stationarity"][index])
            member.seconds = float(arrays["seconds"][index])
            if member.running:
                member.route = self.home._weighted(values)
            members.append(member)
        self.members = members
        self.route.rungs[...] = np.asarray(arrays["rungs"], dtype=np.int64)
        self.route.first_steps[...] = np.asarray(arrays["first_steps"], dtype=np.float64)
        ledger = getattr(self.route.study, "_ledger", None)
        if ledger is not None:
            ledger.restore(arrays)

    def rebuild(self) -> None:
        """One ``blocks`` request at the running members' points: their systems, after a restore.

        Every running member takes part and counts the request; its system at
        its point is formed from it. The first steps the members' lanes hold
        stay those the checkpoint saved wherever the request raised no rung. A
        member whose lane of nonzero weight fails there, or whose system is
        not finite, ends failed, as for any host work that raises.
        """
        running = [member for member in self.members if member.running]
        if not running:
            return
        route = self.route
        rungs, first_steps = route.rungs.copy(), route.first_steps.copy()
        results = route.evaluate(
            "blocks", self._points(), active=self._mask(running), units=self._units()
        )
        same = route.rungs == rungs
        route.first_steps[same] = first_steps[same]
        lost = route._failures(results)
        for member in running:
            member.requests += 1
            self._host(
                member,
                lambda member=member: member.rebuild(results, lost.get(member.index, ())),
            )

    def state(self) -> tuple:
        """``(arrays, texts)``: every member's state at the end of the last iteration.

        ``arrays`` holds numbers stacked over the population (each window
        padded with NaN to the window's length, with its length beside it);
        ``texts`` holds the statuses, reasons, stops, errors and failure logs.
        What a member's system was is not kept: ``rebuild`` forms it again.
        """
        members = self.members
        lanes, n = len(self.route.ids), int(self.route.n)

        def column(read: Any, dtype: Any = np.float64) -> np.ndarray:
            return np.asarray([read(member) for member in members], dtype=dtype)

        def rows(read: Any, width: int) -> np.ndarray:
            stacked = [
                np.full((width,), np.nan) if read(member) is None else read(member)
                for member in members
            ]
            return np.stack([np.asarray(row, dtype=np.float64) for row in stacked])

        longest = max(
            [_WINDOW] + [len(getattr(member, name)) for member in members for name in _WINDOWS]
        )
        windows = np.full((len(members), len(_WINDOWS), longest), np.nan)
        lengths = np.zeros((len(members), len(_WINDOWS)), dtype=np.int64)
        for row, member in enumerate(members):
            for position, name in enumerate(_WINDOWS):
                window = list(getattr(member, name))
                windows[row, position, : len(window)] = window
                lengths[row, position] = len(window)
        arrays = {
            "weights": np.stack([member.weights for member in members]),
            "point": np.stack([np.asarray(member.point, dtype=np.float64) for member in members]),
            "cost": column(lambda member: member.cost),
            "final_cost": column(lambda member: member.final()),
            "has_lane_costs": column(lambda member: member.lane_costs is not None, bool),
            "lane_costs": rows(lambda member: member.lane_costs, lanes),
            "held_costs": rows(lambda member: member.held[0], lanes),
            "held_rows": rows(lambda member: member.held[1], lanes),
            "held_failed": np.stack([np.asarray(member.held[2], dtype=bool) for member in members]),
            "has_scale": column(lambda member: member.scale is not None, bool),
            "scale": rows(lambda member: member.scale, n),
            "radius": column(lambda member: member.radius),
            "accepted_radius": column(
                lambda member: np.nan if member.accepted_radius is None else member.accepted_radius
            ),
            "grad_threshold": column(
                lambda member: np.nan if member.grad_threshold is None else member.grad_threshold
            ),
            "windows": windows,
            "window_lengths": lengths,
            "stationarity": column(lambda member: member.stationarity),
            "seconds": column(lambda member: member.seconds),
            "rungs": np.array(self.route.rungs, dtype=np.int64, copy=True),
            "first_steps": np.array(self.route.first_steps, dtype=np.float64, copy=True),
        }
        for name in _COUNTERS:
            arrays[name] = column(lambda member, name=name: getattr(member, name), np.int64)
        # What the lanes cost so far, for the placement of a later run; never where they ran.
        ledger = getattr(self.route.study, "_ledger", None)
        if ledger is not None:
            arrays.update(ledger.entries())
        fingerprint = getattr(self.route.study, "_plan_fingerprint", None)
        value = fingerprint() if callable(fingerprint) else None
        if value is not None:
            arrays["ledger.plan_fingerprint"] = np.asarray(value)
        texts = {
            "status": [member.status for member in members],
            "reason": [member.reason for member in members],
            "stop": [member.stop for member in members],
            "error": [member.error for member in members],
            "log": [[dict(failure._asdict()) for failure in member.log] for member in members],
        }
        return arrays, texts

    def counts(self) -> dict:
        """The replicates' outcomes so far, ``running`` for those still running."""
        counts: dict = {}
        for member in self.members[: self.replicates]:
            label = "running" if member.running else _label(member.status, member.reason)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def iterate(self, running: list) -> None:
        """One lockstep iteration: one attempt of every member of ``running``, in the six stages."""
        s, route = self.settings, self.route
        # (1) The stationarity test, the step and the trial, per member.
        attempting, trials = [], []
        for member in running:
            valid = self._host(member, lambda member=member: member.propose(s))
            if member.running:
                attempting.append(member)
                if valid:
                    trials.append(member)
        # (2) One value request at the trials.
        results, failures = None, {}
        if trials:
            points = self._points()
            for member in trials:
                points[member.index] = member.attempt.trial
            before = route.rungs.copy()
            results = route.evaluate(
                "value", points, active=self._mask(trials), units=self._units()
            )
            failures = route._failures(results)
            for member in trials:
                member.requests += 1
                member.attempt.raised = np.flatnonzero(
                    route.rungs[member.index] > before[member.index]
                )
            # (3) One value request at the current points over the lanes raised.
            raising = [member for member in trials if member.attempt.raised.size]
            if raising:
                positions = sorted(
                    {int(position) for member in raising for position in member.attempt.raised}
                )
                again = route.evaluate(
                    "value",
                    self._points(),
                    active=self._mask(raising),
                    lanes=tuple(route.ids[position] for position in positions),
                    units=self._units(positions),
                )
                failed: dict = {}
                for entry in again.failures():
                    failed.setdefault(int(entry["member"]), set()).add(str(entry["lane"]))
                for member in raising:
                    member.requests += 1
                    self._host(
                        member,
                        lambda member=member: member.correct(
                            again, failed.get(member.index, set())
                        ),
                    )
        # (4) The trial's cost, its gain ratio, the acceptance and the radius.
        for member in attempting:
            if member.running:
                self._host(
                    member,
                    lambda member=member: member.judge(s, results, failures.get(member.index, ())),
                )
        # (5) One blocks request at the points accepted.
        accepting = [member for member in attempting if member.running and member.attempt.accepting]
        if accepting:
            points = self._points()
            for member in accepting:
                points[member.index] = member.attempt.trial
            blocks = route.evaluate(
                "blocks", points, active=self._mask(accepting), units=self._units()
            )
            lost = route._failures(blocks)
            for member in accepting:
                member.requests += 1
                self._host(
                    member,
                    lambda member=member: member.settle(s, blocks, lost.get(member.index, ())),
                )
        # (6) The attempt's end and the stop tests.
        for member in attempting:
            if member.running:
                self._host(member, lambda member=member: member.conclude(s))

    # What the result reports ----------------------------------------------------------

    def columns(self) -> dict:
        """Every member's record, one entry per population member, as columns.

        ``kind``, ``index`` (the replicate's number, or the member left out),
        ``status``, ``reason``, ``stop``, ``error``, ``attempts``,
        ``accepted``, ``requests``, ``invalid_trials``, ``stationarity``,
        ``radius``, ``active_bounds``, ``failed_lanes``, ``points``,
        ``final_cost``, ``data_cost``, ``data_rows``, ``training_rmse`` and
        ``host_seconds``.
        """
        members, count = self.members, self.count
        bounds = self.home.bounds
        data_cost = np.full((len(members), count), np.nan)
        data_rows = np.full((len(members), count), np.nan)
        rmse = np.full((len(members),), np.nan)
        for row, member in enumerate(members):
            cost, rows = _member_data(self.lane_members, count, member.held)
            data_cost[row], data_rows[row] = cost, rows
            rmse[row] = _training_rmse(member.weights, cost, rows)
        return {
            "kind": tuple(member.kind for member in members),
            "index": np.asarray([member.number for member in members], dtype=np.int64),
            "status": tuple(member.status for member in members),
            "reason": tuple(member.reason for member in members),
            "stop": tuple(member.stop for member in members),
            "error": tuple(member.error for member in members),
            "attempts": np.asarray([member.attempts for member in members], dtype=np.int64),
            "accepted": np.asarray([member.accepted for member in members], dtype=np.int64),
            "requests": np.asarray([member.requests for member in members], dtype=np.int64),
            "invalid_trials": np.asarray(
                [member.invalid_trials for member in members], dtype=np.int64
            ),
            "stationarity": np.asarray(
                [member.stationarity for member in members], dtype=np.float64
            ),
            "radius": np.asarray([member.radius for member in members], dtype=np.float64),
            "active_bounds": tuple(
                tuple(box.active_bounds(member.point, bounds)) for member in members
            ),
            "failed_lanes": tuple(tuple(member.log) for member in members),
            "points": np.stack([np.asarray(member.point, dtype=np.float64) for member in members]),
            "final_cost": np.asarray([member.final() for member in members], dtype=np.float64),
            "data_cost": data_cost,
            "data_rows": data_rows,
            "training_rmse": rmse,
            "host_seconds": np.asarray([member.seconds for member in members], dtype=np.float64),
        }


def _member_data(lane_members: np.ndarray, count: int, held: tuple) -> tuple:
    """``(C, N)``: per member of the top level its lanes' unweighted cost and rows.

    A member with a failed lane (one of weight zero, ignored) has NaN for both.
    """
    costs, rows, failed = held
    cost = np.bincount(lane_members, weights=costs, minlength=count).astype(np.float64)
    total = np.bincount(lane_members, weights=rows, minlength=count).astype(np.float64)
    broken = np.bincount(lane_members, weights=failed.astype(np.float64), minlength=count) > 0
    cost[broken] = np.nan
    total[broken] = np.nan
    return cost, total


def _training_rmse(weights: np.ndarray, cost: np.ndarray, rows: np.ndarray) -> float:
    """``sqrt(sum_m w_m 2 C_m / sum_m w_m N_m)`` over the members of nonzero weight."""
    used = np.asarray(weights, dtype=np.float64) > 0.0
    if not np.any(used):
        return float("nan")
    w = np.asarray(weights, dtype=np.float64)[used]
    denominator = float(np.sum(w * rows[used]))
    if not denominator > 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(w * 2.0 * cost[used]) / denominator))


def fit_training_rmse(products: FitProducts, lane_members: np.ndarray, count: int) -> float:
    """The fit's training RMSE: every member at weight one, from the evaluation at the fit."""
    held = _held_from(products.results, int(products.requested[0]))
    cost, rows = _member_data(lane_members, count, held)
    return _training_rmse(np.ones((count,)), cost, rows)
