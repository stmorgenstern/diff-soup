"""The tolerances a trust-region loop tests, the state it carries, and the rules of an attempt.

The rules of one attempt: its gain ratio, acceptance and next radius; the growth
lock, the windows and the runs after it; the tests that end the loop. All of them
take and return plain numbers and host arrays, whatever objective the loop minimizes.

Every relative number the rules read is relative to the cost scale
``max(|cost|, cost_atol / cost_tol)`` (``cost_scale``): the cost itself, never
below the scale at which ``cost_tol`` reaches the absolute tolerance
``cost_atol``. The stationarity number, the windows of rejected trials, the
evaluation noise, the gradient floor's cost equivalent and the noise-relaxed
gain ratio all divide or multiply by it, so where a fit stops does not depend
on the units of its residual rows, and a relative number at ``cost_tol`` is a
cost change at the cost threshold ``max(cost_tol * |cost|, cost_atol)``
(``cost_threshold``). With ``cost_atol`` 0 (the default) the scale is the cost
itself: a fit to exact data, whose cost falls towards zero, then ends by its
step or gradient test rather than by a cost or stationarity test.

The loop also learns the accuracy floor of the model's gradient from its own small
rejected trials, at no cost in evaluations. A floor sample is such a trial's mismatch
between its actual and predicted decrease per unit of scaled step length, kept when a
gradient error within an a priori bound can explain it. The floor is the median of
the latest samples, capped by the bound. It keeps the radius from shrinking below the
scale at which the model's predicted decrease is still information, and a fit ends at
the accuracy of its derivatives when the model's full step predicts no more decrease
than the floor could take back along it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import numpy as np

from diff_soup._distributed import local_value
from diff_soup.inference.solvers._termination import (
    DEFAULT_COST_ATOL,
    DEFAULT_COST_TOL,
    DEFAULT_STEP_TOL,
    GradientTolerance,
    absolute_tolerance,
    gradient_tolerance,
    relative_tolerance,
)


class LoopTolerances(NamedTuple):
    """A trust-region loop's tolerances, resolved for its decision vector's dtype.

    ``grad_tolerance`` is ``None`` for no gradient test. ``stationarity_tol``
    is kept as given; ``None`` stands for the yardstick ``max(cost_tol, 3 * noise)``,
    a stationarity number relative to the cost scale (``cost_scale``).
    """

    grad_tolerance: GradientTolerance | None
    step_tol: float
    cost_tol: float
    cost_atol: float
    stationarity_tol: Any


def loop_tolerances(
    dtype, *, grad_tol, step_tol, cost_tol, cost_atol, stationarity_tol, label: str
) -> LoopTolerances:
    """Resolve a trust-region loop's tolerance options for a decision vector of ``dtype``.

    ``grad_tol`` (``None`` tests no gradient), ``step_tol``, ``cost_tol`` and
    ``cost_atol`` are resolved in that order by ``gradient_tolerance``,
    ``relative_tolerance`` and ``absolute_tolerance``; ``stationarity_tol`` is
    kept as given. ``label`` names the optimizer in their warnings and refusal.

    Raises:
        ConfigurationError: If ``cost_atol`` is negative or not a finite number.
    """
    return LoopTolerances(
        None if grad_tol is None else gradient_tolerance(grad_tol, dtype, label=label),
        relative_tolerance(step_tol, DEFAULT_STEP_TOL, dtype, name="step_tol", label=label),
        relative_tolerance(cost_tol, DEFAULT_COST_TOL, dtype, name="cost_tol", label=label),
        absolute_tolerance(cost_atol, DEFAULT_COST_ATOL, name="cost_atol", label=label),
        stationarity_tol,
    )


def _saved_window(state: Any, key: str) -> list:
    """A window of rejected trials kept in a saved state; empty when the state has none."""
    if state is None:
        return []
    try:
        values = state[key]
    except (KeyError, TypeError):
        return []
    return [float(value) for value in np.ravel(np.asarray(values, dtype=np.float64))]


def _saved_value(state: Any, key: str) -> Any:
    """One entry of a saved state as a Python scalar, or ``None`` when it is absent."""
    try:
        value = state[key]
    except (KeyError, TypeError):
        return None
    return np.asarray(value).item()


def _saved_count(state: Any, key: str) -> int:
    """A count kept in a saved state; 0 when the state has none."""
    value = _saved_value(state, key)
    return 0 if value is None else int(value)


def _saved_radius(state: Any, key: str) -> Any:
    """A radius kept in a saved state; ``None`` when the state has none or holds ``nan``."""
    value = _saved_value(state, key)
    return None if value is None or not np.isfinite(value) else float(value)


@dataclasses.dataclass(kw_only=True)
class LoopState:
    """What a trust-region loop hands from one attempt to the next, each field saved by name.

    ``attempt`` counts the attempts made. ``scale``, ``grad_threshold`` and
    ``accepted_step_radius`` are ``None`` until the first attempt, gradient
    test and accepted step set them. The windows hold the last rejected and
    noise-dominated trials' changes of the objective (and the rejected ones'
    predicted decreases) relative to the cost scale at the attempt
    (``cost_scale``); the runs count
    invalid and noise-dominated rejected trials in a row.
    ``gradient_floor_samples`` is the window of the last floor samples
    (``_floor_sample``), in cost per unit of scaled step length; it is empty
    while no trial has been admitted as one. ``record`` is the attempt just
    made, saved as the entries ``record.<name>``.
    """

    attempt: int = 0
    accepted: int = 0
    point: Any
    scale: Any
    cost: float
    grad_threshold: Any = None
    radius: float
    growth_lock: int = 0
    cost_history: list = dataclasses.field(default_factory=list)
    grad_history: list = dataclasses.field(default_factory=list)
    radius_history: list
    rejected_cost_change: list = dataclasses.field(default_factory=list)
    rejected_predicted_decrease: list = dataclasses.field(default_factory=list)
    noise_dominated_cost_change: list = dataclasses.field(default_factory=list)
    accepted_step_radius: Any = None
    consecutive_invalid_trials: int = 0
    consecutive_noise_dominated_trials: int = 0
    gradient_floor_samples: list = dataclasses.field(default_factory=list)
    record: dict

    @classmethod
    def start(cls, *, radius: float, **values: Any) -> LoopState:
        """A fresh loop from ``point``, ``cost``, ``scale``, ``radius`` and ``record``.

        No attempt is made; the radius history holds ``radius``, and the other
        histories, the windows and the runs are empty.
        """
        return cls(radius=radius, radius_history=[radius], **values)

    @classmethod
    def restore(cls, state: Any, *, point: Any, cost: float, scale: Any) -> LoopState:
        """The loop that the saved ``state`` continues, at ``point`` with ``cost`` and ``scale``.

        A state saved without the windows (the floor samples' included) resumes
        with empty ones, without the radius of the last accepted step as if no
        step had been accepted, and without the runs with none.
        """
        threshold = float(np.asarray(state["grad_threshold"]))
        return cls(
            attempt=int(np.asarray(state["attempt"])),
            accepted=int(np.asarray(state["accepted"])),
            point=point,
            scale=scale,
            cost=cost,
            grad_threshold=threshold if np.isfinite(threshold) else None,
            radius=float(np.asarray(state["radius"])),
            growth_lock=int(np.asarray(state["growth_lock"])),
            cost_history=[float(value) for value in np.ravel(state["cost_history"])],
            grad_history=[float(value) for value in np.ravel(state["grad_history"])],
            radius_history=[float(value) for value in np.ravel(state["radius_history"])],
            rejected_cost_change=_saved_window(state, "rejected_cost_change"),
            rejected_predicted_decrease=_saved_window(state, "rejected_predicted_decrease"),
            noise_dominated_cost_change=_saved_window(state, "noise_dominated_cost_change"),
            accepted_step_radius=_saved_radius(state, "accepted_step_radius"),
            consecutive_invalid_trials=_saved_count(state, "consecutive_invalid_trials"),
            consecutive_noise_dominated_trials=_saved_count(
                state, "consecutive_noise_dominated_trials"
            ),
            gradient_floor_samples=_saved_window(state, "gradient_floor_samples"),
            record={
                key[len("record.") :]: np.array(value)
                for key, value in state.items()
                if key.startswith("record.")
            },
        )

    def entries(self) -> dict:
        """The saved form: each field as a new host array under its name, the record's last.

        A field annotated ``int`` is saved as ``int64``, every other as
        ``float64`` (``None`` as ``nan``); the record's entries as
        ``record.<name>``.
        """
        entries = {
            field.name: np.array(
                getattr(self, field.name),
                dtype=np.int64 if field.type in ("int", int) else np.float64,
            )
            for field in dataclasses.fields(self)
            if field.name != "record"
        }
        entries.update({f"record.{key}": value for key, value in self.record.items()})
        return entries


# The number of rejected trials whose cost change and predicted decrease a fit keeps,
# for the report of a fit that ends at max_steps; also the number of noise-dominated
# rejected trials whose cost changes give the evaluation noise the fit reports.
_WINDOW = 10

# The number of noise-dominated rejected trials in a row that ends a loop: the model's
# predictions at that radius are below the noise of the cost's evaluation.
_NOISE_RUN = 3

_TINY = float(np.finfo(float).tiny)


def cost_scale(cost: Any, cost_tol: float, cost_atol: float) -> Any:
    """The yardstick of a loop's relative numbers: ``max(|cost|, cost_atol / cost_tol)``.

    A cost-unit quantity (a cost change, a predicted decrease, a noise level)
    divided by it is relative to the cost itself, never to less than the cost
    at which ``cost_tol`` reaches ``cost_atol``; a relative number at
    ``cost_tol`` is then a cost-unit quantity at ``cost_threshold``. With
    ``cost_tol`` 0 the floor is 0, and the scale is never below the smallest
    positive double, so a cost of exactly 0 divides nothing by zero. Works on
    a float and on an array of costs alike.
    """
    floor = max(float(cost_atol) / float(cost_tol) if cost_tol > 0.0 else 0.0, _TINY)
    scale = np.maximum(np.abs(np.asarray(cost, dtype=np.float64)), floor)
    return float(scale) if scale.ndim == 0 else scale


def cost_threshold(cost: float, cost_tol: float, cost_atol: float) -> float:
    """The largest cost change the cost test accepts at ``cost``: ``max(cost_tol * |cost|, cost_atol)``.

    Relative to the cost itself, so where a fit stops does not depend on the
    units of its residual rows; ``cost_atol`` is the absolute floor.
    """
    return max(float(cost_tol) * abs(float(cost)), float(cost_atol))


def cost_small(
    *, accepted: bool, evaluated: bool, cost_change: float, predicted: float, threshold: float
) -> bool:
    """The cost test of a trial: whether it ends its loop at the cost threshold ``threshold``.

    An ``accepted`` trial whose cost changed by at most ``threshold`` ends it,
    and so does an ``evaluated`` one (its cost computed, accepted or rejected)
    whose cost change and ``predicted`` decrease both are at most it.
    """
    return bool(
        (accepted and cost_change <= threshold)
        or (evaluated and cost_change <= threshold and predicted <= threshold)
    )


def step_small(step_norm: float, point: Any, step_tol: float, *, tested: bool = True) -> bool:
    """The step test of a trial: ``tested`` and ``step_norm <= step_tol * (1 + ||point||)``.

    ``point`` is the point the loop holds after the trial; ``tested`` says
    whether the trial takes the test at all (a loop tests a trial whose cost
    was computed, or one the predicate passed when the test comes before the
    evaluation).
    """
    return bool(tested and step_norm <= step_tol * (1.0 + float(np.linalg.norm(point))))


def trial_valid(problem: Any, point: Any) -> bool:
    """Whether ``problem.is_valid(point)`` holds, read on the host.

    Raises:
        ValueError: If the predicate does not return one scalar boolean.
    """
    value = np.asarray(local_value(problem.is_valid(point)))
    if value.shape != ():
        raise ValueError(
            f"problem.is_valid(point) must return one scalar boolean; got shape {value.shape}."
        )
    return bool(value)


def _next_radius(
    radius: float,
    rho: float,
    scaled_norm: float,
    *,
    lock: int,
    radius_max: Any,
    computed: bool,
    floor: float = 0.0,
) -> float:
    """The trust-region radius after an attempt.

    On a trial whose cost was computed, a gain ratio under 0.25 sets the radius
    to a quarter of the trial's scaled step norm; a gain ratio above 0.75 with
    the growth lock 0 and the scaled step at least 0.95 of the radius doubles
    it, up to ``radius_max``; otherwise it stays. A trial the predicate
    rejected or whose objective could not be evaluated quarters it.

    A gain ratio under 0.25 never shrinks the radius below ``floor``, the
    radius at which the model's predicted decrease falls to what the
    gradient's error could take back (``_floor_radius``), nor does the floor
    raise it above ``radius``; ``floor`` 0 is the classical rule. Following
    Sun and Nocedal (2023), the radius stays bounded below while the gradient
    exceeds its error, since shrinking it on rejections that only measure that
    error is how the classical method fails on an inexact objective. A trial
    whose objective was not computed says nothing of the gradient and
    quarters the radius regardless of ``floor``.
    """
    if not computed:
        return max(radius / 4.0, _TINY)
    if not rho >= 0.25:
        return max(0.25 * scaled_norm, min(floor, radius), _TINY)
    if rho > 0.75 and lock == 0 and scaled_norm >= 0.95 * radius:
        return min(2.0 * radius, float(radius_max))
    return radius


# The factor of the noise-relaxed gain ratio: Sun and Nocedal's 2 / (1 - c) at c = 1/2.
_RELAXATION = 4.0


def _gain_ratio(
    actual: Any, predicted: Any, *, noise: float, relative_to: Any, relaxed: bool
) -> Any:
    """The gain ratio of a trial: its cost decrease ``actual`` over the model's ``predicted``.

    With ``relaxed`` and a measured evaluation noise ``noise`` above 0 (relative
    to the cost scale ``relative_to``, ``cost_scale``), both are raised by
    ``4 * noise * relative_to``, the noise in cost units times four, so that a
    trial whose predicted decrease is of the size of that noise is not
    rejected, and does not shrink the radius, for the noise alone; otherwise
    the classical ratio ``actual / predicted``. Works on floats and on arrays
    alike.
    """
    if relaxed and noise > 0.0:
        level = _RELAXATION * noise * relative_to
        return (actual + level) / (predicted + level)
    return actual / predicted


def _noise_dominated(
    *,
    evaluated: bool,
    accepted: bool,
    step_radius: float,
    accepted_radius: Any,
    cost: float,
    trial_cost: float,
    predicted: float,
    relative_to: float,
    limit: Any = None,
) -> bool:
    """Whether a trial's cost change measures the cost's evaluation jitter.

    The trial was evaluated and rejected, its step was solved at a radius
    below the one at which the last accepted step was solved, and its cost
    rose or changed by more than the model predicted. On a smooth cost a step
    that small along the model's descent direction does not raise the cost,
    so such a change is the jitter of its evaluation. Before the first
    accepted step no trial qualifies. When ``limit`` is a number, the
    decrease the model predicted, relative to the cost scale ``relative_to``
    (``cost_scale``), must also be at most ``limit``; ``None`` sets no such
    bound.
    """
    return bool(
        evaluated
        and not accepted
        and accepted_radius is not None
        and step_radius < accepted_radius
        and (trial_cost > cost or abs(cost - trial_cost) > predicted)
        and (limit is None or predicted / relative_to <= limit)
    )


def _median(window: list) -> float:
    """The median of a window, 0 when it is empty."""
    return float(np.median(np.asarray(window, dtype=np.float64))) if window else 0.0


def _jump_level(changes: list, decreases: list, ceiling: Any) -> Any:
    """The median cost jump of the rejected trials, or ``None`` when one of them was no jump.

    ``changes`` and ``decreases`` are the window of rejected trials: the size
    of each one's cost change and the decrease its model predicted, both
    relative to the cost scale at its attempt (``cost_scale``), in the units
    of ``ceiling``. A rejected trial met a jump of the computed
    cost when the model predicted a decrease and the cost rose by more than
    that, by at most ``ceiling``, the relative noise the solvers' tolerance
    allows the cost's evaluation. Such a rise says nothing of the model: at
    any shorter step the computed cost can jump as far, so a decrease of a
    fraction of the jump is beyond what a trial can show. ``None`` when the
    window is empty, ``ceiling`` is ``None``, or a trial in it was no jump.
    """
    if ceiling is None or not changes:
        return None
    rises = np.asarray(changes, dtype=np.float64)
    predicted = np.asarray(decreases, dtype=np.float64)
    if not bool(np.all((predicted > 0.0) & (rises > predicted) & (rises <= float(ceiling)))):
        return None
    return float(np.median(rises))


def _floor_sample(
    *,
    evaluated: bool,
    accepted: bool,
    step_radius: float,
    accepted_radius: Any,
    cost: float,
    trial_cost: float,
    predicted: float,
    scaled_norm: float,
    noise: float,
    relative_to: float,
    bound: Any,
) -> float | None:
    """A rejected trial's measure of the gradient's error, or ``None`` when it gives none.

    The model's predicted decrease is linear in its gradient, so a gradient in
    error by ``e`` per unit of scaled step length misjudges the decrease of a
    step of scaled norm ``scaled_norm`` by up to ``e * scaled_norm``. A trial
    is a floor sample when it was evaluated and rejected, its step was solved
    at a radius below the one at which the last accepted step was solved
    (before the first accepted step no trial qualifies), the model
    ``predicted`` a decrease, its step is not zero, ``bound`` is a number,
    and the mismatch ``m = |(cost - trial_cost) - predicted|`` of the actual
    and the predicted decrease is at most ``noise_abs + bound * scaled_norm``.
    ``noise_abs = noise * relative_to`` is the cost's evaluation noise in
    cost units (``noise`` is relative to the cost scale ``relative_to``,
    ``cost_scale``) and ``bound`` the
    largest error per unit of scaled step length the derivatives' accuracy
    allows the gradient. The sample is ``max(m - noise_abs, 0) / scaled_norm``,
    in cost per unit of scaled step length.

    A larger mismatch is not the gradient's: no gradient error within the bound
    produces it at that step length, so it is a jump of the computed values,
    which the evaluation noise accounts for, and gives no sample.
    """
    if not (
        evaluated
        and not accepted
        and accepted_radius is not None
        and step_radius < accepted_radius
        and predicted > 0.0
        and scaled_norm > 0.0
        and bound is not None
    ):
        return None
    noise_abs = noise * relative_to
    mismatch = abs((cost - trial_cost) - predicted)
    if not mismatch <= noise_abs + bound * scaled_norm:
        return None
    return float(max(mismatch - noise_abs, 0.0) / scaled_norm)


def _floor_level(samples: list, bound: Any) -> Any:
    """The gradient's accuracy floor: the median of the floor ``samples``, at most ``bound``.

    In cost per unit of scaled step length. ``None`` when there is no sample or
    no bound: the bound alone is an upper limit with a margin, not a measurement.
    """
    if bound is None or len(samples) == 0:
        return None
    return min(_median(samples), float(bound))


def _floor_sigma(level: Any, full_step_norm: float, relative_to: float) -> float:
    """The floor's cost equivalent along the model's full step, relative to the cost scale.

    ``level * full_step_norm`` is what a gradient in error by ``level`` along
    the model's full step, of scaled norm ``full_step_norm``, could take back
    of the decrease the step predicts. Over the cost scale ``relative_to``
    (``cost_scale``) it is in the stationarity number's units: a point whose
    stationarity number is at most it is as stationary as the derivatives can
    tell. 0 when ``level`` is ``None``.
    """
    if level is None:
        return 0.0
    return level * full_step_norm / relative_to


def _floor_radius(level: Any, full_step_norm: float, gradient_norm: float) -> float:
    """The radius at which a step along the gradient predicts the floor's cost equivalent.

    ``level * full_step_norm / gradient_norm``, with ``gradient_norm`` the
    gradient's norm in the trust region's scaled coordinates: the Cauchy
    decrease ``gradient_norm * radius`` equals ``level * full_step_norm`` there,
    and a smaller radius predicts decreases the gradient's accuracy does not
    resolve. 0 when ``level`` is ``None`` or the gradient is zero.
    """
    if level is None or not gradient_norm > 0.0:
        return 0.0
    return level * full_step_norm / gradient_norm


def _floor_parameter_radius(
    level: Any, full_step_norm: float, full_step_length: float, predicted: float
) -> Any:
    """How far, in parameter units, the gradient's error moves the minimizer along the full step.

    Along the model's full step, of scaled norm ``full_step_norm``, length
    ``full_step_length`` in parameter units and predicted decrease
    ``predicted``, the model's curvature is ``2 * predicted / full_step_norm**2``,
    so a gradient in error by ``level`` moves the minimizer along it by
    ``level * full_step_norm**2 / (2 * predicted)`` in scaled units, that is
    ``full_step_length * level * full_step_norm / (2 * predicted)`` in
    parameter units. A bound along one direction, not the trust radius.
    ``None`` when ``level`` is ``None``; NaN when ``predicted`` is not positive.
    """
    if level is None:
        return None
    if not predicted > 0.0:
        return float("nan")
    return full_step_length * level * full_step_norm / (2.0 * predicted)


def _judge(
    *,
    cost: float,
    trial_cost: float,
    predicted: float,
    valid: bool,
    invalid: bool,
    scaled_norm: float,
    radius: float,
    lock: int,
    noise_window: list,
    eta: float,
    radius_max: Any,
    relaxed: bool,
    relative_to: float,
    hold: bool = False,
    radius_floor: float = 0.0,
) -> tuple:
    """A trial's gain ratio, acceptance, next radius and rejection reason, at objective ``cost``.

    The ratio is ``-inf`` for an ``invalid`` trial (its objective not
    evaluated) or a ``predicted`` decrease of at most 0, else ``_gain_ratio``
    with the median of ``noise_window`` and the cost scale ``relative_to``
    (``cost_scale`` at ``cost``); a ``valid`` trial (one the predicate
    passed) is accepted at ``eta`` or above. With ``hold`` (a step within the
    step tolerance, not evaluated) the ratio is NaN, the trial rejected and
    the radius kept. A gain ratio under 0.25 shrinks the radius to no less
    than the smaller of ``radius_floor`` and ``radius`` (``_next_radius``'s
    ``floor``). Returns ``(rho, accepted, radius, reason)``.
    """
    computed = bool(valid and not invalid)
    if hold:
        rho = float("nan")
    elif invalid or predicted <= 0.0:
        rho = float("-inf")
    else:
        noise = _median(noise_window)
        rho = float(
            _gain_ratio(
                cost - trial_cost,
                predicted,
                noise=noise,
                relative_to=relative_to,
                relaxed=relaxed,
            )
        )
    accepted = bool(computed and not hold and rho >= eta)
    if accepted:
        reason = None
    elif predicted <= 0.0:
        reason = "the model predicted no decrease"
    else:
        reason = f"the gain ratio {rho:.3g} is below eta {eta:.3g}"
    if not hold:
        radius = _next_radius(
            radius,
            rho,
            scaled_norm,
            lock=lock,
            radius_max=radius_max,
            computed=computed,
            floor=radius_floor,
        )
    return rho, accepted, radius, reason


class Trials(NamedTuple):
    """What a loop keeps of its trials, under the names of ``LoopState``'s fields.

    ``gradient_floor_samples`` may be left out by a loop that keeps no floor
    samples; it then holds the empty list every such instance shares, which
    nothing modifies in place, as each update of a window builds a new list.
    """

    growth_lock: int
    rejected_cost_change: list
    rejected_predicted_decrease: list
    noise_dominated_cost_change: list
    accepted_step_radius: Any
    consecutive_invalid_trials: int
    consecutive_noise_dominated_trials: int
    gradient_floor_samples: list = []


class AfterAttempt(NamedTuple):
    """The trials kept after an attempt, and what the stop tests read of the attempt."""

    trials: Trials
    accepted: bool
    evaluated: bool
    predicted: float
    cost_change: float
    cost_threshold: float


def _after_attempt(
    trials: Trials | LoopState,
    *,
    accepted: bool,
    valid: bool,
    invalid: bool,
    cost: float,
    trial_cost: float,
    predicted: float,
    step_radius: float,
    cost_tol: float,
    cost_atol: float,
    noise_limit: Any,
    scaled_norm: float = 0.0,
    floor_bound: Any = None,
) -> AfterAttempt:
    """The growth lock, the windows and the runs after an attempt made at objective ``cost``.

    ``trials`` holds them before the attempt, as a ``Trials`` or a ``LoopState``.
    An accepted trial lowers the lock by one, an ``invalid`` one sets it to 2.
    A ``valid`` trial that is not invalid and has a finite ``trial_cost`` is
    evaluated; rejected, it enters the windows with its objective change and
    ``predicted`` decrease relative to the cost scale
    ``cost_scale(cost, cost_tol, cost_atol)``, and the noise window with its
    change when ``_noise_dominated``. An evaluated trial ends
    the run of invalid trials and extends or ends the noise-dominated run; an
    invalid one extends the first and ends the second; any other leaves both.

    With ``floor_bound``, the bound on the gradient's error per unit of scaled
    step length, the trial, of scaled step norm ``scaled_norm``, enters the
    window of floor samples when ``_floor_sample`` admits it, judged against
    the evaluation noise measured before this attempt. ``None`` adds no sample.

    The cost test reads ``cost_threshold(cost, cost_tol, cost_atol)``, returned
    as ``AfterAttempt.cost_threshold``.
    """
    lock = trials.growth_lock
    if accepted:
        if lock:
            lock -= 1
    elif invalid:
        lock = 2
    evaluated = bool(valid and not invalid and np.isfinite(trial_cost))
    cost_change = abs(trial_cost - cost)
    relative_to = cost_scale(cost, cost_tol, cost_atol)
    changes, decreases = trials.rejected_cost_change, trials.rejected_predicted_decrease
    if evaluated and not accepted:
        changes = (changes + [cost_change / relative_to])[-_WINDOW:]
        decreases = (decreases + [predicted / relative_to])[-_WINDOW:]
    samples = trials.gradient_floor_samples
    if floor_bound is not None:
        sample = _floor_sample(
            evaluated=evaluated,
            accepted=accepted,
            step_radius=step_radius,
            accepted_radius=trials.accepted_step_radius,
            cost=cost,
            trial_cost=trial_cost,
            predicted=predicted,
            scaled_norm=scaled_norm,
            noise=_median(trials.noise_dominated_cost_change),
            relative_to=relative_to,
            bound=floor_bound,
        )
        if sample is not None:
            samples = [*samples, sample][-_WINDOW:]
    noisy = _noise_dominated(
        evaluated=evaluated,
        accepted=accepted,
        step_radius=step_radius,
        accepted_radius=trials.accepted_step_radius,
        cost=cost,
        trial_cost=trial_cost,
        predicted=predicted,
        relative_to=relative_to,
        limit=noise_limit,
    )
    noise = trials.noise_dominated_cost_change
    if noisy:
        noise = (noise + [cost_change / relative_to])[-_WINDOW:]
    invalid_run = trials.consecutive_invalid_trials
    noise_run = trials.consecutive_noise_dominated_trials
    if evaluated:
        invalid_run = 0
        noise_run = noise_run + 1 if noisy else 0
    elif invalid:
        invalid_run += 1
        noise_run = 0
    radius = step_radius if accepted else trials.accepted_step_radius
    kept = Trials(lock, changes, decreases, noise, radius, invalid_run, noise_run, samples)
    threshold = cost_threshold(cost, cost_tol, cost_atol)
    return AfterAttempt(kept, accepted, evaluated, predicted, cost_change, threshold)


def _stop_after(
    after: AfterAttempt,
    *,
    max_invalid: int,
    step_small: Any,
    accepted_count: int,
    target_steps: Any,
    target_stop: str = "accepted_steps",
    requested: bool = False,
    evaluations: int = 0,
    max_evaluations: Any = None,
) -> str | None:
    """The event that ends a trust-region loop after an attempt, or ``None`` to go on.

    In this order: ``"callback"`` when ``requested``; ``"invalid_trials"`` at
    ``max_invalid`` invalid trials in a row; ``"evaluation_noise"`` at three
    noise-dominated ones; ``"step_tolerance"`` when ``step_small`` (the
    loop's ``step_small`` test of the trial); ``"cost_tolerance"`` when
    ``cost_small`` at the attempt's cost threshold; ``target_stop`` at the
    accepted trial that brings
    ``accepted_count`` to ``target_steps``; ``"max_evaluations"`` when
    ``evaluations`` reach ``max_evaluations``. ``None`` limits test nothing.
    """
    trials = after.trials
    if requested:
        return "callback"
    if trials.consecutive_invalid_trials >= max_invalid:
        return "invalid_trials"
    if trials.consecutive_noise_dominated_trials >= _NOISE_RUN:
        return "evaluation_noise"
    if step_small:
        return "step_tolerance"
    if cost_small(
        accepted=after.accepted,
        evaluated=after.evaluated,
        cost_change=after.cost_change,
        predicted=after.predicted,
        threshold=after.cost_threshold,
    ):
        return "cost_tolerance"
    if after.accepted and target_steps is not None and accepted_count >= int(target_steps):
        return target_stop
    if max_evaluations is not None and evaluations >= max_evaluations:
        return "max_evaluations"
    return None
