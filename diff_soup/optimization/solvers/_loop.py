"""The tolerances a trust-region loop tests, the state it carries, and the rules of an attempt.

The rules of one attempt: its gain ratio, acceptance and next radius; the growth
lock, the windows and the runs after it; the tests that end the loop. Every rule is
written for arrays (``_loop_judge._namespace``): a host loop calls it on NumPy
values, a compiled program on traced ones, a polish on every member at once.

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
from collections.abc import Mapping
from numbers import Real
from typing import Any, NamedTuple

import numpy as np
from jaxtyping import ArrayLike

from diff_soup._distributed import local_value
from diff_soup.optimization._abc import OptimizationProblem
from diff_soup.optimization.solvers._loop_judge import (
    Window,
    _floor_sample,
    _larger,
    _median,
    _missing,
    _namespace,
    _noise_dominated,
    as_window,
    cost_scale,
    window_entries,
    window_push,
)
from diff_soup.optimization.solvers._termination import (
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
    stationarity_tol: Real | None


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


def _saved_window(state: Mapping | None, key: str) -> Window:
    """A window of rejected trials kept in a saved state; empty when the state has none."""
    if state is None:
        return as_window([])
    try:
        values = state[key]
    except (KeyError, TypeError):
        return as_window([])
    return as_window(np.ravel(np.asarray(values, dtype=np.float64)))


def _saved_value(state: Mapping | None, key: str) -> int | float | str | None:
    """One entry of a saved state as a Python scalar, or ``None`` when it is absent."""
    try:
        value = state[key]
    except (KeyError, TypeError):
        return None
    return np.asarray(value).item()


def _saved_count(state: Mapping | None, key: str) -> int:
    """A count kept in a saved state; 0 when the state has none."""
    value = _saved_value(state, key)
    return 0 if value is None else int(value)


def _saved_radius(state: Mapping | None, key: str) -> float | None:
    """A radius kept in a saved state; ``None`` when the state has none or holds ``nan``."""
    value = _saved_value(state, key)
    return None if value is None or not np.isfinite(value) else float(value)


@dataclasses.dataclass(kw_only=True)
class LoopState:
    """What a trust-region loop hands from one attempt to the next, each field saved by name.

    ``attempt`` counts the attempts made. ``scale``, ``grad_threshold`` and
    ``accepted_step_radius`` are ``None`` until the first attempt, gradient
    test and accepted step set them. The windows (``Window``) hold the last
    rejected and noise-dominated trials' changes of the objective (and the
    rejected ones' predicted decreases) relative to the cost scale at the attempt
    (``cost_scale``), saved as the arrays of their entries; the runs count
    invalid and noise-dominated rejected trials in a row.
    ``gradient_floor_samples`` is the window of the last floor samples
    (``_floor_sample``), in cost per unit of scaled step length; it is empty
    while no trial has been admitted as one. ``record`` is the attempt just
    made, saved as the entries ``record.<name>``.
    """

    attempt: int = 0
    accepted: int = 0
    point: ArrayLike
    scale: ArrayLike | None
    cost: float
    grad_threshold: float | None = None
    radius: float
    growth_lock: int = 0
    cost_history: list = dataclasses.field(default_factory=list)
    grad_history: list = dataclasses.field(default_factory=list)
    radius_history: list
    rejected_cost_change: Window = dataclasses.field(default_factory=lambda: as_window([]))
    rejected_predicted_decrease: Window = dataclasses.field(default_factory=lambda: as_window([]))
    noise_dominated_cost_change: Window = dataclasses.field(default_factory=lambda: as_window([]))
    accepted_step_radius: Real | None = None
    consecutive_invalid_trials: int = 0
    consecutive_noise_dominated_trials: int = 0
    gradient_floor_samples: Window = dataclasses.field(default_factory=lambda: as_window([]))
    record: dict

    @classmethod
    def start(cls, *, radius: float, **values: object) -> LoopState:
        """A fresh loop from ``point``, ``cost``, ``scale``, ``radius`` and ``record``.

        No attempt is made; the radius history holds ``radius``, and the other
        histories, the windows and the runs are empty.
        """
        return cls(radius=radius, radius_history=[radius], **values)

    @classmethod
    def restore(
        cls, state: Mapping, *, point: ArrayLike, cost: float, scale: ArrayLike | None
    ) -> LoopState:
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

        A field annotated ``int`` is saved as ``int64``, a window as the
        ``float64`` array of its entries, every other as ``float64`` (``None``
        as ``nan``); the record's entries as ``record.<name>``.
        """
        entries = {}
        for field in dataclasses.fields(self):
            if field.name == "record":
                continue
            value = getattr(self, field.name)
            if isinstance(value, Window):
                entries[field.name] = np.array(window_entries(value), dtype=np.float64)
                continue
            entries[field.name] = np.array(
                np.nan if value is None else value,
                dtype=np.int64 if field.type in ("int", int) else np.float64,
            )
        entries.update({f"record.{key}": value for key, value in self.record.items()})
        return entries

    def trials(self) -> Trials:
        """What the loop keeps of its trials, as the arrays a compiled program takes."""
        return Trials(
            np.asarray(self.growth_lock, dtype=np.int64),
            self.rejected_cost_change,
            self.rejected_predicted_decrease,
            self.noise_dominated_cost_change,
            np.asarray(_missing(self.accepted_step_radius), dtype=np.float64),
            np.asarray(self.consecutive_invalid_trials, dtype=np.int64),
            np.asarray(self.consecutive_noise_dominated_trials, dtype=np.int64),
            self.gradient_floor_samples,
        )

    def keep(self, trials: Trials) -> None:
        """Take the trials kept after an attempt, as host values (``None`` for no accepted step)."""
        radius = float(np.asarray(trials.accepted_step_radius))
        self.growth_lock = int(np.asarray(trials.growth_lock))
        self.rejected_cost_change = _host_window(trials.rejected_cost_change)
        self.rejected_predicted_decrease = _host_window(trials.rejected_predicted_decrease)
        self.noise_dominated_cost_change = _host_window(trials.noise_dominated_cost_change)
        self.accepted_step_radius = None if np.isnan(radius) else radius
        self.consecutive_invalid_trials = int(np.asarray(trials.consecutive_invalid_trials))
        self.consecutive_noise_dominated_trials = int(
            np.asarray(trials.consecutive_noise_dominated_trials)
        )
        self.gradient_floor_samples = _host_window(trials.gradient_floor_samples)


def _host_window(window: Any) -> Window:
    """A window as host arrays."""
    window = as_window(window)
    return Window(np.asarray(window.values), np.asarray(window.count))


# The number of noise-dominated rejected trials in a row that ends a loop: the model's
# predictions at that radius are below the noise of the cost's evaluation.
_NOISE_RUN = 3


def cost_threshold(cost: ArrayLike, cost_tol: float, cost_atol: float) -> ArrayLike:
    """The largest cost change the cost test accepts at ``cost``: ``max(cost_tol * |cost|, cost_atol)``.

    Relative to the cost itself, so where a fit stops does not depend on the
    units of its residual rows; ``cost_atol`` is the absolute floor.
    """
    xp = _namespace(cost)
    if xp is np and np.ndim(cost) == 0:
        return max(float(cost_tol) * abs(float(cost)), float(cost_atol))
    return _larger(float(cost_tol) * xp.abs(cost), float(cost_atol))


def cost_small(
    *,
    accepted: ArrayLike,
    evaluated: ArrayLike,
    cost_change: ArrayLike,
    predicted: ArrayLike,
    threshold: ArrayLike,
) -> ArrayLike:
    """The cost test of a trial: whether it ends its loop at the cost threshold ``threshold``.

    An ``accepted`` trial whose cost changed by at most ``threshold`` ends it,
    and so does an ``evaluated`` one (its cost computed, accepted or rejected)
    whose cost change and ``predicted`` decrease both are at most it.
    """
    xp = _namespace(accepted, evaluated, cost_change, predicted, threshold)
    with np.errstate(invalid="ignore"):
        within = cost_change <= threshold
        return (xp.asarray(accepted) & within) | (
            xp.asarray(evaluated) & within & (predicted <= threshold)
        )


def step_small(
    step_norm: ArrayLike, point_norm: ArrayLike, step_tol: float, *, tested: ArrayLike = True
) -> ArrayLike:
    """The step test of a trial: ``tested`` and ``step_norm <= step_tol * (1 + point_norm)``.

    ``point_norm`` is the norm of the point the loop holds after the trial;
    ``tested`` says whether the trial takes the test at all (a loop tests a
    trial whose cost was computed, or one the predicate passed when the test
    comes before the evaluation).
    """
    xp = _namespace(step_norm, point_norm, tested)
    with np.errstate(invalid="ignore"):
        return xp.asarray(tested) & (step_norm <= step_tol * (1.0 + point_norm))


def trial_valid(problem: OptimizationProblem, point: ArrayLike) -> bool:
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


class Trials(NamedTuple):
    """What a loop keeps of its trials, under the names of ``LoopState``'s fields.

    The windows are ``Window`` values (a sequence of entries is taken as one);
    ``accepted_step_radius`` is ``None`` or ``nan`` before the first accepted
    step. ``gradient_floor_samples`` may be left out by a loop that keeps no
    floor samples; it then holds an empty window, which nothing modifies in
    place, as each update of a window builds a new one.
    """

    growth_lock: ArrayLike
    rejected_cost_change: Window
    rejected_predicted_decrease: Window
    noise_dominated_cost_change: Window
    accepted_step_radius: ArrayLike | None
    consecutive_invalid_trials: ArrayLike
    consecutive_noise_dominated_trials: ArrayLike
    gradient_floor_samples: Window = as_window([])


class AfterAttempt(NamedTuple):
    """The trials kept after an attempt, and what the stop tests read of the attempt."""

    trials: Trials
    accepted: ArrayLike
    evaluated: ArrayLike
    predicted: ArrayLike
    cost_change: ArrayLike
    cost_threshold: ArrayLike


def _after_attempt(
    trials: Trials | LoopState,
    *,
    accepted: ArrayLike,
    valid: ArrayLike,
    invalid: ArrayLike,
    cost: ArrayLike,
    trial_cost: ArrayLike,
    predicted: ArrayLike,
    step_radius: ArrayLike,
    cost_tol: float,
    cost_atol: float,
    noise_limit: Real | None,
    scaled_norm: ArrayLike = 0.0,
    floor_bound: ArrayLike | None = None,
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
    the evaluation noise measured before this attempt. ``None`` or ``nan``
    adds no sample.

    The cost test reads ``cost_threshold(cost, cost_tol, cost_atol)``, returned
    as ``AfterAttempt.cost_threshold``.
    """
    if isinstance(trials, LoopState):
        trials = trials.trials()
    changes = as_window(trials.rejected_cost_change)
    decreases = as_window(trials.rejected_predicted_decrease)
    noise = as_window(trials.noise_dominated_cost_change)
    samples = as_window(trials.gradient_floor_samples)
    accepted_radius = _missing(trials.accepted_step_radius)
    xp = _namespace(
        accepted,
        valid,
        invalid,
        cost,
        trial_cost,
        predicted,
        step_radius,
        scaled_norm,
        floor_bound,
        changes,
        noise,
        samples,
        accepted_radius,
        trials.growth_lock,
    )
    accepted, valid, invalid = xp.asarray(accepted), xp.asarray(valid), xp.asarray(invalid)
    lock = xp.asarray(trials.growth_lock)
    lock = xp.where(accepted, xp.where(lock > 0, lock - 1, lock), xp.where(invalid, 2, lock))
    evaluated = valid & ~invalid & xp.isfinite(trial_cost)
    with np.errstate(invalid="ignore"):
        cost_change = xp.abs(trial_cost - cost)
    relative_to = cost_scale(cost, cost_tol, cost_atol)
    rejected = evaluated & ~accepted
    with np.errstate(all="ignore"):
        relative_change = cost_change / relative_to
        relative_predicted = predicted / relative_to
    sample = _floor_sample(
        evaluated=evaluated,
        accepted=accepted,
        step_radius=step_radius,
        accepted_radius=accepted_radius,
        cost=cost,
        trial_cost=trial_cost,
        predicted=predicted,
        scaled_norm=scaled_norm,
        noise=_median(noise),
        relative_to=relative_to,
        bound=floor_bound,
    )
    noisy = _noise_dominated(
        evaluated=evaluated,
        accepted=accepted,
        step_radius=step_radius,
        accepted_radius=accepted_radius,
        cost=cost,
        trial_cost=trial_cost,
        predicted=predicted,
        relative_to=relative_to,
        limit=noise_limit,
    )
    invalid_run = xp.asarray(trials.consecutive_invalid_trials)
    noise_run = xp.asarray(trials.consecutive_noise_dominated_trials)
    kept = Trials(
        lock,
        window_push(changes, relative_change, rejected),
        window_push(decreases, relative_predicted, rejected),
        window_push(noise, relative_change, noisy),
        xp.where(accepted, step_radius, accepted_radius),
        xp.where(evaluated, 0, xp.where(invalid, invalid_run + 1, invalid_run)),
        xp.where(evaluated, xp.where(noisy, noise_run + 1, 0), xp.where(invalid, 0, noise_run)),
        window_push(samples, sample, ~xp.isnan(sample)),
    )
    threshold = cost_threshold(cost, cost_tol, cost_atol)
    return AfterAttempt(kept, accepted, evaluated, predicted, cost_change, threshold)


# The number of noise-dominated rejected trials in a row that ends a loop: the model's
# predictions at that radius are below the noise of the cost's evaluation.
_NOISE_RUN = 3

# The events that end a loop after an attempt, in the order they are tested; a code is an
# event's position plus one, 0 for none. ``"target"`` is the loop's ``target_stop``.
STOP_EVENTS = (
    "callback",
    "invalid_trials",
    "evaluation_noise",
    "step_tolerance",
    "cost_tolerance",
    "target",
    "max_evaluations",
)


def stop_code(
    after: AfterAttempt,
    *,
    max_invalid: int,
    step_small: ArrayLike,
    accepted_count: ArrayLike,
    target_steps: Real | None,
    requested: ArrayLike = False,
    evaluations: ArrayLike = 0,
    max_evaluations: Real | None = None,
) -> ArrayLike:
    """The code of the event that ends a trust-region loop after an attempt, 0 to go on.

    In this order (``STOP_EVENTS``): ``"callback"`` when ``requested``;
    ``"invalid_trials"`` at ``max_invalid`` invalid trials in a row;
    ``"evaluation_noise"`` at three noise-dominated ones; ``"step_tolerance"``
    when ``step_small`` (the loop's ``step_small`` test of the trial);
    ``"cost_tolerance"`` when ``cost_small`` at the attempt's cost threshold;
    ``"target"`` at the accepted trial that brings ``accepted_count`` to
    ``target_steps``; ``"max_evaluations"`` when ``evaluations`` reach
    ``max_evaluations``. ``None`` limits test nothing.
    """
    trials = after.trials
    xp = _namespace(
        after.accepted,
        after.evaluated,
        after.cost_change,
        trials.consecutive_invalid_trials,
        step_small,
        accepted_count,
        requested,
        evaluations,
    )
    small_cost = cost_small(
        accepted=after.accepted,
        evaluated=after.evaluated,
        cost_change=after.cost_change,
        predicted=after.predicted,
        threshold=after.cost_threshold,
    )
    target = xp.asarray(False)
    if target_steps is not None:
        target = xp.asarray(after.accepted) & (accepted_count >= int(target_steps))
    budget = xp.asarray(False)
    if max_evaluations is not None:
        budget = xp.asarray(evaluations >= max_evaluations)
    events = (
        xp.asarray(requested),
        xp.asarray(trials.consecutive_invalid_trials) >= max_invalid,
        xp.asarray(trials.consecutive_noise_dominated_trials) >= _NOISE_RUN,
        xp.asarray(step_small),
        small_cost,
        target,
        budget,
    )
    code = xp.zeros(xp.shape(events[1]), dtype=np.int64)
    for position in range(len(events) - 1, -1, -1):
        code = xp.where(events[position], position + 1, code)
    return code


def stop_name(code: ArrayLike, target_stop: str = "accepted_steps") -> str | None:
    """The event of a stop ``code`` (``None`` for 0), ``"target"`` named ``target_stop``."""
    code = int(np.asarray(code))
    if code == 0:
        return None
    name = STOP_EVENTS[code - 1]
    return target_stop if name == "target" else name


def _stop_after(
    after: AfterAttempt,
    *,
    max_invalid: int,
    step_small: ArrayLike,
    accepted_count: ArrayLike,
    target_steps: Real | None,
    target_stop: str = "accepted_steps",
    requested: ArrayLike = False,
    evaluations: ArrayLike = 0,
    max_evaluations: Real | None = None,
) -> str | None:
    """``stop_code``'s event by name, on the host: ``None`` to go on."""
    return stop_name(
        stop_code(
            after,
            max_invalid=max_invalid,
            step_small=step_small,
            accepted_count=accepted_count,
            target_steps=target_steps,
            requested=requested,
            evaluations=evaluations,
            max_evaluations=max_evaluations,
        ),
        target_stop,
    )
