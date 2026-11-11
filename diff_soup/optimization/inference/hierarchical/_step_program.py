"""An attempt of the joint trust region as two programs: one proposes the trial, one judges it.

``propose`` takes the system at the current point and the loop's state as arrays.
It first tests whether the attempt before ended the loop (``stop_code``, read at the
point that attempt left, with the callbacks' request and the evaluations made, which
the host knows only after that attempt's records), then takes the attempt's scale and
subproblem coordinates, the gradient test, the solve at zero damping, the stationarity
and derivative-accuracy tests, the trial point and its predicted decrease. ``judge``
takes the trial's outcome (whether the predicate passed, whether a lane failed, its
cost) and gives the gain ratio, the acceptance, the next radius, the windows and the
runs. Between them the host makes the trial's request, turns failures into records,
re-solves the lanes whose rung rose, and on acceptance makes the request for the
matrix; the messages are built on the host from the codes.

Both are written for arrays (``_loop_judge._namespace``): compiled once per system
shape and settings with ``jax.jit``, or called on host values where a part of the
attempt still runs on the host (the inequality rows' active set). The proposal's
zero-damping solve and trial come from ``solve``: the attempt program of
``_level_solve`` (``attempt_solve_traced``), or the host active set under rows.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from numbers import Real
from typing import Any, NamedTuple

import jax
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.fitting.scales import XScale
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._level_solve import _attempt, _vectors
from diff_soup.optimization.inference.hierarchical._trust_step import (
    _bounded_gradient,
    _floor_bound,
    _radius_floor,
    _scale_candidate,
    _subproblem,
    _trial_from,
)
from diff_soup.optimization.inference.hierarchical._verdicts import _stationarity_limit, _within
from diff_soup.optimization.solvers._loop import (
    AfterAttempt,
    Trials,
    _after_attempt,
    step_small,
    stop_code,
)
from diff_soup.optimization.solvers._loop_judge import (
    Window,
    _floor_level,
    _floor_sigma,
    _judge,
    _median,
    _namespace,
    cost_scale,
)

# The tests before a trial that end the loop, as the proposal's ``pre_stop`` code (0: none).
PRE_STOPS = ("gradient_tolerance", "stationarity", "derivative_accuracy")


class StepSettings(NamedTuple):
    """What an attempt's programs take from the solver, fixed for a fit (static arguments).

    ``fixed`` lists the fixed coordinates; ``grad_test`` whether the gradient test
    runs (``grad_tol`` set and no rows); ``equality`` whether the system has
    equality rows; ``target_steps`` and ``max_evaluations`` are ``None`` for no
    limit; ``noise_limit`` ``None`` for no bound.
    """

    fixed: tuple
    grad_test: bool
    equality: bool
    stationarity_tol: Real | None
    cost_tol: float
    cost_atol: float
    step_tol: float
    eta: float
    radius_max: float
    relaxed: bool
    noise_limit: float | None
    max_invalid: int
    target_steps: int | None
    max_evaluations: int | None
    eps: float
    tiny: float


class Before(NamedTuple):
    """The loop at the start of an attempt, as arrays.

    ``scale`` is ``nan`` until the first attempt sets it; ``grad_threshold`` the
    gradient test's threshold (any number when the test is off); ``floor`` the
    lanes' terms of the gradient floor's bound at the point, ``(diagonals,
    costs, known)``.
    """

    point: ArrayLike
    scale: ArrayLike
    cost: ArrayLike
    radius: ArrayLike
    grad_threshold: ArrayLike
    noise: Window
    samples: Window
    floor: tuple


class Previous(NamedTuple):
    """The attempt before, as its stop tests read it: ``made`` false when there is none."""

    made: ArrayLike
    after: AfterAttempt
    step_norm: ArrayLike
    requested: ArrayLike
    evaluations: ArrayLike
    accepted: ArrayLike


class Proposal(NamedTuple):
    """An attempt's proposal, and the numbers at its point.

    ``previous_stop`` is the code of the event that ended the loop after the
    attempt before (``STOP_EVENTS``, 0: none) and ``pre_stop`` that of the test
    that ends it before this trial (``PRE_STOPS``, 0: none). ``scale`` is the
    attempt's scale; ``grad_norm`` and ``stop_gradient`` the gradient's norm and
    the norm the gradient test reads; ``stationarity`` the full step's predicted
    decrease over the cost scale; ``decrease``, ``full_norm`` and
    ``full_length`` that decrease in cost units and the full step's norm in the
    scaled coordinates and in parameter units; ``bound`` and ``level`` the
    gradient floor's bound and level (``nan``: none); ``radius_floor`` the radius
    a gain ratio under 0.25 does not shrink below. ``trial``, ``scaled_norm``,
    ``predicted``, ``multiplier``, ``solves`` and ``step_norm`` are the trial's;
    ``failed`` says the search found no finite step inside the radius.
    """

    previous_stop: ArrayLike
    pre_stop: ArrayLike
    scale: ArrayLike
    grad_norm: ArrayLike
    stop_gradient: ArrayLike
    stationarity: ArrayLike
    decrease: ArrayLike
    full_norm: ArrayLike
    full_length: ArrayLike
    bound: ArrayLike
    level: ArrayLike
    radius_floor: ArrayLike
    trial: ArrayLike
    scaled_norm: ArrayLike
    predicted: ArrayLike
    multiplier: ArrayLike
    solves: ArrayLike
    step_norm: ArrayLike
    failed: ArrayLike


class Zero(NamedTuple):
    """The solve at zero damping as the tests read it: its step ``z`` and the decrease predicted."""

    z: ArrayLike
    decrease: ArrayLike


class Trial(NamedTuple):
    """A trial as the proposal reads it."""

    point: ArrayLike
    scaled_norm: ArrayLike
    predicted: ArrayLike
    multiplier: ArrayLike
    solves: ArrayLike
    failed: ArrayLike


def coordinates(
    system: elimination.LevelSystem,
    x_scale: XScale,
    point: ArrayLike,
    scale: ArrayLike,
    bounds: tuple | None,
) -> tuple:
    """``(gradient, scale, coordinates)``: the attempt's gradient, its scale and its subproblem.

    The scale is the larger of ``scale`` (``nan`` before the first attempt) and
    the candidate the system's diagonal gives (``_scale_candidate``).
    """
    if isinstance(point, jax.core.Tracer):
        gradient, diagonal = _vectors(system)
    else:
        gradient, diagonal = elimination.level_vectors(system)
    xp = _namespace(gradient, scale)
    candidate = _scale_candidate(system, x_scale, diagonal)
    scale = xp.where(xp.isnan(scale), candidate, xp.maximum(scale, candidate))
    return gradient, scale, _subproblem(point, gradient, scale, bounds)


def attempt_solve_traced(
    system: elimination.LevelSystem,
    gradient: ArrayLike,
    coordinates: box.Scaling,
    radius: ArrayLike,
    point: ArrayLike,
    settings: StepSettings,
    bounds: tuple | None,
) -> tuple:
    """``solve`` without inequality rows: the attempt program, then the trial of its step.

    Returns ``(zero, trial)``, ``trial`` a function of no argument, here already made.
    """
    xp = _namespace(point, gradient, radius)
    scale = coordinates.scale
    addition = xp.zeros_like(scale) if coordinates.addition is None else coordinates.addition
    z, _, decrease, step, predicted, multiplier, solves, failed = _attempt(
        system,
        scale,
        addition,
        xp.asarray(radius, dtype=scale.dtype),
        eps=settings.eps,
        tiny=settings.tiny,
    )
    point_, scaled_norm, predicted = _trial_from(
        system,
        point,
        gradient,
        coordinates,
        radius,
        step,
        predicted,
        fixed=settings.fixed,
        bounds=bounds,
        equality=settings.equality,
    )
    trial = Trial(point_, scaled_norm, predicted, multiplier, solves, failed)
    return Zero(z, decrease), lambda: trial


def propose(
    system: elimination.LevelSystem,
    x_scale: XScale,
    bounds: tuple | None,
    floor: tuple | None,
    before: Before,
    previous: Previous,
    *,
    settings: StepSettings,
    solve: Callable = attempt_solve_traced,
) -> Proposal:
    """The proposal of an attempt: the stop tests, the solve at zero damping and the trial.

    In this order: the stop tests of the attempt before (``Previous``) at the
    current point; the scale and subproblem coordinates; the gradient test
    (``settings.grad_test``) on the bounded gradient's norm; the solve at zero
    damping (``solve``), the stationarity test against the yardstick and the
    derivative-accuracy test against the gradient floor's level (``floor``,
    the bound's fixed arrays, ``None`` without a floor); the radius floor; the
    trial. On host values the trial is made only when no test ended the loop.
    """
    point = before.point
    gradient, scale, subproblem = coordinates(system, x_scale, point, before.scale, bounds)
    xp = _namespace(gradient, scale)
    point_norm = xp.linalg.norm(point)
    tested = step_small(
        previous.step_norm, point_norm, settings.step_tol, tested=previous.after.evaluated
    )
    previous_stop = stop_code(
        previous.after,
        max_invalid=settings.max_invalid,
        step_small=tested,
        accepted_count=previous.accepted,
        target_steps=settings.target_steps,
        requested=previous.requested,
        evaluations=previous.evaluations,
        max_evaluations=settings.max_evaluations,
    )
    previous_stop = xp.where(previous.made, previous_stop, 0)
    grad_norm = xp.linalg.norm(gradient)
    stop_gradient = xp.linalg.norm(_bounded_gradient(gradient, subproblem, bounds))
    gradient_stop = xp.asarray(settings.grad_test) & (stop_gradient <= before.grad_threshold)
    zero, make_trial = solve(system, gradient, subproblem, before.radius, point, settings, bounds)
    relative_to = cost_scale(before.cost, settings.cost_tol, settings.cost_atol)
    with np.errstate(all="ignore"):
        stationarity = zero.decrease / relative_to
        full_length = xp.linalg.norm(zero.z / scale)
    full_norm = xp.linalg.norm(zero.z)
    limit = _stationarity_limit(settings.stationarity_tol, settings.cost_tol, _median(before.noise))
    bound = xp.asarray(np.nan, dtype=scale.dtype)
    if floor is not None:
        diagonals, costs, known = before.floor
        value = _floor_bound(*floor, diagonals, costs, scale)
        bound = xp.where(xp.asarray(known) & xp.isfinite(value), value, xp.nan)
    level = _floor_level(before.samples, bound)
    pre_stop = xp.where(
        gradient_stop,
        1,
        xp.where(
            _within(stationarity, limit),
            2,
            xp.where(
                ~xp.isnan(level)
                & _within(stationarity, _floor_sigma(level, full_norm, relative_to)),
                3,
                0,
            ),
        ),
    )
    radius_floor = _radius_floor(level, full_norm, gradient, scale)
    if xp is np and (int(previous_stop) or int(pre_stop)):
        trial = Trial(np.array(point), np.nan, np.nan, np.nan, 0, False)
    else:
        trial = make_trial()
    step_norm = xp.linalg.norm(trial.point - point)
    return Proposal(
        previous_stop,
        pre_stop,
        scale,
        grad_norm,
        stop_gradient,
        stationarity,
        zero.decrease,
        full_norm,
        full_length,
        bound,
        level,
        radius_floor,
        trial.point,
        trial.scaled_norm,
        trial.predicted,
        trial.multiplier,
        trial.solves,
        step_norm,
        trial.failed,
    )


@functools.partial(jax.jit, static_argnames=("settings",))
def propose_program(system, x_scale, bounds, floor, before, previous, *, settings):
    """``propose`` without inequality rows, as one compiled program."""
    return propose(system, x_scale, bounds, floor, before, previous, settings=settings)


class Outcome(NamedTuple):
    """What the host learned of a trial: the predicate, a failed lane, its cost, its proposal."""

    valid: ArrayLike
    invalid: ArrayLike
    trial_cost: ArrayLike
    predicted: ArrayLike
    scaled_norm: ArrayLike
    radius: ArrayLike
    radius_floor: ArrayLike
    bound: ArrayLike


def judge(trials: Trials, cost: ArrayLike, outcome: Outcome, *, settings: StepSettings) -> tuple:
    """The trial's gain ratio, acceptance, next radius and rejection code, and the trials after it.

    ``cost`` is the cost at the current point (after the re-solves of the lanes
    whose rung rose at the trial). Returns ``(rho, accepted, radius, reason,
    after)``: ``_judge``'s numbers and ``_after_attempt``'s.
    """
    rho, accepted, radius, reason = _judge(
        cost=cost,
        trial_cost=outcome.trial_cost,
        predicted=outcome.predicted,
        valid=outcome.valid,
        invalid=outcome.invalid,
        scaled_norm=outcome.scaled_norm,
        radius=outcome.radius,
        lock=trials.growth_lock,
        noise_window=trials.noise_dominated_cost_change,
        eta=settings.eta,
        radius_max=settings.radius_max,
        relaxed=settings.relaxed,
        relative_to=cost_scale(cost, settings.cost_tol, settings.cost_atol),
        radius_floor=outcome.radius_floor,
    )
    after = _after_attempt(
        trials,
        accepted=accepted,
        valid=outcome.valid,
        invalid=outcome.invalid,
        cost=cost,
        trial_cost=outcome.trial_cost,
        predicted=outcome.predicted,
        step_radius=outcome.radius,
        cost_tol=settings.cost_tol,
        cost_atol=settings.cost_atol,
        noise_limit=settings.noise_limit,
        scaled_norm=outcome.scaled_norm,
        floor_bound=outcome.bound,
    )
    return rho, accepted, radius, reason, after


judge_program = jax.jit(judge, static_argnames=("settings",))


def no_previous(dtype: Any) -> Previous:
    """The ``Previous`` of an attempt with none before it, shaped as a real one."""
    zero = np.asarray(0.0, dtype=dtype)
    empty = Window(np.zeros((10,), dtype=np.float64), np.asarray(0, dtype=np.int64))
    trials = Trials(
        np.asarray(0, dtype=np.int64),
        empty,
        empty,
        empty,
        np.asarray(np.nan),
        np.asarray(0, dtype=np.int64),
        np.asarray(0, dtype=np.int64),
        empty,
    )
    after = AfterAttempt(trials, np.False_, np.False_, zero, zero, zero)
    return Previous(
        np.False_,
        after,
        zero,
        np.False_,
        np.asarray(0, dtype=np.int64),
        np.asarray(0, dtype=np.int64),
    )


def as_host(value: Any) -> Any:
    """A program's output as host values: every array leaf on the host."""
    return jax.tree_util.tree_map(np.asarray, jax.device_get(value))


__all__ = [
    "PRE_STOPS",
    "Before",
    "Outcome",
    "Previous",
    "Proposal",
    "StepSettings",
    "as_host",
    "judge",
    "judge_program",
    "no_previous",
    "propose",
    "propose_program",
]
