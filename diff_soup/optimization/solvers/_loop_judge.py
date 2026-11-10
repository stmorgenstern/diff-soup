"""The trust region's judgment of a trial: its gain ratio, its next radius, and the noise floor of its cost.

Every rule here is written for arrays: it takes NumPy values or traced JAX values
and computes with the matching namespace (``_namespace``), elementwise, so one
definition serves a host loop, a compiled program and every member of a polish at
once. ``None`` where a rule documents it (no accepted step, no bound, no limit)
stands for ``nan`` in an array. Windows of trials are ``Window`` values: the last
``count`` entries of a fixed-length array.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

_TINY = float(np.finfo(float).tiny)

# The number of trials a window keeps: the rejected trials' cost changes and predicted
# decreases, for the report of a fit that ends at max_steps; the noise-dominated trials'
# cost changes, whose median is the evaluation noise; the gradient floor's samples.
_WINDOW = 10


def _namespace(*values: Any) -> Any:
    """``jax.numpy`` when a value is traced, else ``numpy``: the namespace a rule computes in."""
    for value in values:
        if isinstance(value, jax.core.Tracer):
            return jnp
        if isinstance(value, Window) and isinstance(value.values, jax.core.Tracer):
            return jnp
    return np


def _missing(value: Any) -> Any:
    """``nan`` for ``None``, else ``value``: an optional number as an array entry."""
    return float("nan") if value is None else value


def _larger(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Python's ``max(a, b)`` elementwise: ``b`` where ``b > a``, else ``a`` (``nan`` in ``b`` loses)."""
    xp = _namespace(a, b)
    return xp.where(b > a, b, a)


def _smaller(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Python's ``min(a, b)`` elementwise: ``b`` where ``b < a``, else ``a``."""
    xp = _namespace(a, b)
    return xp.where(b < a, b, a)


class Window(NamedTuple):
    """A window of the last trials: the last ``count`` entries of ``values``, oldest first.

    ``values`` is ``(..., _WINDOW)`` and ``count`` ``(...)``; leading axes are
    members. Two windows are equal when they hold the same entries, and a window
    reads as the array of its entries.
    """

    values: ArrayLike
    count: ArrayLike

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` holds the same entries (a window, or a sequence of them)."""
        if not isinstance(other, Window):
            other = as_window(other)
        return bool(np.array_equal(window_entries(self), window_entries(other), equal_nan=True))

    __hash__ = None

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """The window's entries, oldest first."""
        return np.asarray(window_entries(self), dtype=dtype)


def as_window(values: Any, members: tuple = ()) -> Window:
    """A ``Window`` of the entries ``values`` (a window is returned as it is); the last ``_WINDOW`` kept."""
    if isinstance(values, Window):
        return values
    entries = np.ravel(np.asarray(values if values is not None else [], dtype=np.float64))
    entries = entries[-_WINDOW:]
    filled = np.zeros((_WINDOW,), dtype=np.float64)
    if entries.size:
        filled[_WINDOW - entries.size :] = entries
    window = Window(filled, np.asarray(entries.size, dtype=np.int64))
    if members:
        window = Window(
            np.broadcast_to(filled, (*members, _WINDOW)).copy(),
            np.full(members, entries.size, dtype=np.int64),
        )
    return window


def window_entries(window: Window) -> np.ndarray:
    """The entries of one window as a host array, oldest first."""
    values = np.asarray(window.values, dtype=np.float64)
    count = int(np.asarray(window.count))
    return values[values.shape[-1] - count :]


def window_push(window: Window, value: ArrayLike, add: ArrayLike) -> Window:
    """``window`` with ``value`` appended where ``add``; the oldest entry leaves a full window."""
    xp = _namespace(window, value, add)
    values = xp.asarray(window.values)
    value = xp.asarray(value, dtype=values.dtype)
    shifted = xp.concatenate(
        [values[..., 1:], xp.broadcast_to(value, values.shape[:-1])[..., None]], axis=-1
    )
    add = xp.asarray(add)
    count = xp.asarray(window.count)
    return Window(
        xp.where(add[..., None], shifted, values),
        xp.where(add, xp.minimum(count + 1, values.shape[-1]), count),
    )


def _median(window: Any) -> ArrayLike:
    """The median of a window's entries, 0 when it is empty, ``nan`` when an entry is ``nan``."""
    window = as_window(window)
    xp = _namespace(window)
    values = xp.asarray(window.values)
    count = xp.asarray(window.count)
    size = values.shape[-1]
    valid = xp.arange(size) >= (size - count)[..., None]
    ordered = xp.sort(xp.where(valid, values, xp.inf), axis=-1)
    low = xp.take_along_axis(ordered, xp.maximum((count - 1) // 2, 0)[..., None], axis=-1)
    high = xp.take_along_axis(ordered, xp.maximum(count // 2, 0)[..., None], axis=-1)
    with np.errstate(invalid="ignore"):
        middle = (low[..., 0] + high[..., 0]) / 2
    middle = xp.where(count == 0, 0.0, middle)
    return xp.where(xp.any(valid & xp.isnan(values), axis=-1), xp.nan, middle)


def cost_scale(cost: ArrayLike, cost_tol: float, cost_atol: float) -> ArrayLike:
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
    xp = _namespace(cost)
    if xp is jnp:
        return jnp.maximum(jnp.abs(cost), floor)
    scale = np.maximum(np.abs(np.asarray(cost, dtype=np.float64)), floor)
    return float(scale) if scale.ndim == 0 else scale


def _next_radius(
    radius: ArrayLike,
    rho: ArrayLike,
    scaled_norm: ArrayLike,
    *,
    lock: ArrayLike,
    radius_max: Real,
    computed: ArrayLike,
    floor: ArrayLike = 0.0,
) -> ArrayLike:
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
    xp = _namespace(radius, rho, scaled_norm, lock, computed, floor)
    quarter = _larger(radius / 4.0, _TINY)
    shrink = _larger(_larger(0.25 * scaled_norm, _smaller(floor, radius)), _TINY)
    grow = _smaller(2.0 * radius, float(radius_max))
    with np.errstate(invalid="ignore"):
        growing = (rho > 0.75) & (lock == 0) & (scaled_norm >= 0.95 * radius)
        judged = xp.where(~(rho >= 0.25), shrink, xp.where(growing, grow, radius))
    return xp.where(computed, judged, quarter)


# The factor of the noise-relaxed gain ratio: Sun and Nocedal's 2 / (1 - c) at c = 1/2.
_RELAXATION = 4.0


def _gain_ratio(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    noise: ArrayLike,
    relative_to: ArrayLike,
    relaxed: bool,
) -> ArrayLike:
    """The gain ratio of a trial: its cost decrease ``actual`` over the model's ``predicted``.

    With ``relaxed`` and a measured evaluation noise ``noise`` above 0 (relative
    to the cost scale ``relative_to``, ``cost_scale``), both are raised by
    ``4 * noise * relative_to``, the noise in cost units times four, so that a
    trial whose predicted decrease is of the size of that noise is not
    rejected, and does not shrink the radius, for the noise alone; otherwise
    the classical ratio ``actual / predicted``. Works on floats and on arrays
    alike.
    """
    xp = _namespace(actual, predicted, noise, relative_to)
    with np.errstate(all="ignore"):
        classical = actual / predicted
        if not relaxed:
            return classical
        level = _RELAXATION * noise * relative_to
        return xp.where(noise > 0.0, (actual + level) / (predicted + level), classical)


def _noise_dominated(
    *,
    evaluated: ArrayLike,
    accepted: ArrayLike,
    step_radius: ArrayLike,
    accepted_radius: ArrayLike | None,
    cost: ArrayLike,
    trial_cost: ArrayLike,
    predicted: ArrayLike,
    relative_to: ArrayLike,
    limit: Real | None = None,
) -> ArrayLike:
    """Whether a trial's cost change measures the cost's evaluation jitter.

    The trial was evaluated and rejected, its step was solved at a radius
    below the one at which the last accepted step was solved, and its cost
    rose or changed by more than the model predicted. On a smooth cost a step
    that small along the model's descent direction does not raise the cost,
    so such a change is the jitter of its evaluation. Before the first
    accepted step (``accepted_radius`` ``None`` or ``nan``) no trial
    qualifies. When ``limit`` is a number, the decrease the model predicted,
    relative to the cost scale ``relative_to`` (``cost_scale``), must also be
    at most ``limit``; ``None`` sets no such bound.
    """
    accepted_radius = _missing(accepted_radius)
    xp = _namespace(evaluated, accepted, step_radius, accepted_radius, cost, trial_cost, predicted)
    with np.errstate(invalid="ignore"):
        noisy = (
            xp.asarray(evaluated)
            & ~xp.asarray(accepted)
            & ~xp.isnan(accepted_radius)
            & (step_radius < accepted_radius)
            & ((trial_cost > cost) | (xp.abs(cost - trial_cost) > predicted))
        )
        if limit is not None:
            noisy = noisy & (predicted / relative_to <= limit)
    return noisy


def _jump_level(changes: Any, decreases: Any, ceiling: Real | None) -> float | None:
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
    The verdict at the fit's end reads it, on the host.
    """
    rises = np.asarray(changes, dtype=np.float64)
    predicted = np.asarray(decreases, dtype=np.float64)
    if ceiling is None or not rises.size:
        return None
    if not bool(np.all((predicted > 0.0) & (rises > predicted) & (rises <= float(ceiling)))):
        return None
    return float(np.median(rises))


def _floor_sample(
    *,
    evaluated: ArrayLike,
    accepted: ArrayLike,
    step_radius: ArrayLike,
    accepted_radius: ArrayLike | None,
    cost: ArrayLike,
    trial_cost: ArrayLike,
    predicted: ArrayLike,
    scaled_norm: ArrayLike,
    noise: ArrayLike,
    relative_to: ArrayLike,
    bound: ArrayLike | None,
) -> ArrayLike:
    """A rejected trial's measure of the gradient's error, or ``nan`` when it gives none.

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
    accepted_radius, bound = _missing(accepted_radius), _missing(bound)
    xp = _namespace(evaluated, step_radius, accepted_radius, cost, trial_cost, predicted, bound)
    noise_abs = noise * relative_to
    with np.errstate(all="ignore"):
        mismatch = xp.abs((cost - trial_cost) - predicted)
        admitted = (
            xp.asarray(evaluated)
            & ~xp.asarray(accepted)
            & ~xp.isnan(accepted_radius)
            & (step_radius < accepted_radius)
            & (predicted > 0.0)
            & (scaled_norm > 0.0)
            & ~xp.isnan(bound)
            & (mismatch <= noise_abs + bound * scaled_norm)
        )
        sample = xp.maximum(mismatch - noise_abs, 0.0) / scaled_norm
    return xp.where(admitted, sample, xp.nan)


def _floor_level(samples: Any, bound: ArrayLike | None) -> ArrayLike:
    """The gradient's accuracy floor: the median of the floor ``samples``, at most ``bound``.

    In cost per unit of scaled step length. ``nan`` when there is no sample or
    no bound: the bound alone is an upper limit with a margin, not a measurement.
    """
    samples, bound = as_window(samples), _missing(bound)
    xp = _namespace(samples, bound)
    return xp.where(
        xp.isnan(bound) | (xp.asarray(samples.count) == 0),
        xp.nan,
        _smaller(_median(samples), bound),
    )


def _floor_sigma(
    level: ArrayLike | None, full_step_norm: ArrayLike, relative_to: ArrayLike
) -> ArrayLike:
    """The floor's cost equivalent along the model's full step, relative to the cost scale.

    ``level * full_step_norm`` is what a gradient in error by ``level`` along
    the model's full step, of scaled norm ``full_step_norm``, could take back
    of the decrease the step predicts. Over the cost scale ``relative_to``
    (``cost_scale``) it is in the stationarity number's units: a point whose
    stationarity number is at most it is as stationary as the derivatives can
    tell. 0 when ``level`` is ``None`` or ``nan``.
    """
    level = _missing(level)
    xp = _namespace(level, full_step_norm, relative_to)
    with np.errstate(invalid="ignore"):
        return xp.where(xp.isnan(level), 0.0, level * full_step_norm / relative_to)


def _floor_radius(
    level: ArrayLike | None, full_step_norm: ArrayLike, gradient_norm: ArrayLike
) -> ArrayLike:
    """The radius at which a step along the gradient predicts the floor's cost equivalent.

    ``level * full_step_norm / gradient_norm``, with ``gradient_norm`` the
    gradient's norm in the trust region's scaled coordinates: the Cauchy
    decrease ``gradient_norm * radius`` equals ``level * full_step_norm`` there,
    and a smaller radius predicts decreases the gradient's accuracy does not
    resolve. 0 when ``level`` is ``None`` or ``nan`` or the gradient is zero.
    """
    level = _missing(level)
    xp = _namespace(level, full_step_norm, gradient_norm)
    with np.errstate(all="ignore"):
        value = level * full_step_norm / gradient_norm
    return xp.where(xp.isnan(level) | ~(gradient_norm > 0.0), 0.0, value)


def _floor_parameter_radius(
    level: Real | None, full_step_norm: float, full_step_length: float, predicted: float
) -> float | None:
    """How far, in parameter units, the gradient's error moves the minimizer along the full step.

    Along the model's full step, of scaled norm ``full_step_norm``, length
    ``full_step_length`` in parameter units and predicted decrease
    ``predicted``, the model's curvature is ``2 * predicted / full_step_norm**2``,
    so a gradient in error by ``level`` moves the minimizer along it by
    ``level * full_step_norm**2 / (2 * predicted)`` in scaled units, that is
    ``full_step_length * level * full_step_norm / (2 * predicted)`` in
    parameter units. A bound along one direction, not the trust radius.
    ``None`` when ``level`` is ``None``; NaN when ``predicted`` is not positive.
    The fit's end report reads it, on the host.
    """
    if level is None:
        return None
    if not predicted > 0.0:
        return float("nan")
    return full_step_length * level * full_step_norm / (2.0 * predicted)


# Why a trial was rejected, as the judgment's code: accepted, no decrease predicted, or a
# gain ratio below eta.
_ACCEPTED, _NO_DECREASE, _LOW_GAIN = 0, 1, 2


def rejection_text(code: ArrayLike, rho: ArrayLike, eta: float) -> str | None:
    """The message of a judgment's rejection ``code`` (``None`` for an accepted trial)."""
    if int(code) == _ACCEPTED:
        return None
    if int(code) == _NO_DECREASE:
        return "the model predicted no decrease"
    return f"the gain ratio {float(rho):.3g} is below eta {float(eta):.3g}"


def _judge(
    *,
    cost: ArrayLike,
    trial_cost: ArrayLike,
    predicted: ArrayLike,
    valid: ArrayLike,
    invalid: ArrayLike,
    scaled_norm: ArrayLike,
    radius: ArrayLike,
    lock: ArrayLike,
    noise_window: Any,
    eta: float,
    radius_max: Real,
    relaxed: bool,
    relative_to: ArrayLike,
    hold: ArrayLike = False,
    radius_floor: ArrayLike = 0.0,
) -> tuple:
    """A trial's gain ratio, acceptance, next radius and rejection code, at objective ``cost``.

    The ratio is ``-inf`` for an ``invalid`` trial (its objective not
    evaluated) or a ``predicted`` decrease of at most 0, else ``_gain_ratio``
    with the median of ``noise_window`` and the cost scale ``relative_to``
    (``cost_scale`` at ``cost``); a ``valid`` trial (one the predicate
    passed) is accepted at ``eta`` or above. With ``hold`` (a step within the
    step tolerance, not evaluated) the ratio is NaN, the trial rejected and
    the radius kept. A gain ratio under 0.25 shrinks the radius to no less
    than the smaller of ``radius_floor`` and ``radius`` (``_next_radius``'s
    ``floor``). Returns ``(rho, accepted, radius, reason)``, ``reason`` a code
    for ``rejection_text``.
    """
    xp = _namespace(
        cost, trial_cost, predicted, valid, invalid, scaled_norm, radius, lock, noise_window
    )
    valid, invalid, hold = xp.asarray(valid), xp.asarray(invalid), xp.asarray(hold)
    computed = valid & ~invalid
    ratio = _gain_ratio(
        cost - trial_cost,
        predicted,
        noise=_median(noise_window),
        relative_to=relative_to,
        relaxed=relaxed,
    )
    with np.errstate(invalid="ignore"):
        no_decrease = predicted <= 0.0
        rho = xp.where(hold, xp.nan, xp.where(invalid | no_decrease, -xp.inf, ratio))
        accepted = computed & ~hold & (rho >= eta)
    reason = xp.where(accepted, _ACCEPTED, xp.where(no_decrease, _NO_DECREASE, _LOW_GAIN))
    radius = xp.where(
        hold,
        radius,
        _next_radius(
            radius,
            rho,
            scaled_norm,
            lock=lock,
            radius_max=radius_max,
            computed=computed,
            floor=radius_floor,
        ),
    )
    return rho, accepted, radius, reason
