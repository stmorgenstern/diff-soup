"""The trust region's judgment of a trial: its gain ratio, its next radius, and the noise floor of its cost."""

from __future__ import annotations

from typing import Any

import numpy as np

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
