"""How a trust-region fit ended: its status, its stop report and its stall message."""

from __future__ import annotations

from typing import Any

import numpy as np

from diff_soup.optimization.inference.fitting.result import (
    TerminationStatus,
)
from diff_soup.optimization.inference.hierarchical._lane_route import _listed

NONFINITE_BLOCKS = "nonfinite_blocks_at_finite_cost"


def _rungs_text(rungs: Any) -> str:
    """``after rung 0`` or ``after rungs 0, 1, 2``; empty when the rungs are not known."""
    rungs = tuple(rungs)
    if not rungs:
        return ""
    label = "rung" if len(rungs) == 1 else "rungs"
    return f", after {label} {', '.join(str(rung) for rung in rungs)}"


def _failures_text(failures: Any) -> str:
    return _listed(
        f"{failure.lane}: {failure.kind} at t = {failure.t_reached:.6g}{_rungs_text(failure.rungs)}"
        for failure in failures
    )


def _stop_report(
    status: Any, cost_changes: list, decreases: list, stalled: Any = None, budget: Any = None
) -> tuple:
    """The message and the ``info`` entries of a fit, from its window of rejected trials.

    A fit that ends at ``max_steps`` or stalled reports how many rejected
    trials its window holds and, when there are any, the medians of their cost
    changes and predicted decreases, each relative to the cost scale
    ``max(|cost|, cost_atol / cost_tol)`` at its attempt. The
    ``max_steps`` message says which ``cost_tol`` would end the fit at that
    level; ``budget``, for a fit that ``max_evaluations`` ended, states the
    evaluations it made right after the status. A stalled fit's message is
    ``stalled`` (what ended the fit and its numbers) followed by the medians.
    Every other fit reports nothing and its message is the status value.
    """
    if status not in (TerminationStatus.MAX_STEPS, TerminationStatus.STALLED):
        return status.value, {}
    head = status.value if budget is None else f"{status.value}: {budget}"
    lead = stalled if status is TerminationStatus.STALLED else head
    k = len(cost_changes)
    entries: dict = {"last_rejected_trials": int(k)}
    if k == 0:
        return lead, entries
    a = float(np.median(np.asarray(cost_changes, dtype=np.float64)))
    p = float(np.median(np.asarray(decreases, dtype=np.float64)))
    if np.isfinite(a):
        entries["median_rejected_cost_change"] = a
    if np.isfinite(p):
        entries["median_rejected_predicted_decrease"] = p
    if not (np.isfinite(a) and np.isfinite(p)):
        return lead, entries
    medians = (
        f"over the last {k} rejected trial(s) the median size of the cost change was {a:.2g} "
        f"and the median predicted decrease was {p:.2g}, both relative to the cost scale "
        "max(|cost|, cost_atol / cost_tol)"
    )
    if status is TerminationStatus.STALLED:
        return f"{lead}; {medians}", entries
    t = max(a, p)
    prefix = f"{status.value}:" if budget is None else f"{head};"
    message = (
        f"{prefix} {medians}; when the cost is evaluated with noise of this size, "
        f"cost_tol={t:.2g} or larger ends the fit at that noise level."
    )
    return message, entries


# The status of a loop-ending event that the verdict does not decide.
_STOP_STATUS = {
    "gradient_tolerance": TerminationStatus.GRAD_TOL,
    "stationarity": TerminationStatus.GRAD_TOL,
    "derivative_accuracy": TerminationStatus.DERIVATIVE_ACCURACY,
    "max_steps": TerminationStatus.MAX_STEPS,
    "max_evaluations": TerminationStatus.MAX_STEPS,
    "accepted_steps": TerminationStatus.ACCEPTED_STEPS,
    "callback": TerminationStatus.STOPPED,
    "invalid_trials": TerminationStatus.STALLED,
}


# What a step or a cost test found within its tolerance, in a stalled fit's message.
_ENDED_BY = {"step_tolerance": "the step", "cost_tolerance": "the cost change"}


def _stationarity_limit(tolerance: Any, cost_tol: float, noise: float) -> float:
    """``stationarity_tol``, or for ``None`` the yardstick ``max(cost_tol, 3 * noise)``."""
    return max(float(cost_tol), 3.0 * float(noise)) if tolerance is None else float(tolerance)


def _verdict(
    stop: str,
    *,
    sigma: float,
    yardstick: float,
    lock: int,
    members: bool = True,
    floor_sigma: float = 0.0,
    floor_known: bool = False,
    jump_yardstick: Any = None,
    jump_members: bool = True,
) -> tuple:
    """The status and the stall reason of a loop that the event ``stop`` ended.

    A step test, a cost test or a run of noise-dominated trials ends the fit
    converged only at a point where the model predicts no more than the
    evaluation can show (``sigma <= yardstick``, and ``members``: in the
    nested mode every member's model too, against the members' yardstick) and
    no invalid trial shrank the radius (the growth lock is 0). Failing that,
    when the gradient floor is known (``floor_known``: a small rejected trial
    measured it), the growth lock is 0 and the model predicts no more than
    the gradient's error along its full step could take back
    (``sigma <= floor_sigma``, and ``members``, every member within its own
    floor), the fit converged to the accuracy of its derivatives. Failing
    that, a step test or a cost test ends the fit converged, with its own
    status, when every rejected trial in the window met a jump of the
    computed cost (``jump_yardstick``, ``max(cost_tol, 3 * the median
    jump)``, is a number), the growth lock is 0 and the model predicts no
    more than that yardstick (and ``jump_members``, every member within it
    too): the decrease left is below what any trial can show. Otherwise
    the fit is stalled, for invalid trials when the lock is set, else for the
    evaluation noise when the noise-dominated trials ended it, else because
    the radius collapsed. A run of invalid trials ends the fit stalled for
    invalid trials; every other event gives its own status.
    """
    if stop not in ("step_tolerance", "cost_tolerance", "evaluation_noise"):
        status = _STOP_STATUS[stop]
        return status, ("invalid_trials" if status is TerminationStatus.STALLED else None)
    if lock == 0 and sigma <= yardstick and members:
        if stop == "step_tolerance":
            return TerminationStatus.STEP_TOL, None
        return TerminationStatus.COST_TOL, None
    if lock == 0 and floor_known and sigma <= floor_sigma and members:
        return TerminationStatus.DERIVATIVE_ACCURACY, None
    if (
        stop != "evaluation_noise"
        and lock == 0
        and jump_yardstick is not None
        and sigma <= jump_yardstick
        and jump_members
    ):
        if stop == "step_tolerance":
            return TerminationStatus.STEP_TOL, None
        return TerminationStatus.COST_TOL, None
    if lock:
        return TerminationStatus.STALLED, "invalid_trials"
    if stop == "evaluation_noise":
        return TerminationStatus.STALLED, "evaluation_noise"
    return TerminationStatus.STALLED, "radius_collapsed"


def _last_invalid(log: list) -> tuple:
    """The failed lanes of the fit's last invalid trial, from its failure log; empty when none.

    An ignored failure (a lane of weight zero) made no trial invalid and is skipped.
    """
    rows: list = []
    last = None
    for attempt, failure, inner in reversed(log):
        if inner or failure.ignored:
            continue
        if last is None:
            last = attempt
        if attempt != last:
            break
        rows.append(failure)
    return tuple(reversed(rows))


def _stalled_message(
    reason: str,
    stop: str,
    *,
    consecutive: int,
    failures: tuple,
    radius: float,
    sigma: float,
    noise: float,
    ceiling: Any,
    member: Any = None,
) -> str:
    """What ended a stalled fit and its numbers, to two significant digits.

    ``member``, in the nested mode, is ``(level, label, number, yardstick)`` of
    the member whose model predicts the most beyond the members' yardstick, or
    ``None``; its clause ends the message.
    """
    clause = ""
    if member is not None:
        level, label, number, limit = member
        clause = (
            f"; member {label!r} of level {level!r} predicts a further relative decrease of "
            f"{number:.2g} of its own cost, above {limit:.2g}"
        )
    return (
        _stalled_text(reason, stop, consecutive, failures, radius, sigma, noise, ceiling) + clause
    )


def _stalled_text(
    reason: str,
    stop: str,
    consecutive: int,
    failures: tuple,
    radius: float,
    sigma: float,
    noise: float,
    ceiling: Any,
) -> str:
    """``_stalled_message`` without the member's clause."""
    if reason == "evaluation_noise":
        tolerance = (
            "" if ceiling is None else f" (lane solvers at a relative tolerance of {ceiling:.2g})"
        )
        return (
            f"stalled: the model predicts a further relative decrease of {sigma:.2g} while the "
            f"cost's evaluation jitter was measured at about {noise:.2g}{tolerance}; a tighter "
            "lane tolerance resolves the optimum further"
        )
    if reason == "radius_collapsed":
        return (
            f"stalled: {_ENDED_BY[stop]} fell within its tolerance while the model still "
            f"predicts a relative decrease of {sigma:.2g}; the radius is {radius:.2g}"
        )
    where = "one" if stop == "invalid_trials" else "invalid trial"
    failed = (
        f"; at the last {where} the lane solve failed for experiment(s) "
        + _listed(
            f"{failure.lane} ({failure.kind}) at t = {failure.t_reached:.2g}"
            f"{_rungs_text(failure.rungs)}"
            for failure in failures
        )
        if failures
        else ""
    )
    if stop == "invalid_trials":
        event = f"the last {consecutive} trial(s) in a row were invalid"
    elif stop == "evaluation_noise":
        event = (
            "three rejected trials in a row were within the noise of the cost's evaluation "
            "after an invalid trial shrank the radius"
        )
    else:
        event = (
            f"{_ENDED_BY[stop]} fell within its tolerance after an invalid trial shrank the radius"
        )
    return (
        f"stalled: {event}{failed}; the radius is {radius:.2g}; the model predicts a further "
        f"relative decrease of {sigma:.2g}"
    )
