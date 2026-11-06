"""The filter line search with second-order corrections, as one `lax.while_loop`.

Each pass of the loop evaluates one trial point. The first trial is the full step to the
boundary fraction; if the filter rejects it while it raised the constraint violation, up to
`P_MAX` second-order corrections follow, each a new right-hand side on the same KKT factor.
After them the step is halved until a trial is accepted or the step falls below the minimum
of [minimum_step][]. A trial whose evaluation failed (a NaN, a refused point) is rejected and
counted, and the search ends once `max_failed` failures come in a row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax

from diff_soup.optimization.solvers import _filter
from diff_soup.optimization.solvers._barrier import Bounds, fraction_to_boundary

KAPPA_SOC = 0.99
P_MAX = 4
MAX_TRIALS = 60

_NORMAL = 0
_CORRECTION = 1


class LineSearchResult(eqx.Module):
    """The outcome of one line search.

    Attributes:
        accepted: Whether a trial was accepted.
        f_type: Whether the accepted step was held to Armijo, so the filter is left alone.
        alpha: The primal step length along `dx`.
        dx: The accepted direction in `x`, the second-order corrected one if it was taken.
        ds: The accepted direction in the slacks.
        dlam: The accepted direction in the constraint multipliers.
        failed: The number of trial evaluations that failed.
        consecutive_failed: The failures in a row at the end of the search.
        trials: The number of trial points evaluated.
    """

    accepted: Any
    f_type: Any
    alpha: Any
    dx: Any
    ds: Any
    dlam: Any
    failed: Any
    consecutive_failed: Any
    trials: Any


class _Carry(eqx.Module):
    """The loop state: the mode (normal or correction), the correction in hand, the outcome."""

    mode: Any
    first: Any
    alpha: Any
    soc_count: Any
    soc_alpha: Any
    soc_dx: Any
    soc_ds: Any
    soc_dlam: Any
    soc_c: Any
    soc_theta_old: Any
    accepted: Any
    stop: Any
    f_type: Any
    result_alpha: Any
    result_dx: Any
    result_ds: Any
    result_dlam: Any
    failed: Any
    consecutive_failed: Any
    trials: Any


def filter_line_search(
    trial_point: Callable,
    correction: Callable,
    x: Any,
    s: Any,
    dx: Any,
    ds: Any,
    dlam: Any,
    *,
    alpha_max: Any,
    theta: Any,
    phi: Any,
    slope: Any,
    constraint: Any,
    filt: _filter.Filter,
    theta_min: Any,
    theta_max: Any,
    tau: Any,
    bounds: Bounds,
    consecutive_failed: Any,
    max_failed: int,
    accept_full: Any,
) -> LineSearchResult:
    """Search along `(dx, ds)` from `(x, s)` for a step the filter accepts.

    Args:
        trial_point: `(x, s) -> (theta, phi, constraint, valid)` at a trial point, where
            `constraint` is the vector `[c_E; c_I + s]` whose 1-norm is `theta`.
        correction: `c_soc -> (dx, ds, dlam)`, the second-order correction on the step's factor.
        x: The current point.
        s: The current slacks.
        dx: The step in `x`.
        ds: The step in the slacks.
        dlam: The step in the constraint multipliers.
        alpha_max: The fraction-to-the-boundary step length.
        theta: The constraint violation at the current point.
        phi: The barrier objective at the current point.
        slope: The directional derivative of `phi` along `(dx, ds)`.
        constraint: `[c_E; c_I + s]` at the current point.
        filt: The filter.
        theta_min: The switching threshold on `theta`.
        theta_max: The largest violation a trial may have.
        tau: The fraction to the boundary.
        bounds: The bounds of `x`.
        consecutive_failed: The failed evaluations in a row before this search.
        max_failed: The failures in a row that end the search.
        accept_full: Accept `alpha_max` without a trial, for a step too small to measure.

    Returns:
        The [LineSearchResult][].
    """
    alpha_min = _filter.minimum_step(theta, slope, theta_min)
    has_rows = constraint.shape[0] > 0
    dtype = x.dtype

    def cond(carry: _Carry):
        return ~carry.accepted & ~carry.stop

    def body(carry: _Carry):
        in_soc = carry.mode == _CORRECTION
        step_x = jnp.where(in_soc, carry.soc_alpha * carry.soc_dx, carry.alpha * dx)
        step_s = jnp.where(in_soc, carry.soc_alpha * carry.soc_ds, carry.alpha * ds)
        theta_t, phi_t, constraint_t, valid = trial_point(x + step_x, s + step_s)
        test_alpha = jnp.where(in_soc, alpha_max, carry.alpha)
        accepted, f_type = _filter.acceptable(
            filt, theta_t, phi_t, test_alpha, theta, phi, slope, theta_min, theta_max
        )
        accepted = accepted & valid
        rejected = ~accepted

        start_soc = (
            has_rows & ~in_soc & carry.first & rejected & valid & (theta_t >= theta) & (theta_t > 0)
        )
        continue_soc = (
            in_soc
            & rejected
            & valid
            & (carry.soc_count < P_MAX)
            & (theta_t <= KAPPA_SOC * carry.soc_theta_old)
        )
        leave_soc = in_soc & rejected & ~continue_soc
        soc_c = jnp.where(
            start_soc,
            alpha_max * constraint + constraint_t,
            carry.soc_alpha * carry.soc_c + constraint_t,
        )
        soc_dx, soc_ds, soc_dlam = correction(soc_c)
        soc_alpha = fraction_to_boundary(x, soc_dx, s, soc_ds, tau, bounds)
        into_soc = start_soc | continue_soc

        halve = rejected & ((~in_soc & ~start_soc) | leave_soc)
        alpha = jnp.where(halve, 0.5 * carry.alpha, carry.alpha)
        consecutive = jnp.where(valid, 0, carry.consecutive_failed + 1).astype(jnp.int32)
        too_short = halve & (alpha < alpha_min)
        stop = (
            too_short | (consecutive >= max_failed) | (carry.trials + 1 >= MAX_TRIALS)
        ) & rejected

        def keep(new, old):
            return jnp.where(into_soc, new, old)

        return _Carry(
            mode=jnp.where(into_soc, _CORRECTION, _NORMAL).astype(jnp.int32),
            first=jnp.asarray(False),
            alpha=alpha,
            soc_count=jnp.where(start_soc, 1, carry.soc_count + continue_soc).astype(jnp.int32),
            soc_alpha=keep(soc_alpha, carry.soc_alpha),
            soc_dx=keep(soc_dx, carry.soc_dx),
            soc_ds=keep(soc_ds, carry.soc_ds),
            soc_dlam=keep(soc_dlam, carry.soc_dlam),
            soc_c=keep(soc_c, carry.soc_c),
            soc_theta_old=jnp.where(
                start_soc, theta, jnp.where(continue_soc, theta_t, carry.soc_theta_old)
            ),
            accepted=accepted,
            stop=stop,
            f_type=f_type & accepted,
            result_alpha=jnp.where(in_soc, carry.soc_alpha, carry.alpha),
            result_dx=jnp.where(in_soc, carry.soc_dx, dx),
            result_ds=jnp.where(in_soc, carry.soc_ds, ds),
            result_dlam=jnp.where(in_soc, carry.soc_dlam, dlam),
            failed=(carry.failed + ~valid).astype(jnp.int32),
            consecutive_failed=consecutive,
            trials=(carry.trials + 1).astype(jnp.int32),
        )

    zero = jnp.zeros((), dtype=dtype)
    start = _Carry(
        mode=jnp.asarray(_NORMAL, jnp.int32),
        first=jnp.asarray(True),
        alpha=alpha_max,
        soc_count=jnp.asarray(0, jnp.int32),
        soc_alpha=zero,
        soc_dx=jnp.zeros_like(dx),
        soc_ds=jnp.zeros_like(ds),
        soc_dlam=jnp.zeros_like(dlam),
        soc_c=jnp.zeros_like(constraint),
        soc_theta_old=theta,
        accepted=jnp.asarray(accept_full),
        stop=jnp.asarray(False),
        f_type=jnp.asarray(False),
        result_alpha=alpha_max,
        result_dx=dx,
        result_ds=ds,
        result_dlam=dlam,
        failed=jnp.asarray(0, jnp.int32),
        consecutive_failed=jnp.asarray(consecutive_failed, jnp.int32),
        trials=jnp.asarray(0, jnp.int32),
    )
    end = lax.while_loop(cond, body, start)
    return LineSearchResult(
        accepted=end.accepted,
        f_type=end.f_type,
        alpha=end.result_alpha,
        dx=end.result_dx,
        ds=end.result_ds,
        dlam=end.result_dlam,
        failed=end.failed,
        consecutive_failed=end.consecutive_failed,
        trials=end.trials,
    )
