"""The filter line search with second-order corrections, as one `lax.while_loop`.

Each pass of the loop evaluates one trial point. The first trial is the full step to the
boundary fraction; if the filter rejects it while it raised the constraint violation, up to
`P_MAX` second-order corrections follow, each a new right-hand side on the same KKT factor
and computed only on the passes that take one. After them the step is halved until a trial
is accepted or the step falls below the minimum of [minimum_step][]. A trial whose
evaluation failed (a NaN, a refused point) is rejected and counted, and the search ends once
`max_failed` failures come in a row. Without a [Correction][] the search only halves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from diff_soup.optimization.solvers import _filter
from diff_soup.optimization.solvers._barrier import Bounds, fraction_to_boundary

#: Ipopt's `kappa_soc`: the violation decrease each further correction must give.
KAPPA_SOC = 0.99
#: Ipopt's `max_soc`: the second-order corrections of one search.
P_MAX = 4
#: Ours: the trial points of one search; halving from 1 reaches 1e-18 in 60 trials.
MAX_TRIALS = 60

_NORMAL = 0
_CORRECTION = 1


class Correction(NamedTuple):
    """What a second-order correction needs from the step.

    Attributes:
        solve: `c_soc -> (dx, ds, dlam)`, the corrected step on the step's KKT factor.
        constraint: `[c_E; c_I + s]` at the current point.
        dlam: The step in the constraint multipliers.
        tau: The fraction to the boundary.
        bounds: The bounds of `x`.
    """

    solve: Callable
    constraint: Float[Array, "m"]
    dlam: Float[Array, "m"]
    tau: Float[Array, ""]
    bounds: Bounds


class LineSearchResult(eqx.Module):
    """The outcome of one line search.

    Attributes:
        accepted: Whether a trial was accepted.
        f_type: Whether the accepted step was held to Armijo, so the filter is left alone.
        alpha: The primal step length along `dx`.
        dx: The accepted direction in `x`, the second-order corrected one if it was taken.
        ds: The accepted direction in the slacks.
        dlam: The accepted direction in the constraint multipliers, `(0,)` without a correction.
        failed: The number of trial evaluations that failed.
        consecutive_failed: The failures in a row at the end of the search.
    """

    accepted: Bool[Array, ""]
    f_type: Bool[Array, ""]
    alpha: Float[Array, ""]
    dx: Float[Array, "n"]
    ds: Float[Array, "n_s"]
    dlam: Float[Array, "m"]
    failed: Int[Array, ""]
    consecutive_failed: Int[Array, ""]


class _Carry(eqx.Module):
    """The loop state: the mode (normal or correction), the correction in hand, the outcome."""

    mode: Int[Array, ""]
    first: Bool[Array, ""]
    alpha: Float[Array, ""]
    soc_count: Int[Array, ""]
    soc_alpha: Float[Array, ""]
    soc_dx: Float[Array, "n"]
    soc_ds: Float[Array, "n_s"]
    soc_dlam: Float[Array, "m"]
    soc_c: Float[Array, "m"]
    soc_theta_old: Float[Array, ""]
    accepted: Bool[Array, ""]
    stop: Bool[Array, ""]
    f_type: Bool[Array, ""]
    result_alpha: Float[Array, ""]
    result_dx: Float[Array, "n"]
    result_ds: Float[Array, "n_s"]
    result_dlam: Float[Array, "m"]
    failed: Int[Array, ""]
    consecutive_failed: Int[Array, ""]
    trials: Int[Array, ""]


def filter_line_search(
    trial_point: Callable,
    x: Float[Array, "n"],
    s: Float[Array, "n_s"],
    dx: Float[Array, "n"],
    ds: Float[Array, "n_s"],
    *,
    alpha_max: Float[Array, ""],
    theta: Float[Array, ""],
    phi: Float[Array, ""],
    slope: Float[Array, ""],
    filt: _filter.Filter,
    theta_min: Float[Array, ""],
    theta_max: Float[Array, ""],
    consecutive_failed: Int[Array, ""],
    max_failed: int,
    accept_full: Bool[Array, ""],
    correction: Correction | None = None,
) -> LineSearchResult:
    """Search along `(dx, ds)` from `(x, s)` for a step the filter accepts.

    Args:
        trial_point: `(x, s) -> (theta, phi, constraint, valid)` at a trial point, where
            `constraint` is the vector `[c_E; c_I + s]` whose 1-norm is `theta` (any vector
            without a correction).
        x: The current point.
        s: The current slacks, `(0,)` for a solver without them.
        dx: The step in `x`.
        ds: The step in the slacks.
        alpha_max: The fraction-to-the-boundary step length.
        theta: The constraint violation at the current point.
        phi: The (barrier) objective at the current point.
        slope: The directional derivative of `phi` along `(dx, ds)`.
        filt: The filter.
        theta_min: The switching threshold on `theta`.
        theta_max: The largest violation a trial may have.
        consecutive_failed: The failed evaluations in a row before this search.
        max_failed: The failures in a row that end the search.
        accept_full: Accept `alpha_max` without a trial, for a step too small to measure.
        correction: The second-order correction, or `None` to only halve the step.

    Returns:
        The [LineSearchResult][].
    """
    alpha_min = _filter.minimum_step(theta, slope, theta_min)
    dtype = x.dtype
    corrects = correction is not None and correction.constraint.shape[0] > 0
    if correction is None:
        dlam = jnp.zeros((0,), dtype)
        constraint = dlam
    else:
        dlam, constraint = correction.dlam, correction.constraint

    def corrected(soc_c):
        soc_dx, soc_ds, soc_dlam = correction.solve(soc_c)
        alpha = fraction_to_boundary(x, soc_dx, s, soc_ds, correction.tau, correction.bounds)
        return alpha, soc_dx, soc_ds, soc_dlam

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

        if corrects:
            start_soc = (
                ~in_soc & carry.first & rejected & valid & (theta_t >= theta) & (theta_t > 0)
            )
            continue_soc = (
                in_soc
                & rejected
                & valid
                & (carry.soc_count < P_MAX)
                & (theta_t <= KAPPA_SOC * carry.soc_theta_old)
            )
            soc_c = jnp.where(
                start_soc,
                alpha_max * constraint + constraint_t,
                carry.soc_alpha * carry.soc_c + constraint_t,
            )
            into_soc = start_soc | continue_soc
            soc = lax.cond(
                into_soc,
                corrected,
                lambda _: (carry.soc_alpha, carry.soc_dx, carry.soc_ds, carry.soc_dlam),
                soc_c,
            )
            soc_c = jnp.where(into_soc, soc_c, carry.soc_c)
        else:
            start_soc = continue_soc = into_soc = jnp.asarray(False)
            soc = (carry.soc_alpha, carry.soc_dx, carry.soc_ds, carry.soc_dlam)
            soc_c = carry.soc_c
        leave_soc = in_soc & rejected & ~continue_soc

        halve = rejected & ((~in_soc & ~start_soc) | leave_soc)
        alpha = jnp.where(halve, 0.5 * carry.alpha, carry.alpha)
        consecutive = jnp.where(valid, 0, carry.consecutive_failed + 1).astype(jnp.int32)
        too_short = halve & (alpha < alpha_min)
        stop = (
            too_short | (consecutive >= max_failed) | (carry.trials + 1 >= MAX_TRIALS)
        ) & rejected
        soc_alpha, soc_dx, soc_ds, soc_dlam = soc
        return _Carry(
            mode=jnp.where(into_soc, _CORRECTION, _NORMAL).astype(jnp.int32),
            first=jnp.asarray(False),
            alpha=alpha,
            soc_count=jnp.where(start_soc, 1, carry.soc_count + continue_soc).astype(jnp.int32),
            soc_alpha=soc_alpha,
            soc_dx=soc_dx,
            soc_ds=soc_ds,
            soc_dlam=soc_dlam,
            soc_c=soc_c,
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
        alpha=jnp.asarray(alpha_max, dtype),
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
        result_alpha=jnp.asarray(alpha_max, dtype),
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
    )
