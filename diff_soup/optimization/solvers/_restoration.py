"""Ipopt's feasibility restoration phase, as the same interior point on the restoration problem.

When the line search fails, the solver looks for a point the filter accepts by minimizing the
constraint violation near the current point `(x_R, s_R)`:

    min  rho * sum(p + n) + zeta / 2 * (|D_x (x - x_R)|^2 + |D_s (s - s_R)|^2)
    s.t. c(x, s) - p + n = 0,   p, n >= 0,   s >= 0,   the bounds of x,

where `c(x, s) = [c_E(x); c_I(x) + s]`, `zeta = sqrt(mu)` and `D = min(1, 1 / |x_R|)`
(Wächter and Biegler 2006, section 3.3). The restoration problem is a numeric problem in
`y = (x, s, p, n)` with only equality rows, and the interior point solves it in a nested
`while_loop` with its own budget. The phase ends as soon as the original filter accepts the
restoration iterate with its violation reduced by `KAPPA_RESTO`; it reports the original
problem infeasible when the restoration problem converges first.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax

from diff_soup.optimization.solvers import _filter
from diff_soup.optimization.solvers._barrier import Bounds, barrier_value
from diff_soup.optimization.solvers._closures import Closures
from diff_soup.optimization.solvers._protocol import SolveStatus

RHO = 1000.0
KAPPA_RESTO = 0.9
MAX_STEPS = 200


class Reference(eqx.Module):
    """The point the restoration problem stays near, carried in its `args`.

    Attributes:
        x: `x_R`.
        s: `s_R`.
        zeta: The weight of the proximity term, `sqrt(mu)`.
        weight_x: `D_x`, `(n,)`.
        weight_s: `D_s`, `(n_ineq,)`.
    """

    x: Any
    s: Any
    zeta: Any
    weight_x: Any
    weight_s: Any


def _split(y: Any, n: int, n_ineq: int, m: int) -> tuple[Any, Any, Any, Any]:
    """`(x, s, p, n)` from the restoration vector."""
    return y[:n], y[n : n + n_ineq], y[n + n_ineq : n + n_ineq + m], y[n + n_ineq + m :]


def restoration_closures(closures: Closures, n: int) -> Closures:
    """The closures of the restoration problem, taking `args = (args, Reference)`."""
    n_eq, n_ineq = closures.n_equality, closures.n_inequality
    m = n_eq + n_ineq

    def proximity_weights(reference):
        return jnp.concatenate([reference.weight_x, reference.weight_s]) ** 2

    def value_and_gradient(y, packed):
        args, reference = packed
        x, s, p, q = _split(y, n, n_ineq, m)
        offset = jnp.concatenate([x - reference.x, s - reference.s])
        weights = proximity_weights(reference)
        value = RHO * jnp.sum(p + q) + 0.5 * reference.zeta * jnp.sum(weights * offset**2)
        gradient = jnp.concatenate(
            [reference.zeta * weights * offset, jnp.full((2 * m,), RHO, dtype=y.dtype)]
        )
        return value, gradient

    def value(y, packed):
        return value_and_gradient(y, packed)[0]

    def residual(y, args, equality, inequality):
        _, s, p, q = _split(y, n, n_ineq, m)
        return jnp.concatenate([equality, inequality + s]) - p + q

    def constraints(y, packed):
        args, _ = packed
        equality, inequality = closures.constraints(y[:n], args)
        return residual(y, args, equality, inequality), jnp.zeros((0,), y.dtype)

    def constraints_and_jacobians(y, packed):
        args, _ = packed
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(y[:n], args)
        identity = jnp.eye(m, dtype=y.dtype)
        slack_columns = jnp.concatenate(
            [jnp.zeros((n_eq, n_ineq), y.dtype), jnp.eye(n_ineq, dtype=y.dtype)]
        )
        jacobian = jnp.concatenate(
            [jnp.concatenate([jac_eq, jac_ineq]), slack_columns, -identity, identity], axis=1
        )
        return (
            residual(y, args, equality, inequality),
            jnp.zeros((0,), y.dtype),
            jacobian,
            jnp.zeros((0, y.shape[0]), y.dtype),
        )

    def proximity_hessian(y, reference):
        diagonal = jnp.concatenate(
            [reference.zeta * proximity_weights(reference), jnp.zeros((2 * m,), y.dtype)]
        )
        return jnp.diag(diagonal)

    def lagrangian_hessian(y, lam, _lam_i, packed):
        args, reference = packed
        x = y[:n]
        zeros_e = jnp.zeros((n_eq,), y.dtype)
        zeros_i = jnp.zeros((n_ineq,), y.dtype)
        curvature = closures.lagrangian_hessian(
            x, lam[:n_eq], lam[n_eq:], args
        ) - closures.lagrangian_hessian(x, zeros_e, zeros_i, args)
        return proximity_hessian(y, reference).at[:n, :n].add(curvature)

    def gauss_newton(y, packed):
        f, g = value_and_gradient(y, packed)
        return f, g, proximity_hessian(y, packed[1]), 0

    def is_valid(y, packed):
        return closures.is_valid(y[:n], packed[0])

    return Closures(
        value=value,
        value_and_gradient=value_and_gradient,
        constraints=constraints,
        constraints_and_jacobians=constraints_and_jacobians,
        lagrangian_hessian=None if closures.lagrangian_hessian is None else lagrangian_hessian,
        gauss_newton=gauss_newton,
        is_valid=None if closures.is_valid is None else is_valid,
        n_equality=m,
        n_inequality=0,
    )


class Outcome(eqx.Module):
    """What the restoration phase returns to the regular phase.

    Attributes:
        ok: Whether the original filter accepted a restoration iterate.
        infeasible: Whether the restoration problem converged without that.
        x: The restoration iterate's `x`.
        s: Its slacks.
        steps: The restoration steps taken.
        failed: The failed trial evaluations in the restoration phase.
    """

    ok: Any
    infeasible: Any
    x: Any
    s: Any
    steps: Any
    failed: Any


def skipped(x: Any, s: Any) -> Outcome:
    """The outcome where the phase did not run."""
    return Outcome(
        jnp.asarray(False),
        jnp.asarray(False),
        x,
        s,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def restore(
    inner: Any,
    closures: Closures,
    bounds: Bounds,
    state: Any,
    args: Any,
    *,
    run: Any,
    theta: Any,
    filt: _filter.Filter,
) -> Outcome:
    """Run the restoration phase from the regular phase's `state`.

    Args:
        inner: The interior point that solves the restoration problem, without restoration.
        closures: The closures of the (scaled) original problem.
        bounds: The bounds of `x`.
        state: The regular phase's state at `x_R`.
        args: The (scaled) original problem's `args`.
        run: Whether this lane needs the phase; a lane that does not takes no step.
        theta: The constraint violation at `x_R`.
        filt: The original filter, already augmented with the current point.

    Returns:
        The [Outcome][].
    """
    n = state.x.shape[0]
    n_ineq = closures.n_inequality
    m = closures.n_equality + n_ineq
    dtype = state.x.dtype
    mu = state.mu
    x, s, evaluation = state.x, state.slacks, state.evaluation
    violation = jnp.concatenate([evaluation.equality, evaluation.inequality + s])
    mu_r = jnp.maximum(mu, jnp.max(jnp.abs(violation), initial=0.0))
    half = (mu_r - RHO * violation) / (2.0 * RHO)
    q = half + jnp.sqrt(half**2 + mu_r * violation / (2.0 * RHO))
    p = violation + q
    reference = Reference(
        x=x,
        s=s,
        zeta=jnp.sqrt(mu),
        weight_x=jnp.minimum(1.0, 1.0 / jnp.abs(x)),
        weight_s=jnp.minimum(1.0, 1.0 / jnp.abs(s)),
    )
    resto_args = (args, reference)
    resto = restoration_closures(closures, n)
    zero_block = jnp.zeros((n_ineq + 2 * m,), dtype)
    resto_bounds = Bounds(
        lower=jnp.concatenate([bounds.lower, zero_block]),
        upper=jnp.concatenate([bounds.upper, zero_block]),
        has_lower=jnp.concatenate([bounds.has_lower, jnp.ones((n_ineq + 2 * m,), bool)]),
        has_upper=jnp.concatenate([bounds.has_upper, jnp.zeros((n_ineq + 2 * m,), bool)]),
    )
    y0 = jnp.concatenate([x, s, p, q])
    z_lower = jnp.concatenate(
        [jnp.minimum(RHO, state.z_lower), jnp.minimum(RHO, state.v), mu_r / p, mu_r / q]
    )
    z_upper = jnp.concatenate(
        [jnp.minimum(RHO, state.z_upper), jnp.zeros((n_ineq + 2 * m,), dtype)]
    )
    start = inner._start(
        resto,
        resto_bounds,
        y0,
        args=resto_args,
        lam_e=jnp.zeros((m,), dtype),
        lam_i=jnp.zeros((0,), dtype),
        z_lower=z_lower,
        z_upper=z_upper,
        v=jnp.zeros((0,), dtype),
        mu=mu_r,
        scales=None,
    )

    def accepted(current):
        xr, sr, pr, qr = _split(current.x, n, n_ineq, m)
        constraint = current.evaluation.equality + pr - qr
        theta_r = jnp.sum(jnp.abs(constraint))
        phi_r = closures.value(xr, args) + barrier_value(xr, sr, mu, bounds)
        return (
            jnp.isfinite(phi_r)
            & ~_filter.blocks(filt, theta_r, phi_r)
            & (theta_r <= KAPPA_RESTO * theta)
        )

    def cond(carry):
        current, ok = carry
        return run & ~current.done & ~ok

    def body(carry):
        current, _ = carry
        current = inner._advance(resto, resto_bounds, current, resto_args)
        return current, accepted(current)

    end, ok = lax.while_loop(cond, body, (start, jnp.asarray(False)))
    xr, sr, pr, qr = _split(end.x, n, n_ineq, m)
    theta_end = jnp.sum(jnp.abs(end.evaluation.equality + pr - qr))
    converged = end.status == SolveStatus.KKT_TOL
    return Outcome(
        ok=run & ok,
        infeasible=run & ~ok & converged & (theta_end > inner.tol),
        x=xr,
        s=sr,
        steps=end.step,
        failed=end.failed_evaluations,
    )
