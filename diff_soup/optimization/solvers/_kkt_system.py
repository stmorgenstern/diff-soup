"""The KKT system at a constrained solution, which both forms of the implicit derivative solve.

The active set is read from the multipliers and the slacks: a row is active where its
multiplier exceeds its slack. Where both are below `DEGENERATE * sqrt(mu)` the active set is
degenerate, and the system is instead the barrier solution's at the final `mu`, whose blocks
replace the active-set masks with the barrier diagonals `Z / d` and `s / lam`; the record's
`degenerate` says so. The exact Lagrangian Hessian is computed once, at the solution.

The Gauss-Newton form, for a least-squares objective whose solver steps on Gauss-Newton,
replaces that Hessian with `J_r^T J_r` of the residual rows and drops the constraint
curvature: its derivatives in `args` are first order, the rows' (`rows_of`) and the
constraints'. At a solution with zero residual and linear rows the two forms agree.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from diff_soup.optimization.kkt import cholesky
from diff_soup.optimization.solvers._barrier import Bounds, distances
from diff_soup.optimization.solvers._closures import Closures
from diff_soup.optimization.solvers._protocol import SolveRecord

#: Ours: a multiplier and its slack both below `DEGENERATE * sqrt(mu)` make the active set degenerate.
DEGENERATE = 1e2
#: Ours: the cap on the degenerate system's barrier diagonals, so the branch that is not
#: selected stays finite and its zero cotangent never meets an infinity; a dtype whose
#: largest finite value is smaller (float32) is capped there instead.
_CAP = 1e200


class ActiveSet(eqx.Module):
    """The active set at a solution.

    Attributes:
        inequality: Where an inequality row is active, `(n_ineq,)`.
        lower: Where a lower bound is active, `(n,)`.
        upper: Where an upper bound is active, `(n,)`.
        degenerate: Whether a multiplier and its slack (or bound distance) are both near zero.
        pinned: Where equal bounds fix a coordinate; it counts as active on its lower bound
            when that multiplier is the larger, else on its upper, and is never degenerate.
    """

    inequality: Bool[Array, "n_ineq"]
    lower: Bool[Array, "n"]
    upper: Bool[Array, "n"]
    degenerate: Bool[Array, ""]
    pinned: Bool[Array, "n"]


def active_set(
    x: Float[Array, "n"],
    record: SolveRecord,
    bounds: Bounds,
    jac_eq: Float[Array, "n_eq n"] | None = None,
    jac_ineq: Float[Array, "n_ineq n"] | None = None,
) -> ActiveSet:
    """Read the active set from the record's multipliers and slacks.

    With the constraint Jacobians, an active set whose rows are linearly dependent on the free
    coordinates (two identical active rows, say) counts as degenerate too: its KKT system is
    singular, and the barrier system at `mu` is not.
    """
    threshold = DEGENERATE * jnp.sqrt(record.mu)
    below, above = distances(x, bounds)
    z_lower, z_upper = record.bound_multipliers
    lam, slack = record.multipliers_inequality, record.slacks

    def side(multiplier, gap, mask):
        active = mask & (multiplier > gap)
        degenerate = mask & (multiplier < threshold) & (gap < threshold)
        return active, jnp.any(degenerate)

    pinned = bounds.has_lower & bounds.has_upper & (bounds.lower == bounds.upper)
    inequality, deg_i = side(lam, slack, jnp.ones_like(lam, dtype=bool))
    lower, deg_l = side(z_lower, below, bounds.has_lower & ~pinned)
    upper, deg_u = side(z_upper, above, bounds.has_upper & ~pinned)
    on_lower = z_lower >= z_upper
    lower = lower | (pinned & on_lower)
    upper = upper | (pinned & ~on_lower)
    degenerate = deg_i | deg_l | deg_u
    if jac_eq is not None:
        free = (~(lower | upper)).astype(x.dtype)
        rows = jnp.concatenate([jac_eq, jac_ineq * inequality[:, None]]) * free[None, :]
        idle = jnp.concatenate([jnp.zeros(jac_eq.shape[0], bool), ~inequality])
        _, independent = cholesky(rows @ rows.T + jnp.diag(idle.astype(x.dtype)))
        degenerate = degenerate | ~independent
    return ActiveSet(inequality, lower, upper, degenerate, pinned)


def _lagrangian_gradient(
    closures: Closures,
    x: Float[Array, "n"],
    lam_e: Float[Array, "n_eq"],
    lam_i: Float[Array, "n_ineq"],
    args: PyTree,
) -> Float[Array, "n"]:
    """`grad_x (f + lam_e . c_E + lam_i . c_I)`."""
    _, gradient, _, _, jac_eq, jac_ineq = closures.evaluate_with_derivatives(x, args)
    return gradient + jac_eq.T @ lam_e + jac_ineq.T @ lam_i


def _hessian(
    closures: Closures,
    x: Float[Array, "n"],
    lam_e: Float[Array, "n_eq"],
    lam_i: Float[Array, "n_ineq"],
    args: PyTree,
) -> Float[Array, "n n"]:
    """The exact Lagrangian Hessian: the problem's closure, or forward over its gradient."""
    if closures.lagrangian_hessian is not None:
        return closures.lagrangian_hessian(x, lam_e, lam_i, args)
    return jax.jacfwd(lambda y: _lagrangian_gradient(closures, y, lam_e, lam_i, args))(x)


class KKTSystem(eqx.Module):
    """The KKT system at a solution and what its tangent map reads besides.

    Attributes:
        gradient: The objective's gradient.
        jac_eq: The equality rows' Jacobian.
        jac_ineq: The inequality rows' Jacobian.
        hessian: The exact Lagrangian Hessian, or `J_r^T J_r` in the Gauss-Newton form.
        primal: The primal block: the Hessian on the free coordinates, or plus the barrier diagonals.
        rows: The constraint rows of the system, masked to the free coordinates and active rows.
        dual: The dual diagonal: 0 for an equality, 1 for an inactive row, `s / lam` degenerate.
        keep: 1 on a free coordinate, 0 on one an active bound fixes.
        rows_i: 1 on an inequality row in the system, else 0.
        sigma_lower: The lower barrier diagonal `z / d`.
        sigma_upper: The upper barrier diagonal.
        slack_ratio: `s / lam` on the inequality rows.
        active: The active set.
    """

    gradient: Float[Array, "n"]
    jac_eq: Float[Array, "n_eq n"]
    jac_ineq: Float[Array, "n_ineq n"]
    hessian: Float[Array, "n n"]
    primal: Float[Array, "n n"]
    rows: Float[Array, "m n"]
    dual: Float[Array, "m"]
    keep: Float[Array, "n"]
    rows_i: Float[Array, "n_ineq"]
    sigma_lower: Float[Array, "n"]
    sigma_upper: Float[Array, "n"]
    slack_ratio: Float[Array, "n_ineq"]
    active: ActiveSet


def residual_of(closures: Closures, x: Float[Array, "n"], record: SolveRecord) -> Callable:
    """`args -> ((f, grad_x L, c_E, c_I), (grad f, J_E, J_I))` from one evaluation at `x`."""
    lam_e, lam_i = record.multipliers_equality, record.multipliers_inequality

    def residual(a):
        f, g, equality, inequality, jac_eq, jac_ineq = closures.evaluate_with_derivatives(x, a)
        lagrangian = g + jac_eq.T @ lam_e + jac_ineq.T @ lam_i
        return (f, lagrangian, equality, inequality), (g, jac_eq, jac_ineq)

    return residual


def rows_of(closures: Closures, x: Float[Array, "n"]) -> Callable:
    """`args -> ((r, c_E, c_I), (J_E, J_I))` at `x`: the Gauss-Newton form's first-order terms."""

    def rows(a):
        equality, inequality, jac_eq, jac_ineq = closures.constraints_and_jacobians(x, a)
        return (closures.residual(x, a), equality, inequality), (jac_eq, jac_ineq)

    return rows


def system_at(
    closures: Closures,
    bounds: Bounds,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    derivatives: tuple[Float[Array, "n"], Float[Array, "n_eq n"], Float[Array, "n_ineq n"]],
    jacobian: Float[Array, "m n"] | None = None,
) -> KKTSystem:
    """The KKT system at the solution.

    With the residual rows' Jacobian `jacobian` the primal block is `J_r^T J_r`, the
    Gauss-Newton form; without it the exact Lagrangian Hessian, computed once.
    """
    gradient, jac_eq, jac_ineq = derivatives
    lam_e, lam_i = record.multipliers_equality, record.multipliers_inequality
    dtype = x.dtype
    active = active_set(x, record, bounds, jac_eq, jac_ineq)
    degenerate = active.degenerate
    if jacobian is None:
        hessian = _hessian(closures, x, lam_e, lam_i, args)
    else:
        hessian = jacobian.T @ jacobian
    below, above = distances(x, bounds)
    z_lower, z_upper = record.bound_multipliers
    tiny = jnp.finfo(dtype).tiny
    cap = min(_CAP, float(jnp.finfo(dtype).max))
    barrier_lower = bounds.has_lower & ~active.pinned
    barrier_upper = bounds.has_upper & ~active.pinned
    sigma_lower = jnp.where(
        barrier_lower, jnp.minimum(z_lower / jnp.maximum(below, tiny), cap), 0.0
    )
    sigma_upper = jnp.where(
        barrier_upper, jnp.minimum(z_upper / jnp.maximum(above, tiny), cap), 0.0
    )
    # At `mu = 0` (SQP, the augmented Lagrangian) an active slack can be exactly 0, and two
    # identical active rows would then give an exact zero pivot; the floor keeps it finite.
    slack_ratio = jnp.clip(record.slacks / jnp.maximum(lam_i, tiny), jnp.finfo(dtype).eps, cap)
    # A coordinate equal bounds fix is fixed in the degenerate system too, without a barrier.
    fixed = ((active.lower | active.upper) & ~degenerate) | active.pinned
    keep = (~fixed).astype(dtype)
    curvature = hessian + jnp.where(degenerate, jnp.diag(sigma_lower + sigma_upper), 0.0)
    primal = keep[:, None] * curvature * keep[None, :] + jnp.diag(1.0 - keep)
    rows_i = jnp.where(degenerate, 1.0, active.inequality.astype(dtype))
    rows = jnp.concatenate([jac_eq * keep[None, :], rows_i[:, None] * jac_ineq * keep[None, :]])
    dual_i = jnp.where(degenerate, slack_ratio, jnp.where(active.inequality, 0.0, 1.0))
    dual = jnp.concatenate([jnp.zeros((lam_e.shape[0],), dtype), dual_i])
    return KKTSystem(
        gradient, jac_eq, jac_ineq, hessian, primal, rows, dual, keep, rows_i,
        sigma_lower, sigma_upper, slack_ratio, active,
    )  # fmt: skip
