"""The implicit derivative of a constrained solution with respect to `args`.

A compiled solver's `run` is one `custom_jvp`. Its rule never differentiates the iterations:
at the solution `(x*, lam*)` it solves the KKT system

    [ H       J_E^T   J_A^T ] [dx   ]     [ d_args grad_x L ]
    [ J_E     0       0     ] [dlamE] = - [ d_args c_E      ]
    [ J_A     0       0     ] [dlamA]     [ d_args c_A      ]

with `H` the exact Lagrangian Hessian, computed once, and `A` the active inequalities; an
active bound fixes its coordinate (`dx_j = 0`) and its multiplier takes the stationarity row.
The active set is read from the multipliers and the slacks: a row is active where its
multiplier exceeds its slack. Where both are below `DEGENERATE * sqrt(mu)` the active set is
degenerate, and the rule instead differentiates the barrier solution at the final `mu`, whose
system replaces the active-set masks with the barrier diagonals `Z / d` and `s / lam`; the
record's `degenerate` says so.

The rule is linear in the tangents (one `jvp` of the KKT residual in `args`, one dense solve),
so JAX transposes it for reverse mode, as in `diff_soup.problem.implicit.root_ift`. Its primal
runs through the entry, so differentiating the rule again differentiates the solution by the
rule. A run that did not end with `KKT_TOL` has no supported derivative: its tangents are NaN,
and its input tangents are zeroed first so a reverse pass gives it exact zero cotangents.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.custom_derivatives import zero_from_primal
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree

from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import KKTSolve, cholesky
from diff_soup.optimization.solvers._barrier import Bounds, distances
from diff_soup.optimization.solvers._closures import Closures
from diff_soup.optimization.solvers._protocol import SolveRecord

#: Ours: a multiplier and its slack both below `DEGENERATE * sqrt(mu)` make the active set degenerate.
DEGENERATE = 1e2
#: Ours: the cap on the degenerate system's barrier diagonals, so the branch that is not
#: selected stays finite and its zero cotangent never meets an infinity.
_CAP = 1e200


class ActiveSet(eqx.Module):
    """The active set at a solution.

    Attributes:
        inequality: Where an inequality row is active, `(n_ineq,)`.
        lower: Where a lower bound is active, `(n,)`.
        upper: Where an upper bound is active, `(n,)`.
        degenerate: Whether a multiplier and its slack (or bound distance) are both near zero.
    """

    inequality: Bool[Array, "n_ineq"]
    lower: Bool[Array, "n"]
    upper: Bool[Array, "n"]
    degenerate: Bool[Array, ""]


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

    inequality, deg_i = side(lam, slack, jnp.ones_like(lam, dtype=bool))
    lower, deg_l = side(z_lower, below, bounds.has_lower)
    upper, deg_u = side(z_upper, above, bounds.has_upper)
    degenerate = deg_i | deg_l | deg_u
    if jac_eq is not None:
        free = (~(lower | upper)).astype(x.dtype)
        rows = jnp.concatenate([jac_eq, jac_ineq * inequality[:, None]]) * free[None, :]
        idle = jnp.concatenate([jnp.zeros(jac_eq.shape[0], bool), ~inequality])
        _, independent = cholesky(rows @ rows.T + jnp.diag(idle.astype(x.dtype)))
        degenerate = degenerate | ~independent
    return ActiveSet(inequality, lower, upper, degenerate)


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


def kkt_tangent(
    closures: Closures,
    bounds: Bounds,
    kkt: KKTSolve,
    x: Float[Array, "n"],
    record: SolveRecord,
    args: PyTree,
    dargs: PyTree,
) -> tuple[Float[Array, "n"], SolveRecord, Bool[Array, ""]]:
    """The tangent of `(x*, record)` along `dargs`, linear in `dargs`, and whether its system solved.

    One forward derivative of one evaluation gives the derivatives of the objective, of the
    Lagrangian gradient and of the rows in `args`, and its primal gives the Jacobians. The
    system goes through the solver's [KKTSolve][diff_soup.optimization.solvers.KKTSolve], with
    the inactive rows' dual diagonal 1, and does not solve when its inertia is not `(n, m, 0)`.
    """
    lam_e, lam_i = record.multipliers_equality, record.multipliers_inequality
    n_eq = lam_e.shape[0]
    dtype = x.dtype

    def residual(a):
        f, g, equality, inequality, jac_eq, jac_ineq = closures.evaluate_with_derivatives(x, a)
        lagrangian = g + jac_eq.T @ lam_e + jac_ineq.T @ lam_i
        return (f, lagrangian, equality, inequality), (g, jac_eq, jac_ineq)

    _, (d_f, d_grad, d_eq, d_ineq), (gradient, jac_eq, jac_ineq) = jax.jvp(
        residual, (args,), (dargs,), has_aux=True
    )
    active = active_set(x, record, bounds, jac_eq, jac_ineq)
    degenerate = active.degenerate
    hessian = _hessian(closures, x, lam_e, lam_i, args)
    below, above = distances(x, bounds)
    z_lower, z_upper = record.bound_multipliers
    tiny = jnp.finfo(dtype).tiny
    sigma_lower = jnp.where(
        bounds.has_lower, jnp.minimum(z_lower / jnp.maximum(below, tiny), _CAP), 0.0
    )
    sigma_upper = jnp.where(
        bounds.has_upper, jnp.minimum(z_upper / jnp.maximum(above, tiny), _CAP), 0.0
    )
    sigma = sigma_lower + sigma_upper
    # At `mu = 0` (SQP, the augmented Lagrangian) an active slack can be exactly 0, and two
    # identical active rows would then give an exact zero pivot; the floor keeps it finite.
    slack_ratio = jnp.clip(record.slacks / jnp.maximum(lam_i, tiny), jnp.finfo(dtype).eps, _CAP)

    fixed = (active.lower | active.upper) & ~degenerate
    keep = (~fixed).astype(dtype)
    primal = jnp.where(
        degenerate,
        hessian + jnp.diag(sigma),
        keep[:, None] * hessian * keep[None, :] + jnp.diag(1.0 - keep),
    )
    rows_i = jnp.where(degenerate, 1.0, active.inequality.astype(dtype))
    jac_eq_m = jac_eq * keep[None, :]
    jac_ineq_m = rows_i[:, None] * jac_ineq * keep[None, :]
    dual_i = jnp.where(degenerate, slack_ratio, jnp.where(active.inequality, 0.0, 1.0))
    dx, dlam, solved = kkt.solve(
        primal,
        jnp.concatenate([jac_eq_m, jac_ineq_m]),
        0.0,
        jnp.concatenate([jnp.zeros((n_eq,), dtype), dual_i]),
        -d_grad * keep,
        -jnp.concatenate([d_eq, rows_i * d_ineq]),
    )
    dlam_e, dlam_i = dlam[:n_eq], dlam[n_eq:]

    stationarity = hessian @ dx + jac_eq.T @ dlam_e + jac_ineq.T @ dlam_i + d_grad
    dz_lower = jnp.where(
        degenerate,
        -sigma_lower * dx,
        jnp.where(active.lower, stationarity, 0.0),
    )
    dz_upper = jnp.where(
        degenerate,
        sigma_upper * dx,
        jnp.where(active.upper, -stationarity, 0.0),
    )
    ds = jnp.where(
        degenerate,
        -slack_ratio * dlam_i,
        jnp.where(active.inequality, 0.0, -(jac_ineq @ dx + d_ineq)),
    )
    tangent = eqx.tree_at(
        lambda r: (
            r.multipliers_equality,
            r.multipliers_inequality,
            r.slacks,
            r.bound_multipliers,
            r.value,
        ),
        jax.tree.map(_zero, record),
        (dlam_e, dlam_i, ds, (dz_lower, dz_upper), d_f + gradient @ dx),
    )
    return dx, tangent, solved


def _zero(leaf: Array) -> ArrayLike:
    """The zero tangent of a gradient-dead leaf (`float0` for an integer or bool)."""
    return zero_from_primal(leaf, symbolic_zeros=False)


def _poison(tree: PyTree, valid: Bool[Array, ""]) -> PyTree:
    """NaN in every inexact leaf of `tree` where `valid` is false."""

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(valid, leaf, jnp.full_like(leaf, jnp.nan))

    return jax.tree.map(gate, tree)


def _zero_where_not(keep: Bool[Array, ""], tree: PyTree) -> PyTree:
    """Exact zeros in every inexact leaf of `tree` where `keep` is false."""

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(keep, leaf, jnp.zeros_like(leaf))

    return jax.tree.map(gate, tree)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _entry(primal, prepare, kkt, static, dynamic, x0, args, warm):
    """`primal(problem, x0, args, warm) -> (x, record)` with the implicit derivative in `args`."""
    del prepare, kkt
    return primal(eqx.combine(dynamic, static), x0, args, warm)


@_entry.defjvp
def _entry_jvp(primal, prepare, kkt, static, primals, tangents):
    """The KKT tangent at the solution; `x0`, `warm` and the problem's arrays are gradient-dead."""
    dynamic, x0, args, warm = primals
    _, _, dargs, _ = tangents
    out = _entry(primal, prepare, kkt, static, dynamic, x0, args, warm)
    x, record = out
    closures, bounds = prepare(eqx.combine(dynamic, static), x)
    valid = record.status == TerminationStatus.KKT_TOL
    dargs = _zero_where_not(valid, dargs)
    dx, drecord, solved = kkt_tangent(closures, bounds, kkt, x, record, args, dargs)
    return out, _poison((dx, drecord), valid & solved)


def implicit_run(
    primal: Callable,
    prepare: Callable,
    kkt: KKTSolve,
    problem: PyTree,
    x0: Float[ArrayLike, "n"],
    args: PyTree,
    warm: SolveRecord | None,
) -> tuple[Float[Array, "n"], SolveRecord]:
    """Run `primal` with the implicit derivative of its solution in `args`.

    Args:
        primal: `(problem, x0, args, warm) -> (x, record)`, the solver's loop; never differentiated.
        prepare: `(problem, x) -> (closures, bounds)`, the raw closures and bounds of `problem`.
        kkt: The backend that solves the derivative's KKT system.
        problem: The numeric problem; its arrays (the bounds) are gradient-dead.
        x0: The start point, gradient-dead.
        args: What the derivative is taken in.
        warm: The warm-start record or `None`, gradient-dead.

    Returns:
        `(x, record)`.
    """
    dynamic, static = eqx.partition(problem, eqx.is_array)
    return _entry(primal, prepare, kkt, static, dynamic, x0, args, warm)
