"""The implicit-function derivatives of the initialization roots.

Each route has one derivative entry, a ``custom_jvp``
([consistent_ic_root][diff_soup.problem.implicit.consistent_init.rules.consistent_ic_root]
for the joint solve,
[consistent_ic_root_reduced][diff_soup.problem.implicit.consistent_init.rules.consistent_ic_root_reduced]
for the algebraic-block elimination). Its rule is linear in the tangents, so JAX derives
reverse mode by transposing it: the cotangent solve factors the transposed matrix
directly, on the plan's reverse precision policy. The rule computes its primal through
the entry itself, so a further differentiation (``jax.hessian``, reverse over forward)
differentiates the root by the same rule, never through the Newton loop. A solve that
fails to converge, or whose sensitivity system is singular, has no supported derivative:
its tangents are NaN, its reverse-mode cotangents are exact zeros (the rule zeroes the
tangents of an invalid root, and a transposition cannot tell a consumed cotangent from a
discarded one), and the returned diagnostics record the failure.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero, zero_from_primal

from diff_soup.nonlinear._kkt import RouteFactor, transposable_solve
from diff_soup.problem.implicit.consistent_init.newton import (
    _g_root_values,
    _k_root_values,
    _newton_primal,
    _reduced_newton_primal,
    _routes,
)
from diff_soup.problem.implicit.consistent_init.plan import ConsistentInitPlan


def _root_tangent(icp: ConsistentInitPlan, values, n: int, rhs):
    """The tangent solve ``x = M^-1 rhs`` at the root.

    A [transposable_solve][diff_soup.nonlinear._kkt.transposable_solve]
    (linear in ``rhs``; JAX's transposition of it factors ``M^T`` directly on
    the plan's reverse route at its adjoint dtype).
    """
    values = jnp.asarray(values)
    route, reverse = _routes(icp, values, n)
    return transposable_solve(
        route, values, rhs, reverse_route=reverse, adjoint_dtype=icp.adjoint_dtype
    )


def _root_solvable(icp: ConsistentInitPlan, values, n: int):
    """Whether the matrix at the root solves.

    The checked solve of a constant right-hand side with a factor of ``M``
    (see [RouteFactor.solves][diff_soup.nonlinear._kkt.RouteFactor.solves]), a
    function of primal values only, so a rule that gates on it stays linear
    in the tangents.
    """
    route, _reverse = _routes(icp, jnp.asarray(values), n)
    return RouteFactor(route, jnp.asarray(values)).solves()


def _zero_where_not(keep, tree):
    """Replace every inexact leaf of ``tree`` with exact zeros where ``keep`` is false.

    A selection, not a product: an infinite or NaN leaf that is not kept
    becomes zero instead of NaN, and so does its cotangent under
    transposition. Integer (``float0``) leaves and ``None`` pass unchanged.
    """
    keep = jnp.asarray(keep)

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(keep, leaf, jnp.zeros_like(leaf))

    return jax.tree_util.tree_map(gate, tree)


def _gated_root_tangent(icp: ConsistentInitPlan, values, n: int, rhs_dot, valid):
    """The tangent solve ``M dz = -rhs_dot`` with its right-hand side and result gated on ``valid``.

    On an invalid root both are exact zeros, so the transposed solve neither
    receives nor returns a cotangent there: a singular matrix cannot turn a
    discarded cotangent into NaN.
    """
    rhs = _zero_where_not(valid, -rhs_dot)
    dz = _root_tangent(icp, values, n, rhs)
    return _zero_where_not(valid, dz)


def _pinned_residual_fn(icp: ConsistentInitPlan, t0, y_star, ydot_star):
    """The residual at the frozen root as a function of the differentiable inputs."""
    diff_mask = jnp.asarray(icp.diff_mask)

    def pinned_residual(y0_in, theta_in, control_in):
        """Residual with pinned rows from ``y0_in`` and solved values frozen."""
        y = jnp.where(diff_mask, y0_in, y_star)
        return icp.plan._residual(t0, y, ydot_star, theta_in, control_in)

    return pinned_residual


def _instantiated(tangents):
    """``tangents`` with every symbolic zero replaced by an array of zeros."""

    def fill(t):
        if not isinstance(t, SymbolicZero):
            return t
        if t.dtype == jax.dtypes.float0:
            return np.zeros(t.shape, dtype=jax.dtypes.float0)
        return jnp.zeros(t.shape, t.dtype)

    return jax.tree_util.tree_map(fill, tangents, is_leaf=lambda x: isinstance(x, SymbolicZero))


def _zero_cotangent(x):
    """Zero cotangent (or tangent) matching ``x``'s pytree and leaf avals."""
    return jax.tree_util.tree_map(lambda leaf: zero_from_primal(leaf, symbolic_zeros=False), x)


def _poison(x, valid):
    """Replace ``x`` with NaN when the root or its sensitivity solve is invalid."""
    return jnp.where(valid, x, jnp.full_like(x, jnp.nan))


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def consistent_ic_root(icp, t0, y0, ydot_guess, theta, control):
    """Joint consistent-initialization root with its implicit-function derivative.

    Returns ``(y, ydot, residual_norm, iterations, converged)``. Tangents come
    from the implicit-function rule at the converged root; reverse mode is
    JAX's transposition of that rule, whose cotangent solve factors ``K^T``
    directly on the plan's reverse policy. Diagnostics are gradient-dead.
    """
    return _newton_primal(icp, t0, y0, ydot_guess, theta, control)


@functools.partial(consistent_ic_root.defjvp, symbolic_zeros=True)
def _consistent_ic_root_jvp(icp, primals, tangents):
    """Tangent: exact implicit solve in the joint unknowns at the root."""
    t0, y0, ydot_guess, theta, control = primals
    dt0, dy0, _dydot_guess, dtheta, dcontrol = tangents
    fixed_time = isinstance(dt0, SymbolicZero)
    dt0, dy0, dtheta, dcontrol = _instantiated((dt0, dy0, dtheta, dcontrol))
    # The primal through the entry itself: differentiating this rule again
    # differentiates the root by this rule, never through the Newton loop.
    out = consistent_ic_root(icp, t0, y0, ydot_guess, theta, control)
    y_star, ydot_star, norm, iterations, converged = out
    diff_mask = jnp.asarray(icp.diff_mask)

    values = _k_root_values(icp, t0, y_star, ydot_star, theta, control)
    valid = jnp.logical_and(converged, _root_solvable(icp, values, icp.n))
    pinned_residual = _pinned_residual_fn(icp, t0, y_star, ydot_star)
    dy0_pinned = _zero_where_not(valid, jnp.where(diff_mask, dy0, 0.0))
    dtheta, dcontrol = _zero_where_not(valid, (dtheta, dcontrol))
    _, rhs_dot = jax.jvp(pinned_residual, (y0, theta, control), (dy0_pinned, dtheta, dcontrol))
    if not fixed_time:
        _, time_dot = jax.jvp(
            lambda t: icp.plan._residual(t, y_star, ydot_star, theta, control),
            (t0,),
            (_zero_where_not(valid, dt0),),
        )
        rhs_dot = rhs_dot + time_dot
    dz = _gated_root_tangent(icp, values, icp.n, rhs_dot, valid)

    dy = jnp.where(diff_mask, dy0, dz)
    dydot = jnp.where(diff_mask, dz, jnp.zeros_like(dz))
    return out, (
        _poison(dy, valid),
        _poison(dydot, valid),
        _zero_cotangent(norm),
        _zero_cotangent(iterations),
        _zero_cotangent(converged),
    )


def _pinned_alg_residual_fn(icp: ConsistentInitPlan, t0, y_star):
    """The frozen-root algebraic rows as a function of the differentiable inputs."""
    diff_mask = jnp.asarray(icp.diff_mask)
    a_idx = jnp.asarray(icp.a_idx)
    zeros = jnp.zeros_like(y_star)

    def pinned_residual(y0_in, theta_in, control_in):
        """Algebraic rows with pinned values from ``y0_in``, solved values frozen."""
        y = jnp.where(diff_mask, y0_in, y_star)
        return icp.plan._residual(t0, y, zeros, theta_in, control_in)[a_idx]

    return pinned_residual


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _reduced_ic_root(icp, t0, y0, theta, control):
    """Reduced algebraic root with its implicit-function derivative.

    Returns ``(y, iterations, converged)`` where ``y`` carries the pinned
    differential values and the solved algebraic values. Tangents come from
    the implicit-function rule on the algebraic block at the converged root
    (reverse mode is its transposition); diagnostics are gradient-dead.
    """
    return _reduced_newton_primal(icp, t0, y0, theta, control)


@functools.partial(_reduced_ic_root.defjvp, symbolic_zeros=True)
def _reduced_ic_root_jvp(icp, primals, tangents):
    """Tangent: exact implicit solve on the algebraic block at the root."""
    t0, y0, theta, control = primals
    dt0, dy0, dtheta, dcontrol = tangents
    fixed_time = isinstance(dt0, SymbolicZero)
    dt0, dy0, dtheta, dcontrol = _instantiated((dt0, dy0, dtheta, dcontrol))
    # The primal through the entry itself, as in the joint rule.
    out = _reduced_ic_root(icp, t0, y0, theta, control)
    y_star, iterations, converged = out
    diff_mask = jnp.asarray(icp.diff_mask)
    a_idx = jnp.asarray(icp.a_idx)

    values = _g_root_values(icp, t0, y_star, theta, control)
    valid = jnp.logical_and(converged, _root_solvable(icp, values, icp.n_a))
    pinned_residual = _pinned_alg_residual_fn(icp, t0, y_star)
    dy0_pinned = _zero_where_not(valid, jnp.where(diff_mask, dy0, 0.0))
    dtheta, dcontrol = _zero_where_not(valid, (dtheta, dcontrol))
    _, rhs_dot = jax.jvp(pinned_residual, (y0, theta, control), (dy0_pinned, dtheta, dcontrol))
    if not fixed_time:
        _, time_dot = jax.jvp(
            lambda t: icp.plan._residual(t, y_star, jnp.zeros_like(y_star), theta, control)[a_idx],
            (t0,),
            (_zero_where_not(valid, dt0),),
        )
        rhs_dot = rhs_dot + time_dot
    dz = _gated_root_tangent(icp, values, icp.n_a, rhs_dot, valid)

    dz_full = jnp.zeros_like(y_star).at[a_idx].set(dz, unique_indices=True, indices_are_sorted=True)
    dy = jnp.where(diff_mask, dy0, dz_full)
    return out, (
        _poison(dy, valid),
        _zero_cotangent(iterations),
        _zero_cotangent(converged),
    )


def _reconstruct_slopes(icp: ConsistentInitPlan, t0, y, theta, control):
    """Reconstruct the differential slopes ``ydot_d = f_d / D_dd``.

    Plain differentiable code: the diagonal comes from the derivative
    operator applied to ones (exact for diagonal support) and ``f`` from the
    residual at zero slopes. Invertibility is a value property checked here,
    at the solved point: a non-finite or zero diagonal entry on a
    differential slot fails loudly — the returned slopes are NaN and
    ``diag_ok`` is false.
    """
    diff_mask = jnp.asarray(icp.diff_mask)
    diag = icp._derivative_apply(t0, y, theta, control, jnp.ones_like(y))
    f = -icp.plan._residual(t0, y, jnp.zeros_like(y), theta, control)
    on_diff = jnp.isfinite(diag) & (diag != 0.0)
    diag_ok = jnp.all(jnp.where(diff_mask, on_diff, True))
    safe_diag = jnp.where(diff_mask & on_diff, diag, jnp.ones_like(diag))
    ydot = jnp.where(diff_mask, f / safe_diag, 0.0)
    return _poison(ydot, diag_ok), diag_ok


def _reduced_root_and_slopes(icp: ConsistentInitPlan, root, t0, y0, ydot_guess, theta, control):
    """Compose one reduced root entry with slope reconstruction.

    Same five outputs as the joint entries. The slope guess is unused: the
    reduced route reconstructs the slopes exactly, so the guess carries no
    derivative by construction. The reported norm is the full residual at
    the composed point (the common post-solve metric), gradient-dead.
    """
    del ydot_guess
    if icp.n_a == 0:
        y_star = jnp.asarray(y0)
        iterations = jnp.asarray(0, jnp.int32)
        alg_ok = jnp.asarray(True)
    else:
        y_star, iterations, alg_ok = root(icp, t0, y0, theta, control)
    ydot, diag_ok = _reconstruct_slopes(icp, t0, y_star, theta, control)
    r = icp.plan._residual(t0, y_star, ydot, theta, control)
    norm = jax.lax.stop_gradient(jnp.linalg.norm(r))
    converged = alg_ok & diag_ok & jnp.isfinite(norm)
    return y_star, ydot, norm, iterations, converged


def consistent_ic_root_reduced(icp, t0, y0, ydot_guess, theta, control):
    """Reduced consistent-initialization solve with its implicit-function derivative.

    Available for a diagonal invertible mass only (see
    [reduced_route_eligible][diff_soup.problem.implicit.consistent_init.reduced_route_eligible]): the algebraic values solve on the exact
    algebraic block and the differential slopes reconstruct from the mass
    diagonal. Same outputs and derivative contract as
    [consistent_ic_root][diff_soup.problem.implicit.consistent_init.consistent_ic_root].
    """
    return _reduced_root_and_slopes(icp, _reduced_ic_root, t0, y0, ydot_guess, theta, control)
