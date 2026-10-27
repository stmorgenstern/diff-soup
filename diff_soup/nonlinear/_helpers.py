"""Branchless helpers for the unified box-constrained Newton.

These primitives back [diff_soup.nonlinear.box_newton][] and are
shared between the forward primal and the reduced-space IFT VJP/JVP rules.

* [benson_munson_bind][diff_soup.nonlinear._helpers.benson_munson_bind] -- residual-sign bind rule (Benson-Munson 2006);
  the same definition is used at every Newton iter (forward) and at the
  converged point (IFT).
* [projected_armijo_scan][diff_soup.nonlinear._helpers.projected_armijo_scan] -- fixed-alpha schedule scanned via
  `jax.vmap` to avoid host-side branching during the line search.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def benson_munson_bind(u, F, lb, ub, mask, *, tol_b=1e-6, tol_F=1e-12):
    """Return ``(bind_lower, bind_upper)`` Boolean masks (Benson-Munson 2006).

    A row is bound to its lower bound iff ``u_i <= lb + tol_b`` *and*
    ``F_i >= -tol_F`` (i.e., the residual wants ``u_i`` to stay at or
    below ``lb``). Symmetrically for the upper bound.

    Branchless: every output is a `jnp.where`. Used both forward
    (to pin rows in the iteration matrix) and in the IFT (to define the
    fixed-active-set reduced system).

    Args:
        u: Current iterate, shape ``(n,)``.
        F: Residual ``R(u)`` at the iterate, shape ``(n,)``.
        lb: Per-DOF lower bound, shape ``(n,)``.
        ub: Per-DOF upper bound, shape ``(n,)``.
        mask: Per-DOF constrained mask (bool), shape ``(n,)``.
        tol_b: Bound-proximity tolerance.
        tol_F: Residual-sign tolerance.

    Returns:
        ``(bind_l, bind_u)`` boolean arrays each of shape ``(n,)``.
    """
    at_lb = mask & (u <= lb + tol_b)
    at_ub = mask & (u >= ub - tol_b)
    bind_l = at_lb & (F >= -tol_F)
    bind_u = at_ub & (F <= tol_F)
    return bind_l, bind_u


def projected_armijo_scan(
    x,
    d,
    lb,
    ub,
    merit: Callable,
    g0,
    *,
    alphas,
    c1: float = 1e-4,
):
    """Sync-free projected Armijo over a fixed step-size schedule.

    Evaluates ``merit(clip(x + alpha*d, lb, ub))`` at every alpha in
    ``alphas`` in parallel via `jax.vmap`, then picks the *first*
    (largest) alpha whose projected step satisfies the Armijo descent
    condition ``merit(x_a) <= merit(x) + c1 * alpha * g^T d``. If no
    alpha accepts, the smallest is returned (degraded but
    progress-making, matches PETSc SNESLineSearchBT at the schedule
    floor).

    Args:
        x: Current iterate, shape ``(n,)``.
        d: Search direction, shape ``(n,)``.
        lb: Per-DOF lower bound, shape ``(n,)``.
        ub: Per-DOF upper bound, shape ``(n,)``.
        merit: Scalar merit function of the projected iterate.
        g0: Surrogate gradient whose ``g0^T d`` is the descent slope.
        alphas: Fixed step-size schedule (descending), shape ``(n_alpha,)``.
        c1: Armijo descent constant.

    Returns:
        ``(alpha, x_new)`` — the accepted step size and projected iterate.
    """
    f0 = merit(x)
    descent = jnp.dot(g0, d)

    def at(a):
        x_a = jnp.clip(x + a * d, lb, ub)
        return merit(x_a), x_a

    f_vec, x_vec = jax.vmap(at)(alphas)
    armijo_rhs = f0 + c1 * alphas * descent
    accept = f_vec <= armijo_rhs
    idx = jnp.where(jnp.any(accept), jnp.argmax(accept), alphas.size - 1)
    return alphas[idx], x_vec[idx]
