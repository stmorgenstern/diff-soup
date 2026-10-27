"""Shared implicit-function-theorem boundary for the consistent-IC paths.

Both consistent-IC entries -- the pin-differential projection and the
Euler-Newton subset -- must be differentiable through the *converged* root, never
through the Newton iteration that reaches it. This module provides the one shared
autodiff boundary they route their projected state through:
[consistent_project][diff_soup.timestepping.consistent_ic.ift.consistent_project],
a ``custom_jvp`` that serves every differentiation mode -- forward tangents
directly, reverse cotangents through JAX's transposition of its tangent solve,
and repeated differentiation of the rule, so ``jax.hessian``-style compositions
work over it.

The primal value comes from a caller-supplied ``primal_fn`` (the path-specific
Newton solve). The derivative comes from the implicit-function rule linearized at
the very root ``primal_fn`` returns -- no second Newton solve: the parameter,
time, and fixed-derivative sensitivities flow through the shared linearization
([linearize_at_root_jvp][diff_soup.nonlinear.solver.linearize_at_root_jvp]), and
the coupling
from moving the pinned differential rows into the local- and global-algebraic
solution is added on top -- the corrector holds those rows as a
gradient-dead initial guess, so their reprojection response is linearized here at
the root and solved through the same pinned factor. Every factorization runs on the factor contract of
[diff_soup.linear][diff_soup.linear]; no transposed matrix is factored.

The pinned-operator construction differs per path (which rows are held fixed, and
whether the state Jacobian keeps the mass term), but the boundary is shared: the
caller passes the all-static base operator and a ``pin_kind`` naming which rows to
hold, and the rule builds the pinned operator at the frozen primal point.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.nonlinear._kkt import kkt_solve
from diff_soup.nonlinear.solver import _factor_state_block, linearize_at_root_jvp
from diff_soup.timestepping.consistent_ic.pin_differential import PinnedOperator


def _diff_mask_for(base_operator, pin_kind):
    """Boolean state mask of the rows the projection holds fixed (structural).

    ``pin_kind == "differential"`` holds the differential rows (positive mass /
    ``id_mask == 1``); ``pin_kind == "none"`` holds nothing (the full-state solve
    used when every ``y`` is solved against a fixed derivative). Read host-side
    from the problem's structural ``id_mask`` and returned as a plain host array,
    so the mask carried on the pinned operator is a genuine constant rather than a
    traced value -- the operator is the corrector's non-differentiable argument
    and a traced leaf there is rejected under ``jax.jit``. The mask still drives
    ``jnp.where`` selections unchanged.
    """
    n_u = int(base_operator.n_u)
    if pin_kind == "none":
        return np.zeros((n_u,), dtype=bool)
    id_arr = np.asarray(base_operator.problem.id_mask)[:n_u]
    return id_arr > 0


def _pinned_operator(base_operator, diff_mask):
    """Wrap the base operator to hold the ``diff_mask`` rows fixed.

    The pin target is threaded through the step context at solve time, not baked
    into the operator, so the returned operator is a trace-free static object and
    can be passed as the corrector's non-differentiable operator under
    ``jax.jit``.
    """
    return PinnedOperator(base=base_operator, pin_mask=diff_mask)


def _pinned_state_response_jvp(
    base_operator,
    ciop,
    backend,
    analysis,
    ncfg,
    diff_mask,
    u_star,
    lam_star,
    theta,
    ctx,
    u_diff_dot,
):
    """Tangent of the algebraic projection from moving the pinned rows (forward).

    The corrector holds the pinned rows as a gradient-dead initial guess, so the
    coupling from those rows into the local- and global-algebraic solution is
    added here. Linearizes the residual at the converged projection
    ``(u_star, lam_star)``, zeros the pinned rows (held by an identity residual,
    they carry no driving term), and solves the response through the same pinned
    factor. Returns the algebraic-state tangent (zero on the held rows) and the
    bordered-global tangent.
    """

    def residual_of_pinned_rows(u_inject):
        """Residual as a function of the pinned rows only."""
        u = jnp.where(diff_mask, u_inject, u_star)
        return base_operator.residual(u, lam_star, theta, ctx)

    (_R, _g), (r_dot, g_dot) = jax.jvp(residual_of_pinned_rows, (u_star,), (u_diff_dot,))
    r_dot = jnp.where(diff_mask, 0.0, r_dot)

    avals, b, c, d, _topo = ciop.jacobian_blocks(u_star, lam_star, theta, ctx)
    held = _factor_state_block(backend, analysis, avals)
    du, dlam = kkt_solve(held, b, c, d, -r_dot, -g_dot, ncfg.schur_eps, linear=True)
    return du, dlam


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def consistent_project(
    base_operator, backend, analysis, ncfg, pin_kind, primal_fn, u0, lam0, theta, ctx
):
    """Project ``(u0, lam0)`` onto the constraint manifold (forward-mode entry).

    Returns the projected ``(u, lam)``. The primal value is ``primal_fn(u0, lam0,
    theta, ctx)``; the forward tangent comes from the implicit-function rule at
    the converged projection, reverse mode works by JAX transposing that
    tangent solve, and ``jax.hessian``-style compositions differentiate the
    rule again.
    """
    return primal_fn(u0, lam0, theta, ctx)


@consistent_project.defjvp
def _consistent_project_jvp(
    base_operator, backend, analysis, ncfg, pin_kind, primal_fn, primals, tangents
):
    """Forward tangent: implicit-function rule AT the returned root, plus pinned coupling."""
    u0, lam0, theta, ctx = primals
    du0, _dlam0, dtheta, dctx = tangents
    diff_mask = _diff_mask_for(base_operator, pin_kind)
    ciop = _pinned_operator(base_operator, diff_mask)
    # The primal through the entry itself: a second differentiation
    # differentiates the projection by this rule, never through its Newton loop.
    u_out, lam_out = consistent_project(
        base_operator, backend, analysis, ncfg, pin_kind, primal_fn, u0, lam0, theta, ctx
    )

    # Linearize the pinned system at the root the primal actually returns. The
    # pin target enters as a gradient-dead value, exactly as the corrector holds
    # it; its coupling is added explicitly below.
    ctx_pin = ctx.replace_pin(jax.lax.stop_gradient(u0))
    dctx_pin = dctx.replace_pin(jnp.zeros_like(u0))
    du_ift, dlam_ift = linearize_at_root_jvp(
        ciop, backend, analysis, ncfg, u_out, lam_out, theta, ctx_pin, dtheta, dctx_pin
    )
    du_ci = jnp.where(diff_mask, du0, du_ift)
    dlam_ci = dlam_ift

    # Add the reprojection coupling the corrector drops: how the algebraic and
    # constraint rows move when the pinned rows move. The held rows already pass
    # through via the identity restore above.
    u_diff_dot = jnp.where(diff_mask, du0, 0.0)
    du_pin, dlam_pin = _pinned_state_response_jvp(
        base_operator,
        ciop,
        backend,
        analysis,
        ncfg,
        diff_mask,
        u_out,
        lam_out,
        theta,
        ctx,
        u_diff_dot,
    )
    du = du_ci + jnp.where(diff_mask, 0.0, du_pin)
    dlam = dlam_ci + dlam_pin
    return (u_out, lam_out), (du, dlam)


__all__ = ["consistent_project"]
