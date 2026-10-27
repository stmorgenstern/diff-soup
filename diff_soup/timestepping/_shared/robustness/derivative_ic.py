"""Backend-neutral consistent-derivative recovery for index-1 mass-matrix DAEs.

A differential-algebraic problem ``M(t, y) ydot = f(t, y)`` needs a consistent
initial derivative ``ydot`` to seed a multistep history. On differential rows
(positive mass), including differential globals, the derivative comes from the
canonical startup slope. Local algebraic rows and algebraic bordered globals
have no direct slope: they move as the differential state evolves, so their
derivatives must be recovered from the manifold itself.

[recover_algebraic_derivative][] is the finite-difference recovery shared by
the IDA-class and BDF-class engines so they cannot diverge:

  1. take the projected state ``(u_proj, lam_proj)`` already on the algebraic
     manifold and the local differential-row derivative ``udot_diff``;
  2. take a tiny Euler step on the differential rows: ``u_eps_diff = u_diff +
     eps * udot_diff``;
  3. re-project the local algebraic state and algebraic globals at ``t0 + eps``
     with the Euler-stepped differential rows held fixed (the caller's
     ``reproject``);
  4. finite-difference those algebraic derivatives across the probe:
     ``ydot_alg = (u_alg_eps - u_alg) / eps``, ``lamdot = (lam_eps - lam) / eps``.

The helper is purely numeric: it takes the projected state, the mass partition,
the differential derivative, and a ``reproject`` callback, and returns the full
derivative and the algebraic-global derivative. The two callers adapt their own
problem/operator interfaces (and their own projection machinery) into the
``reproject`` callback, so the recovery itself is single-source.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def recover_algebraic_derivative(
    *,
    u_proj,
    lam_proj,
    udot_diff_full,
    alg_mask,
    reproject,
    eps_du,
):
    """Return ``(ydot, lamdot)`` recovered by an Euler probe + finite difference.

    Args:
        u_proj: The state projected onto the algebraic manifold at ``t0``.
        lam_proj: The algebraic bordered-global values at that projection.
        udot_diff_full: The differential-row derivative ``M^{-1} f`` over the full
            state vector, zero on the algebraic rows. Its differential rows drive
            the Euler probe; its values are returned unchanged on the differential
            rows of the result.
        alg_mask: Boolean mask, ``True`` on the algebraic (zero-mass) rows whose
            derivative is recovered by finite difference.
        reproject: Callback ``(u_eps_seed, lam_seed) -> (u_eps, lam_eps)`` that
            re-projects onto the algebraic manifold at ``t0 + eps_du`` holding the
            Euler-stepped differential rows of ``u_eps_seed`` fixed. The caller
            bakes ``t0 + eps_du`` and its projection machinery into the closure.
        eps_du: The Euler / finite-difference step. ``1e-8`` is the float64 sweet
            spot: smaller is more accurate but more sensitive to roundoff.

    Returns:
        ``(ydot, lamdot)`` -- the full consistent derivative (differential rows
        from ``udot_diff_full``, algebraic rows from the finite difference) and
        the algebraic-global derivative (empty when there are no algebraic
        globals).
    """
    # Euler step on the differential rows only: udot_diff_full is zero on the
    # algebraic rows, so adding it moves the differential rows and leaves the
    # algebraic rows at their projected (seed) values.
    u_eps_seed = u_proj + eps_du * udot_diff_full
    u_eps, lam_eps = reproject(u_eps_seed, lam_proj)

    # The recovered rows are a startup heuristic and stay gradient-dead.
    # Differentiating the quotient scales every cotangent by ``1/eps_du``
    # across two independently-iterated projections whose sensitivities only
    # cancel in exact arithmetic; a baseline already at its root takes a
    # zero-update early exit whose unrolled derivative is the seed's, not the
    # root's, so the mismatch is order one and the channel explodes. The
    # trajectory's parameter gradient is owned by the per-step corrector
    # roots, not the history seed.
    ydot_alg = jax.lax.stop_gradient((u_eps - u_proj) / eps_du)
    ydot = jnp.where(alg_mask, ydot_alg, udot_diff_full)
    # Empty (shape (0,)) when there are no algebraic globals; the subtraction
    # is well defined either way.
    lamdot = jax.lax.stop_gradient((lam_eps - lam_proj) / eps_du)
    return ydot, lamdot


__all__ = ["recover_algebraic_derivative"]
