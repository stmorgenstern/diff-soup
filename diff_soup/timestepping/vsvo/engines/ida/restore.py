"""The exact inverse of the coefficient/history mutations the coefficient recurrence applies.

[ida_restore][] undoes the psi shift and the phi -> phi* rescale that
[ida_set_coeffs][diff_soup.timestepping.vsvo.engines.ida.coeffs.ida_set_coeffs]
applied, so the next attempt re-predicts from the pre-attempt history.

``ida_restore`` handles ONLY the ``psi``/``phi`` history inversion. The
current time and the per-step convergence/error-test failure counters are a
step-loop concern the caller restores and resets separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import MAXORD


def ida_restore(
    phi: jax.Array,
    psi: jax.Array,
    beta: jax.Array,
    kk: jax.Array,
    hh: jax.Array,
    ns: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Undo one attempt's ``psi`` shift and ``phi`` -> ``phi*`` rescale.

    The ``psi`` inverse shift writes ``psi[j - 1] = psi[j] - hh`` for ``j =
    1 .. kk`` in ascending order, reading each ``psi[j]`` before any later
    iteration could touch it -- the exact inverse of ``ida_set_coeffs``'s
    forward shift. Because the source only ever writes ``psi[j - 1]`` for
    ``j`` up to ``kk``, ``psi[kk]`` itself is never touched: it keeps the
    shifted value ``ida_set_coeffs`` left there, since the next
    ``ida_set_coeffs`` call overwrites it before anything reads it.

    The ``phi`` inverse scaling divides each row in ``ns .. kk`` by the
    ``beta`` factor ``ida_set_coeffs`` used to scale it, over the identical
    index range; rows outside ``ns .. kk`` -- including row 0, which is
    never scaled -- are left untouched.

    Both loops are fixed-length unrolls over ``1 .. MAXORD`` masked by the
    traced ``kk``/``ns`` bounds, so no shape here depends on their runtime
    values. The restored ``(phi, psi)`` is the pre-attempt history that the
    next attempt re-predicts from, so it keeps the upstream tangents ``phi``
    and ``psi`` carry from every accepted step before the rejected one. Only
    the step-control scale factors this inversion subtracts and divides by
    (``hh``, ``beta``) are gradient-dead, and each is stopped at its point of
    use -- ``phi`` and ``psi`` themselves are never stopped (the same handling
    [ida_set_coeffs][diff_soup.timestepping.vsvo.engines.ida.coeffs.ida_set_coeffs]
    applies to ``beta`` when it forms ``phi*``).

    Args:
        phi: The ``phi*`` history left by the failed attempt's
            ``ida_set_coeffs`` call, shape ``(MXORDP1, n)``.
        psi: The shifted step-size history left by the same call, shape
            ``(MXORDP1,)``.
        beta: The ``phi`` -> ``phi*`` scale factors that call used, shape
            ``(MXORDP1,)``.
        kk: The order the failed attempt used.
        hh: The step size the failed attempt used.
        ns: The consecutive-constant-step count the failed attempt's
            ``ida_set_coeffs`` call produced.

    Returns:
        The ``(phi, psi)`` pair with the attempt's mutations undone, carrying
        the accepted-step history's tangents (only ``hh``/``beta`` are stopped).
    """
    kk = jnp.asarray(kk)
    ns = jnp.asarray(ns)
    # hh is a step-control scale factor: stop its gradient at the point of use
    # (the psi subtract below) so it stays gradient-dead while psi keeps its
    # own tangent.
    hh = jax.lax.stop_gradient(jnp.asarray(hh))

    psi_restored = psi
    for j in range(1, MAXORD + 1):
        active = j <= kk
        psi_restored = psi_restored.at[j - 1].set(
            jnp.where(active, psi_restored[j] - hh, psi_restored[j - 1])
        )

    phi_restored = phi
    for j in range(1, MAXORD + 1):
        scale_j = (j >= ns) & (j <= kk)
        # beta is a gradient-dead scale factor; stop it at the point of use so
        # phi keeps its tangent through the (1/beta) rescale. Inactive lanes
        # (masked out below) carry beta == 0 before the history warms up:
        # guard the denominator so their masked division is 0-free — the
        # selected value is unchanged (active lanes always have beta != 0),
        # and reverse-mode cotangents through the mask stay finite instead of
        # producing 0/0 in the dead lanes.
        beta_j = jax.lax.stop_gradient(beta[j])
        beta_safe = jnp.where(beta_j == 0, jnp.ones_like(beta_j), beta_j)
        phi_restored = phi_restored.at[j].set(
            jnp.where(scale_j, phi_restored[j] / beta_safe, phi_restored[j])
        )

    return phi_restored, psi_restored


__all__ = ["ida_restore"]
