"""Dense output for the fully-implicit DAE engine.

[ida_get_solution][diff_soup.timestepping.vsvo.engines.ida.dense.ida_get_solution] evaluates ``(y(t), ydot(t))`` at any query time
``t`` on the accepted interval ``[tn - hused, tn]`` from the
divided-difference history ``phi`` and the post-step ``psi`` array left
by the last accepted step. The driver's dense-output seam calls this on every
accepted step to fill the explicit save grid.

Every state and bordered row is reported at interpolation accuracy. Strict
algebraic constraint satisfaction belongs to the projected output policy.
Interpolated algebraic-row derivatives remain internal and must be zeroed if
they are ever exposed publicly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import MAXORD


def _dense_coeffs(
    t: jax.Array,
    tn: jax.Array,
    psi: jax.Array,
    kused: jax.Array,
    maxord: int = MAXORD,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Newton-form interpolation weights ``(cvals, dvals, kord)``.

    ``cvals`` has shape ``(maxord + 1,)`` and ``dvals`` has shape ``(maxord,)``
    -- fixed shapes matching the ``phi`` history regardless of the traced
    order ``kused``. The order actually used is ``kord = max(kused, 1)``: the
    very first accepted step reports ``kused == 0``, which is treated as
    order 1 for dense output.

    The recurrence carries three running scalars
    ``c``, ``d``, ``gam`` through a fixed Python loop over ``j = 1 ..
    maxord``, active only while ``j <= kord``; entries at ``j > kord`` are
    left at their zero initialization. A row's denominator (``psi[j-1]`` or
    ``psi[j]``) is only ever zero outside the active range (unused history
    slots), so the division is guarded there to keep the whole computation
    finite -- the guarded branch is always discarded by the ``j <= kord``
    mask.
    """
    dtype = psi.dtype
    kord = jnp.maximum(jnp.asarray(kused, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    delt = t - tn

    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)

    def _safe(denom: jax.Array) -> jax.Array:
        return jnp.where(denom == 0, one, denom)

    c = one
    d = zero
    gam = delt / _safe(psi[0])

    cvals = jnp.zeros(maxord + 1, dtype=dtype).at[0].set(one)
    dvals = jnp.zeros(maxord, dtype=dtype)

    for j in range(1, maxord + 1):
        active = j <= kord
        psi_prev = psi[j - 1]
        psi_here = psi[j]

        d_new = d * gam + c / _safe(psi_prev)
        c_new = c * gam
        gam_new = (delt + psi_prev) / _safe(psi_here)

        c = jnp.where(active, c_new, c)
        d = jnp.where(active, d_new, d)
        gam = jnp.where(active, gam_new, gam)

        cvals = cvals.at[j].set(jnp.where(active, c, cvals[j]))
        dvals = dvals.at[j - 1].set(jnp.where(active, d, dvals[j - 1]))

    return cvals, dvals, kord


def ida_get_solution(
    t: jax.Array,
    tn: jax.Array,
    phi: jax.Array,
    psi: jax.Array,
    kused: jax.Array,
    maxord: int = MAXORD,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the dense interpolant ``(y(t), ydot(t))`` on the last accepted step.

    Valid for ``t`` on ``[tn - hused, tn]`` (``hused = psi[0]`` after a step),
    up to a fuzz tolerance of ``100 * uround * (|tn| + |hh|)``; this function
    does not itself clamp or validate the query time.

    ``yret`` is the Newton-form combination ``sum_{j=0..kord} cvals[j] *
    phi[j]``; ``ypret`` is ``sum_{j=1..kord} dvals[j-1] * phi[j]``, with
    ``cvals``/``dvals`` from [_dense_coeffs][diff_soup.timestepping.vsvo.engines.ida.dense._dense_coeffs]. Both sums run over the
    fixed ``phi`` leading axis (``cvals``/``dvals`` are zero past ``kord``),
    so ``kused`` stays a traced integer while every array keeps its
    compile-time shape.

    ``phi`` is not stopped, so both outputs keep the history's tangent (each
    is linear in ``phi``); ``t`` is not stopped either, so a query time that
    itself carries a tangent (e.g. in an adjoint replay) flows through
    ``delt`` into ``cvals``/``dvals`` and both outputs.

    Args:
        t: The query time.
        tn: The current internal time (end of the last accepted step).
        phi: The post-step divided-difference history, shape ``(maxord + 1,
            n)``.
        psi: The post-step step-size history, shape ``(maxord + 1,)``.
        kused: The order used on the last accepted step (``int32``); ``0`` is
            treated as order 1.
        maxord: The maximum integration order (static; matches the ``phi``
            leading dimension minus one).

    Returns:
        The interpolated state and its derivative, each shape ``(n,)``.
    """
    cvals, dvals, _kord = _dense_coeffs(t, tn, psi, kused, maxord=maxord)
    yret = jnp.sum(cvals[:, None] * phi, axis=0)
    ypret = jnp.sum(dvals[:, None] * phi[1:], axis=0)
    return yret, ypret


__all__ = ["ida_get_solution"]
