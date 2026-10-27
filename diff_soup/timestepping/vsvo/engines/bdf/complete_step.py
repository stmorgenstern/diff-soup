"""The post-accept history update for the backward-differentiation-formula engine.

[diff_soup.timestepping.vsvo.engines.bdf.complete_step.bdf_complete_step][] commits the correction into the Nordsieck history and
advances the step-size/order bookkeeping: once a step attempt has passed the
local-error test, it applies the following updates. In order:

1. bump the successful-step and constant-step-run counters, record the step
   size and order just used;
2. shift the recent step-size history ``tau`` down one slot (with the order-1
   special that keeps a two-deep history);
3. add ``l[j] * acor`` into every active Nordsieck column ``zn[j]``;
4. decrement the order-change wait counter;
5. when that counter reaches 1 (and the order is below the maximum), save the
   correction into the spare Nordsieck column so a later order increase can use
   it.

The next-step order and step size are NOT decided here -- that is the
controller
([bdf_prepare_next_step][diff_soup.timestepping.vsvo.engines.bdf.controller.bdf_prepare_next_step]).

The Nordsieck update ``zn[j] += l[j]*acor`` is the accepted trajectory advance:
it stays differentiable so the correction ``acor`` carries the solution's
parameter sensitivity into the history. Every other field written here -- the
counters, the recorded step size and order, ``tau``, ``qwait``, and the saved
correction bookkeeping -- is a step-control quantity and is stopped.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .state import QMAX, BDFState


def bdf_complete_step(state: BDFState, acor: jax.Array, dtype: type) -> BDFState:
    """Commit an accepted step into the carried history and bookkeeping.

    Args:
        state: The carried step state, read for the pre-accept history and
            written with the post-accept update.
        acor: The accepted correction ``y_n - y_n(0)``, shape ``(n,)``. Its
            gradient flows through the ``zn`` update.
        dtype: The floating-point dtype of the state's array leaves.

    Returns:
        The updated [BDFState][diff_soup.timestepping.vsvo.engines.bdf.state.BDFState]
        with the counters bumped, ``tau`` shifted, ``zn`` corrected, ``qwait``
        decremented, and -- when due -- the correction saved for a possible
        order increase.
    """
    q = state.q
    h = jnp.asarray(state.h, dtype=dtype)
    acor = jnp.asarray(acor, dtype=dtype)

    # --- step and constant-step-run counters, last-step size and order ---
    nst_new = state.nst + 1
    nscon_new = state.nscon + 1
    hu_new = h
    qu_new = q

    # --- tau down-shift: tau[i] = tau[i-1] for i = q down to 2 ---
    # The descending index order matters: each write reads the still-original
    # tau[i-1]. Indices above q keep their prior value (masked out).
    tau = state.tau
    for i in range(QMAX, 1, -1):
        tau = tau.at[i].set(jnp.where(i <= q, tau[i - 1], tau[i]))
    # Order-1 special: keep a two-deep history once past the first step.
    # Reads the old tau[1] before it is overwritten.
    tau = tau.at[2].set(jnp.where((q == 1) & (nst_new > 1), tau[1], tau[2]))
    tau = tau.at[1].set(h)

    # --- zn[j] += l[j]*acor for j = 0..q ---
    l = state.l  # noqa: E741 -- the BDF coefficient array's reference name.
    zn = state.zn
    for j in range(0, QMAX + 1):
        zn = zn.at[j].set(jnp.where(j <= q, zn[j] + l[j] * acor, zn[j]))

    # --- qwait decrement and the qwait == 1 save of the correction ---
    qwait_new = state.qwait - 1
    save = (qwait_new == 1) & (q != QMAX)
    zn = zn.at[QMAX].set(jnp.where(save, acor, zn[QMAX]))
    saved_tq5_new = jnp.where(save, state.tq[5], state.saved_tq5)
    indx_acor_new = jnp.where(save, jnp.asarray(QMAX, jnp.int32), state.indx_acor)

    return eqx.tree_at(
        lambda s: (
            s.nst,
            s.nscon,
            s.hu,
            s.qu,
            s.tau,
            s.zn,
            s.qwait,
            s.saved_tq5,
            s.indx_acor,
        ),
        state,
        (
            jax.lax.stop_gradient(nst_new),
            jax.lax.stop_gradient(nscon_new),
            jax.lax.stop_gradient(hu_new),
            jax.lax.stop_gradient(qu_new),
            jax.lax.stop_gradient(tau),
            zn,  # the accepted history advance keeps its gradient through acor
            jax.lax.stop_gradient(qwait_new),
            jax.lax.stop_gradient(saved_tq5_new),
            jax.lax.stop_gradient(indx_acor_new),
        ),
    )


__all__ = ["bdf_complete_step"]
