"""The carried step state for the NDF engine.

[NDFState][diff_soup.timestepping.vsvo.engines.ndf.state.NDFState] is the
fixed-shape pytree that persists across accepted
steps: the accepted-point backward-difference table and its grid spacing,
the order and step-size state, the controller counters, the Newton
contraction memory, and the two-tier linear-solver gate memory.
[init_ndf_state][diff_soup.timestepping.vsvo.engines.ndf.state.
init_ndf_state] builds the record a solve starts from.

The difference table is stored once, at the last accepted point on that
step's own grid (``dt_basis``); every attempt derives its working table by
rescaling this basis to the attempt's step. A rejected attempt therefore
never has to restore anything — the basis was never touched — which also
makes repeated rejections at different trial steps exact rather than
compounded through incremental rescales.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .tables import D_ROWS, seed_d


class NDFState(eqx.Module):
    """The carried step state threaded through the NDF engine.

    Every leaf is a fixed-shape JAX array; the augmented state length is
    ``n = n_u + n_lam`` — the full bordered global partition. Algebraic
    global rows ride along with zero mass; differential-global rows carry
    unit mass and genuine derivative history.

    Attributes:
        d_basis: The difference table at the last accepted point, shape
            ``(D_ROWS, n)``: row 0 the state, row ``j`` its ``j``-th
            backward difference on the ``dt_basis`` grid.
        dt_basis: The (signed) grid spacing ``d_basis`` lives on.
        k: The order for the next attempt (``int32``).
        k_used: The order of the last accepted step (``int32``) — the
            dense-output rows of ``d_basis`` that are current.
        dt: The (signed) step size for the next attempt.
        nconsteps: Consecutive accepted steps at the current step size
            (``int32``; the accept controller's constant-step counter).
        consfailcnt: Consecutive error-test rejections of the current step
            (``int32``; reset by an accepted step).
        theta_mem: The Newton contraction estimate carried across attempts
            (reset to 1 after any rejected attempt).
        nst: Accepted internal steps so far (``int32``).
        nstlp: The step number of the most recent factorization (``int32``).
        nstlj: The step number of the most recent Jacobian evaluation
            (``int32``).
        gdt_prev: The leading coefficient ``dt*beta0(k)`` at the most recent
            factorization (the setup gate's drift reference).
    """

    d_basis: jax.Array
    dt_basis: jax.Array
    k: jax.Array
    k_used: jax.Array
    dt: jax.Array
    nconsteps: jax.Array
    consfailcnt: jax.Array
    theta_mem: jax.Array
    nst: jax.Array
    nstlp: jax.Array
    nstlj: jax.Array
    gdt_prev: jax.Array


def init_ndf_state(
    y0_aug: jax.Array,
    ydot0_aug: jax.Array,
    h0: jax.Array,
    dtype: type = jnp.float64,
) -> NDFState:
    """Build the first-call carried state: order 1, a two-row seeded table.

    Args:
        y0_aug: The (consistent) initial augmented state, shape ``(n,)``.
        ydot0_aug: The initial augmented state derivative, shape ``(n,)``.
        h0: The (signed) initial step size.
        dtype: The floating-point dtype for every array leaf.

    Returns:
        The seeded
        [NDFState][diff_soup.timestepping.vsvo.engines.ndf.state.NDFState].
    """
    y0_aug = jnp.asarray(y0_aug, dtype=dtype)
    ydot0_aug = jnp.asarray(ydot0_aug, dtype=dtype)
    h0 = jnp.asarray(h0, dtype=dtype)
    return NDFState(
        d_basis=seed_d(y0_aug, ydot0_aug, h0),
        dt_basis=h0,
        k=jnp.asarray(1, jnp.int32),
        k_used=jnp.asarray(1, jnp.int32),
        dt=h0,
        nconsteps=jnp.asarray(0, jnp.int32),
        consfailcnt=jnp.asarray(0, jnp.int32),
        theta_mem=jnp.asarray(1.0, dtype),
        nst=jnp.asarray(0, jnp.int32),
        nstlp=jnp.asarray(0, jnp.int32),
        nstlj=jnp.asarray(0, jnp.int32),
        gdt_prev=h0,
    )


__all__ = ["NDFState", "init_ndf_state", "D_ROWS"]
