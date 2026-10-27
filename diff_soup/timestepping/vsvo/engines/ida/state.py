"""The carried step state for the fully-implicit DAE engine.

[IDAState][diff_soup.timestepping.vsvo.engines.ida.state.IDAState] is the fixed-shape pytree that persists across accepted
steps: the divided-difference history, the corrector coefficient arrays, the
current solution/predictor/correction vectors, the order and step-size
state, the leading-coefficient triple, and the Newton rate memory. Every
downstream piece of the engine -- the predictor, the corrector, the error
test, and the accept/reject/complete-step machinery -- reads and writes this
record. [init_ida_state][diff_soup.timestepping.vsvo.engines.ida.state.init_ida_state] builds the record a solve starts from.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

MAXORD = 5
"""The maximum integration order this engine supports (BDF methods of
order above 5 are unstable, per Petzold 1982)."""

MXORDP1 = MAXORD + 1
"""Length of the ``phi`` divided-difference history array."""

_ORDER_VEC = MXORDP1
"""Length of the ``psi``/``alpha``/``beta``/``sigma``/``gamma`` coefficient
arrays. Equal to ``MXORDP1``, matching the ``phi`` history length."""

NEWTON_CONV_COEFF = 0.33
"""The Newton convergence-test coefficient used during stepping (the
reference algorithm's default ``epcon``)."""

DIRECT_TEST_TOL = 1e-4 * NEWTON_CONV_COEFF
"""The direct-test constant for a one-iteration Newton acceptance: ``1e-4``
times [NEWTON_CONV_COEFF][diff_soup.timestepping.vsvo.engines.ida.state.NEWTON_CONV_COEFF]."""


class IDAState(eqx.Module):
    """The carried step state threaded through the IDA-tracking engine.

    Every leaf is a fixed-shape JAX array, so the whole record moves through
    a compiled step loop without any data-dependent shape. The augmented
    state length is ``n = n_u + n_lam``: the differential/algebraic
    unknowns plus the bordered global rows. ``phi``, ``ee``, ``yy``, and
    ``yp`` all carry the *full* augmented vector. Algebraic globals contribute
    no time derivative to the residual; differential globals carry unit mass
    and participate in the history like differential state rows.

    Two per-step-attempt counters -- the convergence-failure and
    error-test-failure counts -- are deliberately absent from this record.
    They reset at the start of every accepted step and belong to the
    step-loop's own local carry, not to the state that persists across
    steps.
    """

    # --- divided-difference history ---
    phi: jax.Array
    """Scaled divided differences of ``y``, shape ``(MXORDP1, n)``.
    ``phi[0]`` is the current solution; ``phi[1:]`` are the higher-order
    differences the predictor and the corrector-update recurrence consume."""
    psi: jax.Array
    """Step-size history, shape ``(MXORDP1,)``. ``psi[i]`` sums the ``i + 1``
    most recent step sizes, including the step being attempted."""
    alpha: jax.Array
    """Current-step-to-history step ratios, shape ``(MXORDP1,)``.
    ``alpha[0] = 1``; ``alpha[i] = hh / psi[i]`` for ``i >= 1``."""
    beta: jax.Array
    """The ``phi`` -> ``phi*`` rescale factors, shape ``(MXORDP1,)``.
    ``beta[0] = 1``; each ``beta[i]`` is the ratio of the new to the old
    ``psi[i - 1]``."""
    sigma: jax.Array
    """Order-dependent scale factors the local-error estimate uses, shape
    ``(MXORDP1,)``. ``sigma[0] = 1``; ``sigma[i] = i * sigma[i - 1] *
    alpha[i]``."""
    gamma: jax.Array
    """Predictor-derivative coefficients, shape ``(MXORDP1,)``. ``gamma[0] =
    0``; ``gamma[i] = gamma[i - 1] + alpha[i - 1] / hh``."""

    # --- solution pair + predictors + correction ---
    yy: jax.Array
    """The current ``y`` iterate, shape ``(n,)``: ``yypredict + ee``."""
    yp: jax.Array
    """The current ``y'`` iterate, shape ``(n,)``: ``yppredict + cj * ee``."""
    yypredict: jax.Array
    """The predicted ``y`` at the top of the current step attempt, shape
    ``(n,)``."""
    yppredict: jax.Array
    """The predicted ``y'`` at the top of the current step attempt, shape
    ``(n,)``."""
    ee: jax.Array
    """The accumulated Newton correction, shape ``(n,)``. Once a step is
    accepted, ``ck * ee`` is the estimated local error for that step."""

    # --- order state ---
    kk: jax.Array
    """The integration order for the step being attempted (``int32``)."""
    kused: jax.Array
    """The order used on the last successful step (``int32``)."""
    knew: jax.Array
    """The order the error-test decrease rule proposes for the next attempt
    (``int32``)."""

    # --- step state ---
    hh: jax.Array
    """The signed step size for the current attempt."""
    hused: jax.Array
    """The step size used on the last successful step."""
    eta: jax.Array
    """The chosen ratio of the next step size to the last used step size."""

    # --- leading-coefficient state ---
    cj: jax.Array
    """The leading coefficient of the iteration matrix, ``-alphas / hh``."""
    cjlast: jax.Array
    """``cj`` from the previous step attempt."""
    cjold: jax.Array
    """``cj`` as of the most recent linear-solver setup call."""
    cjratio: jax.Array
    """``cj / cjold``, used to gate whether a fresh linear-solver setup is
    needed."""

    # --- Newton rate memory ---
    ss: jax.Array
    """Newton convergence-rate memory: 20 right after a fresh setup, 100
    when ``cj`` changed without a setup, otherwise the carried ``rate / (1 -
    rate)`` from the previous iteration."""
    oldnrm: jax.Array
    """The weighted-RMS norm of the first Newton correction update in the
    current solve, used to compute the convergence rate."""

    # --- startup / phase ---
    phase: jax.Array
    """``0`` during the startup phase (raise the order and double the step
    every step); ``1`` once normal local-error control has taken over
    (``int32``)."""
    ns: jax.Array
    """The number of consecutive steps taken at constant step size and
    order, capped at ``kused + 2`` (``int32``)."""

    # --- counters ---
    nst: jax.Array
    """The number of successful steps taken so far (``int32``)."""


def init_ida_state(
    y0_aug: jax.Array,
    ydot0_aug: jax.Array,
    h0: jax.Array | float,
    n: int,
    dtype: type = jnp.float64,
) -> IDAState:
    """Build the first-call carried state for an IDA-tracking solve.

    Seeds the divided-difference history from the (consistent) initial
    state and derivative, and initializes every coefficient array and
    scalar to the standard first-step values: order 1, the startup phase,
    a fresh Newton rate memory, and a leading coefficient ``cj = 1 / h0``.

    Args:
        y0_aug: The initial augmented state, shape ``(n,)``.
        ydot0_aug: The initial augmented state derivative, shape ``(n,)``.
        h0: The initial step size.
        n: The augmented-state length (``n_u + n_lam``).
        dtype: The floating-point dtype for every array leaf.

    Returns:
        The seeded
        [IDAState][diff_soup.timestepping.vsvo.engines.ida.state.IDAState].
    """
    y0_aug = jnp.asarray(y0_aug, dtype=dtype)
    ydot0_aug = jnp.asarray(ydot0_aug, dtype=dtype)
    h0 = jnp.asarray(h0, dtype=dtype)

    phi = jnp.zeros((MXORDP1, n), dtype=dtype)
    phi = phi.at[0].set(y0_aug)
    phi = phi.at[1].set(h0 * ydot0_aug)

    psi = jnp.zeros(_ORDER_VEC, dtype=dtype).at[0].set(h0)
    alpha = jnp.zeros(_ORDER_VEC, dtype=dtype).at[0].set(1.0)
    beta = jnp.zeros(_ORDER_VEC, dtype=dtype).at[0].set(1.0)
    sigma = jnp.zeros(_ORDER_VEC, dtype=dtype).at[0].set(1.0)
    gamma = jnp.zeros(_ORDER_VEC, dtype=dtype)

    cj = 1.0 / h0

    return IDAState(
        phi=phi,
        psi=psi,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        yy=y0_aug,
        yp=ydot0_aug,
        yypredict=y0_aug,
        yppredict=ydot0_aug,
        ee=jnp.zeros(n, dtype=dtype),
        kk=jnp.asarray(1, dtype=jnp.int32),
        kused=jnp.asarray(0, dtype=jnp.int32),
        knew=jnp.asarray(1, dtype=jnp.int32),
        hh=h0,
        hused=jnp.asarray(0.0, dtype=dtype),
        eta=jnp.asarray(1.0, dtype=dtype),
        cj=cj,
        cjlast=cj,
        cjold=cj,
        cjratio=jnp.asarray(1.0, dtype=dtype),
        ss=jnp.asarray(20.0, dtype=dtype),
        oldnrm=jnp.asarray(0.0, dtype=dtype),
        phase=jnp.asarray(0, dtype=jnp.int32),
        ns=jnp.asarray(0, dtype=jnp.int32),
        nst=jnp.asarray(0, dtype=jnp.int32),
    )


__all__ = [
    "IDAState",
    "init_ida_state",
    "MAXORD",
    "MXORDP1",
    "NEWTON_CONV_COEFF",
    "DIRECT_TEST_TOL",
]
