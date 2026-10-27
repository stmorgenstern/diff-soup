"""The carried step state for the backward-differentiation-formula engine.

[BDFState][diff_soup.timestepping.vsvo.engines.bdf.state.BDFState] is the
fixed-shape pytree that persists across accepted steps: the Nordsieck history,
the corrector coefficient arrays, the order and step-size state, the
leading-coefficient triple, and the corrector rate memory. Every downstream
piece of the engine -- the predictor, the corrector, the error test, and the
accept/reject/complete-step machinery -- reads and writes this record.
[init_bdf_state][diff_soup.timestepping.vsvo.engines.bdf.state.init_bdf_state]
builds the record a solve starts from.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

QMAX = 5
"""The maximum integration order this engine supports."""

L_ZN = QMAX + 2
"""Number of Nordsieck history columns: holds ``zn[0..q+1]`` for every
supported order plus one extra slot (index ``QMAX``) that doubles as the
saved-correction slot after a step completes."""

_ORDER_VEC = QMAX + 2
"""Length of the ``tau``/``l`` coefficient arrays. Sized to match the trace
record's order-indexed vectors; index 0 of ``tau``/``l`` and index
``q + 1`` and above are unused at a given order."""

_TQ_LEN = 6
"""Length of the ``tq`` test-quantity array: indices ``1..5`` are used
(``NUM_TESTS = 5``); index 0 is never written."""

ETA_MAX_FS = 10000.0
"""The step-size-ratio cap in force before any step has been taken."""

CORTES = 0.1
"""The default nonlinear-convergence-test coefficient."""


class BDFState(eqx.Module):
    """The carried step state threaded through the BDF-tracking engine.

    Every leaf is a fixed-shape JAX array, so the whole record moves through
    a compiled step loop without any data-dependent shape. The augmented
    state length is ``n = n_u + n_lam``: the state unknowns plus the full
    bordered global partition. ``zn`` and ``acor`` both carry the *full*
    augmented vector -- algebraic global rows contribute no time derivative
    (zero mass) while differential-global rows carry unit mass and genuine
    derivative history; both ride the history and the correction the same
    way state rows do.

    Several per-attempt or per-step-attempt quantities are deliberately
    absent from this record:

    - ``delp``, ``etaq``, ``etaqm1``, ``etaqp1`` are locals the corrector
      and the step-size controller compute and consume within a single
      attempt; they never need to survive past that attempt.
    - ``nef``/``ncf`` (the error-test/convergence-failure counts) reset at
      the start of every accepted step and belong to the step loop's own
      local carry, not to state that persists across steps.
    - ``jcur``, ``nje``, ``nsetups``, ``nni``, ``nnf`` are corrector/setup
      outputs the corrector returns and the step loop threads into
      diagnostics and the engine's own carry, not fields of this state.
    - The saved-Jacobian blocks and the assembled/factored iteration matrix
      also live on the engine's own carry, not on this state.
    """

    # --- Nordsieck history + correction ---
    zn: jax.Array
    """The Nordsieck history, shape ``(L_ZN, n)``. ``zn[j] = (h**j / j!) *
    y^(j)`` of the interpolating polynomial for ``j`` in ``0..q+1``; column
    ``QMAX`` doubles as the slot that holds the saved correction after a
    step completes."""
    acor: jax.Array
    """The corrector correction ``y_n(m) - y_n(0)``, shape ``(n,)``. After a
    step is accepted, ``tq[2] * acor`` is that step's estimated local
    error."""
    tau: jax.Array
    """The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
    ``tau[1]`` is the most recent step size."""
    l: jax.Array  # noqa: E741 -- the BDF coefficient array's reference name.
    """Coefficients of the degree-``q`` polynomial the corrector-update
    recurrence uses, shape ``(_ORDER_VEC,)``."""
    tq: jax.Array
    """Test quantities, shape ``(_TQ_LEN,)``. Indices ``1..5`` are used:
    the local-error coefficients at order ``q - 1``, ``q``, and ``q + 1``,
    the nonlinear-convergence-test constant, and the coefficient used to
    build the order-``q + 2`` derivative estimate. Index 0 is unused."""

    # --- order state ---
    q: jax.Array
    """The current integration order (``int32``)."""
    qprime: jax.Array
    """The order chosen for the next step attempt (``int32``)."""
    qwait: jax.Array
    """The number of steps left to wait before an order change is
    considered again (``int32``)."""
    L: jax.Array
    """``q + 1`` (``int32``)."""

    # --- step state ---
    h: jax.Array
    """The step size for the current attempt."""
    hprime: jax.Array
    """The step size chosen for the next attempt."""
    eta: jax.Array
    """``hprime / h``, the chosen step-size ratio."""
    hscale: jax.Array
    """The step size the Nordsieck history ``zn`` is currently scaled
    with."""
    hu: jax.Array
    """The step size used on the last successful step."""
    qu: jax.Array
    """The order used on the last successful step (``int32``)."""

    # --- leading-coefficient state (the pencil scalars) ---
    rl1: jax.Array
    """``1 / l[1]``."""
    gamma: jax.Array
    """``h * rl1``: the pencil coefficient this engine calls ``hinvGak``
    (the iteration matrix is ``M - hinvGak * J``; the materialized leading
    coefficient is ``cj = 1 / hinvGak``, never the inverted relation)."""
    gammap: jax.Array
    """``gamma`` as of the most recent linear-solver setup call."""
    gamrat: jax.Array
    """``gamma / gammap`` (``1`` while ``nst == 0``): the ratio that gates
    whether a fresh linear-solver setup is needed."""

    # --- nonlinear / convergence state ---
    crate: jax.Array
    """The estimated corrector convergence rate (reset to ``1`` whenever a
    linear-solver setup runs)."""
    acnrm: jax.Array
    """``||acor||`` in the weighted-RMS norm."""

    # --- controller state ---
    etamax: jax.Array
    """The upper cap on ``eta`` for this step: ``10000`` before any step has
    been taken, ``10`` for early/general steps, ``1`` right after a
    convergence or error-test failure."""
    saved_tq5: jax.Array
    """``tq[5]`` saved at the step where ``qwait`` reached ``1``, consumed
    one step later by the order-``q + 1`` step-ratio candidate."""

    # --- reuse / step counters ---
    nst: jax.Array
    """The number of successful internal steps taken so far (``int32``)."""
    nstlp: jax.Array
    """The step number of the most recent linear-solver setup call
    (``int32``)."""
    nstlj: jax.Array
    """The step number of the most recent Jacobian update (``int32``)."""
    nscon: jax.Array
    """The number of successful steps taken at constant step size and
    order (``int32``); carried for the stability-limit detector, which is
    not wired by this engine yet."""
    indx_acor: jax.Array
    """The ``zn`` column index that holds the saved correction, always
    ``QMAX`` (``int32``)."""


def init_bdf_state(
    y0_aug: jax.Array,
    ydot0_aug: jax.Array,
    h0: jax.Array | float,
    n: int,
    dtype: type = jnp.float64,
) -> BDFState:
    """Build the first-call carried state for a BDF-tracking solve.

    Seeds the Nordsieck history from the (consistent) initial state and
    derivative, and initializes every coefficient array and scalar to the
    BDF corrector's first-step values: order 1, a fresh corrector rate
    memory, and a leading-coefficient triple consistent with what the
    coefficient recurrence would produce on the very first call (``l[1] =
    1``, so ``rl1 = 1`` and ``gamma = h0``).

    Args:
        y0_aug: The initial augmented state, shape ``(n,)``.
        ydot0_aug: The initial augmented state derivative, shape ``(n,)``.
        h0: The initial step size.
        n: The augmented-state length (``n_u + n_lam``).
        dtype: The floating-point dtype for every array leaf.

    Returns:
        The seeded
        [BDFState][diff_soup.timestepping.vsvo.engines.bdf.state.BDFState].
    """
    y0_aug = jnp.asarray(y0_aug, dtype=dtype)
    ydot0_aug = jnp.asarray(ydot0_aug, dtype=dtype)
    h0 = jnp.asarray(h0, dtype=dtype)

    zn = jnp.zeros((L_ZN, n), dtype=dtype)
    zn = zn.at[0].set(y0_aug)
    zn = zn.at[1].set(h0 * ydot0_aug)

    tau = jnp.zeros(_ORDER_VEC, dtype=dtype)
    l = jnp.zeros(_ORDER_VEC, dtype=dtype)  # noqa: E741
    tq = jnp.zeros(_TQ_LEN, dtype=dtype)

    rl1 = jnp.asarray(1.0, dtype=dtype)
    gamma = h0

    return BDFState(
        zn=zn,
        acor=jnp.zeros(n, dtype=dtype),
        tau=tau,
        l=l,
        tq=tq,
        q=jnp.asarray(1, dtype=jnp.int32),
        qprime=jnp.asarray(1, dtype=jnp.int32),
        qwait=jnp.asarray(2, dtype=jnp.int32),
        L=jnp.asarray(2, dtype=jnp.int32),
        h=h0,
        hprime=h0,
        eta=jnp.asarray(1.0, dtype=dtype),
        hscale=h0,
        hu=jnp.asarray(0.0, dtype=dtype),
        qu=jnp.asarray(0, dtype=jnp.int32),
        rl1=rl1,
        gamma=gamma,
        gammap=gamma,
        gamrat=jnp.asarray(1.0, dtype=dtype),
        crate=jnp.asarray(1.0, dtype=dtype),
        acnrm=jnp.asarray(0.0, dtype=dtype),
        etamax=jnp.asarray(ETA_MAX_FS, dtype=dtype),
        saved_tq5=jnp.asarray(0.0, dtype=dtype),
        nst=jnp.asarray(0, dtype=jnp.int32),
        nstlp=jnp.asarray(0, dtype=jnp.int32),
        nstlj=jnp.asarray(0, dtype=jnp.int32),
        nscon=jnp.asarray(0, dtype=jnp.int32),
        indx_acor=jnp.asarray(QMAX, dtype=jnp.int32),
    )


__all__ = [
    "BDFState",
    "init_bdf_state",
    "QMAX",
    "L_ZN",
    "ETA_MAX_FS",
    "CORTES",
]
