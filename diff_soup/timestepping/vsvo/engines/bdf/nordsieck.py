"""Nordsieck-array manipulation for the backward-differentiation-formula engine.

The Nordsieck history ``zn`` packs the interpolating polynomial's scaled
derivatives, ``zn[j] = (h**j / j!) * y^(j)``, for orders ``j = 0 .. q + 1``.
Every routine here is a pure transform of that array (plus the small
scalars that describe the step-size history): the predictor's repeated
Pascal additions ([bdf_predict_history][]), their exact inverse
([bdf_restore_history][]), the step-size rescale ([bdf_rescale_history][]), the two
order-change correction polynomials ([bdf_increase_order][],
[bdf_decrease_order][]) and their composition ([bdf_adjust_order][]),
and the full per-attempt composition that applies a pending order change
and then always rescales ([bdf_adjust_params][]). These are the history
transforms the step assembly sequences around the nonlinear corrector; none
of them advances the current time or clamps against a stop time -- that is
the caller's job.

``q`` is a traced integration order, but ``zn`` has a fixed ``(L_ZN, n)``
shape, so every loop over ``1 .. q`` (or a sub-range) is a Python-unrolled
loop over the full static range with a per-iteration active mask threaded
through ``jnp.where``. The Pascal double loop for the predictor and its
inverse is order-sensitive within a single outer pass: each inner-loop write
at column ``j - 1`` is read back as column ``j`` by the very next
inner-loop iteration. Realizing both loops as plain, sequential
``zn.at[...].set(...)`` calls (rather than a vectorized or reordered form)
preserves that dependency automatically, because each iteration reads the
array as the prior iteration left it -- the same discipline the
accepted-step history advance uses (``engines/ida/complete_step.py``).

Every output here is gradient-dead in the coefficient/order/step controls
(``eta``, ``tau``, ``hscale``, ``q``, ``qprime``, ``deltaq``, ``indx_acor``
are stopped at the point of use), while ``zn`` carries the state's gradient
straight through: every transform here is linear in ``zn``, so it is never
wrapped in ``stop_gradient``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import _ORDER_VEC, QMAX, BDFState


def bdf_predict_history(zn: jax.Array, q: jax.Array) -> jax.Array:
    """Advance the Nordsieck array by repeated Pascal additions (``zn`` only).

    Advances the Nordsieck array by repeated Pascal additions: for
    ``k = 1 .. q``, for ``j = q`` down to ``k``, ``zn[j - 1] += zn[j]``. The
    current-time advance and the stop-time clamp are not part of this
    routine -- it is the pure ``zn`` transform the step assembly wraps with
    those.

    Args:
        zn: The Nordsieck history, shape ``(L_ZN, n)``.
        q: The order for this attempt (``int32``).

    Returns:
        The predicted ``zn``, carrying the same gradient ``zn`` carries in
        (this transform is linear and exact).
    """
    q = jax.lax.stop_gradient(jnp.asarray(q))
    for k in range(1, QMAX + 1):
        for j in range(QMAX, 0, -1):
            active = (k <= q) & (k <= j) & (j <= q)
            zn = zn.at[j - 1].set(jnp.where(active, zn[j - 1] + zn[j], zn[j - 1]))
    return zn


def bdf_restore_history(zn: jax.Array, q: jax.Array) -> jax.Array:
    """Invert [bdf_predict_history][] exactly (Pascal subtractions).

    Inverts [bdf_predict_history][] exactly: the same nested loop, with
    subtraction in place of addition. The current-time restore is not part
    of this routine.

    Args:
        zn: The Nordsieck history left by [bdf_predict_history][], shape
            ``(L_ZN, n)``.
        q: The order the matching [bdf_predict_history][] call used (``int32``).

    Returns:
        The restored ``zn``. ``bdf_restore_history(bdf_predict_history(zn, q), q)``
        reproduces the input ``zn`` exactly.
    """
    q = jax.lax.stop_gradient(jnp.asarray(q))
    for k in range(1, QMAX + 1):
        for j in range(QMAX, 0, -1):
            active = (k <= q) & (k <= j) & (j <= q)
            zn = zn.at[j - 1].set(jnp.where(active, zn[j - 1] - zn[j], zn[j - 1]))
    return zn


def bdf_rescale_history(
    zn: jax.Array,
    h: jax.Array,
    hscale: jax.Array,
    eta: jax.Array,
    q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Rescale the Nordsieck array by ``eta**j`` and rebase ``h``/``hscale``.

    Rescales the Nordsieck array by ``eta**j`` and rebases ``h``/``hscale``:
    builds the running powers ``eta**1 .. eta**q`` by repeated multiplication
    (``cval = eta * cval`` at each order) rather than a general float power,
    since these scale factors are exact products of the same ``eta``, scales
    ``zn[j]`` by ``eta**j`` for ``j = 1 .. q`` (``zn[0]`` is untouched), and
    rebases the step size: ``h = hscale * eta``, ``hscale = h``. The
    ``next_h`` reporting copy is left to the caller, since it is not
    consumed by any later computation this engine performs.

    Args:
        zn: The Nordsieck history, shape ``(L_ZN, n)``.
        h: The step size before this rescale.
        hscale: The step size ``zn`` is currently scaled with.
        eta: The step-size ratio to rescale by.
        q: The order to rescale over (``int32``).

    Returns:
        A 4-tuple ``(zn, h_new, hscale_new, nscon)``: the rescaled history,
        the rebased step size and its ``hscale`` copy, and the reset
        constant-step-and-order counter (always ``0``).
    """
    h = jax.lax.stop_gradient(jnp.asarray(h))
    hscale = jax.lax.stop_gradient(jnp.asarray(hscale))
    eta = jax.lax.stop_gradient(jnp.asarray(eta))
    q = jax.lax.stop_gradient(jnp.asarray(q))

    dtype = zn.dtype
    eta = eta.astype(dtype)
    cval = jnp.asarray(1.0, dtype)
    for j in range(1, QMAX + 1):
        active = j <= q
        cval = jnp.where(active, cval * eta, cval)
        zn = zn.at[j].set(jnp.where(active, zn[j] * cval, zn[j]))

    h_new = hscale.astype(dtype) * eta
    hscale_new = h_new
    nscon = jnp.asarray(0, dtype=jnp.int32)
    return zn, h_new, hscale_new, nscon


def bdf_increase_order(
    zn: jax.Array,
    tau: jax.Array,
    hscale: jax.Array,
    q: jax.Array,
    indx_acor: jax.Array,
) -> jax.Array:
    """Apply the order-raise (``q`` -> ``q + 1``) Nordsieck correction.

    Builds the coefficients of ``x * x * (x + xi_1) * ... * (x + xi_{q-1})``
    into a scratch ``l`` array (interior loop ``j = 1 .. q - 1``, walking
    ``tau[j + 1]`` and using the prior iteration's ``xi`` -- ``xiold`` --
    for the descending ``l`` update), sets the new top column
    ``zn[q + 1] = A1 * zn[indx_acor]`` (the saved correction from the step
    that just completed), then corrects columns ``2 .. q`` by
    ``zn[j] += l[j] * zn[q + 1]``. At ``q == 1`` the interior loop never
    runs, so ``alpha0 = -1``, ``alpha1 = 1``, ``prod = 1`` keep their seed
    values and ``A1 = 0``: the new top column is zero and no lower column is
    touched, exactly matching the ``q > 1`` guard structure.

    Args:
        zn: The Nordsieck history, shape ``(L_ZN, n)``.
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        hscale: The step size ``zn`` is currently scaled with.
        q: The order being raised from (``int32``).
        indx_acor: The ``zn`` column holding the saved correction (``int32``).

    Returns:
        The order-raised ``zn``, carrying ``zn``'s own gradient (the
        correction is linear in ``zn``; every scalar multiplier is built
        from gradient-dead control inputs).
    """
    tau = jax.lax.stop_gradient(jnp.asarray(tau))
    hscale = jax.lax.stop_gradient(jnp.asarray(hscale))
    q = jax.lax.stop_gradient(jnp.asarray(q))
    indx_acor = jax.lax.stop_gradient(jnp.asarray(indx_acor))

    dtype = zn.dtype
    one = jnp.asarray(1.0, dtype)
    zero = jnp.asarray(0.0, dtype)

    l = jnp.zeros(_ORDER_VEC, dtype=dtype)  # noqa: E741
    l = l.at[2].set(one)  # noqa: E741
    alpha1 = one
    prod = one
    xiold = one
    alpha0 = -one
    hsum = hscale.astype(dtype)

    q_gt_1 = q > 1
    for j in range(1, QMAX):  # j = 1 .. QMAX - 1 (covers q - 1 for q up to QMAX)
        active_j = q_gt_1 & (j <= q - 1)
        hsum = hsum + jnp.where(active_j, tau[j + 1], zero)
        xi = jnp.where(active_j, hsum / hscale.astype(dtype), xiold)
        prod = jnp.where(active_j, prod * xi, prod)
        alpha0 = alpha0 - jnp.where(active_j, one / (j + 1), zero)
        alpha1 = jnp.where(active_j, alpha1 + one / xi, alpha1)
        for i in range(j + 2, 1, -1):
            l = l.at[i].set(jnp.where(active_j, l[i] * xiold + l[i - 1], l[i]))  # noqa: E741
        xiold = jnp.where(active_j, xi, xiold)

    a1 = (-alpha0 - alpha1) / prod
    L = q + 1
    zn = zn.at[L].set(a1 * zn[indx_acor])
    zn_l = zn[L]
    for j in range(2, QMAX + 1):
        active = j <= q
        zn = zn.at[j].set(jnp.where(active, zn[j] + l[j] * zn_l, zn[j]))
    return zn


def bdf_decrease_order(
    zn: jax.Array,
    tau: jax.Array,
    hscale: jax.Array,
    q: jax.Array,
) -> jax.Array:
    """Apply the order-lower (``q`` -> ``q - 1``) Nordsieck correction.

    Builds the coefficients of ``x * x * (x + xi_1) * ... * (x + xi_j)``
    into a scratch ``l`` array (interior loop ``j = 1 .. q - 2``, walking
    ``tau[j]``), then corrects columns ``2 .. q - 1`` by
    ``zn[j] += (-l[j]) * zn[q]``. Column ``zn[q]`` itself is never written
    -- it is simply excluded from the range the caller predicts/corrects
    once the order has dropped. At ``q == 2`` the interior loop is empty
    (only the seed ``l[2] = 1`` stands) and the correction range
    ``2 .. q - 1`` is also empty, so this is a no-op on ``zn``.

    Args:
        zn: The Nordsieck history, shape ``(L_ZN, n)``.
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        hscale: The step size ``zn`` is currently scaled with.
        q: The order being lowered from (``int32``).

    Returns:
        The order-lowered ``zn``, carrying ``zn``'s own gradient.
    """
    tau = jax.lax.stop_gradient(jnp.asarray(tau))
    hscale = jax.lax.stop_gradient(jnp.asarray(hscale))
    q = jax.lax.stop_gradient(jnp.asarray(q))

    dtype = zn.dtype
    one = jnp.asarray(1.0, dtype)
    zero = jnp.asarray(0.0, dtype)

    l = jnp.zeros(_ORDER_VEC, dtype=dtype)  # noqa: E741
    l = l.at[2].set(one)  # noqa: E741
    hsum = zero

    for j in range(1, QMAX - 1):  # j = 1 .. QMAX - 2 (covers q - 2 for q up to QMAX)
        active_j = j <= q - 2
        hsum = hsum + jnp.where(active_j, tau[j], zero)
        xi = hsum / hscale.astype(dtype)
        for i in range(j + 2, 1, -1):
            l = l.at[i].set(jnp.where(active_j, l[i] * xi + l[i - 1], l[i]))  # noqa: E741

    zn_q = zn[q]
    for j in range(2, QMAX):  # j = 2 .. QMAX - 1 (covers q - 1 for q up to QMAX)
        active = j <= q - 1
        zn = zn.at[j].set(jnp.where(active, zn[j] - l[j] * zn_q, zn[j]))
    return zn


def bdf_adjust_order(
    zn: jax.Array,
    tau: jax.Array,
    hscale: jax.Array,
    q: jax.Array,
    deltaq: jax.Array,
    indx_acor: jax.Array,
) -> jax.Array:
    """Dispatch the order-change correction by sign of ``deltaq``.

    ``deltaq == +1`` runs [bdf_increase_order][], ``deltaq == -1`` runs
    [bdf_decrease_order][]. At ``q == 2`` a decrease (``deltaq != 1``)
    returns ``zn`` unchanged -- BDF's order-2-to-1 decrease needs no history
    correction, only the order/``qwait`` bookkeeping the caller applies
    afterward.

    Args:
        zn: The Nordsieck history, shape ``(L_ZN, n)``.
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        hscale: The step size ``zn`` is currently scaled with.
        q: The order being changed from (``int32``).
        deltaq: The order change, ``+1`` or ``-1`` (``int32``).
        indx_acor: The ``zn`` column holding the saved correction (``int32``).

    Returns:
        The order-changed ``zn``, carrying ``zn``'s own gradient.
    """
    q = jax.lax.stop_gradient(jnp.asarray(q))
    deltaq = jax.lax.stop_gradient(jnp.asarray(deltaq))

    skip = (q == 2) & (deltaq != 1)
    zn_up = bdf_increase_order(zn, tau, hscale, q, indx_acor)
    zn_down = bdf_decrease_order(zn, tau, hscale, q)
    zn_changed = jnp.where(deltaq == 1, zn_up, zn_down)
    return jnp.where(skip, zn, zn_changed)


def bdf_adjust_params(
    state: BDFState,
    first_step_after_resize: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply a pending order change, then always rescale.

    When ``qprime != q``, applies [bdf_adjust_order][] for
    ``qprime - q`` (skipped when ``first_step_after_resize`` is set, since
    that case's history adjustment was already applied elsewhere) and
    updates ``q``, ``L = q + 1``, ``qwait = L`` to the new order; then always
    calls [bdf_rescale_history][] -- including when the order did not change,
    since a step-size-only change (``eta != 1``) is what triggers this
    routine in the first place.

    This function reads ``BDFState`` but never constructs one: it returns
    the individual updated arrays/scalars for the caller to write back,
    matching the read/write split the accepted-step history advance uses
    (``engines/ida/complete_step.py``).

    Args:
        state: The carried step state. Reads ``zn``, ``tau``, ``hscale``,
            ``q``, ``qprime``, ``L``, ``qwait``, ``h``, ``eta``,
            ``indx_acor``.
        first_step_after_resize: When set, skips the order-change history
            correction (its effect was already applied by a resize this
            engine does not implement) while still moving ``q``/``L``/
            ``qwait`` to the pending order.

    Returns:
        A 7-tuple ``(zn, q, L, qwait, h, hscale, nscon)``: the fully
        adjusted-and-rescaled history and the updated order/step-size
        bookkeeping.
    """
    zn = state.zn
    tau = jax.lax.stop_gradient(state.tau)
    hscale = jax.lax.stop_gradient(state.hscale)
    q = jax.lax.stop_gradient(state.q)
    qprime = jax.lax.stop_gradient(state.qprime)
    l_state = jax.lax.stop_gradient(state.L)
    qwait_state = jax.lax.stop_gradient(state.qwait)
    h = jax.lax.stop_gradient(state.h)
    eta = jax.lax.stop_gradient(state.eta)
    indx_acor = jax.lax.stop_gradient(state.indx_acor)

    order_changes = qprime != q
    deltaq = qprime - q

    if first_step_after_resize:
        zn_candidate = zn
    else:
        zn_candidate = bdf_adjust_order(zn, tau, hscale, q, deltaq, indx_acor)
    zn_after_order = jnp.where(order_changes, zn_candidate, zn)

    l_candidate = qprime + 1
    q_new = jnp.where(order_changes, qprime, q).astype(jnp.int32)
    l_new = jnp.where(order_changes, l_candidate, l_state).astype(jnp.int32)
    qwait_new = jnp.where(order_changes, l_candidate, qwait_state).astype(jnp.int32)

    zn_final, h_new, hscale_new, nscon = bdf_rescale_history(zn_after_order, h, hscale, eta, q_new)

    return zn_final, q_new, l_new, qwait_new, h_new, hscale_new, nscon


__all__ = [
    "bdf_predict_history",
    "bdf_restore_history",
    "bdf_rescale_history",
    "bdf_increase_order",
    "bdf_decrease_order",
    "bdf_adjust_order",
    "bdf_adjust_params",
]
