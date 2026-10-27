"""The corrector coefficient recurrence for the backward-differentiation formula engine.

[bdf_set_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_coeffs] builds the BDF corrector coefficients: given the step-size
history ``tau`` and the current order and step size, it builds the polynomial coefficients ``l`` ([bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs]) and the
test-quantity vector ``tq`` ([bdf_set_test_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_test_coeffs]), then derives the
leading-coefficient triple ``rl1``, ``gamma``, ``gammap``, ``gamrat``.

This engine's pencil coefficient (what the rest of this codebase calls
``hinvGak``) is *not* the fixed constant-step harmonic-sum table this
package's ``coefficients`` module tabulates for the quasi-constant-step
family -- it is this recurrence's live ``gamma = h * rl1 = h / l[1]``,
derived from the actual (possibly non-uniform) recent step-size history. At
a constant step size the two agree (``l[1]`` reduces to the harmonic
partial sum), but in general they do not, so this module never reads that
fixed table. The leading coefficient ``cj`` is always materialized as
``cj_from_hinvgak(gamma)`` (``diff_soup.timestepping.vsvo.coefficients``),
never by inverting a ``cj * gamma == 1`` relation.

Every output here is a step-control quantity -- a coefficient, a test
constant, or a leading-coefficient scalar -- and carries no gradient with
respect to the solution or any model parameter: every argument is stopped
before use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import CORTES, QMAX


def bdf_set_order_coeffs(
    tau: jax.Array,
    l: jax.Array,  # noqa: E741 -- the BDF coefficient array's reference name.
    h: jax.Array,
    q: jax.Array,
    proj_disabled: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the BDF polynomial coefficients ``l`` and their supporting scalars.

    Builds the coefficients
    of ``Lambda(x) = (1 + x/xi*_q) * prod_{i=1}^{q-1} (1 + x/xi_i)``, where
    ``xi_i = [t_n - t_(n-i)] / h``, by repeated in-place multiplication of
    the binomial factors into ``l``. ``q`` is a traced order, so the ``for
    j = 2..q-1`` and ``for j = q`` source loops are realized as fixed
    Python loops over the full ``2..QMAX`` range with a per-``j`` active
    mask: the running sums (``hsum``, ``alpha0``) accumulate a masked-zero
    term on inactive iterations (so they freeze once ``j`` exceeds the
    active range), and ``l``'s array writes fall back to their unchanged
    value when inactive.

    Projection is not implemented -- the projection-only polynomial
    construction required only when projection is enabled is not built here.

    Args:
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        l: The incoming coefficient array, shape ``(_ORDER_VEC,)`` -- used
            only for its shape and dtype, and as the base every untouched
            entry (index ``> q``) is carried through from, exactly matching
            an in-place update scheme where untouched entries retain their
            prior value.
        h: The step size for this attempt.
        q: The order for this attempt (``int32``).
        proj_disabled: Must be ``True`` -- projection is not implemented.

    Returns:
        A 6-tuple ``(l, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv)``: the
        updated coefficient array and the scalars [bdf_set_test_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_test_coeffs]
        needs to build the test-quantity vector.
    """
    if not proj_disabled:
        raise NotImplementedError(
            "bdf_set_order_coeffs does not build the projection-only coefficient "
            "polynomial; call with proj_disabled=True."
        )

    tau = jax.lax.stop_gradient(jnp.asarray(tau))
    l = jax.lax.stop_gradient(jnp.asarray(l))  # noqa: E741
    h = jax.lax.stop_gradient(jnp.asarray(h))
    q = jax.lax.stop_gradient(jnp.asarray(q))

    dtype = l.dtype
    one = jnp.asarray(1.0, dtype)
    zero = jnp.asarray(0.0, dtype)

    l = l.at[0].set(one)  # noqa: E741
    l = l.at[1].set(one)  # noqa: E741
    for i in range(2, QMAX + 1):
        l = l.at[i].set(jnp.where(i <= q, zero, l[i]))  # noqa: E741

    xi_inv = one
    xistar_inv = one
    alpha0 = -one
    alpha0_hat = -one
    hsum = h

    q_gt_1 = q > 1

    # Interior recurrence j = 2 .. q - 1.
    for j in range(2, QMAX):
        active_j = q_gt_1 & (j <= q - 1)
        hsum = hsum + jnp.where(active_j, tau[j - 1], zero)
        xi_inv = jnp.where(active_j, h / hsum, xi_inv)
        alpha0 = alpha0 - jnp.where(active_j, one / j, zero)
        for i in range(j, 0, -1):
            l = l.at[i].set(jnp.where(active_j, l[i] + l[i - 1] * xi_inv, l[i]))  # noqa: E741

    # j = q term, active only when q > 1.
    alpha0 = alpha0 - jnp.where(q_gt_1, one / q.astype(dtype), zero)
    xistar_inv = jnp.where(q_gt_1, -l[1] - alpha0, xistar_inv)
    hsum = hsum + jnp.where(q_gt_1, tau[q - 1], zero)
    xi_inv = jnp.where(q_gt_1, h / hsum, xi_inv)
    alpha0_hat = jnp.where(q_gt_1, -l[1] - xi_inv, alpha0_hat)
    for i in range(QMAX, 0, -1):
        active_i = q_gt_1 & (i <= q)
        l = l.at[i].set(jnp.where(active_i, l[i] + l[i - 1] * xistar_inv, l[i]))  # noqa: E741

    return l, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv


def bdf_set_test_coeffs(
    l: jax.Array,  # noqa: E741 -- the BDF coefficient array's reference name.
    tau: jax.Array,
    h: jax.Array,
    q: jax.Array,
    qwait: jax.Array,
    hsum: jax.Array,
    alpha0: jax.Array,
    alpha0_hat: jax.Array,
    xi_inv: jax.Array,
    xistar_inv: jax.Array,
    tq: jax.Array,
    nlscoef: float = CORTES,
) -> jax.Array:
    """Build the test-quantity vector ``tq`` from [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs]'s output.

    Builds the test-quantity vector from the BDF coefficient recurrence's
    output. ``tq[2]`` (the order-``q``
    local-error coefficient) and ``tq[5]`` (the order-``q+2`` derivative
    coefficient) are recomputed on every call. ``tq[1]`` (the order-``q-1``
    local-error coefficient) and ``tq[3]`` (the order-``q+1`` local-error
    coefficient) are refreshed only when ``qwait == 1`` -- otherwise they
    keep the value the incoming ``tq`` already carries, since
    the order-change step-ratio candidates consume exactly the values
    computed at the step where
    ``qwait`` reached ``1``, one step later. ``q`` and ``qwait`` are traced,
    so both the ``q > 1``/``q == 1`` branches and the ``qwait == 1`` refresh
    are realized by computing every candidate and selecting with
    ``jnp.where``.

    Args:
        l: The coefficient array [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned, shape
            ``(_ORDER_VEC,)``.
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        h: The step size for this attempt.
        q: The order for this attempt (``int32``).
        qwait: The number of steps left to wait before an order change is
            considered again (``int32``).
        hsum: The accumulated step-size sum [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned.
        alpha0: The accumulated coefficient [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned.
        alpha0_hat: The accumulated coefficient [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned.
        xi_inv: The reciprocal history spacing [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned.
        xistar_inv: The order-condition reciprocal spacing
            [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] returned.
        tq: The incoming test-quantity vector, shape ``(_TQ_LEN,)`` --
            ``tq[1]`` and ``tq[3]`` are read back from here when
            ``qwait != 1``.
        nlscoef: The nonlinear-convergence-test coefficient (default
            ``CORTES``).

    Returns:
        The updated ``tq``, shape ``(_TQ_LEN,)``.
    """
    l = jax.lax.stop_gradient(jnp.asarray(l))  # noqa: E741
    tau = jax.lax.stop_gradient(jnp.asarray(tau))
    h = jax.lax.stop_gradient(jnp.asarray(h))
    q = jax.lax.stop_gradient(jnp.asarray(q))
    qwait = jax.lax.stop_gradient(jnp.asarray(qwait))
    hsum = jax.lax.stop_gradient(jnp.asarray(hsum))
    alpha0 = jax.lax.stop_gradient(jnp.asarray(alpha0))
    alpha0_hat = jax.lax.stop_gradient(jnp.asarray(alpha0_hat))
    xi_inv = jax.lax.stop_gradient(jnp.asarray(xi_inv))
    xistar_inv = jax.lax.stop_gradient(jnp.asarray(xistar_inv))
    tq = jax.lax.stop_gradient(jnp.asarray(tq))

    dtype = l.dtype
    one = jnp.asarray(1.0, dtype)
    q_f = q.astype(dtype)
    l_q = l[q]

    a1 = one - alpha0_hat + alpha0
    a2 = one + q_f * a1
    tq2 = jnp.abs(a1 / (alpha0 * a2))
    tq5 = jnp.abs(a2 * xistar_inv / (l_q * xi_inv))

    q_gt_1 = q > 1
    c = xistar_inv / l_q
    a3 = alpha0 + one / q_f
    a4 = alpha0_hat + xi_inv
    cpinv = (one - a4 + a3) / a3
    tq1_candidate = jnp.where(q_gt_1, jnp.abs(c * cpinv), one)

    hsum_qp1 = hsum + tau[q]
    xi_inv_qp1 = h / hsum_qp1
    a5 = alpha0 - one / (q_f + one)
    a6 = alpha0_hat - xi_inv_qp1
    cppinv = (one - a6 + a5) / a2
    tq3_candidate = jnp.abs(cppinv / (xi_inv_qp1 * (q_f + 2.0) * a5))

    qwait_eq_1 = qwait == 1
    tq1 = jnp.where(qwait_eq_1, tq1_candidate, tq[1])
    tq3 = jnp.where(qwait_eq_1, tq3_candidate, tq[3])
    tq4 = jnp.asarray(nlscoef, dtype) / tq2

    tq_out = tq
    tq_out = tq_out.at[1].set(tq1)
    tq_out = tq_out.at[2].set(tq2)
    tq_out = tq_out.at[3].set(tq3)
    tq_out = tq_out.at[4].set(tq4)
    tq_out = tq_out.at[5].set(tq5)

    return tq_out


def bdf_set_coeffs(
    l: jax.Array,  # noqa: E741 -- the BDF coefficient array's reference name.
    tau: jax.Array,
    h: jax.Array,
    q: jax.Array,
    qwait: jax.Array,
    nst: jax.Array,
    gammap: jax.Array,
    tq: jax.Array,
    proj_disabled: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Advance the corrector coefficients by one step attempt.

    Composes [bdf_set_order_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_order_coeffs] and
    [bdf_set_test_coeffs][diff_soup.timestepping.vsvo.engines.bdf.coeffs.bdf_set_test_coeffs], then derives the leading-coefficient triple.
    ``rl1 = 1 / l[1]`` and ``gamma = h * rl1`` are this engine's pencil
    coefficient (``hinvGak``); on the very first step (``nst == 0``),
    ``gammap`` is seeded to that same ``gamma`` and ``gamrat`` is fixed at
    ``1`` to avoid a ``0 / 0``. Do not materialize the leading coefficient
    ``cj`` here -- return ``gamma`` and let the caller call
    ``cj_from_hinvgak(gamma)`` (``diff_soup.timestepping.vsvo.coefficients``).

    Args:
        l: The incoming coefficient array, shape ``(_ORDER_VEC,)``.
        tau: The previous ``q + 1`` step sizes, shape ``(_ORDER_VEC,)``.
        h: The step size for this attempt.
        q: The order for this attempt (``int32``).
        qwait: The number of steps left to wait before an order change is
            considered again (``int32``).
        nst: The number of successful internal steps taken so far
            (``int32``).
        gammap: ``gamma`` as of the most recent linear-solver setup call.
        tq: The incoming test-quantity vector, shape ``(_TQ_LEN,)``.
        proj_disabled: Must be ``True`` -- projection is not implemented.

    Returns:
        A 6-tuple ``(l, tq, rl1, gamma, gammap, gamrat)``.
    """
    l, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv = bdf_set_order_coeffs(  # noqa: E741
        tau, l, h, q, proj_disabled=proj_disabled
    )
    tq_out = bdf_set_test_coeffs(
        l, tau, h, q, qwait, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv, tq
    )

    h = jax.lax.stop_gradient(jnp.asarray(h))
    nst = jax.lax.stop_gradient(jnp.asarray(nst))
    gammap = jax.lax.stop_gradient(jnp.asarray(gammap))

    rl1 = 1.0 / l[1]
    gamma = h * rl1
    gammap_out = jnp.where(nst == 0, gamma, gammap)
    gamrat = jnp.where(nst > 0, gamma / gammap_out, jnp.asarray(1.0, dtype=gamma.dtype))

    return l, tq_out, rl1, gamma, gammap_out, gamrat


__all__ = ["bdf_set_order_coeffs", "bdf_set_test_coeffs", "bdf_set_coeffs"]
