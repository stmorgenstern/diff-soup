"""The step-retry ladders for the backward-differentiation-formula engine.

When a step attempt fails, one of two ladders decides whether the next attempt
retries -- and with what step size and order -- or whether the solve gives up.

* [bdf_handle_nflag][diff_soup.timestepping.vsvo.engines.bdf.failure.bdf_handle_nflag] is the convergence-failure ladder. An unrecoverable
  nonlinear-solver, linear-setup, or linear-solve failure maps straight to its
  own termination code. A recoverable failure aborts once the per-attempt
  convergence-failure count reaches its cap or the step is already at the
  minimum size; otherwise it shrinks the step and asks for a re-predict. The
  cap test is an equality on the incremented count and is checked *before* the
  step is reduced.

* [bdf_error_test_fail][diff_soup.timestepping.vsvo.engines.bdf.failure.bdf_error_test_fail] is the error-test-failure ladder. It walks three rungs keyed
  on the incremented error-test-failure count, tested strictly in order: up to
  the first three failures shrink the step at the current order with the biased
  error formula; past that, the order is dropped by one (or, already at order 1,
  the history is rebuilt from scratch with a long order-change wait).

Neither ladder touches the Nordsieck history: the step assembly owns the
restore that undoes a failed attempt and, on the order-1 restart, the
rebuild of the first history column from a fresh right-hand-side evaluation.
[bdf_error_test_fail][diff_soup.timestepping.vsvo.engines.bdf.failure.bdf_error_test_fail] returns the decision -- the step-size ratio, the new
order and wait counter, and the ``order_drop`` / ``restart_q1`` flags -- and the
step assembly applies it (an order drop through the order-decrease history
routine; the order-1 restart as a direct step/history update with no rescale).

Both ladders force the step-growth cap ``etamax`` to 1, which persists through
the retry attempts so the eventual successful attempt's controller defers its
order and step changes. Every output of both ladders is a gradient-dead
step-control decision and is stopped.

[kflag_to_result_code][diff_soup.timestepping.vsvo.engines.bdf.failure.kflag_to_result_code] collapses a termination code into the driver's
step-result space: a repeated error-test failure is a step-size collapse; every
other termination is a nonlinear-convergence failure; the retry sentinels are
read as non-terminal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .flags import NonlinearFlag, StepFlag
from .state import BDFState

SUN_NLS_CONV_RECVR = 902
"""The recoverable-convergence return code a nonlinear solver reports; an abort
triggered by it is a genuine convergence failure rather than a repeated
recoverable right-hand-side error."""

_ONE_PLUS_UROUND = 1.000001
"""The ``|h| <= hmin*ONEPSM`` "already at the minimum step" fuzz factor."""

_ETA_CF = 0.25
"""Step-size ratio after a recoverable convergence failure."""

_ETA_MIN_EF = 0.1
"""Floor of the step-size ratio after an error-test failure."""

_ETA_MAX_EF = 0.2
"""Cap of the step-size ratio once the error-test-failure count reaches
``SMALL_NEF``."""

_BIAS2 = 6.0
"""Bias in the error-test-failure step-ratio formula."""

_ADDON = 1e-6
"""Safety addend in the step-ratio formula."""

_SMALL_NEF = 2
"""Error-test-failure count at or above which the step ratio is capped."""

_LONG_WAIT = 10
"""Order-change wait imposed after an order-1 restart."""

_MXNCF = 10
"""Convergence failures per step attempt before aborting."""

_MXNEF = 7
"""Error-test failures per step attempt before aborting."""

_MXNEF1 = 3
"""Error-test failures before a forced order reduction."""

# Driver step-result codes (see diff_soup.timestepping.vsvo.common.driver):
# 1 success / non-terminal, 2 step-size collapse, 3 nonlinear divergence.
_CODE_SUCCESS = 1
_CODE_STEP_COLLAPSE = 2
_CODE_NONLINEAR_DIVERGENCE = 3


def bdf_handle_nflag(
    nflag: jax.Array,
    h: jax.Array,
    eta: jax.Array,
    ncf: jax.Array,
    hmin: jax.Array,
    dtype: type,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decide how -- or whether -- a step retries after a corrector failure.

    Args:
        nflag: The corrector's return flag. A success routes to the error test;
            a negative flag is unrecoverable; a positive flag
            (`SUN_NLS_CONV_RECVR` or
            [`RHSFUNC_RECVR`][diff_soup.timestepping.vsvo.engines.bdf.flags.NonlinearFlag.RHSFUNC_RECVR])
            is recoverable.
        h: The step size the failed attempt used.
        eta: The step-size ratio carried in (returned unchanged on every path
            but the recoverable retry).
        ncf: The convergence-failure count so far this step (reset on every
            accepted step).
        hmin: The minimum step size.
        dtype: The floating-point dtype.

    Returns:
        ``(kflag, eta, ncf, etamax, nflag_out)``: the termination or retry
        sentinel ([StepFlag][diff_soup.timestepping.vsvo.engines.bdf.flags.StepFlag]
        / [NonlinearFlag][diff_soup.timestepping.vsvo.engines.bdf.flags.NonlinearFlag]); the
        step-size ratio, updated failure count, and next corrector-reason flag
        the retry should use; and ``etamax``, forced to 1 on any failure. The
        ``etamax``, ``eta``, and ``nflag_out`` outputs are meaningful on the
        recoverable-retry path; on a success or an abort the caller reads only
        ``kflag``.
    """
    nflag = jnp.asarray(nflag, jnp.int32)
    h = jnp.asarray(h, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    hmin = jnp.asarray(hmin, dtype=dtype)
    ncf = jnp.asarray(ncf, jnp.int32)
    abs_h = jnp.abs(h)

    is_success = nflag == StepFlag.STEP_SUCCESS
    fatal = nflag < 0
    fatal_kflag = jnp.where(
        nflag == StepFlag.LINEAR_SETUP_FAIL,
        jnp.asarray(StepFlag.LINEAR_SETUP_FAIL, jnp.int32),
        jnp.where(
            nflag == StepFlag.LINEAR_SOLVE_FAIL,
            jnp.asarray(StepFlag.LINEAR_SOLVE_FAIL, jnp.int32),
            jnp.where(
                nflag == StepFlag.RHS_FAIL,
                jnp.asarray(StepFlag.RHS_FAIL, jnp.int32),
                jnp.asarray(StepFlag.NONLINEAR_FAIL, jnp.int32),
            ),
        ),
    )

    ncf_after = ncf + 1
    abort = (abs_h <= hmin * _ONE_PLUS_UROUND) | (ncf_after == _MXNCF)
    abort_kflag = jnp.where(
        nflag == SUN_NLS_CONV_RECVR,
        jnp.asarray(StepFlag.CONVERGENCE_FAIL, jnp.int32),
        jnp.asarray(StepFlag.REPEATED_RHS_ERR, jnp.int32),
    )
    eta_cf = jnp.maximum(jnp.asarray(_ETA_CF, dtype=dtype), hmin / abs_h)

    kflag = jnp.where(
        is_success,
        jnp.asarray(NonlinearFlag.DO_ERROR_TEST, jnp.int32),
        jnp.where(
            fatal,
            fatal_kflag,
            jnp.where(abort, abort_kflag, jnp.asarray(NonlinearFlag.PREDICT_AGAIN, jnp.int32)),
        ),
    ).astype(jnp.int32)

    eta_out = jnp.where(is_success | fatal | abort, eta, eta_cf)
    ncf_out = jnp.where(is_success | fatal, ncf, ncf_after).astype(jnp.int32)
    etamax_out = jnp.ones((), dtype=dtype)
    nflag_out = jnp.where(
        kflag == NonlinearFlag.PREDICT_AGAIN,
        jnp.asarray(NonlinearFlag.PREV_CONV_FAIL, jnp.int32),
        nflag,
    ).astype(jnp.int32)

    return jax.lax.stop_gradient((kflag, eta_out, ncf_out, etamax_out, nflag_out))


def bdf_error_test_fail(
    dsm: jax.Array,
    state: BDFState,
    nef: jax.Array,
    hmin: jax.Array,
    dtype: type,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Walk the error-test-failure ladder and decide the retry.

    The incremented failure count and the current order pick the rung, tested
    in the order abort > (count within the first three) > (order above 1) >
    (order 1 restart):

    * within the first three failures: shrink at the current order with
      ``eta = 1/((BIAS2*dsm)^(1/L) + ADDON)`` floored at 0.1, additionally
      capped at 0.2 once the count reaches ``SMALL_NEF``;
    * past that with order above 1: ``order_drop`` -- drop the order by one with
      a flat ``eta = max(0.1, hmin/|h|)`` and reset the wait counter to the old
      order;
    * past that at order 1: ``restart_q1`` -- flat ``eta``, order held, wait
      counter set to ``LONG_WAIT`` and the constant-step-run counter reset (the
      step assembly rebuilds the first history column from a fresh right-hand
      side, with no rescale).

    An abort (count equals ``MXNEF``, or the step is already at the minimum
    size) returns ``ERROR_TEST_FAIL``; the count still increments on the aborting
    attempt.

    Args:
        dsm: The scaled local error that failed the test.
        state: The carried step state (read for ``q``, ``L``, ``h``, ``qwait``,
            ``nscon``).
        nef: The error-test-failure count so far this step.
        hmin: The minimum step size.
        dtype: The floating-point dtype.

    Returns:
        ``(kflag, eta, q, L, qwait, nef, etamax, order_drop, restart_q1,
        nscon)``: the termination or retry sentinel; the step-size ratio; the
        new order, ``L = q+1``, and wait counter; the incremented failure count;
        ``etamax`` forced to 1; the two decision flags; and the constant-step-run
        counter. The ``eta`` and order/counter fields are meaningful on the
        retry (``TRY_AGAIN``) path.
    """
    q = state.q
    L = state.L
    h = jnp.asarray(state.h, dtype=dtype)
    qwait = state.qwait
    nscon = state.nscon
    dsm = jnp.asarray(dsm, dtype=dtype)
    hmin = jnp.asarray(hmin, dtype=dtype)
    nef = jnp.asarray(nef, jnp.int32)
    abs_h = jnp.abs(h)
    L_f = L.astype(dtype)

    nef_after = nef + 1
    abort = (abs_h <= hmin * _ONE_PLUS_UROUND) | (nef_after == _MXNEF)
    within_first = (~abort) & (nef_after <= _MXNEF1)
    past_first = (~abort) & (nef_after > _MXNEF1)
    order_drop = past_first & (q > 1)
    restart_q1 = past_first & (q == 1)

    eta_rescale = 1.0 / (jnp.power(_BIAS2 * dsm, 1.0 / L_f) + _ADDON)
    eta_rescale = jnp.maximum(
        jnp.asarray(_ETA_MIN_EF, dtype=dtype), jnp.maximum(eta_rescale, hmin / abs_h)
    )
    eta_rescale = jnp.where(
        nef_after >= _SMALL_NEF,
        jnp.minimum(eta_rescale, jnp.asarray(_ETA_MAX_EF, dtype=dtype)),
        eta_rescale,
    )
    eta_flat = jnp.maximum(jnp.asarray(_ETA_MIN_EF, dtype=dtype), hmin / abs_h)

    eta_out = jnp.where(within_first, eta_rescale, eta_flat)
    kflag = jnp.where(
        abort,
        jnp.asarray(StepFlag.ERROR_TEST_FAIL, jnp.int32),
        jnp.asarray(NonlinearFlag.TRY_AGAIN, jnp.int32),
    ).astype(jnp.int32)
    q_out = jnp.where(order_drop, q - 1, q).astype(jnp.int32)
    L_out = jnp.where(order_drop, q, L).astype(jnp.int32)
    qwait_out = jnp.where(
        order_drop, q, jnp.where(restart_q1, jnp.asarray(_LONG_WAIT, jnp.int32), qwait)
    ).astype(jnp.int32)
    nscon_out = jnp.where(restart_q1, jnp.zeros((), jnp.int32), nscon).astype(jnp.int32)
    etamax_out = jnp.ones((), dtype=dtype)

    return jax.lax.stop_gradient(
        (
            kflag,
            eta_out,
            q_out,
            L_out,
            qwait_out,
            nef_after.astype(jnp.int32),
            etamax_out,
            order_drop,
            restart_q1,
            nscon_out,
        )
    )


def kflag_to_result_code(kflag: jax.Array) -> jax.Array:
    """Collapse a termination ``kflag`` into the driver's step-result code.

    A repeated error-test failure (``ERROR_TEST_FAIL``) is a step-size collapse
    (2). Every other termination code -- a stalled or diverged corrector, a
    repeated recoverable right-hand-side error, or a fatal linear-setup /
    linear-solve / right-hand-side failure -- is a nonlinear-convergence failure
    (3). The retry sentinels (``PREDICT_AGAIN`` / ``TRY_AGAIN`` /
    ``DO_ERROR_TEST``) are non-terminal and map to the success code (1). The
    finer distinction between the failure families stays visible through the
    engine's convergence- and error-test-failure counters.

    Args:
        kflag: A termination or retry code from one of the ladders.

    Returns:
        The matching driver step-result code, ``int32``.
    """
    kflag = jnp.asarray(kflag, jnp.int32)
    is_retry = (
        (kflag == NonlinearFlag.PREDICT_AGAIN)
        | (kflag == NonlinearFlag.TRY_AGAIN)
        | (kflag == NonlinearFlag.DO_ERROR_TEST)
    )
    is_step_collapse = kflag == StepFlag.ERROR_TEST_FAIL
    return jnp.where(
        is_retry,
        jnp.asarray(_CODE_SUCCESS, jnp.int32),
        jnp.where(
            is_step_collapse,
            jnp.asarray(_CODE_STEP_COLLAPSE, jnp.int32),
            jnp.asarray(_CODE_NONLINEAR_DIVERGENCE, jnp.int32),
        ),
    ).astype(jnp.int32)


__all__ = [
    "SUN_NLS_CONV_RECVR",
    "bdf_handle_nflag",
    "bdf_error_test_fail",
    "kflag_to_result_code",
]
