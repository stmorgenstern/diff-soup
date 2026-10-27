"""The IDA step-retry decision: convergence and error-test failure ladders.

The step loop reaches this module only after a step attempt has already
failed -- either the corrector did not converge (or diverged, or a linear
solve/setup call failed) or the local-error test rejected the attempt.
[ida_handle_nflag][diff_soup.timestepping.vsvo.engines.ida.failure.ida_handle_nflag]
is the single entry point for both cases: it
decides whether the next attempt should retry with a smaller step (and, for
the error-test ladder, a possibly different order), or whether the solve
should give up.

* A nonlinear-solver failure bumps the local convergence-failure counter.
  A nonrecoverable failure (a fatal residual, linear-setup, or linear-solve
  error) maps straight through to its own termination code. A recoverable
  failure aborts once the counter reaches its cap or the step is already at
  the minimum size; otherwise it shrinks the step and asks for a retry.
* An error-test failure bumps the local error-test-failure counter and
  walks a three-rung ladder keyed on that counter's value, tested in order:
  the first failure keeps or lowers the order with a difference-based step
  cut; the second flattens to a fixed cut at the same order choice; the
  third and every later one forces order 1 with the same fixed cut. The
  ladder only aborts once the counter reaches its cap -- the ``==1``,
  ``==2``, ``< cap`` tests are evaluated strictly in that order, so a
  reject count of 1 or 2 always retries no matter how the cap is
  configured.

[ida_reset][diff_soup.timestepping.vsvo.engines.ida.failure.ida_reset] performs
the small first-step-only rescale the step loop
applies after [ida_restore][diff_soup.timestepping.vsvo.engines.ida.restore.ida_restore]
when a retry is requested while no step has yet been accepted.

Every output of [ida_handle_nflag][diff_soup.timestepping.vsvo.engines.ida.failure.ida_handle_nflag]
is a step-control decision -- a flag,
an order, a step ratio, or a failure counter -- and carries no gradient, so
that function stop-gradients its whole return value.
[ida_reset][diff_soup.timestepping.vsvo.engines.ida.failure.ida_reset]
rescales the pre-attempt ``(phi, psi)`` history instead, which the retry
re-predicts from and, once accepted, carries into the trajectory; it keeps
that history's tangents and stops only the step-control scale factors it
subtracts and divides by (``hh``, ``eta``) at their point of use, the same way
[restore][diff_soup.timestepping.vsvo.engines.ida.restore] does.

Driver termination-code mapping
--------------------------------
The abort branches below return one of the termination codes listed on
[NonlinearFlag][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag],
which form this module's own return-value contract. The
driver's step loop, which owns the solve-wide result code, collapses that
family into two outcomes: a local-error-test abort (too many rejected
steps in a row) is a step-size collapse; every nonlinear-solver or
residual-failure abort (a stalled Newton corrector, a repeated recoverable
residual failure, or a fatal linear-solve/setup/residual failure) is a
nonlinear-convergence failure.
[kflag_to_result_code][diff_soup.timestepping.vsvo.engines.ida.failure.kflag_to_result_code]
performs that
collapse; the finer distinction between the two failure families stays
visible through the separate convergence- and error-test-failure counters
this module returns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "NonlinearFlag",
    "ida_handle_nflag",
    "ida_reset",
    "kflag_to_result_code",
]


class NonlinearFlag:
    """Sentinel integers for the flag this module reads and the ``kflag`` it returns.

    The *input* ``nflag`` is either
    [ERROR_TEST_FAIL][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.ERROR_TEST_FAIL]
    or one of the
    corrector's recoverable (positive) or nonrecoverable (negative)
    failure codes. The *returned* ``kflag`` is either
    [PREDICT_AGAIN][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.PREDICT_AGAIN]
    (retry) or one of the termination codes below.
    """

    # --- input flag: which failure fired ---
    ERROR_TEST_FAIL = 7
    RESIDUAL_RECOVERABLE = 1
    LINEAR_SETUP_RECOVERABLE = 2
    LINEAR_SOLVE_RECOVERABLE = 3
    NONLINEAR_SETUP_RECOVERABLE = 4

    # --- returned kflag: retry sentinel ---
    PREDICT_AGAIN = 20

    # --- returned kflag: termination codes. LINEAR_SOLVE_FAIL, LINEAR_SETUP_FAIL,
    # and RESIDUAL_FAIL also double as nonrecoverable INPUT flags: the fatal
    # branch passes them straight through unchanged. ---
    ERROR_TEST_FAIL = -3
    CONVERGENCE_FAIL = -4
    LINEAR_SETUP_FAIL = -6
    LINEAR_SOLVE_FAIL = -7
    RESIDUAL_FAIL = -8
    REPEATED_RESIDUAL_ERR = -9
    NONLINEAR_FAIL = -17


_ETA_LOW = 0.9
"""Upper bound on the step-size ratio after the first error-test failure."""

_ETA_MIN_EF = 0.25
"""Floor of the step-size ratio after an error-test failure."""

_ETA_CF = 0.25
"""Step-size ratio after a recoverable nonlinear-solver failure."""

_ONE_PLUS_UROUND = 1.000001
"""The ``|hh| <= hmin*ONEPSM`` "already at the minimum step" test factor."""

_FIRST_FAIL_SAFETY = 0.9
"""The safety factor on the first-error-test-failure step-ratio formula."""

_ERROR_RATIO_BIAS = 1e-4
"""The bias term in the first-error-test-failure step-ratio formula."""


def ida_handle_nflag(
    nflag: jax.Array,
    err_k: jax.Array,
    err_km1: jax.Array,
    kk: jax.Array,
    knew: jax.Array,
    hh: jax.Array,
    eta: jax.Array,
    ncf: jax.Array,
    nef: jax.Array,
    hmin: jax.Array,
    maxncf: int = 10,
    maxnef: int = 10,
    eta_ef_override: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decide how -- or whether -- the failed step attempt retries.

    Dispatches on ``nflag``: anything other than
    [NonlinearFlag.ERROR_TEST_FAIL][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.ERROR_TEST_FAIL]
    names a nonlinear-solver failure and walks the convergence-failure
    branch;
    [NonlinearFlag.ERROR_TEST_FAIL][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.ERROR_TEST_FAIL]
    walks the ordered error-test
    ladder. Both branches are evaluated on every call and blended with
    ``jnp.where`` so ``nflag``/``ncf``/``nef`` stay traced integers with no
    data-dependent control flow.

    Args:
        nflag: The incoming failure flag --
            [NonlinearFlag.ERROR_TEST_FAIL][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.ERROR_TEST_FAIL]
            for a local-error rejection, else a nonlinear-solver flag
            (negative for a nonrecoverable failure, otherwise
            recoverable).
        err_k: The current-order sigma-scaled error estimate from the
            error test.
        err_km1: The order-``kk-1`` sigma-scaled error estimate.
        kk: The order the failed attempt used.
        knew: The order the error test proposed for the next attempt.
        hh: The step size the failed attempt used.
        eta: The step-size ratio carried in from the caller (returned
            unchanged on every abort path).
        ncf: The convergence-failure count so far this step (local to the
            step loop's retry sequence -- reset on every accepted step).
        nef: The error-test-failure count so far this step (same scope).
        hmin: The minimum step size.
        maxncf: The convergence-failure count that aborts the step.
        maxnef: The error-test-failure count that aborts the step.
        eta_ef_override: Optional replacement for the first error-test
            failure's step-size ratio. ``None`` (the default) uses the
            reference shrink formula. A controller that owns the step-size
            law passes its own rejection ratio here; the ``hmin`` floor
            still applies, and the later rungs (the flat cut, the forced
            order-1 fall-back) and every abort are unchanged.

    Returns:
        A 7-tuple ``(kflag, kk_new, eta_new, hh_new, ncf_new, nef_new,
        phase_new)``: the retry sentinel or a termination code (see
        [NonlinearFlag][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag]);
        the order, step-size ratio, and step size the
        next attempt should use (each left unchanged from the input on an
        abort); the updated failure counters; and ``phase_new`` (always
        ``1`` -- any failure forces normal local-error control on the next
        attempt).
    """
    dtype = jnp.asarray(hh).dtype
    hh = jnp.asarray(hh, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    hmin = jnp.asarray(hmin, dtype=dtype)
    err_k = jnp.asarray(err_k, dtype=dtype)
    err_km1 = jnp.asarray(err_km1, dtype=dtype)
    kk = jnp.asarray(kk)
    knew = jnp.asarray(knew)
    nflag = jnp.asarray(nflag)
    abs_hh = jnp.abs(hh)

    is_error_test_fail = nflag == NonlinearFlag.ERROR_TEST_FAIL

    # ---- nonlinear-solver (or residual) failure branch ----
    ncf_after = ncf + 1
    fatal = nflag < 0
    fatal_kflag = jnp.where(
        nflag == NonlinearFlag.LINEAR_SOLVE_FAIL,
        jnp.asarray(NonlinearFlag.LINEAR_SOLVE_FAIL, jnp.int32),
        jnp.where(
            nflag == NonlinearFlag.LINEAR_SETUP_FAIL,
            jnp.asarray(NonlinearFlag.LINEAR_SETUP_FAIL, jnp.int32),
            jnp.where(
                nflag == NonlinearFlag.RESIDUAL_FAIL,
                jnp.asarray(NonlinearFlag.RESIDUAL_FAIL, jnp.int32),
                jnp.asarray(NonlinearFlag.NONLINEAR_FAIL, jnp.int32),
            ),
        ),
    )
    recoverable_abort = jnp.logical_or(ncf_after == maxncf, abs_hh <= hmin * _ONE_PLUS_UROUND)
    recoverable_abort_kflag = jnp.where(
        nflag == NonlinearFlag.RESIDUAL_RECOVERABLE,
        jnp.asarray(NonlinearFlag.REPEATED_RESIDUAL_ERR, jnp.int32),
        jnp.asarray(NonlinearFlag.CONVERGENCE_FAIL, jnp.int32),
    )
    eta_cf = jnp.maximum(jnp.asarray(_ETA_CF, dtype=dtype), hmin / abs_hh)
    hh_cf = hh * eta_cf

    nls_abort = jnp.logical_or(fatal, recoverable_abort)
    kflag_nls = jnp.where(
        fatal,
        fatal_kflag,
        jnp.where(
            recoverable_abort,
            recoverable_abort_kflag,
            jnp.asarray(NonlinearFlag.PREDICT_AGAIN, jnp.int32),
        ),
    )
    eta_nls = jnp.where(nls_abort, eta, eta_cf)
    hh_nls = jnp.where(nls_abort, hh, hh_cf)
    kk_nls = kk  # order never changes on a nonlinear-solver failure

    # ---- error-test failure ladder: ==1, ==2, <maxnef, else -- in that order ----
    nef_after = nef + 1
    is_rung1 = nef_after == 1
    is_rung2 = nef_after == 2
    is_rung3 = nef_after < maxnef

    err_knew = jnp.where(kk == knew, err_k, err_km1)
    kk_rung1 = knew  # kk is set to knew BEFORE the exponent below reads it
    if eta_ef_override is None:
        tmp1 = (2.0 * err_knew + _ERROR_RATIO_BIAS) ** (-1.0 / (kk_rung1.astype(dtype) + 1.0))
        eta_rung1 = jnp.clip(_FIRST_FAIL_SAFETY * tmp1, _ETA_MIN_EF, _ETA_LOW)
    else:
        eta_rung1 = jnp.asarray(eta_ef_override, dtype=dtype)
    eta_rung1 = jnp.maximum(eta_rung1, hmin / abs_hh)

    eta_flat = jnp.maximum(jnp.asarray(_ETA_MIN_EF, dtype=dtype), hmin / abs_hh)

    kk_ladder = jnp.where(
        is_rung1, kk_rung1, jnp.where(is_rung2, knew, jnp.asarray(1, dtype=kk.dtype))
    )
    eta_ladder = jnp.where(is_rung1, eta_rung1, eta_flat)
    hh_ladder = hh * eta_ladder

    ef_retry = jnp.logical_or(is_rung1, jnp.logical_or(is_rung2, is_rung3))
    kflag_ef = jnp.where(
        ef_retry,
        jnp.asarray(NonlinearFlag.PREDICT_AGAIN, jnp.int32),
        jnp.asarray(NonlinearFlag.ERROR_TEST_FAIL, jnp.int32),
    )
    kk_ef = jnp.where(ef_retry, kk_ladder, kk)
    eta_ef = jnp.where(ef_retry, eta_ladder, eta)
    hh_ef = jnp.where(ef_retry, hh_ladder, hh)

    # ---- select the active branch ----
    kflag = jnp.where(is_error_test_fail, kflag_ef, kflag_nls).astype(jnp.int32)
    kk_new = jnp.where(is_error_test_fail, kk_ef, kk_nls).astype(kk.dtype)
    eta_new = jnp.where(is_error_test_fail, eta_ef, eta_nls)
    hh_new = jnp.where(is_error_test_fail, hh_ef, hh_nls)
    ncf_new = jnp.where(is_error_test_fail, ncf, ncf_after)
    nef_new = jnp.where(is_error_test_fail, nef_after, nef)
    phase_new = jnp.asarray(1, dtype=jnp.int32)

    return jax.lax.stop_gradient((kflag, kk_new, eta_new, hh_new, ncf_new, nef_new, phase_new))


def ida_reset(
    phi: jax.Array, psi: jax.Array, hh: jax.Array, eta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rescale ``phi[1]`` and ``psi[0]`` before re-predicting the very first step.

    Called only when a retry is requested while no step has yet been
    accepted (``nst == 0``), after
    [ida_restore][diff_soup.timestepping.vsvo.engines.ida.restore.ida_restore]
    has already undone the failed attempt's coefficient mutations. ``hh``
    is the already-shrunk step size --
    [ida_handle_nflag][diff_soup.timestepping.vsvo.engines.ida.failure.ida_handle_nflag]
    has
    already multiplied it by ``eta`` -- so this only keeps ``psi[0]`` and
    ``phi[1] = hh * y0'`` consistent with that new step size before the
    predictor runs again.

    Args:
        phi: The divided-difference history, shape ``(MXORDP1, n)``.
        psi: The step-size history, shape ``(MXORDP1,)``.
        hh: The already-shrunk step size for the retry.
        eta: The step-size ratio that produced ``hh``.

    Returns:
        The ``(phi, psi)`` pair with ``phi[1]`` rescaled by ``eta`` and
        ``psi[0]`` set to ``hh``. ``phi`` keeps its tangent through the
        rescale -- ``phi[1] = hh * y0'`` carries the initial derivative's
        parameter sensitivity into the retry, and the retry, once accepted,
        continues the trajectory -- while the step-control scale factors
        ``hh`` and ``eta`` are stopped at their point of use.
    """
    hh = jax.lax.stop_gradient(hh)
    eta = jax.lax.stop_gradient(eta)
    psi_new = psi.at[0].set(hh)
    phi_new = phi.at[1].set(eta * phi[1])
    return phi_new, psi_new


def kflag_to_result_code(kflag: jax.Array) -> jax.Array:
    """Map a termination ``kflag`` to the driver's step-result code space.

    The driver's per-attempt result code (see
    [diff_soup.timestepping.vsvo.common.driver][]) uses the space ``0``
    unknown, ``1`` success, ``2`` step collapse, ``3`` nonlinear
    divergence, ``4`` max steps, ``5`` event stop, ``6`` inconsistent IC.
    A repeated local-error-test failure
    ([NonlinearFlag.ERROR_TEST_FAIL][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.ERROR_TEST_FAIL])
    is a
    step-size collapse. Every other termination code this module can
    return -- a stalled Newton corrector, a repeated recoverable residual
    failure, or a fatal linear-solve/setup/residual failure -- is a
    nonlinear-convergence failure.

    Args:
        kflag: A termination code returned by
            [ida_handle_nflag][diff_soup.timestepping.vsvo.engines.ida.failure.ida_handle_nflag]
            (not the
            [NonlinearFlag.PREDICT_AGAIN][diff_soup.timestepping.vsvo.engines.ida.failure.NonlinearFlag.PREDICT_AGAIN]
            retry sentinel -- the
            step loop retries on that flag instead of reporting a result).

    Returns:
        The matching driver result code, ``int32``.
    """
    kflag = jnp.asarray(kflag)
    step_collapse = kflag == NonlinearFlag.ERROR_TEST_FAIL
    return jnp.where(
        step_collapse, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)
    )
