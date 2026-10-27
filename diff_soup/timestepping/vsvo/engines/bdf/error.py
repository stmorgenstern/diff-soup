"""The local-error accept predicate for the backward-differentiation-formula engine.

[bdf_do_error_test_accept][diff_soup.timestepping.vsvo.engines.bdf.error.bdf_do_error_test_accept]
forms the scaled local-error estimate ``dsm``
and reports whether the step passes the test. This engine keeps the accept
test, the history restore, and the failure ladder in separate modules -- the
accept test here, the history restore in the step assembly, and the failure
ladder in [diff_soup.timestepping.vsvo.engines.bdf.failure][] -- so each is
a small, independently checkable piece.

The scaled error and the accept flag are step-control decisions and carry no
gradient with respect to the solution: the returned values are stopped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bdf_do_error_test_accept(
    acnrm: jax.Array, tq: jax.Array, dtype: type
) -> tuple[jax.Array, jax.Array]:
    """Form the scaled local error and decide whether the step is accepted.

    Computes ``dsm = acnrm * tq[2]`` -- the weighted-RMS norm of the accepted
    correction scaled by the order-``q`` local-error coefficient -- and accepts
    the step exactly when ``dsm <= 1``. A rejected step is routed by the step assembly into the
    error-test failure ladder in
    [bdf_error_test_fail][diff_soup.timestepping.vsvo.engines.bdf.failure.bdf_error_test_fail].

    Args:
        acnrm: The weighted-RMS norm of the accepted correction ``||acor||``.
        tq: The test-quantity vector, shape ``(6,)``; ``tq[2]`` is the
            order-``q`` local-error coefficient.
        dtype: The floating-point dtype for the scaled error.

    Returns:
        A pair ``(dsm, error_test_pass)``: the scaled local error and the
        boolean accept flag. Both are gradient-dead step-control quantities.
    """
    acnrm = jnp.asarray(acnrm, dtype=dtype)
    tq = jnp.asarray(tq, dtype=dtype)
    dsm = acnrm * tq[2]
    error_test_pass = dsm <= jnp.asarray(1.0, dtype=dtype)
    return jax.lax.stop_gradient((dsm, error_test_pass))


__all__ = ["bdf_do_error_test_accept"]
