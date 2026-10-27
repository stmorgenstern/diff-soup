"""Fixed-shape per-step trace records for the scalar VSVO solver suite.

A trace records, per internal step *attempt* (accepted or rejected), the
decision variables a parity harness compares against a reference integrator.
Allocation happens only when a solve is asked to trace, so the default solve
path pays nothing; when a trace is requested the buffers are statically sized to
the step budget so the compiled solve loop never allocates data-dependently.

Layout
------
* [StepTraceHeader][diff_soup.timestepping.vsvo.common.trace.StepTraceHeader]
  — the engine-independent fields every solver records each attempt (step
  index, time, step size, order, accept flag, status code,
  corrector-iteration count, error estimate, step-size ratio).
* [NDFTraceRecord][diff_soup.timestepping.vsvo.common.trace.NDFTraceRecord],
  [BDFTraceRecord][diff_soup.timestepping.vsvo.common.trace.BDFTraceRecord],
  [IDATraceRecord][diff_soup.timestepping.vsvo.common.trace.IDATraceRecord]
  — one record family per engine. Each nests the shared header and adds
  that engine's decision variables. The field lists mirror the
  numerical-differentiation-formula, backward-differentiation-formula,
  and implicit-DAE reference algorithms; the engines populate them as
  they are wired.
* [alloc_trace][diff_soup.timestepping.vsvo.common.trace.alloc_trace] /
  [write_trace][diff_soup.timestepping.vsvo.common.trace.write_trace] /
  [finalize_trace][diff_soup.timestepping.vsvo.common.trace.finalize_trace]
  — allocate a step-indexed buffer, write one attempt into a slot, and
  slice the buffer down to the number of attempts recorded.
* [select_trace][diff_soup.timestepping.vsvo.common.trace.select_trace]
  — map ``"ndf"`` / ``"bdf"`` / ``"ida"`` to the record type.

Order-indexed coefficient vectors are statically sized to the maximum
integration order (5) with a small headroom for the ``+1`` index ranges the
recurrences use, so every buffer leaf has a compile-time-fixed shape. Diagnostic
quantities that are produced once per corrector iteration are recorded as the
terminal-iteration value here; an engine may widen them to per-iteration vectors
when it wires its corrector, since only then is the iteration cap fixed.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal._loop.common import _is_buffer, buffer_at_set

# The suite caps the integration order at 5. Order-indexed coefficient vectors
# (psi[0..k], gamma[1..k], tau[1..q+1], l[0..q], the dif column norms, dense
# output coefficients) are sized to hold every index the recurrences touch at
# the maximum order, including the ``+1`` slot the ladders reach into.
_MAX_ORDER = 5
_ORDER_VEC = _MAX_ORDER + 2


def _f(dtype) -> jax.Array:
    return jnp.asarray(0.0, dtype)


def _i() -> jax.Array:
    return jnp.asarray(0, jnp.int32)


def _b() -> jax.Array:
    return jnp.asarray(False)


def _v(dtype) -> jax.Array:
    return jnp.zeros(_ORDER_VEC, dtype)


class StepTraceHeader(eqx.Module):
    """Engine-independent per-attempt fields common to every solver.

    Attributes:
        nst: Attempt index (0-based).
        t: Time after the attempt.
        h: Step size the attempt used.
        k: Integration order the attempt used.
        accepted: Whether the attempt was accepted.
        result_code: Termination code for the attempt (the code space
            [result_from_code][diff_soup._solution.result_from_code]
            consumes).
        newton_iters: Corrector iterations the attempt took.
        err_estimate: The accept/reject error estimate.
        eta: Step-size ratio chosen for the next attempt.
    """

    nst: jax.Array
    t: jax.Array
    h: jax.Array
    k: jax.Array
    accepted: jax.Array
    result_code: jax.Array
    newton_iters: jax.Array
    err_estimate: jax.Array
    eta: jax.Array

    @classmethod
    def zeros(cls, dtype=jnp.float64) -> "StepTraceHeader":
        """Build an all-zero header."""
        return cls(
            nst=_i(),
            t=_f(dtype),
            h=_f(dtype),
            k=_i(),
            accepted=_b(),
            result_code=_i(),
            newton_iters=_i(),
            err_estimate=_f(dtype),
            eta=_f(dtype),
        )


class NDFTraceRecord(eqx.Module):
    """Per-attempt trace for the numerical-differentiation-formula engine.

    The fields record the attempt's order and step, the basis rescale and
    its ratio, the controller counters, the leading-coefficient scale, the
    corrector's convergence quantities and setup/Jacobian decisions, the
    three per-order error estimates, and the accept/reject decision the
    controllers took (next order/step, the hold flag, and the
    history-restart flag).
    """

    header: StepTraceHeader
    k: jax.Array
    dt_attempt: jax.Array
    rho: jax.Array
    rescale_fired: jax.Array
    nconsteps: jax.Array
    consfailcnt: jax.Array
    invgdt: jax.Array
    theta: jax.Array
    ndz: jax.Array
    newton_converged: jax.Array
    call_setup: jax.Array
    jbad: jax.Array
    jcur: jax.Array
    nsetups: jax.Array
    nje: jax.Array
    eest: jax.Array
    eest1: jax.Array
    eest2: jax.Array
    k_next: jax.Array
    dt_next: jax.Array
    held: jax.Array
    restart_history: jax.Array

    @classmethod
    def zeros(cls, dtype=jnp.float64) -> "NDFTraceRecord":
        """Build an all-zero record (the engine fills the fields as wired)."""
        return cls(
            header=StepTraceHeader.zeros(dtype),
            k=_i(),
            dt_attempt=_f(dtype),
            rho=_f(dtype),
            rescale_fired=_b(),
            nconsteps=_i(),
            consfailcnt=_i(),
            invgdt=_f(dtype),
            theta=_f(dtype),
            ndz=_f(dtype),
            newton_converged=_b(),
            call_setup=_b(),
            jbad=_b(),
            jcur=_b(),
            nsetups=_i(),
            nje=_i(),
            eest=_f(dtype),
            eest1=_f(dtype),
            eest2=_f(dtype),
            k_next=_i(),
            dt_next=_f(dtype),
            held=_b(),
            restart_history=_b(),
        )


class BDFTraceRecord(eqx.Module):
    """Per-attempt trace for the backward-differentiation-formula engine.

    The fields record the order and its
    ``L = q+1``, the ``tau`` history and the ``l`` coefficients, the
    ``gamma``/``gammap``/``gamrat`` leading-coefficient triple and ``rl1``, the
    test-quotient vector, the setup gate and its condition, the Jacobian
    freshness flags and setup counters, the corrector convergence quantities, the
    accepted-correction norms, the status flags, the scaled error ``dsm``, the
    error-test failure counters, the step-growth cap and the per-order growth
    candidates with the chosen branch, the next order/step, the wait counter, the
    stability counter, the first-step size search, and the dense-output
    coefficients.
    """

    header: StepTraceHeader
    q: jax.Array
    L: jax.Array
    tau: jax.Array
    l: jax.Array  # noqa: E741 -- the BDF coefficient array's reference name.
    rl1: jax.Array
    gamma: jax.Array
    gammap: jax.Array
    gamrat: jax.Array
    tq: jax.Array
    call_setup: jax.Array
    setup_condition: jax.Array
    jbad: jax.Array
    jcur: jax.Array
    nsetups: jax.Array
    nje: jax.Array
    nstlj: jax.Array
    delnrm: jax.Array
    crate: jax.Array
    dcon: jax.Array
    acnrm: jax.Array
    acnrmcur: jax.Array
    nflag: jax.Array
    kflag: jax.Array
    dsm: jax.Array
    nef: jax.Array
    netf: jax.Array
    etamax: jax.Array
    etaq: jax.Array
    etaqm1: jax.Array
    etaqp1: jax.Array
    chosen_branch: jax.Array
    qprime: jax.Array
    hprime: jax.Array
    qwait: jax.Array
    nscon: jax.Array
    hin: jax.Array
    dky_coeffs: jax.Array

    @classmethod
    def zeros(cls, dtype=jnp.float64) -> "BDFTraceRecord":
        """Build an all-zero record (the engine fills the fields as wired)."""
        return cls(
            header=StepTraceHeader.zeros(dtype),
            q=_i(),
            L=_i(),
            tau=_v(dtype),
            l=_v(dtype),
            rl1=_f(dtype),
            gamma=_f(dtype),
            gammap=_f(dtype),
            gamrat=_f(dtype),
            tq=_v(dtype),
            call_setup=_b(),
            setup_condition=_i(),
            jbad=_b(),
            jcur=_b(),
            nsetups=_i(),
            nje=_i(),
            nstlj=_i(),
            delnrm=_f(dtype),
            crate=_f(dtype),
            dcon=_f(dtype),
            acnrm=_f(dtype),
            acnrmcur=_f(dtype),
            nflag=_i(),
            kflag=_i(),
            dsm=_f(dtype),
            nef=_i(),
            netf=_i(),
            etamax=_f(dtype),
            etaq=_f(dtype),
            etaqm1=_f(dtype),
            etaqp1=_f(dtype),
            chosen_branch=_i(),
            qprime=_i(),
            hprime=_f(dtype),
            qwait=_i(),
            nscon=_i(),
            hin=_v(dtype),
            dky_coeffs=_v(dtype),
        )


class IDATraceRecord(eqx.Module):
    """Per-attempt trace for the implicit-DAE engine.

    The fields record the ``ns`` reset counter, the
    ``psi`` history and the ``alpha``/``sigma``/``gamma`` coefficients with
    ``alphas``/``alpha0`` and the error constant ``ck``, the leading-coefficient
    triple ``cjlast``/``cj``/``cjratio``, the setup decision and counter, the
    corrector convergence rate memory ``ss`` and its per-iteration ``delnrm`` and
    ``rate``, the per-order error norms and the resulting order choice ``knew``,
    the error-test product ``ck * enorm_k``, the convergence/error failure
    counters, the phase state and phase action, and the dense-output
    coefficients.
    """

    header: StepTraceHeader
    ns: jax.Array
    psi: jax.Array
    alpha: jax.Array
    sigma: jax.Array
    gamma: jax.Array
    alphas: jax.Array
    alpha0: jax.Array
    ck: jax.Array
    cjlast: jax.Array
    cj: jax.Array
    cjratio: jax.Array
    call_lsetup: jax.Array
    nsetups: jax.Array
    ss: jax.Array
    delnrm: jax.Array
    rate: jax.Array
    err_k: jax.Array
    err_km1: jax.Array
    err_km2: jax.Array
    knew: jax.Array
    error_test_ck_enorm: jax.Array
    ncf: jax.Array
    nef: jax.Array
    phase: jax.Array
    phase_action: jax.Array
    cvals: jax.Array
    dvals: jax.Array

    @classmethod
    def zeros(cls, dtype=jnp.float64) -> "IDATraceRecord":
        """Build an all-zero record (the engine fills the fields as wired)."""
        return cls(
            header=StepTraceHeader.zeros(dtype),
            ns=_i(),
            psi=_v(dtype),
            alpha=_f(dtype),
            sigma=_f(dtype),
            gamma=_v(dtype),
            alphas=_f(dtype),
            alpha0=_f(dtype),
            ck=_f(dtype),
            cjlast=_f(dtype),
            cj=_f(dtype),
            cjratio=_f(dtype),
            call_lsetup=_b(),
            nsetups=_i(),
            ss=_f(dtype),
            delnrm=_f(dtype),
            rate=_f(dtype),
            err_k=_f(dtype),
            err_km1=_f(dtype),
            err_km2=_f(dtype),
            knew=_i(),
            error_test_ck_enorm=_f(dtype),
            ncf=_i(),
            nef=_i(),
            phase=_i(),
            phase_action=_i(),
            cvals=_v(dtype),
            dvals=_v(dtype),
        )


_RECORD_TYPES = {
    "ndf": NDFTraceRecord,
    "bdf": BDFTraceRecord,
    "ida": IDATraceRecord,
}


def select_trace(kind: str):
    """Return the per-attempt trace record type for a trace ``kind``.

    Args:
        kind: One of ``"ndf"``, ``"bdf"``, ``"ida"``.

    Returns:
        The matching `eqx.Module` record type.

    Raises:
        ValueError: If ``kind`` names no trace family.
    """
    try:
        return _RECORD_TYPES[kind]
    except KeyError:
        raise ValueError(
            f"unknown trace kind {kind!r}; expected one of {sorted(_RECORD_TYPES)}"
        ) from None


def zero_record(kind: str, dtype=jnp.float64):
    """Build a single all-zero per-attempt record for a trace ``kind``."""
    return select_trace(kind).zeros(dtype)


def alloc_trace(kind: str, max_steps: int, dtype=jnp.float64):
    """Allocate a step-indexed trace buffer sized to ``max_steps`` attempts.

    Every leaf of the ``kind`` record is broadcast to a leading ``max_steps``
    axis, preserving its per-field dtype (counters and flags stay integer/bool,
    coefficients take ``dtype``). The result is a record of the same type whose
    leaves are the fixed-shape buffers
    [write_trace][diff_soup.timestepping.vsvo.common.trace.write_trace]
    fills.

    Args:
        kind: One of ``"ndf"``, ``"bdf"``, ``"ida"``.
        max_steps: The static step budget (buffer leading dimension).
        dtype: Floating dtype for the coefficient fields.

    Returns:
        A record whose leaves are zero buffers of shape ``(max_steps, ...)``.
    """
    rec = zero_record(kind, dtype)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((max_steps, *jnp.shape(leaf)), leaf.dtype), rec
    )


def write_trace(buffers: Any, i, record: Any):
    """Write one per-attempt ``record`` into slot ``i`` of ``buffers``.

    ``buffers`` and ``record`` share the record type; each buffer leaf has the
    record leaf's shape with a leading step axis, so the write is a per-leaf
    ``.at[i].set(...)``. Under a replaying (checkpointed) solve loop the
    buffer leaves arrive as the loop's write-only buffer wrappers rather than
    plain arrays; those wrappers are themselves pytree nodes, so the write
    must treat them as leaves and go through the wrapper-aware setter.

    Args:
        buffers: The step-indexed buffer record from
            [alloc_trace][diff_soup.timestepping.vsvo.common.trace.alloc_trace].
        i: The slot index to write (traced integer is fine).
        record: A single per-attempt record of the same ``kind``.

    Returns:
        The buffer record with slot ``i`` overwritten.
    """
    return jax.tree_util.tree_map(
        lambda buf, val: buffer_at_set(buf, i, val), buffers, record, is_leaf=_is_buffer
    )


def finalize_trace(buffers: Any, n_steps):
    """Slice a trace buffer down to the number of attempts actually recorded.

    The buffer is allocated to the static step budget; only ``n_steps`` slots
    hold real attempts. When ``n_steps`` is known concretely (eager execution)
    the buffer is sliced to that length; under a trace the count is not a static
    integer, so the full fixed-shape buffer is returned unchanged to keep the
    compiled path free of data-dependent shapes.

    Args:
        buffers: The step-indexed buffer record from
            [alloc_trace][diff_soup.timestepping.vsvo.common.trace.alloc_trace].
        n_steps: The number of attempts recorded.

    Returns:
        The buffer sliced to ``n_steps`` rows when that count is concrete, else
        the full buffer.
    """
    try:
        n = int(n_steps)
    except (TypeError, jax.errors.ConcretizationTypeError):
        return buffers
    return jax.tree_util.tree_map(lambda buf: buf[:n], buffers)


__all__ = [
    "StepTraceHeader",
    "NDFTraceRecord",
    "BDFTraceRecord",
    "IDATraceRecord",
    "select_trace",
    "zero_record",
    "alloc_trace",
    "write_trace",
    "finalize_trace",
]
