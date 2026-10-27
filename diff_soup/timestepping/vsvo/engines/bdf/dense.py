"""Dense output and initial-step search for the BDF-tracking engine.

[bdf_get_dky][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_get_dky] evaluates the interpolating polynomial: given the Nordsieck
history ``zn`` and the step size ``h`` and order ``q`` a completed step left
behind, it evaluates the ``k``-th derivative of the interpolating polynomial
at any query time ``t`` on the accepted interval. [bdf_dense][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_dense] is a small
wrapper that returns both the state and its derivative, mirroring the
IDA-tracking engine's ``ida_get_solution``.

[bdf_initial_step][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_initial_step] searches for an initial step size ``h0`` with ``||h0**2 *
ydd/2||`` (weighted RMS) close to 1, together with its two helpers
[bdf_upper_bound_h0][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_upper_bound_h0] and [bdf_ydd_norm][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_ydd_norm].

``bdf_initial_step`` differs from a mass-free initial-step search in two ways, both
required by running under a mass matrix and a right-hand side that cannot
itself signal a recoverable failure:

- The right-hand side this engine integrates is a plain JAX function, not a
  callback that can report a recoverable evaluation failure. A mass-free
  search's inner retry loop (cut the trial step by a fixed factor and
  re-evaluate, up to four times) therefore never has anything to recover
  from, and is not realized here -- every second-derivative estimate is
  computed by the one right-hand-side evaluation the outer loop calls for.
- For a system ``M(t, y) y' = f(t, y)``, the quantity whose second time
  derivative this search is estimating is ``y``, not ``f``. Every place a
  mass-free search reads or evaluates the derivative, this module instead
  reads or evaluates the mass-consistent derivative ``f / M`` on rows with
  nonzero mass (zero on rows with zero mass, e.g. algebraic constraint
  rows). For an identity mass matrix this reduces exactly to the mass-free
  form.

Every quantity these two functions compute is a step-control scalar or
weight, not part of the solution trajectory; none of the arguments are
stopped, so callers that want a gradient-dead initial step should stop the
result themselves.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from diff_soup.timestepping.vsvo.common.weights import wrms_norm

from .state import QMAX

FUZZ_FACTOR = 100.0
"""Slack factor for the dense-output query-time window."""

HLB_FACTOR = 100.0
"""Lower-bound factor for the initial-step search: ``hlb = HLB_FACTOR *
tround``."""

HUB_FACTOR = 0.1
"""Upper-bound factor for the initial-step search: both the cap on ``hub``
relative to the distance to the target time and the allowed relative growth
of ``y0`` a forward-Euler step may produce."""

H_BIAS = 0.5
"""The initial-step search's final bias, and also its ``hrat``
convergence-window's lower bound."""

MAX_ITERS = 4
"""Iteration cap for the initial-step search's outer loop."""

POINT2 = 0.2
"""The step-size cut applied inside the initial-step search when a
right-hand-side evaluation reports a recoverable failure. This engine's
right-hand side cannot report a recoverable failure (see the module
docstring), so the cut is never applied; the constant is unused but
retained."""

TWO = 2.0
"""Threshold the initial-step search uses both to decide whether the
second-derivative estimate alone determines ``hnew`` and for the ``hrat``
convergence window."""


def bdf_get_dky(
    t: jax.Array,
    tn: jax.Array,
    h: jax.Array,
    q: jax.Array,
    zn: jax.Array,
    k: int = 0,
) -> jax.Array:
    """Interpolate the ``k``-th derivative of ``y`` at ``t`` from the Nordsieck array.

    Evaluates ``dky = sum_{j=k}^{q} c(j, k) * s**(j - k) * zn[j]`` with ``s
    = (t - tn) / h`` and ``c(j, k) = j * (j - 1) * ... * (j - k + 1)`` the
    falling factorial, then ``dky *= h**(-k)`` for ``k > 0``. Both the falling
    factorial and the power of ``h`` are built by exact repeated
    multiplication, never a floating-point power function.

    Valid for query times on ``[tn - hu - tfuzz, tn + tfuzz]``, ``tfuzz =
    FUZZ_FACTOR * uround * (|tn| + |hu|)``. This engine's
    driver always calls with a query time already known to lie in the last
    accepted step's interval, so this function does not itself validate or
    clamp ``t``.

    The order actually used, ``q``, may be a traced value; ``k`` must be a
    plain Python integer known at trace time (this engine only ever calls
    with ``k=0`` for the state and ``k=1`` for its derivative). The sum runs
    over the compile-time-static range ``j = k .. QMAX``, masked term-by-term
    by ``q >= j`` so every array involved keeps a fixed shape regardless of
    the traced order.

    Args:
        t: The query time.
        tn: The current internal time (end of the last accepted step).
        h: The step size the Nordsieck history is currently scaled with.
        q: The current integration order (``int32``, ``1 <= q <= QMAX``).
        zn: The Nordsieck history, shape ``(L_ZN, n)`` -- rows ``0..QMAX``
            are read; row ``QMAX + 1`` (the saved-correction slot) is not.
        k: The derivative order to interpolate (``0`` for the state).

    Returns:
        The interpolated ``k``-th derivative, shape ``(n,)``.
    """
    if k < 0:
        raise ValueError(f"bdf_get_dky requires k >= 0; got k={k}.")

    dtype = zn.dtype
    t = jnp.asarray(t, dtype=dtype)
    tn = jnp.asarray(tn, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    s = (t - tn) / h

    zero = jnp.zeros((), dtype=dtype)
    dky = jnp.zeros((zn.shape[1],), dtype=dtype)
    for j in range(k, QMAX + 1):
        active = q >= j

        c = 1.0
        for i in range(j, j - k, -1):
            c *= i

        s_pow = jnp.ones((), dtype=dtype)
        for _ in range(j - k):
            s_pow = s_pow * s

        term = jnp.asarray(c, dtype=dtype) * s_pow
        dky = dky + jnp.where(active, term, zero) * zn[j]

    if k == 0:
        return dky

    denom = jnp.ones((), dtype=dtype)
    for _ in range(k):
        denom = denom * h
    return dky / denom


def bdf_dense(state: Any, tn: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Evaluate the dense interpolant ``(y(t), ydot(t))`` on the last accepted step.

    A small wrapper around two [bdf_get_dky][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_get_dky] calls (``k=0`` and ``k=1``),
    mirroring the IDA-tracking engine's ``ida_get_solution``. ``state`` is
    read only for its ``h``, ``q``, and ``zn`` fields -- the current internal
    time ``tn`` is not part of the carried step state, so the caller supplies
    it separately.

    Args:
        state: The carried BDF step state (any object with ``h``, ``q``, and
            ``zn`` fields, e.g. [BDFState][diff_soup.timestepping.vsvo.engines.bdf.state.BDFState]).
        tn: The current internal time (end of the last accepted step).
        t: The query time.

    Returns:
        The interpolated state and its derivative, each shape ``(n,)``.
    """
    y = bdf_get_dky(t, tn, state.h, state.q, state.zn, k=0)
    ydot = bdf_get_dky(t, tn, state.h, state.q, state.zn, k=1)
    return y, ydot


def bdf_upper_bound_h0(
    tdist: jax.Array,
    zn0: jax.Array,
    zn1: jax.Array,
    ewt: jax.Array,
) -> jax.Array:
    """Bound the initial step size from the initial state and its derivative.

    Bounds a forward-Euler step, allowed to grow ``|y0|`` by at most a
    factor of ``HUB_FACTOR``, using ``1/ewt`` as a floor against zero
    components of ``y0``; the result is further capped at ``HUB_FACTOR *
    tdist``.

    Args:
        tdist: The distance to the target time.
        zn0: The initial state, shape ``(n,)``.
        zn1: The initial derivative (mass-consistent for a non-identity mass
            matrix), shape ``(n,)``.
        ewt: The error-weight vector at ``zn0``, shape ``(n,)``.

    Returns:
        The upper bound on the initial step size (unsigned).
    """
    dtype = zn0.dtype
    hub_factor = jnp.asarray(HUB_FACTOR, dtype=dtype)
    denom = hub_factor * jnp.abs(zn0) + 1.0 / ewt
    hub_inv = jnp.max(jnp.abs(zn1) / denom)
    hub = hub_factor * tdist
    return jnp.where(hub * hub_inv > 1.0, 1.0 / hub_inv, hub)


def bdf_slope_fn(
    problem: Any,
    theta: Any,
    *,
    tol: float,
    control: Any = None,
    backend: Any = "auto",
    layout: Any = None,
    precision: Any = None,
    reverse_precision: Any = None,
):
    """Build this engine's slope evaluation ``(t, y, ydot_guess) -> ydot``.

    The slope seeds the first Nordsieck history column, sizes the initial
    step, and rebuilds history on an order-one restart. It is evaluated only
    through the solver problem's capability-routed startup operation.

    Args:
        problem: The engine-facing problem view.
        theta: The learnable parameters the residual reads.
        tol: Residual gate for the consistent-initialization solve.
        control: The control-value pytree the residual reads.
        backend: The linear route for the initialization solve — pass the engine's own backend so one solver
            configuration governs every linear solve of the run.
        layout: Device-placement policy for the initialization solve's
            factor; ``None`` leaves the backend's default placement.
        precision: Dtype policy for the initialization solve's factor and
            its tangent solve; ``None`` leaves the backend's default (fp64).
        reverse_precision: Dtype policy for the transposed factor a reverse
            sweep solves with; ``None`` uses ``precision``.

    Returns:
        A callable ``slope(t, y, ydot_guess) -> ydot`` of shape ``(n,)``.
    """

    def slope(t, y, ydot_guess):
        return problem.startup_slope(
            t,
            y,
            theta,
            control,
            ydot0=ydot_guess,
            tol=tol,
            backend=backend,
            layout=layout,
            precision=precision,
            reverse_precision=reverse_precision,
        )

    return slope


def bdf_ydd_norm(
    problem: Any,
    theta: Any,
    t0: jax.Array,
    hg: jax.Array,
    zn0: jax.Array,
    zn1: jax.Array,
    ewt: jax.Array,
    *,
    slope_fn: Any,
) -> jax.Array:
    """Estimate ``||y''||`` (weighted RMS) by a forward difference quotient.

    Estimates the second derivative, mass-aware (see the module docstring):
    the trial state ``y = zn0 + hg * zn1`` is advanced by the *signed* trial
    step ``hg``, the
    mass-consistent derivative is evaluated there, and the norm of its
    difference from ``zn1`` divided by ``hg`` estimates ``y''``.

    Args:
        problem: The solver problem being integrated.
        theta: The learnable parameters the residual reads.
        t0: The time ``zn0``/``zn1`` are evaluated at.
        hg: The signed trial step (``hg = hg_magnitude * sign``).
        zn0: The initial state, shape ``(n,)``.
        zn1: The initial mass-consistent derivative, shape ``(n,)``.
        ewt: The error-weight vector at ``zn0``, shape ``(n,)``.
        slope_fn: Startup slope evaluation ``(t, y, ydot_guess) -> ydot``.

    Returns:
        The weighted-RMS norm of the estimated second derivative.
    """
    y = zn0 + hg * zn1
    ydot1 = slope_fn(t0 + hg, y, zn1)
    return wrms_norm((ydot1 - zn1) / hg, ewt)


def bdf_initial_step(
    problem: Any,
    theta: Any,
    t0: jax.Array,
    t1: jax.Array,
    zn0: jax.Array,
    zn1: jax.Array,
    ewt: jax.Array,
    uround: jax.Array,
    hmax_inv: jax.Array,
    dtype: type,
    *,
    slope_fn: Any,
) -> jax.Array:
    """Search for an initial step size ``h0`` with ``||h0**2 * y''/2|| ~ 1``.

    Mass-aware (see the module docstring). Bounds ``hlb = HLB_FACTOR *
    tround`` and ``hub`` ([bdf_upper_bound_h0][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_upper_bound_h0]) bracket the search; if
    they cross, their geometric mean is returned immediately. Otherwise an
    outer loop of at most ``MAX_ITERS`` passes calls [bdf_ydd_norm][diff_soup.timestepping.vsvo.engines.bdf.dense.bdf_ydd_norm] and
    proposes a new trial step, stopping when two successive proposals agree
    within a factor of two, when the loop is out of iterations, or (after the
    first pass) when the proposal grows by more than a factor of two -- taken
    as a sign the estimate is unreliable, in which case the trial step itself
    is kept instead. The final proposal is biased by ``H_BIAS``, clamped to
    ``[hlb, hub]``, and signed toward ``t1``.

    Every one of these iterations is unrolled as a fixed-length Python loop
    (``MAX_ITERS`` is a compile-time constant): each iteration computes its
    candidate values unconditionally and blends them against the
    already-decided ones with ``jnp.where``, so the traced early-stopping
    decisions never change the shape or the trace of the computation.

    ``hmax_inv`` is accepted for call-site symmetry with this engine's other
    step-control entry points but is not read: the ``hmax``/``hmin`` bounds
    are applied to the returned value by the caller, not inside this search.

    Args:
        problem: The solver problem being integrated.
        theta: The learnable parameters the residual reads.
        t0: The initial time.
        t1: The target time (direction and distance only -- integration does
            not reach ``t1`` here).
        zn0: The initial state, shape ``(n,)``.
        zn1: The initial mass-consistent derivative, shape ``(n,)``.
        ewt: The error-weight vector at ``zn0``, shape ``(n,)``.
        uround: The unit roundoff.
        hmax_inv: Unused (see above).
        dtype: The floating-point dtype every intermediate array uses.
        slope_fn: Startup slope evaluation ``(t, y, ydot_guess) -> ydot``
            used by the second-derivative estimates.

    Returns:
        The signed initial step size ``h0``.
    """
    del hmax_inv

    t0 = jnp.asarray(t0, dtype=dtype)
    t1 = jnp.asarray(t1, dtype=dtype)
    zn0 = jnp.asarray(zn0, dtype=dtype)
    zn1 = jnp.asarray(zn1, dtype=dtype)
    ewt = jnp.asarray(ewt, dtype=dtype)
    uround = jnp.asarray(uround, dtype=dtype)

    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(TWO, dtype=dtype)
    half = jnp.asarray(H_BIAS, dtype=dtype)

    tdiff = t1 - t0
    sign = jnp.where(tdiff > 0, one, -one)
    tdist = jnp.abs(tdiff)
    tround = uround * jnp.maximum(jnp.abs(t0), jnp.abs(t1))

    hlb = jnp.asarray(HLB_FACTOR, dtype=dtype) * tround
    hub = bdf_upper_bound_h0(tdist, zn0, zn1, ewt)

    hg0 = jnp.sqrt(hlb * hub)
    early_exit = hub < hlb

    hg = hg0
    hnew = hg0
    done = jnp.asarray(False)

    for count1 in range(1, MAX_ITERS + 1):
        yddnrm = bdf_ydd_norm(problem, theta, t0, hg * sign, zn0, zn1, ewt, slope_fn=slope_fn)

        hnew_candidate = jnp.where(
            yddnrm * hub * hub > two,
            jnp.sqrt(two / yddnrm),
            jnp.sqrt(hg * hub),
        )

        is_last = count1 == MAX_ITERS
        hrat = hnew_candidate / hg
        converged = (hrat > half) & (hrat < two)
        unreliable = jnp.asarray(not is_last and count1 > 1) & (hrat > two) & (~converged)
        stop = jnp.asarray(is_last) | converged | unreliable

        hnew_this = jnp.where(unreliable, hg, hnew_candidate)

        hnew = jnp.where(done, hnew, hnew_this)
        hg = jnp.where(done, hg, jnp.where(stop, hg, hnew_this))
        done = done | stop

    h0 = half * hnew
    h0 = jnp.clip(h0, hlb, hub)
    h0 = sign * h0

    return jnp.where(early_exit, sign * hg0, h0)


__all__ = [
    "FUZZ_FACTOR",
    "HLB_FACTOR",
    "HUB_FACTOR",
    "H_BIAS",
    "MAX_ITERS",
    "POINT2",
    "TWO",
    "bdf_get_dky",
    "bdf_dense",
    "bdf_slope_fn",
    "bdf_upper_bound_h0",
    "bdf_ydd_norm",
    "bdf_initial_step",
]
