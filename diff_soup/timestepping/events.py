# >500 LOC justified: single cohesive event/affect subsystem (root-find, affect arms, reverse rules share private helpers).
"""events.py — Unified diffrax/SciML event contract for diff_soup.

Provides four public types for stopping or modifying an ongoing solve:

* [Event][diff_soup.timestepping.events.Event] — the core frozen module (``cond_fn`` + optional ``affect``
  + optional ``root_finder`` + ``direction``).  Return-type discrimination
  follows diffrax: a ``bool`` cond_fn gives a *discrete* event checked at
  each accepted-step boundary; a real-valued cond_fn gives a *continuous*
  event located by bisection on the dense interpolant.
  ``affect=None`` → terminate (diffrax-parity);
  ``affect`` set → mutate-and-continue (SciML-parity).
* [EventSet][diff_soup.timestepping.events.EventSet] — composes multiple [Event][diff_soup.timestepping.events.Event] objects (SciML
  ``CallbackSet`` spirit).
* [EventObservable][diff_soup.timestepping.events.EventObservable] — thin adapter that wraps an arbitrary scalar
  functional of ``(t, y, args)`` and exposes ``.crosses(level)``,
  ``.below(level)``, ``.above(level)`` returning ready-made ``cond_fn``
  closures.  The classmethods ``.soc`` / ``.voltage`` accept a user-supplied
  functional so battery observables integrate naturally.
* [terminate_steady_state][diff_soup.timestepping.events.terminate_steady_state] — factory returning an [Event][diff_soup.timestepping.events.Event] whose
  ``cond_fn`` tests ``‖f(t,y)‖ < atol + rtol·‖y‖`` (diffrax
  ``steady_state_event`` / SciML ``TerminateSteadyState``).  The user
  supplies the residual/vector-field callable explicitly via ``args`` or the
  ``residual_fn`` keyword.

All callable fields ride as **static auxiliary data** (``eqx.field(static=True)``)
so an [Event][diff_soup.timestepping.events.Event] or [EventSet][diff_soup.timestepping.events.EventSet] adds **zero array leaves** to the
JAX pytree.  The default path (no ``event=``) is therefore byte-identical to
solves that carry no event.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._errors import ConfigurationError
from diff_soup.timestepping.saveat import DenseOutputPolicy


def _bisection_iteration_count(rtol: float, max_iters_cap: int) -> int:
    """Return the static halving count shared by solve and status replay."""
    return int(
        min(
            max_iters_cap,
            max(1, int(math.ceil(-math.log2(rtol))) + 2),
        )
    )


# ---------------------------------------------------------------------------
# Bisection root finder (continuous root-find termination)
# ---------------------------------------------------------------------------


class Bisection(eqx.Module):
    """Root finder that locates an event crossing by bisection on the step's interval.

    It runs a fixed number of halvings, set by `rtol` and capped by `max_iters_cap`, so it works under `jit` and `vmap`.

    Args:
        rtol: Tolerance relative to the bracket width, which sets the number of halvings.
        atol: Absolute time tolerance that `JNDF`, `JBDF`, and `JIDA` use to match the crossing against other crossings and save times.
        max_iters_cap: Upper bound on the number of halvings.
    """

    rtol: float = eqx.field(default=1e-6, static=True)
    atol: float = eqx.field(default=1e-12, static=True)
    max_iters_cap: int = eqx.field(default=100, static=True)

    def solve(
        self,
        fn: Callable,
        lo: jax.Array,
        hi: jax.Array,
    ) -> jax.Array:
        """Locate ``t*`` with ``fn(t*) ≈ 0`` in ``[lo, hi]`` by bisection.

        The bracket ``[lo, hi]`` is assumed to contain a sign change
        (``fn(lo) * fn(hi) <= 0``); the caller (the continuous event hook)
        guarantees this by only invoking the root find on a detected sign
        change.  The loop is branchless: each iteration evaluates ``fn`` at
        the midpoint and updates ``lo``/``hi`` with `jax.numpy.where`,
        keeping the subinterval that still straddles zero.

        Args:
            fn: Scalar function ``t -> real``.  Must be traceable (no Python
                control flow on ``t``); evaluated ``max_iters`` times.
            lo: Lower bracket endpoint (scalar).
            hi: Upper bracket endpoint (scalar).

        Returns:
            jax.Array: The midpoint of the final bracket — the located root,
            within ``rtol``/``atol`` of the true crossing.
        """
        lo = jnp.asarray(lo)
        hi = jnp.asarray(hi)
        # The bracket runs in the solve's own time dtype (a Python float is weak).
        dtype = jnp.result_type(lo, hi, float)
        lo = lo.astype(dtype)
        hi = hi.astype(dtype)

        # Static trip count: halvings needed to shrink the bracket below the
        # bracket-relative ``rtol``. Computed from the static Python float
        # ``self.rtol`` (NOT a traced array) so the loop is static-shape and
        # JIT/vmap-safe. After ``n`` halvings the half-bracket is below
        # ``rtol`` * (initial width); +2 margin, capped for safety.
        n_iters = _bisection_iteration_count(self.rtol, self.max_iters_cap)

        f_lo = fn(lo)
        sign_lo = jnp.sign(f_lo)

        def _body(_, carry):  # noqa: ANN001, ANN202
            lo_c, hi_c = carry
            mid = 0.5 * (lo_c + hi_c)
            f_mid = fn(mid)
            # Keep the half-interval that still brackets the root: if f(mid)
            # has the same sign as f(lo), the root is in [mid, hi]; else in
            # [lo, mid]. Branchless via where (sign_lo is fixed at the
            # original lo so the test is purely on f_mid's sign).
            same_sign = jnp.sign(f_mid) == sign_lo
            lo_new = jnp.where(same_sign, mid, lo_c)
            hi_new = jnp.where(same_sign, hi_c, mid)
            return (lo_new, hi_new)

        lo_f, hi_f = jax.lax.fori_loop(0, n_iters, _body, (lo, hi))
        return 0.5 * (lo_f + hi_f)


# ---------------------------------------------------------------------------
# Caller-supplied-root-finder variant (used by _locate_crossing)
# ---------------------------------------------------------------------------


def root_with_ift(g: Callable, root_finder: Any, lo: jax.Array, hi: jax.Array) -> jax.Array:
    r"""``t* = root_finder.solve(g, lo, hi)`` with the IFT switch-time adjoint attached.

    The AD-exact switch-time root used by the in-loop continuous-event hook
    ([_locate_crossing][diff_soup.timestepping.events._locate_crossing]).  The **primal is EXACTLY** ``root_finder.solve``
    (bit-identical to the raw, un-wired call the event-accuracy bands pin) — only
    the AD behaviour changes.  The tangent / cotangent use the
    implicit-function-theorem switch-time derivative

    .. math::
        \\frac{dt^*}{d\\theta} = -\\frac{\\partial g/\\partial\\theta}
                                       {\\partial g/\\partial t}
        \\quad\\text{at } t^*,

    one extra ``g`` linearisation rather than back-propagating through the
    bisection halvings (whose ``jnp.sign`` comparisons give a zero/garbage
    derivative).  The differentiable parameters ``θ`` are the constants captured
    by the closure ``g`` (here: the live ``control`` pytree and the one-step
    interpolant's differentiable data).  The bracket endpoints carry zero tangent
    (the root is fixed by ``g(t*)=0``).

    Implementation: `jax.lax.custom_root`. It carries the implicit-diff
    plumbing as a primitive, so
    JAX manages those int consts correctly for BOTH forward (``jax.jvp``) and
    reverse (``jax.grad``/``jax.vjp``, under a checkpointed outer loop) modes,
    and the rule composes under repeated differentiation — the switch-time
    derivative flows through ``jax.hessian``-style forward-over-reverse too.

    Args:
        g: Scalar function ``t -> real`` closing over the differentiable
            parameters ``θ``.  Must be traceable (no Python control flow on
            ``t``).
        root_finder: A [Bisection][diff_soup.timestepping.events.Bisection] (or any object with a
            ``solve(fn, lo, hi)`` method).  Static — used verbatim in the primal.
        lo: Lower bracket endpoint (scalar).
        hi: Upper bracket endpoint (scalar).

    Returns:
        jax.Array: The located root ``t*`` (within ``root_finder``'s tolerance).
    """
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    x0 = 0.5 * (lo + hi)

    def _solve(g_fn, _x0):  # noqa: ANN001, ANN202
        # PRIMAL — bit-identical to the raw ``root_finder.solve(g, lo, hi)`` the
        # un-wired ``_locate_crossing`` made (the event-accuracy bands pin it).
        # ``custom_root`` returns this verbatim for the forward value.
        return root_finder.solve(g_fn, lo, hi)

    def _tangent_solve(g_lin, b):  # noqa: ANN001, ANN202
        # Solve the linearised scalar root system ``g'(t*) · dt = b`` for ``dt``.
        # ``g_lin`` is the linearisation of ``g`` at the root (linear through the
        # origin), so its slope is ``g_lin(1) = g'(t*)`` — one extra ``g``
        # linearisation, the IFT denominator.  Scalar ⇒ a single divide.
        #
        # The caller root-finds branchlessly on EVERY step and discards the
        # result on non-firing steps via ``where(fired, ...)`` — but a
        # degenerate slope on a discarded step still poisons cotangent
        # accumulation with ``0 * (b / 0)``. Guard the zero-slope lane; at a
        # genuine transversal crossing the slope is bounded away from zero
        # and the guard is inert.
        slope = g_lin(jnp.ones_like(b))
        degenerate = slope == 0.0
        safe_slope = jnp.where(degenerate, jnp.ones_like(slope), slope)
        return jnp.where(degenerate, jnp.zeros_like(b), b / safe_slope)

    return jax.lax.custom_root(g, x0, _solve, _tangent_solve)


# ---------------------------------------------------------------------------
# Continuous-crossing locator (used by the JNDF core event hook)
# ---------------------------------------------------------------------------


def _bisection_evaluations_ok(
    fn: Callable,
    status_fn: Callable,
    root_finder: Bisection,
    lo: jax.Array,
    hi: jax.Array,
) -> jax.Array:
    """Replay Bisection's candidate schedule and fold evaluation status.

    Root-finder APIs return only a root, so status cannot ride through
    [root_with_ift][diff_soup.timestepping.events.root_with_ift] without changing its established primal. For the
    supported [Bisection][diff_soup.timestepping.events.Bisection] locator this small diagnostic replay follows
    the exact same fixed candidate schedule and proves that every state used
    to choose the bracket was valid. It runs only when the caller supplies a
    projected-state status callback; the legacy event path is untouched.
    """
    n_iters = _bisection_iteration_count(root_finder.rtol, root_finder.max_iters_cap)
    # Diagnostics are nondifferentiable control metadata. Stop their values so
    # the replay cannot add a second sensitivity path beside root_with_ift.
    f_lo = jax.lax.stop_gradient(fn(lo))
    sign_lo = jnp.sign(f_lo)
    ok0 = jnp.asarray(status_fn(lo), dtype=jnp.bool_)

    def _body(_, carry):  # noqa: ANN001, ANN202
        lo_c, hi_c, ok_c = carry
        mid = 0.5 * (lo_c + hi_c)
        f_mid = jax.lax.stop_gradient(fn(mid))
        ok_mid = jnp.asarray(status_fn(mid), dtype=jnp.bool_)
        same_sign = jnp.sign(f_mid) == sign_lo
        lo_new = jnp.where(same_sign, mid, lo_c)
        hi_new = jnp.where(same_sign, hi_c, mid)
        return lo_new, hi_new, jnp.logical_and(ok_c, ok_mid)

    _, _, evaluations_ok = jax.lax.fori_loop(0, n_iters, _body, (lo, hi, ok0))
    return evaluations_ok


def _locate_crossing(
    cond_fn: Callable,
    interp_fn: Callable,
    root_finder: Any,
    t_lo: jax.Array,
    t_hi: jax.Array,
    cond_lo: jax.Array,
    cond_hi: jax.Array,
    direction: str = "any",
    *,
    control: Any = None,
    control_at: Optional[Callable] = None,
    event: Any = None,
    state_at: Optional[Callable] = None,
    state_status: Optional[Callable] = None,
    active: Any = None,
):
    """Detect a sign change of ``cond_fn`` over ``[t_lo, t_hi]`` and root-find it.

    Helper for the JNDF core's continuous (real-valued ``cond_fn``) event hook.
    Given the condition values already evaluated at the two step
    endpoints (``cond_lo`` at ``t_lo``, ``cond_hi`` at ``t_hi``) and a
    one-step interpolant ``interp_fn(t) -> y`` valid over the step, this:

    1. tests for a (direction-filtered) sign change between the endpoints, and
    2. if one is present, root-finds the exact crossing time ``t*`` of
       ``g(t) = cond_fn(t, state(t), control)`` over ``[t_lo, t_hi]`` with
       ``root_finder`` (a [Bisection][diff_soup.timestepping.events.Bisection]).
       ``state(t)`` is the one-step interpolant unless an event-aware
       ``state_at`` evaluator is supplied.

    The endpoint sign test is cheap (two precomputed condition values); the
    root find — ``root_finder.solve`` on the interpolant, with a per-candidate
    algebraic projection when the event's state policy is ``projected`` — is
    the expensive arm, so it runs under `jax.lax.cond`, only on a step
    whose endpoints actually bracket a direction-matching crossing (and, when
    the caller supplies ``active``, only on an accepted attempt).  A solve
    fires its terminating event on exactly one step, so the located arm's cost
    is paid once per solve rather than once per step.  Under ``vmap`` the
    ``cond`` lowers to a select (both arms execute per lane), which reproduces
    the previous always-on cost — never worse.

    Args:
        cond_fn: The event condition ``(t, y, args, *, control, **kw) -> real``.
            Called with ``args=None`` here (the in-loop ``lam`` is folded into
            the interpolated state by the caller when needed).
        interp_fn: One-step state interpolant ``t -> y`` over ``[t_lo, t_hi]``.
        root_finder: A [Bisection][diff_soup.timestepping.events.Bisection] (or any object with a
            ``solve(fn, lo, hi)`` method).
        t_lo: Step start time (scalar).
        t_hi: Step end time (scalar).
        cond_lo: ``cond_fn`` value at ``t_lo`` (scalar, precomputed).
        cond_hi: ``cond_fn`` value at ``t_hi`` (scalar, precomputed).
        direction: ``"up"`` (negative→positive), ``"down"`` (positive→negative),
            or ``"any"`` (either).  Filters which crossings fire.
        control: Optional fixed control pytree threaded to ``cond_fn``.
        control_at: Optional ``t -> control`` resolver for a time-dependent
            Recipe control. When present it takes precedence over ``control``.
        event: Event owning this crossing. Passed unchanged to ``state_at``;
            ignored on the legacy interpolated-state path.
        state_at: Optional ``(event, t) -> y`` evaluator. This lets a caller
            construct the state required by one event's ``state_policy`` at
            every root candidate. ``None`` preserves the legacy
            ``interp_fn(t)`` path exactly.
        state_status: Optional ``(event, t) -> bool`` companion reporting
            whether the event-specific state evaluation succeeded. For
            [Bisection][diff_soup.timestepping.events.Bisection], every bracket candidate and the located root
            are checked (on the located arm only). The endpoints are not:
            ``cond_lo`` / ``cond_hi`` come precomputed from the caller's exact
            accepted endpoint states, which involve no state evaluation that
            can fail. ``None`` means evaluation is supported and successful.
        active: Optional bool scalar further gating the located arm (the
            caller's step-accepted flag). ``None`` gates on the bracket test
            alone. The returned ``fired`` is NOT masked by ``active`` — the
            caller keeps its own accepted-step masking semantics.

    Returns:
        tuple: ``(fired, t_star, crossing_direction, evaluation_ok)`` where
        ``fired`` is a bool scalar (True iff a direction-matching crossing
        occurred this step), ``t_star`` is the located crossing time
        (meaningful only when ``fired``; defaults to ``t_hi`` otherwise),
        ``crossing_direction`` is ``+1`` for a negative-to-positive crossing,
        ``-1`` for positive-to-negative, ``0`` when nothing fired, and
        ``evaluation_ok`` reports event-specific state-evaluation success
        (always True on a step whose located arm did not run).
    """
    cond_lo = jnp.asarray(cond_lo)
    cond_hi = jnp.asarray(cond_hi)

    # Include an exact root at either endpoint, but do not report an interval
    # that remains identically on the event surface as a crossing.
    up_cross = jnp.logical_and(
        jnp.logical_and(cond_lo <= 0.0, cond_hi >= 0.0),
        jnp.logical_or(cond_lo < 0.0, cond_hi > 0.0),
    )
    down_cross = jnp.logical_and(
        jnp.logical_and(cond_lo >= 0.0, cond_hi <= 0.0),
        jnp.logical_or(cond_lo > 0.0, cond_hi < 0.0),
    )
    if direction == "up":
        fired = up_cross
    elif direction == "down":
        fired = down_cross
    else:
        fired = jnp.logical_or(up_cross, down_cross)

    def _g(t):  # noqa: ANN001, ANN202
        control_t = control if control_at is None else control_at(t)
        state_t = interp_fn(t) if state_at is None else state_at(event, t)
        return jnp.asarray(cond_fn(t, state_t, None, control=control_t)).reshape(())

    # AD-exact switch time: the primal is byte-identical to the raw
    # ``root_finder.solve(_g, t_lo, t_hi)`` (the event-accuracy bands pin it),
    # but ``root_with_ift`` attaches the implicit-function-theorem custom_jvp /
    # custom_vjp so ``dt*/dθ = -(∂g/∂θ)/(∂g/∂t)`` flows to the closed-over
    # differentiable parameters (the live ``control`` pytree + the one-step
    # interpolant's data) — instead of the near-zero/garbage cotangent that
    # back-propagating through bisection's ``jnp.sign`` comparisons gives.
    #
    # The root find (and its evaluation-status replay) runs under ``lax.cond``:
    # only a step whose endpoints bracket a direction-matching crossing — and,
    # when the caller distinguishes them, only an accepted attempt — pays for
    # it. With a ``projected`` state policy each root candidate is a full
    # algebraic Newton projection, so an always-on root find multiplies every
    # step's cost by the candidate count (the ~90× cutoff-protocol slowdown
    # this gate removed); the skipped arm is two constants.
    gate = fired if active is None else jnp.logical_and(fired, jnp.asarray(active, jnp.bool_))
    t_dtype = jnp.result_type(jnp.asarray(t_lo), jnp.asarray(t_hi), float)

    def _located(_):  # noqa: ANN001, ANN202
        t_star = root_with_ift(_g, root_finder, t_lo, t_hi)
        located_ok = jnp.asarray(True)
        if state_status is not None:
            candidate_ok = jnp.asarray(True)
            if isinstance(root_finder, Bisection):
                candidate_ok = _bisection_evaluations_ok(
                    _g,
                    lambda t: state_status(event, t),
                    root_finder,
                    t_lo,
                    t_hi,
                )
            root_ok = jnp.asarray(state_status(event, t_star), dtype=jnp.bool_)
            located_ok = jnp.logical_and(candidate_ok, root_ok)
        return jnp.asarray(t_star, dtype=t_dtype), located_ok

    def _skipped(_):  # noqa: ANN001, ANN202
        # A non-located step leaves the accepted-step endpoint untouched.
        return jnp.asarray(t_hi, dtype=t_dtype), jnp.asarray(True)

    t_star, evaluation_ok = jax.lax.cond(gate, _located, _skipped, None)
    crossing_direction = jnp.where(
        fired,
        jnp.where(up_cross, jnp.asarray(1, jnp.int32), jnp.asarray(-1, jnp.int32)),
        jnp.asarray(0, jnp.int32),
    )
    return fired, t_star, crossing_direction, evaluation_ok


# ---------------------------------------------------------------------------
# In-loop event resolution (factored from jndf/core.py)
# ---------------------------------------------------------------------------

# ``kind`` codes surfaced by [resolve_in_loop][diff_soup.timestepping.events.resolve_in_loop] (the winning-event type).
RESOLVE_KIND_NONE = 0
RESOLVE_KIND_DISCRETE = 1
RESOLVE_KIND_CONTINUOUS = 2


def resolve_in_loop(
    events,  # noqa: ANN001
    t_lo,  # noqa: ANN001
    t_hi,  # noqa: ANN001
    cond_lo_fn: Callable,
    cond_hi_fn: Callable,
    interp: Callable,  # noqa: ARG001 (held for the affect arm / signature stability)
    control: Any = None,  # noqa: ARG001 (folded into cond_*_fn closures by the caller)
    control_at: Optional[Callable] = None,
    direction_default: str = "any",
    *,
    state_at: Optional[Callable] = None,
    state_status: Optional[Callable] = None,
    active: Any = None,
):
    """Resolve which event fires this accepted step + where (SciML CallbackSet semantics).

    Factored to share one implementation across the variable-step engines with
    the same operations and ordering — the engine forward path is bit-identical.

    Two tiers, discriminated per event by whether it carries a ``root_finder``:

    * **Discrete** (``root_finder is None``): the bool ``cond_fn`` is evaluated
      at the step-end boundary (via ``cond_hi_fn``) and folded in declaration
      order — any fire stops the solve at ``t_hi``.
    * **Continuous** (``root_finder`` present): the real-valued ``cond_fn`` is
      evaluated at both endpoints (``cond_lo_fn`` / ``cond_hi_fn``) and, on a
      direction-matching sign change, the exact crossing time is root-found on
      the one-step interpolant via [_locate_crossing][diff_soup.timestepping.events._locate_crossing].  Among several
      that fire this step, the **earliest** crossing wins.

    The discrete fold and the continuous earliest-crossing are computed with
    the identical ops as the original inline block; the caller reconstructs the
    terminal ``(t, u)`` / ``done`` / ``term_code`` updates from the returned
    tuple (continuous ⇒ rewrite the endpoint to ``t_star``; discrete ⇒ stop at
    the boundary).

    Args:
        events: Ordered tuple of [Event][diff_soup.timestepping.events.Event] (declaration order).
        t_lo: Step-start time (scalar).
        t_hi: Step-end time (scalar).
        cond_lo_fn: Closure ``ev_cond_fn -> scalar`` evaluating an event's
            condition at the step start ``(t_lo, u_lo, lam_lo)`` (continuous
            low endpoint), with ``control`` already folded in.  The caller
            MUST evaluate on the exact accepted state at ``t_lo`` (the loop
            carry), never the one-step interpolant: an accepted DAE state is
            algebraically consistent, so the endpoint condition needs no
            projection. Mid-step candidates evaluate every row through the
            fast interpolant and can bracket algebraic-global crossings.
        cond_hi_fn: Closure ``ev_cond_fn -> scalar`` evaluating an event's
            condition at the step end ``(t_hi, u_hi, lam_hi)`` (discrete check
            and continuous high endpoint), with ``control`` already folded in.
            Same exact-accepted-state requirement as ``cond_lo_fn``.
        interp: One-step state interpolant ``t -> y`` over ``[t_lo, t_hi]``.
        control: Optional fixed control-state pytree.
        control_at: Optional ``t -> control`` resolver used while locating a
            crossing for a time-dependent Recipe control.
        direction_default: Fallback crossing direction when an event does not
            declare one (``"up"`` / ``"down"`` / ``"any"``).
        state_at: Optional ``(event, t) -> y`` evaluator used for every
            continuous root candidate while locating a crossing. The event
            object exposes its static ``state_policy``, so this hook can
            project only events that request projection.  Endpoint conditions
            never use it — they always come from ``cond_lo_fn`` /
            ``cond_hi_fn`` on the exact accepted states, which need no
            projection.  ``None`` root-finds on the interpolant directly.
        state_status: Optional ``(event, t) -> bool`` companion for
            ``state_at``. With [Bisection][diff_soup.timestepping.events.Bisection], a false intermediate
            bracket candidate or located-root status makes the returned
            ``evaluation_ok`` false, allowing the driver to surface
            projection failure instead of accepting an unprojected condition.
            Consulted only on the step whose located arm runs.
        active: Optional bool scalar (the caller's step-accepted flag) gating
            each continuous event's expensive located arm; the returned
            ``fired`` keeps its unmasked semantics.

    Returns:
        tuple: ``(fired, idx, t_star, kind, direction, evaluation_ok)`` where

        * ``fired`` (bool scalar) — any event fired this step (discrete OR
          continuous);
        * ``idx`` (int32 scalar) — declaration-order index of the winning
          event (the earliest-crossing continuous event if one fired, else the
          first discrete event that fired, else ``0``);
        * ``t_star`` (scalar) — the surfaced stop time: the earliest continuous
          crossing if a continuous event fired, otherwise ``t_hi``;
        * ``kind`` (int32 scalar) — winning-event type:
          ``RESOLVE_KIND_CONTINUOUS`` if a continuous crossing fired,
          ``RESOLVE_KIND_DISCRETE`` if (only) a discrete event fired,
          ``RESOLVE_KIND_NONE`` if nothing fired.
        * ``evaluation_ok`` (bool scalar) — all event-specific state
          evaluations needed by this step succeeded. It is always true on
          the legacy fast path.
    """
    # Continuous earliest-crossing accumulation (parity-safe defaults: an
    # unfired step leaves the accepted endpoint untouched -> t_star == t_hi).
    _any_cont_fired = jnp.asarray(False)
    _min_cross_t = jnp.asarray(t_hi)
    cont_idx = jnp.asarray(0, dtype=jnp.int32)
    cont_dir = jnp.asarray(0, dtype=jnp.int32)
    cont_time_tol = jnp.asarray(0.0, dtype=jnp.asarray(t_hi).dtype)
    # Discrete fold (any discrete fire -> stop at the boundary).
    _any_disc_fired = jnp.asarray(False)
    disc_idx = jnp.asarray(0, dtype=jnp.int32)
    _disc_seen = jnp.asarray(False)
    _all_evaluations_ok = jnp.asarray(True)

    for _i, _ev in enumerate(events):
        _i32 = jnp.asarray(_i, dtype=jnp.int32)
        if getattr(_ev, "root_finder", None) is None:
            # Discrete (bool) event — ordered; evaluated at the step end on
            # the exact accepted state (already algebraically consistent).
            event_stop_user = jnp.asarray(cond_hi_fn(_ev.cond_fn), dtype=jnp.bool_)
            # First discrete fire wins the surfaced idx (declaration order).
            _take = jnp.logical_and(event_stop_user, jnp.logical_not(_disc_seen))
            disc_idx = jnp.where(_take, _i32, disc_idx)
            _disc_seen = jnp.logical_or(_disc_seen, event_stop_user)
            _any_disc_fired = jnp.logical_or(_any_disc_fired, event_stop_user)
        else:
            # Continuous (real-valued) event — endpoints from the exact
            # accepted states (consistent; no projection), then the exact
            # crossing located on the (projected) one-step interpolant only
            # when they bracket one; keep the EARLIEST fired t* (SciML
            # earliest-crossing).
            cond_lo = jnp.asarray(cond_lo_fn(_ev.cond_fn)).reshape(())
            cond_hi = jnp.asarray(cond_hi_fn(_ev.cond_fn)).reshape(())
            fired_cont, t_star, cross_dir, evaluation_ok = _locate_crossing(
                _ev.cond_fn,
                interp,
                _ev.root_finder,
                t_lo,
                t_hi,
                cond_lo,
                cond_hi,
                getattr(_ev, "direction", direction_default),
                control=control,
                control_at=control_at,
                event=_ev,
                state_at=state_at,
                state_status=state_status,
                active=active,
            )
            _all_evaluations_ok = jnp.logical_and(_all_evaluations_ok, evaluation_ok)
            _rf_atol = float(getattr(_ev.root_finder, "atol", 0.0))
            _rf_rtol = float(getattr(_ev.root_finder, "rtol", 0.0))
            _candidate_tol = jnp.maximum(
                jnp.asarray(_rf_atol, dtype=jnp.asarray(t_hi).dtype),
                jnp.asarray(_rf_rtol, dtype=jnp.asarray(t_hi).dtype)
                * jnp.abs(jnp.asarray(t_hi) - jnp.asarray(t_lo)),
            )
            _tie_tol = jnp.maximum(cont_time_tol, _candidate_tol)
            _is_tie = jnp.abs(t_star - _min_cross_t) <= _tie_tol
            _strictly_earlier = t_star < (_min_cross_t - _tie_tol)
            _take = jnp.logical_and(
                fired_cont,
                jnp.logical_or(
                    jnp.logical_not(_any_cont_fired),
                    jnp.logical_or(
                        _strictly_earlier,
                        jnp.logical_and(_is_tie, _i32 < cont_idx),
                    ),
                ),
            )
            cont_idx = jnp.where(_take, _i32, cont_idx)
            cont_dir = jnp.where(_take, cross_dir, cont_dir)
            _min_cross_t = jnp.where(_take, t_star, _min_cross_t)
            cont_time_tol = jnp.where(_take, _candidate_tol, cont_time_tol)
            _any_cont_fired = jnp.logical_or(_any_cont_fired, fired_cont)

    fired = jnp.logical_or(_any_cont_fired, _any_disc_fired)
    # A continuous crossing before the boundary precedes every discrete event,
    # which is checked at that boundary. At the same time, declaration order
    # resolves the tie across kinds just as it does within each kind.
    _cont_before_boundary = _min_cross_t < (jnp.asarray(t_hi) - cont_time_tol)
    _cont_tied_at_boundary = jnp.abs(_min_cross_t - jnp.asarray(t_hi)) <= cont_time_tol
    _continuous_wins = jnp.logical_and(
        _any_cont_fired,
        jnp.logical_or(
            jnp.logical_not(_any_disc_fired),
            jnp.logical_or(
                _cont_before_boundary,
                jnp.logical_and(_cont_tied_at_boundary, cont_idx < disc_idx),
            ),
        ),
    )
    idx = jnp.where(_continuous_wins, cont_idx, disc_idx).astype(jnp.int32)
    t_star = jnp.where(_continuous_wins, _min_cross_t, jnp.asarray(t_hi))
    kind = jnp.where(
        _continuous_wins,
        jnp.asarray(RESOLVE_KIND_CONTINUOUS, dtype=jnp.int32),
        jnp.where(
            _any_disc_fired,
            jnp.asarray(RESOLVE_KIND_DISCRETE, dtype=jnp.int32),
            jnp.asarray(RESOLVE_KIND_NONE, dtype=jnp.int32),
        ),
    ).astype(jnp.int32)
    direction = jnp.where(_continuous_wins, cont_dir, jnp.asarray(0, jnp.int32))
    return fired, idx, t_star, kind, direction, _all_evaluations_ok


# ---------------------------------------------------------------------------
# EventSet resolution (SciML CallbackSet semantics)
# ---------------------------------------------------------------------------


def _events_of(event: Any) -> tuple:
    """Normalize a single [Event][diff_soup.timestepping.events.Event] or an [EventSet][diff_soup.timestepping.events.EventSet] to a tuple.

    A bare [Event][diff_soup.timestepping.events.Event] is wrapped as a one-element tuple; an
    [EventSet][diff_soup.timestepping.events.EventSet] (or anything exposing an ``events`` attribute) yields its
    ordered ``events`` tuple.  This is the single normalization used by both the
    core event block and the affect driver so event *ordering* is consistent
    everywhere (the SciML ``CallbackSet`` order is the declaration order).

    Args:
        event: An [Event][diff_soup.timestepping.events.Event], an [EventSet][diff_soup.timestepping.events.EventSet], or ``None``.

    Returns:
        tuple: The ordered tuple of [Event][diff_soup.timestepping.events.Event] objects (empty for ``None``).
    """
    if event is None:
        return ()
    if hasattr(event, "events"):
        return tuple(event.events)
    return (event,)


def _cond_is_discrete(ev: Any) -> bool:
    """Return True iff ``ev`` is a discrete (bool) event.

    Return-type discrimination follows diffrax: an event with **no**
    ``root_finder`` is discrete (its ``cond_fn`` returns a bool checked at the
    step boundary); an event **with** a ``root_finder`` is continuous (its
    real-valued ``cond_fn`` is root-found for the exact crossing).

    Args:
        ev: An [Event][diff_soup.timestepping.events.Event].

    Returns:
        bool: ``True`` for a discrete event, ``False`` for a continuous one.
    """
    return getattr(ev, "root_finder", None) is None


def resolve_event_mask(
    event: Any,
    t,  # noqa: ANN001
    y,  # noqa: ANN001
    args=None,  # noqa: ANN001
    *,
    control: Any = None,
) -> tuple[bool, ...]:
    """Re-evaluate an [Event][diff_soup.timestepping.events.Event] / [EventSet][diff_soup.timestepping.events.EventSet] and report which fired.

    Given the *terminal* solver state ``(t, y)`` surfaced by an ``EVENT_STOP``
    solve, re-evaluates every event's ``cond_fn`` in declaration order and
    returns one bool per event — the diffrax ``Solution.event_mask`` (which
    event fired).

    SciML ``CallbackSet`` resolution semantics are applied:

    * **discrete** events (no ``root_finder``) fire on a truthy ``cond_fn`` and
      are resolved **in declaration order** — the first that fires wins, and any
      later events are masked ``False`` (a single solve stops on one event);
    * **continuous** events (real-valued ``cond_fn`` + ``root_finder``) fire on a
      direction-matching crossing, surfaced here as a non-negative cond value at
      the (root-found) terminal time; among several that fire, the earliest
      crossing is the one the core surfaced as the terminal ``t``, so the
      re-evaluation at that ``t`` flags exactly the winning event.

    Because the core stops at the single winning event, this returns a mask with
    **at most one** ``True``.  It is a pure function of the terminal state — no
    new array leaves on the hot path — so it is computed post-hoc in
    ``_build_solution`` (and mirrored by the affect driver's host loop).

    Args:
        event: The [Event][diff_soup.timestepping.events.Event] or [EventSet][diff_soup.timestepping.events.EventSet] the solve ran with.
        t: Terminal time (scalar).
        y: Terminal state array.
        args: Solver args / parameter pytree passed through to ``cond_fn``.
        control: Optional control-state pytree threaded to ``cond_fn``.

    Returns:
        tuple[bool, ...]: One bool per event (declaration order); ``True`` for
        the single event that fired, ``False`` elsewhere.  Empty tuple when
        ``event`` is ``None``.
    """
    events = _events_of(event)
    if not events:
        return ()
    mask: list[bool] = []
    already_fired = False
    for ev in events:
        if already_fired:
            mask.append(False)
            continue
        c = ev.cond_fn(t, y, args, control=control)
        if _cond_is_discrete(ev):
            fired = bool(jnp.asarray(c))
        else:
            # Continuous: at the surfaced (root-found) terminal time the cond is
            # ~0; a crossing is flagged by a non-negative value (>= a small
            # tolerance band around the located root).
            cval = float(jnp.asarray(c).reshape(()))
            fired = cval >= -1e-9
        mask.append(fired)
        if fired:
            already_fired = True
    return tuple(mask)


# ---------------------------------------------------------------------------
# Event (the unified type)
# ---------------------------------------------------------------------------


class Event(eqx.Module):
    """A condition that stops a solve, or rewrites its state and continues, when it fires.

    Pass it, or an `EventSet`, as `event=` to `pdesolve`. Without a `root_finder` the condition is read as a boolean at the end of each accepted step, and with one its sign change is located inside the step.

    Args:
        cond_fn: `cond_fn(t, y, args, *, control=None)`, returning a boolean, or a real scalar for a continuous event.
        affect: `affect(t, y, args, control)` returning `(y_new, control_new)`, or `None` to stop the solve at the event.
        root_finder: Root finder that locates a continuous crossing, for example `Bisection()`.
        direction: Crossing that fires a continuous event, `"up"` (negative to positive), `"down"`, or `"any"`.
        state_policy: A `DenseOutputPolicy` whose `"projected"` mode projects mid-step states onto the algebraic constraints, and needs a `root_finder`.
    """

    cond_fn: Callable = eqx.field(static=True)
    affect: Optional[Callable] = eqx.field(default=None, static=True)
    root_finder: Optional[Any] = eqx.field(default=None, static=True)
    direction: Literal["up", "down", "any"] = eqx.field(default="any", static=True)
    state_policy: Optional[DenseOutputPolicy] = eqx.field(default=None, static=True)

    def __init__(
        self,
        cond_fn: Callable,
        affect: Optional[Callable] = None,
        root_finder: Optional[Any] = None,
        direction: Literal["up", "down", "any"] = "any",
        state_policy: Any = None,
    ) -> None:
        """Construct and eagerly validate one static event declaration.

        A projected state is meaningful only for a continuous condition whose
        crossing is localized. Discrete boundary checks retain their native
        accepted-state semantics.

        Args:
            cond_fn: Scalar condition callable.
            affect: Optional mutate-and-continue callback.
            root_finder: Root locator for a continuous condition, or ``None``
                for a discrete boundary condition.
            direction: Crossing direction: ``"up"``, ``"down"``, or ``"any"``.
            state_policy: Optional dense-output state policy. Projected mode
                requires ``root_finder``.

        Raises:
            ConfigurationError: If ``state_policy`` is not a
                [DenseOutputPolicy][diff_soup.timestepping.saveat.DenseOutputPolicy], or projected mode is attached to
                a discrete event.
        """
        if state_policy is not None and not isinstance(state_policy, DenseOutputPolicy):
            raise ConfigurationError(
                "Event.state_policy must be a DenseOutputPolicy instance or None; "
                f"got {type(state_policy).__name__}. Construct "
                "DenseOutputPolicy(mode='fast'|'projected') and pass that object."
            )
        if state_policy is not None and state_policy.mode == "projected" and root_finder is None:
            raise ConfigurationError(
                "Event.state_policy with mode='projected' requires a continuous "
                "event with root_finder=... so candidate states can be projected; "
                "add a root finder such as Bisection(), or omit state_policy for "
                "a discrete boundary event."
            )
        self.cond_fn = cond_fn
        self.affect = affect
        self.root_finder = root_finder
        self.direction = direction
        self.state_policy = state_policy


# ---------------------------------------------------------------------------
# EventSet (composer; SciML CallbackSet spirit)
# ---------------------------------------------------------------------------


class EventSet(eqx.Module):
    """Several events checked together in one solve.

    When several fire in one step the earliest wins, and declaration order breaks ties.

    Args:
        events: Tuple of `Event` objects.
    """

    events: tuple = eqx.field(static=True)

    def __init__(self, events: tuple[Event, ...]) -> None:
        """Initialise with a tuple of [Event][diff_soup.timestepping.events.Event] objects.

        Args:
            events: One or more [Event][diff_soup.timestepping.events.Event] instances.
        """
        self.events = tuple(events)


# ---------------------------------------------------------------------------
# EventObservable (thin adapter for scalar functionals)
# ---------------------------------------------------------------------------


class EventObservable(eqx.Module):
    """A scalar function of the state that builds event conditions.

    `crosses` gives a condition for a continuous event, and `below` and `above` give boolean ones.

    Attributes:
        fn: The scalar function `fn(t, y, args)`.
    """

    fn: Callable = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def soc(cls, fn: Callable) -> "EventObservable":
        """Wrap a state-of-charge function, the same as `EventObservable(fn=fn)`.

        Args:
            fn: The function `fn(t, y, args)`.

        Returns:
            An `EventObservable`.
        """
        return cls(fn=fn)

    @classmethod
    def voltage(cls, fn: Callable) -> "EventObservable":
        """Wrap a voltage function, the same as `EventObservable(fn=fn)`.

        Args:
            fn: The function `fn(t, y, args)`.

        Returns:
            An `EventObservable`.
        """
        return cls(fn=fn)

    # ------------------------------------------------------------------
    # Condition-function factories
    # ------------------------------------------------------------------

    def crosses(self, level: float) -> Callable:
        """Return the real-valued condition `fn(t, y, args) - level` for a continuous event.

        Args:
            level: Level whose crossing fires the event.

        Returns:
            A condition for `Event(cond_fn=...)`, which needs a `root_finder`.
        """
        _fn = self.fn
        _level = level

        def _cond(t, y, args, *, control=None, **kw):  # noqa: ANN001, ANN202
            return _fn(t, y, args) - _level

        return _cond

    def below(self, level: float) -> Callable:
        """Return a boolean condition, true while `fn(t, y, args) < level`.

        Args:
            level: Threshold.

        Returns:
            A condition for a discrete `Event`.
        """
        _fn = self.fn
        _level = level

        def _cond(t, y, args, *, control=None, **kw):  # noqa: ANN001, ANN202
            return _fn(t, y, args) < _level

        return _cond

    def above(self, level: float) -> Callable:
        """Return a boolean condition, true while `fn(t, y, args) > level`.

        Args:
            level: Threshold.

        Returns:
            A condition for a discrete `Event`.
        """
        _fn = self.fn
        _level = level

        def _cond(t, y, args, *, control=None, **kw):  # noqa: ANN001, ANN202
            return _fn(t, y, args) > _level

        return _cond


# ---------------------------------------------------------------------------
# terminate_steady_state preset
# ---------------------------------------------------------------------------


def terminate_steady_state(
    rtol: float,
    atol: float,
    norm: Optional[Callable] = None,
) -> Event:
    """Return a terminating event that fires once `norm(f) < atol + rtol * norm(y)`.

    The condition reads `f` from a `residual_fn` keyword, or else from `args`, as a function `args(t, y)`, a sequence starting with one, or the vector itself.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        norm: Norm of a vector, the root mean square when `None`.

    Returns:
        A discrete `Event` with no affect.
    """
    _rtol = float(rtol)
    _atol = float(atol)
    _norm: Callable

    if norm is None:

        def _norm(v):  # noqa: ANN001, ANN202
            """RMS norm."""
            v = jnp.asarray(v)
            return jnp.sqrt(jnp.mean(v**2))
    else:
        _norm = norm

    def _steady_state_cond(
        t,  # noqa: ANN001
        y,  # noqa: ANN001
        args,  # noqa: ANN001
        *,
        residual_fn: Optional[Callable] = None,
        control=None,  # noqa: ANN001
        **kw,  # noqa: ANN003
    ) -> bool:
        """Return True when steady-state criterion is satisfied.

        Args:
            t: Current time (scalar).
            y: Current state array.
            args: Solver args.  When ``residual_fn`` is None, the first
                element of ``args`` is used as the residual callable, or
                ``args`` itself is interpreted as the residual vector.
            residual_fn: Optional explicit residual callable
                ``(t, y, args) -> vector``.  Takes priority over ``args``.
            control: Optional control state (currently unused).
            **kw: Additional keyword arguments (ignored).

        Returns:
            bool: ``True`` iff the steady-state criterion is satisfied.
        """
        if residual_fn is not None:
            f_vec = residual_fn(t, y, args)
        elif callable(args):
            # args is the residual callable
            f_vec = args(t, y)
        elif isinstance(args, (list, tuple)) and len(args) > 0 and callable(args[0]):
            f_vec = args[0](t, y)
        else:
            # Fallback: treat args as the residual vector directly (useful
            # for unit tests where the residual is precomputed).
            f_vec = jnp.asarray(args)

        y_arr = jnp.asarray(y)
        f_arr = jnp.asarray(f_vec)
        threshold = _atol + _rtol * _norm(y_arr)
        return _norm(f_arr) < threshold

    return Event(cond_fn=_steady_state_cond, affect=None, root_finder=None, direction="any")


__all__ = [
    "Event",
    "EventSet",
    "EventObservable",
    "Bisection",
    "root_with_ift",
    "resolve_in_loop",
    "terminate_steady_state",
    "resolve_event_mask",
]
