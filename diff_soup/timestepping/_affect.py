"""apply_affect — SciML affect! over a static Event tuple.

Provides a single function
[apply_affect][diff_soup.timestepping._affect.apply_affect] that dispatches the
fired event's affect callable (or terminate sentinel) via ``jax.lax.switch``
over the static event tuple.  The branch index ``idx`` is a traced integer
(array scalar), so the dispatch is fully JIT- and vmap-compatible.

Design:
  - ``terminate = True`` iff the fired event's ``affect`` is ``None`` (the
    diffrax/terminate arm).  This is the unified in-loop discriminant: the
    caller checks ``terminate`` to decide whether to stop or continue.
  - All branches return the same pytree shape ``(u', control', terminate)``
    so ``lax.switch`` can lower to a single XLA ``conditional``.
  - The function is **stepper-agnostic**: it only touches ``(t, u, args,
    control)`` — the per-stepper ``reinit_after_affect`` hook (the
    variable-step engines) is *not* called here.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def apply_affect(
    events: tuple,
    idx: jax.Array,
    t_star: jax.Array,
    u_star: jax.Array,
    args: Any,
    control: Any,
    normalize_control=None,
) -> tuple:
    """Apply the fired event's affect (or set terminate=True if affect=None).

    Dispatches via ``jax.lax.switch`` over the *static* ``events`` tuple,
    using the *traced* integer ``idx`` to select the fired branch.  All
    branches share the pytree structure ``(u', control', terminate)`` so the
    switch lowers correctly to a single XLA ``conditional``.

    Args:
        events: Static tuple of
            [Event][diff_soup.timestepping.events.Event]
            objects (the full declaration-order set from an
            [EventSet][diff_soup.timestepping.events.EventSet]).
        idx: Traced integer scalar — the index of the fired event (0-based,
            declaration order).  Typically the output of ``resolve_in_loop``
            or the JNDF event block.
        t_star: Scalar crossing time (or accepted-step end for discrete events).
        u_star: State at the crossing time (output of the one-step interpolant
            for continuous events, or the accepted step end for discrete).
        args: Solver args / parameter pytree threaded to the affect callable.
        control: Current ``control_state`` carry value.
        normalize_control: Optional eager/static normalizer applied to every
            affected control payload before branches rejoin.

    Returns:
        tuple: ``(u_prime, control_prime, terminate)`` where

        * ``u_prime`` — possibly mutated state (same shape as ``u_star``).
        * ``control_prime`` — possibly mutated control (same pytree shape as
          ``control``).
        * ``terminate`` — bool scalar; ``True`` iff the fired event has
          ``affect=None`` (i.e. the event is terminate-only).
    """

    # Build one branch callable per event.  Each branch captures its event
    # index at Python (trace) time via a default argument, so the closure is
    # correct when the list is handed to lax.switch.
    def _branch(i: int) -> tuple:  # noqa: ANN001
        e = events[i]
        if e.affect is None:
            # Terminate arm: return state/control unchanged, flip terminate.
            return (u_star, control, jnp.asarray(True))
        u_p, c_p = e.affect(t_star, u_star, args, control)
        if normalize_control is not None:
            c_p = normalize_control(c_p)
        return (u_p, c_p, jnp.asarray(False))

    branches = [(lambda _i=i: _branch(_i)) for i in range(len(events))]
    return jax.lax.switch(idx, branches)
