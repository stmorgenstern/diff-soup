"""The solver contract: the problem surface, engine capabilities, validation.

This module declares what each timestepper supports
([EngineCapabilities]
[diff_soup.timestepping.vsvo.common.contract.EngineCapabilities]),
and a fail-loud pre-solve check
([validate_requirements]
[diff_soup.timestepping.vsvo.common.contract.validate_requirements])
that rejects a problem an engine cannot solve
*before* the solve begins -- a wrapper's limitation is an error message, never a
silently wrong answer.

The augmented state is ``x = [u; lam]`` of length ``n_u + n_lam``: ``u`` are the
differential/local-algebraic unknowns and ``lam`` the bordered global unknowns.
[pack_state][diff_soup.timestepping.vsvo.common.contract.pack_state] and
[unpack_state][diff_soup.timestepping.vsvo.common.contract.unpack_state] move
between the split
and concatenated views. Tolerances live on this (problem) side; the error-weight
*law* is engine-owned and lives in
[diff_soup.timestepping.vsvo.common.weights][].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Augmented-state helpers
# ---------------------------------------------------------------------------
def pack_state(u: jax.Array, lam: jax.Array) -> jax.Array:
    """Concatenate ``(u, lam)`` into the augmented state ``x = [u; lam]``."""
    return jnp.concatenate((u, lam), axis=0)


def unpack_state(x: jax.Array, n_u: int) -> tuple[jax.Array, jax.Array]:
    """Split the augmented state ``x`` back into ``(u, lam)`` at ``n_u``."""
    return x[:n_u], x[n_u:]


# ---------------------------------------------------------------------------
# EngineCapabilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineCapabilities:
    """What a timestepper supports. No defaults: callers state every flag."""

    supports_factor_reuse: bool
    supports_events: bool
    supports_fully_implicit: bool
    supports_global_constraints: bool
    supports_state_dependent_mass: bool
    supports_dense_output: bool


def ic_gate_tolerance(atol: Any, ic_tol: Optional[float]) -> float:
    """The scalar residual gate consistent initialization solves to.

    ``ic_tol`` (the explicitly resolved initialization tolerance) wins when
    set. Otherwise the engine's absolute tolerance serves double duty — valid
    only while it is a scalar. A per-row absolute tolerance without an
    explicit initialization tolerance is rejected: silently collapsing the
    vector here would hide which rows governed the gate.
    """
    if ic_tol is not None:
        return float(ic_tol)
    # Python scalars stay off the JAX path entirely: wrapping them in an
    # array inside an active trace would turn the constant into a tracer.
    if isinstance(atol, (int, float)):
        return float(atol)
    if getattr(atol, "ndim", 1) == 0:
        return float(atol)
    raise ValueError(
        "consistent initialization needs a scalar residual gate, but the "
        "engine carries a per-row absolute tolerance and no explicit "
        "initialization tolerance. Resolve the solve's tolerances (which "
        "sets the initialization gate), or pass a scalar atol."
    )


def closure_moved(
    problem: Any,
    t: Any,
    state: jax.Array,
    differential: jax.Array,
    theta: Any,
    control: Any,
    previous_control: Any,
    gate: float,
) -> jax.Array:
    """Whether a change of control moves the algebraic residual at ``state``.

    The algebraic rows of ``problem.residual`` at ``(t, state)`` under
    ``previous_control`` and under ``control`` differ, in the 2-norm, by more
    than ``gate``: the switch changes the closure the algebraic unknowns
    satisfy, so a history taken under the previous control does not continue
    smoothly. ``gate`` is the absolute residual gate the consistent
    initialization solves to (``ic_gate_tolerance``: the solve's ``ic_tol``,
    else its scalar ``atol``), in the units of the residual: a change below it
    leaves a state that initialization would accept as consistent. The
    algebraic rows do not read the state's derivative, which is passed as
    zeros. A decision: it carries no derivative.

    An operational row is an observation minus its target. Under one mode the
    difference is the change of the target itself, whatever the state and the
    parameters, so a perturbation of the parameters (a derivative stencil in
    them, say) never moves the decision; a change of one segment's target
    moves it as soon as the change exceeds ``gate``.

    Args:
        problem: The solver problem; its ``residual(t, y, ydot, theta, control)``.
        t: The time of the switch.
        state: The augmented state ``[u; lam]`` at the switch.
        differential: The differential-row mask over the augmented state
            (``1`` differential, ``0`` algebraic).
        theta: The parameters.
        control: The control after the switch.
        previous_control: The control before the switch.
        gate: The residual gate, the consistent initialization's
            (``ic_gate_tolerance``).

    Returns:
        A boolean scalar.
    """
    state = jax.lax.stop_gradient(state)
    t = jax.lax.stop_gradient(jnp.asarray(t, dtype=state.dtype))
    slope = jnp.zeros_like(state)
    change = problem.residual(t, state, slope, theta, control) - problem.residual(
        t, state, slope, theta, previous_control
    )
    algebraic = 1.0 - jnp.asarray(differential, dtype=state.dtype)
    return jnp.linalg.norm(jax.lax.stop_gradient(algebraic * change)) > gate


def consume_consistent_init(result) -> tuple[jax.Array, jax.Array]:
    """Return a converged initialization state, NaN-poisoning a failed root.

    ``consistent_initial_state`` deliberately reports its last Newton iterate
    for diagnostics when it fails. Engines and output projection must not
    mistake that off-root iterate for a valid primal state, so every consumer
    passes through this gate before using ``y`` or ``ydot``.
    """
    y = jnp.asarray(result.y)
    ydot = jnp.asarray(result.ydot)
    converged = jnp.asarray(result.converged)
    return (
        jnp.where(converged, y, jnp.full_like(y, jnp.nan)),
        jnp.where(converged, ydot, jnp.full_like(ydot, jnp.nan)),
    )


def fy_reuse_window(plan, engine_window: int) -> int:
    """Intersect an engine's F_y reuse window with the problem declaration."""
    engine_window = int(engine_window)
    declared = getattr(plan, "max_fy_reuse_steps", None)
    return engine_window if declared is None else min(engine_window, int(declared))


# ---------------------------------------------------------------------------
# Pre-solve validation
# ---------------------------------------------------------------------------
def validate_requirements(problem, caps: EngineCapabilities) -> None:
    """Reject a problem the selected timestepper cannot solve, before solving.

    Reads a minimal attribute set off ``problem`` -- ``mass_dependence``
    (default ``"identity"``), ``n_lam`` (default 0) and ``events`` (default
    ``None``) -- and raises `ValueError` naming the violated requirement
    and what would satisfy it.
    """
    # A lowered problem that is fully implicit — a residual nonlinear in the
    # state derivative, recognizable by the absence of a mass operator —
    # needs a fully-implicit engine. A solver problem in mass form is
    # consumed by every engine directly: startup, correction, and
    # differentiation all run against it directly.
    model = getattr(problem, "problem", None)
    fully_implicit = model is not None and not hasattr(model, "mass")
    if fully_implicit and not caps.supports_fully_implicit:
        raise ValueError(
            "the problem lowers to a fully implicit residual but the selected "
            "timestepper solves the mass form. Use JIDA (a fully-implicit "
            "engine), or restate the equations linearly in the state "
            "derivative."
        )
    md = getattr(problem, "mass_dependence", "identity")
    if md in ("state_weak", "state_strong") and not caps.supports_state_dependent_mass:
        raise ValueError(
            f"problem declares state-dependent mass (mass_dependence={md!r}) but the "
            f"selected timestepper evaluates only identity, constant, or time-dependent "
            f"mass. Use JIDA or JNDF (which evaluate a state-dependent mass), or declare "
            f"mass_dependence in {{'identity', 'constant', 'time'}}."
        )
    n_lam = int(getattr(problem, "n_lam", 0) or 0)
    if n_lam > 0 and not caps.supports_global_constraints:
        raise ValueError(
            f"problem has {n_lam} bordered-global row(s) but the selected timestepper "
            f"does not solve bordered globals. Use an engine whose capabilities include "
            f"supports_global_constraints, or remove the global variables."
        )
    if getattr(problem, "events", None) and not caps.supports_events:
        raise ValueError(
            "problem defines events but the selected timestepper does not handle events. "
            "Use an engine whose capabilities include supports_events, or remove the events."
        )


# ---------------------------------------------------------------------------
# Problem-coercion injection seam
# ---------------------------------------------------------------------------
# The driver accepts either a problem that already satisfies the solver contract
# (it exposes the parameter accessor ``default_theta``) or a higher-level
# problem object that has to be wrapped in a solver adapter first. That wrapping
# belongs to the problem layer, which this layer must not import. The problem
# layer registers its coercion here at its own import time; until it does,
# ``coerce_problem`` is the identity -- which is already correct for any problem
# that satisfies the contract.
_PROBLEM_COERCION: Optional[Callable[[Any], Any]] = None


def register_problem_coercion(coercion: Optional[Callable[[Any], Any]]) -> None:
    """Register the coercion the driver applies to a problem before solving.

    The problem layer calls this at import time so a direct ``solve`` accepts
    the same problem objects the top-level solve entry point does. Pass ``None``
    to clear the registered coercion (the test / session-reset hook).

    Args:
        coercion: ``coerce(problem) -> problem`` mapping any accepted problem to
            one that satisfies the solver contract, or ``None`` to clear it.
    """
    global _PROBLEM_COERCION
    _PROBLEM_COERCION = coercion


def coerce_problem(problem: Any) -> Any:
    """Return a problem that satisfies the solver contract.

    Applies the registered coercion, if any. With none registered this is the
    identity, which is correct for a problem that already satisfies the contract
    -- the wrap is only needed for a higher-level problem object, and having
    imported the layer that defines those objects is what registers the
    coercion in the first place.
    """
    if _PROBLEM_COERCION is None:
        return problem
    return _PROBLEM_COERCION(problem)


__all__ = [
    "EngineCapabilities",
    "validate_requirements",
    "register_problem_coercion",
    "coerce_problem",
    "pack_state",
    "unpack_state",
]
