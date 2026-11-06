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
[diff_soup.utils.weights][].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.linear import SparseTopology
from diff_soup.problem.implicit.consistent_init.pinned import _structure
from diff_soup.timestepping.vsvo.common.backends import analyze_sensitivity, initialization_policy
from diff_soup.utils.weights import error_weight

#: The endpoint clamp shortens a step to just shy of the endpoint by this many rounding
#: errors (ours: a margin of a few ulps so ``t + h`` never lands past ``t1``).
ENDPOINT_CLAMP_FUZZ = 4.0


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


def where_state(cond: jax.Array, a: Any, b: Any) -> Any:
    """Select between two carried states leaf by leaf on a scalar predicate."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


#: Each engine's capability flags, every flag stated.
ENGINE_CAPABILITIES = {
    "ida": EngineCapabilities(
        # The factored iteration matrix persists across attempts and is
        # refactored only where a setup replaces it.
        supports_factor_reuse=True,
        supports_events=True,
        supports_fully_implicit=True,
        supports_global_constraints=True,
        supports_state_dependent_mass=True,
        supports_dense_output=True,
    ),
    "bdf": EngineCapabilities(
        supports_factor_reuse=True,
        supports_events=True,
        # The mass-matrix ODE/DAE form (a correction-form residual built from
        # the mass and right-hand side), not a truly implicit F(t, y, ydot).
        supports_fully_implicit=False,
        # The augmented Schur machinery handles bordered-global rows.
        supports_global_constraints=True,
        # The mass is evaluated as identity, constant, or purely
        # time-dependent; a state-dependent mass is rejected before the solve.
        supports_state_dependent_mass=False,
        supports_dense_output=True,
    ),
    "ndf": EngineCapabilities(
        supports_factor_reuse=True,
        supports_events=True,
        # The mass-matrix ODE/DAE form, not a truly implicit F(t, y, ydot).
        supports_fully_implicit=False,
        supports_global_constraints=True,
        # The diagonal mass is re-evaluated at the current iterate inside the
        # residual; a state-dependent mass is handled in weak (chord) form.
        supports_state_dependent_mass=True,
        supports_dense_output=True,
    ),
}


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


def ic_gate_rtol(rtol: float, ic_rtol: Optional[float], ic_tol: Optional[float]) -> float:
    """The relative part of the consistent-initialization gate.

    ``ic_rtol`` wins when set. Otherwise an explicit absolute ``ic_tol`` keeps
    the gate absolute, and without one the step's ``rtol`` serves: the rule
    [ToleranceSpec][diff_soup.timestepping.tolerances.ToleranceSpec] resolves
    ``ic_rtol`` by.
    """
    if ic_rtol is not None:
        return float(ic_rtol)
    return 0.0 if ic_tol is not None else float(rtol)


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


class ConsistentStart(NamedTuple):
    """The consistent state an engine starts from, with the weights of its first step.

    Attributes:
        u: The state rows.
        lam: The bordered-global rows.
        y: ``u`` and ``lam`` packed.
        ydot: The packed slopes, zero on algebraic rows.
        alg_norm: The residual norm initialization left.
        atol_rows: The per-row absolute tolerance, or ``None`` for a scalar one.
        ewt: The error weights at ``y``.
    """

    u: jax.Array
    lam: jax.Array
    y: jax.Array
    ydot: jax.Array
    alg_norm: jax.Array
    atol_rows: Optional[jax.Array]
    ewt: jax.Array


def initialization_policies(problem: Any, precision: Any) -> tuple[Any, Any, Any]:
    """The layout and the forward and reverse precisions of the initialization factor.

    The projection factors the problem exactly as the corrector does, so it is
    analysed with the lowered problem's placement and the accepted-root
    sensitivity precision the solve declared for each derivative direction,
    never the backend's defaults.
    """
    layout, forward = initialization_policy(problem, precision, mode="forward")
    _, reverse = initialization_policy(problem, precision, mode="reverse")
    return layout, forward, reverse


def project_start(
    engine: Any,
    problem: Any,
    t0: jax.Array,
    u0: jax.Array,
    lam0: jax.Array,
    ydot0: jax.Array,
    theta: Any,
    control: Any,
    policies: tuple[Any, Any, Any],
    *,
    max_iters: Optional[int] = None,
) -> Any:
    """The startup projection: the solver problem's consistent initialization at the engine's gate."""
    layout, precision, reverse_precision = policies
    budget = {} if max_iters is None else {"max_iters": max_iters}
    return problem.consistent_initial_state(
        t0,
        pack_state(u0, lam0),
        theta,
        control,
        ydot0=jnp.concatenate([ydot0, jnp.zeros(lam0.shape, dtype=ydot0.dtype)]),
        tol=ic_gate_tolerance(engine.atol, engine.ic_tol),
        rtol=ic_gate_rtol(engine.rtol, engine.ic_rtol, engine.ic_tol),
        backend=engine.backend,
        layout=layout,
        precision=precision,
        reverse_precision=reverse_precision,
        **budget,
    )


def consistent_start(engine: Any, cfg: Any, ic: Any, n_u: int, dtype: Any) -> ConsistentStart:
    """Unpack an initialization result into the state and weights an engine starts from.

    A failed root is NaN-poisoned (see
    [consume_consistent_init][diff_soup.timestepping.vsvo.common.contract.consume_consistent_init]).
    The slopes cover every differential row, the differential globals of the
    bordered partition included, and are consumed whole. A per-row absolute
    tolerance arrives as a vector on the driver configuration and rides the
    carry, so accepted-step weight refreshes keep consuming it.
    """
    ic_y, ic_ydot = consume_consistent_init(ic)
    u = jnp.asarray(ic_y[:n_u], dtype=dtype)
    lam = jnp.asarray(ic_y[n_u:], dtype=dtype)
    y = pack_state(u, lam)
    atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
    return ConsistentStart(
        u=u,
        lam=lam,
        y=y,
        ydot=jnp.asarray(ic_ydot, dtype=dtype),
        alg_norm=jnp.asarray(ic.residual_norm, dtype=dtype),
        atol_rows=atol_rows,
        ewt=error_weight(y, engine.rtol, engine.atol if atol_rows is None else atol_rows),
    )


def refuse_unprojected_start(problem: Any, skip: bool, refused: str) -> None:
    """Refuse to start a problem with algebraic rows from a state that is not projected.

    Args:
        problem: The solver problem.
        skip: Whether this solve skips the startup projection.
        refused: The options that skip it, with their verb, as the message names them.

    Raises:
        ConfigurationError: If ``skip`` holds and the problem has algebraic rows or
            bordered globals.
    """
    if skip and _structure(problem)[4]:
        raise ConfigurationError(
            f"this problem has algebraic rows; {refused} not supported for it. "
            "Enable consistent initialization."
        )


def sensitivity_analyses(engine: Any, problem: Any, backend: Any, layout: Any) -> tuple[Any, Any]:
    """The forward and reverse analyses of the accepted root's sensitivity solves.

    The per-step root derivative factors the full-system matrix at the accepted
    root exactly, the tangent solve the matrix and a reverse sweep its transpose,
    each under the precision policy of its direction, through the engine's own
    backend; a dense-mode plan hands them the full dense pattern. The symbolic
    analysis is problem-static.
    """
    topology = problem.full_topology
    if topology is None:
        topology = SparseTopology.full(int(problem.n))
    forward, reverse = (
        analyze_sensitivity(backend, topology, layout=layout, precision=engine.precision, mode=mode)
        for mode in ("forward", "reverse")
    )
    return forward, reverse


def problem_layout(problem: Any) -> Layout:
    """The device placement of every analysis of a solve: the problem's lowered layout.

    The lowering-declared layout places each corrector rung's factor and the
    sensitivity pattern alike, so a multi-GPU factorization request reaches the
    adjoint's transpose factor too; a problem without one takes the default.
    """
    layout = getattr(problem, "layout", None)
    return Layout() if layout is None else layout


def clamp_initial_step(
    h: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    hmax_inv: jax.Array,
    hmin: jax.Array,
    uround: Any,
    step: Any,
    dtype: Any,
) -> jax.Array:
    """Clamp a first step to ``hmax`` and ``hmin``, then short of ``t1``, or replay a recorded one.

    Args:
        h: The proposed first step.
        t0: The start time.
        t1: The end time.
        hmax_inv: ``1 / hmax``, or zero for no maximum.
        hmin: The minimum step magnitude.
        uround: The unit roundoff of ``dtype``.
        step: ``None``, or ``(replay, recorded)``: a replayed restart enters with the
            step its forward sweep recorded where ``replay`` holds.
        dtype: The state dtype.

    Returns:
        The first step.
    """
    rh = jnp.abs(h) * hmax_inv
    h = jnp.where(rh > 1.0, h / jnp.where(rh > 1.0, rh, 1.0), h)
    at_min = jnp.abs(h) < hmin
    h = jnp.where(at_min, h * (hmin / jnp.where(at_min, jnp.abs(h), 1.0)), h)
    overshoot = (t0 + h - t1) * h > 0
    h = jnp.where(overshoot, (t1 - t0) * (1.0 - ENDPOINT_CLAMP_FUZZ * uround), h)
    if step is not None:
        replay, recorded = step
        recorded = jax.lax.stop_gradient(jnp.asarray(recorded).astype(dtype))
        h = jnp.where(replay, recorded, h)
    return h


__all__ = [
    "ENDPOINT_CLAMP_FUZZ",
    "ENGINE_CAPABILITIES",
    "ConsistentStart",
    "clamp_initial_step",
    "EngineCapabilities",
    "coerce_problem",
    "consistent_start",
    "initialization_policies",
    "pack_state",
    "problem_layout",
    "project_start",
    "refuse_unprojected_start",
    "register_problem_coercion",
    "sensitivity_analyses",
    "unpack_state",
    "validate_requirements",
    "where_state",
]
