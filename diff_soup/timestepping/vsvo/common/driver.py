"""The shared solve-loop driver for the scalar VSVO solver suite.

One driver runs the time loop for every engine in the suite. The engine kernels
own the numerics of a step (history, coefficients, predictor, corrector, error
estimate, accept/reject, order and step selection, failure recovery); the driver
owns everything around the step that is the same for all of them:

* coercing the problem to the solver contract and resolving the parameters,
* validating the engine's capabilities against the problem's requirements and
  failing loud *before* the loop,
* building the explicit-save grid and routing dense output to it,
* running a bounded step loop with termination mapped to a status code,
* stopping cleanly and reporting the achieved tolerance scale factor when the
  requested accuracy exceeds what the current solution's magnitude can
  resolve,
* accumulating the shared statistics and, when asked, the per-step trace,
* selecting the differentiation strategy at an explicit, control-dead seam,
* and packaging the shared `Solution`.

The engine is a black box behind the `Engine` protocol: the driver reads
the `StepReport` each attempt returns, plus a handful of fields it duck-
types off the carry the same way (the initial-condition residual norm, and the
accuracy-limit guard's error-weight vector, ``ewt`` -- present on every engine
in the suite, absent and so a no-op on one that carries no such field) --
never anything else about the engine's internal state. The loop is bounded
with statically sized buffers, so it compiles to a fixed-shape program; the
adaptive controls (step, order, accept/reject, counters) are integer/boolean
state the driver never differentiates.

The differentiation strategy selects the loop primitive. The default
forward-only strategy runs a true early-exit ``jax.lax.while_loop``: primal
solves and forward-mode (JVP) differentiation in O(1) memory. Passing
``adjoint=CheckpointAdjoint()`` runs a checkpointed replay loop instead, whose
reverse rule makes ``jax.grad`` / VJP work through the whole accepted
trajectory.

An event with a mutate-and-continue affect switches the solve instead of
stopping it: the driver locates the crossing on the dense interpolant, applies
the affect to the state and the control channel, and restarts the engine at
the switch time through the same ``init_carry`` entry used at ``t0``. Both
differentiation modes see the switch, and a root-found (continuous) crossing
time carries its exact implicit-function-theorem derivative, so gradients flow
through the switch time as well as the states around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._adjoints import ForwardAdjoint
from diff_soup._errors import ConfigurationError
from diff_soup._precision import Precision, cast_to_compute_dtype
from diff_soup.problem.implicit.solver_problem import solver_view
from diff_soup.timestepping.control_state import ControlState
from diff_soup.timestepping.saveat import _build_save_plan, default_output_plan, output_emitter
from diff_soup.timestepping.vsvo.common._driver_ad import _composable_solve, _forward_loop
from diff_soup.timestepping.vsvo.common._driver_finalize import _finalize
from diff_soup.timestepping.vsvo.common._driver_loop import _loop_fns
from diff_soup.timestepping.vsvo.common._driver_state import _loop_init

from ..solution import (
    SavedOutputs,
    Solution,
    SolveHistory,
)
from .backends import initialization_policy
from .checkpointed_loop import checkpointed_while_loop
from .contract import (
    consume_consistent_init,
    ic_gate_rtol,
    ic_gate_tolerance,
    validate_requirements,
)
from .decisions import init_decision_tape
from .trace import (
    finalize_trace,
)


@dataclass(frozen=True)
class DriverConfig:
    """The solve configuration the driver reads.

    The field names match the legacy scalar-solver ``solve`` signature so a
    per-class solver facade passes its arguments through unchanged. The step
    budget and step/order bounds default to the legacy solver defaults.

    Attributes:
        max_steps: Maximum step attempts (the loop bound and buffer size).
        max_order: Maximum integration order (at most 5).
        max_events: Static capacity of the bounded event log. Additional fires
            set ``event_log.overflow`` and leave recorded entries intact.
        trace: Retain a per-step trace when ``True`` (zero cost otherwise).
        adjoint: The differentiation strategy. ``None`` (the default) selects
            the forward-only strategy: an early-exit loop supporting primal
            solves and forward-mode JVP. A strategy declaring online
            checkpointing (``CheckpointAdjoint``) selects the checkpointed
            replay loop, which supports reverse-mode VJP through the solve.
        hmin: Lower step-size bound.
        hmax: Upper step-size bound (``0`` auto-resolves in the engine).
        dt0: Initial step override (``0`` lets the engine estimate it).
        atol: Absolute tolerance input, or ``None`` to defer to the problem.
        rtol: Relative tolerance input, or ``None`` to defer to the problem.
        lam_atol: Absolute tolerance for the multipliers, or ``None``.
        lam_rtol: Relative tolerance for the multipliers, or ``None``.
        event: Event or event set to monitor, or ``None``.
        control_state0: Initial operating-protocol pytree, or ``None``. On an
            affect-bearing event solve the control channel defaults to the
            parameter pytree when this is ``None`` — an affect that rewrites
            its control then rewrites the post-switch dynamics; an explicit
            value feeds the event conditions only. A parameter-folded channel
            has no control epoch; its samples use epoch zero while the affect's
            engine restart independently resets numerical caches.
        consistent_ic: Project the initial state onto the constraint manifold.
        theta: Parameter pytree to integrate against, or ``None`` to use the
            problem's default parameters.
        save_every: Accepted-step output stride, or ``0`` for an explicit
            query grid. This is populated from the shared save plan.
        save_initial: Whether accepted-step output starts with the state at
            ``t0``. This is populated from the shared save plan.
        history: ``None`` (no history), ``True`` (start afresh and return
            the record a later solve continues from as ``Solution.history``),
            or a [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
            from an earlier solve to continue from (and return the new one).
        keep_history: Whether the solve returns ``Solution.history``. Set by
            ``run_engine`` from ``history``; the record itself enters the
            compiled solve as an argument.
        outputs: The [OutputPlan][diff_soup.timestepping.saveat.OutputPlan]
            naming what is stored at every save time, or ``None`` for the
            default (the state and the bordered globals). Set by
            ``run_engine`` from the save plan.
    """

    max_steps: int = 10_000
    max_order: int = 5
    trace: bool = False
    adjoint: Any = None
    hmin: float = 0.0
    hmax: float = 0.0
    dt0: Any = 0.0
    atol: Any = None
    rtol: Any = None
    lam_atol: Any = None
    lam_rtol: Any = None
    event: Any = None
    control_state0: Any = None
    consistent_ic: bool = True
    theta: Any = None
    max_events: int = 16
    dense: bool = False
    epoch_projected_output: bool = False
    save_every: int = 0
    save_initial: bool = False
    history: Any = None
    keep_history: bool = False
    outputs: Any = None


def _history_request(history: Any, engine: Any) -> tuple:
    """``(record, keep)`` for a solve's ``history=`` value.

    ``record`` is the [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
    to continue from, or ``None``; ``keep`` says whether the solve returns
    ``Solution.history``.

    Raises:
        TypeError: If ``history`` is not ``None``, ``True`` or a ``SolveHistory``.
        ConfigurationError: If ``history`` asks for a record and ``engine``
            cannot keep one.
    """
    if history is None or history is False:
        return None, False
    if history is not True and not isinstance(history, SolveHistory):
        raise TypeError(
            "history must be None, True, or the SolveHistory of an earlier solve "
            f"(its Solution.history); got {type(history).__name__}."
        )
    if not callable(getattr(engine, "history_at_exit", None)):
        raise ConfigurationError(
            f"{type(engine).__name__} keeps no history a later solve can continue from; "
            "pass history=None."
        )
    if history is True:
        return None, True
    return (
        jax.tree_util.tree_map(
            lambda leaf: jnp.asarray(leaf) if eqx.is_array_like(leaf) else leaf, history
        ),
        True,
    )


def _select_adjoint(adjoint: Any) -> Any:
    """Resolve the differentiation strategy at an explicit, control-dead seam.

    ``None`` selects the default forward-only strategy: a true early-exit
    ``jax.lax.while_loop`` supporting primal solves and forward-mode (JVP)
    differentiation in O(1) memory. A strategy object routes by its declared
    predicates — ``is_forward_only`` keeps the early-exit loop;
    ``uses_online_checkpoints`` (``CheckpointAdjoint``) runs the checkpointed
    replay loop whose reverse rule supports ``jax.grad`` / VJP through the
    whole solve. The choice is a static configuration value, never a
    differentiated quantity.
    """
    if adjoint is None:
        return ForwardAdjoint()
    if getattr(adjoint, "is_forward_only", False) or getattr(
        adjoint, "uses_online_checkpoints", False
    ):
        return adjoint
    raise TypeError(
        "the supplied adjoint does not declare a loop strategy this solver can "
        "run: it must declare is_forward_only=True (forward mode, e.g. "
        "ForwardAdjoint) or uses_online_checkpoints=True (reverse mode, e.g. "
        f"CheckpointAdjoint); got {type(adjoint).__name__}."
    )


def _loop_buffers(cfg, record_decisions=False, outputs_buffered=False):
    """The write-only buffer selector for the checkpointed loop, or ``None``."""
    fields = []
    if cfg.trace:
        fields.append(lambda s: s.trace)
    if cfg.dense:
        fields.append(lambda s: s.dense_nodes)
    if record_decisions:
        fields.append(lambda s: s.decisions)
    if outputs_buffered:
        fields.append(lambda s: s.outputs)
    if not fields:
        return None
    if len(fields) == 1:
        return fields[0]
    return lambda s: tuple(field(s) for field in fields)


def _event_has_affect(event: Any) -> bool:
    """Return ``True`` iff ``event`` carries a mutate-and-continue affect.

    Discriminates an affect-bearing event (one that rewrites the carry and
    continues) from a terminate-only event (``affect is None``). Handles a bare
    event, an event set (any member with an affect), and ``None``. Duck-typed so
    the driver never imports the event or top-level solve module.
    """
    if event is None:
        return False
    members = getattr(event, "events", None)
    if members is not None:
        return any(getattr(ev, "affect", None) is not None for ev in members)
    return getattr(event, "affect", None) is not None


def run_engine(engine, problem, t0, t1, y0, saveat, cfg: DriverConfig) -> Solution:
    """Drive ``engine`` over ``[t0, t1]`` and return the shared solution.

    Owns coercion, capability validation, save-grid and dense-output routing,
    the bounded step loop with termination mapping, statistics and optional
    trace accumulation, differentiation-strategy selection, and static buffer
    sizing.

    Args:
        engine: An `Engine`-conforming step kernel.
        problem: The problem (or built operator) to integrate.
        t0: Integration start time.
        t1: Integration end time.
        y0: Initial augmented state ``[u; lam]``.
        saveat: The output policy (a save grid, endpoints, or every step).
        cfg: The `DriverConfig`.

    Returns:
        The shared `Solution`.

    Raises:
        ValueError: If the engine cannot solve the problem (a capability the
            problem needs that the engine does not provide).
        ConfigurationError: If ``cfg.max_events`` is not a positive event-log
            capacity.
    """
    if int(cfg.max_events) < 1:
        raise ConfigurationError(
            "max_events must be at least 1 so the bounded event log has "
            f"capacity for a fired event; got {cfg.max_events!r}."
        )
    problem = solver_view(problem)
    # A coerced problem may carry a user-to-internal state translation (unit
    # rescale / pytree flattening) that the entry owns: the engines and their
    # consistent-initialization iterate in the problem's internal units. Raw
    # contract problems carry no such hook and pass through unchanged.
    # The resolved precision owns every dtype of the solve: the initial state
    # and the floating parameter leaves are cast to its compute dtype here, in
    # both directions, so an input's own dtype never decides the precision.
    _precision = getattr(engine, "precision", None)
    if _precision is not None:
        y0 = cast_to_compute_dtype(y0, _precision, what="initial state")
    if hasattr(problem, "to_internal_state"):
        y0 = problem.to_internal_state(y0)
    if _precision is not None:
        y0 = cast_to_compute_dtype(y0, _precision, what="initial state")

    # Fail loud before any solving: a capability the problem needs but the
    # engine does not provide is an error naming the requirement and a
    # satisfying engine, never a silently wrong answer.
    validate_requirements(problem, engine.capabilities)
    # A lowered problem's Layout is the single source of truth for device
    # placement; a layout the resolved backend cannot serve fails here,
    # before the step loop, instead of being silently ignored.
    _layout = getattr(problem, "layout", None)
    _backend0 = getattr(engine, "backend", None)
    if _layout is not None and _backend0 is not None:
        _backend0._validate_capabilities(_layout, Precision() if _precision is None else _precision)
    if cfg.event is not None and not engine.capabilities.supports_events:
        raise ValueError(
            "an event was supplied but the selected timestepper does not handle "
            "events. Use an engine whose capabilities include supports_events, or "
            "remove the event."
        )
    adjoint = _select_adjoint(cfg.adjoint)
    # The composable forward-sensitivity mode runs its own solve core.
    composable = bool(getattr(adjoint, "composable", False)) and getattr(
        adjoint, "is_forward_only", False
    )
    # A history to continue from is runtime data: it leaves the static
    # configuration and enters the compiled solve as an argument, so solves
    # that continue and solves that start afresh share one program.
    history_in, keep_history = _history_request(cfg.history, engine)

    theta = cfg.theta if cfg.theta is not None else problem.default_theta()
    # Numeric parameter leaves ride the traced side of the compile cache, so a
    # solve with new parameter VALUES reuses the compiled program; non-numeric
    # leaves stay static.
    theta = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf) if eqx.is_array_like(leaf) else leaf, theta
    )
    if _precision is not None:
        theta = cast_to_compute_dtype(theta, _precision, what="parameters")

    # Mutate-and-continue affect support is resolved statically. An affect
    # solve threads a control channel through the loop carry and restarts the
    # engine at each switch. With no explicit ``control_state0`` the control
    # channel IS the parameter pytree, so an affect that rewrites its control
    # rewrites the post-switch dynamics; an explicit ``control_state0`` feeds
    # the event conditions only and leaves the dynamics parameters unchanged.
    _has_affect = _event_has_affect(cfg.event)
    _operational_controls = bool(getattr(problem, "has_operational_control", False))
    _recipe_controls = bool(getattr(problem, "has_recipe_controls", False))
    _recipe_controls = _recipe_controls or _operational_controls
    _dense = bool(getattr(saveat, "dense", False))
    if _dense and _has_affect:
        raise ConfigurationError(
            "dense output is not supported together with a mutate-and-continue "
            "event: the affect jumps the state, and a single smooth "
            "interpolant cannot represent the discontinuity. Sample the two "
            "segments with explicit save times, or split the solve at the "
            "switch."
        )
    _output_policy = getattr(saveat, "output", None)
    _epoch_projected_output = bool(
        _output_policy is not None and _output_policy.mode == "projected" and _has_affect
    )
    x0 = jnp.asarray(y0)
    dtype = x0.dtype
    t0_arr = jnp.asarray(t0, dtype)
    t1_arr = jnp.asarray(t1, dtype)

    # A Recipe's declared controls always occupy the explicit control channel.
    # A ``None`` value means the authored callable is evaluated at each model
    # time; an explicit mapping is normalized once here. Both forms carry an
    # epoch so every affect is a semantic cache-generation boundary.
    _control_folds_theta = _has_affect and cfg.control_state0 is None and not _recipe_controls
    if _operational_controls:
        control0 = problem.normalize_control(cfg.control_state0, dtype=dtype)
        n_u = int(problem.n_u)
        lam0 = problem.operational_seed_multipliers(t0_arr, x0[:n_u], x0[n_u:], theta, control0)
        x0 = jnp.concatenate([x0[:n_u], lam0])
    elif _recipe_controls:
        values0 = (
            None if cfg.control_state0 is None else problem.normalize_control(cfg.control_state0)
        )
        control0 = ControlState.initial(values0)
    elif cfg.control_state0 is not None:
        control0 = jax.tree_util.tree_map(jnp.asarray, cfg.control_state0)
    elif _has_affect:
        control0 = theta
    else:
        control0 = None

    # One shared plan owns the sorted, deduplicated query times and the
    # structural terminal reserve. The output-time buffer starts empty and is
    # filled as its aligned state rows become valid.
    save_plan = _build_save_plan(
        saveat,
        t0,
        t1,
        dtype,
        event=cfg.event,
        max_steps=cfg.max_steps,
        problem=problem,
    )
    # Projection after the loop acts on the stored state, so a solve that
    # projects its samples there stores the state and globals in the loop and
    # evaluates its named outputs from the projected samples.
    _project_after_loop = (
        _output_policy is not None
        and _output_policy.mode == "projected"
        and not _epoch_projected_output
        and not bool(save_plan.step_every)
    )
    output_plan = save_plan.outputs
    loop_output_plan = output_plan
    if _project_after_loop and output_plan.requested:
        loop_output_plan = default_output_plan(int(problem.n_u), int(problem.n_lam))
    if save_plan.step_every and not getattr(adjoint, "is_forward_only", False):
        raise ConfigurationError(
            "SaveAt.steps is not supported with CheckpointAdjoint because the "
            "step-budget-sized differentiable trajectory would multiply "
            "checkpoint memory. Use an undifferentiated solve or forward-mode "
            "differentiation, or select an explicit SaveAt.ts grid for reverse mode."
        )
    ts_grid = save_plan.query_ts
    ts_output = save_plan.output_ts

    # Everything from here on runs inside one compiled program, cached on the
    # engine configuration, the problem object's identity, the stripped
    # driver configuration, and the argument shapes/dtypes. Repeat solves with
    # the same problem object and solver configuration -- including different
    # spans, initial states, and parameter values -- reuse the compiled
    # program without re-tracing.
    # A per-row absolute tolerance is runtime data: it leaves the stripped
    # (static, hashable) configuration and rides the compiled solve as its
    # own traced argument; loop initialization reinstates it as ``cfg.atol``.
    atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
    # ``dt0`` is runtime schedule data just like a per-row tolerance. Keeping
    # it out of the static configuration lets a mapped/windowed optimization
    # choose one initial step per segment without retracing or converting a
    # tracer to Python ``float``.
    dt0_value = jax.lax.stop_gradient(jnp.asarray(cfg.dt0, dtype=dtype))
    cfg_static = _dc_replace(
        cfg,
        theta=None,
        control_state0=None,
        atol=None if atol_rows is not None else cfg.atol,
        dt0=0.0,
        dense=_dense,
        epoch_projected_output=_epoch_projected_output and not bool(save_plan.step_every),
        save_every=save_plan.step_every,
        save_initial=save_plan.save_initial,
        history=None,
        keep_history=keep_history,
        outputs=loop_output_plan,
    )
    # Passed only when a record arrived, so a solve without one calls the
    # compiled core exactly as before.
    history_args = () if history_in is None else (history_in,)
    solve_core = _composable_solve if composable else _solve_core
    sol = solve_core(
        engine,
        problem,
        cfg_static,
        adjoint,
        _has_affect,
        _control_folds_theta,
        ts_grid,
        ts_output,
        t0_arr,
        t1_arr,
        x0,
        theta,
        control0,
        atol_rows,
        dt0_value,
        *history_args,
    )
    # The compiled core cannot trim the step-budget-sized trace buffers (a
    # data-dependent shape); slice them here, where the recorded attempt
    # count is concrete. Inside an enclosing jit the count is a tracer and
    # the full fixed-shape buffers pass through unchanged.
    if cfg.trace and sol.trace is not None:
        trimmed = finalize_trace(sol.trace, sol.stats.trial_steps)
        if trimmed is not sol.trace:
            sol = eqx.tree_at(lambda s: s.trace, sol, trimmed)
    if _project_after_loop:
        sol = _project_saved_samples(
            sol, engine, problem, cfg, theta, None if _control_folds_theta else control0
        )
        if output_plan.requested:
            sol = _emit_saved_outputs(sol, problem, output_plan, theta)
    return sol


def _emit_saved_outputs(sol, problem, plan, theta):
    """Evaluate the named outputs of ``plan`` from a solution's stored state samples.

    Invalid (sentinel) slots stay ``NaN``. ``Solution.ys`` and
    ``Solution.lambdas`` are kept only when ``plan`` names them.
    """
    ts = sol.ts
    ys = sol.ys
    capacity = int(ts.shape[0])
    lams = (
        sol.lambdas
        if sol.lambdas is not None
        else jnp.zeros((capacity, int(problem.n_lam)), dtype=ys.dtype)
    )
    valid = sol.valid_mask
    # Evaluate sentinel slots at the (always valid) terminal state so every
    # lane is in-domain; the select below discards them.
    t_safe = jnp.where(valid, ts, sol.terminal.t)
    y_safe = jnp.where(valid[:, None], ys, sol.terminal.y[None, :])
    lam_safe = jnp.where(valid[:, None], lams, sol.terminal.lam.astype(lams.dtype)[None, :])
    emit = output_emitter(plan, problem, theta, ys.dtype)
    rows = jax.vmap(lambda t, u, lam: emit(t, u, lam if plan.needs_lam else None))(
        t_safe, y_safe, lam_safe
    )
    values = tuple(
        jnp.where(valid.reshape((-1,) + (1,) * (row.ndim - 1)), row, jnp.nan) for row in rows
    )
    by_name = dict(zip(plan.names, values))
    lambdas = by_name.get("lambdas")
    if lambdas is not None and lambdas.shape[-1] == 0:
        lambdas = None
    return eqx.tree_at(
        lambda s: (s.ys, s.lambdas, s.outputs),
        sol,
        (by_name.get("state"), lambdas, SavedOutputs(plan.names, values)),
        is_leaf=lambda x: x is None,
    )


def _project_saved_samples(sol, engine, problem, cfg, theta, control):
    """Project every valid saved sample onto the algebraic constraint manifold.

    The fast samples interpolate the differential rows and hold or
    interpolate the algebraic ones, so they satisfy the constraints only to
    interpolation accuracy. This projection runs the joint consistent-initialization solve
    at each valid sample time with the differential rows pinned, replacing
    the algebraic rows with values that meet the algebraic gate. A problem
    without algebraic structure is returned unchanged; never-reached
    (sentinel) samples stay sentinels.

    The projection solves go through the engine's own linear backend. Every
    sample carries its own iteration matrix, so the samples run as one
    ``vmap`` when that backend factors a batch natively
    (``vmap_factors="native"``: dense, cuDSS, the structured backends) and
    sequentially (``jax.lax.map``) otherwise. A backend that factors a batch
    lane by lane (``"sequential"``: KLU, SuperLU_MT) keeps a native factor
    slot per lane of each call site, so under one ``vmap`` the samples would
    leave one slot per sample in the solve context; the sequential route
    reuses one slot per call site. The projected values are the same either
    way; only the execution differs. An enclosing ``vmap`` over independent
    problems is a separate contract and stays subject to the backend's own
    batching rules.
    """
    id_mask = np.asarray(problem.id_mask).astype(bool)
    if bool(id_mask.all()):
        return sol
    backend = getattr(engine, "backend", "auto")
    lanes_batch = _projection_route_batches_lanes(backend, problem)
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)
    ts = sol.ts
    ys = sol.ys
    lams = sol.lambdas if sol.lambdas is not None else jnp.zeros((ts.shape[0], 0), dtype=ys.dtype)
    y_rows = jnp.concatenate([ys, lams.astype(ys.dtype)], axis=1)
    valid = jnp.logical_not(jnp.any(jnp.isnan(y_rows), axis=1))
    # Guard sentinel rows with the (always valid) terminal state so every
    # vmapped projection lane solves an in-domain system; the select below
    # discards those lanes.
    safe_row = jnp.concatenate([sol.terminal.y, sol.terminal.lam.astype(sol.terminal.y.dtype)])
    guarded = jnp.where(valid[:, None], y_rows, safe_row[None, :])
    t_guarded = jnp.where(valid, ts, sol.terminal.t)
    tol = ic_gate_tolerance(engine.atol, getattr(engine, "ic_tol", None))
    # The gate of the solve's own initialization, so a sample at the residual's
    # round-off floor passes exactly when the start did.
    rtol = ic_gate_rtol(
        getattr(engine, "rtol", 0.0),
        getattr(engine, "ic_rtol", None),
        getattr(engine, "ic_tol", None),
    )
    ic_layout, ic_precision = initialization_policy(
        problem, getattr(engine, "precision", None), mode="forward"
    )
    _, ic_reverse_precision = initialization_policy(
        problem, getattr(engine, "precision", None), mode="reverse"
    )

    def _project(t_i, y_i):
        ic = problem.consistent_initial_state(
            t_i,
            y_i,
            theta,
            control,
            tol=tol,
            rtol=rtol,
            # Project through the engine's own linear backend so output
            # sampling shares the run's solver configuration (and its
            # cached initialization plan).
            backend=backend,
            layout=ic_layout,
            precision=ic_precision,
            reverse_precision=ic_reverse_precision,
        )
        ic_y, _ic_ydot = consume_consistent_init(ic)
        return jnp.asarray(ic_y)

    if lanes_batch:
        projected = jax.vmap(_project)(t_guarded, guarded)
    else:
        projected = jax.lax.map(lambda args: _project(*args), (t_guarded, guarded))
    ys_new = jnp.where(valid[:, None], projected[:, :n_u], ys)
    sol = eqx.tree_at(lambda s: s.ys, sol, ys_new)
    if sol.lambdas is not None and n_lam > 0:
        lam_new = jnp.where(
            valid[:, None], projected[:, n_u:].astype(sol.lambdas.dtype), sol.lambdas
        )
        sol = eqx.tree_at(lambda s: s.lambdas, sol, lam_new)
    return sol


def _projection_route_batches_lanes(backend, problem) -> bool:
    """Whether the projection's linear route may run every sample under one ``vmap``.

    Asks the problem to resolve ``backend`` (a name or a linear-solver
    instance) exactly as its consistent-initialization solve does
    (``problem.initialization_backend``) and reads the resolved backend's
    ``vmap_factors`` contract flag: a backend that factors a batch natively
    (``"native"``) serves every sample under one ``vmap``; one that factors
    lane by lane (``"sequential"``, whose lanes each keep a native factor
    slot) and one that cannot batch (``"unsupported"``) keep the samples on
    the sequential route. A problem without that resolver is asked nothing:
    a backend instance answers for itself and a bare name keeps the batched
    route.
    """
    resolve = getattr(problem, "initialization_backend", None)
    if resolve is not None:
        _name, instance = resolve(backend)
    else:
        instance = None if isinstance(backend, str) else backend
    return instance is None or getattr(instance, "vmap_factors", "native") == "native"


@eqx.filter_jit
def _solve_core(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    ts_grid,
    ts_output,
    t0,
    t1,
    x0,
    theta,
    control0,
    atol_rows=None,
    dt0_value=None,
    history_in=None,
):
    """The trace-once core of `run_engine`.

    Array arguments (``ts_grid``, ``t0``, ``t1``, ``x0``, ``theta`` leaves,
    ``control0`` leaves, ``atol_rows``) are traced; the engine, the problem
    (by identity), the theta/control-stripped configuration, and the adjoint
    are static. ``cfg.theta`` and ``cfg.control_state0`` are ``None`` here
    by construction -- the resolved values arrive as ``theta`` /
    ``control0`` -- and a per-row absolute tolerance arrives as
    ``atol_rows`` (the initialization reinstates it as ``cfg.atol``). A
    history to continue from arrives as ``history_in``.
    """
    init_state = _loop_init(
        engine,
        problem,
        cfg,
        has_affect,
        control_folds_theta,
        ts_grid,
        ts_output,
        t0,
        t1,
        x0,
        theta,
        control0,
        atol_rows,
        dt0_value,
        history_in,
    )
    record_decisions = not adjoint.is_forward_only and callable(
        getattr(engine, "decision_template", None)
    )
    if record_decisions:
        init_state = eqx.tree_at(
            lambda s: s.decisions,
            init_state,
            init_decision_tape(engine.decision_template(), int(cfg.max_steps)),
            is_leaf=lambda x: x is None,
        )
    cond_fun, body_fun = _loop_fns(
        engine,
        problem,
        cfg,
        has_affect,
        control_folds_theta,
        ts_grid,
        t1,
        theta,
        control0,
        record_decisions=record_decisions,
    )

    if adjoint.is_forward_only:
        # True early-exit primal; forward-mode (JVP) differentiable. The
        # custom-JVP wrapper only turns reverse mode into a public error.
        final = _forward_loop(
            engine,
            problem,
            cfg,
            has_affect,
            control_folds_theta,
            init_state,
            ts_grid,
            t1,
            theta,
            control0,
        )
    else:
        # Checkpointed replay loop: the primal early-exits the same way, and
        # the attached reverse rule replays segments from online checkpoints
        # so VJP works through the whole trajectory. The per-step trace and
        # (on a solve without a mutate-and-continue event or per-epoch
        # projected output) the stored outputs are written once per slot and
        # never read back, so they ride the loop's write-only buffer channel
        # instead of the checkpointed carry.
        checkpoints = getattr(adjoint, "checkpoints", None)
        final = checkpointed_while_loop(
            cond_fun,
            body_fun,
            init_state,
            max_steps=int(cfg.max_steps),
            checkpoints=int(checkpoints) if checkpoints is not None else None,
            buffers=_loop_buffers(cfg, record_decisions, init_state.outputs_buffered),
        )

    return _finalize(
        engine, problem, cfg, adjoint, has_affect, control_folds_theta, ts_grid, init_state, final
    )


def _build_saveat_grid(saveat, t0, t1, dtype) -> jax.Array:
    """Return the query times from the shared save plan."""
    return _build_save_plan(saveat, t0, t1, dtype).query_ts


__all__ = ["DriverConfig", "run_engine"]
