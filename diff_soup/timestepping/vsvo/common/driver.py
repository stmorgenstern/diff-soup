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

import inspect
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import Any, Optional, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._adjoints import ForwardAdjoint
from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import Primitive as _Primitive
from diff_soup._jax_compat import ad as _ad
from diff_soup._jax_compat import batching as _batching
from diff_soup._jax_compat import mlir as _mlir
from diff_soup._jax_compat import partial_eval as _partial_eval
from diff_soup._precision import Precision, cast_to_compute_dtype
from diff_soup._solution import result_from_code
from diff_soup.timestepping._affect import apply_affect
from diff_soup.timestepping.control_state import ControlState
from diff_soup.timestepping.event_records import EventLog, TerminalRecord
from diff_soup.timestepping.events import (
    RESOLVE_KIND_CONTINUOUS,
    RESOLVE_KIND_DISCRETE,
    _events_of,
    resolve_in_loop,
)
from diff_soup.timestepping.saveat import _build_save_plan, default_output_plan, output_emitter
from diff_soup.utils.lanes import lane_cond

from ..interpolation import CubicHermiteInterpolation
from ..solution import (
    SavedOutputs,
    Solution,
    SolveHistory,
    SolverStats,
    build_solution,
    commit_pending_sample,
    fill_saveat_times,
    fill_saveat_window,
    hold_pending_sample,
    init_saveat_outputs,
    place_saveat_sample,
    terminal_sample_slot,
)
from .backends import initialization_policy
from .checkpointed_loop import checkpointed_while_loop
from .contract import (
    EngineCapabilities,
    coerce_problem,
    consume_consistent_init,
    ic_gate_tolerance,
    pack_state,
    unpack_state,
    validate_requirements,
)
from .decisions import REPLAY_DIVERGED_MESSAGE, DecisionTape, init_decision_tape
from .trace import (
    StepTraceHeader,
    alloc_trace,
    buffer_at_set,
    finalize_trace,
    write_trace,
    zero_record,
)
from .weights import wrms_norm

# Integer termination codes in the space result_from_code maps (see
# diff_soup._solution.result_from_code): 0 unknown, 1 success, 2 step collapse,
# 3 nonlinear divergence, 4 max steps, 5 event stop, 6 inconsistent IC,
# 7 accuracy limit.
_CODE_UNKNOWN = 0
_CODE_SUCCESS = 1
_CODE_MAX_STEPS = 4
_CODE_EVENT_STOP = 5
_CODE_INCONSISTENT_IC = 6
_CODE_ACCURACY_LIMIT = 7

# Endpoint tolerance so the final save time, reached up to a rounding error, is
# still claimed by the last step, and so an exact landing on t1 terminates.
_SAVE_TOL = 1e-10

# Interpolant evaluations per attempt on the incremental save-fill fast path;
# a step covering more save times than this falls back to the full-grid fill.
_SAVE_WINDOW = 16


# ---------------------------------------------------------------------------
# The engine protocol and the driver/engine per-attempt data contract
# ---------------------------------------------------------------------------
@runtime_checkable
class Engine(Protocol):
    """The step kernel the driver runs. Opaque behind three methods + a flag.

    An engine owns one internal step: it builds its own carry (history,
    coefficients, step and order state), advances one attempt, and evaluates its
    dense output on the last accepted interval. The driver never inspects the
    carry; it reads only the `StepReport` each attempt returns.

    Attributes:
        capabilities: The `EngineCapabilities` the driver validates the
            problem against before the loop.

    An engine may also expose a ``trace_kind`` string (``"ndf"`` / ``"bdf"`` /
    ``"ida"``) naming its per-step trace family; the driver reads it only when a
    trace is requested and defaults to ``"ndf"`` when it is absent. An engine
    whose ``init_carry`` consistent-initializes the state may expose
    ``state_at_entry(carry)`` returning the augmented ``[u; lam]`` vector the
    solve starts from; the driver records it as the ``t0`` output sample and
    falls back to the raw input state when the method is absent. An engine
    that exposes ``decision_template()`` (the zero-valued decision inputs of
    one attempt) and accepts ``attempt_step(carry, decision=(replay,
    recorded))`` has its decisions recorded under a checkpointed reverse solve,
    so the backward replay decides as the forward sweep did. Such an engine also
    exposes ``step_at_entry(carry)`` and accepts ``init_carry(..., step=(replay,
    recorded))``, so a replayed event restart enters with the step the forward
    sweep's restart chose instead of searching for it again. An engine whose
    ``attempt_step`` takes a ``live`` keyword receives the loop's condition for
    the lane: under ``vmap`` the loop discards the attempt of a lane where it is
    false, so the engine may leave that lane out of any decision it takes for
    the whole batch, provided the lanes it keeps compute the same values. An
    engine that exposes ``history_at_exit(carry)`` (the record a later solve
    continues from) and accepts ``init_carry(..., history=...)`` (a
    [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory] to
    continue from) serves a solve's ``history=`` request.
    """

    capabilities: EngineCapabilities

    def init_carry(self, problem, t0, t1, y0, ydot0, lam0, theta, cfg):
        """Build the engine's initial per-step carry.

        ``y0``/``ydot0``/``lam0`` are the driver's consistent initial
        differential state, its derivative, and the initial multipliers.
        """
        ...

    def attempt_step(self, carry):
        """Advance one internal step attempt (accepted or rejected).

        Returns ``(carry, report)`` where ``report`` is a `StepReport`.
        Pure of the driver's save-grid, event, and statistics bookkeeping.
        """
        ...

    def dense(self, carry, t):
        """Evaluate ``(y(t), ydot(t))`` on the last accepted step's interval."""
        ...


class StepReport(eqx.Module):
    """What one step attempt reports back to the driver.

    All leaves are fixed-shape arrays. The adaptive controls (``accepted``,
    ``h_new``, ``k_new``, the counters) are integer/boolean state the driver
    accumulates and threads but never differentiates.

    Attributes:
        accepted: Whether the attempt was accepted.
        t_new: Time after the attempt (equal to the prior time if rejected).
        y_new: Accepted differential state (unchanged if rejected).
        ydot_new: Accepted state derivative.
        lam_new: Multipliers after the attempt.
        h_new: Step size chosen for the next attempt.
        k_new: Order chosen for the next attempt.
        err_estimate: The accept/reject error estimate.
        result: Termination code for this attempt (the code space
            [`result_from_code`][diff_soup._solution.result_from_code] consumes).
        newton_iters: Corrector-iteration delta for the shared statistics.
        jac_refreshes: Delta of fresh state-Jacobian evaluations. An attempt
            that factorizes a matrix recomposed from a saved state Jacobian
            contributes to ``factorizations`` but not to this.
        factorizations: Iteration-matrix-factorization delta.
        linear_solves: Linear-solve delta.
        rhs_evals: Right-hand-side-evaluation delta.
        residual_evals: Residual-evaluation delta.
        trace_record: The engine's per-attempt trace record, or ``None`` when
            the solve is not tracing (the driver supplies the shared header).
        precision_promotions: Per-cause promotion-counter delta of the
            adaptive linear-precision policy (``int32[4]``), or ``None`` for
            concrete solve modes.
        ir_refinement_steps: Mixed-precision refinement-step delta of this
            attempt's held-factor linear solves (fp64 scalar), or ``None``
            for an engine/configuration that does not report it.
        fp64_fallbacks: Reduced-precision fp64-fallback delta of this
            attempt's held-factor linear solves (fp64 scalar), or ``None``.
        multirhs_solves: Multi-right-hand-side linear-solve delta (the
            Schur-complement pre-builds this attempt ran), or ``None`` for an
            engine that does not report it.
        newton_failures: Nonlinear-corrector convergence-failure delta, or
            ``None`` for an engine that does not report it.
        multirhs_columns: Right-hand-side column delta across this attempt's
            multi-RHS solves, or ``None``.
        du_b_builds: Coupling-block border-solve construction delta, or
            ``None``.
        schur_builds: Schur-complement construction delta, or ``None``.
        decisions: The decision inputs this attempt used (the engine's
            ``decision_template`` shape), recorded by a checkpointed reverse
            solve so its replay decides as the forward sweep did, or ``None``
            for an engine that does not record them.
    """

    accepted: jax.Array
    t_new: jax.Array
    y_new: jax.Array
    ydot_new: jax.Array
    lam_new: jax.Array
    h_new: jax.Array
    k_new: jax.Array
    err_estimate: jax.Array
    result: jax.Array
    newton_iters: jax.Array
    jac_refreshes: jax.Array
    factorizations: jax.Array
    linear_solves: jax.Array
    rhs_evals: jax.Array
    residual_evals: jax.Array
    trace_record: Optional[Any] = None
    precision_promotions: Optional[jax.Array] = None
    ir_refinement_steps: Optional[jax.Array] = None
    fp64_fallbacks: Optional[jax.Array] = None
    multirhs_solves: Optional[jax.Array] = None
    newton_failures: Optional[jax.Array] = None
    multirhs_columns: Optional[jax.Array] = None
    du_b_builds: Optional[jax.Array] = None
    schur_builds: Optional[jax.Array] = None
    decisions: Optional[Any] = None


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


# ---------------------------------------------------------------------------
# Driver-owned loop state
# ---------------------------------------------------------------------------
class _DriverState(eqx.Module):
    """The bounded step loop's carry (all fixed-shape leaves)."""

    engine_carry: Any
    t_prev: jax.Array
    h_prev: jax.Array
    y_last: jax.Array
    lam_last: jax.Array
    control_state: Any
    event_log: Any
    ts_output: jax.Array
    # One buffer per stored output, ``(capacity + 1, *shape)`` (the last row
    # discards masked writes), and the held-back latest sample ``(slot, rows)``
    # when the buffers are write-only (``None`` otherwise).
    outputs: Any
    pending: Any
    save_cursor: jax.Array
    valid_count: jax.Array
    control_epochs: jax.Array
    event_indices: jax.Array
    event_kinds: jax.Array
    event_directions: jax.Array
    result_codes: jax.Array
    dense_nodes: Any
    trace: Any
    step: jax.Array
    done: jax.Array
    term_code: jax.Array
    event_idx: jax.Array
    event_kind: jax.Array
    event_direction: jax.Array
    t_reached: jax.Array
    tolsf: jax.Array
    accepts: jax.Array
    rejects: jax.Array
    trial_steps: jax.Array
    newton_iters: jax.Array
    jac_refreshes: jax.Array
    factorizations: jax.Array
    linear_solves: jax.Array
    rhs_evals: jax.Array
    residual_evals: jax.Array
    precision_promotions: jax.Array
    # Reduced-precision observability accumulators (fp64 scalars), or ``None``
    # when the engine's configuration reports no such counters — keeping the
    # default fp64 solve's compiled program unchanged.
    ir_refinement_steps: Any
    fp64_fallbacks: Any
    multirhs_solves: jax.Array
    newton_failures: jax.Array
    multirhs_columns: jax.Array
    du_b_builds: jax.Array
    schur_builds: jax.Array
    saveat_interpolations: jax.Array
    event_solves: jax.Array
    # The per-attempt decision record of a checkpointed reverse solve (a
    # write-only loop buffer), or ``None``.
    decisions: Any = None
    # Whether ``outputs`` is written only by slot (each slot once, never read
    # in the loop), so it rides the loop's write-only buffer channel.
    outputs_buffered: bool = eqx.field(static=True, default=False)


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


def _safe_ratio(num: jax.Array, den: jax.Array) -> jax.Array:
    """Return ``num / den`` with a unit fallback when ``den`` is zero."""
    ok = den != 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), jnp.ones_like(num))


def _freeze_finished(finished, leaf):
    """Freeze an inexact leaf on finished lanes, pass others through.

    Only inexact array leaves carry a tangent, so on a finished lane they are
    replaced by their stopped-gradient value (no cotangent can flow back through
    a lane that is only being kept alive by a slower sibling lane in a batched
    loop). Integer/bool control state is left as-is so the loop's own
    bookkeeping is untouched.
    """
    if eqx.is_inexact_array(leaf):
        return jnp.where(finished, jax.lax.stop_gradient(leaf), leaf)
    return leaf


def _buffer_fields(state):
    """The carry's write-only buffer subtrees: trace, dense nodes, decision record, outputs.

    The output buffers belong to the set only when they are written by slot
    alone (``state.outputs_buffered``).
    """
    fields = (state.trace, state.dense_nodes, state.decisions)
    return fields + (state.outputs,) if state.outputs_buffered else fields


def _strip_buffers(state):
    """``state`` with its write-only buffer subtrees replaced by ``None``."""
    width = len(_buffer_fields(state))
    return eqx.tree_at(_buffer_fields, state, (None,) * width, is_leaf=lambda x: x is None)


def _map_no_buffer(f, state):
    """Apply ``f`` to every leaf of ``state`` except the write-only buffers.

    The buffer subtrees (see ``_buffer_fields``) are the ones the
    checkpointed loop marks as write-only accumulation buffers; inside the
    loop body they support only ``.at[].set()``, so mapping over them would
    violate that contract. They are lifted out, left untouched, and
    restored. When absent they are ``None`` and both lifts are no-ops.
    """
    saved = _buffer_fields(state)
    mapped = jax.tree_util.tree_map(f, _strip_buffers(state))
    return eqx.tree_at(_buffer_fields, mapped, saved, is_leaf=lambda x: x is None)


def _select_no_buffer(pred, new, old):
    """Per-leaf ``where(pred, old_leaf, new_leaf)`` skipping both trees' write-only buffers.

    The buffers of ``new`` are kept as-is: the body's ``.at[].set()`` writes
    must survive; only the non-buffer leaves revert to ``old`` on lanes where
    ``pred`` is true.
    """
    saved = _buffer_fields(new)
    picked = jax.tree_util.tree_map(
        lambda n, o: jnp.where(pred, o, n), _strip_buffers(new), _strip_buffers(old)
    )
    return eqx.tree_at(_buffer_fields, picked, saved, is_leaf=lambda x: x is None)


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


def _output_plan(cfg, problem):
    """The solve's output layout: ``cfg.outputs``, or the default state and globals."""
    if cfg.outputs is not None:
        return cfg.outputs
    return default_output_plan(int(problem.n_u), int(problem.n_lam))


def _outputs_buffered(cfg, has_affect) -> bool:
    """Whether the output buffers are written by slot alone, each slot once.

    A mutate-and-continue event rewrites the row at its switch time with the
    post-affect state, and per-epoch projected output reverts a failed
    segment's rows; both read the buffers back, so those solves keep the
    buffers on the ordinary carry.
    """
    return not has_affect and not bool(cfg.epoch_projected_output)


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


def _control_epoch(control: Any) -> jax.Array:
    """Return a control's traced epoch, or zero without an epoch-bearing channel."""
    if control is None:
        epoch = 0
    elif isinstance(control, dict):
        epoch = control.get("epoch", 0)
    else:
        epoch = getattr(control, "epoch", 0)
    return jnp.asarray(epoch, dtype=jnp.int32)


def _event_projection_options(engine: Any) -> Any:
    """Build the shared algebraic-projection policy for event candidates.

    Returns an [AlgebraicProjectionOptions][diff_soup.projection.AlgebraicProjectionOptions].
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionOptions

    projection_atol = ic_gate_tolerance(engine.atol, getattr(engine, "ic_tol", None))
    projection_backend = getattr(getattr(engine, "backend", None), "backend_name", "auto")
    if projection_backend not in ("dense", "klu", "cudss"):
        projection_backend = "auto"
    return AlgebraicProjectionOptions(
        rtol=float(getattr(engine, "rtol", 1e-8)),
        atol=projection_atol,
        max_steps=30,
        linear_solver=projection_backend,
        # Root-time IFT needs a time-direction JVP even when the surrounding
        # solve is transposed. This internal projection rule is transpose-safe.
        ad_mode="forward",
    )


def _selected_event_tolerance(events, event_index, t_lo, t_hi, dtype):
    """Return the winning event's time-collision tolerance."""
    selected = jnp.asarray(_SAVE_TOL, dtype=dtype)
    width = jnp.abs(jnp.asarray(t_hi, dtype=dtype) - jnp.asarray(t_lo, dtype=dtype))
    for index, event in enumerate(events):
        root_finder = getattr(event, "root_finder", None)
        if root_finder is None:
            tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
        else:
            tolerance = jnp.maximum(
                jnp.asarray(float(getattr(root_finder, "atol", 0.0)), dtype=dtype),
                jnp.asarray(float(getattr(root_finder, "rtol", 0.0)), dtype=dtype) * width,
            )
        selected = jnp.where(event_index == index, tolerance, selected)
    return selected


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
    problem = coerce_problem(problem)
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


def _loop_init(
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
    atol_rows=None,
    dt0_value=None,
    history_in=None,
):
    """Build the initial driver-loop state (everything before the first attempt).

    ``atol_rows`` is the per-row absolute-tolerance vector of a
    layout-resolved solve, or ``None``. It arrives as a separate traced
    argument (the stripped configuration is static and cannot hold an
    array) and is reinstated as ``cfg.atol`` here, inside the trace, so the
    engine reads one configuration surface.

    ``history_in`` is the [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
    the solve continues from when its flag is set, or ``None``; the engine
    builds its first carry from it.

    Runs the engine's ``init_carry`` (including any consistent-initialization
    projection), seeds the save grid from the state the solve actually starts
    from, and packs the full `_DriverState`. Shared by the compiled
    solve core and by the composable forward rule, which derives the initial
    state's tangent by differentiating this function directly.
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionStatus, _project_algebraic_bidirectional

    if atol_rows is not None:
        cfg = _dc_replace(cfg, atol=atol_rows)
    if dt0_value is not None:
        cfg = _dc_replace(cfg, dt0=jax.lax.stop_gradient(jnp.asarray(dt0_value)))
    _has_affect = has_affect
    n_u = int(problem.n_u)
    dtype = x0.dtype
    u0, lam0 = unpack_state(x0, n_u)
    # The engine obtains the physical startup slope from the view's
    # residual/operator route. This value is only a shape-correct Newton seed.
    ydot0 = jnp.zeros((n_u,), dtype=dtype)
    control_channel0 = (
        control0
        if (
            _has_affect
            or problem.has_recipe_controls
            or getattr(problem, "has_operational_control", False)
        )
        else None
    )
    # The engine evaluates the model with the explicitly supplied control values;
    # with the control channel folded into the parameters (an affect solve
    # with no explicit ``control_state0``) the dynamics read them from theta
    # and the engine's control stays ``None``.
    engine_control0 = None if control_folds_theta else control0

    # Statically size the trace to the step budget, only when a trace is asked
    # for (zero cost on the default path).
    max_steps = int(cfg.max_steps)
    kind = getattr(engine, "trace_kind", "ndf")
    trace0 = alloc_trace(kind, max_steps, dtype) if cfg.trace else None

    continuing = {} if history_in is None else {"history": history_in}
    engine_carry0 = engine.init_carry(
        problem, t0, t1, u0, ydot0, lam0, theta, cfg, control=engine_control0, **continuing
    )

    # Seed the output grid from the state the solve actually starts from: the
    # engine's consistent-initialization runs inside ``init_carry``, so the
    # ``t0`` sample must read the projected state off the carry, not the raw
    # input (for a DAE the two differ on every algebraic row).
    if hasattr(engine, "state_at_entry"):
        u0_entry, lam0_entry = unpack_state(engine.state_at_entry(engine_carry0), n_u)
    else:
        u0_entry, lam0_entry = u0, lam0
    entry_state = pack_state(u0_entry, lam0_entry)
    if hasattr(engine, "slope_at_entry"):
        entry_slope = jnp.asarray(engine.slope_at_entry(engine_carry0), dtype)
    else:
        entry_slope = jnp.concatenate([ydot0, jnp.zeros((lam0.shape[0],), dtype)])
    if history_in is not None and cfg.dense:
        # A continued history starts on the interpolant of the earlier steps.
        continued_slope = jnp.asarray(engine.dense_full(engine_carry0, t0)[1], dtype)
        entry_slope = jnp.where(history_in.continues, continued_slope, entry_slope)
    capacity = int(ts_output.shape[0])
    output_plan = _output_plan(cfg, problem)
    outputs_buffered = _outputs_buffered(cfg, _has_affect)
    emit = output_emitter(output_plan, problem, theta, dtype)
    needs_lam = output_plan.needs_lam
    ts_output0, outputs0 = init_saveat_outputs(
        output_plan,
        ts_grid,
        u0_entry,
        lam0_entry,
        t0,
        capacity=capacity,
        emit=emit,
    )

    h0 = jnp.asarray(cfg.dt0, dtype)
    if history_in is not None and hasattr(engine, "step_at_entry"):
        # A continued history enters with the step its controller chose.
        carried_step = jax.lax.stop_gradient(
            jnp.asarray(engine.step_at_entry(engine_carry0), dtype)
        )
        h0 = jnp.where(history_in.continues, carried_step, h0)
    z = jnp.asarray(0, jnp.int32)
    # The fired-event index starts at -1 ("nothing fired"): a non-negative
    # value marks the declaration index of the last event that fired, whether
    # it terminated the solve or switched it and continued.
    no_event = jnp.asarray(-1, jnp.int32)
    event_log0 = (
        EventLog.empty(int(cfg.max_events), int(x0.shape[0]), dtype=dtype)
        if cfg.event is not None
        else None
    )
    # Dense-output node buffers: one (time, state, derivative) node per
    # accepted step plus the entry node. Sized by the step budget; slots
    # beyond the recorded count hold +inf times so segment lookup skips them.
    if cfg.dense:
        n_full = int(x0.shape[0])
        cap = max_steps + 1
        dense_nodes0 = (
            jnp.full((cap,), jnp.inf, dtype).at[0].set(t0),
            jnp.zeros((cap, n_full), dtype).at[0].set(entry_state),
            jnp.zeros((cap, n_full), dtype).at[0].set(entry_slope),
        )
    else:
        dense_nodes0 = None

    # Reduced-precision counter channel: engaged exactly when the engine's
    # corrector solves report in-band counters (a reduced or adaptive solve
    # mode); ``None`` otherwise so the default fp64 solve carries no counters.
    _prec = getattr(engine, "precision", None)
    _reports_reduced = getattr(_prec, "solve_mode", "fp64") in ("fp32", "fp32_ir", "adaptive")
    reduced_zero = jnp.zeros((), jnp.result_type(float)) if _reports_reduced else None

    # The explicit grid is consumed once, front to back. Accepted-step output
    # instead owns a direct-write prefix and may seed its first slot with the
    # exact entry state; it never runs the interpolation fill machinery.
    if cfg.save_every:
        save_cursor0 = jnp.asarray(int(cfg.save_initial), jnp.int32)
        valid_count0 = save_cursor0
        if cfg.save_initial:
            ts_output0 = ts_output0.at[0].set(t0)
            outputs0, _ = place_saveat_sample(
                outputs0,
                None,
                0,
                emit(t0, u0_entry, lam0_entry if needs_lam else None),
                jnp.asarray(True),
            )
    else:
        save_cursor0 = jnp.searchsorted(ts_grid, t0, side="right").astype(jnp.int32)
        valid_count0 = save_cursor0
        ts_output0 = fill_saveat_times(ts_grid, ts_output0, save_cursor0)
    sample_slots = jnp.arange(capacity, dtype=jnp.int32)
    epoch0 = _control_epoch(control_channel0 if control_channel0 is not None else control0)
    control_epochs0 = jnp.where(sample_slots < valid_count0, epoch0, no_event)
    event_indices0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    event_kinds0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    event_directions0 = jnp.full((capacity,), -1, dtype=jnp.int32)
    result_codes0 = jnp.full((capacity,), -1, dtype=jnp.int32)

    # A terminating event already satisfied at the integration start owns an
    # exact terminal row without taking a numerical step. Continuous events
    # use their projected state policy here as they do during localization.
    initial_fired = jnp.asarray(False)
    initial_projection_failed = jnp.logical_not(
        jnp.all(jnp.isfinite(entry_state)) & jnp.all(jnp.isfinite(entry_slope))
    )
    initial_event_idx = no_event
    initial_event_kind = no_event
    initial_event_direction = no_event
    initial_state = entry_state
    initial_root_tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
    events = _events_of(cfg.event)
    recipe_controls = bool(getattr(problem, "has_recipe_controls", False)) or bool(
        getattr(problem, "has_operational_control", False)
    )
    declared_control0 = control_channel0 if (_has_affect or recipe_controls) else control0
    event_control0 = (
        problem.control_values(t0, theta, declared_control0)
        if recipe_controls
        else declared_control0
    )
    projected_initial = any(
        getattr(getattr(event, "state_policy", None), "mode", None) == "projected"
        and getattr(event, "affect", None) is None
        for event in events
    )
    projection_options = (
        _event_projection_options(engine)
        if projected_initial or cfg.epoch_projected_output
        else None
    )
    if cfg.epoch_projected_output:
        has_initial_output = save_cursor0 > 0

        def _project_entry(_):
            projected = _project_algebraic_bidirectional(
                problem,
                t=t0,
                state=entry_state,
                state_dot=entry_slope,
                seed=entry_state,
                theta=theta,
                control=event_control0,
                options=projection_options,
            )
            return projected.state, projected.status

        projected_entry_state, projected_entry_status = jax.lax.cond(
            has_initial_output,
            _project_entry,
            lambda _: (
                entry_state,
                jnp.asarray(int(AlgebraicProjectionStatus.SUCCESS), jnp.int32),
            ),
            None,
        )
        initial_output_projection_failed = jnp.logical_and(
            has_initial_output,
            projected_entry_status != int(AlgebraicProjectionStatus.SUCCESS),
        )
        projected_entry_y, projected_entry_lam = unpack_state(projected_entry_state, n_u)
        projected_rows = emit(t0, projected_entry_y, projected_entry_lam if needs_lam else None)
        keep_projected = jnp.logical_and(
            jnp.arange(capacity + 1, dtype=jnp.int32) < save_cursor0,
            ~initial_output_projection_failed,
        )
        outputs0 = tuple(
            jnp.where(keep_projected.reshape((-1,) + (1,) * row.ndim), row[None], buffer)
            for buffer, row in zip(outputs0, projected_rows)
        )
        initial_projection_failed = jnp.logical_or(
            initial_projection_failed, initial_output_projection_failed
        )
        save_cursor0 = jnp.where(initial_output_projection_failed, z, save_cursor0)
        valid_count0 = jnp.where(initial_output_projection_failed, z, valid_count0)
        ts_output0 = jnp.where(
            initial_output_projection_failed,
            jnp.full_like(ts_output0, jnp.nan),
            ts_output0,
        )
        outputs0 = tuple(
            jnp.where(initial_output_projection_failed, jnp.full_like(buffer, jnp.nan), buffer)
            for buffer in outputs0
        )
        control_epochs0 = jnp.where(
            initial_output_projection_failed,
            jnp.full_like(control_epochs0, -1),
            control_epochs0,
        )
    for index, event in enumerate(events):
        if getattr(event, "affect", None) is not None:
            continue
        active = jnp.logical_not(jnp.logical_or(initial_fired, initial_projection_failed))
        policy = getattr(event, "state_policy", None)
        if policy is not None and policy.mode == "projected":
            projected = _project_algebraic_bidirectional(
                problem,
                t=t0,
                state=entry_state,
                state_dot=entry_slope,
                seed=entry_state,
                theta=theta,
                control=event_control0,
                options=projection_options,
            )
            evaluation_ok = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
            event_state = jnp.where(evaluation_ok, projected.state, entry_state)
            initial_projection_failed = jnp.logical_or(
                initial_projection_failed,
                jnp.logical_and(active, jnp.logical_not(evaluation_ok)),
            )
        else:
            evaluation_ok = jnp.asarray(True)
            event_state = entry_state
        value = jnp.asarray(event.cond_fn(t0, event_state, None, control=event_control0)).reshape(
            ()
        )
        root_finder = getattr(event, "root_finder", None)
        if root_finder is None:
            satisfied = jnp.asarray(value, dtype=jnp.bool_)
            kind = RESOLVE_KIND_DISCRETE
            crossing_direction = jnp.asarray(0, jnp.int32)
            tolerance = jnp.asarray(_SAVE_TOL, dtype=dtype)
        else:
            value_tolerance = max(
                float(getattr(root_finder, "atol", 0.0)),
                10.0 * float(np.finfo(dtype).eps),
            )

            def _condition_along_entry(tq):
                delta = tq - t0
                state_seed = entry_state + delta * entry_slope
                control_tq = (
                    problem.control_values(tq, theta, declared_control0)
                    if recipe_controls
                    else declared_control0
                )
                if policy is not None and policy.mode == "projected":
                    projected_tq = _project_algebraic_bidirectional(
                        problem,
                        t=tq,
                        state=state_seed,
                        state_dot=entry_slope,
                        seed=state_seed,
                        theta=theta,
                        control=control_tq,
                        options=projection_options,
                    )
                    state_tq = projected_tq.state
                else:
                    state_tq = state_seed
                return jnp.asarray(event.cond_fn(tq, state_tq, None, control=control_tq)).reshape(
                    ()
                )

            _, outgoing_slope = jax.jvp(
                _condition_along_entry,
                (jnp.asarray(t0, dtype=dtype),),
                (jnp.asarray(1.0, dtype=dtype),),
            )
            crossing_direction = jnp.where(
                outgoing_slope > 0,
                jnp.asarray(1, jnp.int32),
                jnp.where(
                    outgoing_slope < 0,
                    jnp.asarray(-1, jnp.int32),
                    jnp.asarray(0, jnp.int32),
                ),
            )
            declared_direction = getattr(event, "direction", "any")
            if declared_direction == "up":
                direction_matches = crossing_direction > 0
            elif declared_direction == "down":
                direction_matches = crossing_direction < 0
            else:
                direction_matches = jnp.asarray(True)
            satisfied = jnp.logical_and(
                jnp.abs(value) <= jnp.asarray(value_tolerance, dtype=value.dtype),
                direction_matches,
            )
            kind = RESOLVE_KIND_CONTINUOUS
            tolerance = jnp.maximum(
                jnp.asarray(float(getattr(root_finder, "atol", 0.0)), dtype=dtype),
                jnp.asarray(float(getattr(root_finder, "rtol", 0.0)), dtype=dtype),
            )
        take = jnp.logical_and(
            active,
            jnp.logical_and(evaluation_ok, satisfied),
        )
        initial_fired = jnp.logical_or(initial_fired, take)
        initial_event_idx = jnp.where(take, jnp.asarray(index, jnp.int32), initial_event_idx)
        initial_event_kind = jnp.where(take, jnp.asarray(kind, jnp.int32), initial_event_kind)
        initial_event_direction = jnp.where(take, crossing_direction, initial_event_direction)
        initial_state = jnp.where(take, event_state, initial_state)
        initial_root_tolerance = jnp.where(take, tolerance, initial_root_tolerance)

    initial_y, initial_lam = unpack_state(initial_state, n_u)
    terminal_slot0, terminal_collides0, terminal_room0 = terminal_sample_slot(
        ts_output0,
        valid_count0,
        t0,
        initial_root_tolerance,
    )
    safe_terminal_slot0 = jnp.minimum(terminal_slot0, jnp.asarray(capacity - 1, jnp.int32))
    write_initial_terminal = jnp.logical_and(initial_fired, terminal_room0)

    def _write_initial(buffer, value):
        written = buffer.at[safe_terminal_slot0].set(value)
        return jnp.where(write_initial_terminal, written, buffer)

    ts_output0 = _write_initial(ts_output0, jnp.asarray(t0, dtype=dtype))
    outputs0, _ = place_saveat_sample(
        outputs0,
        None,
        safe_terminal_slot0,
        emit(t0, initial_y, initial_lam if needs_lam else None),
        write_initial_terminal,
    )
    control_epochs0 = _write_initial(control_epochs0, epoch0)
    event_indices0 = _write_initial(event_indices0, initial_event_idx)
    event_kinds0 = _write_initial(event_kinds0, initial_event_kind)
    event_directions0 = _write_initial(event_directions0, initial_event_direction)
    result_codes0 = _write_initial(result_codes0, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
    valid_count0 = jnp.where(
        write_initial_terminal,
        valid_count0 + jnp.logical_not(terminal_collides0).astype(jnp.int32),
        valid_count0,
    )
    if cfg.event is not None:
        appended0 = event_log0.append(
            t=t0,
            index=initial_event_idx,
            kind=initial_event_kind,
            direction=initial_event_direction,
            y=initial_state,
            pre_y=initial_state,
            pre_epoch=epoch0,
            post_epoch=epoch0,
        )
        event_log0 = jax.tree_util.tree_map(
            lambda new, old: jnp.where(initial_fired, new, old),
            appended0,
            event_log0,
        )
    # Write-only buffers take each slot once: hold the latest sample back in
    # the carry, where an exact terminal sample at its time may replace it.
    pending0 = None
    if outputs_buffered:
        outputs0, pending0 = hold_pending_sample(outputs0, valid_count0)
    initial_done = jnp.logical_or(initial_fired, initial_projection_failed)
    initial_term_code = jnp.where(
        initial_fired,
        jnp.asarray(_CODE_EVENT_STOP, jnp.int32),
        jnp.where(
            initial_projection_failed,
            jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
            jnp.asarray(_CODE_UNKNOWN, jnp.int32),
        ),
    )
    init_state = _DriverState(
        engine_carry=engine_carry0,
        t_prev=t0,
        h_prev=h0,
        y_last=jnp.where(initial_fired, initial_y, u0_entry),
        lam_last=jnp.where(initial_fired, initial_lam, lam0_entry),
        control_state=control_channel0,
        event_log=event_log0,
        ts_output=ts_output0,
        save_cursor=save_cursor0,
        valid_count=valid_count0,
        control_epochs=control_epochs0,
        event_indices=event_indices0,
        event_kinds=event_kinds0,
        event_directions=event_directions0,
        result_codes=result_codes0,
        dense_nodes=dense_nodes0,
        outputs=outputs0,
        pending=pending0,
        trace=trace0,
        step=z,
        done=initial_done,
        term_code=initial_term_code,
        event_idx=jnp.where(initial_fired, initial_event_idx, no_event),
        event_kind=jnp.where(initial_fired, initial_event_kind, no_event),
        event_direction=jnp.where(initial_fired, initial_event_direction, no_event),
        t_reached=t0,
        tolsf=jnp.asarray(0.0, dtype),
        accepts=z,
        rejects=z,
        trial_steps=z,
        newton_iters=z,
        jac_refreshes=z,
        factorizations=z,
        linear_solves=z,
        rhs_evals=z,
        residual_evals=z,
        precision_promotions=jnp.zeros((4,), jnp.int32),
        ir_refinement_steps=reduced_zero,
        fp64_fallbacks=reduced_zero,
        multirhs_solves=z,
        newton_failures=z,
        multirhs_columns=z,
        du_b_builds=z,
        schur_builds=z,
        saveat_interpolations=z,
        event_solves=z,
        outputs_buffered=outputs_buffered,
    )
    return init_state


def _check_replay(report: StepReport, tape: DecisionTape, slot, replay) -> StepReport:
    """Raise when a replayed attempt reaches a different time or next step.

    A replay uses the recorded decision inputs, so its time and next step size
    agree with the forward sweep's to rounding. A larger difference means the
    replay integrates a different trajectory; the solve raises instead of
    returning the gradient of that other trajectory.
    """
    eps = jnp.finfo(jnp.asarray(report.t_new).dtype).eps

    def close(value, recorded):
        value = jnp.asarray(value).astype(recorded.dtype)
        scale = jnp.maximum(jnp.abs(value), jnp.abs(recorded))
        return jnp.abs(value - recorded) <= 64.0 * eps * scale

    same = jnp.logical_and(
        close(report.t_new, tape.t_new[slot]), close(report.h_new, tape.h_new[slot])
    )
    diverged = jnp.logical_and(replay, jnp.logical_not(same))
    t_new = eqx.error_if(report.t_new, diverged, REPLAY_DIVERGED_MESSAGE)
    return eqx.tree_at(lambda r: r.t_new, report, t_new)


def _record_decisions(tape: DecisionTape, slot, report: StepReport, finished) -> DecisionTape:
    """Write one attempt's decision inputs, time and next step into its row.

    A lane that finished before the rest of a batched loop keeps its row as it
    was.
    """

    def write(buffer, value):
        value = jnp.asarray(value).astype(buffer.dtype)
        return buffer.at[slot].set(jnp.where(finished, buffer[slot], value))

    return DecisionTape(
        written=write(tape.written, True),
        t_new=write(tape.t_new, report.t_new),
        h_new=write(tape.h_new, report.h_new),
        h_restart=tape.h_restart,
        values=tuple(write(buffer, value) for buffer, value in zip(tape.values, report.decisions)),
    )


def _record_restart_step(tape: DecisionTape, slot, step, finished) -> DecisionTape:
    """Write the entry step of the carry an attempt hands on into its row.

    After an event restart this is the restarted engine's entry step, which a
    replay of the attempt reuses. A lane that finished before the rest of a
    batched loop keeps its row as it was.
    """
    buffer = tape.h_restart
    row = jnp.where(finished, buffer[slot], jnp.asarray(step).astype(buffer.dtype))
    return DecisionTape(
        written=tape.written,
        t_new=tape.t_new,
        h_new=tape.h_new,
        h_restart=buffer.at[slot].set(row),
        values=tape.values,
    )


def _write_collide_rows(buffers, collide, rows):
    """Overwrite the rows flagged by ``collide`` (``(capacity,)``) with one sample's rows.

    Used for the save time at a mutate-and-continue switch, whose buffers ride
    the ordinary carry.
    """
    flagged = jnp.concatenate([collide, jnp.zeros((1,), dtype=bool)])
    return tuple(
        jnp.where(flagged.reshape((-1,) + (1,) * row.ndim), row[None], buffer)
        for buffer, row in zip(buffers, rows)
    )


def _loop_fns(
    engine,
    problem,
    cfg,
    has_affect,
    control_folds_theta,
    ts_grid,
    t1,
    theta,
    control0,
    record_decisions=False,
):
    """Build the step loop's ``cond`` and ``body`` over the traced solve inputs.

    Every traced value the loop body reads (the save grid, the horizon, the
    parameters, the event control seed) arrives as an argument, so a caller
    can differentiate one step by differentiating through this factory's
    ``body``. Shared by the compiled solve core and by the composable forward
    rule, whose augmented loop differentiates ``body`` once per step.

    With ``record_decisions`` the carry holds a
    [DecisionTape][diff_soup.timestepping.vsvo.common.decisions.DecisionTape]:
    each attempt writes its decision inputs, time and next step to the row of
    its attempt index, and an attempt whose row is already written (the
    backward replay of a checkpointed loop) uses the recorded decision inputs
    and raises if it still reaches a different time or next step.
    """
    # The projection sits above timestepping in the layer order.
    from diff_soup.projection import AlgebraicProjectionStatus, _project_algebraic_bidirectional

    _has_affect = has_affect
    _control_folds_theta = control_folds_theta
    _operational_controls = bool(getattr(problem, "has_operational_control", False))
    _recipe_controls = bool(getattr(problem, "has_recipe_controls", False))
    _recipe_controls = _recipe_controls or _operational_controls
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)
    dtype = ts_grid.dtype
    max_steps = int(cfg.max_steps)
    kind = getattr(engine, "trace_kind", "ndf")
    _events = _events_of(cfg.event)
    _has_projected_events = any(
        getattr(getattr(event, "state_policy", None), "mode", None) == "projected"
        for event in _events
    )
    _needs_projection = _has_projected_events or bool(cfg.epoch_projected_output)
    projection_options = _event_projection_options(engine) if _needs_projection else None
    output_plan = _output_plan(cfg, problem)
    needs_lam = output_plan.needs_lam

    # An engine whose ``attempt_step`` takes ``live`` is told which lanes the
    # loop keeps (see the engine protocol).
    takes_live = "live" in inspect.signature(type(engine).attempt_step).parameters

    def cond_fun(state: _DriverState) -> jax.Array:
        return jnp.logical_and(state.step < max_steps, jnp.logical_not(state.done))

    def _step_body(state: _DriverState, theta_local, live) -> _DriverState:
        lanes = {} if live is None else {"live": live}
        if record_decisions:
            tape = state.decisions
            slot = state.step
            replay = jnp.logical_and(tape.written[slot], jnp.logical_not(state.done))
            recorded = tuple(buffer[slot] for buffer in tape.values)
            new_carry, report = engine.attempt_step(
                state.engine_carry, decision=(replay, recorded), **lanes
            )
            report = _check_replay(report, tape, slot, replay)
            decisions = _record_decisions(tape, slot, report, state.done)
        else:
            new_carry, report = engine.attempt_step(state.engine_carry, **lanes)
            decisions = None
        accepted = jnp.asarray(report.accepted, jnp.bool_)
        t_new = report.t_new

        def interp_u(tq):
            y_t, _ = engine.dense(new_carry, tq)
            return y_t

        if n_lam > 0:

            def interp_lam(tq):
                y_full, _ = engine.dense_full(new_carry, tq)
                return y_full[n_u:]

        else:
            interp_lam = None
        emit = output_emitter(output_plan, problem, theta_local, dtype)
        # The bordered-row interpolant runs for the saved outputs only when
        # one of them reads the globals.
        if needs_lam:
            save_interp_lam = interp_lam
            if save_interp_lam is None:

                def save_interp_lam(tq):
                    return jnp.zeros((0,), dtype)

        else:
            save_interp_lam = None

        def output_rows(t, u, lam):
            return emit(t, u, lam if needs_lam else None)

        # Event resolution (skipped entirely on the default no-event path).
        # On an affect solve it runs BEFORE the grid fill so a mid-step switch
        # caps the grid claim at the crossing — the segment past the switch
        # belongs to the pre-switch dynamics and is integrated again by the
        # restarted engine.
        if cfg.event is not None:
            control = state.control_state if (_has_affect or _recipe_controls) else control0
            events = _events

            def control_at(tq):
                if _recipe_controls:
                    return problem.control_values(tq, theta_local, control)
                return control

            def projected_output_state(tq, active_control):
                """Project a sample before its control epoch leaves the loop carry."""
                interpolated, interpolated_dot = engine.dense_full(new_carry, tq)
                projected_control = (
                    problem.control_values(tq, theta_local, active_control)
                    if _recipe_controls
                    else active_control
                )
                return _project_algebraic_bidirectional(
                    problem,
                    t=tq,
                    state=interpolated,
                    state_dot=interpolated_dot,
                    seed=interpolated,
                    theta=theta_local,
                    control=projected_control,
                    options=projection_options,
                )

            def interp_full(tq):
                lam_tq = report.lam_new if n_lam == 0 else interp_lam(tq)
                return pack_state(interp_u(tq), lam_tq)

            def projected_event_state(event, tq):
                del event
                interpolated, interpolated_dot = engine.dense_full(new_carry, tq)
                return _project_algebraic_bidirectional(
                    problem,
                    t=tq,
                    state=interpolated,
                    state_dot=interpolated_dot,
                    seed=interpolated,
                    theta=theta_local,
                    control=control_at(tq),
                    options=projection_options,
                )

            def event_state_at(event, tq):
                policy = getattr(event, "state_policy", None)
                if policy is None or policy.mode == "fast":
                    return interp_full(tq)
                projected = projected_event_state(event, tq)
                succeeded = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
                return jnp.where(
                    succeeded, projected.state, jnp.full_like(projected.state, jnp.nan)
                )

            def event_state_status(event, tq):
                policy = getattr(event, "state_policy", None)
                if policy is None or policy.mode == "fast":
                    return jnp.asarray(True)
                return projected_event_state(event, tq).status == int(
                    AlgebraicProjectionStatus.SUCCESS
                )

            def selected_event_state(event_index, tq):
                selected = interp_full(tq)
                for declared_index, declared_event in enumerate(events):
                    candidate = event_state_at(declared_event, tq)
                    selected = jnp.where(event_index == declared_index, candidate, selected)
                return selected

            # Endpoint conditions read the EXACT accepted states — the loop
            # carry at ``t_prev`` and the just-accepted ``report`` at
            # ``t_new`` — never the one-step interpolant.  Accepted DAE states
            # are algebraically consistent, so endpoint bracketing needs no
            # projection even for a ``projected`` state policy. Mid-step root
            # candidates read every row from the engine interpolant; projection
            # is reserved for policies that require strict constraint
            # satisfaction. ``resolve_in_loop`` enters that arm only on an
            # accepted step whose endpoints bracket a crossing.
            def cond_lo_fn(cf):
                return jnp.asarray(
                    cf(
                        state.t_prev,
                        pack_state(state.y_last, state.lam_last),
                        None,
                        control=control_at(state.t_prev),
                    )
                )

            def cond_hi_fn(cf):
                return jnp.asarray(
                    cf(
                        t_new,
                        pack_state(report.y_new, report.lam_new),
                        None,
                        control=control_at(t_new),
                    )
                )

            fired, idx, t_star, ev_kind, ev_direction, event_evaluation_ok = resolve_in_loop(
                events,
                state.t_prev,
                t_new,
                cond_lo_fn,
                cond_hi_fn,
                interp_full,
                control=control,
                control_at=control_at if _recipe_controls else None,
                state_at=event_state_at if _has_projected_events else None,
                state_status=event_state_status if _has_projected_events else None,
                active=accepted,
            )
            # Root-finder tolerances are static Python values and may be
            # strong float64. Keep the crossing on the solve's numeric lane
            # before it enters restart state or loop-carry selections.
            t_star = jnp.asarray(t_star, dtype=dtype)
            event_projection_failed = jnp.logical_and(accepted, ~event_evaluation_ok)
            fired = jnp.logical_and(jnp.logical_and(accepted, fired), event_evaluation_ok)

        pre_output_projection_failed = jnp.asarray(False)
        if _has_affect:
            # Apply the fired event's affect at the crossing state. A
            # terminate-only member reports ``terminate`` instead, keeping
            # the stop semantics; the affect outputs are used only under
            # ``do_affect``.
            #
            # Build the crossing state only on the firing step: under the
            # ``projected`` state policy ``selected_event_state`` is a dense
            # interpolant evaluation plus a full Newton algebraic projection,
            # and running it branchlessly on every attempted step is the same
            # per-step leak the located arm and the terminate arm carried
            # before they were gated.  The non-firing branch feeds the accepted
            # endpoint state -- algebraically consistent by construction and a
            # valid in-domain point, the guarded-inputs rule of the restart
            # below -- and every consumer is already masked by ``do_affect`` /
            # ``log_fire``.  ``apply_affect`` itself stays branchless: it is
            # the user's traced affect arithmetic, cheap and previously
            # branchless too, and its ``terminate`` flag is value-independent.
            def _crossing_state(_):
                return selected_event_state(idx, t_star)

            def _endpoint_state(_):
                return pack_state(report.y_new, report.lam_new)

            u_star = lane_cond(fired, _crossing_state, _endpoint_state, None, false_is_free=True)
            affect_control = control if _operational_controls else control_at(t_star)
            u_prime, control_prime, terminate = apply_affect(
                events,
                idx,
                t_star,
                u_star,
                theta_local,
                affect_control,
                normalize_control=(
                    problem.normalize_control
                    if _recipe_controls and not _operational_controls
                    else None
                ),
            )
            if _recipe_controls and not _operational_controls:
                control_prime = state.control_state.with_values(control_prime)
            u_p, lam_p = unpack_state(u_prime, n_u)
            pre_epoch = _control_epoch(state.control_state)
            post_epoch = _control_epoch(control_prime)
            # A switch within the endpoint tolerance of ``t1`` has nothing
            # left to integrate: let the solve complete instead of restarting
            # a zero-length interval.
            before_end = t_star < t1 - _SAVE_TOL
            do_affect = jnp.logical_and(
                jnp.logical_and(fired, jnp.logical_not(terminate)), before_end
            )
            if _operational_controls:
                # An event condition can remain on the newly active control
                # manifold after a switch (for example, a voltage cutoff that
                # becomes the voltage-hold row). Numerical noise may then
                # report the same crossing again. A no-op operating update
                # keeps its epoch, so it must not restart the engine or
                # invalidate a factor generation a second time.
                epoch_changed = control_prime.epoch != state.control_state.epoch
                state_changed = jnp.any(u_prime != u_star)
                do_affect = jnp.logical_and(
                    do_affect,
                    jnp.logical_or(epoch_changed, state_changed),
                )
            event_stop = jnp.logical_and(fired, terminate)
            t_fill = jnp.where(jnp.logical_or(do_affect, event_stop), t_star, t_new)
            # One bounded log records every fire: a mutate-and-continue fire
            # holds the post-affect state, a terminate-member fire holds the
            # crossing (stopping) state.
            log_fire = jnp.logical_or(do_affect, event_stop)
            log_y = jnp.where(do_affect, u_prime, u_star)
            if cfg.epoch_projected_output:

                def _project_pre_log(_):
                    safe_t = jnp.where(log_fire, t_star, state.t_prev)
                    projected = projected_output_state(safe_t, state.control_state)
                    return projected.state, projected.status

                pre_log_y, pre_log_status = lane_cond(
                    log_fire,
                    _project_pre_log,
                    lambda _: (
                        u_star,
                        jnp.asarray(int(AlgebraicProjectionStatus.SUCCESS), jnp.int32),
                    ),
                    None,
                    false_is_free=True,
                )
                pre_output_projection_failed = jnp.logical_and(
                    log_fire,
                    pre_log_status != int(AlgebraicProjectionStatus.SUCCESS),
                )
            else:
                pre_log_y = u_star
                pre_output_projection_failed = jnp.asarray(False)
            appended = state.event_log.append(
                t=t_star,
                index=idx,
                kind=ev_kind,
                direction=ev_direction,
                y=log_y,
                pre_y=pre_log_y,
                pre_epoch=pre_epoch,
                post_epoch=jnp.where(do_affect, post_epoch, pre_epoch),
            )
            event_log = jax.tree_util.tree_map(
                lambda new, old: jnp.where(log_fire, new, old),
                appended,
                state.event_log,
            )
        else:
            if cfg.event is not None:
                event_stop = fired
                # A terminating crossing stops the output at the crossing:
                # the remainder of the step belongs to a trajectory the solve
                # rejected by stopping (a discrete stop keeps the boundary,
                # where t_star equals the step end).
                t_fill = jnp.where(event_stop, t_star, t_new)
            else:
                t_fill = t_new

        # A step whose event-state projection failed cannot be accepted into
        # the returned trajectory: the solver does not know whether it crossed
        # a terminating surface. Stop at the prior accepted point instead.
        trajectory_accepted = (
            jnp.logical_and(accepted, jnp.logical_not(event_projection_failed))
            if cfg.event is not None
            else accepted
        )
        sample_control = state.control_state if (_has_affect or _recipe_controls) else control0
        sample_epoch = _control_epoch(sample_control)

        if cfg.event is not None:
            root_tolerance = _selected_event_tolerance(events, idx, state.t_prev, t_new, dtype)
            if not cfg.save_every:
                # Use the root tolerance for the single requested sample
                # allowed to collide with a terminal crossing. This prevents
                # a broader endpoint band from admitting post-termination
                # queries and then appending the earlier exact terminal time.
                next_query = jnp.searchsorted(ts_grid, t_star, side="right").astype(jnp.int32)
                safe_next_query = jnp.minimum(
                    next_query, jnp.asarray(int(ts_grid.shape[0]) - 1, jnp.int32)
                )
                next_query_collides = jnp.logical_and(
                    next_query < int(ts_grid.shape[0]),
                    jnp.abs(ts_grid[safe_next_query] - t_star) <= root_tolerance,
                )
                terminal_fill_bound = jnp.where(
                    next_query_collides, ts_grid[safe_next_query], t_star
                )
                t_fill = jnp.where(event_stop, terminal_fill_bound, t_fill)

        # A mutate-and-continue switch owns the pre-switch flow only up to
        # the crossing itself: the endpoint tolerance band must not claim
        # save times just beyond ``t*`` (they belong to the restarted
        # post-switch flow), so the crossing step's fill drops the band.
        if _has_affect:
            step_save_tol = jnp.where(
                jnp.logical_or(do_affect, event_stop), jnp.asarray(0.0, dtype), _SAVE_TOL
            )
        elif cfg.event is not None:
            step_save_tol = jnp.where(event_stop, jnp.asarray(0.0, dtype), _SAVE_TOL)
        else:
            step_save_tol = _SAVE_TOL
        capacity = int(state.ts_output.shape[0])
        slots = jnp.arange(capacity, dtype=jnp.int32)
        pending = state.pending
        if cfg.save_every:
            outputs = state.outputs
            save_cursor = state.save_cursor
            valid_count = state.valid_count
            ts_output = state.ts_output
            control_epochs = state.control_epochs
            fill_evals = jnp.asarray(0, jnp.int32)
            pre_owned = jnp.zeros((capacity,), dtype=bool)
            if _has_affect:
                collide = jnp.zeros((capacity,), dtype=bool)
        else:
            old_save_cursor = state.save_cursor
            outputs, save_cursor, fill_evals, pending = fill_saveat_window(
                ts_grid,
                state.outputs,
                cursor=state.save_cursor,
                t_prev=state.t_prev,
                t_acc=t_fill,
                accept=trajectory_accepted,
                interp_u=interp_u,
                emit=emit,
                save_tol=step_save_tol,
                window=min(int(ts_grid.shape[0]), _SAVE_WINDOW),
                interp_lam=save_interp_lam,
                pending=pending,
            )
            ts_output = fill_saveat_times(ts_grid, state.ts_output, save_cursor)
            valid_count = jnp.maximum(state.valid_count, save_cursor)
            newly_filled = jnp.logical_and(slots >= old_save_cursor, slots < save_cursor)
            control_epochs = jnp.where(newly_filled, sample_epoch, state.control_epochs)
            pre_owned = newly_filled
            if _has_affect:
                # A row at the switch is the right limit: only the restarted
                # epoch projects and writes it. EventLog retains the left limit.
                collide_query = jnp.logical_and(do_affect, jnp.abs(ts_grid - t_star) <= _SAVE_TOL)
                collide = (
                    jnp.zeros((capacity,), dtype=bool).at[: ts_grid.shape[0]].set(collide_query)
                )
                pre_owned = jnp.logical_and(pre_owned, jnp.logical_not(collide))
        sample_projection_failed = jnp.asarray(False)
        if cfg.epoch_projected_output:
            # Project only rows claimed by this accepted segment. A static
            # for-loop keeps reverse-mode support while its scalar conditional
            # avoids running a Newton solve for untouched grid slots.
            def _project_saved_row(slot, carried):
                buffers, failed = carried

                def _project(_):
                    safe_t = jnp.where(pre_owned[slot], ts_grid[slot], state.t_prev)
                    projected = projected_output_state(safe_t, state.control_state)
                    projected_y, projected_lam = unpack_state(projected.state, n_u)
                    succeeded = projected.status == int(AlgebraicProjectionStatus.SUCCESS)
                    rows = output_rows(safe_t, projected_y, projected_lam)
                    return (
                        tuple(
                            jnp.where(succeeded, buffer.at[slot].set(row), buffer)
                            for buffer, row in zip(buffers, rows)
                        ),
                        jnp.logical_or(failed, jnp.logical_not(succeeded)),
                    )

                return lane_cond(
                    pre_owned[slot], _project, lambda _: carried, None, false_is_free=True
                )

            outputs, sample_projection_failed = jax.lax.fori_loop(
                0,
                int(ts_grid.shape[0]),
                _project_saved_row,
                (outputs, sample_projection_failed),
            )
        output_projection_failed = jnp.logical_or(
            pre_output_projection_failed, sample_projection_failed
        )
        if cfg.epoch_projected_output:
            # A failed projected row invalidates the whole accepted segment.
            # Restore every aligned output leaf now; the final aggregate below
            # also covers a failure discovered while constructing the right
            # limit after restart.
            outputs = jax.tree_util.tree_map(
                lambda old, new: jnp.where(output_projection_failed, old, new),
                state.outputs,
                outputs,
            )
            ts_output = jnp.where(output_projection_failed, state.ts_output, ts_output)
            save_cursor = jnp.where(output_projection_failed, state.save_cursor, save_cursor)
            valid_count = jnp.where(output_projection_failed, state.valid_count, valid_count)
            control_epochs = jnp.where(
                output_projection_failed, state.control_epochs, control_epochs
            )
            trajectory_accepted = jnp.logical_and(
                trajectory_accepted, jnp.logical_not(output_projection_failed)
            )
        event_indices = state.event_indices
        event_kinds = state.event_kinds
        event_directions = state.event_directions
        result_codes = state.result_codes
        if _has_affect:
            # A save time colliding with a nonterminal event time records the
            # post-affect state (the pre-affect state stays in the event log).
            control_epochs = jnp.where(collide, post_epoch, control_epochs)

        if cfg.dense:
            # One node per accepted step, at slot ``accepts + 1``. A rejected
            # attempt writes its (discarded) values into the same pending
            # slot, which the eventually accepted attempt overwrites; the
            # node count never includes a pending slot.
            slot = state.accepts + 1
            node_t = jnp.where(trajectory_accepted, t_new, jnp.asarray(jnp.inf, dtype))
            node_y = pack_state(report.y_new, report.lam_new)
            node_yd = pack_state(report.ydot_new, jnp.zeros_like(report.lam_new))
            dn_t, dn_y, dn_yd = state.dense_nodes
            dense_nodes = (
                buffer_at_set(dn_t, slot, node_t),
                buffer_at_set(dn_y, slot, node_y),
                buffer_at_set(dn_yd, slot, node_yd),
            )
        else:
            dense_nodes = state.dense_nodes

        # Termination: an engine-fatal attempt stops with its own code; an
        # accepted step reaching t1 succeeds; an event overrides both. A
        # mid-step switch rolls the step back to the crossing, so that step
        # no longer reaches the end.
        attempt_failed = jnp.logical_and(
            report.result != _CODE_SUCCESS, report.result != _CODE_EVENT_STOP
        )
        if cfg.event is not None:
            attempt_failed = jnp.logical_or(
                attempt_failed,
                jnp.logical_or(event_projection_failed, output_projection_failed),
            )
            attempt_result = jnp.where(
                jnp.logical_or(event_projection_failed, output_projection_failed),
                jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
                report.result,
            )
        else:
            attempt_result = report.result
        reached_end = jnp.logical_and(trajectory_accepted, t_new >= t1 - _SAVE_TOL)
        post_output_projection_failed = jnp.asarray(False)
        if _has_affect:
            reached_end = jnp.logical_and(reached_end, jnp.logical_not(do_affect))
        term_code = jnp.where(
            attempt_failed,
            attempt_result,
            jnp.where(reached_end, jnp.asarray(_CODE_SUCCESS, jnp.int32), state.term_code),
        )
        done = jnp.logical_or(attempt_failed, reached_end)
        t_reached = jnp.where(trajectory_accepted, t_new, state.t_reached)
        y_last = jnp.where(trajectory_accepted, report.y_new, state.y_last)
        lam_last = jnp.where(trajectory_accepted, report.lam_new, state.lam_last)
        t_prev = jnp.where(trajectory_accepted, t_new, state.t_prev)

        if cfg.event is not None:
            # A terminal fire stops the solve at the event time and records
            # the winning event (declaration order) for the solution mask.
            # The exact stopping point is the crossing state — the remainder
            # of the accepted step belongs to a trajectory the stop rejected.
            term_code = jnp.where(event_stop, jnp.asarray(_CODE_EVENT_STOP, jnp.int32), term_code)
            done = jnp.logical_or(done, event_stop)
            t_reached = jnp.where(event_stop, t_star, t_reached)
            event_idx = jnp.where(event_stop, idx, state.event_idx)
            event_kind = jnp.where(event_stop, ev_kind, state.event_kind)
            event_direction = jnp.where(event_stop, ev_direction, state.event_direction)
            # The crossing state is consumed only by the (at most one) step
            # whose terminal event fired, and constructing it evaluates the
            # dense interpolant — plus a full algebraic Newton projection for
            # a ``projected`` state policy — so it runs under ``lane_cond``
            # (under vmap, only when some lane's event fires): an always-on
            # build here cost one projection on EVERY attempted step (~10 s
            # of a 23 s full-scale cutoff solve). The skipped arm's value is
            # never read.
            u_stop = lane_cond(
                event_stop,
                lambda _: selected_event_state(idx, t_star),
                lambda _: pack_state(report.y_new, report.lam_new),
                None,
                false_is_free=True,
            )
            y_stop, lam_stop = unpack_state(u_stop, n_u)
            y_last = jnp.where(event_stop, y_stop, y_last)
            lam_last = jnp.where(event_stop, lam_stop, lam_last)

            terminal_slot, terminal_collides, terminal_room = terminal_sample_slot(
                ts_output,
                valid_count,
                t_star,
                root_tolerance,
            )
            safe_terminal_slot = jnp.minimum(terminal_slot, jnp.asarray(capacity - 1, jnp.int32))
            write_terminal = jnp.logical_and(event_stop, terminal_room)

            def _write_terminal(buffer, value):
                written = buffer.at[safe_terminal_slot].set(value)
                return jnp.where(write_terminal, written, buffer)

            ts_output = _write_terminal(ts_output, t_star)
            outputs, pending = place_saveat_sample(
                outputs,
                pending,
                safe_terminal_slot,
                output_rows(t_star, y_stop, lam_stop),
                write_terminal,
            )
            control_epochs = _write_terminal(control_epochs, sample_epoch)
            event_indices = _write_terminal(event_indices, idx)
            event_kinds = _write_terminal(event_kinds, ev_kind)
            event_directions = _write_terminal(event_directions, ev_direction)
            result_codes = _write_terminal(result_codes, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
            valid_count = jnp.where(
                write_terminal,
                valid_count + jnp.logical_not(terminal_collides).astype(jnp.int32),
                valid_count,
            )

            # Record every fire in the bounded log: a terminal fire holds the
            # crossing (stopping) state; a mutate-and-continue fire holds the
            # post-affect state and is appended below, where that state
            # exists.
            if not _has_affect:
                appended = state.event_log.append(
                    t=t_star,
                    index=idx,
                    kind=ev_kind,
                    direction=ev_direction,
                    y=u_stop,
                    pre_y=u_stop,
                    pre_epoch=sample_epoch,
                    post_epoch=sample_epoch,
                )
                event_log = jax.tree_util.tree_map(
                    lambda new, old: jnp.where(event_stop, new, old),
                    appended,
                    state.event_log,
                )
            # With an affect in play the log was already updated where the
            # post-affect state exists (the affect block above).
        else:
            event_idx = state.event_idx
            event_kind = state.event_kind
            event_direction = state.event_direction
            event_log = state.event_log

        if _has_affect:
            # The switch: the accepted endpoint rolls back to the crossing
            # and the engine restarts there from the affected state, through
            # the same entry that built the first carry at ``t0`` — so the
            # engine's own startup (fresh step size, first order, consistent
            # initialization) applies at the switch.
            theta_re = control_prime if _control_folds_theta else theta_local

            # On lanes where no event fires (``do_affect`` is False under a
            # batched solve, where the restart runs for the whole batch once
            # any lane fires and each lane keeps its own branch) feed the restart the lane's own last
            # accepted state -- a valid in-domain point -- so its forward stays
            # finite; the branch select below discards these values there.
            # Guarding the inputs (rather than stop-gradient-ing the discarded
            # outputs) is what removes the ``0 * inf`` the branch's own backward
            # would otherwise form at a degenerate restart point.
            t_star_g = jnp.where(do_affect, t_star, state.t_prev)
            u_p_g = jnp.where(do_affect, u_p, state.y_last)
            lam_p_g = jnp.where(do_affect, lam_p, state.lam_last)
            # The restarted engine evaluates the model with the affected
            # control values; under the theta fold the rewritten values ride
            # ``theta_re`` instead and the engine's control stays ``None``.
            if _control_folds_theta:
                control_re = None
            else:
                control_re = jax.tree_util.tree_map(
                    lambda nw, od: jnp.where(do_affect, nw, od),
                    control_prime,
                    state.control_state,
                )

            def _restart_carry(_):
                # The engine freezes its internal time grid (step-size
                # controls carry no derivative), so a parameter-dependent
                # switch time cannot reach the post-switch trajectory through
                # the restart time itself. Transfer it through the state:
                # restarting at the gradient-frozen time from
                # ``u - (t* - stop_gradient(t*)) * du/dt`` is primal-identical
                # (the offset is exactly zero) and carries the switch-time
                # derivative via the flow-map identity
                # ``d y(t1) / d t0 = -Phi(t1, t0) f(t0, y0)`` — which, with
                # the crossing state's own tangent, realizes the switching
                # (saltation) correction ``(f_pre - f_post) * dt*``.
                t_frozen = jax.lax.stop_gradient(t_star_g)
                if _operational_controls:
                    # The new operating point owns the restart seed for its
                    # algebraic current. Evaluate the policy only on guarded
                    # lane values so inactive vmapped lanes cannot poison the
                    # selected branch with an out-of-domain default.
                    lam_start = problem.operational_seed_multipliers(
                        t_star_g,
                        u_p_g,
                        lam_p_g,
                        theta_re,
                        control_re,
                    )
                else:
                    lam_start = lam_p_g
                restart_state = pack_state(u_p_g, lam_start)
                # The crossing slope realizes the switching correction and
                # therefore must be the true post-affect flow
                # derivative at the affected state.
                slope_layout, slope_precision = initialization_policy(
                    problem, getattr(engine, "precision", None), mode="forward"
                )
                _, slope_reverse_precision = initialization_policy(
                    problem, getattr(engine, "precision", None), mode="reverse"
                )
                ydot_p = problem.startup_slope(
                    t_star_g,
                    restart_state,
                    theta_re,
                    control_re,
                    tol=ic_gate_tolerance(engine.atol, getattr(engine, "ic_tol", None)),
                    backend=getattr(engine, "backend", "auto"),
                    layout=slope_layout,
                    precision=slope_precision,
                    reverse_precision=slope_reverse_precision,
                )[:n_u]
                u_start = u_p_g - (t_star_g - t_frozen) * ydot_p
                # The entry step is searched over the affected state, which a
                # replay recomputes and may round differently; a replay enters
                # with the step the forward sweep's restart recorded.
                entry = {"step": (replay, tape.h_restart[slot])} if record_decisions else {}
                return engine.init_carry(
                    problem,
                    t_frozen,
                    t1,
                    u_start,
                    ydot_p,
                    lam_start,
                    theta_re,
                    cfg,
                    control=control_re,
                    **entry,
                )

            def _keep_carry(_):
                return new_carry

            new_carry = lane_cond(do_affect, _restart_carry, _keep_carry, None, false_is_free=True)
            if record_decisions:
                decisions = _record_restart_step(
                    decisions, slot, engine.step_at_entry(new_carry), state.done
                )
            post_y = u_p
            post_lam = lam_p
            if cfg.epoch_projected_output:
                # The restart owns the right limit. Read its entry state and
                # slope, then project once more under the post-affect control so
                # this remains correct even when consistent initialization was
                # disabled for the integration itself.
                if hasattr(engine, "state_at_entry"):
                    post_seed = engine.state_at_entry(new_carry)
                else:
                    post_seed = pack_state(u_p_g, lam_p_g)
                if hasattr(engine, "slope_at_entry"):
                    post_slope = engine.slope_at_entry(new_carry)
                else:
                    post_slope = jnp.zeros_like(post_seed)
                theta_post = (
                    jax.tree_util.tree_map(
                        lambda new, old: jnp.where(do_affect, new, old), theta_re, theta_local
                    )
                    if _control_folds_theta
                    else theta_local
                )
                post_projection_control = (
                    problem.control_values(t_star_g, theta_post, control_re)
                    if _recipe_controls
                    else control_re
                )
                post_projection = _project_algebraic_bidirectional(
                    problem,
                    t=t_star_g,
                    state=post_seed,
                    state_dot=post_slope,
                    seed=post_seed,
                    theta=theta_post,
                    control=post_projection_control,
                    options=projection_options,
                )
                post_output_projection_failed = jnp.logical_and(
                    do_affect,
                    post_projection.status != int(AlgebraicProjectionStatus.SUCCESS),
                )
                post_y, post_lam = unpack_state(post_projection.state, n_u)

                # A requested sample at the switch is one right-continuous row:
                # the left limit remains solely in EventLog.pre_ys.
                outputs = _write_collide_rows(
                    outputs, collide, output_rows(t_star, post_y, post_lam)
                )

                log_slot = jnp.minimum(
                    state.event_log.count,
                    jnp.asarray(int(state.event_log.ts.shape[0]) - 1, jnp.int32),
                )
                log_has_room = state.event_log.count < int(state.event_log.ts.shape[0])
                log_post_ys = state.event_log.ys.at[log_slot].set(pack_state(post_y, post_lam))
                event_log = eqx.tree_at(
                    lambda log: log.ys,
                    event_log,
                    jnp.where(
                        jnp.logical_and(do_affect, log_has_room),
                        log_post_ys,
                        event_log.ys,
                    ),
                )
            else:
                outputs = _write_collide_rows(outputs, collide, output_rows(t_star, u_p, lam_p))
            t_prev = jnp.where(do_affect, t_star, t_prev)
            t_reached = jnp.where(do_affect, t_star, t_reached)
            y_last = jnp.where(do_affect, post_y, y_last)
            lam_last = jnp.where(do_affect, post_lam, lam_last)
            # A mutate-and-continue fire also records the fired event's
            # declaration index, so a switched solve that then runs to the
            # horizon surfaces the last-fired affect in the solution's event
            # mask (matching the other engine families).
            event_idx = jnp.where(do_affect, idx, event_idx)
            control_state = jax.tree_util.tree_map(
                lambda nw, od: jnp.where(do_affect, nw, od),
                control_prime,
                state.control_state,
            )
        else:
            control_state = state.control_state

        if cfg.epoch_projected_output:
            projection_failed = jnp.logical_or(
                output_projection_failed, post_output_projection_failed
            )
            # Projection is part of accepting an output-bearing segment. A
            # failed row, left limit, or right limit rolls the whole attempt
            # back to its prior accepted point and reports the same public
            # inconsistent-initialization result used by projected events.
            term_code = jnp.where(
                projection_failed,
                jnp.asarray(_CODE_INCONSISTENT_IC, jnp.int32),
                term_code,
            )
            done = jnp.logical_or(done, projection_failed)
            t_prev = jnp.where(projection_failed, state.t_prev, t_prev)
            t_reached = jnp.where(projection_failed, state.t_reached, t_reached)
            y_last = jnp.where(projection_failed, state.y_last, y_last)
            lam_last = jnp.where(projection_failed, state.lam_last, lam_last)
            ts_output = jnp.where(projection_failed, state.ts_output, ts_output)
            outputs = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new), state.outputs, outputs
            )
            save_cursor = jnp.where(projection_failed, state.save_cursor, save_cursor)
            valid_count = jnp.where(projection_failed, state.valid_count, valid_count)
            control_epochs = jnp.where(projection_failed, state.control_epochs, control_epochs)
            event_indices = jnp.where(projection_failed, state.event_indices, event_indices)
            event_kinds = jnp.where(projection_failed, state.event_kinds, event_kinds)
            event_directions = jnp.where(
                projection_failed, state.event_directions, event_directions
            )
            result_codes = jnp.where(projection_failed, state.result_codes, result_codes)
            event_idx = jnp.where(projection_failed, state.event_idx, event_idx)
            event_kind = jnp.where(projection_failed, state.event_kind, event_kind)
            event_direction = jnp.where(projection_failed, state.event_direction, event_direction)
            event_log = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new),
                state.event_log,
                event_log,
            )
            control_state = jax.tree_util.tree_map(
                lambda old, new: jnp.where(projection_failed, old, new),
                state.control_state,
                control_state,
            )

        # Accuracy-limit guard: at the top of this attempt, form the
        # weighted-RMS norm of the last accepted solution against its
        # error weights and scale it by the unit roundoff -- the factor by
        # which the tolerances would have to grow for the requested accuracy
        # to be achievable at this magnitude (the guard fires when that
        # factor exceeds 1). The carry's ``ewt`` field -- the per-attempt
        # error-weight vector every engine in the suite carries under that
        # name -- duck-types whether this engine supports the guard, so no
        # engine file needs an edit. Every engine in the suite
        # refreshes ``state.engine_carry.ewt`` only on an accepted step, so it
        # always pairs with ``state.y_last``/``state.lam_last`` as the
        # last-accepted point's own weights. Gated on at least one prior
        # attempt so the very
        # first attempt always gets to run; this takes precedence over
        # whatever this attempt itself reports, matching the convention that
        # this accuracy check runs before a step is attempted. When the engine
        # carries a ``suppressalg`` policy (JIDA), the norm masks out
        # algebraic rows via the engine's resolved differential-row mask --
        # an explicit engine-level ``id`` override wins over the problem's
        # ``id_mask``, exactly as in the engine's own error test; JBDF/JNDF
        # never mask this norm, matching their own unmasked error tests.
        # Every input and the result are wrapped in ``stop_gradient``: this
        # is a termination diagnostic, not part of the differentiated
        # trajectory, and must add no traced leaf to the loop's tangent.
        if hasattr(state.engine_carry, "ewt"):
            uround = jnp.asarray(jnp.finfo(dtype).eps, dtype)
            prior_y = jax.lax.stop_gradient(pack_state(state.y_last, state.lam_last))
            ewt = jax.lax.stop_gradient(state.engine_carry.ewt)
            if getattr(engine, "suppressalg", False):
                resolve_mask = getattr(engine, "_id_mask", None)
                raw_mask = (
                    resolve_mask(problem, dtype)
                    if resolve_mask is not None
                    else jnp.asarray(problem.id_mask, dtype=dtype)
                )
                guard_mask = jax.lax.stop_gradient(raw_mask)
            else:
                guard_mask = None
            nrm = jax.lax.stop_gradient(wrms_norm(prior_y, ewt, mask=guard_mask))
            tolsf_now = jax.lax.stop_gradient(uround * nrm)
            too_much_acc = jnp.logical_and(state.step > 0, tolsf_now > 1.0)
        else:
            tolsf_now = state.tolsf
            too_much_acc = jnp.asarray(False)

        done = jnp.logical_or(done, too_much_acc)
        term_code = jnp.where(too_much_acc, jnp.asarray(_CODE_ACCURACY_LIMIT, jnp.int32), term_code)

        one = jnp.asarray(1, jnp.int32)
        acc_i = accepted.astype(jnp.int32)

        if cfg.save_every:
            # Accepted-step output is a direct prefix write of the trajectory
            # state the driver retained. In particular, a switch writes the
            # exact post-affect right limit at the crossing, while a terminal
            # crossing has already used the structural reserve above. The
            # final accepted state is retained even when its accept count is
            # not a multiple of the requested stride (or a later rejected
            # attempt exhausts the budget).
            next_accepts = state.accepts + acc_i
            periodic = jnp.logical_and(
                trajectory_accepted,
                jnp.equal(jnp.mod(next_accepts, int(cfg.save_every)), 0),
            )
            event_stopped = event_stop if cfg.event is not None else jnp.asarray(False)
            leaving_loop = jnp.logical_or(done, state.step + one >= max_steps)
            final_non_event = jnp.logical_and(
                jnp.logical_and(leaving_loop, jnp.logical_not(event_stopped)),
                next_accepts > 0,
            )
            want_step_sample = jnp.logical_or(periodic, final_non_event)
            previous_slot = jnp.clip(valid_count - 1, 0, capacity - 1)
            collides_previous = jnp.logical_and(
                valid_count > 0,
                jnp.abs(ts_output[previous_slot] - t_reached) <= _SAVE_TOL,
            )
            step_slot = jnp.where(collides_previous, previous_slot, valid_count)
            step_room = jnp.logical_or(collides_previous, valid_count < capacity)
            write_step = jnp.logical_and(want_step_sample, step_room)
            safe_step_slot = jnp.minimum(step_slot, jnp.asarray(capacity - 1, jnp.int32))
            final_sample_epoch = _control_epoch(
                control_state if (_has_affect or _recipe_controls) else control0
            )

            def _write_step(buffer, value):
                written = buffer.at[safe_step_slot].set(value)
                return jnp.where(write_step, written, buffer)

            ts_output = _write_step(ts_output, t_reached)
            outputs, pending = place_saveat_sample(
                outputs,
                pending,
                safe_step_slot,
                output_rows(t_reached, y_last, lam_last),
                write_step,
            )
            control_epochs = _write_step(control_epochs, final_sample_epoch)
            valid_count = jnp.where(
                write_step,
                valid_count + jnp.logical_not(collides_previous).astype(jnp.int32),
                valid_count,
            )
            save_cursor = valid_count

        if cfg.trace:
            eta = _safe_ratio(report.h_new, state.h_prev)
            header = StepTraceHeader(
                nst=state.step,
                t=t_new,
                h=report.h_new,
                k=jnp.asarray(report.k_new, jnp.int32),
                accepted=accepted,
                result_code=jnp.asarray(report.result, jnp.int32),
                newton_iters=jnp.asarray(report.newton_iters, jnp.int32),
                err_estimate=report.err_estimate,
                eta=eta,
            )
            body_rec = (
                report.trace_record if report.trace_record is not None else zero_record(kind, dtype)
            )
            record = eqx.tree_at(lambda r: r.header, body_rec, header)
            trace = write_trace(state.trace, state.step, record)
        else:
            trace = None

        return _DriverState(
            engine_carry=new_carry,
            t_prev=t_prev,
            h_prev=report.h_new,
            y_last=y_last,
            lam_last=lam_last,
            control_state=control_state,
            event_log=event_log,
            ts_output=ts_output,
            outputs=outputs,
            pending=pending,
            save_cursor=save_cursor,
            valid_count=valid_count,
            control_epochs=control_epochs,
            event_indices=event_indices,
            event_kinds=event_kinds,
            event_directions=event_directions,
            result_codes=result_codes,
            dense_nodes=dense_nodes,
            trace=trace,
            step=state.step + one,
            done=done,
            term_code=term_code,
            event_idx=event_idx,
            event_kind=event_kind,
            event_direction=event_direction,
            t_reached=t_reached,
            tolsf=tolsf_now,
            accepts=state.accepts + acc_i,
            rejects=state.rejects + (one - acc_i),
            trial_steps=state.trial_steps + one,
            newton_iters=state.newton_iters + jnp.asarray(report.newton_iters, jnp.int32),
            jac_refreshes=state.jac_refreshes + jnp.asarray(report.jac_refreshes, jnp.int32),
            factorizations=state.factorizations + jnp.asarray(report.factorizations, jnp.int32),
            linear_solves=state.linear_solves + jnp.asarray(report.linear_solves, jnp.int32),
            rhs_evals=state.rhs_evals + jnp.asarray(report.rhs_evals, jnp.int32),
            residual_evals=state.residual_evals + jnp.asarray(report.residual_evals, jnp.int32),
            precision_promotions=(
                state.precision_promotions
                if report.precision_promotions is None
                else state.precision_promotions
                + jnp.asarray(report.precision_promotions, jnp.int32)
            ),
            ir_refinement_steps=(
                state.ir_refinement_steps
                if state.ir_refinement_steps is None or report.ir_refinement_steps is None
                else state.ir_refinement_steps
                + jnp.asarray(report.ir_refinement_steps, jnp.result_type(float))
            ),
            fp64_fallbacks=(
                state.fp64_fallbacks
                if state.fp64_fallbacks is None or report.fp64_fallbacks is None
                else state.fp64_fallbacks + jnp.asarray(report.fp64_fallbacks, jnp.result_type(float))
            ),
            multirhs_solves=(
                state.multirhs_solves
                if report.multirhs_solves is None
                else state.multirhs_solves + jnp.asarray(report.multirhs_solves, jnp.int32)
            ),
            newton_failures=(
                state.newton_failures
                if report.newton_failures is None
                else state.newton_failures + jnp.asarray(report.newton_failures, jnp.int32)
            ),
            multirhs_columns=(
                state.multirhs_columns
                if report.multirhs_columns is None
                else state.multirhs_columns + jnp.asarray(report.multirhs_columns, jnp.int32)
            ),
            du_b_builds=(
                state.du_b_builds
                if report.du_b_builds is None
                else state.du_b_builds + jnp.asarray(report.du_b_builds, jnp.int32)
            ),
            schur_builds=(
                state.schur_builds
                if report.schur_builds is None
                else state.schur_builds + jnp.asarray(report.schur_builds, jnp.int32)
            ),
            # The incremental fill evaluates one bounded window per attempt
            # (the full grid only on a window overflow); count the work
            # actually performed.
            saveat_interpolations=state.saveat_interpolations + fill_evals,
            event_solves=state.event_solves
            + (do_affect.astype(jnp.int32) if _has_affect else jnp.asarray(0, jnp.int32)),
            decisions=decisions,
            outputs_buffered=state.outputs_buffered,
        )

    def body_fun(state: _DriverState) -> _DriverState:
        # Freeze finished lanes through the loop body. ``state.done`` is a
        # per-lane scalar bool -- ``False`` on every entry of an unbatched solve
        # (``cond_fun`` guarantees it), so the freeze and the output-select are
        # exact identities there. Under a batched loop a lane kept alive by a
        # slower sibling has ``done == True``; freezing its inexact inputs (and
        # the closure ``theta`` the affect path reads) keeps its now-meaningless
        # step from manufacturing cotangents, and the output-select reverts its
        # state to the finish point.
        finished = state.done
        frozen = _map_no_buffer(lambda leaf: _freeze_finished(finished, leaf), state)
        theta_local = jax.tree_util.tree_map(lambda leaf: _freeze_finished(finished, leaf), theta)
        # A batched loop keeps the body's result only where its condition holds.
        live = cond_fun(state) if takes_live else None
        stepped = _step_body(frozen, theta_local, live)
        return _select_no_buffer(finished, stepped, state)

    return cond_fun, body_fun


def _finalize(
    engine, problem, cfg, adjoint, has_affect, control_folds_theta, ts_grid, init_state, final
):
    """Package the loop's final state into the shared solution record."""
    _has_affect = has_affect
    _control_folds_theta = control_folds_theta
    dtype = ts_grid.dtype
    has_ewt = hasattr(init_state.engine_carry, "ewt")
    engine_carry0 = init_state.engine_carry

    # The loop exits either because an attempt terminated it (use that code) or
    # because the step budget ran out (max steps before t1).
    term_code = jnp.where(final.done, final.term_code, jnp.asarray(_CODE_MAX_STEPS, jnp.int32))
    result = result_from_code(term_code)

    # The realized linear backend (post-``auto`` routing), the linear solve
    # mode, and the loop strategy actually in effect, recorded as static
    # metadata so a consumer can prove which backend and precision a solve
    # ran against.
    _backend = getattr(engine, "backend", None)
    _linear_name = getattr(type(_backend), "backend_name", None) if _backend is not None else None
    _solve_mode = getattr(getattr(engine, "precision", None), "solve_mode", None)
    _adjoint_ckpts = getattr(adjoint, "checkpoints", None)
    _zero_f64 = jnp.zeros((), jnp.result_type(float))

    stats = SolverStats(
        accepts=final.accepts,
        rejects=final.rejects,
        trial_steps=final.trial_steps,
        newton_iters=final.newton_iters,
        jac_refreshes=final.jac_refreshes,
        factorizations=final.factorizations,
        linear_solves=final.linear_solves,
        rhs_evals=final.rhs_evals,
        residual_evals=final.residual_evals,
        active_set_iters=jnp.asarray(0, jnp.int32),
        active_set_changes=jnp.asarray(0, jnp.int32),
        # An engine that runs a consistent-initialization pre-pass stashes the
        # achieved algebraic/global-constraint residual on its initial carry;
        # an engine that does not leaves this at zero.
        ic_alg_residual_norm=jnp.asarray(
            getattr(engine_carry0, "ic_alg_residual_norm", 0.0), dtype
        ),
        t_reached=final.t_reached,
        status=result,
        precision_promotions=final.precision_promotions,
        multirhs_solves=final.multirhs_solves,
        newton_failures=final.newton_failures,
        multirhs_columns=final.multirhs_columns,
        du_b_builds=final.du_b_builds,
        schur_builds=final.schur_builds,
        saveat_interpolations=final.saveat_interpolations,
        ic_solves=jnp.asarray(int(bool(cfg.consistent_ic)), jnp.int32),
        event_solves=final.event_solves,
        ir_refinement_steps=(
            _zero_f64 if final.ir_refinement_steps is None else final.ir_refinement_steps
        ),
        fp64_fallbacks=(_zero_f64 if final.fp64_fallbacks is None else final.fp64_fallbacks),
        # The step the first attempt takes, read on the initial carry: the
        # engine's estimate or the dt0 override, after its clamps.
        first_step=(
            jax.lax.stop_gradient(jnp.asarray(engine.step_at_entry(engine_carry0), _zero_f64.dtype))
            if hasattr(engine, "step_at_entry")
            else _zero_f64
        ),
        # The step the next attempt would take, read on the final carry.
        last_step=(
            jax.lax.stop_gradient(
                jnp.asarray(engine.step_at_entry(final.engine_carry), _zero_f64.dtype)
            )
            if hasattr(engine, "step_at_entry")
            else _zero_f64
        ),
        linear_backend=_linear_name,
        adjoint=type(adjoint).__name__,
        adjoint_checkpoints=None if _adjoint_ckpts is None else int(_adjoint_ckpts),
        linear_solve_mode=_solve_mode,
    )

    # The held-back latest sample goes to its slot, and the discard row of
    # every buffer is dropped.
    output_plan = _output_plan(cfg, problem)
    capacity = int(final.ts_output.shape[0])
    stored = tuple(
        buffer[:capacity] for buffer in commit_pending_sample(final.outputs, final.pending)
    )
    by_name = dict(zip(output_plan.names, stored))
    ys = by_name.get("state")
    lambdas = by_name.get("lambdas")
    if lambdas is not None and lambdas.shape[-1] == 0:
        lambdas = None
    saved_outputs = SavedOutputs(output_plan.names, stored) if output_plan.requested else None

    # Fixed-shape buffers leave the compiled core untrimmed; run_engine slices
    # them to the recorded attempt count where the count is concrete.
    trace_out = final.trace if cfg.trace else None

    # One flag per event (declaration order): the event that terminated the
    # solve, or — on a mutate-and-continue solve that ran to the horizon — the
    # last affect that fired (all-False when nothing fired before the end).
    # Built from the in-loop winner index with traced ops, so it works inside
    # jit and under both differentiation modes.
    event_mask = None
    if cfg.event is not None:
        stopped = jnp.equal(term_code, jnp.asarray(_CODE_EVENT_STOP, jnp.int32))
        completed = jnp.equal(term_code, jnp.asarray(_CODE_SUCCESS, jnp.int32))
        fired_any = jnp.logical_and(final.event_idx >= 0, jnp.logical_or(stopped, completed))
        event_mask = tuple(
            jnp.logical_and(fired_any, final.event_idx == i)
            for i in range(len(_events_of(cfg.event)))
        )

    # The final operating protocol is surfaced only for an affect solve seeded
    # with an explicit control pytree (matching the other engine families); a
    # control channel folded from the parameters stays internal.
    control_state = final.control_state if (_has_affect and not _control_folds_theta) else None

    # The achieved tolerance scale factor, for an engine whose carry supports
    # the accuracy-limit guard; ``None`` (no info payload at all, matching
    # today's every-solve behavior) for one that does not.
    info = {"tolsf": final.tolsf} if has_ewt else None

    # The exact stopping point, independent of the requested output grid: an
    # event stop holds the crossing state, a completed solve the endpoint
    # state, a failed solve the last accepted point.
    terminal = TerminalRecord.empty(
        int(final.y_last.shape[0]), int(final.lam_last.shape[0]), dtype=dtype
    ).write(
        t=final.t_reached,
        y=final.y_last,
        lam=final.lam_last,
        result_code=term_code,
        control_epoch=_control_epoch(final.control_state),
        event_index=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_idx,
            jnp.asarray(-1, jnp.int32),
        ),
        event_kind=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_kind,
            jnp.asarray(-1, jnp.int32),
        ),
        event_direction=jnp.where(
            term_code == _CODE_EVENT_STOP,
            final.event_direction,
            jnp.asarray(-1, jnp.int32),
        ),
    )

    interpolation = None
    if cfg.dense and final.dense_nodes is not None:
        dn_t, dn_y, dn_yd = final.dense_nodes
        interpolation = CubicHermiteInterpolation(
            ts=dn_t, ys=dn_y, ydots=dn_yd, count=final.accepts + 1
        )

    # The record a later solve continues from: the final carry's, which a
    # later solve may continue only when this one reached its end time.
    history = None
    if cfg.keep_history:
        history = SolveHistory(
            continues=jnp.equal(term_code, jnp.asarray(_CODE_SUCCESS, jnp.int32)),
            state=engine.history_at_exit(final.engine_carry),
        )

    return build_solution(
        ts=final.ts_output,
        ys=ys,
        lambdas=lambdas,
        valid_mask=(jnp.arange(final.ts_output.shape[0], dtype=jnp.int32) < final.valid_count),
        valid_count=final.valid_count,
        control_epochs=final.control_epochs,
        event_indices=final.event_indices,
        event_kinds=final.event_kinds,
        event_directions=final.event_directions,
        result_codes=final.result_codes,
        stats=stats,
        result=result,
        info=info,
        interpolation=interpolation,
        trace=trace_out,
        control_state=control_state,
        event_mask=event_mask,
        event_log=final.event_log,
        terminal=terminal,
        alg=engine,
        prob=problem,
        history=history,
        outputs=saved_outputs,
    )


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


_COMPOSABLE_REVERSE_ERROR = (
    "Reverse-mode differentiation was applied directly to a solve running the "
    "composable forward-sensitivity mode (ForwardAdjoint(composable=True)). "
    "This mode computes forward sensitivities (jax.jvp / jax.jacfwd) and "
    "supports differentiating functions of those sensitivities, for example "
    "jax.jacrev(jax.jacfwd(solve)), or jax.grad of a scalar criterion of "
    "jax.jacfwd(solve). For a reverse-mode gradient of the solution itself, "
    "solve with adjoint=CheckpointAdjoint() instead."
)

_COMPOSABLE_NESTED_FORWARD_ERROR = (
    "A second forward-mode differentiation was applied to a solve running the "
    "composable forward-sensitivity mode (ForwardAdjoint(composable=True)). "
    "This mode supports one forward differentiation of the solve plus "
    "reverse-mode differentiation of the resulting sensitivities. For "
    "forward-over-forward compositions, solve with the default "
    "ForwardAdjoint() instead."
)

# The reverse-linearization check below recognises the partial-evaluation
# tracer through the public ``jax.interpreters`` module. If a JAX release
# stops exporting it, the check degrades gracefully: bare reverse mode on the
# composable atom then surfaces jax's own trace-time error instead of the
# public message above.
_JAXPR_TRACER = getattr(_partial_eval, "JaxprTracer", None)


def _is_reverse_linearize_tangent(leaf) -> bool:
    """Whether ``leaf`` is an unknown partial-eval tangent (reverse linearization).

    Reverse-mode differentiation linearizes a function with symbolic (unknown)
    tangent inputs before transposing; a concrete forward differentiation
    carries known tangent values. Batch wrappers are unwrapped so the check
    also catches ``vmap``-of-``grad`` compositions.
    """
    if _JAXPR_TRACER is None:
        return False
    depth = 0
    while leaf is not None and depth < 8:
        if isinstance(leaf, _JAXPR_TRACER):
            return not leaf.pval.is_known()
        leaf = getattr(leaf, "val", None)
        depth += 1
    return False


_FORWARD_ONLY_REVERSE_ERROR = (
    "Reverse-mode differentiation (jax.grad, jax.vjp, jax.jacrev) was applied to a "
    "solve running the forward-only adjoint, which is the default and supports "
    "jax.jvp and jax.jacfwd only. For a reverse-mode gradient, solve with "
    "adjoint=ds.CheckpointAdjoint(). For reverse mode over forward sensitivities, "
    "use adjoint=ds.ForwardAdjoint(composable=True)."
)

# An identity on the forward-only loop's input tangents. Forward mode passes
# through it unchanged. Reverse mode partially evaluates it with unknown
# tangents before it reaches the loop (top level and inside jit alike), and
# its rule raises the public message there instead of JAX's raw while-loop
# error. The transpose rule is a second line of defence.
_forward_only_tangent_p = _Primitive("diffsoup_forward_only_tangent")
_forward_only_tangent_p.def_impl(lambda x: x)
_forward_only_tangent_p.def_abstract_eval(lambda aval: aval)
_mlir.register_lowering(
    _forward_only_tangent_p, _mlir.lower_fun(lambda x: x, multiple_results=False)
)
_batching.primitive_batchers[_forward_only_tangent_p] = lambda args, dims: (
    _forward_only_tangent_p.bind(args[0]),
    dims[0],
)


def _forward_only_tangent_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    out = _forward_only_tangent_p.bind(x)
    return out, (dx if type(dx) is _ad.Zero else _forward_only_tangent_p.bind(dx))


def _forward_only_tangent_partial_eval(trace, tracer):
    if not tracer.pval.is_known():
        raise ValueError(_FORWARD_ONLY_REVERSE_ERROR)
    return trace.default_process_primitive(_forward_only_tangent_p, (tracer,), {})


def _forward_only_tangent_transpose(cotangent, x):
    raise ValueError(_FORWARD_ONLY_REVERSE_ERROR)


_ad.primitive_jvps[_forward_only_tangent_p] = _forward_only_tangent_jvp
_ad.primitive_transposes[_forward_only_tangent_p] = _forward_only_tangent_transpose
_partial_eval.custom_partial_eval_rules[_forward_only_tangent_p] = (
    _forward_only_tangent_partial_eval
)


def _forward_loop_impl(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The forward-only step loop: the plain early-exit ``jax.lax.while_loop``."""
    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )
    return jax.lax.while_loop(cond_fun, body_fun, init_state)


@eqx.filter_custom_jvp
def _forward_loop(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The forward-only step loop, with a forward rule that refuses reverse mode.

    The primal and the forward rule run the same plain loop, so forward-mode
    results are unchanged. The rule marks the input tangents so that reverse
    mode stops with a message naming the reverse adjoint.
    """
    return _forward_loop_impl(
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


@_forward_loop.def_jvp
def _forward_loop_jvp(primals, tangents):
    tangents = jax.tree_util.tree_map(
        lambda t: None if t is None else _forward_only_tangent_p.bind(t),
        tangents,
        is_leaf=lambda t: t is None,
    )
    final, d_final = eqx.filter_jvp(_forward_loop_impl, primals, tangents)
    return final, _canon_tangent(final, d_final)


def _canon_tangent(primal, tangent):
    """Normalize a tangent tree: arrays at every inexact leaf, ``None`` elsewhere.

    A checkpointed loop carry must keep one fixed structure across
    iterations, while ``filter_jvp`` prunes symbolically zero tangents to
    ``None``; materialising those to zeros (and forcing ``None`` at
    non-inexact leaves) restores the fixed structure.
    """

    def one(p, t):
        if eqx.is_inexact_array(p):
            return jnp.zeros_like(p) if t is None else t
        return None

    return jax.tree_util.tree_map(one, primal, tangent, is_leaf=lambda x: x is None)


def _composable_solve(
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
    """The composable forward-sensitivity solve.

    Three differentiable segments: the loop-state construction and the
    solution packaging are ordinary compiled functions (differentiable in
    both modes through their implicit-function-theorem atoms), while the
    step loop itself is `_composable_loop` — the atom whose forward
    rule makes ``jax.jacrev(jax.jacfwd(solve))`` compose. The undifferentiated
    solve runs the same early-exit primal loop as the default forward mode.
    """
    init_state = _composable_init(
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
    final = _composable_loop(
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    )
    return _composable_finalize(
        engine, problem, cfg, adjoint, has_affect, control_folds_theta, ts_grid, init_state, final
    )


_composable_init = eqx.filter_jit(_loop_init)


@eqx.filter_jit
def _composable_loop_primal(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The composable atom's primal: the early-exit forward step loop."""
    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )
    return jax.lax.while_loop(cond_fun, body_fun, init_state)


_composable_finalize = eqx.filter_jit(_finalize)


@eqx.filter_custom_jvp
def _composable_loop(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    init_state,
    ts_grid,
    t1,
    theta,
    control0,
):
    """The composable forward-sensitivity step-loop atom.

    The primal is the early-exit ``jax.lax.while_loop`` — full adaptivity,
    O(1) memory, the same loop an undifferentiated forward solve runs. The
    attached forward rule (`_composable_loop_jvp`) re-integrates the
    trajectory together with its directional tangent inside a checkpointed
    loop, so the sensitivities it produces support reverse-mode
    differentiation on top.
    """
    del adjoint  # read by the differentiation rule, not by the primal
    return _composable_loop_primal(
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


@_composable_loop.def_jvp
def _composable_loop_jvp(primals, tangents):
    """Forward rule: integrate the primal and its tangent together, checkpointed.

    Rejects, with public errors, the applications outside this mode's
    contract: reverse-mode differentiation of the solve itself, tangents for
    arrays embedded in the problem/solver objects, and per-step tracing under
    differentiation.
    """
    (
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    ) = primals
    (d_engine, d_problem, d_cfg, d_adjoint, _d_ha, _d_cf, *loop_tangents) = tangents

    for leaf in jax.tree_util.tree_leaves(tuple(loop_tangents)):
        if _is_reverse_linearize_tangent(leaf):
            raise ValueError(_COMPOSABLE_REVERSE_ERROR)
    if jax.tree_util.tree_leaves((d_engine, d_problem, d_cfg, d_adjoint)):
        raise ValueError(
            "The composable forward-sensitivity mode differentiates the solve "
            "with respect to its array inputs (the time span, the initial "
            "state, theta, the save grid, and the event control state); a "
            "tangent arrived for an array embedded in the problem or solver "
            "object instead. Route differentiated parameters through theta."
        )
    if cfg.trace:
        raise ValueError(
            "The per-step trace is not recorded while differentiating through "
            "the composable forward-sensitivity mode. Run the traced solve "
            "undifferentiated, or differentiate with trace=False."
        )

    final, _joint_final, d_final = _composable_loop_jvp_core(
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        (init_state, ts_grid, t1, theta, control0),
        tuple(loop_tangents),
    )
    # The joint loop's primal half is discarded: the primal output must reach
    # the caller through exactly one differentiable path (the primal-only
    # loop below), or an outer reverse pass double-counts the first-order
    # Hessian block of a criterion that reads both the solution and its
    # sensitivities. It is still returned from the core so tests can check
    # that both loops realized the same adaptive schedule.
    return final, d_final


@eqx.filter_jit
def _composable_loop_jvp_core(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    loop_primals,
    loop_tangents,
):
    """Trace-once body of the composable forward rule.

    Integrates the augmented ``(state, tangent)`` pair step-by-step inside a
    checkpointed while loop whose supported differentiation mode is exactly
    the reverse transposition that ``jacrev``-of-``jacfwd`` needs. The primal
    trajectory is recomputed alongside its tangent (no values are
    gradient-stopped); the adaptive controls remain integer/boolean state
    with no tangent, and the early-exit condition reads only the primal half
    of the carry.

    Returns ``(final, joint_final, d_final)``: the primal output, the joint
    loop's (discarded) primal half, and the tangent output. The primal output
    comes from a *separate* primal-only checkpointed loop rather than from
    the joint loop: when a custom-JVP rule returns both primal and tangent
    from one checkpointed loop carry, an outer
    reverse-mode linearization applies the primal output's cotangent twice
    (the equinox checkpointed loop's saved-residual replay retains a second
    reverse path to the joint carry's primal half), which double-counts the
    Gauss-Newton block of any criterion that reads the solution together
    with its sensitivities. Splitting the loops leaves exactly one reverse
    path to each output. The extra primal integration runs only when this
    rule fires (i.e. under differentiation) and has unbatched operands under
    an outer ``vmap``, so it runs once for all tangent seeds. Both loops are
    separately compiled programs of the same step math; the joint primal is
    returned so tests can gate that they realize the same adaptive schedule.
    """
    init_state, ts_grid, t1, theta, control0 = loop_primals
    d_init, d_ts, d_t1, d_theta, d_control0 = _canon_tangent(loop_primals, loop_tangents)

    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )

    def aug_body(carry):
        state, dstate = carry

        def step(state_in, ts_grid_in, t1_in, theta_in, control0_in):
            _, body_fun = _loop_fns(
                engine,
                problem,
                cfg,
                has_affect,
                control_folds_theta,
                ts_grid_in,
                t1_in,
                theta_in,
                control0_in,
            )
            return body_fun(state_in)

        new, d_new = eqx.filter_jvp(
            step,
            (state, ts_grid, t1, theta, control0),
            (dstate, d_ts, d_t1, d_theta, d_control0),
        )
        return new, _canon_tangent(new, d_new)

    def aug_cond(carry):
        return cond_fun(carry[0])

    checkpoints = getattr(adjoint, "checkpoints", None)
    checkpoints = int(checkpoints) if checkpoints is not None else None
    try:
        joint_final, d_final = checkpointed_while_loop(
            aug_cond,
            aug_body,
            (init_state, d_init),
            max_steps=int(cfg.max_steps),
            checkpoints=checkpoints,
        )
    except TypeError as exc:
        if "custom_vjp" in str(exc):
            raise ValueError(_COMPOSABLE_NESTED_FORWARD_ERROR) from exc
        raise
    final = checkpointed_while_loop(
        cond_fun,
        body_fun,
        init_state,
        max_steps=int(cfg.max_steps),
        checkpoints=checkpoints,
    )
    return final, joint_final, d_final


def _build_saveat_grid(saveat, t0, t1, dtype) -> jax.Array:
    """Return the query times from the shared save plan."""
    return _build_save_plan(saveat, t0, t1, dtype).query_ts


__all__ = [
    "Engine",
    "StepReport",
    "DriverConfig",
    "run_engine",
]
