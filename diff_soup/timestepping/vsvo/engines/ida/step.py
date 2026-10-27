"""Step assembly and driver wiring for the fully-implicit DAE engine.

This module assembles the landed step pieces -- the coefficient recurrence, the
predictor, the fully-implicit corrector, the local-error test, the accepted-step
controller, and the failure ladders -- into an [Engine][] the shared driver
runs. [IDAEngine][] owns the array-free static configuration; the carried
per-attempt state (the divided-difference history, the retry counters, the error
weights, the learnable parameters, and the problem it integrates) rides on
[IDAEngineCarry][].

The decomposition follows the driver contract: one [IDAEngine.attempt_step][]
is exactly one pass of the step-attempt retry loop: a converged, error-test-passing
attempt reports ``accepted=True`` and hands the driver the next order and step; a
corrector or error-test failure restores the history, walks the failure ladder, and
reports ``accepted=False`` -- either a retry sentinel (the driver re-invokes with the
shrunk step already in the carry) or a termination code. The driver's outer loop
counts accepted and rejected attempts.

Before the first step [IDAEngine.init_carry][] runs the mandatory
consistent-initialization gate when the problem has algebraic structure, seeds
the divided-difference history from the (projected) state and derivative, and
sizes the initial step from the standard 0.001*tdist heuristic with a
derivative-norm cap.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup._precision import Precision
from diff_soup.linear._krylov import exact_backend
from diff_soup.nonlinear.border_cache import initial_border_cache
from diff_soup.nonlinear.generations import certify_exact_root, initial_generations
from diff_soup.nonlinear.result import LinearizationDirective
from diff_soup.timestepping.consistent_ic.api import (
    ICRequest,
    ICState,
    InitMode,
    consistent_init,
)
from diff_soup.timestepping.consistent_ic.pin_differential import (
    ICConfig,
    _full_dense_topology,
    _structure,
)
from diff_soup.timestepping.vsvo.common.adaptive import (
    CAUSE_ESCALATION,
    CAUSE_FALSE_CONVERGENCE,
    CAUSE_STAGNATION,
    TOP_RUNG,
    advance_rung_state,
    initial_rung_state,
)
from diff_soup.timestepping.vsvo.common.backends import (
    analyze_sensitivity,
    initialization_policy,
    resolve_backend,
)
from diff_soup.timestepping.vsvo.common.contract import (
    EngineCapabilities,
    closure_moved,
    consume_consistent_init,
    ic_gate_tolerance,
    pack_state,
)
from diff_soup.timestepping.vsvo.common.decisions import replay_decisions
from diff_soup.timestepping.vsvo.common.driver import StepReport
from diff_soup.timestepping.vsvo.common.flash_controller import (
    FLASH_ERR_SEED,
    flash_step_control,
)
from diff_soup.timestepping.vsvo.common.linear_ops import (
    empty_held,
    problem_linear,
    state_block_size,
)
from diff_soup.timestepping.vsvo.common.trace import IDATraceRecord, StepTraceHeader
from diff_soup.timestepping.vsvo.common.weights import error_weight, wrms_norm
from diff_soup.utils import safe_div

from .coeffs import ida_set_coeffs
from .complete_step import ACTION_UNSET, ida_complete_step
from .dense import ida_get_solution
from .error import ida_test_error
from .failure import NonlinearFlag, ida_handle_nflag, ida_reset, kflag_to_result_code
from .nonlinear import IDANlsContext, ida_corrector_root
from .predict import ida_predict
from .restore import ida_restore
from .state import MAXORD, MXORDP1, IDAState, init_ida_state

# Driver step-result codes (see diff_soup.timestepping.vsvo.common.driver): 1 is
# the success sentinel the driver reads as "not terminal" -- a retry reports it so
# the driver re-invokes the engine with the shrunk step already in the carry.
_CODE_SUCCESS = 1

# Snap the internal time onto the endpoint when it lands within this many
# rounding errors of it.
_TN_SNAP_FUZZ = 100.0

# The endpoint clamp shortens the step to just shy of the endpoint by this many
# rounding errors.
_ENDPOINT_CLAMP_FUZZ = 4.0

# The endpoint save band. Must equal the driver's save tolerance (its
# ``t_new >= t1 - 1e-10`` finished-step test) so a step already within this band
# of the endpoint is treated as finished rather than re-clamped to a vanishing
# ``hh`` -- which would blow up the leading coefficient ``cj = -alphas/hh`` when
# a finished lane is re-run inside a batched loop.
_ENDPOINT_BAND = 1e-10

# The dense-output fuzz base the interpolant weights use, sized to the trace
# record's order-indexed vectors (which reserve one slot past the maximum order).


class _ConsistentInitAdapter(NamedTuple):
    """Common result shape consumed by the engine's NaN-poison gate.

    The Euler-Newton route reports ``residual_norm`` as the root-mean-square of
    the algebraic residual rows, while the default projection reports their
    absolute 2-norm. Both are compared against the same initialization
    tolerance, so on that shared value the Euler gate is the looser of the two
    by roughly the square root of the algebraic row count.
    """

    y: jax.Array
    ydot: jax.Array
    residual_norm: jax.Array
    converged: jax.Array


def _capabilities() -> EngineCapabilities:
    """The IDA-tracking engine's capability flags (every flag stated)."""
    return EngineCapabilities(
        # The factored iteration matrix persists across attempts and is
        # refactored only where a setup replaces it.
        supports_factor_reuse=True,
        supports_events=True,
        supports_fully_implicit=True,
        supports_global_constraints=True,
        supports_state_dependent_mass=True,
        supports_dense_output=True,
    )


def _pad_to(vec: jax.Array, length: int) -> jax.Array:
    """Right-pad (or truncate) ``vec`` to ``length`` along its single axis."""
    have = int(vec.shape[0])
    if have < length:
        return jnp.concatenate([vec, jnp.zeros((length - have,), vec.dtype)])
    return vec[:length]


def _where_state(cond: jax.Array, a: IDAState, b: IDAState) -> IDAState:
    """Select between two carried states leaf-by-leaf on a scalar predicate."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def _advance_linearization_provenance(generations, *, setups, converged, correction):
    """Advance the carried linearization generations for one corrector attempt.

    A fired setup re-assembles the iteration matrix at the attempt's
    predictor and rebuilds its numeric factor, so the point, matrix, and
    factor counters advance together; a gate-closed attempt leaves every
    counter in place. The returned certificate attests exactness at the
    accepted root: it authorizes factor reuse only when the attempt converged
    on the freshly assembled matrix without applying a Newton correction —
    the one case in which the accepted root coincides with the factorization
    point. Every other accepted step carries a certificate that never
    authorizes, so downstream consumers rebuild exact values at the root.

    Args:
        generations: The carry's generation record at attempt entry.
        setups: Whether the attempt performed at least one linear-solver
            setup (traced boolean).
        converged: The corrector's convergence predicate (traced boolean).
        correction: The converged Newton correction ``ee``; an exactly zero
            correction means the root equals the setup-time predictor.

    Returns:
        ``(generations, certificate)`` — the advanced record and the
        certificate issued at this attempt's acceptance decision.
    """
    bumped = generations.bump("point", "matrix", "factor")
    advanced = jax.tree_util.tree_map(
        lambda new, old: jnp.where(setups, new, old), bumped, generations
    )
    root_exact = jnp.logical_and(
        jnp.asarray(converged),
        jnp.logical_and(jnp.asarray(setups), jnp.all(correction == 0.0)),
    )
    return advanced, certify_exact_root(advanced, root_exact)


def _decision_part(decision, start, stop):
    """The ``[start, stop)`` slice of a recorded decision, or ``None``."""
    if decision is None:
        return None
    replay, recorded = decision
    return replay, tuple(recorded)[start:stop]


class IDAEngineCarry(eqx.Module):
    """The per-attempt carry threaded through the IDA-tracking engine.

    The divided-difference/order/step record
    [IDAState][diff_soup.timestepping.vsvo.engines.ida.state.IDAState]
    persists across
    accepted steps; the remaining fields are the step loop's own local state.
    ``ncf`` and ``nef`` are the per-step convergence- and error-test-failure
    counts the failure ladders bump and every accepted step resets. ``ewt`` is
    the error-weight vector, refreshed only after an accepted step and held
    constant across every retry of a step. ``tn`` is the current internal time
    at the top of an attempt. ``theta`` is the learnable-parameter pytree the
    residual reads (a runtime leaf, handed in once by the driver). The
    iteration matrix in effect, ``A = F_y + matrix_cj * F_ydot``, is carried
    as its factor (whose values are ``A``) beside the derivative partial
    ``F_ydot``: a gate-closed attempt solves with the held factor as it is,
    and a setup that fires only because the leading coefficient moved
    recovers the state Jacobian as ``A - matrix_cj * F_ydot``, so no separate
    copy of ``F_y`` is stored in the loop state (or in a reverse-mode
    checkpoint). ``problem`` rides as identity-hashable static
    configuration.

    Attributes:
        state: The carried divided-difference/order/step state.
        tn: The current internal time (top of the attempt = last accepted or
            restored time).
        ncf: The local convergence-failure count (``int32``; reset on accept).
        nef: The local error-test-failure count (``int32``; reset on accept).
        ewt: The error-weight vector, shape ``(n,)``.
        theta: The learnable-parameter pytree (a runtime leaf).
        nstlj: The step number at which the carried state Jacobian was last
            evaluated (``int32``). It persists with ``saved_blocks`` and
            bounds how long that state Jacobian may be reused.
        saved_blocks: The iteration matrix in effect, ``(held, B, C, D)``: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block (its factor, starting empty) and the
            global-coupling blocks. A gate-closed attempt reuses it as it is;
            a setup that fires only because the leading coefficient moved
            recomposes from it and ``saved_fydot`` at the step's exact leading
            coefficient, so the state Jacobian is not re-evaluated. Zero-seeded
            at the first call, where the setup gate fires fresh.
        saved_fydot: The derivative partial ``F_ydot`` of the matrix in effect.
        matrix_cj: The leading coefficient at which the iteration matrix in
            effect was last assembled: this attempt's ``cj`` whenever the
            corrector assembled a matrix (a setup the gate fired, or the
            fresh-Jacobian retry after a convergence failure, which assembles
            without moving the gate's ``cjold``), the carried value otherwise.
        ic_alg_residual_norm: The local- and global-algebraic residual norm the
            consistent-initialization gate achieved (reported in the stats).
        hmax_inv: The reciprocal maximum step size (``0`` for no bound).
        hmin: The minimum step size (``0`` for no bound).
        t1_end: The integration endpoint, used as the endpoint clamp target.
        problem: The solver-contract problem (identity-hashable static config).
        trace_enabled: Whether to build the per-attempt trace record.
        flash_mem: The flash controller's memory ``(err_prev, h_prev,
            last_accept, order_run)`` — the boundary-normalized error of the
            last attempt whose corrector converged, the step size of the
            last accepted step, whether the last attempt was accepted, and
            the count of consecutive accepted steps at the current order
            (the flash order-raise gate; capped like the reference
            constant-``(h, k)`` counter). ``None`` under the source
            controller, so a source-profile carry has no extra leaves.
    """

    state: IDAState
    tn: jax.Array
    ncf: jax.Array
    nef: jax.Array
    ewt: jax.Array
    theta: Any
    control: Any
    saved_blocks: tuple
    saved_fydot: jax.Array
    matrix_cj: jax.Array
    nstlj: jax.Array
    ic_alg_residual_norm: jax.Array
    hmax_inv: jax.Array
    hmin: jax.Array
    t1_end: jax.Array
    problem: Any = eqx.field(static=True)
    trace_enabled: bool = eqx.field(static=True, default=False)
    flash_mem: Any = None
    prec_state: Any = None
    # Linearization provenance: the generation record of the persisted blocks
    # and the exact-root certificate issued at the last acceptance decision.
    # Consumers gate factor reuse on the certificate, never on block identity.
    generations: Any = None
    root_certificate: Any = None
    # The generation-keyed border/Schur cache for a problem with
    # bordered-global rows, or ``None`` without them. Plain array leaves,
    # so each vmap lane keys and invalidates its own cache; a rebuilt carry
    # starts cold.
    border_cache: Any = None
    # Per-row absolute-tolerance vector consumed by the error-weight law when
    # the solve's tolerances were resolved against the problem layout, or
    # ``None`` for the engine's uniform scalar. Runtime data, so it rides the
    # carry (the engine itself stays static configuration).
    atol_rows: Any = None


class IDAHistory(eqx.Module):
    """What a later solve continues from: the IDA carry without the solve's configuration.

    Everything that persists from one step to the next inside a solve: the
    divided-difference, order and step record, the iteration matrix in effect
    (its held factor and global-coupling blocks) with the derivative partial,
    the leading coefficient and the step at which it was assembled, the
    generation record and border cache that key it, the controller memories,
    and the control the steps were taken under.

    Attributes:
        state: The carried [IDAState][diff_soup.timestepping.vsvo.engines.ida.state.IDAState].
        saved_blocks: The iteration matrix in effect, ``(held, B, C, D)``:
            the [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block, whose factors name the corrector's shared
            storage, and the global-coupling blocks.
        saved_fydot: The derivative partial ``F_ydot`` of the matrix in effect.
        matrix_cj: The leading coefficient of the iteration matrix in effect.
        nstlj: The step at which the saved state Jacobian was evaluated.
        generations: The generation record of the carried linearization.
        border_cache: The border/Schur cache, or ``None``.
        flash_mem: The flash controller's memory, or ``None``.
        prec_state: The adaptive precision state, or ``None``.
        control: The control the steps were taken under, or ``None``.
    """

    state: IDAState
    saved_blocks: tuple
    saved_fydot: jax.Array
    matrix_cj: jax.Array
    nstlj: jax.Array
    generations: Any
    border_cache: Any
    flash_mem: Any
    prec_state: Any
    control: Any


class IDAEngine(eqx.Module):
    """The implicit-DAE step kernel the shared driver runs.

    Holds the solve configuration: the linear backend, the tolerances, the
    algebraic-suppression and identification-mask policy, the
    consistent-initialization mode, the corrector loop shape, the Jacobian
    setup policy, and the capability flags. The problem, the learnable
    parameters, and the per-step arrays live on [IDAEngineCarry][]; the
    one exception is ``atol_rows``, a per-row absolute-tolerance vector that
    rides the compiled solve as data when the solve's tolerances were
    resolved against the problem layout.

    Attributes:
        backend: The linear solver used for the state-block factor / solve.
        rtol: Relative integration tolerance.
        atol: Scalar absolute integration tolerance. When the solve carries
            a per-row vector (via the driver configuration) this is its
            tightest row, serving the scalar gates.
        ic_tol: Scalar residual gate for consistent initialization, or
            ``None`` to reuse ``atol``.
        ic_rtol: Relative tolerance for consistent initialization, or
            ``None`` to reuse ``rtol``.
        suppressalg: Keep algebraic rows out of the local-error norms.
        id_override: An explicit differential-row mask (a tuple of floats) that
            overrides the problem's own ``id_mask``, or ``None``.
        ic_mode: The consistent-initialization mode -- ``"ya_ydp"`` solves the
            algebraic state and differential derivative through the
            pin-differential projection (a relinearizing Newton on the reduced
            algebraic block), ``"ya_ydp_euler"`` solves the same unknowns with
            the fixed-Jacobian Euler-Newton kernel, ``"y_fixed_ydot"``
            solves the full state at a fixed derivative, ``"assume_consistent"``
            skips the projection.
        loop_form: ``"fixed"`` (bounded freeze-on-done) or ``"adaptive"``
            (early-exit) corrector loop.
        jacobian_policy: ``"source"`` runs the coefficient-window setup gate;
            any other value refreshes the Jacobian every attempt.
        capabilities: The capability flags the driver validates against.
        controller_mode: ``"source"`` runs the reference accepted-step
            order/step controller; ``"flash"`` keeps the reference order
            decision but re-sizes the step on every accepted step — and on
            the first error-test rejection of a step — with the flash
            controller. Because the step size then moves on every accepted
            step, the order-raise gate reads a run-of-constant-order counter
            carried in the flash memory instead of the constant-``(h, k)``
            counter ``ns`` (which would never accumulate); ``ns`` itself and
            the coefficient recurrences it gates are untouched.
        controller_law: The flash controller's step-size law (``"pi"`` or
            ``"h211b"``); read only under ``controller_mode="flash"``.
        trace_kind: The per-step trace family name.
    """

    backend: Any = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    suppressalg: bool = eqx.field(static=True)
    id_override: Any = eqx.field(static=True)
    ic_mode: str = eqx.field(static=True)
    loop_form: str = eqx.field(static=True)
    jacobian_policy: str = eqx.field(static=True)
    capabilities: EngineCapabilities = eqx.field(static=True)
    controller_mode: str = eqx.field(static=True, default="source")
    controller_law: str = eqx.field(static=True, default="pi")
    trace_kind: str = eqx.field(static=True, default="ida")
    newton_tol_factor: float = eqx.field(static=True, default=1.0)
    precision: Precision = eqx.field(static=True, default_factory=Precision)
    jacobian_values_dtype: Any = eqx.field(static=True, default=None)
    ic_tol: Optional[float] = eqx.field(static=True, default=None)
    ic_rtol: Optional[float] = eqx.field(static=True, default=None)

    # -- helpers -----------------------------------------------------------
    def _id_mask(self, problem, dtype) -> jax.Array:
        """Resolve the differential-row mask (override tuple, else the problem)."""
        if self.id_override is not None:
            return jnp.asarray(self.id_override, dtype=dtype)
        return jnp.asarray(problem.id_mask, dtype=dtype)

    def _directive(self) -> LinearizationDirective:
        """The corrector setup policy: window-gated reuse (stock) or fresh."""
        if self.jacobian_policy == "source":
            return LinearizationDirective.reuse()
        return LinearizationDirective.fresh()

    # -- initial carry -----------------------------------------------------
    def init_carry(
        self, problem, t0, t1, y0, ydot0, lam0, theta, cfg, *, control=None, step=None, history=None
    ):
        """Build the first carry: afresh, or continuing an earlier solve's history.

        Without ``history`` this is the fresh start of
        [_fresh_carry][diff_soup.timestepping.vsvo.engines.ida.step.IDAEngine._fresh_carry].
        ``history`` is a [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
        whose ``state`` is an [IDAHistory][diff_soup.timestepping.vsvo.engines.ida.step.IDAHistory].
        When its ``continues`` flag is set and the new control leaves the
        algebraic closure at the carried state unchanged, the carry continues
        it (see
        [_continued_carry][diff_soup.timestepping.vsvo.engines.ida.step.IDAEngine._continued_carry]).
        When the flag is set and the closure changes (a jump of the target, a
        new mode: ``closure_moved``), the solve starts afresh at order one from
        the step the history carries; otherwise it starts afresh as without
        one. The flag may be traced.

        Raises:
            TypeError: If ``history`` holds the record of another solver.
        """
        if history is None:
            return self._fresh_carry(
                problem, t0, t1, y0, ydot0, lam0, theta, cfg, control=control, step=step
            )
        record = history.state
        if not isinstance(record, IDAHistory):
            raise TypeError(
                "history holds the record of another solver "
                f"({type(record).__name__}); a JIDA solve continues only the "
                "Solution.history of an earlier JIDA solve."
            )

        def fresh(first_step):
            fresh_cfg = dataclasses.replace(cfg, dt0=first_step)
            return self._fresh_carry(
                problem, t0, t1, y0, ydot0, lam0, theta, fresh_cfg, control=control, step=step
            )

        def continued(_):
            return self._continued_carry(problem, record, t0, t1, y0, lam0, theta, cfg, control)

        # A switch that changes the algebraic closure (a jump of the target, a
        # new mode) puts a kink in the solution: the solve then restarts at
        # order one from the step the history carries.
        continues = jnp.asarray(history.continues, dtype=jnp.bool_)
        moved = continues & closure_moved(
            problem,
            t0,
            self._continued_state(problem, record.state, y0, lam0),
            self._id_mask(problem, record.state.phi.dtype),
            theta,
            control,
            record.control,
            ic_gate_tolerance(self.atol, self.ic_tol),
        )
        first_step = jnp.where(moved, record.state.hh, jnp.asarray(cfg.dt0, record.state.hh.dtype))
        return jax.lax.cond(
            continues & jnp.logical_not(moved), continued, lambda _: fresh(first_step), None
        )

    def _continued_state(self, problem, st, y0, lam0):
        """The state a continued solve starts from.

        The differential rows come from ``(y0, lam0)``, the algebraic rows from
        the record's current state.
        """
        dtype = st.phi.dtype
        given = pack_state(jnp.asarray(y0, dtype=dtype), jnp.asarray(lam0, dtype=dtype))
        differential = self._id_mask(problem, dtype)
        return differential * given + (1.0 - differential) * st.phi[0]

    def _continued_carry(self, problem, record, t0, t1, y0, lam0, theta, cfg, control):
        """The carry of a solve that continues ``record`` from ``t0``.

        The record passes as it passes between two steps of one solve: the
        divided differences, the order, the error-estimate and corrector
        memory, the iteration matrix in effect (its held factor, whose first
        refactor makes the shared storage hold it again) with the derivative
        partial, the leading coefficient and the step it was assembled at,
        the generation record and border cache, and the controller memories.
        The differential rows of the current state come from ``(y0, lam0)``
        (the earlier solve's end state), the algebraic rows from the record.
        The first step is the step the earlier solve took before the one that
        landed on its end time. The solve's own configuration is this solve's:
        the time ``t0``, the endpoint ``t1``, the parameters, the control, the
        step bounds for the steps after the first, and the tolerance rows; the
        endpoint clamps the first step at the attempt, as it clamps every step. The
        error weights are recomputed from the state, the per-step failure
        counters start at zero, the exact-root certificate is off-root, and no
        initialization runs (its residual norm reads 0).
        """
        st = record.state
        dtype = st.phi.dtype
        current = self._continued_state(problem, st, y0, lam0)
        state = eqx.tree_at(lambda s: (s.phi, s.yy), st, (st.phi.at[0].set(current), current))
        # The earlier solve's last step was shortened to land on its end time.
        # The controller keeps a step until it can double or must shrink it, so
        # the step it chose after the shortened one would hold the solve off the
        # steps one solve through the switch takes. Take the step before the
        # shortened one, when the history holds one.
        before = st.psi[1] - st.psi[0]
        usable = (st.kused >= 1) & (before * st.hh > 0)
        state = eqx.tree_at(lambda s: s.hh, state, jnp.where(usable, before, st.hh))
        atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
        hmax = float(cfg.hmax)
        hmax_inv = jnp.asarray(1.0 / hmax if hmax > 0.0 else 0.0, dtype=dtype)
        hmin = jnp.asarray(float(cfg.hmin), dtype=dtype)
        return IDAEngineCarry(
            state=state,
            tn=jnp.asarray(t0, dtype=dtype),
            ncf=jnp.asarray(0, jnp.int32),
            nef=jnp.asarray(0, jnp.int32),
            ewt=error_weight(current, self.rtol, self.atol if atol_rows is None else atol_rows),
            theta=theta,
            control=control,
            saved_blocks=record.saved_blocks,
            saved_fydot=record.saved_fydot,
            matrix_cj=record.matrix_cj,
            nstlj=record.nstlj,
            ic_alg_residual_norm=jnp.zeros((), dtype=dtype),
            hmax_inv=hmax_inv,
            hmin=hmin,
            t1_end=jnp.asarray(t1, dtype=dtype),
            problem=problem,
            trace_enabled=bool(cfg.trace),
            flash_mem=record.flash_mem,
            prec_state=record.prec_state,
            generations=record.generations,
            root_certificate=certify_exact_root(record.generations, jnp.asarray(False)),
            border_cache=record.border_cache,
            atol_rows=atol_rows,
        )

    def _fresh_carry(
        self, problem, t0, t1, y0, ydot0, lam0, theta, cfg, *, control=None, step=None
    ):
        """Build the first-call carry: consistent-IC gate + initial-step heuristic.

        ``y0`` is the differential state ``u`` the driver unpacked, ``ydot0`` the
        driver's raw right-hand side on the differential rows, and ``lam0`` the
        initial bordered globals. When the problem carries algebraic structure and
        the gate is enabled, the state, bordered globals, and derivative are projected onto
        the manifold using the consistent-initialization solve; otherwise the
        derivative is taken mass-aware (``f / mass`` on the differential rows) and
        the state is used as given. The initial step follows the standard
        0.001*tdist first-call heuristic with a derivative-norm cap.

        ``step`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the entry step is the recorded one, the step
        the forward sweep's carry entered with, instead of a new search.
        """
        n_u = int(problem.n_u)
        n_lam = int(problem.n_lam)
        n = n_u + n_lam
        u0 = jnp.asarray(y0)
        dtype = u0.dtype
        lam0 = jnp.asarray(lam0, dtype=dtype)
        ydot0 = jnp.asarray(ydot0, dtype=dtype)
        t0 = jnp.asarray(t0, dtype=dtype)
        t1 = jnp.asarray(t1, dtype=dtype)
        uround = jnp.finfo(dtype).eps
        id_mask = self._id_mask(problem, dtype)

        _, _, _diff, _alg_index, needs_projection = _structure(problem)
        if (
            not bool(cfg.consistent_ic) or self.ic_mode == "assume_consistent"
        ) and needs_projection:
            raise ConfigurationError(
                "this problem has algebraic rows; "
                "consistent_ic=False and ic_mode='assume_consistent' are "
                "not supported for it. Enable consistent initialization."
            )
        # The projection factors this problem exactly as the corrector does,
        # so it is analyzed with the lowered problem's placement and the
        # accepted-root sensitivity precision this solve declared for each
        # derivative direction — never the backend's defaults.
        ic_layout, ic_precision = initialization_policy(problem, self.precision, mode="forward")
        _, ic_reverse_precision = initialization_policy(problem, self.precision, mode="reverse")
        if self.ic_mode in ("ya_ydp_euler", "y_fixed_ydot"):
            gate_tol = ic_gate_tolerance(cfg.atol, self.ic_tol)
            mode = (
                InitMode.ALG_AND_YDOT
                if self.ic_mode == "ya_ydp_euler"
                else InitMode.ALL_Y_GIVEN_YDOT
            )
            euler_ic = consistent_init(
                problem,
                ICState(y=u0, lam=lam0, ydot=ydot0),
                theta,
                ICRequest(
                    mode=mode,
                    t0=t0,
                    cfg=ICConfig(rtol=self.rtol, atol=gate_tol),
                    t1=t1,
                ),
                exact_backend(self.backend),
                layout=ic_layout,
                precision=ic_precision,
            )
            packed_y = pack_state(euler_ic.y, euler_ic.lam)
            packed_ydot = jnp.concatenate([euler_ic.ydot, jnp.zeros((n_lam,), dtype=dtype)])
            alg_norm = jnp.asarray(euler_ic.alg_residual_norm, dtype=dtype)
            converged = (
                jnp.all(jnp.isfinite(packed_y))
                & jnp.all(jnp.isfinite(packed_ydot))
                & jnp.isfinite(alg_norm)
                & (alg_norm <= jnp.asarray(gate_tol, dtype=dtype))
            )
            ic = _ConsistentInitAdapter(
                y=packed_y,
                ydot=packed_ydot,
                residual_norm=alg_norm,
                converged=converged,
            )
        else:
            ic = problem.consistent_initial_state(
                t0,
                pack_state(u0, lam0),
                theta,
                control,
                ydot0=jnp.concatenate([ydot0, jnp.zeros((n_lam,), dtype=dtype)]),
                tol=ic_gate_tolerance(self.atol, self.ic_tol),
                backend=self.backend,
                layout=ic_layout,
                precision=ic_precision,
                reverse_precision=ic_reverse_precision,
            )
        ic_y, ic_ydot = consume_consistent_init(ic)
        u_c = jnp.asarray(ic_y[:n_u], dtype=dtype)
        lam_c = jnp.asarray(ic_y[n_u:], dtype=dtype)
        alg_norm = jnp.asarray(ic.residual_norm, dtype=dtype)

        y_aug0 = pack_state(u_c, lam_c)
        # The consistent initialization computes derivatives for every
        # differential row, including differential globals in the bordered
        # partition; algebraic rows arrive as zeros. Consume it whole.
        ydot_aug0 = jnp.asarray(ic_ydot, dtype=dtype)

        # A per-row absolute tolerance arrives as a vector on the driver
        # configuration; it rides the carry so accepted-step weight refreshes
        # keep consuming it.
        atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
        ewt0 = error_weight(y_aug0, self.rtol, self.atol if atol_rows is None else atol_rows)

        hmax = float(cfg.hmax)
        hmax_inv = jnp.asarray(1.0 / hmax if hmax > 0.0 else 0.0, dtype=dtype)
        hmin = jnp.asarray(float(cfg.hmin), dtype=dtype)

        # Initial step: an explicit dt0 override is used as-is; otherwise the
        # 0.001*tdist heuristic with the derivative-norm cap. The derivative
        # norm is masked only under suppressalg.
        dt0 = jax.lax.stop_gradient(jnp.asarray(cfg.dt0, dtype=dtype))

        def _auto_initial_step(_):
            tdist = jnp.abs(t1 - t0)
            candidate = jnp.asarray(0.001, dtype=dtype) * tdist
            yp_mask = id_mask if self.suppressalg else None
            ypnorm = wrms_norm(ydot_aug0, ewt0, mask=yp_mask)
            candidate = jnp.where(
                ypnorm > 0.5 / candidate,
                safe_div(jnp.asarray(0.5, dtype=dtype), ypnorm),
                candidate,
            )
            return jnp.where(t1 < t0, -candidate, candidate)

        # Dynamic and mapped-safe: no Python float/bool conversion. The
        # realized schedule remains gradient-dead.
        hh = jax.lax.stop_gradient(
            jax.lax.cond(dt0 != 0.0, lambda _: dt0, _auto_initial_step, operand=None)
        )

        # hmax / hmin clamps, then the endpoint clamp against t1.
        rh = jnp.abs(hh) * hmax_inv
        hh = jnp.where(rh > 1.0, hh / jnp.where(rh > 1.0, rh, 1.0), hh)
        at_min = jnp.abs(hh) < hmin
        hh = jnp.where(at_min, hh * (hmin / jnp.where(at_min, jnp.abs(hh), 1.0)), hh)
        overshoot = (t0 + hh - t1) * hh > 0
        hh = jnp.where(overshoot, (t1 - t0) * (1.0 - _ENDPOINT_CLAMP_FUZZ * uround), hh)

        if step is not None:
            # A replayed restart enters with the step its forward sweep recorded.
            replay, recorded = step
            recorded = jax.lax.stop_gradient(jnp.asarray(recorded).astype(dtype))
            hh = jnp.where(replay, recorded, hh)

        state = init_ida_state(y_aug0, ydot_aug0, hh, n, dtype.type)

        # The flash controller's memory exists only under the flash profile;
        # a source-profile carry keeps the field empty (no extra leaves).
        if self.controller_mode == "flash":
            flash_mem = (
                jnp.asarray(FLASH_ERR_SEED, dtype=dtype),
                hh,
                jnp.asarray(True),
                jnp.asarray(0, jnp.int32),
            )
        else:
            flash_mem = None

        return IDAEngineCarry(
            state=state,
            tn=t0,
            ncf=jnp.asarray(0, jnp.int32),
            nef=jnp.asarray(0, jnp.int32),
            ewt=ewt0,
            theta=theta,
            control=control,
            saved_blocks=(
                empty_held(
                    problem_linear(self.backend, problem, precision=self.precision, dtype=dtype),
                    state_block_size(problem),
                    dtype,
                ),
                jnp.zeros((n_u, n_lam), dtype=dtype),
                jnp.zeros((n_lam, n_u), dtype=dtype),
                jnp.zeros((n_lam, n_lam), dtype=dtype),
            ),
            saved_fydot=problem.zero_state_partials(dtype)[1],
            matrix_cj=state.cj,
            nstlj=jnp.asarray(0, dtype=jnp.int32),
            ic_alg_residual_norm=alg_norm,
            hmax_inv=hmax_inv,
            hmin=hmin,
            t1_end=t1,
            problem=problem,
            trace_enabled=bool(cfg.trace),
            flash_mem=flash_mem,
            prec_state=(initial_rung_state() if self.precision.solve_mode == "adaptive" else None),
            generations=initial_generations(),
            root_certificate=certify_exact_root(initial_generations(), jnp.asarray(False)),
            # The border cache starts cold on every carry construction —
            # including an event restart, which rebuilds the carry here.
            border_cache=initial_border_cache(n_u, n_lam, dtype) if n_lam > 0 else None,
            atol_rows=atol_rows,
        )

    # -- one retry-loop pass ----------------------------------------------
    def decision_template(self):
        """Zero-valued decision inputs of one attempt, the shape a replay records.

        The convergence flag and the error-test outputs (failure flag,
        candidate order, the three order-local error norms and the scaled
        norm), the accepted branch's next order, step and step ratio (chosen
        from one more error norm), plus the precision-attribution flags under
        the adaptive mode. Everything the attempt decides is a function of
        these.
        """
        flag = jnp.zeros((), jnp.bool_)
        count = jnp.zeros((), jnp.int32)
        norm = jnp.zeros((), jnp.float32)
        base = (flag, flag, count, norm, norm, norm, norm, count, norm, norm)
        if self.precision.solve_mode == "adaptive":
            return base + (flag, flag, flag)
        return base

    def attempt_step(self, carry: IDAEngineCarry, decision=None, live=None):
        """Advance one pass of the step-attempt retry loop; report accept or fail.

        Runs the endpoint clamp, the coefficient recurrence, the predictor, the
        fully-implicit corrector, and the local-error test. A converged,
        error-test-passing attempt completes the step (new order/step, advanced
        history, refreshed weights) and reports ``accepted=True``. A corrector or
        error-test failure restores the history, walks the failure ladder, and
        reports ``accepted=False`` -- a retry sentinel when the ladder asks for
        another attempt (the shrunk step already threaded into the carry), else a
        termination code.

        ``decision`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the attempt uses the recorded decision inputs
        (see [decision_template][]) so it decides as the forward sweep did.

        ``live`` is ``None`` or the step loop's condition for this lane. Under
        ``vmap`` the corrector's branches then run for the batch only when a
        lane the loop keeps takes them; the attempt of such a lane is
        unchanged.
        """
        problem = carry.problem
        st = carry.state
        theta = carry.theta
        ewt = carry.ewt
        dtype = st.hh.dtype
        n_u = int(problem.n_u)
        uround = jnp.finfo(dtype).eps
        id_mask = self._id_mask(problem, dtype)
        hmax_inv = carry.hmax_inv
        hmin = carry.hmin
        t1 = carry.t1_end
        saved_t = carry.tn

        backend = self.backend
        # The problem's lowering-declared Layout is the single source of
        # truth for device placement: it places EVERY analysis of this solve
        # — each corrector rung's factor and the sensitivity pattern alike —
        # so a multi-GPU factorization request reaches the adjoint's
        # transpose factor too, never silently staying single-device.
        layout = getattr(problem, "layout", None)
        if layout is None:
            layout = Layout()
        adaptive = self.precision.solve_mode == "adaptive"
        # The corrector's routes: one per precision rung (two under the
        # adaptive ladder, which switches between them on the carried rung).
        linear = problem_linear(backend, problem, precision=self.precision, dtype=dtype)
        prec = carry.prec_state

        kk_idx = jnp.clip(st.kk, 0, MXORDP1 - 1)

        # Step 0 -- endpoint clamp on hh BEFORE the coefficients, so the clamped
        # step feeds the psi shift: shorten a step that would overshoot the
        # endpoint.
        hh_in = st.hh
        overshoot = (saved_t + hh_in - t1) * hh_in > 0
        # Clamp only strictly outside the endpoint band: a lane already at (or
        # within ``_ENDPOINT_BAND`` of) ``t1`` keeps its full step ``hh_in`` so
        # the leading coefficient ``cj = -alphas/hh`` stays finite when a
        # finished lane is re-run inside a batched loop. The sign-free band test
        # also holds for backward-in-time integration.
        outside_band = jnp.abs(t1 - saved_t) > _ENDPOINT_BAND
        do_clamp = jnp.logical_and(overshoot, outside_band)
        hh = jnp.where(do_clamp, (t1 - saved_t) * (1.0 - _ENDPOINT_CLAMP_FUZZ * uround), hh_in)

        # Step 2 -- advance the coefficients.
        (phi_star, psi, alpha, beta, sigma, gamma, cjlast, cj, ck, ns, alphas, alpha0) = (
            ida_set_coeffs(
                st.phi,
                st.psi,
                st.alpha,
                st.beta,
                st.sigma,
                st.gamma,
                st.cj,
                st.kk,
                st.kused,
                hh,
                st.hused,
                st.ns,
            )
        )

        # Step 3 -- advance the internal time, clamping onto the endpoint.
        tn = saved_t + hh
        tn = jnp.where((tn - t1) * hh > 0, t1, tn)

        # Step 4 -- predictor.
        yypredict, yppredict = ida_predict(phi_star, gamma, st.kk)

        # Step 5 -- fully-implicit corrector. The setup-gate memory rides on the
        # state; the new leading coefficient is handed in via cjlast/cj and ctx.cj.
        corr_state = eqx.tree_at(lambda s: (s.cjlast, s.cj), st, (cjlast, cj))
        ctx = IDANlsContext(
            t=tn,
            theta=theta,
            cj=cj,
            yypredict=yypredict,
            yppredict=yppredict,
            ewt=ewt,
            prec_rung=None if prec is None else prec.rung,
            rung_changed=None if prec is None else prec.entered,
            control=carry.control,
        )
        # The sensitivity solves of the per-step root derivative factor the
        # full-system matrix at the accepted root exactly — the tangent solve
        # the matrix, a reverse sweep its transpose, each under the precision
        # policy of its direction — through this engine's own backend; a
        # dense-mode plan hands it the full dense pattern. The symbolic
        # analysis is problem-static.
        full_topology = problem.full_topology
        if full_topology is None:
            full_topology = _full_dense_topology(int(problem.n))
        sens_analysis = analyze_sensitivity(
            backend,
            full_topology,
            layout=layout,
            precision=self.precision,
            mode="forward",
        )
        rev_analysis = analyze_sensitivity(
            backend,
            full_topology,
            layout=layout,
            precision=self.precision,
            mode="reverse",
        )
        # When algebraic rows participate in the error test, stale algebraic
        # couplings can cost more Newton work than they save. A supplied exact
        # state-partial callback is already the cheap assembly path, so do not
        # offer its saved partials for coefficient-only recomposition in that
        # configuration. Gate-closed attempts still reuse the factored matrix;
        # every setup simply evaluates the exact state partial fresh, matching
        # the historical IDA setup semantics.
        allow_partial_reuse = self.suppressalg or not problem.plan.has_exact_fy_values
        nls = ida_corrector_root(
            problem,
            backend,
            linear,
            sens_analysis,
            corr_state,
            ctx,
            self._directive(),
            self.loop_form,
            frozen_blocks=carry.saved_blocks,
            frozen_fydot=carry.saved_fydot,
            frozen_cj=carry.matrix_cj,
            prior_failure=(carry.ncf > 0) | (carry.nef > 0),
            nstlj=carry.nstlj,
            reverse_analysis=rev_analysis,
            generations=carry.generations,
            border_cache=carry.border_cache,
            allow_partial_reuse=allow_partial_reuse,
            newton_tol_factor=self.newton_tol_factor,
            jacobian_values_dtype=self.jacobian_values_dtype,
            detect_precision=adaptive,
            live=live,
        )

        # The leading-coefficient bookkeeping the setup gate would carry forward.
        cjold_prev = jnp.where(st.nst == 0, cj, st.cjold)
        cjratio_raw = cj / cjold_prev
        cjratio = jnp.where(nls.call_lsetup, jnp.asarray(1.0, dtype), cjratio_raw)
        cjold_new = jnp.where(nls.call_lsetup, cj, cjold_prev)

        # Step 6 -- local-error test.
        error_test_fail, knew, err_k, err_km1, err_km2, enorm_k = ida_test_error(
            ck, nls.ee, phi_star, sigma, st.kk, ewt, id_mask, self.suppressalg
        )
        did_converge = jnp.asarray(nls.converged, jnp.bool_)
        if adaptive:
            # Attribute failures to precision on a reduced rung (the NDF
            # engine's twin block): false convergence rejects the sweep, both
            # detector causes retry promoted at the SAME step, an escalation
            # on a healthy sweep promotes forward.
            reduced = prec.rung < TOP_RUNG
            prec_false_conv = jnp.asarray(nls.prec_false_conv) & reduced
            prec_stagnant = jnp.asarray(nls.prec_stagnant) & reduced
            lin_escalated = (
                (jnp.asarray(nls.lin_fallbacks) > 0) & reduced
                if nls.lin_fallbacks is not None
                else jnp.asarray(False)
            )
            did_converge = did_converge & jnp.logical_not(prec_false_conv)
        tested = (did_converge, error_test_fail, knew, err_k, err_km1, err_km2, enorm_k)
        tested = replay_decisions(_decision_part(decision, 0, 7), tested)
        did_converge, error_test_fail, knew, err_k, err_km1, err_km2, enorm_k = tested
        if adaptive:
            attributed = replay_decisions(
                _decision_part(decision, 10, 13), (prec_false_conv, prec_stagnant, lin_escalated)
            )
            prec_false_conv, prec_stagnant, lin_escalated = attributed
            prec_retry = prec_false_conv | (prec_stagnant & jnp.logical_not(did_converge))
        else:
            prec_retry = None
        accept = jnp.logical_and(did_converge, jnp.logical_not(error_test_fail))

        # Flash profile: one controller evaluation per attempt, driven by the
        # boundary-1 error the accept test uses, sizes both the accepted
        # step's successor and a first error-test rejection's retry. The
        # order-run counter is this attempt's view of "consecutive accepted
        # steps at the current order" (reset when the order moved, capped
        # like the reference constant-(h, k) counter): it feeds the
        # order-raise gate in place of ns, which never accumulates once the
        # step size moves every accepted step. Static dispatch: under the
        # source profile this branch is never built and the source paths are
        # unchanged.
        if self.controller_mode == "flash":
            (flash_err_prev, flash_h_prev, flash_last_accept, flash_k_run) = carry.flash_mem
            flash_err_now = jax.lax.stop_gradient(ck * enorm_k)
            flash_res = flash_step_control(
                flash_err_now,
                flash_err_prev,
                h=hh,
                h_prev=flash_h_prev,
                order=st.kk,
                last_accept=flash_last_accept,
                first_step=st.nst == 0,
                hmax_inv=hmax_inv,
                hmin=hmin,
                max_order=MAXORD,
                law=self.controller_law,
            )
            flash_k_run_now = jnp.where(
                st.kk == st.kused,
                jnp.minimum(flash_k_run + 1, st.kk + 2),
                jnp.asarray(1, jnp.int32),
            )

        # --- accept branch: complete the step ---
        st_acc_in = eqx.tree_at(
            lambda s: (
                s.phi,
                s.psi,
                s.alpha,
                s.beta,
                s.sigma,
                s.gamma,
                s.cj,
                s.cjlast,
                s.ns,
                s.hh,
                s.ee,
                s.yy,
                s.yp,
            ),
            st,
            (phi_star, psi, alpha, beta, sigma, gamma, cj, cjlast, ns, hh, nls.ee, nls.yy, nls.yp),
        )
        new_state_acc, phase_action_acc, _err_knew = ida_complete_step(
            st_acc_in,
            err_k,
            err_km1,
            knew,
            ewt,
            id_mask,
            self.suppressalg,
            hmax_inv,
            hmin,
            ns_for_raise_gate=(flash_k_run_now if self.controller_mode == "flash" else None),
        )
        next_step = replay_decisions(
            _decision_part(decision, 7, 10),
            (new_state_acc.kk, new_state_acc.hh, new_state_acc.eta),
        )
        new_state_acc = eqx.tree_at(
            lambda s: (s.cjold, s.cjratio, s.knew, s.ss, s.kk, s.hh, s.eta),
            new_state_acc,
            (cjold_new, cjratio, knew, nls.ss) + tuple(next_step),
        )
        decided = tested + tuple(next_step)
        if adaptive:
            decided = decided + tuple(attributed)
        # Flash profile: keep the reference order decision (the startup order
        # climb and the LOWER/MAINTAIN/RAISE rule inside the completed step,
        # with the raise gate reading the flash order-run counter) but
        # re-size the step on every accepted step with the flash controller
        # instead of the reference step-ratio deadband. The memory advance
        # sits below, after the accept flag is settled.
        if self.controller_mode == "flash":
            new_state_acc = eqx.tree_at(
                lambda s: (s.hh, s.eta),
                new_state_acc,
                (flash_res.h_new, flash_res.eta),
            )
            # The error memory refreshes on every attempt whose corrector
            # converged (a diverged attempt's estimate is not meaningful),
            # the step-size memory and the order-run counter on accepted
            # steps only, and the last-accept flag on every attempt — a
            # rejection suppresses growth on the next accepted step.
            flash_mem_next = (
                jnp.where(did_converge, flash_err_now, flash_err_prev),
                jax.lax.stop_gradient(jnp.where(accept, hh, flash_h_prev)),
                accept,
                jnp.where(accept, flash_k_run_now, flash_k_run),
            )
        else:
            flash_mem_next = None
        # The scaled local-error vector is not carried here: the driver reports
        # ck*enorm_k as the error estimate and exposes no consumer for a full
        # scaled-error vector, so it is never materialized.
        new_ewt = error_weight(
            new_state_acc.phi[0],
            self.rtol,
            self.atol if carry.atol_rows is None else carry.atol_rows,
        )
        tn_snapped = jnp.where(
            jnp.abs(tn - t1) <= _TN_SNAP_FUZZ * uround * (jnp.abs(tn) + jnp.abs(hh)), t1, tn
        )
        y_new_acc = new_state_acc.phi[0][:n_u]
        ydot_acc = nls.yp[:n_u]
        lam_acc = new_state_acc.phi[0][n_u:]

        # --- failure branch: restore, walk the ladder ---
        phi_restored, psi_restored = ida_restore(phi_star, psi, beta, st.kk, hh, ns)
        # A non-converged corrector is a recoverable nonlinear-solver failure on
        # the dense path (no fatal residual / linear-solve returns); the abort
        # branch maps it to a nonlinear-convergence termination. A converged
        # attempt that fails the error test walks the error-test ladder instead.
        nflag_nls = jnp.asarray(NonlinearFlag.NONLINEAR_SETUP_RECOVERABLE, jnp.int32)
        nflag_hn = jnp.where(
            did_converge, jnp.asarray(NonlinearFlag.ERROR_TEST_FAIL, jnp.int32), nflag_nls
        )
        # Flash profile: the first error-test rejection of a step retries at
        # the flash controller's rejection ratio (capped at 0.9) instead of
        # the reference shrink formula; the ladder's escalation rungs and
        # aborts are unchanged.
        kflag, kk_new_f, eta_new_f, hh_new_f, ncf_new, nef_new, phase_new = ida_handle_nflag(
            nflag_hn,
            err_k,
            err_km1,
            st.kk,
            knew,
            hh,
            st.eta,
            carry.ncf,
            carry.nef,
            hmin,
            eta_ef_override=(flash_res.eta_reject if self.controller_mode == "flash" else None),
        )
        if adaptive:
            # A precision-attributed failure retries at the promoted rung with
            # the SAME step, order, and phase, and does not count as a
            # convergence failure (the rung is monotone within a step, so
            # these retries are bounded).
            kflag = jnp.where(
                prec_retry, jnp.asarray(NonlinearFlag.PREDICT_AGAIN, jnp.int32), kflag
            )
            kk_new_f = jnp.where(prec_retry, st.kk, kk_new_f).astype(kk_new_f.dtype)
            eta_new_f = jnp.where(prec_retry, jnp.asarray(1.0, dtype), eta_new_f)
            hh_new_f = jnp.where(prec_retry, hh, hh_new_f)
            ncf_new = jnp.where(prec_retry, carry.ncf, ncf_new).astype(jnp.int32)
            phase_new = jnp.where(prec_retry, st.phase, phase_new).astype(phase_new.dtype)
        predict_again = kflag == NonlinearFlag.PREDICT_AGAIN
        do_reset = jnp.logical_and(predict_again, st.nst == 0)
        phi_reset, psi_reset = ida_reset(phi_restored, psi_restored, hh_new_f, eta_new_f)
        phi_f = jnp.where(do_reset, phi_reset, phi_restored)
        psi_f = jnp.where(do_reset, psi_reset, psi_restored)
        # cjold and the Newton rate memory ss are persistent solver state, not
        # step history, so a rejected attempt carries them forward exactly as an
        # accepted one does -- cjold is this attempt's cj when a setup fired, and
        # ss is the attempt's final rate memory the retry reads.
        st_fail = eqx.tree_at(
            lambda s: (
                s.phi,
                s.psi,
                s.kk,
                s.hh,
                s.eta,
                s.phase,
                s.cj,
                s.cjlast,
                s.ns,
                s.cjold,
                s.ss,
            ),
            st,
            (
                phi_f,
                psi_f,
                kk_new_f,
                hh_new_f,
                eta_new_f,
                phase_new,
                cj,
                cjlast,
                ns,
                cjold_new,
                nls.ss,
            ),
        )
        result_fail = jnp.where(predict_again, _CODE_SUCCESS, kflag_to_result_code(kflag))

        # --- select the branch ---
        new_state = _where_state(accept, new_state_acc, st_fail)
        if adaptive:
            # Attribute this attempt's promotion cause and advance the policy.
            none_cause = jnp.asarray(-1, jnp.int32)
            cause = jnp.where(
                prec_false_conv,
                jnp.asarray(CAUSE_FALSE_CONVERGENCE, jnp.int32),
                jnp.where(
                    prec_stagnant & jnp.logical_not(did_converge),
                    jnp.asarray(CAUSE_STAGNATION, jnp.int32),
                    jnp.where(lin_escalated, jnp.asarray(CAUSE_ESCALATION, jnp.int32), none_cause),
                ),
            )
            prec_next = advance_rung_state(prec, cause=cause, accepted=accept)
            promo_delta = prec_next.promo_counts - prec.promo_counts
        else:
            prec_next = None
            promo_delta = None
        provenance, certificate = _advance_linearization_provenance(
            carry.generations,
            setups=nls.nsetups > 0,
            converged=nls.converged,
            correction=nls.ee,
        )
        new_carry = IDAEngineCarry(
            state=new_state,
            tn=jnp.where(accept, tn_snapped, saved_t),
            ncf=jnp.where(accept, jnp.asarray(0, jnp.int32), ncf_new),
            nef=jnp.where(accept, jnp.asarray(0, jnp.int32), nef_new),
            ewt=jnp.where(accept, new_ewt, ewt),
            theta=theta,
            control=carry.control,
            # The factored iteration matrix persists across attempts and
            # accepted steps with its derivative partial and the leading
            # coefficient it was assembled at: the corrector reports the
            # matrix in effect at exit (refactored where it assembled, the
            # carried factor otherwise). The corrector assembles on a gate
            # setup or on its fresh-Jacobian retry (``nsetups`` counts both);
            # the retry does not move the gate's ``cjold``.
            saved_blocks=nls.blocks_out,
            saved_fydot=nls.saved_fydot_out,
            matrix_cj=jnp.where(nls.nsetups > 0, cj, carry.matrix_cj),
            nstlj=nls.nstlj_out,
            ic_alg_residual_norm=carry.ic_alg_residual_norm,
            hmax_inv=carry.hmax_inv,
            hmin=carry.hmin,
            t1_end=carry.t1_end,
            problem=problem,
            trace_enabled=carry.trace_enabled,
            flash_mem=flash_mem_next,
            prec_state=prec_next,
            generations=provenance,
            root_certificate=certificate,
            border_cache=nls.border_cache_out,
            atol_rows=carry.atol_rows,
        )

        err_estimate = ck * enorm_k
        result_code = jnp.where(accept, jnp.asarray(_CODE_SUCCESS, jnp.int32), result_fail).astype(
            jnp.int32
        )

        trace_record = None
        if carry.trace_enabled:
            trace_record = self._trace_record(
                st,
                ns,
                psi,
                alpha,
                sigma,
                gamma,
                alphas,
                alpha0,
                ck,
                cjlast,
                cj,
                cjratio,
                nls,
                err_k,
                err_km1,
                err_km2,
                knew,
                enorm_k,
                accept,
                ncf_new,
                nef_new,
                new_state_acc,
                phase_action_acc,
                phase_new,
                kk_idx,
                dtype,
            )

        # Border/Schur accounting: the corrector reports the coupling-block
        # work it actually performed — the generation-keyed cache rebuilds a
        # tier only when its keys moved, so unchanged generations add nothing.
        # Jacobian accounting: every setup factorizes, but a setup that fires
        # only because the leading coefficient moved recomposes the iteration
        # matrix from the saved state Jacobian, so it refreshes nothing.
        report = StepReport(
            accepted=accept,
            t_new=jnp.where(accept, tn_snapped, saved_t),
            y_new=jnp.where(accept, y_new_acc, st.phi[0][:n_u]),
            ydot_new=jnp.where(accept, ydot_acc, st.yp[:n_u]),
            lam_new=jnp.where(accept, lam_acc, st.phi[0][n_u:]),
            h_new=jnp.where(accept, new_state_acc.hh, hh_new_f),
            k_new=jnp.where(accept, new_state_acc.kk, kk_new_f).astype(jnp.int32),
            err_estimate=err_estimate,
            result=result_code,
            newton_iters=nls.nni,
            jac_refreshes=nls.njac,
            factorizations=nls.nsetups,
            linear_solves=nls.nni,
            rhs_evals=nls.nni,
            residual_evals=nls.nni,
            trace_record=trace_record,
            precision_promotions=promo_delta,
            ir_refinement_steps=nls.lin_ir_steps,
            fp64_fallbacks=nls.lin_fallbacks,
            multirhs_solves=nls.du_b_builds,
            newton_failures=nls.nnf,
            # Each border rebuild solves one column per bordered-global row.
            multirhs_columns=nls.du_b_builds * jnp.asarray(int(problem.n_lam), jnp.int32),
            du_b_builds=nls.du_b_builds,
            schur_builds=nls.schur_builds,
            decisions=decided,
        )
        return new_carry, report

    # -- trace -------------------------------------------------------------
    def _trace_record(
        self,
        st,
        ns,
        psi,
        alpha,
        sigma,
        gamma,
        alphas,
        alpha0,
        ck,
        cjlast,
        cj,
        cjratio,
        nls,
        err_k,
        err_km1,
        err_km2,
        knew,
        enorm_k,
        accept,
        ncf_new,
        nef_new,
        new_state_acc,
        phase_action_acc,
        phase_new,
        kk_idx,
        dtype,
    ) -> IDATraceRecord:
        """Fill the engine's per-attempt trace fields (the driver adds the header).

        Records the coefficient outputs, the leading-coefficient triple and the
        setup decision, the corrector rate memory, the per-order error norms and
        the order choice, and the branch-selected failure counters and phase
        state.
        """
        zeros = IDATraceRecord.zeros(dtype)
        vlen = int(zeros.psi.shape[0])
        return IDATraceRecord(
            header=StepTraceHeader.zeros(dtype),
            ns=ns.astype(jnp.int32),
            psi=_pad_to(psi, vlen),
            alpha=alpha[kk_idx],
            sigma=sigma[kk_idx],
            gamma=_pad_to(gamma, vlen),
            alphas=alphas,
            alpha0=alpha0,
            ck=ck,
            cjlast=cjlast,
            cj=cj,
            cjratio=cjratio,
            call_lsetup=jnp.asarray(nls.call_lsetup, jnp.bool_),
            nsetups=nls.nsetups.astype(jnp.int32),
            ss=nls.ss,
            delnrm=nls.delnrm,
            rate=nls.rate,
            err_k=err_k,
            err_km1=err_km1,
            err_km2=err_km2,
            knew=knew.astype(jnp.int32),
            error_test_ck_enorm=ck * enorm_k,
            ncf=jnp.where(accept, jnp.asarray(0, jnp.int32), ncf_new).astype(jnp.int32),
            nef=jnp.where(accept, jnp.asarray(0, jnp.int32), nef_new).astype(jnp.int32),
            phase=jnp.where(accept, new_state_acc.phase, phase_new).astype(jnp.int32),
            phase_action=jnp.where(
                accept, phase_action_acc, jnp.asarray(ACTION_UNSET, jnp.int32)
            ).astype(jnp.int32),
            cvals=jnp.zeros(vlen, dtype),
            dvals=jnp.zeros(vlen, dtype),
        )

    def state_at_entry(self, carry: IDAEngineCarry):
        """The augmented ``[u; lam]`` state the solve starts from.

        Read on the freshly built carry, this is the consistent-initialized
        state (``phi[0]``; the driver records it as the ``t0`` output sample).
        """
        return carry.state.phi[0]

    def step_at_entry(self, carry: IDAEngineCarry):
        """The step size the carry's next attempt starts from.

        Read on a freshly built carry, this is the entry step ``init_carry``
        chose; the driver records it after an event restart so a replayed
        restart enters with the same step.
        """
        return carry.state.hh

    def history_at_exit(self, carry: IDAEngineCarry) -> IDAHistory:
        """The [IDAHistory][diff_soup.timestepping.vsvo.engines.ida.step.IDAHistory] a later solve continues from.

        Read on the final carry: everything that persists from one step to the
        next, without the solve's configuration.
        """
        return IDAHistory(
            state=carry.state,
            saved_blocks=carry.saved_blocks,
            saved_fydot=carry.saved_fydot,
            matrix_cj=carry.matrix_cj,
            nstlj=carry.nstlj,
            generations=carry.generations,
            border_cache=carry.border_cache,
            flash_mem=carry.flash_mem,
            prec_state=carry.prec_state,
            control=carry.control,
        )

    def slope_at_entry(self, carry: IDAEngineCarry):
        """The augmented state derivative the solve starts from.

        Read on the freshly built carry: the divided-difference history
        seeds ``phi[1] = h0 * ydot`` with ``psi[0] = h0``, so the entry
        slope divides the step back out.
        """
        st = carry.state
        return st.phi[1] / jnp.where(st.psi[0] != 0, st.psi[0], 1.0)

    # -- dense output ------------------------------------------------------
    def dense(self, carry: IDAEngineCarry, t):
        """Evaluate the differential state and derivative at ``t`` on the last step.

        Delegates to the dense interpolant over the accepted interval and returns
        the local state block. Callers that need bordered rows use
        :meth:`dense_full`, which also interpolates differential globals.
        """
        n_u = int(carry.problem.n_u)
        y_full, ydot_full = self.dense_full(carry, t)
        return y_full[:n_u], ydot_full[:n_u]

    def dense_full(self, carry: IDAEngineCarry, t):
        """Evaluate the full augmented interpolant (state and bordered rows) at ``t``.

        Every row is reported at interpolation accuracy. Strict algebraic
        constraint satisfaction is the projected output policy's job.
        Interpolated algebraic-row derivatives are internal only and must be
        zeroed if a public derivative surface is added.
        """
        return ida_get_solution(t, carry.tn, carry.state.phi, carry.state.psi, carry.state.kused)


def _build_ida_engine(
    profile,
    rtol,
    atol,
    linear,
    suppress_alg,
    id_override,
    ic_mode,
    newton_tol_factor=1.0,
    precision=Precision(),
    jacobian_values_dtype=None,
    ic_tol=None,
    ic_rtol=None,
):
    """Construct the engine (uncached).

    See [ida_engine][diff_soup.timestepping.vsvo.engines.ida.step.ida_engine].
    """
    backend = resolve_backend(linear)
    backend._validate_capabilities(Layout(), precision)
    return IDAEngine(
        backend=backend,
        rtol=float(rtol),
        atol=float(atol),
        suppressalg=bool(suppress_alg),
        id_override=id_override,
        ic_mode=str(ic_mode),
        loop_form=profile.nonlinear_mode,
        jacobian_policy=profile.jacobian_policy,
        capabilities=_capabilities(),
        controller_mode=profile.controller_mode,
        controller_law=profile.controller_law,
        newton_tol_factor=float(newton_tol_factor),
        precision=precision,
        jacobian_values_dtype=jacobian_values_dtype,
        ic_tol=None if ic_tol is None else float(ic_tol),
        ic_rtol=None if ic_rtol is None else float(ic_rtol),
    )


_ida_engine_cached = functools.lru_cache(maxsize=None)(_build_ida_engine)


def ida_engine(
    profile,
    *,
    rtol: float,
    atol: float,
    linear: Any,
    suppress_alg: bool = True,
    id: Optional[Any] = None,  # noqa: A002 -- matches the public JIDA field name.
    ic_mode: str = "ya_ydp",
    newton_tol_factor: float = 1.0,
    precision: Precision = Precision(),
    jacobian_values_dtype=None,
    ic_tol: Optional[float] = None,
    ic_rtol: Optional[float] = None,
) -> IDAEngine:
    """Build the implicit-DAE engine kernel from a solver profile.

    Resolves the linear backend from ``linear`` (the dense CPU backend for the
    default and ``"auto"`` names), reads the corrector loop shape, the Jacobian
    setup policy, and the accepted-step controller mode from ``profile``, and
    carries the algebraic-suppression flag, the identification-mask override,
    and the consistent-initialization mode through to the engine.

    Identical configurations return one memoised engine object, so repeated
    solves share the driver's compiled program instead of recompiling per
    construction. An unhashable ``id`` mask falls back to an uncached build.

    Args:
        profile: The solver execution profile (``nonlinear_mode`` selects the
            corrector loop shape; ``jacobian_policy`` selects the setup
            policy; ``controller_mode`` selects the accepted-step step-size
            controller).
        rtol: Relative integration tolerance.
        atol: Absolute integration tolerance.
        linear: Linear-backend name (``"auto"``, ``"cudss"``, ``"klu"``,
            ``"superlu"``, ``"dense"``) or a backend instance.
        suppress_alg: Keep algebraic rows out of the local-error norms.
        id: An explicit differential-row mask overriding the problem's own, or
            ``None``.
        ic_mode: The consistent-initialization mode (``"ya_ydp"``,
            ``"ya_ydp_euler"``, ``"y_fixed_ydot"``, ``"assume_consistent"``).
        newton_tol_factor: Corrector-band scale — widens the Newton
            convergence-test tolerance relative to the step tolerance
            (``1.0`` = the standard band).
        precision: Linear-solve precision policy handed to the backend's
            ``analyze``. A mode the backend does not declare is rejected
            here, eagerly.
        jacobian_values_dtype: Optional dtype for the Jacobian-values
            kernels (``None`` = the state dtype).
        ic_tol: Absolute tolerance of the consistent-initialization gate
            (``None`` = derived from ``atol``).
        ic_rtol: Relative tolerance of the consistent-initialization solve
            (``None`` = ``rtol``).

    Returns:
        The configured [IDAEngine][].
    """
    id_key = tuple(id) if isinstance(id, list) else id
    try:
        return _ida_engine_cached(
            profile,
            float(rtol),
            float(atol),
            linear if not isinstance(linear, str) else str(linear),
            bool(suppress_alg),
            id_key,
            ic_mode,
            float(newton_tol_factor),
            precision,
            jacobian_values_dtype,
            None if ic_tol is None else float(ic_tol),
            None if ic_rtol is None else float(ic_rtol),
        )
    except TypeError:
        # A per-row atol (or another unhashable input) cannot key the memo;
        # an uncached build with identical static configuration still shares
        # the driver's compiled program (jit keys on structure, not identity).
        return _build_ida_engine(
            profile,
            rtol,
            atol,
            linear,
            suppress_alg,
            id,
            ic_mode,
            newton_tol_factor,
            precision,
            jacobian_values_dtype,
            ic_tol,
            ic_rtol,
        )


__all__ = ["IDAEngine", "IDAEngineCarry", "IDAHistory", "ida_engine"]
