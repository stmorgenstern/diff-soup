"""Step assembly and driver wiring for the NDF engine.

Assembles the landed pieces — the coefficient tables, the basis-rescale and
fold operations, the correction-form Newton corrector, and the
accept/reject controllers — into an `Engine` the shared driver runs.
`NDFEngine` owns the array-free static configuration; the carried
per-attempt state rides on `NDFEngineCarry`.

One `NDFEngine.attempt_step` is exactly one pass of the step-attempt
retry loop. Every attempt derives its working difference table by rescaling
the last accepted step's basis table onto the attempt's step, so rejected
attempts never mutate history. A converged, error-test-passing attempt
folds the correction into a new basis and asks the accept controller for
the next order and step; an error-test rejection asks the reject
controller (which may force the order down or restart the history); a
Newton failure halves the step. The constant-step counter the accept
controller's preference window reads accumulates only across steps whose
size did not change, matching the rescale-resets-the-counter behavior of
the reference (QNDF, OrdinaryDiffEq.jl, MIT; see
``THIRD_PARTY_NOTICES.md``).

Before the first step `NDFEngine.init_carry` runs the mandatory
consistent-initialization gate when the problem has algebraic structure,
seeds the two-row difference table from the (projected) state and its
consistent derivative — a solver problem takes both from its joint
full-residual root — and sizes the initial step with the shared
second-derivative search.

Each attempt's corrector runs behind the per-step root derivative boundary
(see `ndf_corrector_root`): the accepted state carries the
exact implicit-function derivative. The carry names its held matrix/factor
with a generation record whose counters advance exactly with rebuild
events, and pairs it with a certificate that truthfully marks
provably-exact-at-root factors for downstream reuse decisions; the engine's
own reuse windows remain governed by its convergence monitoring.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup._precision import Precision
from diff_soup.nonlinear.border_cache import initial_border_cache
from diff_soup.nonlinear.generations import certify_exact_root, initial_generations
from diff_soup.nonlinear.result import LinearizationDirective
from diff_soup.timestepping.consistent_ic.pin_differential import (
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
from diff_soup.timestepping.vsvo.common.trace import NDFTraceRecord, StepTraceHeader
from diff_soup.timestepping.vsvo.common.weights import error_weight, wrms_norm
from diff_soup.utils.lanes import lane_cond

from ..bdf.dense import bdf_initial_step, bdf_slope_fn
from .controller import accept_step_control, reject_step_control
from .dense import ndf_interpolant
from .nonlinear import NDFNlsContext, ndf_corrector_root
from .state import NDFState, init_ndf_state
from .tables import (
    BETA0,
    D_ROWS,
    ERROR_CONST,
    GAMMA,
    MAX_ORDER,
    rescale_d,
    seed_d,
    update_d,
)

# Driver step-result codes (see diff_soup.timestepping.vsvo.common.driver):
# 1 is the non-terminal sentinel; 2 is step collapse; 3 nonlinear divergence.
_CODE_SUCCESS = 1
_CODE_STEP_COLLAPSE = 2
_CODE_NONLINEAR = 3

# nflag reason codes into the next attempt's corrector.
_FIRST_CALL = 0
_PREV_CONV_FAIL = 1
_PREV_ERR_FAIL = 2
# A precision-attributed corrector failure: retry at the promoted rung with a
# fresh setup, without shrinking the step.
_PREV_PREC_FAIL = 3

# The internal time snaps onto the endpoint within this many rounding errors.
_TN_SNAP_FUZZ = 100.0

# The endpoint clamp shortens a step to just shy of the endpoint by this
# many rounding errors.
_ENDPOINT_CLAMP_FUZZ = 4.0

# The step-size floor is this many spacings of the current time (the hard
# ``hmin`` bound still applies on top when set).
_FLOOR_SPACINGS = 10.0

# The step cut after a Newton (convergence) failure.
_NEWTON_FAIL_SHRINK = 0.5

# Consecutive Newton failures of one step before the attempt aborts (an own
# resource bound; the step floor usually triggers first).
_MAX_NEWTON_FAILS = 10

# Rows 0..k masks are built against this static index vector. Host numpy:
# the engines are imported lazily by the solver entry, so this module may
# first load inside an active jit trace, where a module-level jax operation
# would leak a tracer into the global.
_ROW_IDX = np.arange(D_ROWS)


# GAMMA padded to the table rows so the history term is one masked matvec.
_GAMMA_ROWS = np.concatenate([GAMMA, np.zeros(D_ROWS - GAMMA.shape[0])])


def _capabilities() -> EngineCapabilities:
    """The NDF engine's capability flags (every flag stated)."""
    return EngineCapabilities(
        # The factored iteration matrix persists across attempts and is
        # refactored only where a setup replaces it.
        supports_factor_reuse=True,
        supports_events=True,
        # This engine integrates the mass-matrix ODE/DAE form, not a
        # user-supplied truly-implicit residual F(t, y, ydot).
        supports_fully_implicit=False,
        supports_global_constraints=True,
        # The diagonal mass is re-evaluated at the current iterate inside the
        # residual; state-dependent mass is handled in weak (chord) form.
        supports_state_dependent_mass=True,
        supports_dense_output=True,
    )


def _where_state(cond: jax.Array, a: NDFState, b: NDFState) -> NDFState:
    """Select between two carried states leaf-by-leaf on a scalar predicate."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


class NDFEngineCarry(eqx.Module):
    """The per-attempt carry threaded through the NDF engine.

    Attributes:
        state: The carried table/order/step state.
        tn: The current internal time (the last accepted time).
        ncf: Consecutive Newton failures of the current step (``int32``).
        ewt: The error-weight vector, refreshed on accepted steps only.
        theta: The learnable-parameter pytree (a runtime leaf).
        control: The control-value pytree the residual reads (a
            runtime leaf; ``None`` for a problem without a control channel).
        nflag: The reason code into the next attempt (``int32``).
        saved_jac: The saved raw ``(J_f, B, C, D)`` Jacobian blocks.
        pencil: The iteration matrix ``(held, B, C, D)``: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block (its factor, whose values are ``A_aug``,
            starting empty) and the global-coupling blocks.
        ic_alg_residual_norm: The consistent-initialization residual.
        hmax_inv: The reciprocal maximum step size (``0`` for no bound).
        hmin: The minimum step size (``0`` for no bound).
        t1_end: The integration endpoint.
        problem: The solver-contract problem (identity-hashable static).
        trace_enabled: Whether to build the per-attempt trace record.
        max_order: The order cap for this solve (static; at most 5).
        flash_mem: The flash controller's memory, or ``None`` under the
            source controller.
        prec_state: The adaptive precision policy's carried
            [RungState][diff_soup.timestepping.vsvo.common.adaptive.RungState],
            or ``None`` for concrete solve modes.
        generations: The
            [Generations][diff_soup.nonlinear.generations.Generations] record
            naming the carried linearization artifacts (``saved_jac`` /
            ``pencil`` and the factor built from them); counters advance
            exactly with the corrector's rebuild events.
        factor_certificate: The
            [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate]
            issued at the last root-acceptance decision. It authorizes only
            when the held factor was provably built from values evaluated
            exactly at the accepted root, so downstream consumers can trust
            it unconditionally; the engine's own reuse windows are governed
            by its convergence monitoring, not by the certificate. The
            first-call carry holds an off-root certificate that authorizes
            nothing.
        border_cache: The generation-keyed border/Schur cache
            ([BorderSchurCache][diff_soup.nonlinear.border_cache.BorderSchurCache]) for
            a problem with bordered-global rows, or ``None`` without them.
            Plain array leaves, so each ``vmap`` lane keys and invalidates
            its own cache; a rebuilt carry starts cold.
    """

    state: NDFState
    tn: jax.Array
    ncf: jax.Array
    ewt: jax.Array
    theta: Any
    control: Any
    nflag: jax.Array
    saved_jac: tuple
    pencil: tuple
    ic_alg_residual_norm: jax.Array
    hmax_inv: jax.Array
    hmin: jax.Array
    t1_end: jax.Array
    problem: Any = eqx.field(static=True)
    trace_enabled: bool = eqx.field(static=True, default=False)
    max_order: int = eqx.field(static=True, default=MAX_ORDER)
    flash_mem: Any = None
    prec_state: Any = None
    generations: Any = None
    factor_certificate: Any = None
    border_cache: Any = None
    # Per-row absolute-tolerance vector consumed by the error-weight law when
    # the solve's tolerances were resolved against the problem layout, or
    # ``None`` for the engine's uniform scalar. Runtime data, so it rides the
    # carry (the engine itself stays static configuration).
    atol_rows: Any = None


class NDFHistory(eqx.Module):
    """What a later solve continues from: the NDF carry without the solve's configuration.

    Everything that persists from one step to the next inside a solve: the
    difference-table, order and step record, the saved Jacobian blocks and the
    iteration matrix in effect, the reason code of the next attempt, the
    generation record and border cache that key them, the controller
    memories, and the control the steps were taken under.

    Attributes:
        state: The carried [NDFState][diff_soup.timestepping.vsvo.engines.ndf.state.NDFState].
        nflag: The reason code the next attempt's corrector reads.
        saved_jac: The saved raw Jacobian blocks.
        pencil: The iteration matrix in effect, ``(held, B, C, D)``: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block, whose factors name the corrector's shared
            storage, and the global-coupling blocks.
        generations: The generation record of the carried linearization.
        border_cache: The border/Schur cache, or ``None``.
        flash_mem: The flash controller's memory, or ``None``.
        prec_state: The adaptive precision state, or ``None``.
        control: The control the steps were taken under, or ``None``.
    """

    state: NDFState
    nflag: jax.Array
    saved_jac: tuple
    pencil: tuple
    generations: Any
    border_cache: Any
    flash_mem: Any
    prec_state: Any
    control: Any


class NDFEngine(eqx.Module):
    """The NDF step kernel the shared driver runs.

    Holds only array-free static configuration; the problem, the learnable
    parameters, and every array live on `NDFEngineCarry` (a per-row
    absolute-tolerance vector, when the solve's tolerances were resolved
    against the problem layout, arrives via the driver configuration and
    rides the carry). ``atol`` is the scalar absolute tolerance — the
    tightest row when a vector is in play — and ``ic_tol`` / ``ic_rtol``
    carry the scalar consistent-initialization gate (``None`` reuses
    ``atol`` / ``rtol``).
    """

    backend: Any = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    loop_form: str = eqx.field(static=True)
    jacobian_policy: str = eqx.field(static=True)
    capabilities: EngineCapabilities = eqx.field(static=True)
    controller_mode: str = eqx.field(static=True, default="source")
    controller_law: str = eqx.field(static=True, default="pi")
    trace_kind: str = eqx.field(static=True, default="ndf")
    newton_tol_factor: float = eqx.field(static=True, default=1.0)
    ic_max_newton_iters: int = eqx.field(static=True, default=10)
    precision: Precision = eqx.field(static=True, default_factory=Precision)
    jacobian_values_dtype: Any = eqx.field(static=True, default=None)
    ic_tol: Optional[float] = eqx.field(static=True, default=None)
    ic_rtol: Optional[float] = eqx.field(static=True, default=None)

    # -- helpers -----------------------------------------------------------
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
        [_fresh_carry][diff_soup.timestepping.vsvo.engines.ndf.step.NDFEngine._fresh_carry].
        ``history`` is a [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
        whose ``state`` is a [NDFHistory][diff_soup.timestepping.vsvo.engines.ndf.step.NDFHistory].
        When its ``continues`` flag is set and the new control leaves the
        algebraic closure at the carried state unchanged, the carry continues
        it (see
        [_continued_carry][diff_soup.timestepping.vsvo.engines.ndf.step.NDFEngine._continued_carry]).
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
        if not isinstance(record, NDFHistory):
            raise TypeError(
                "history holds the record of another solver "
                f"({type(record).__name__}); a JNDF solve continues only the "
                "Solution.history of an earlier JNDF solve."
            )
        dtype = record.state.d_basis.dtype
        differential = jnp.asarray(problem.id_mask, dtype=dtype)
        current = self._continued_state(record.state, y0, lam0, differential)
        # A switch that changes the algebraic closure (a jump of the target, a
        # new mode) puts a kink in the solution: the solve then restarts at
        # order one from the step the history carries.
        continues = jnp.asarray(history.continues, dtype=jnp.bool_)
        moved = continues & closure_moved(
            problem,
            t0,
            current,
            differential,
            theta,
            control,
            record.control,
            ic_gate_tolerance(self.atol, self.ic_tol),
        )
        carried_step = record.state.dt
        first_step = jnp.where(moved, carried_step, jnp.asarray(cfg.dt0, carried_step.dtype))
        return jax.lax.cond(
            continues & jnp.logical_not(moved),
            lambda _: self._continued_carry(problem, record, t0, t1, y0, lam0, theta, cfg, control),
            lambda _: self._fresh_carry(
                problem,
                t0,
                t1,
                y0,
                ydot0,
                lam0,
                theta,
                dataclasses.replace(cfg, dt0=first_step),
                control=control,
                step=step,
            ),
            None,
        )

    def _continued_state(self, st, y0, lam0, differential):
        """The state a continued solve starts from.

        The differential rows come from ``(y0, lam0)``, the algebraic rows from
        the record's current state.
        """
        dtype = st.d_basis.dtype
        given = pack_state(jnp.asarray(y0, dtype=dtype), jnp.asarray(lam0, dtype=dtype))
        return differential * given + (1.0 - differential) * st.d_basis[0]

    def _continued_carry(self, problem, record, t0, t1, y0, lam0, theta, cfg, control):
        """The carry of a solve that continues ``record`` from ``t0``.

        The record passes as it passes between two steps of one solve: the
        difference table, the order, the step the controller chose, the
        error-estimate and corrector memory, the saved Jacobian blocks and the
        iteration matrix (its held factor, whose first refactor makes the
        shared storage hold it again) with their age, the generation record and border
        cache, and the controller memories. The differential rows of the
        current state come from ``(y0, lam0)`` (the earlier solve's end
        state), the algebraic rows from the record. The solve's own
        configuration is this solve's: the time ``t0``, the endpoint ``t1``,
        the parameters, the control, the step bounds for the steps after the
        carried one, and the tolerance rows; the endpoint clamps the carried
        step at the attempt, as it clamps every step. The error weights are
        recomputed from the state, the per-step failure counters start at
        zero, the exact-root certificate is off-root, and no initialization
        runs (its residual norm reads 0).
        """
        st = record.state
        dtype = st.d_basis.dtype
        current = self._continued_state(st, y0, lam0, jnp.asarray(problem.id_mask, dtype=dtype))
        state = eqx.tree_at(lambda s: s.d_basis, st, st.d_basis.at[0].set(current))
        atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
        hmax = float(cfg.hmax)
        return NDFEngineCarry(
            state=state,
            tn=jnp.asarray(t0, dtype=dtype),
            ncf=jnp.asarray(0, jnp.int32),
            ewt=error_weight(current, self.rtol, self.atol if atol_rows is None else atol_rows),
            theta=theta,
            control=control,
            nflag=record.nflag,
            saved_jac=record.saved_jac,
            pencil=record.pencil,
            ic_alg_residual_norm=jnp.zeros((), dtype=dtype),
            hmax_inv=jnp.asarray(1.0 / hmax if hmax > 0.0 else 0.0, dtype=dtype),
            hmin=jnp.asarray(float(cfg.hmin), dtype=dtype),
            t1_end=jnp.asarray(t1, dtype=dtype),
            problem=problem,
            trace_enabled=bool(cfg.trace),
            max_order=min(int(cfg.max_order), MAX_ORDER),
            flash_mem=record.flash_mem,
            prec_state=record.prec_state,
            generations=record.generations,
            factor_certificate=certify_exact_root(record.generations, jnp.asarray(False)),
            border_cache=record.border_cache,
            atol_rows=atol_rows,
        )

    def _fresh_carry(
        self, problem, t0, t1, y0, ydot0, lam0, theta, cfg, *, control=None, step=None
    ):
        """Build the first-call carry: consistent-IC gate + initial-step search.

        ``step`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the entry step is the recorded one, the step
        the forward sweep's carry entered with, instead of a new search.
        """
        n_u = int(problem.n_u)
        n_lam = int(problem.n_lam)
        u0 = jnp.asarray(y0)
        dtype = u0.dtype
        lam0 = jnp.asarray(lam0, dtype=dtype)
        ydot0 = jnp.asarray(ydot0, dtype=dtype)
        t0 = jnp.asarray(t0, dtype=dtype)
        t1 = jnp.asarray(t1, dtype=dtype)
        uround = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

        _, _, _diff, _alg_index, needs_projection = _structure(problem)
        if not bool(cfg.consistent_ic) and needs_projection:
            raise ConfigurationError(
                "this problem has algebraic rows; "
                "consistent_ic=False is not supported for it. Enable "
                "consistent initialization."
            )
        # The projection factors this problem exactly as the corrector does,
        # so it is analyzed with the lowered problem's placement and the
        # accepted-root sensitivity precision this solve declared for each
        # derivative direction — never the backend's defaults.
        ic_layout, ic_precision = initialization_policy(problem, self.precision, mode="forward")
        _, ic_reverse_precision = initialization_policy(problem, self.precision, mode="reverse")
        ic = problem.consistent_initial_state(
            t0,
            pack_state(u0, lam0),
            theta,
            control,
            ydot0=jnp.concatenate([ydot0, jnp.zeros((n_lam,), dtype=dtype)]),
            tol=ic_gate_tolerance(self.atol, self.ic_tol),
            max_iters=self.ic_max_newton_iters,
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

        dt0 = jax.lax.stop_gradient(jnp.asarray(cfg.dt0, dtype=dtype))

        def _auto_initial_step(_):
            # Step-size control is gradient-dead by convention. Trial-state
            # slopes use the same startup operation as the initial
            # history and restart paths.
            step_layout, step_precision = initialization_policy(
                problem, self.precision, mode="forward"
            )
            slope_fn = bdf_slope_fn(
                problem,
                theta,
                tol=ic_gate_tolerance(self.atol, self.ic_tol),
                control=control,
                backend=self.backend,
                layout=step_layout,
                precision=step_precision,
                reverse_precision=ic_reverse_precision,
            )
            return bdf_initial_step(
                problem,
                theta,
                t0,
                t1,
                y_aug0,
                ydot_aug0,
                ewt0,
                uround,
                hmax_inv,
                dtype.type,
                slope_fn=slope_fn,
            )

        hh = jax.lax.stop_gradient(
            jax.lax.cond(dt0 != 0.0, lambda _: dt0, _auto_initial_step, operand=None)
        )

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

        state = init_ndf_state(y_aug0, ydot_aug0, hh, dtype.type)

        # The state block is held as its factor, which starts empty (no
        # factorization) and is filled by the first setup's refactor.
        linear = problem_linear(self.backend, problem, precision=self.precision, dtype=dtype)
        zeros_pencil = (
            empty_held(linear, state_block_size(problem), dtype),
            jnp.zeros((n_u, n_lam), dtype=dtype),
            jnp.zeros((n_lam, n_u), dtype=dtype),
            jnp.zeros((n_lam, n_lam), dtype=dtype),
        )
        zeros_jac = (problem.zero_state_partials(dtype),) + zeros_pencil[1:]

        if self.controller_mode == "flash":
            flash_mem = (jnp.asarray(FLASH_ERR_SEED, dtype=dtype), hh, jnp.asarray(True))
        else:
            flash_mem = None

        # A fresh generation record for the zero-seeded carry, paired with an
        # off-root certificate: the carry structure is fixed from the first
        # call, and nothing authorizes reusing the zero seed. The border
        # cache starts cold on every carry construction — including an event
        # restart, which rebuilds the carry through this entry.
        generations0 = initial_generations(coupling_blocks=n_lam > 0)
        certificate0 = certify_exact_root(generations0, jnp.asarray(False))
        border_cache0 = initial_border_cache(n_u, n_lam, dtype) if n_lam > 0 else None

        return NDFEngineCarry(
            state=state,
            tn=t0,
            ncf=jnp.asarray(0, jnp.int32),
            ewt=ewt0,
            theta=theta,
            control=control,
            nflag=jnp.asarray(_FIRST_CALL, jnp.int32),
            saved_jac=zeros_jac,
            pencil=zeros_pencil,
            ic_alg_residual_norm=alg_norm,
            hmax_inv=hmax_inv,
            hmin=hmin,
            t1_end=t1,
            problem=problem,
            trace_enabled=bool(cfg.trace),
            max_order=min(int(cfg.max_order), MAX_ORDER),
            flash_mem=flash_mem,
            prec_state=(initial_rung_state() if self.precision.solve_mode == "adaptive" else None),
            generations=generations0,
            factor_certificate=certificate0,
            border_cache=border_cache0,
            atol_rows=atol_rows,
        )

    # -- one retry-loop pass ----------------------------------------------
    def decision_template(self):
        """Zero-valued decision inputs of one attempt, the shape a replay records.

        The convergence flag and the three error estimates (current order,
        one below, one above), plus the precision-attribution flags under the
        adaptive mode. Everything the attempt decides is a function of these.
        """
        flag = jnp.zeros((), jnp.bool_)
        norm = jnp.zeros((), jnp.float32)
        base = (flag, norm, norm, norm)
        if self.precision.solve_mode == "adaptive":
            return base + (flag, flag, flag)
        return base

    def attempt_step(self, carry: NDFEngineCarry, decision=None):
        """Advance one pass of the step-attempt retry loop; report accept or fail.

        ``decision`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the attempt uses the recorded decision inputs
        (see [decision_template][]) so it decides as the forward sweep did.
        """
        problem = carry.problem
        st = carry.state
        theta = carry.theta
        ewt = carry.ewt
        dtype = st.dt.dtype
        dtype_t = dtype.type
        n_u = int(problem.n_u)
        uround = jnp.finfo(dtype).eps
        t1 = carry.t1_end
        tn = carry.tn
        nflag_in = carry.nflag
        max_order = jnp.asarray(carry.max_order, jnp.int32)

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

        dt = st.dt
        k = st.k
        tdir = jnp.sign(st.dt_basis)
        dt_abs = jnp.abs(dt)
        # The step floor is a magnitude scale (controller state, gradient-dead
        # by the suite's discipline); ``jnp.spacing`` lowers to ``nextafter``,
        # which has no differentiation rule, so its input must carry no tangent.
        floor = jnp.maximum(
            carry.hmin, _FLOOR_SPACINGS * jnp.spacing(jax.lax.stop_gradient(jnp.abs(tn)))
        )

        # working table: the accepted basis moved onto this attempt's step
        rho = dt / st.dt_basis
        rescale_fired = rho != 1.0
        d_work = rescale_d(st.d_basis, rho, k)

        # endpoint snap on the predicted time (reporting/accept only)
        tn_raw = tn + dt
        tn_new = jnp.where(
            jnp.abs(tn_raw - t1) <= _TN_SNAP_FUZZ * uround * (jnp.abs(tn_raw) + dt_abs),
            t1,
            tn_raw,
        )

        # predictor and history term from rows 0..k / 1..k
        mask_le_k = (_ROW_IDX <= k).astype(dtype)
        u0 = mask_le_k @ d_work
        gamma_w = _GAMMA_ROWS.astype(dtype) * mask_le_k  # row 0 weight is 0
        phi = gamma_w @ d_work

        beta0 = jnp.asarray(BETA0, dtype=dtype)[k]
        gdt = dt * beta0
        invgdt = 1.0 / gdt
        err_const_k = jnp.asarray(ERROR_CONST, dtype=dtype)[k]

        ctx = NDFNlsContext(
            t=tn_new,
            theta=theta,
            invgdt=invgdt,
            gdt=gdt,
            phi_dt=phi / dt,
            pred_y=u0,
            ewt=ewt,
            err_const_k=err_const_k,
            theta_mem=st.theta_mem,
            first_call=nflag_in == _FIRST_CALL,
            # A precision-attributed retry re-enters like a convergence
            # failure (fresh setup, fresh Jacobian) — only the step size and
            # the failure counters are handled differently below.
            prev_conv_fail=(nflag_in == _PREV_CONV_FAIL) | (nflag_in == _PREV_PREC_FAIL),
            prev_err_fail=nflag_in == _PREV_ERR_FAIL,
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
        nls = ndf_corrector_root(
            problem,
            backend,
            linear,
            sens_analysis,
            ctx,
            self._directive(),
            self.loop_form,
            frozen_saved_jac=carry.saved_jac,
            frozen_pencil=carry.pencil,
            reverse_analysis=rev_analysis,
            generations=carry.generations,
            border_cache=carry.border_cache,
            nst=st.nst,
            nstlp=st.nstlp,
            nstlj=st.nstlj,
            gdt_prev=st.gdt_prev,
            newton_tol_factor=self.newton_tol_factor,
            jacobian_values_dtype=self.jacobian_values_dtype,
            detect_precision=adaptive,
        )
        did_converge = jnp.asarray(nls.converged, jnp.bool_)
        if adaptive:
            # Attribute failures to precision on a reduced rung: false
            # convergence rejects the converged-by-increment sweep, and both
            # detector causes retry promoted without shrinking the step.
            # A refinement escalation on a healthy sweep promotes forward.
            reduced = prec.rung < TOP_RUNG
            prec_false_conv = jnp.asarray(nls.prec_false_conv) & reduced
            prec_stagnant = jnp.asarray(nls.prec_stagnant) & reduced
            lin_escalated = (
                (jnp.asarray(nls.lin_fallbacks) > 0) & reduced
                if nls.lin_fallbacks is not None
                else jnp.asarray(False)
            )
            did_converge = did_converge & jnp.logical_not(prec_false_conv)

        # candidate table and the three error estimates (post-fold rows)
        d_cand = update_d(d_work, nls.dd, k)
        eest = jax.lax.stop_gradient(err_const_k * wrms_norm(nls.dd, ewt))
        err_const = jnp.asarray(ERROR_CONST, dtype=dtype)
        eest1 = jax.lax.stop_gradient(err_const[jnp.maximum(k - 1, 0)] * wrms_norm(d_cand[k], ewt))
        eest2 = jax.lax.stop_gradient(
            err_const[jnp.minimum(k + 1, MAX_ORDER)] * wrms_norm(d_cand[k + 2], ewt)
        )
        decided = (did_converge, eest, eest1, eest2)
        if adaptive:
            decided = decided + (prec_false_conv, prec_stagnant, lin_escalated)
        decided = replay_decisions(decision, decided)
        did_converge, eest, eest1, eest2 = decided[:4]
        if adaptive:
            prec_false_conv, prec_stagnant, lin_escalated = decided[4:]
            prec_retry = prec_false_conv | (prec_stagnant & jnp.logical_not(did_converge))
        else:
            prec_retry = None
        error_pass = eest <= 1.0
        accept = jnp.logical_and(did_converge, error_pass)

        if self.controller_mode == "flash":
            flash_err_prev, flash_h_prev, flash_last_accept = carry.flash_mem
            flash_res = flash_step_control(
                eest,
                flash_err_prev,
                h=dt,
                h_prev=flash_h_prev,
                order=k,
                last_accept=flash_last_accept,
                first_step=st.nst == 0,
                hmax_inv=carry.hmax_inv,
                hmin=carry.hmin,
                max_order=MAX_ORDER,
                law=self.controller_law,
            )

        # --- accept branch: fold the basis, ask the accept controller ---
        acc = accept_step_control(
            dt=dt_abs,
            k=k,
            eest=eest,
            eest1=eest1,
            eest2=eest2,
            nconsteps=st.nconsteps,
            max_order=max_order,
        )
        dt_acc_abs = acc.dt_next
        k_acc = acc.k_next
        if self.controller_mode == "flash":
            # keep the order decision; the flash controller sizes the step
            dt_acc_abs = jnp.abs(flash_res.h_new)
        # hmax / hmin clamps, then the endpoint clamp
        over = dt_acc_abs * carry.hmax_inv
        dt_acc_abs = jnp.where(
            over > 1.0, dt_acc_abs / jnp.where(over > 1.0, over, 1.0), dt_acc_abs
        )
        dt_acc_abs = jnp.maximum(dt_acc_abs, floor)
        dt_acc = tdir * dt_acc_abs
        at_end = jnp.abs(tn_new - t1) <= _TN_SNAP_FUZZ * uround * (jnp.abs(tn_new) + dt_abs)
        overtake = (tn_new + dt_acc - t1) * dt > 0
        clamp = jnp.logical_and(jnp.logical_not(at_end), overtake)
        dt_acc = jnp.where(clamp, (t1 - tn_new) * (1.0 - _ENDPOINT_CLAMP_FUZZ * uround), dt_acc)
        # the constant-step counter survives only an unchanged step size
        nconsteps_acc = jnp.where(dt_acc == dt, acc.nconsteps_next, jnp.zeros((), jnp.int32))
        accept_state = NDFState(
            d_basis=d_cand,
            dt_basis=dt,
            k=k_acc,
            k_used=k,
            dt=dt_acc,
            nconsteps=nconsteps_acc,
            consfailcnt=jnp.zeros((), jnp.int32),
            theta_mem=nls.theta_out,
            nst=st.nst + 1,
            nstlp=nls.nstlp_out,
            nstlj=nls.nstlj_out,
            gdt_prev=nls.gdt_prev_out,
        )
        y_new_full = d_cand[0]
        ydot_new_full = nls.dd * invgdt + phi / dt

        # --- error-test-rejection branch ---
        rej = reject_step_control(
            dt=dt_abs, k=k, eest=eest, eest1=eest1, consfailcnt=st.consfailcnt
        )
        dt_rej_abs = rej.dt_next
        if self.controller_mode == "flash":
            # the first rejection of a step retries at the flash ratio
            first_rej = st.consfailcnt == 0
            dt_rej_abs = jnp.where(first_rej, dt_abs * flash_res.eta_reject, dt_rej_abs)
        abort_ef = dt_rej_abs < floor
        dt_rej = tdir * jnp.maximum(dt_rej_abs, floor)
        restart = jnp.logical_and(rej.restart_history, jnp.logical_not(abort_ef))
        y_basis = st.d_basis[0]
        ydot_guess = st.d_basis[1] / jnp.where(st.dt_basis != 0, st.dt_basis, 1.0)

        slope_layout, slope_precision = initialization_policy(
            problem, self.precision, mode="forward"
        )
        _, slope_reverse_precision = initialization_policy(problem, self.precision, mode="reverse")

        def _restart_slope(_op):
            return problem.startup_slope(
                tn,
                y_basis,
                theta,
                carry.control,
                ydot0=ydot_guess,
                tol=ic_gate_tolerance(self.atol, self.ic_tol),
                backend=self.backend,
                layout=slope_layout,
                precision=slope_precision,
                reverse_precision=slope_reverse_precision,
            )

        # Under vmap the slope runs only when some lane restarts.
        ydot_restart = lane_cond(
            restart,
            _restart_slope,
            lambda _op: jnp.zeros_like(y_basis),
            None,
            false_is_free=True,
        )
        d_restart = seed_d(y_basis, ydot_restart, dt_rej)
        reject_state = NDFState(
            d_basis=jnp.where(restart, d_restart, st.d_basis),
            dt_basis=jnp.where(restart, dt_rej, st.dt_basis),
            k=rej.k_next,
            k_used=st.k_used,
            dt=dt_rej,
            nconsteps=jnp.zeros((), jnp.int32),
            consfailcnt=jnp.where(restart, jnp.zeros((), jnp.int32), rej.consfailcnt_next),
            theta_mem=jnp.ones((), dtype),
            nst=st.nst,
            nstlp=nls.nstlp_out,
            nstlj=nls.nstlj_out,
            gdt_prev=nls.gdt_prev_out,
        )

        # --- Newton-failure branch: halve and retry on the untouched basis ---
        # A precision-attributed failure retries at the promoted rung with the
        # SAME step (an fp64 re-solve of a fresh Jacobian is far cheaper than
        # a step rejection); the rung is monotone within a step, so these
        # retries are structurally bounded and never count as Newton failures.
        dt_nf_abs = jnp.maximum(_NEWTON_FAIL_SHRINK * dt_abs, floor)
        ncf_next = carry.ncf + 1
        abort_nf = jnp.logical_or(
            _NEWTON_FAIL_SHRINK * dt_abs < floor, ncf_next >= _MAX_NEWTON_FAILS
        )
        if adaptive:
            dt_nf_abs = jnp.where(prec_retry, dt_abs, dt_nf_abs)
            ncf_next = jnp.where(prec_retry, carry.ncf, ncf_next).astype(jnp.int32)
            abort_nf = abort_nf & jnp.logical_not(prec_retry)
        dt_nf = tdir * dt_nf_abs
        newton_fail_state = NDFState(
            d_basis=st.d_basis,
            dt_basis=st.dt_basis,
            k=k,
            k_used=st.k_used,
            dt=dt_nf,
            nconsteps=jnp.zeros((), jnp.int32),
            consfailcnt=st.consfailcnt,
            theta_mem=jnp.ones((), dtype),
            nst=st.nst,
            nstlp=nls.nstlp_out,
            nstlj=nls.nstlj_out,
            gdt_prev=nls.gdt_prev_out,
        )

        # --- select the branch ---
        fail_state = _where_state(did_converge, reject_state, newton_fail_state)
        new_state = _where_state(accept, accept_state, fail_state)

        new_ewt = error_weight(
            y_new_full, self.rtol, self.atol if carry.atol_rows is None else carry.atol_rows
        )
        nonconv_flag = jnp.asarray(_PREV_CONV_FAIL, jnp.int32)
        if adaptive:
            nonconv_flag = jnp.where(
                prec_retry, jnp.asarray(_PREV_PREC_FAIL, jnp.int32), nonconv_flag
            )
        nflag_next = jnp.where(
            accept,
            jnp.asarray(_FIRST_CALL, jnp.int32),
            jnp.where(
                did_converge,
                jnp.asarray(_PREV_ERR_FAIL, jnp.int32),
                nonconv_flag,
            ),
        ).astype(jnp.int32)
        ncf_out = jnp.where(
            accept,
            jnp.zeros((), jnp.int32),
            jnp.where(did_converge, jnp.zeros((), jnp.int32), ncf_next),
        ).astype(jnp.int32)

        if self.controller_mode == "flash":
            flash_mem_next = (
                jax.lax.stop_gradient(jnp.where(did_converge, eest, flash_err_prev)),
                jax.lax.stop_gradient(jnp.where(accept, dt, flash_h_prev)),
                accept,
            )
        else:
            flash_mem_next = None

        if adaptive:
            # Attribute this attempt's promotion cause (false convergence
            # takes precedence over stagnation over a forward-looking
            # refinement escalation) and advance the rung policy.
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

        new_carry = NDFEngineCarry(
            state=new_state,
            tn=jnp.where(accept, tn_new, tn),
            ncf=ncf_out,
            ewt=jnp.where(accept, new_ewt, ewt),
            theta=theta,
            control=carry.control,
            nflag=nflag_next,
            saved_jac=nls.saved_jac_out,
            pencil=nls.pencil_out,
            ic_alg_residual_norm=carry.ic_alg_residual_norm,
            hmax_inv=carry.hmax_inv,
            hmin=carry.hmin,
            t1_end=carry.t1_end,
            problem=problem,
            trace_enabled=carry.trace_enabled,
            max_order=carry.max_order,
            flash_mem=flash_mem_next,
            prec_state=prec_next,
            generations=nls.generations_out,
            factor_certificate=nls.certificate_out,
            border_cache=nls.border_cache_out,
            atol_rows=carry.atol_rows,
        )

        result_code = jnp.where(
            accept,
            jnp.asarray(_CODE_SUCCESS, jnp.int32),
            jnp.where(
                did_converge,
                jnp.where(
                    abort_ef,
                    jnp.asarray(_CODE_STEP_COLLAPSE, jnp.int32),
                    jnp.asarray(_CODE_SUCCESS, jnp.int32),
                ),
                jnp.where(
                    abort_nf,
                    jnp.asarray(_CODE_NONLINEAR, jnp.int32),
                    jnp.asarray(_CODE_SUCCESS, jnp.int32),
                ),
            ),
        ).astype(jnp.int32)

        y_prev = st.d_basis[0]
        ydot_prev_scale = jnp.where(st.dt_basis != 0, st.dt_basis, 1.0)
        ydot_prev = st.d_basis[1] / ydot_prev_scale
        dt_fail = jnp.where(did_converge, dt_rej, dt_nf)
        k_fail = jnp.where(did_converge, rej.k_next, k).astype(jnp.int32)

        restart_extra = jnp.where(
            jnp.logical_and(jnp.logical_not(accept), restart),
            jnp.asarray(1, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

        trace_record = None
        if carry.trace_enabled:
            trace_record = NDFTraceRecord(
                header=StepTraceHeader.zeros(dtype_t),
                k=k.astype(jnp.int32),
                dt_attempt=dt,
                rho=rho,
                rescale_fired=rescale_fired,
                nconsteps=st.nconsteps.astype(jnp.int32),
                consfailcnt=st.consfailcnt.astype(jnp.int32),
                invgdt=invgdt,
                theta=nls.theta_out,
                ndz=nls.ndz_last,
                newton_converged=did_converge,
                call_setup=jnp.asarray(nls.call_setup, jnp.bool_),
                jbad=jnp.asarray(nls.jbad, jnp.bool_),
                jcur=jnp.asarray(nls.jcur, jnp.bool_),
                nsetups=nls.nsetups.astype(jnp.int32),
                nje=nls.nje.astype(jnp.int32),
                eest=eest,
                eest1=eest1,
                eest2=eest2,
                k_next=jnp.where(accept, k_acc, k_fail).astype(jnp.int32),
                dt_next=jnp.where(accept, dt_acc, dt_fail),
                held=jnp.where(accept, acc.held, jnp.asarray(False)),
                restart_history=jnp.where(accept, jnp.asarray(False), restart),
            )

        # Border/Schur accounting: the corrector reports the coupling-block
        # work it actually performed — the generation-keyed cache rebuilds a
        # tier only when its keys moved, so unchanged generations add nothing.
        report = StepReport(
            accepted=accept,
            t_new=jnp.where(accept, tn_new, tn),
            y_new=jnp.where(accept, y_new_full[:n_u], y_prev[:n_u]),
            ydot_new=jnp.where(accept, ydot_new_full[:n_u], ydot_prev[:n_u]),
            lam_new=jnp.where(accept, y_new_full[n_u:], y_prev[n_u:]),
            h_new=jnp.where(accept, dt_acc, dt_fail),
            k_new=jnp.where(accept, k_acc, k_fail).astype(jnp.int32),
            err_estimate=eest,
            result=result_code,
            newton_iters=nls.nni,
            jac_refreshes=nls.nje,
            factorizations=nls.nsetups,
            linear_solves=nls.nni,
            rhs_evals=nls.nni + restart_extra,
            residual_evals=nls.nni + restart_extra,
            trace_record=trace_record,
            precision_promotions=promo_delta,
            ir_refinement_steps=nls.lin_ir_steps,
            fp64_fallbacks=nls.lin_fallbacks,
            multirhs_solves=nls.du_b_builds,
            multirhs_columns=nls.du_b_builds * jnp.asarray(int(problem.n_lam), jnp.int32),
            du_b_builds=nls.du_b_builds,
            schur_builds=nls.schur_builds,
            decisions=decided,
        )
        return new_carry, report

    def state_at_entry(self, carry: NDFEngineCarry):
        """The augmented ``[u; lam]`` state the solve starts from."""
        return carry.state.d_basis[0]

    def step_at_entry(self, carry: NDFEngineCarry):
        """The step size the carry's next attempt starts from.

        Read on a freshly built carry, this is the entry step ``init_carry``
        chose; the driver records it after an event restart so a replayed
        restart enters with the same step.
        """
        return carry.state.dt

    def history_at_exit(self, carry: NDFEngineCarry) -> NDFHistory:
        """The [NDFHistory][diff_soup.timestepping.vsvo.engines.ndf.step.NDFHistory] a later solve continues from.

        Read on the final carry: everything that persists from one step to the
        next, without the solve's configuration.
        """
        return NDFHistory(
            state=carry.state,
            nflag=carry.nflag,
            saved_jac=carry.saved_jac,
            pencil=carry.pencil,
            generations=carry.generations,
            border_cache=carry.border_cache,
            flash_mem=carry.flash_mem,
            prec_state=carry.prec_state,
            control=carry.control,
        )

    def slope_at_entry(self, carry: NDFEngineCarry):
        """The augmented state derivative the solve starts from.

        Read on the freshly built carry: the seeded first difference row is
        ``h * ydot`` over the initial step, so the entry slope divides it
        back out (the consistent-initialization slope on the
        startup route).
        """
        st = carry.state
        return st.d_basis[1] / jnp.where(st.dt_basis != 0, st.dt_basis, 1.0)

    # -- dense output ------------------------------------------------------
    def dense(self, carry: NDFEngineCarry, t):
        """Evaluate the differential state and derivative at ``t`` on the last step."""
        n_u = int(carry.problem.n_u)
        y_full, ydot_full = self.dense_full(carry, t)
        return y_full[:n_u], ydot_full[:n_u]

    def dense_full(self, carry: NDFEngineCarry, t):
        """Evaluate the full augmented interpolant (state and bordered rows) at ``t``.

        Every row is reported at interpolation accuracy. Strict algebraic
        constraint satisfaction is the projected output policy's job.
        Interpolated algebraic-row derivatives are internal only and must be
        zeroed if a public derivative surface is added.
        """
        st = carry.state
        return ndf_interpolant(st.d_basis, st.k_used, carry.tn, st.dt_basis, t)


@functools.lru_cache(maxsize=None)
def ndf_engine(
    profile,
    *,
    rtol: float,
    atol: Any,
    linear: Any,
    newton_tol_factor: float = 1.0,
    ic_max_newton_iters: int = 10,
    precision: Precision = Precision(),
    jacobian_values_dtype=None,
    ic_tol: Optional[float] = None,
    ic_rtol: Optional[float] = None,
) -> NDFEngine:
    """Build the NDF engine kernel from a solver execution profile.

    Resolves the linear backend from ``linear`` and reads the corrector loop
    shape, the Jacobian setup policy, and the accepted-step controller mode
    from ``profile``. ``newton_tol_factor`` widens the corrector's acceptance
    band relative to the step tolerance (``1.0`` = the standard band).
    ``ic_max_newton_iters`` caps the consistent-initialization projection's
    Newton iteration (the projection exits early once converged, so the cap
    binds only for seeds far from the algebraic manifold).
    ``precision`` selects the linear solve mode; a mode the backend does not
    declare is rejected here, eagerly. Identical configurations return one
    memoised engine object, so repeated solves share the driver's compiled
    program.
    """
    backend = resolve_backend(linear)
    backend._validate_capabilities(Layout(), precision)
    return NDFEngine(
        backend=backend,
        rtol=float(rtol),
        atol=float(atol),
        loop_form=profile.nonlinear_mode,
        jacobian_policy=profile.jacobian_policy,
        capabilities=_capabilities(),
        controller_mode=profile.controller_mode,
        controller_law=profile.controller_law,
        newton_tol_factor=float(newton_tol_factor),
        ic_max_newton_iters=int(ic_max_newton_iters),
        precision=precision,
        jacobian_values_dtype=jacobian_values_dtype,
        ic_tol=None if ic_tol is None else float(ic_tol),
        ic_rtol=None if ic_rtol is None else float(ic_rtol),
    )


__all__ = ["NDFEngine", "NDFEngineCarry", "NDFHistory", "ndf_engine"]
