"""Step assembly and driver wiring for the backward-differentiation-formula engine.

This module assembles the landed step pieces -- the coefficient recurrence, the
predictor, the correction-form corrector, the local-error test, the accepted-step
controller, and the failure ladders -- into an [Engine][] the shared driver
runs. [BDFEngine][] owns the array-free static configuration; the carried
per-attempt state (the Nordsieck history, the retry counters, the error weights,
the two-tier saved-Jacobian / iteration-matrix blocks, the learnable parameters,
and the problem it integrates) rides on [BDFEngineCarry][].

The decomposition follows the driver contract: one [BDFEngine.attempt_step][]
is exactly one pass of the step-attempt retry loop. A converged, error-test-passing attempt reports
``accepted=True`` and hands the driver the next order and step; a corrector or
error-test failure restores the history, walks the failure ladder, and reports
``accepted=False`` -- either a retry sentinel (the driver re-invokes with the
shrunk step already in the carry) or a termination code. The driver's outer loop
counts accepted and rejected attempts.

Two ordering rules distinguish the Nordsieck history from a divided-difference
one. First, the step size is baked into the history (``zn[j] = (h**j / j!) *
y^(j)``), so ``h`` may change only through a rescale of that history or a
failure ladder -- never by editing ``h`` directly at the top of an attempt.
The endpoint clamp therefore shrinks the *next* step's ``hprime``/``eta`` at the
end of a successful attempt, and the following attempt's rescale moves the
history onto it. Second, the order-1 error-test restart does not
rescale: it updates ``h`` directly and rebuilds the first history column from a
fresh right-hand-side evaluation.

Before the first step [BDFEngine.init_carry][] runs the mandatory
consistent-initialization gate when the problem has algebraic structure, seeds
the Nordsieck history from the (projected) state and mass-consistent derivative,
and sizes the initial step from an automatic search over the mass-consistent
second derivative.
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
from diff_soup.timestepping.vsvo.coefficients import cj_from_hinvgak
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
from diff_soup.timestepping.vsvo.common.trace import BDFTraceRecord, StepTraceHeader
from diff_soup.timestepping.vsvo.common.weights import error_weight
from diff_soup.utils.lanes import lane_cond

from .coeffs import bdf_set_coeffs
from .complete_step import bdf_complete_step
from .controller import ChosenBranch, bdf_prepare_next_step
from .dense import bdf_dense, bdf_initial_step, bdf_slope_fn
from .error import bdf_do_error_test_accept
from .failure import (
    SUN_NLS_CONV_RECVR,
    bdf_error_test_fail,
    bdf_handle_nflag,
    kflag_to_result_code,
)
from .flags import NonlinearFlag, StepFlag
from .nonlinear import BDFNlsContext, bdf_corrector_root
from .nordsieck import (
    bdf_adjust_params,
    bdf_decrease_order,
    bdf_predict_history,
    bdf_rescale_history,
    bdf_restore_history,
)
from .state import QMAX, BDFState, init_bdf_state

# Driver step-result codes (see diff_soup.timestepping.vsvo.common.driver): 1 is
# the success sentinel the driver reads as "not terminal" -- a retry reports it so
# the driver re-invokes the engine with the shrunk step already in the carry.
_CODE_SUCCESS = 1

# The internal time snaps onto the endpoint when it lands within this many
# rounding errors of it.
_TN_SNAP_FUZZ = 100.0

# The endpoint clamp shortens a step to just shy of the endpoint by this many
# rounding errors.
_ENDPOINT_CLAMP_FUZZ = 4.0

# The step-growth cap reset after every successful step -- the same value
# applies on early and general steps.
_ETA_MAX_POST_SUCCESS = 10.0


def _capabilities() -> EngineCapabilities:
    """The BDF-tracking engine's capability flags (every flag stated)."""
    return EngineCapabilities(
        # The factored iteration matrix persists across attempts and is
        # refactored only where a setup replaces it.
        supports_factor_reuse=True,
        supports_events=True,
        # This engine integrates the mass-matrix ODE/DAE form (a correction-form
        # residual built from the mass and right-hand side), not a user-supplied
        # truly-implicit residual F(t, y, ydot).
        supports_fully_implicit=False,
        # The augmented Schur machinery handles bordered-global rows.
        supports_global_constraints=True,
        # The mass is evaluated as identity, constant, or purely time-dependent;
        # a state-dependent mass is rejected before the solve.
        supports_state_dependent_mass=False,
        supports_dense_output=True,
    )


def _reject_unsupported_problem(problem: Any) -> None:
    """Reject problem features this engine does not integrate, loudly and eagerly.

    Runs before any array work, so an unsupported problem fails as a Python
    error at configuration time — never as a traced-in NaN or a wrong
    linearization. This engine integrates the mass form ``M(t) @ dy/dt =
    f(t, y)`` with identity, constant, or purely time-dependent ``M``.

    Raises:
        ConfigurationError: For a solver problem whose residual is fully
            implicit in the derivative, or for any problem declaring
            state-dependent derivative coefficients.
    """
    model = getattr(problem, "problem", None)
    if model is not None and not hasattr(model, "mass"):
        raise ConfigurationError(
            "this solver integrates the mass form M(t) @ dy/dt = f(t, y), but "
            "the problem lowers to a fully implicit residual "
            "F(t, y, dy/dt). Select a fully-implicit-capable solver, or author "
            "the model in mass form."
        )
    mass_dependence = getattr(problem, "mass_dependence", "identity")
    if mass_dependence in ("state_weak", "state_strong"):
        raise ConfigurationError(
            f"the problem declares state-dependent derivative coefficients "
            f"(mass_dependence={mass_dependence!r}), but this solver supports "
            f"only identity, constant, and time-dependent coefficients. Supply "
            f"a problem whose derivative coefficients do not read the state, "
            f"or select a solver that supports a state-dependent mass."
        )


def _pad_to(vec: jax.Array, length: int) -> jax.Array:
    """Right-pad (or truncate) ``vec`` to ``length`` along its single axis."""
    have = int(vec.shape[0])
    if have < length:
        return jnp.concatenate([vec, jnp.zeros((length - have,), vec.dtype)])
    return vec[:length]


def _where_state(cond: jax.Array, a: BDFState, b: BDFState) -> BDFState:
    """Select between two carried states leaf-by-leaf on a scalar predicate."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def _decision_part(decision, start, stop):
    """The ``[start, stop)`` slice of a recorded decision, or ``None``."""
    if decision is None:
        return None
    replay, recorded = decision
    return replay, tuple(recorded)[start:stop]


class BDFEngineCarry(eqx.Module):
    """The per-attempt carry threaded through the BDF-tracking engine.

    The Nordsieck/order/step record [BDFState][] persists across accepted
    steps; the remaining fields are the step loop's own local state. ``ncf`` and
    ``nef`` are the per-step convergence- and error-test-failure counts the
    failure ladders bump and every accepted step resets. ``ewt`` is the
    error-weight vector, refreshed only after an accepted step and held constant
    across every retry of a step. ``tn`` is the current internal time at the top
    of an attempt. ``theta`` is the learnable-parameter pytree the residual reads
    (a runtime leaf). ``nflag`` is the reason code the next attempt's corrector
    reads: ``FIRST_CALL`` on the first attempt of every step, ``PREV_CONV_FAIL``
    or ``PREV_ERR_FAIL`` on the respective retries. ``saved_jac`` is the saved
    raw right-hand-side Jacobian and its global blocks; ``pencil`` is the
    assembled iteration matrix. Both persist across attempts and accepted steps
    (they are re-evaluated only inside a setup) and are never rolled
    back by a rejected attempt.

    Attributes:
        state: The carried Nordsieck/order/step state.
        tn: The current internal time (top of the attempt = last accepted time).
        ncf: The local convergence-failure count (``int32``; reset on accept).
        nef: The local error-test-failure count (``int32``; reset on accept).
        ewt: The error-weight vector, shape ``(n,)``.
        theta: The learnable-parameter pytree (a runtime leaf).
        nflag: The reason code into the next attempt (``int32``).
        saved_jac: The saved ``(J_f, B, C, D)`` raw Jacobian blocks, zero-seeded
            at the first call where the setup gate always fires and never reads
            the seed.
        pencil: The iteration matrix ``(held, B, C, D)``: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block (its factor, whose values are ``A_aug``,
            starting empty) and the global-coupling blocks, zero-seeded at the
            first call.
        ic_alg_residual_norm: The local- and global-algebraic residual norm the
            consistent-initialization gate achieved (reported in the stats).
        hmax_inv: The reciprocal maximum step size (``0`` for no bound).
        hmin: The minimum step size (``0`` for no bound).
        t1_end: The integration endpoint, used as the endpoint clamp target.
        problem: The solver-contract problem (identity-hashable static config).
        trace_enabled: Whether to build the per-attempt trace record.
        flash_mem: The flash controller's memory ``(err_prev, h_prev,
            last_accept)`` — the normalized error of the last attempt whose
            corrector converged, the step size of the last accepted step,
            and whether the last attempt was accepted. ``None`` under the
            source controller, so a source-profile carry has no extra
            leaves.
        generations: The [Generations][diff_soup.nonlinear.generations.Generations]
            record naming the carried ``saved_jac`` / ``pencil`` / held-factor
            artifacts: the point, matrix, and factor counters advance
            together exactly when a setup rebuilds the iteration matrix and
            its factor, and the coupling-block counters (tracked when the
            problem has bordered-global rows) advance when a fresh
            Jacobian re-evaluates the coupling blocks.
        factor_certificate: The
            [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate]
            issued at the last attempt's root-acceptance decision. It attests
            exactness only when the factorization point provably IS the
            accepted root: the corrector converged, the matrix was rebuilt
            from a FRESH Jacobian evaluation this attempt (a saved-Jacobian
            pencil rebuild carries values from an earlier point and never
            attests), and the total applied correction is exactly zero. It
            exists for generation-keyed consumers of the carried factor,
            not for the engine's own window-gated reuse policy. The initial
            carry holds an off-root certificate, which never authorizes.
        border_cache: The generation-keyed border/Schur cache
            ([BorderSchurCache][diff_soup.nonlinear.border_cache.BorderSchurCache]) for
            a problem with bordered-global rows, or ``None`` without them.
            Plain array leaves, so each ``vmap`` lane keys and invalidates
            its own cache; a rebuilt carry starts cold.
    """

    state: BDFState
    tn: jax.Array
    ncf: jax.Array
    nef: jax.Array
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


class BDFHistory(eqx.Module):
    """What a later solve continues from: the BDF carry without the solve's configuration.

    Everything that persists from one step to the next inside a solve: the
    Nordsieck, order and step record, the saved Jacobian blocks and the
    iteration matrix in effect, the reason code of the next attempt, the
    generation record and border cache that key them, the controller
    memories, and the control the steps were taken under.

    Attributes:
        state: The carried [BDFState][diff_soup.timestepping.vsvo.engines.bdf.state.BDFState].
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

    state: BDFState
    nflag: jax.Array
    saved_jac: tuple
    pencil: tuple
    generations: Any
    border_cache: Any
    flash_mem: Any
    prec_state: Any
    control: Any


class BDFEngine(eqx.Module):
    """The BDF step kernel the shared driver runs.

    Holds only array-free static configuration: the linear backend, the
    tolerances, the corrector loop shape, the Jacobian setup policy, and the
    capability flags. The problem, the learnable parameters, and every array live
    on
    [BDFEngineCarry][diff_soup.timestepping.vsvo.engines.bdf.step.BDFEngineCarry].

    Attributes:
        backend: The linear solver used for the state-block factor / solve.
        rtol: Relative integration tolerance.
        atol: Absolute integration tolerance.
        loop_form: ``"fixed"`` (bounded freeze-on-done) or ``"adaptive"``
            (early-exit) corrector loop.
        jacobian_policy: ``"source"`` runs the coefficient-window setup gate;
            any other value refreshes the Jacobian every attempt.
        capabilities: The capability flags the driver validates against.
        controller_mode: ``"source"`` runs the reference accepted-step
            order/step controller; ``"flash"`` keeps the reference order
            decision but re-sizes the step on every accepted step — and on
            the first error-test rejection of a step — with the flash
            controller.
        controller_law: The flash controller's step-size law (``"pi"`` or
            ``"h211b"``); read only under ``controller_mode="flash"``.
        trace_kind: The per-step trace family name.
        ic_max_newton_iters: Newton iteration cap for the consistent-
            initialization projection; the projection exits early once
            converged, so the cap binds only for far-off-manifold seeds.
    """

    backend: Any = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    loop_form: str = eqx.field(static=True)
    jacobian_policy: str = eqx.field(static=True)
    capabilities: EngineCapabilities = eqx.field(static=True)
    controller_mode: str = eqx.field(static=True, default="source")
    controller_law: str = eqx.field(static=True, default="pi")
    trace_kind: str = eqx.field(static=True, default="bdf")
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
        [_fresh_carry][diff_soup.timestepping.vsvo.engines.bdf.step.BDFEngine._fresh_carry].
        ``history`` is a [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory]
        whose ``state`` is a [BDFHistory][diff_soup.timestepping.vsvo.engines.bdf.step.BDFHistory].
        When its ``continues`` flag is set and the new control leaves the
        algebraic closure at the carried state unchanged, the carry continues
        it (see
        [_continued_carry][diff_soup.timestepping.vsvo.engines.bdf.step.BDFEngine._continued_carry]).
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
        if not isinstance(record, BDFHistory):
            raise TypeError(
                "history holds the record of another solver "
                f"({type(record).__name__}); a JBDF solve continues only the "
                "Solution.history of an earlier JBDF solve."
            )
        dtype = record.state.zn.dtype
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
        carried_step = record.state.h
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
        dtype = st.zn.dtype
        given = pack_state(jnp.asarray(y0, dtype=dtype), jnp.asarray(lam0, dtype=dtype))
        return differential * given + (1.0 - differential) * st.zn[0]

    def _continued_carry(self, problem, record, t0, t1, y0, lam0, theta, cfg, control):
        """The carry of a solve that continues ``record`` from ``t0``.

        The record passes as it passes between two steps of one solve: the
        Nordsieck array, the order, the step the controller chose, the
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
        dtype = st.zn.dtype
        current = self._continued_state(st, y0, lam0, jnp.asarray(problem.id_mask, dtype=dtype))
        state = eqx.tree_at(lambda s: s.zn, st, st.zn.at[0].set(current))
        atol_rows = cfg.atol if getattr(cfg.atol, "ndim", 0) == 1 else None
        hmax = float(cfg.hmax)
        return BDFEngineCarry(
            state=state,
            tn=jnp.asarray(t0, dtype=dtype),
            ncf=jnp.asarray(0, jnp.int32),
            nef=jnp.asarray(0, jnp.int32),
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
        """Build the first-call carry: consistent-IC gate + initial-step heuristic.

        ``y0`` is the differential state the driver unpacked, ``ydot0`` the
        driver's raw right-hand side on the differential rows, and ``lam0`` the
        initial bordered globals. When the problem carries algebraic structure and
        the gate is enabled, the state, bordered globals, and mass-consistent derivative
        are projected onto the manifold (the pin-differential path); otherwise the
        derivative is taken mass-aware (``f / mass`` on the differential rows) and
        the state is used as given. The initial step follows an automatic search
        over the mass-consistent second derivative.

        ``step`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the entry step is the recorded one, the step
        the forward sweep's carry entered with, instead of a new search.
        """
        _reject_unsupported_problem(problem)
        n_u = int(problem.n_u)
        n_lam = int(problem.n_lam)
        n = n_u + n_lam
        u0 = jnp.asarray(y0)
        dtype = u0.dtype
        lam0 = jnp.asarray(lam0, dtype=dtype)
        ydot0 = jnp.asarray(ydot0, dtype=dtype)
        t0 = jnp.asarray(t0, dtype=dtype)
        t1 = jnp.asarray(t1, dtype=dtype)
        uround = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

        _, _, _diff, _alg_index, needs_projection = _structure(problem)

        # These solves factor this problem exactly as the corrector does, so
        # they are analyzed with the lowered problem's placement and the
        # accepted-root sensitivity precision this solve declared for each
        # derivative direction — never the backend's defaults.
        ic_layout, ic_precision = initialization_policy(problem, self.precision, mode="forward")
        _, ic_reverse_precision = initialization_policy(problem, self.precision, mode="reverse")
        slope_fn = bdf_slope_fn(
            problem,
            theta,
            tol=ic_gate_tolerance(self.atol, self.ic_tol),
            control=control,
            backend=self.backend,
            layout=ic_layout,
            precision=ic_precision,
            reverse_precision=ic_reverse_precision,
        )
        if not bool(cfg.consistent_ic) and needs_projection:
            raise ConfigurationError(
                "this problem has algebraic rows; "
                "consistent_ic=False is not supported for it. Enable "
                "consistent initialization."
            )
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

        # Initial step: an explicit dt0 override is used as-is; otherwise the
        # mass-consistent second-derivative search sizes it.
        dt0 = jax.lax.stop_gradient(jnp.asarray(cfg.dt0, dtype=dtype))

        def _auto_initial_step(_):
            # The initial step is a step-size control: stop its gradient here
            # (``bdf_initial_step`` documents that duty as the caller's). Beyond keeping
            # the step grid gradient-dead, this prunes the search's reverse
            # path — its ``sqrt(2/yddnrm)`` branch is singular when the second
            # derivative is exactly zero (a linear trajectory), and a
            # cotangent entering that subgraph would turn into 0 * inf = nan.
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

        # hmax / hmin clamps, then the endpoint clamp against t1, all BEFORE the
        # history seed scales zn[1] by h.
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

        state = init_bdf_state(y_aug0, ydot_aug0, hh, n, dtype.type)

        # The persisted two-tier blocks. Zero-seeded: the first attempt has
        # nst == 0, so both the setup gate and the Jacobian gate fire and
        # assemble fresh blocks without ever reading these zeros. The state
        # block is held as its factor, which starts empty (no factorization)
        # and is filled by the first setup's refactor.
        linear = problem_linear(self.backend, problem, precision=self.precision, dtype=dtype)
        zeros_pencil = (
            empty_held(linear, state_block_size(problem), dtype),
            jnp.zeros((n_u, n_lam), dtype=dtype),
            jnp.zeros((n_lam, n_u), dtype=dtype),
            jnp.zeros((n_lam, n_lam), dtype=dtype),
        )
        zeros_jac = (problem.zero_state_partials(dtype),) + zeros_pencil[1:]

        # The generation record naming the zero-seeded carry (tracking the
        # coupling-block counters exactly when the problem has
        # bordered-global rows), and its seed certificate — issued off-root
        # (no root has been accepted yet), so it never authorizes consuming
        # the seed. The border cache starts cold on every carry construction
        # — including an event restart, which rebuilds the carry here.
        generations0 = initial_generations(coupling_blocks=n_lam > 0)
        certificate0 = certify_exact_root(generations0, jnp.asarray(False))
        border_cache0 = initial_border_cache(n_u, n_lam, dtype) if n_lam > 0 else None

        # The flash controller's memory exists only under the flash profile;
        # a source-profile carry keeps the field empty (no extra leaves).
        if self.controller_mode == "flash":
            flash_mem = (jnp.asarray(FLASH_ERR_SEED, dtype=dtype), hh, jnp.asarray(True))
        else:
            flash_mem = None

        return BDFEngineCarry(
            state=state,
            tn=t0,
            ncf=jnp.asarray(0, jnp.int32),
            nef=jnp.asarray(0, jnp.int32),
            ewt=ewt0,
            theta=theta,
            control=control,
            nflag=jnp.asarray(NonlinearFlag.FIRST_CALL, jnp.int32),
            saved_jac=zeros_jac,
            pencil=zeros_pencil,
            ic_alg_residual_norm=alg_norm,
            hmax_inv=hmax_inv,
            hmin=hmin,
            t1_end=t1,
            problem=problem,
            trace_enabled=bool(cfg.trace),
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

        Three groups, in order: the convergence flag (plus the
        precision-attribution flags under the adaptive mode); the scaled
        local error and its test; and the next-step controls the accepted
        branch chooses from the order-local error norms (step ratio, order,
        next step, wait counter, the three order ratios, the chosen branch and
        the constant-step counter). Everything the attempt decides is a
        function of these.
        """
        flag = jnp.zeros((), jnp.bool_)
        count = jnp.zeros((), jnp.int32)
        real = jnp.zeros((), jnp.float32)
        converged = (flag, flag, flag, flag) if self.precision.solve_mode == "adaptive" else (flag,)
        error_test = (real, flag)
        next_step = (real, count, real, count, real, real, real, count, count)
        return converged + error_test + next_step

    def attempt_step(self, carry: BDFEngineCarry, decision=None):
        """Advance one pass of the step-attempt retry loop; report accept or fail.

        Runs the pending order/step adjustment, the predictor, the coefficient
        recurrence, the correction-form corrector, and the local-error test. A
        converged, error-test-passing attempt completes the step (new order/step,
        advanced history, refreshed weights, endpoint-clamped next step) and
        reports ``accepted=True``. A corrector or error-test failure restores the
        history, walks the failure ladder, and reports ``accepted=False`` -- a
        retry sentinel when the ladder asks for another attempt (the shrunk step
        already threaded into the carry), else a termination code.

        ``decision`` is ``None`` or ``(replay, recorded)``: on a replay of a
        checkpointed reverse pass the attempt uses the recorded decision inputs
        (see [decision_template][]) so it decides as the forward sweep did.
        """
        problem = carry.problem
        st = carry.state
        theta = carry.theta
        ewt = carry.ewt
        dtype = st.h.dtype
        dtype_t = dtype.type
        n_u = int(problem.n_u)
        uround = jnp.finfo(dtype).eps
        hmax_inv = carry.hmax_inv
        hmin = carry.hmin
        t1 = carry.t1_end
        saved_t = carry.tn
        nflag_in = carry.nflag
        one_f = jnp.ones((), dtype)

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

        # Apply a pending order change and rescale the history, but ONLY on the
        # first attempt of a step and only when the step size or order actually
        # changed (``nst > 0 && hprime != h``). The rescale is how the history
        # moves onto the endpoint-clamped step the previous step chose.
        apply_adjust = (nflag_in == NonlinearFlag.FIRST_CALL) & (st.nst > 0) & (st.hprime != st.h)
        zn_a, q_a, l_int_a, qwait_a, h_a, hscale_a, nscon_a = bdf_adjust_params(st)
        zn_top = jnp.where(apply_adjust, zn_a, st.zn)
        q_adj = jnp.where(apply_adjust, q_a, st.q).astype(jnp.int32)
        L_adj = jnp.where(apply_adjust, l_int_a, st.L).astype(jnp.int32)
        qwait_adj = jnp.where(apply_adjust, qwait_a, st.qwait).astype(jnp.int32)
        h_adj = jnp.where(apply_adjust, h_a, st.h)
        hscale_adj = jnp.where(apply_adjust, hscale_a, st.hscale)
        nscon_adj = jnp.where(apply_adjust, nscon_a, st.nscon).astype(jnp.int32)
        st_adj = eqx.tree_at(
            lambda s: (s.zn, s.q, s.L, s.qwait, s.h, s.hscale, s.nscon),
            st,
            (zn_top, q_adj, L_adj, qwait_adj, h_adj, hscale_adj, nscon_adj),
        )

        # Advance the internal time with the endpoint snap (touches tn only),
        # then the Pascal additions on zn.
        tn_pred = saved_t + h_adj
        tn_pred = jnp.where((tn_pred - t1) * h_adj > 0, t1, tn_pred)
        zn_pred = bdf_predict_history(st_adj.zn, q_adj)

        # Compute the corrector coefficients and the leading-coefficient triple.
        l_new, tq_new, rl1, gamma, gammap_set, gamrat = bdf_set_coeffs(
            st_adj.l, st_adj.tau, h_adj, q_adj, qwait_adj, st_adj.nst, st_adj.gammap, st_adj.tq
        )
        cj = cj_from_hinvgak(gamma)

        # The correction-form corrector. The setup-gate memory
        # (nst/nstlp/nstlj/gammap/crate) rides on st_adj; the two-tier blocks ride
        # on the carry.
        ctx = BDFNlsContext(
            t=tn_pred,
            theta=theta,
            gamma=gamma,
            cj=cj,
            gamrat=gamrat,
            rl1_zn1=rl1 * zn_pred[1],
            pred_y=zn_pred[0],
            ewt=ewt,
            tq4=tq_new[4],
            nflag=nflag_in,
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
        nls = bdf_corrector_root(
            problem,
            backend,
            linear,
            sens_analysis,
            st_adj,
            ctx,
            self._directive(),
            self.loop_form,
            frozen_saved_jac=carry.saved_jac,
            frozen_pencil=carry.pencil,
            reverse_analysis=rev_analysis,
            generations=carry.generations,
            border_cache=carry.border_cache,
            newton_tol_factor=self.newton_tol_factor,
            jacobian_values_dtype=self.jacobian_values_dtype,
            detect_precision=adaptive,
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
            converged = (did_converge, prec_false_conv, prec_stagnant, lin_escalated)
        else:
            converged = (did_converge,)
        n_conv = len(converged)
        converged = replay_decisions(_decision_part(decision, 0, n_conv), converged)
        did_converge = converged[0]
        if adaptive:
            prec_false_conv, prec_stagnant, lin_escalated = converged[1:]
            prec_retry = prec_false_conv | (prec_stagnant & jnp.logical_not(did_converge))
        else:
            prec_retry = None

        # Generation discipline on the factor/Jacobian carry: the point,
        # matrix, and factor counters advance together once per setup that
        # rebuilt the carried iteration matrix and its factor (an attempt may
        # run up to three setup stages — the counters advance by the exact
        # setup count, so two factors built within one attempt never share a
        # counter value), and the coupling-block counters advance once per
        # fresh-Jacobian evaluation. The certificate is issued only here, at
        # this attempt's root-acceptance decision, and it attests exactness
        # only when the factorization point provably IS the accepted root:
        # the corrector converged, the matrix was rebuilt from a FRESH
        # Jacobian evaluation this attempt, and the total applied correction
        # is exactly zero. Issuing anywhere else would falsely attest matrix
        # values evaluated at a different point; the engine's own
        # window-gated reuse policy never reads the certificate — it exists
        # for generation-keyed consumers of the carried factor.
        generations_next = carry.generations.advance(
            point=nls.nsetups, matrix=nls.nsetups, factor=nls.nsetups
        )
        if int(problem.n_lam) > 0:
            # A differential-global border recombines its coupling blocks at
            # every setup (cj enters D through the global mass), so the
            # coupling counters track setups; a pure-algebraic border keeps
            # the fresh-Jacobian cadence.
            border_tracks_factor = bool(np.asarray(problem.id_mask)[int(problem.n_u) :].any())
            coupling_events = nls.nsetups if border_tracks_factor else nls.nje
            generations_next = generations_next.advance(
                b=coupling_events, c=coupling_events, d=coupling_events
            )
        # Only a PRIMARY setup that evaluated a fresh Jacobian qualifies: a
        # saved-Jacobian pencil rebuild carries J values from an earlier
        # trajectory point, so even a zero-correction root would falsely
        # attest them, and proving exactness across a retry setup is not
        # worth the plumbing (under-issuance is always safe).
        primary_fresh_jacobian = jnp.logical_and(nls.call_setup, nls.jbad)
        no_retry_setup = (nls.nsetups - nls.call_setup.astype(jnp.int32)) == 0
        root_exact = jnp.logical_and(
            did_converge,
            jnp.logical_and(
                jnp.logical_and(primary_fresh_jacobian, no_retry_setup),
                jnp.all(nls.acor == 0.0),
            ),
        )
        certificate_next = certify_exact_root(generations_next, root_exact)

        # On a corrector failure the convergence-failure ladder decides a retry
        # or an abort. A dense-path corrector reports only a recoverable
        # convergence failure.
        nflag_for_handle = jnp.where(
            did_converge,
            jnp.asarray(StepFlag.STEP_SUCCESS, jnp.int32),
            jnp.asarray(SUN_NLS_CONV_RECVR, jnp.int32),
        )
        kflag_cf, eta_cf, ncf_new, _etamax_cf, nflag_cf = bdf_handle_nflag(
            nflag_for_handle, h_adj, st.eta, carry.ncf, hmin, dtype_t
        )
        if adaptive:
            # A precision-attributed failure retries at the promoted rung with
            # the SAME step and does not count as a convergence failure (the
            # rung is monotone within a step, so these retries are bounded).
            one_dt = jnp.asarray(1.0, dtype_t)
            kflag_cf = jnp.where(
                prec_retry, jnp.asarray(NonlinearFlag.PREDICT_AGAIN, jnp.int32), kflag_cf
            )
            eta_cf = jnp.where(prec_retry, one_dt, eta_cf)
            ncf_new = jnp.where(prec_retry, carry.ncf, ncf_new).astype(jnp.int32)
            nflag_cf = jnp.where(
                prec_retry, jnp.asarray(NonlinearFlag.PREV_PREC_FAIL, jnp.int32), nflag_cf
            )

        # On convergence, form the scaled error and test it; on failure walk
        # the error-test ladder.
        dsm, error_test_pass = bdf_do_error_test_accept(nls.acnrm, tq_new, dtype_t)
        dsm, error_test_pass = replay_decisions(
            _decision_part(decision, n_conv, n_conv + 2), (dsm, error_test_pass)
        )
        accept = jnp.logical_and(did_converge, error_test_pass)
        # Flash profile: one controller evaluation per attempt, driven by the
        # scaled local error dsm (accept boundary 1), sizes both the accepted
        # step's successor and a first error-test rejection's retry. Static
        # dispatch: under the source profile this branch is never built and
        # the source paths are unchanged.
        if self.controller_mode == "flash":
            flash_err_prev, flash_h_prev, flash_last_accept = carry.flash_mem
            flash_res = flash_step_control(
                dsm,
                flash_err_prev,
                h=h_adj,
                h_prev=flash_h_prev,
                order=q_adj,
                last_accept=flash_last_accept,
                first_step=st.nst == 0,
                hmax_inv=hmax_inv,
                hmin=hmin,
                max_order=QMAX,
                law=self.controller_law,
            )
        (
            kflag_ef,
            eta_ef,
            q_ef,
            L_ef,
            qwait_ef,
            nef_new,
            _etamax_ef,
            order_drop,
            restart_q1,
            _nscon_ef_ladder,
        ) = bdf_error_test_fail(dsm, st_adj, carry.nef, hmin, dtype_t)
        # Flash profile: the first error-test rejection of a step retries at
        # the flash controller's rejection ratio (capped at 0.9) instead of
        # the reference shrink formula; the ladder's escalation rungs (the
        # capped repeats, the forced order drop, the order-1 restart) and its
        # abort are unchanged.
        if self.controller_mode == "flash":
            first_ef_retry = (carry.nef == 0) & (kflag_ef == NonlinearFlag.TRY_AGAIN)
            eta_ef = jnp.where(first_ef_retry, flash_res.eta_reject, eta_ef)

        # Restore the predicted history for BOTH failure paths: undo the Pascal
        # additions at the order this attempt predicted with.
        zn_restored = bdf_restore_history(zn_pred, q_adj)

        # --- accept branch: complete the step, decide the next order/step ---
        st_complete_in = eqx.tree_at(lambda s: (s.zn, s.l, s.tq), st_adj, (zn_pred, l_new, tq_new))
        new_state_complete = bdf_complete_step(st_complete_in, nls.acor, dtype_t)
        # Under the flash profile the reference post-failure deferral (the
        # carried step-growth cap forced to 1 by the failure ladders, holding
        # the step AND the order on the next accepted step) never fires: the
        # flash controller's own post-rejection growth suppression replaces
        # the step hold, and the order decision follows the order ladder on
        # its normal wait-counter cadence.
        prep_etamax = (
            st_adj.etamax
            if self.controller_mode != "flash"
            else jnp.asarray(_ETA_MAX_POST_SUCCESS, dtype)
        )
        (
            eta_p,
            qprime_p,
            hprime_p,
            qwait_p,
            etaq_p,
            etaqm1_p,
            etaqp1_p,
            branch_p,
            nscon_p,
            zn_next,
        ) = bdf_prepare_next_step(
            new_state_complete, nls.acor, dsm, prep_etamax, hmax_inv, hmin, ewt, dtype_t
        )
        next_step = replay_decisions(
            _decision_part(decision, n_conv + 2, n_conv + 11),
            (eta_p, qprime_p, hprime_p, qwait_p, etaq_p, etaqm1_p, etaqp1_p, branch_p, nscon_p),
        )
        (eta_p, qprime_p, hprime_p, qwait_p, etaq_p, etaqm1_p, etaqp1_p, branch_p, nscon_p) = (
            next_step
        )
        # The order-raise history refresh follows the branch actually taken,
        # recorded or recomputed, so a replayed order change carries the
        # history it expects. Off a replay this equals the controller's own
        # ``zn_out``; the refresh keeps its gradient through ``acor``.
        zn_next = jnp.where(
            branch_p == ChosenBranch.RAISE,
            new_state_complete.zn.at[QMAX].set(nls.acor),
            new_state_complete.zn,
        )
        decided = converged + (dsm, error_test_pass) + next_step
        # Flash profile: keep the reference order decision (qprime / qwait /
        # the order-raise history refresh in zn) but re-size the step on
        # every accepted step with the flash controller instead of the
        # reference eta laws and their step-retention deadband.
        if self.controller_mode == "flash":
            eta_p = flash_res.eta
            hprime_p = flash_res.h_new
        # Reset the step-growth cap after a successful step; early and general
        # steps share the value 10.
        etamax_next = jnp.asarray(_ETA_MAX_POST_SUCCESS, dtype)

        # The scaled correction is the exported local-error estimate
        # (``acor * tq[2]``). The driver reports dsm as the error
        # estimate and exposes no consumer for the full scaled vector, so it is
        # not materialized here (mirroring the implicit-DAE engine).

        # Endpoint clamp on the NEXT step: never edit h mid-attempt (the Nordsieck
        # history bakes h in). If the next step would overtake the endpoint,
        # shrink hprime/eta so the following attempt's rescale step moves the
        # history onto it.
        tn_snapped = jnp.where(
            jnp.abs(tn_pred - t1) <= _TN_SNAP_FUZZ * uround * (jnp.abs(tn_pred) + jnp.abs(h_adj)),
            t1,
            tn_pred,
        )
        troundoff = _TN_SNAP_FUZZ * uround * (jnp.abs(tn_snapped) + jnp.abs(h_adj))
        at_end = jnp.abs(tn_snapped - t1) <= troundoff
        overtake = (tn_snapped + hprime_p - t1) * h_adj > 0
        clamp = jnp.logical_and(jnp.logical_not(at_end), overtake)
        hprime_c = jnp.where(
            clamp, (t1 - tn_snapped) * (1.0 - _ENDPOINT_CLAMP_FUZZ * uround), hprime_p
        )
        eta_c = jnp.where(clamp, hprime_c / h_adj, eta_p)

        accept_state = eqx.tree_at(
            lambda s: (s.zn, s.qprime, s.hprime, s.eta, s.qwait, s.nscon, s.etamax, s.rl1, s.gamma),
            new_state_complete,
            (zn_next, qprime_p, hprime_c, eta_c, qwait_p, nscon_p, etamax_next, rl1, gamma),
        )
        y_new_acc = new_state_complete.zn[0][:n_u]
        lam_acc = new_state_complete.zn[0][n_u:]
        ydot_acc = new_state_complete.zn[1][:n_u] / h_adj

        # --- convergence-failure branch: restore + rescale for the retry ---
        zn_cf, h_cf, hscale_cf, nscon_cf = bdf_rescale_history(
            zn_restored, h_adj, hscale_adj, eta_cf, q_adj
        )
        conv_fail_state = eqx.tree_at(
            lambda s: (s.zn, s.h, s.hscale, s.eta, s.nscon, s.etamax, s.l, s.tq, s.rl1, s.gamma),
            st_adj,
            (zn_cf, h_cf, hscale_cf, eta_cf, nscon_cf, one_f, l_new, tq_new, rl1, gamma),
        )

        # --- error-test-failure branch: restore, then rescale / order-drop /
        # order-1 restart, keyed on the ladder's decision flags ---
        zn_wf, h_wf, hscale_wf, _nscon_wf = bdf_rescale_history(
            zn_restored, h_adj, hscale_adj, eta_ef, q_adj
        )
        zn_od0 = bdf_decrease_order(zn_restored, st_adj.tau, hscale_adj, q_adj)
        zn_od, h_od, hscale_od, _nscon_od = bdf_rescale_history(
            zn_od0, h_adj, hscale_adj, eta_ef, q_ef
        )
        # The order-1 restart does NOT rescale: h is updated directly and zn[1] is
        # rebuilt from a fresh slope evaluation with zn[0] unchanged. This
        # engine's Nordsieck convention is zn[1] = h * ydot with
        # F(t, y, ydot) = 0 (see init_carry and nonlinear.py). The view's
        # startup operation is run only when the ladder chooses the restart.
        h_q1 = h_adj * eta_ef
        slope_layout, slope_precision = initialization_policy(
            problem, self.precision, mode="forward"
        )
        _, slope_reverse_precision = initialization_policy(problem, self.precision, mode="reverse")
        slope_fn = bdf_slope_fn(
            problem,
            theta,
            tol=ic_gate_tolerance(self.atol, self.ic_tol),
            control=carry.control,
            backend=self.backend,
            layout=slope_layout,
            precision=slope_precision,
            reverse_precision=slope_reverse_precision,
        )
        run_restart = jnp.logical_and(
            restart_q1, jnp.logical_and(did_converge, jnp.logical_not(accept))
        )
        # Under vmap the slope runs only when some lane restarts.
        ydot_q1 = lane_cond(
            run_restart,
            lambda ops: slope_fn(ops[0], ops[1], ops[2]),
            lambda ops: ops[2],
            (saved_t, zn_restored[0], zn_restored[1] / h_adj),
            false_is_free=True,
        )
        zn_q1 = zn_restored.at[1].set(h_q1 * ydot_q1)

        zn_ef = jnp.where(order_drop, zn_od, jnp.where(restart_q1, zn_q1, zn_wf))
        h_ef = jnp.where(order_drop, h_od, jnp.where(restart_q1, h_q1, h_wf))
        hscale_ef = jnp.where(order_drop, hscale_od, jnp.where(restart_q1, h_q1, hscale_wf))
        # Every error-test failure resets the constant-step-run counter (through
        # the rescale on the first two rungs, explicitly on the restart).
        error_fail_state = eqx.tree_at(
            lambda s: (
                s.zn,
                s.h,
                s.hscale,
                s.eta,
                s.q,
                s.L,
                s.qwait,
                s.nscon,
                s.etamax,
                s.l,
                s.tq,
                s.rl1,
                s.gamma,
            ),
            st_adj,
            (
                zn_ef,
                h_ef,
                hscale_ef,
                eta_ef,
                q_ef,
                L_ef,
                qwait_ef,
                jnp.zeros((), jnp.int32),
                one_f,
                l_new,
                tq_new,
                rl1,
                gamma,
            ),
        )

        # --- select the branch ---
        fail_state = _where_state(did_converge, error_fail_state, conv_fail_state)
        new_state = _where_state(accept, accept_state, fail_state)
        # The two-tier solver state persists across attempts and accepted steps:
        # it is re-evaluated only inside a setup, so carry the corrector's
        # post-setup values (fresh where a gate fired, the prior values
        # otherwise) regardless of accept or reject.
        new_state = eqx.tree_at(
            lambda s: (s.gammap, s.gamrat, s.crate, s.nstlp, s.nstlj, s.acnrm),
            new_state,
            (
                nls.gammap_out,
                nls.gamrat_out,
                nls.crate,
                nls.nstlp_out,
                nls.nstlj_out,
                nls.acnrm,
            ),
        )

        # Failure counters: each ladder bumps only its own counter; both reset on
        # accept.
        ncf_next = jnp.where(
            accept, jnp.asarray(0, jnp.int32), jnp.where(did_converge, carry.ncf, ncf_new)
        ).astype(jnp.int32)
        nef_next = jnp.where(
            accept, jnp.asarray(0, jnp.int32), jnp.where(did_converge, nef_new, carry.nef)
        ).astype(jnp.int32)

        # nflag into the next attempt: FIRST_CALL after an accept, PREV_ERR_FAIL /
        # PREV_CONV_FAIL on the respective retries.
        nflag_next = jnp.where(
            accept,
            jnp.asarray(NonlinearFlag.FIRST_CALL, jnp.int32),
            jnp.where(did_converge, jnp.asarray(NonlinearFlag.PREV_ERR_FAIL, jnp.int32), nflag_cf),
        ).astype(jnp.int32)

        new_ewt = error_weight(
            new_state_complete.zn[0],
            self.rtol,
            self.atol if carry.atol_rows is None else carry.atol_rows,
        )

        # Flash memory advance: the error memory refreshes on every attempt
        # whose corrector converged (a diverged attempt's estimate is not
        # meaningful), the step-size memory on accepted steps only, and the
        # last-accept flag on every attempt — a rejection suppresses growth
        # on the next accepted step.
        if self.controller_mode == "flash":
            flash_mem_next = (
                jax.lax.stop_gradient(jnp.where(did_converge, dsm, flash_err_prev)),
                jax.lax.stop_gradient(jnp.where(accept, h_adj, flash_h_prev)),
                accept,
            )
        else:
            flash_mem_next = None

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

        new_carry = BDFEngineCarry(
            state=new_state,
            tn=jnp.where(accept, tn_snapped, saved_t),
            ncf=ncf_next,
            nef=nef_next,
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
            flash_mem=flash_mem_next,
            prec_state=prec_next,
            generations=generations_next,
            factor_certificate=certificate_next,
            border_cache=nls.border_cache_out,
            atol_rows=carry.atol_rows,
        )

        # Result code: accept and both retry sentinels map to success (1); an
        # abort maps to its termination code (step collapse / nonlinear failure).
        result_fail = jnp.where(
            did_converge, kflag_to_result_code(kflag_ef), kflag_to_result_code(kflag_cf)
        )
        result_code = jnp.where(accept, jnp.asarray(_CODE_SUCCESS, jnp.int32), result_fail).astype(
            jnp.int32
        )

        # Reported next step / order and the pre-attempt state on a reject.
        y_prev = st.zn[0][:n_u]
        lam_prev = st.zn[0][n_u:]
        ydot_prev = st.zn[1][:n_u] / st.h
        h_fail = jnp.where(did_converge, h_ef, h_cf)
        q_fail = jnp.where(did_converge, q_ef, q_adj).astype(jnp.int32)

        # The order-1 restart does one extra right-hand-side evaluation.
        restart_extra = jnp.where(
            jnp.logical_and(jnp.logical_and(jnp.logical_not(accept), did_converge), restart_q1),
            jnp.asarray(1, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

        trace_record = None
        if carry.trace_enabled:
            trace_record = self._trace_record(
                accept=accept,
                did_converge=did_converge,
                q_adj=q_adj,
                L_adj=L_adj,
                tau=st_adj.tau,
                l_new=l_new,
                tq_new=tq_new,
                rl1=rl1,
                gamma=gamma,
                gamrat=gamrat,
                nls=nls,
                nflag_in=nflag_in,
                kflag_ef=kflag_ef,
                kflag_cf=kflag_cf,
                dsm=dsm,
                nef_next=nef_next,
                etamax_used=st_adj.etamax,
                etaq_p=etaq_p,
                etaqm1_p=etaqm1_p,
                etaqp1_p=etaqp1_p,
                branch_p=branch_p,
                qprime_p=qprime_p,
                hprime_c=hprime_c,
                qwait_p=qwait_p,
                qwait_ef=qwait_ef,
                qwait_adj=qwait_adj,
                nscon_p=nscon_p,
                q_fail=q_fail,
                h_fail=h_fail,
                dtype=dtype,
            )

        # Border/Schur accounting: the corrector reports the coupling-block
        # work it actually performed — the generation-keyed cache rebuilds a
        # tier only when its keys moved, so unchanged generations add nothing.
        report = StepReport(
            accepted=accept,
            t_new=jnp.where(accept, tn_snapped, saved_t),
            y_new=jnp.where(accept, y_new_acc, y_prev),
            ydot_new=jnp.where(accept, ydot_acc, ydot_prev),
            lam_new=jnp.where(accept, lam_acc, lam_prev),
            h_new=jnp.where(accept, hprime_c, h_fail),
            k_new=jnp.where(accept, qprime_p, q_fail).astype(jnp.int32),
            err_estimate=dsm,
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

    # -- trace -------------------------------------------------------------
    def _trace_record(
        self,
        *,
        accept,
        did_converge,
        q_adj,
        L_adj,
        tau,
        l_new,
        tq_new,
        rl1,
        gamma,
        gamrat,
        nls,
        nflag_in,
        kflag_ef,
        kflag_cf,
        dsm,
        nef_next,
        etamax_used,
        etaq_p,
        etaqm1_p,
        etaqp1_p,
        branch_p,
        qprime_p,
        hprime_c,
        qwait_p,
        qwait_ef,
        qwait_adj,
        q_fail,
        h_fail,
        nscon_p,
        dtype,
    ) -> BDFTraceRecord:
        """Fill the engine's per-attempt trace fields (the driver adds the header).

        Records the order and coefficients, the leading-coefficient triple and
        the setup decision, the corrector convergence quantities and the
        accepted-correction norm, the branch-selected failure counters and status
        flags, and the next order/step choice.
        """
        zeros = BDFTraceRecord.zeros(dtype)
        vlen = int(zeros.tau.shape[0])
        one_f = jnp.ones((), dtype)
        zero_f = jnp.zeros((), dtype)
        zero_i = jnp.asarray(0, jnp.int32)
        # The corrector resets jcur to false on any converged solve; the raw
        # last-fired-arm value is reported only on a non-converged attempt.
        jcur = jnp.where(nls.converged, jnp.asarray(False), nls.jcur)
        kflag = jnp.where(
            accept,
            jnp.asarray(StepFlag.STEP_SUCCESS, jnp.int32),
            jnp.where(did_converge, kflag_ef, kflag_cf),
        ).astype(jnp.int32)
        qwait_fail = jnp.where(did_converge, qwait_ef, qwait_adj)
        return BDFTraceRecord(
            header=StepTraceHeader.zeros(dtype),
            q=q_adj.astype(jnp.int32),
            L=L_adj.astype(jnp.int32),
            tau=_pad_to(tau, vlen),
            l=_pad_to(l_new, vlen),
            rl1=rl1,
            gamma=gamma,
            gammap=nls.gammap_out,
            gamrat=gamrat,
            tq=_pad_to(tq_new, vlen),
            call_setup=jnp.asarray(nls.call_setup, jnp.bool_),
            setup_condition=zero_i,
            jbad=jnp.asarray(nls.jbad, jnp.bool_),
            jcur=jnp.asarray(jcur, jnp.bool_),
            nsetups=nls.nsetups.astype(jnp.int32),
            nje=nls.nje.astype(jnp.int32),
            nstlj=nls.nstlj_out.astype(jnp.int32),
            delnrm=nls.delnrm,
            crate=nls.crate,
            dcon=nls.dcon,
            acnrm=nls.acnrm,
            acnrmcur=jnp.asarray(nls.acnrmcur, jnp.bool_),
            nflag=jnp.asarray(nflag_in, jnp.int32),
            kflag=kflag,
            dsm=dsm,
            nef=nef_next.astype(jnp.int32),
            netf=nef_next.astype(jnp.int32),
            etamax=jnp.where(accept, etamax_used, one_f),
            etaq=jnp.where(accept, etaq_p, zero_f),
            etaqm1=jnp.where(accept, etaqm1_p, zero_f),
            etaqp1=jnp.where(accept, etaqp1_p, zero_f),
            chosen_branch=jnp.where(
                accept, branch_p, jnp.asarray(ChosenBranch.HOLD, jnp.int32)
            ).astype(jnp.int32),
            qprime=jnp.where(accept, qprime_p, q_fail).astype(jnp.int32),
            hprime=jnp.where(accept, hprime_c, h_fail),
            qwait=jnp.where(accept, qwait_p, qwait_fail).astype(jnp.int32),
            nscon=jnp.where(accept, nscon_p, zero_i).astype(jnp.int32),
            hin=jnp.zeros(vlen, dtype),
            dky_coeffs=jnp.zeros(vlen, dtype),
        )

    def state_at_entry(self, carry: BDFEngineCarry):
        """The augmented ``[u; lam]`` state the solve starts from.

        Read on the freshly built carry, this is the consistent-initialized
        state (``zn[0]``; the driver records it as the ``t0`` output sample).
        """
        return carry.state.zn[0]

    def step_at_entry(self, carry: BDFEngineCarry):
        """The step size the carry's next attempt starts from.

        Read on a freshly built carry, this is the entry step ``init_carry``
        chose; the driver records it after an event restart so a replayed
        restart enters with the same step.
        """
        return carry.state.h

    def history_at_exit(self, carry: BDFEngineCarry) -> BDFHistory:
        """The [BDFHistory][diff_soup.timestepping.vsvo.engines.bdf.step.BDFHistory] a later solve continues from.

        Read on the final carry: everything that persists from one step to the
        next, without the solve's configuration.
        """
        return BDFHistory(
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

    def slope_at_entry(self, carry: BDFEngineCarry):
        """The augmented state derivative the solve starts from.

        Read on the freshly built carry: the Nordsieck convention seeds
        ``zn[1] = h * ydot``, so the entry slope divides the step back out.
        """
        st = carry.state
        return st.zn[1] / jnp.where(st.h != 0, st.h, 1.0)

    # -- dense output ------------------------------------------------------
    def dense(self, carry: BDFEngineCarry, t):
        """Evaluate the differential state and derivative at ``t`` on the last step.

        Delegates to the dense interpolant over the accepted interval and returns
        the local state block. Callers that need bordered rows use
        [dense_full][diff_soup.timestepping.vsvo.engines.bdf.step.BDFEngine.dense_full],
        which also interpolates differential globals.
        """
        n_u = int(carry.problem.n_u)
        y_full, ydot_full = self.dense_full(carry, t)
        return y_full[:n_u], ydot_full[:n_u]

    def dense_full(self, carry: BDFEngineCarry, t):
        """Evaluate the full augmented interpolant (state and bordered rows) at ``t``.

        Every row is reported at interpolation accuracy. Strict algebraic
        constraint satisfaction is the projected output policy's job.
        Interpolated algebraic-row derivatives are internal only and must be
        zeroed if a public derivative surface is added.
        """
        return bdf_dense(carry.state, carry.tn, t)


@functools.lru_cache(maxsize=None)
def bdf_engine(
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
) -> BDFEngine:
    """Build the backward-differentiation-formula engine kernel from a profile.

    Resolves the linear backend from ``linear`` (the dense CPU backend for the
    default and ``"auto"`` names) and reads the corrector loop shape, the
    Jacobian setup policy, and the accepted-step controller mode from
    ``profile``. Identical configurations return one memoised engine object,
    so repeated solves share the driver's compiled program instead of
    recompiling per construction.

    Args:
        profile: The solver execution profile (``nonlinear_mode`` selects the
            corrector loop shape; ``jacobian_policy`` selects the setup
            policy; ``controller_mode`` selects the accepted-step step-size
            controller).
        rtol: Relative integration tolerance.
        atol: Absolute integration tolerance.
        linear: Linear-backend name (``"auto"``, ``"cudss"``, ``"klu"``,
            ``"superlu"``, ``"dense"``) or a backend instance.
        newton_tol_factor: Corrector-band scale — widens the Newton
            convergence-test tolerance relative to the step tolerance
            (``1.0`` = the standard band).
        ic_max_newton_iters: Newton iteration cap for the consistent-
            initialization projection; the projection exits early once
            converged, so the cap binds only for seeds far from the
            algebraic manifold.
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
        The configured [BDFEngine][].
    """
    backend = resolve_backend(linear)
    backend._validate_capabilities(Layout(), precision)
    return BDFEngine(
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


__all__ = ["BDFEngine", "BDFEngineCarry", "BDFHistory", "bdf_engine"]
