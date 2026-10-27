"""The fully-implicit corrector for the DAE engine.

This module implements the fully-implicit Newton corrector: the
residual ``F(t, y, ydot, theta)``, the exact materialized iteration matrix
``A(cj) = dF/dy + cj dF/dy'`` read from the problem's linearization plan, the
linear-solver setup gate driven by the leading-coefficient window, and an
inexact-Newton convergence test using a rate-based estimate (the rate memory
``ss``, the first-iteration direct pass, and the ``rate > 0.9`` divergence
trip).

The corrector solves ``F = 0`` over the augmented state ``x = [u; lam]`` for the
correction ``ee`` under the fixed-leading-coefficient update ``yy = yypredict +
ee`` and ``yp = yppredict + cj*ee``. The problem's mass mask makes algebraic
rows derivative-free while differential rows, including differential globals
in the bordered partition, contribute their derivative. It runs its own Newton
loop against the linear backend's factor / solve plus the Schur reduction over
the global block,
so IDA's iteration counts, rate memory, and setup decisions are emitted natively
and the shared nonlinear layer is untouched.

The reuse decision lives here: a
[LinearizationDirective][diff_soup.nonlinear.result.LinearizationDirective]
selects the *setup policy*. Its static ``refresh_jacobian`` flag distinguishes
an always-fresh factorization from the leading-coefficient-window gate; the
traced per-attempt setup decision (``call_lsetup``) executes that policy inside
the corrector. The ``2/(1+cjratio)`` update rescale that compensates a stale
leading coefficient on a reused factor is also computed here from the runtime
coefficient ratio.

The iteration matrix persists across attempts and steps: the corrector reuses a
set of carried blocks whenever its gate stays closed. When the gate fires only
because the leading coefficient moved -- no first step, no failure preceding the
attempt -- it reuses the state Jacobian ``F_y`` saved at the last fresh setup and
re-evaluates only ``F_ydot``, recomposing ``A(cj)`` at the exact leading
coefficient without a fresh ``F_y`` evaluation. A first step, a failure-triggered
setup, and the fresh-Jacobian retry each evaluate both split partials fresh. The
blocks and the saved partials in effect at exit come back on the result so the
caller can thread them into the next attempt.

Two loop shapes are available through ``loop_form``. ``"fixed"`` runs a bounded
``fori_loop`` that freezes once the convergence test is met; ``"adaptive"`` runs
an early-exit ``while_loop``. Both use the same rate-based Newton convergence
test, so the choice never changes the accepted iterate or the reported
iteration counts.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.nonlinear.border_cache import border_cache_solve, refresh_border_cache
from diff_soup.nonlinear.generations import Generations
from diff_soup.nonlinear.result import LinearizationDirective
from diff_soup.nonlinear.solver import _stable_small_solve
from diff_soup.timestepping.vsvo.common.adaptive import (
    STAGNATION_LATCH,
    TOP_RUNG,
    stagnation_flag,
)
from diff_soup.timestepping.vsvo.common.linear_ops import (
    empty_held,
    linearize,
)
from diff_soup.timestepping.vsvo.common.weights import wrms_norm
from diff_soup.utils import safe_div, safe_power
from diff_soup.utils.lanes import live_lane_cond

from .state import DIRECT_TEST_TOL, NEWTON_CONV_COEFF

# --- corrector constants ---
MAX_NEWTON_ITERS = 4
"""Maximum Newton iterations per step attempt."""

NEWTON_DIVERGENCE_RATE = 0.9
"""Divergence threshold on the estimated Newton convergence rate."""

_SS_FRESH = 20.0
"""Rate memory seeded right after a fresh linear-solver setup."""

_SS_CJ_CHANGED = 100.0
"""Rate memory when the leading coefficient changed without a setup."""

_CJ_WINDOW_HALFWIDTH = 0.25
"""Half-width of the leading-coefficient no-setup window; ``dcj = 0.25`` gives
the window ``(0.6, 1.6667)``."""

MSBJ = 51
"""Maximum number of steps a reused state Jacobian may be carried.

A setup that fires only because the leading coefficient moved recomposes the
iteration matrix from the state Jacobian saved at the last fresh setup rather
than re-evaluating it. This window bounds how stale that saved Jacobian may
become: once ``nst >= nstlj + MSBJ`` the next setup evaluates it fresh, so a
reused state Jacobian is never more than ``MSBJ`` steps old.

The bound is needed because the setup gate itself watches only the leading
coefficient and cannot observe how far the state has moved. Reusing the state
Jacobian across setups is a deliberate departure from the reference DAE
integrator, which evaluates both partials at every setup; the multistep
engines use their own nearby resource windows. In every engine a problem's
``max_fy_reuse_steps`` declaration further caps the engine default.

This is a safety bound on worst-case staleness, not a performance tuning knob.
On a problem whose state Jacobian is expensive and slowly varying, reuse is
most of the win and the window rarely binds. The IDA step does not offer this
recomposition path to a supplied exact state partial when algebraic variables
participate in error control: those callbacks are already the cheap path, and
fresh values at each setup avoid excess Newton iterations from stale algebraic
couplings. Algebraic-suppressed solves retain bounded reuse."""

_DIRECT_PASS_FACTOR = 1e-4
"""First-iteration direct-acceptance factor on ``toldel``; the direct test is
``delnrm <= 1e-4 * toldel``."""

_SCHUR_EPS = 1e-12
"""Tikhonov regularization for the small Schur-complement solve over the global
block (matches the shared nonlinear layer's default)."""


class IDANlsContext(NamedTuple):
    """The per-attempt inputs the corrector solves against.

    Built once per step attempt at the API boundary and held constant across
    every retry of the attempt (in particular ``ewt``, refreshed only after an
    accepted step, never on a rejected retry).

    Attributes:
        t: The endpoint time of the step attempt (after any endpoint clamp).
        theta: The learnable-parameter pytree passed to the residual.
        cj: The leading coefficient of the iteration matrix for this attempt.
        yypredict: The predicted state at the endpoint, shape ``(n,)``.
        yppredict: The predicted derivative at the endpoint, shape ``(n,)``.
        ewt: The error-weight vector used for the Newton norm, shape ``(n,)``.
        control: The control-value pytree the residual reads, or
            ``None`` for a problem without a control channel.
    """

    t: Any
    theta: Any
    cj: jax.Array
    yypredict: jax.Array
    yppredict: jax.Array
    ewt: jax.Array
    prec_rung: Any = None
    rung_changed: Any = None
    control: Any = None


class IDANlsResult(NamedTuple):
    """The outcome of one fully-implicit corrector attempt.

    Attributes:
        ee: The converged Newton correction, shape ``(n,)``.
        yy: The corrected state ``yypredict + ee``, shape ``(n,)``.
        yp: The corrected derivative ``yppredict + cj*ee``, shape ``(n,)``.
        converged: Whether the convergence test was met.
        diverged: Whether the estimated rate exceeded the divergence threshold.
        iterations: The first-convergence iteration count (``int32``).
        delnrm: The last Newton-update weighted-RMS norm.
        rate: The last estimated convergence rate.
        ss: The rate memory carried to the next step.
        nsetups: Linear-solver setups performed this attempt (``int32``).
        nni: Newton iterations performed this attempt (``int32``).
        nnf: Newton convergence failures counted this attempt (``int32``).
        call_lsetup: The setup decision taken on the primary attempt.
        blocks_out: The ``(held, B, C, D)`` iteration matrix in effect at exit:
            the [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block, refactored from the assembled values when a
            setup (the gate or the retry) fired this attempt, the supplied
            ``frozen_blocks`` otherwise, and the global-coupling blocks. The
            caller persists these so the next attempt reuses the same factored
            matrix when its gate stays closed.
        saved_fydot_out: The derivative partial ``F_ydot`` of the matrix in
            effect at exit -- freshly evaluated when a setup fired, the carried
            ``frozen_fydot`` otherwise. With the matrix and its leading
            coefficient it defines the state Jacobian a later
            leading-coefficient-only setup reuses.
        nstlj_out: The step number of the most recent state-Jacobian
            evaluation (``int32``): this attempt's step number when the attempt
            evaluated ``F_y``, the incoming ``nstlj`` otherwise. The caller
            persists it beside the saved partials it describes.
        njac: State-Jacobian (``F_y``) evaluations performed this attempt
            (``int32``): one per fresh setup (the gate on a first step or after a
            failure, and the fresh-Jacobian retry), zero on a
            leading-coefficient-only setup that recomposes from the saved
            partials.
        border_cache_out: The carried border/Schur cache with every tier
            valid for the generations in effect at exit, or ``None`` when the
            caller supplied no cache.
        du_b_builds: Border multi-RHS solves performed this attempt
            (``int32``): one per stale-tier rebuild under a cache, one per
            corrector sweep without one, zero without bordered-global rows.
        schur_builds: Schur complements assembled (and factored) this
            attempt (``int32``), on the same accounting.
        lin_failures: Linear solves of the final sweep that missed their
            tolerance (``int32``) on an iterative (Krylov) route, else ``None``.
        krylov_iters: The largest iteration count of a linear solve of the
            attempt (``int32``) on an iterative route, else ``None``.
    """

    ee: jax.Array
    yy: jax.Array
    yp: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array
    delnrm: jax.Array
    rate: jax.Array
    ss: jax.Array
    nsetups: jax.Array
    nni: jax.Array
    nnf: jax.Array
    call_lsetup: jax.Array
    blocks_out: tuple
    saved_fydot_out: Any = None
    njac: Any = None
    nstlj_out: Any = None
    prec_stagnant: Any = None
    prec_false_conv: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    border_cache_out: Any = None
    du_b_builds: Any = None
    schur_builds: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def ida_residual(problem, t, y, ydot, theta, control=None) -> jax.Array:
    """The fully-implicit augmented residual ``F(t, y, ydot, theta, control)``.

    The problem states one residual. Algebraic global rows are its negated
    constraint equations; differential globals carry unit mass and therefore
    contribute their ``lam`` derivative.
    """
    return problem.residual(t, y, ydot, theta, control)


def ida_state_jacobian(problem, t, y, ydot, theta, cj, values_dtype=None, control=None):
    """The exact iteration-matrix blocks at ``(t, y, ydot)``.

    Returns ``(A_aug, B, C, D)``: the state block of
    ``A(cj) = dF/dy + cj*dF/dydot`` (dense, or values over the problem's
    ``state_topology``) and the dense bordered-global couplings, read from the
    problem's exact linearization plan. ``cj`` is applied exactly once inside
    the plan, and the coupling blocks arrive already in the solver's Schur
    convention for both algebraic and differential global rows.
    """
    return problem.state_jacobian_blocks(
        t, y, ydot, theta, cj, values_dtype=values_dtype, control=control
    )


def ida_nls_setup(nst, cj, cjlast, cjold, ss, refresh_jacobian, dcj=_CJ_WINDOW_HALFWIDTH):
    """The leading-coefficient setup gate and rate-memory selection.

    Returns ``(call_lsetup, ss, cjold, cjratio)`` for one step attempt.

    A setup is forced on the first step, whenever ``cjratio = cj/cjold`` leaves
    the window ``((1-dcj)/(1+dcj), (1+dcj)/(1-dcj))``, or when the caller's
    static policy requests an always-fresh factorization (``refresh_jacobian``).
    A leading-coefficient change without a setup bumps the rate memory to the
    conservative value. When a setup fires it resets the rate memory and the
    coefficient ratio: the returned ``ss`` and ``cjratio`` are already
    post-reset, so they feed the Newton loop and the ``2/(1+cjratio)`` rescale
    directly.
    """
    cj = jnp.asarray(cj)
    dtype = cj.dtype
    is_first = nst == 0
    cjold = jnp.where(is_first, cj, jnp.asarray(cjold, dtype=dtype))
    ss = jnp.where(is_first, jnp.asarray(_SS_FRESH, dtype=dtype), jnp.asarray(ss, dtype=dtype))

    cjratio = cj / cjold
    lower = jnp.asarray((1.0 - dcj) / (1.0 + dcj), dtype=dtype)
    upper = 1.0 / lower
    window_trip = (cjratio < lower) | (cjratio > upper)
    call_lsetup = jnp.logical_or(is_first, window_trip)
    call_lsetup = jnp.logical_or(call_lsetup, jnp.asarray(bool(refresh_jacobian)))

    # A cj change without a setup uses the conservative rate memory.
    ss = jnp.where(
        cj != jnp.asarray(cjlast, dtype=dtype), jnp.asarray(_SS_CJ_CHANGED, dtype=dtype), ss
    )
    # When a setup fires, the rate memory and the coefficient ratio reset.
    ss = jnp.where(call_lsetup, jnp.asarray(_SS_FRESH, dtype=dtype), ss)
    cjratio = jnp.where(call_lsetup, jnp.asarray(1.0, dtype=dtype), cjratio)
    cjold = jnp.where(call_lsetup, cj, cjold)
    return call_lsetup, ss, cjold, cjratio


class _NewtonCarry(NamedTuple):
    """The loop state threaded through one Newton sweep."""

    m: jax.Array  # completed iterations (0 on entry)
    ee: jax.Array
    oldnrm: jax.Array
    ss: jax.Array
    rate: jax.Array
    delnrm: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array  # first-convergence count (m+1)
    nni: jax.Array
    stag_count: Any = None
    resnrm_first: Any = None
    resnrm_last: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def ida_corrector(
    problem,
    linear,
    state,
    ctx: IDANlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_blocks: Optional[tuple] = None,
    *,
    frozen_fydot: Optional[jax.Array] = None,
    frozen_cj: Optional[jax.Array] = None,
    prior_failure: Optional[jax.Array] = None,
    nstlj: Optional[jax.Array] = None,
    generations: Optional[Generations] = None,
    border_cache=None,
    allow_partial_reuse: bool = True,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
    live: Optional[jax.Array] = None,
) -> IDANlsResult:
    """Solve ``F = 0`` for the fixed-leading-coefficient correction ``ee``.

    Runs the Newton corrector for one step attempt: the setup gate decides
    whether to factorize a fresh iteration matrix at the predictor or reuse the
    supplied ``frozen_blocks``; the Newton loop iterates the fixed-leading-
    coefficient update using the rate-based convergence test; and a single
    fresh-Jacobian retry recovers a recoverable failure taken with a stale
    Jacobian.

    The factored iteration matrix persists across attempts and steps: between
    setups the same matrix and its factorization are reused, and every setup
    refactors the held matrix in the storage the held factor names. When the
    setup gate fires only because the leading coefficient
    moved, the state Jacobian of the held matrix is reused and only
    ``F_ydot`` is re-evaluated to recompose ``A(cj)`` at the exact leading
    coefficient; a first step, a failure-triggered setup, and the
    fresh-Jacobian retry evaluate the state Jacobian fresh. The matrix and the
    derivative partial in effect at exit are returned on the result so the
    caller can carry them into the next attempt.

    Args:
        problem: The solver-contract problem (residual, Jacobian blocks, mass).
        linear: The state block's
            [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear]
            (its precision-rung routes).
        state: The carried step state; the setup-gate memory (``nst``,
            ``cjlast``, ``cjold``, ``ss``) is read from it.
        ctx: The per-attempt inputs (endpoint time, ``theta``, ``cj``,
            predictors, error weights, mass class).
        directive: The setup policy. Its static ``refresh_jacobian`` selects an
            always-fresh factorization (flash) versus the coefficient-window
            gate (stock); the traced setup decision executes it.
        loop_form: ``"fixed"`` for a bounded freeze-on-done ``fori_loop`` or
            ``"adaptive"`` for an early-exit ``while_loop``. Both use the same
            rate-based Newton convergence test.
        frozen_blocks: A supplied ``(held, B, C, D)`` to reuse when the gate
            does not fire: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the factored state block and the global-coupling blocks.
            ``None`` forces a fresh factorization every attempt (the
            standalone corrector path).
        frozen_fydot: The derivative partial ``F_ydot`` of the supplied matrix.
            When the setup gate fires only because the leading coefficient
            moved and no failure preceded this attempt, the matrix is
            recomposed at the step's exact leading coefficient from the
            supplied matrix's state Jacobian
            ``F_y = A(frozen_cj) - frozen_cj * frozen_fydot`` and a re-evaluated
            ``F_ydot``, so the state Jacobian is not re-evaluated. ``None``
            evaluates both partials at every setup.
        frozen_cj: The leading coefficient at which the supplied matrix was
            last assembled (by a setup the gate fired, or by the fresh-Jacobian
            retry).
        allow_partial_reuse: Whether a coefficient-only setup may recompose
            from the supplied matrix and ``frozen_fydot``. ``False`` keeps
            factored-matrix reuse on gate-closed attempts but evaluates both
            partials at every setup.
        prior_failure: Whether a convergence or error-test failure preceded this
            step attempt. A failure-triggered setup evaluates the state Jacobian
            fresh; only a failure-free leading-coefficient move recomposes from
            the saved partials. ``None`` is treated as no prior failure.
        nstlj: The step number at which the saved state Jacobian was last
            evaluated. A leading-coefficient-only setup recomposes from the
            saved state Jacobian only while ``nst < nstlj + MSBJ``; past that
            window it evaluates the state Jacobian fresh. ``None`` applies no
            window bound, for callers that supply no saved partials and so
            cannot reuse in the first place.
        generations: The carry's generation record at attempt entry, required
            alongside ``border_cache``. This engine reassembles every
            coupling block together with the state block on each setup, so
            the factor counter also names the coupling blocks and keys every
            cache tier.
        border_cache: The carried generation-keyed border/Schur cache for a
            problem with bordered-global rows: each setup stage revalidates
            it and rebuilds exactly the stale tiers, and every Newton
            iteration solves the bordered system by triangular solves
            against the held small factor. ``None`` rebuilds the border data
            per sweep and refactors the small system per iteration; the
            iterates are identical either way.
        newton_tol_factor: Corrector-band scale — widens the rate-based
            convergence bound and the one-iteration direct-acceptance floor
            by the same factor (``1.0`` = the standard band). The divergence
            rate test is a norm ratio and is unaffected.
        jacobian_values_dtype: Optional dtype for fresh Jacobian-values
            evaluations (``None`` = the state dtype).
        detect_precision: Arm the adaptive policy's failure detectors
            (increment stagnation, false convergence) and report them on the
            result; ``False`` leaves the compiled program unchanged.
        live: Whether the step loop keeps this attempt's result (per lane
            under ``vmap``), or ``None``. Under ``vmap``, the setup, Newton
            iteration and retry branches then run for the batch only when a
            live lane takes them; a live lane computes exactly what it
            computes with ``None``, and a lane that is not live may receive
            another branch's result. Unbatched it is not read.

    Returns:
        The [IDANlsResult]
        [diff_soup.timestepping.vsvo.engines.ida.nonlinear.IDANlsResult] for
        the attempt. Its ``blocks_out`` holds the
        factored iteration matrix in effect at exit (refactored where a setup
        fired, the carried ``frozen_blocks`` otherwise) for the caller to
        persist.
    """
    cj = ctx.cj
    dtype = ctx.yypredict.dtype

    # Adaptive precision: the rung switch runs over the route ladder, and the
    # solves carry the in-band escalation flag. A concrete reduced-precision
    # solve rides the same flagged channel so its fp64 fallbacks and
    # refinement steps reach the solver statistics.
    adaptive = linear.adaptive
    flagged = adaptive or linear.reduced
    # An iterative route (a Krylov backend) solves with the current matrix at
    # every attempt and a preconditioner rebuilt only where a setup fires;
    # each solve reports whether it met the linear tolerance.
    inexact = linear.inexact
    if inexact and int(problem.n_lam) > 0:
        raise ConfigurationError(
            "the iterative (Krylov) corrector solves the state block of a problem without "
            f"bordered-global rows; this problem has {int(problem.n_lam)}. Use an exact linear "
            "backend (cudss, klu, dense) for problems with global variables."
        )
    caching = border_cache is not None and int(problem.n_lam) > 0
    if border_cache is not None and generations is None:
        raise ValueError(
            "a border cache is generation-keyed: pass the carry's generations "
            "record alongside border_cache so the cache can be validated "
            "against the counters naming the carried factor and coupling "
            "blocks."
        )
    t = ctx.t
    theta = ctx.theta
    control = ctx.control
    ewt = ctx.ewt
    yypredict = ctx.yypredict
    yppredict = ctx.yppredict
    n = int(yypredict.shape[0])
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)

    conv_bound = NEWTON_CONV_COEFF
    direct_floor = _DIRECT_PASS_FACTOR * DIRECT_TEST_TOL
    if newton_tol_factor != 1.0:
        conv_bound = NEWTON_CONV_COEFF * float(newton_tol_factor)
        direct_floor = direct_floor * float(newton_tol_factor)
    epsnewt = jnp.asarray(conv_bound, dtype=dtype)
    direct_pass = jnp.asarray(direct_floor, dtype=dtype)
    ratemax = jnp.asarray(NEWTON_DIVERGENCE_RATE, dtype=dtype)

    # A gate-closed attempt reuses the supplied held matrix and coupling blocks
    # as they are. Without them there is nothing to reuse, so the attempt
    # factorizes fresh unconditionally (the standalone corrector path); the
    # flash policy forces a setup on every attempt through its directive.
    has_frozen = frozen_blocks is not None
    force_setup = bool(directive.refresh_jacobian) or not has_frozen
    call_lsetup, ss0, _cjold_out, cjratio = ida_nls_setup(
        state.nst, cj, state.cjlast, state.cjold, state.ss, refresh_jacobian=force_setup
    )
    step_scale = jnp.asarray(2.0, dtype=dtype) / (jnp.asarray(1.0, dtype=dtype) + cjratio)
    if inexact:
        # The operator is the matrix at the current cj: no stale-matrix rescale.
        step_scale = jnp.asarray(1.0, dtype=dtype)
        # The Newton test's norm and threshold set the linear tolerance.
        linear_test = dict(weights=ewt[:n_u], newton_tol=float(conv_bound))
    else:
        linear_test = {}
    topology = problem.state_topology
    a_shape = (n_u, n_u) if topology is None else (int(topology.rows.shape[0]),)
    if has_frozen:
        held_frozen, b_frozen, c_frozen, d_frozen = frozen_blocks
        a_frozen = held_frozen.values.reshape(a_shape)
    else:
        held_frozen = empty_held(linear, int(np.prod(a_shape)), dtype)

    # The linear-solver setup evaluates the iteration matrix at the predictor
    # iterate (ee = 0, so yy = yypredict and yp = yppredict). Assembly runs
    # only where a setup consumes it: with frozen blocks to reuse, the gate
    # branches to the assembly instead of selecting between two computed
    # sides, so a gate-closed attempt does zero Jacobian evaluations.
    #
    # A fresh setup evaluates both split partials and composes A(cj). When the
    # gate fires only because the leading coefficient moved -- no first step,
    # no failure preceding this attempt -- the state Jacobian of the held
    # matrix is reused: with F_y = A(cj_old) - cj_old * F_ydot_old,
    #     A(cj) = A(cj_old) + (cj * F_ydot - cj_old * F_ydot_old),
    # so only F_ydot is re-evaluated and the matrix is updated in place of a
    # fresh F_y evaluation.
    is_first = state.nst == 0
    if prior_failure is None:
        prior_failure_flag = jnp.asarray(False)
    else:
        prior_failure_flag = jnp.asarray(prior_failure, dtype=jnp.bool_)
    # The recompose fast path needs a saved derivative partial and applies only
    # to the window-gated policy (the flash policy and the standalone corrector
    # path force a fresh setup every attempt).
    can_recompose = (
        allow_partial_reuse
        and has_frozen
        and (frozen_fydot is not None)
        and (frozen_cj is not None)
        and not force_setup
    )
    # The saved state Jacobian may only be carried MSBJ steps: the setup gate
    # watches the leading coefficient and cannot see how far the state moved.
    if nstlj is None:
        within_window = jnp.asarray(True)
    else:
        reuse_window = problem.plan.max_fy_reuse_steps
        if reuse_window is None:
            reuse_window = MSBJ
        within_window = state.nst.astype(jnp.int32) < (
            jnp.asarray(nstlj, jnp.int32) + jnp.asarray(reuse_window, jnp.int32)
        )
    recompose_within = (
        jnp.logical_not(is_first) & jnp.logical_not(prior_failure_flag) & within_window
        if can_recompose
        else jnp.asarray(False)
    )
    one_i32_pack = jnp.asarray(1, jnp.int32)
    zero_i32_pack = jnp.asarray(0, jnp.int32)

    def _round(values):
        """Round matrix values through ``jacobian_values_dtype`` (the fresh assembly's rule)."""
        if jacobian_values_dtype is None:
            return values
        return values.astype(jacobian_values_dtype).astype(dtype)

    def _assemble_fresh_pack(_operand):
        fy, fydot = problem.state_partials(t, yypredict, yppredict, theta, None, control)
        blocks = problem.recomposed_state_blocks(fy, fydot, cj, values_dtype=jacobian_values_dtype)
        return blocks, fydot, one_i32_pack

    def _recompose_pack(_operand):
        fydot = problem.fydot_partial(t, yypredict, yppredict, theta, control=control)
        zero_fy = problem.zero_state_partials(dtype)[0]
        delta = problem.recomposed_state_blocks(zero_fy, cj * fydot - frozen_cj * frozen_fydot, 1.0)
        held_blocks = (a_frozen, b_frozen, c_frozen, d_frozen)
        blocks = tuple(_round(old + new) for old, new in zip(held_blocks, delta))
        return blocks, fydot, zero_i32_pack

    def _factored(pack):
        """The pack with its state block refactored into the held matrix (concrete modes)."""
        blocks, fydot, njac = pack
        return blocks, fydot, njac

    def _setup_pack(_operand):
        return _factored(
            live_lane_cond(
                jnp.logical_not(recompose_within),
                _assemble_fresh_pack,
                _recompose_pack,
                None,
                live=live,
            )
        )

    if not has_frozen or inexact:
        # Without blocks to reuse, and on an iterative route (whose operator is
        # the current matrix at every attempt, the setup gate deciding only the
        # preconditioner's rebuild), the attempt assembles the matrix fresh.
        blocks_use, fydot_use, njac_primary = _factored(_assemble_fresh_pack(None))
    else:
        fydot_reuse = (
            frozen_fydot if frozen_fydot is not None else problem.zero_state_partials(dtype)[1]
        )
        a_frozen_primary = a_frozen

        def _reuse_pack(_operand):
            return (a_frozen_primary, b_frozen, c_frozen, d_frozen), fydot_reuse, zero_i32_pack

        blocks_use, fydot_use, njac_primary = live_lane_cond(
            call_lsetup, _setup_pack, _reuse_pack, None, live=live, false_is_free=True
        )
    a_use, b_use, c_use, d_use = blocks_use

    def _linearize(held, a_block, b_block, c_block, d_block, refresh, cache, factor_key):
        """Refactor the held state block where it was replaced; make the border/Schur data current.

        ``refresh`` is true exactly when ``a_block`` carries values replaced
        this attempt: the held factor is refactored from them, and otherwise
        the solves use the factor already held. With a cache, ``factor_key`` is the factor
        generation in effect for this stage —
        in this engine every setup reassembles the coupling blocks with the
        state block, so it keys every cache tier. Without one, the border
        solve and Schur complement are rebuilt here and the small system is
        refactored on every iteration.
        """
        rung = ctx.prec_rung if adaptive else None
        held, solve, solve_many = linearize(
            held,
            a_block,
            refresh,
            linear=linear,
            rung=rung,
            rung_changed=ctx.rung_changed if adaptive else None,
            flagged=flagged,
            **linear_test,
        )
        if n_lam > 0:
            if cache is not None:
                cache, du_b_built, schur_built = refresh_border_cache(
                    cache,
                    factor_generation=factor_key,
                    b_generation=factor_key,
                    c_generation=factor_key,
                    d_generation=factor_key,
                    rung=ctx.prec_rung if adaptive else None,
                    solve_many=solve_many,
                    b_block=b_block,
                    c_block=c_block,
                    d_block=d_block,
                    regularization=_SCHUR_EPS,
                )
                du_b = cache.du_b

                def schur_solve(rhs, _cache=cache):
                    return border_cache_solve(_cache, rhs)
            else:
                du_b = solve_many(b_block.T).T
                schur = d_block - c_block @ du_b
                du_b_built = schur_built = jnp.asarray(1, jnp.int32)

                def schur_solve(rhs, _schur=schur):
                    return _stable_small_solve(_schur, rhs, _SCHUR_EPS)
        else:
            du_b = jnp.zeros((n_u, 0), dtype=dtype)
            du_b_built = schur_built = jnp.asarray(0, jnp.int32)

            def schur_solve(rhs):
                return rhs

        return held, solve, du_b, schur_solve, cache, du_b_built, schur_built

    def _run_loop(solve, du_b, c_block, schur_solve, ss_init, scale):
        """One Newton sweep with the rate-based convergence test."""

        def newton_delta(ee):
            yy = yypredict + ee
            yp = yppredict + cj * ee
            resid = ida_residual(problem, t, yy, yp, theta, control)
            r = resid[:n_u]
            g = resid[n_u:]
            if flagged or inexact:
                du_r, lflag, lsteps = solve(-r)
            else:
                du_r = solve(-r)
                lflag = lsteps = None
            if n_lam > 0:
                dlam = schur_solve(-g - c_block @ du_r)
                du = du_r - du_b @ dlam
            else:
                dlam = jnp.zeros((0,), dtype=dtype)
                du = du_r
            resnrm = jnp.linalg.norm(resid) if detect_precision else None
            return jnp.concatenate([du, dlam]) * scale, resnrm, lflag, lsteps

        def take_step(carry: _NewtonCarry) -> _NewtonCarry:
            delta, resnrm, lflag, lsteps = newton_delta(carry.ee)
            ee_new = carry.ee + delta
            dn = wrms_norm(delta, ewt, mask=None)
            is_first = carry.m == 0
            oldnrm_new = jnp.where(is_first, dn, carry.oldnrm)
            m_denom = jnp.maximum(carry.m, 1).astype(dtype)
            # ``safe_div``/``safe_power``: on a trajectory the formula
            # integrates exactly the Newton updates are exactly zero, so the
            # unguarded 0/0 and the fractional power's infinite slope at a
            # zero base — reached only in the frozen/discarded lanes of the
            # bounded corrector — would poison reverse-mode cotangents of the
            # live update.
            rate_actual = safe_power(safe_div(dn, oldnrm_new), 1.0 / m_denom)
            diverged_now = jnp.logical_and(jnp.logical_not(is_first), rate_actual > ratemax)
            # ss = rate/(1-rate) is written only on a non-diverging m >= 1 step;
            # a diverging step keeps the carried memory. The clamp only guards
            # the unused diverging branch.
            ss_updated = rate_actual / jnp.maximum(
                1.0 - rate_actual, jnp.asarray(1e-12, dtype=dtype)
            )
            ss_step = jnp.where(diverged_now, carry.ss, ss_updated)
            ss_new = jnp.where(is_first, carry.ss, ss_step)
            rate_new = jnp.where(is_first, jnp.asarray(0.0, dtype=dtype), rate_actual)
            direct_conv = jnp.logical_and(is_first, dn <= direct_pass)
            tol_conv = (ss_new * dn) <= epsnewt
            converged_now = jnp.logical_or(
                direct_conv, jnp.logical_and(jnp.logical_not(diverged_now), tol_conv)
            )
            if inexact:
                # A linear solve that missed its tolerance ends the sweep as a
                # failure: the caller rebuilds a stale preconditioner and retries.
                missed = lflag > 0
                converged_now = converged_now & jnp.logical_not(missed)
                diverged_now = diverged_now | missed
                lin_failures = carry.lin_failures + missed.astype(jnp.int32)
                krylov_iters = jnp.maximum(carry.krylov_iters, lsteps.astype(jnp.int32))
            else:
                lin_failures = krylov_iters = None
            m_next = carry.m + 1
            iters_new = jnp.where(converged_now, m_next, carry.iterations)
            if detect_precision:
                stag = stagnation_flag(dn, carry.delnrm, converged_now)
                stag_count = jnp.where(stag, carry.stag_count + 1, 0).astype(jnp.int32)
                resnrm_first = jnp.where(carry.m == 0, resnrm, carry.resnrm_first)
                resnrm_last = resnrm
            else:
                stag_count = resnrm_first = resnrm_last = None
            if inexact:
                lin_fallbacks = lin_ir_steps = None
            else:
                lin_fallbacks = None if lflag is None else carry.lin_fallbacks + lflag
                lin_ir_steps = None if lsteps is None else carry.lin_ir_steps + lsteps
            return _NewtonCarry(
                m=m_next,
                ee=ee_new,
                oldnrm=oldnrm_new,
                ss=ss_new,
                rate=rate_new,
                delnrm=dn,
                converged=converged_now,
                diverged=diverged_now,
                iterations=iters_new,
                nni=carry.nni + 1,
                stag_count=stag_count,
                resnrm_first=resnrm_first,
                resnrm_last=resnrm_last,
                lin_fallbacks=lin_fallbacks,
                lin_ir_steps=lin_ir_steps,
                lin_failures=lin_failures,
                krylov_iters=krylov_iters,
            )

        init = _NewtonCarry(
            m=jnp.asarray(0, dtype=jnp.int32),
            ee=jnp.zeros(n, dtype=dtype),
            oldnrm=jnp.asarray(jnp.inf, dtype=dtype),
            ss=jnp.asarray(ss_init, dtype=dtype),
            rate=jnp.asarray(0.0, dtype=dtype),
            delnrm=jnp.asarray(jnp.inf, dtype=dtype),
            converged=jnp.asarray(False),
            diverged=jnp.asarray(False),
            iterations=jnp.asarray(0, dtype=jnp.int32),
            nni=jnp.asarray(0, dtype=jnp.int32),
            stag_count=jnp.asarray(0, jnp.int32) if detect_precision else None,
            resnrm_first=jnp.asarray(0.0, dtype=dtype) if detect_precision else None,
            resnrm_last=jnp.asarray(0.0, dtype=dtype) if detect_precision else None,
            lin_fallbacks=jnp.zeros((), jnp.float64) if flagged else None,
            lin_ir_steps=jnp.zeros((), jnp.float64) if flagged else None,
            lin_failures=jnp.asarray(0, jnp.int32) if inexact else None,
            krylov_iters=jnp.asarray(0, jnp.int32) if inexact else None,
        )

        def done(carry: _NewtonCarry) -> jax.Array:
            return jnp.logical_or(carry.converged, carry.diverged)

        if loop_form == "fixed":

            def fori_body(_i, carry):
                return live_lane_cond(
                    jnp.logical_not(done(carry)),
                    take_step,
                    lambda c: c,
                    carry,
                    live=live,
                    false_is_free=True,
                )

            return jax.lax.fori_loop(0, MAX_NEWTON_ITERS, fori_body, init)
        if loop_form == "adaptive":

            def cond(carry: _NewtonCarry) -> jax.Array:
                return jnp.logical_and(
                    jnp.logical_not(done(carry)),
                    carry.m < jnp.asarray(MAX_NEWTON_ITERS, dtype=carry.m.dtype),
                )

            return jax.lax.while_loop(cond, take_step, init)
        raise ValueError(f"loop_form must be 'fixed' or 'adaptive', got {loop_form!r}.")

    # Generation counters in effect for each stage: the primary setup (when
    # the gate fires) replaces the factor and every coupling block BEFORE the
    # sweep consumes them, and the retry — which runs only when the gate
    # stayed closed — replaces them once more. The caller advances the
    # carried record by exactly the setup count this attempt reports, so the
    # keys stored in the cache equal the record the next attempt validates
    # against.
    zero_key = jnp.asarray(0, jnp.int32)
    if caching:
        factor_key0 = generations.factor + call_lsetup.astype(jnp.int32)
        factor_key_retry = factor_key0 + jnp.asarray(1, jnp.int32)
    else:
        factor_key0 = factor_key_retry = zero_key
    cache_in = border_cache if caching else None

    held1, solve0, du_b0, schur_solve0, cache1, dub0, sb0 = _linearize(
        held_frozen,
        a_use,
        b_use,
        c_use,
        d_use,
        call_lsetup,
        cache_in,
        factor_key0,
    )
    carry1 = _run_loop(solve0, du_b0, c_use, schur_solve0, ss0, step_scale)

    # A recoverable failure taken with a stale Jacobian retries once with a
    # fresh factorization from the predictor.
    # The retry assembles its own fresh blocks inside the branch — at the same
    # predictor point, so the values are identical to a primary-setup assembly
    # — and returns them so the caller persists the matrix actually in effect.
    need_retry = jnp.logical_and(jnp.logical_not(carry1.converged), jnp.logical_not(call_lsetup))

    def _retry(_operand):
        if inexact:
            # The matrix is already current: the retry only rebuilds the
            # preconditioner from it.
            (a_r, b_r, c_r, d_r), fydot_r = blocks_use, fydot_use
        else:
            (a_r, b_r, c_r, d_r), fydot_r, _njac_r = _assemble_fresh_pack(None)
        # ``need_retry`` holds wherever this branch's result is kept; a lane of
        # a batch that runs the branch without needing it keeps its held
        # factor.
        held_r, solve_r, du_b_r, schur_solve_r, cache_r, dub_r, sb_r = _linearize(
            held1, a_r, b_r, c_r, d_r, need_retry, cache1, factor_key_retry
        )
        carry_r = _run_loop(
            solve_r,
            du_b_r,
            c_r,
            schur_solve_r,
            jnp.asarray(_SS_FRESH, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        )
        return carry_r, (held_r, b_r, c_r, d_r), fydot_r, cache_r, dub_r, sb_r

    retried = live_lane_cond(
        need_retry,
        _retry,
        lambda _op: (
            carry1,
            (held1, b_use, c_use, d_use),
            fydot_use,
            cache1,
            zero_key,
            zero_key,
        ),
        None,
        live=live,
        false_is_free=True,
    )
    carry2, blocks_after_retry, fydot_after_retry, cache_out, dub_retry, sb_retry = retried
    du_b_builds = dub0 + dub_retry
    schur_builds = sb0 + sb_retry

    converged = jnp.where(need_retry, carry2.converged, carry1.converged)
    diverged = jnp.where(need_retry, carry2.diverged, carry1.diverged)
    ee = jnp.where(need_retry, carry2.ee, carry1.ee)
    iterations = jnp.where(need_retry, carry2.iterations, carry1.iterations)
    delnrm = jnp.where(need_retry, carry2.delnrm, carry1.delnrm)
    rate = jnp.where(need_retry, carry2.rate, carry1.rate)
    ss_out = jnp.where(need_retry, carry2.ss, carry1.ss)

    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    one_i32 = jnp.asarray(1, dtype=jnp.int32)
    nni = carry1.nni + jnp.where(need_retry, carry2.nni, zero_i32)
    nsetups = call_lsetup.astype(jnp.int32) + jnp.where(need_retry, one_i32, zero_i32)
    # Newton convergence-failure count: 0 when the primary attempt converges;
    # 1 when it fails with a fresh Jacobian (no retry possible) or when the
    # retry recovers; 2 when the retry also fails.
    nnf = jnp.where(
        need_retry,
        jnp.where(carry2.converged, one_i32, jnp.asarray(2, dtype=jnp.int32)),
        jnp.where(carry1.converged, zero_i32, one_i32),
    )

    # The factored iteration matrix in effect at exit: refactored from the
    # assembled blocks when a setup fired (the gate on the primary attempt, or
    # the fresh-Jacobian retry), else the carried matrix left untouched —
    # already threaded through the branch results above. The caller persists
    # it so the next attempt reuses the same factor while its gate stays
    # closed.
    blocks_out = blocks_after_retry
    saved_fydot_out = fydot_after_retry
    # State-Jacobian evaluations this attempt: the primary setup evaluates F_y
    # only on a fresh setup (a first step, a failure-triggered setup, or a
    # standalone corrector), never on a leading-coefficient-only recompose; the
    # fresh-Jacobian retry always evaluates it.
    njac = njac_primary + jnp.where(need_retry, one_i32, zero_i32)
    if inexact:
        # The matrix is assembled once per attempt (a retry reuses it).
        njac = njac_primary

    # A fresh state-Jacobian evaluation this attempt -- the primary setup or the
    # fresh-Jacobian retry -- restarts the reuse window.
    nstlj_in = jnp.asarray(0, jnp.int32) if nstlj is None else jnp.asarray(nstlj, jnp.int32)
    nstlj_out = jnp.where(njac > 0, state.nst.astype(jnp.int32), nstlj_in)

    yy = yypredict + ee
    yp = yppredict + cj * ee
    if detect_precision:
        prec_stagnant = (carry2.stag_count >= STAGNATION_LATCH) & jnp.logical_not(converged)
        prec_false_conv = (
            converged & (carry2.iterations >= 2) & (carry2.resnrm_last >= carry2.resnrm_first)
        )
    else:
        prec_stagnant = prec_false_conv = None

    return IDANlsResult(
        ee=ee,
        yy=yy,
        yp=yp,
        converged=converged,
        diverged=diverged,
        iterations=iterations,
        delnrm=delnrm,
        rate=rate,
        ss=ss_out,
        nsetups=nsetups,
        nni=nni,
        nnf=nnf,
        call_lsetup=call_lsetup,
        blocks_out=blocks_out,
        saved_fydot_out=saved_fydot_out,
        njac=njac,
        nstlj_out=nstlj_out,
        prec_stagnant=prec_stagnant,
        prec_false_conv=prec_false_conv,
        lin_fallbacks=carry2.lin_fallbacks,
        lin_ir_steps=carry2.lin_ir_steps,
        border_cache_out=cache_out,
        du_b_builds=du_b_builds,
        schur_builds=schur_builds,
        lin_failures=jnp.where(need_retry, carry2.lin_failures, carry1.lin_failures)
        if inexact
        else None,
        krylov_iters=jnp.maximum(carry1.krylov_iters, carry2.krylov_iters) if inexact else None,
    )


def ida_corrector_root(
    problem,
    backend,
    linear,
    sensitivity_analysis,
    state,
    ctx: IDANlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_blocks: Optional[tuple] = None,
    *,
    frozen_fydot: Optional[jax.Array] = None,
    frozen_cj: Optional[jax.Array] = None,
    prior_failure: Optional[jax.Array] = None,
    nstlj: Optional[jax.Array] = None,
    reverse_analysis=None,
    generations: Optional[Generations] = None,
    border_cache=None,
    allow_partial_reuse: bool = True,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
    live: Optional[jax.Array] = None,
) -> IDANlsResult:
    """One corrector attempt with the accepted root behind its derivative boundary.

    Runs [ida_corrector]
    [diff_soup.timestepping.vsvo.engines.ida.nonlinear.ida_corrector] as the
    primal solve and binds the returned root
    to the implicit-function rule at that root: the accepted step solves
    ``F(t, y, cj*y + q, theta) = 0`` with the history aggregate
    ``q = yppredict - cj*yypredict``, so the corrected state, correction, and
    derivative carry the exact root derivative while every primal bookkeeping
    output (convergence flags, counters, rate memory, setup decisions, carried
    matrix blocks) stays gradient-dead. ``linear`` is the state block's
    [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear],
    ``sensitivity_analysis`` the ``backend``'s forward-direction analysis of
    the problem's full-system pattern (``None`` on the dense path), which
    serves the tangent solve, and ``reverse_analysis`` its reverse-direction
    analysis, which serves the transposed solve of a reverse sweep (``None``
    reuses ``sensitivity_analysis``).

    ``generations`` / ``border_cache`` carry the generation-keyed
    border/Schur cache into the primal solve (see
    [ida_corrector][diff_soup.timestepping.vsvo.engines.ida.nonlinear.ida_corrector]);
    the cache never reaches the sensitivity rules, which rebuild exact
    values at the accepted root. ``frozen_fydot``, ``frozen_cj`` and
    ``prior_failure`` thread the coefficient-only fast path into the primal
    solve: they select
    whether a setup recomposes the iteration matrix from the saved state
    Jacobian or evaluates it fresh, and stay gradient-dead like every other
    primal bookkeeping input. ``live`` (the step loop's condition for this
    lane, or ``None``) reaches the primal solve the same way, so every
    derivative rule that runs the corrector again counts the same lanes.
    """

    def corrector_at(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector at the reconstructed predictors (never differentiated).

        Every traced solver value (setup-gate memory, error weights, carried
        matrix blocks, precision-rung state) arrives through ``primal_args``
        so this closure captures only static configuration.
        """
        (
            p_state,
            ewt,
            p_blocks,
            p_generations,
            p_border_cache,
            prec_rung,
            rung_changed,
            p_fydot,
            p_prior_failure,
            p_nstlj,
            p_cj,
            p_live,
        ) = primal_args
        ctx_p = IDANlsContext(
            t=t,
            theta=theta,
            cj=cj,
            yypredict=guess,
            yppredict=cj * guess + q,
            ewt=ewt,
            prec_rung=prec_rung,
            rung_changed=rung_changed,
            control=control,
        )
        return ida_corrector(
            problem,
            linear,
            p_state,
            ctx_p,
            directive,
            loop_form,
            frozen_blocks=p_blocks,
            frozen_fydot=p_fydot,
            frozen_cj=p_cj,
            prior_failure=p_prior_failure,
            nstlj=p_nstlj,
            generations=p_generations,
            border_cache=p_border_cache,
            allow_partial_reuse=allow_partial_reuse,
            newton_tol_factor=newton_tol_factor,
            jacobian_values_dtype=jacobian_values_dtype,
            detect_precision=detect_precision,
            live=p_live,
        )

    def primal_fn(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector and expose its converged root to the derivative rule."""
        res = corrector_at(t, cj, q, theta, control, guess, primal_args)
        return res.yy, res.converged, res

    q = ctx.yppredict - ctx.cj * ctx.yypredict
    primal_args = (
        state,
        ctx.ewt,
        frozen_blocks,
        generations,
        border_cache,
        ctx.prec_rung,
        ctx.rung_changed,
        frozen_fydot,
        prior_failure,
        nstlj,
        frozen_cj,
        live,
    )

    def offering_extractor(aux, t, cj, theta, control, primal_args):
        """Build the iteration-matrix offering from the corrector result.

        Under the strict at-root predicate, evaluates the split partials at
        the accepted root and recomposes ``A(cj) = F_y + cj * F_ydot`` at the
        step's leading coefficient. A failed predicate executes no partials
        sweep. The offering is assembled inside the
        derivative rule from the ``aux`` the primal already returned, so no
        second corrector run is needed to source it. It supplies only the
        sensitivity solve's matrix VALUE for a first derivative — the matrix's
        own tangent always rebuilds at the frozen root, keeping higher-order
        derivatives exact.
        """
        prec_rung = primal_args[5]
        root_ok = _ida_root_success(aux, prec_rung, detect_precision)
        zeros = problem.zero_state_partials(aux.yy.dtype)
        fy_saved, fydot_saved = jax.lax.cond(
            root_ok,
            lambda _op: problem.state_partials(t, aux.yy, aux.yp, theta, control=control),
            lambda _op: zeros,
            None,
        )
        return problem.iteration_matrix_offering(
            fy_saved,
            fydot_saved,
            cj,
            root_ok,
        )

    yy, _ok, aux = problem.bind_step_root(
        backend,
        sensitivity_analysis,
        primal_fn,
        ctx.t,
        ctx.cj,
        q,
        ctx.theta,
        ctx.control,
        ctx.yypredict,
        primal_args,
        reverse_analysis=reverse_analysis,
        certificate_extractor=offering_extractor,
    )
    ee = yy - ctx.yypredict
    yp = ctx.yppredict + ctx.cj * ee
    return aux._replace(ee=ee, yy=yy, yp=yp)


def _ida_root_success(res: IDANlsResult, prec_rung, detect_precision: bool) -> jax.Array:
    """The exact-at-root predicate that gates the iteration-matrix offering.

    The offered values may be consumed only when they were evaluated exactly
    at the accepted root. A setup may now recompose from saved ``F_y``, so
    issuance requires the corrector to have converged (minus a reduced-
    precision false convergence when the detectors are armed), to have
    evaluated fresh state partials this attempt, and to have applied an
    exactly zero correction. Under-issuance is always safe — a rejected
    offering rebuilds the exact matrix — while over-issuance would let the
    sensitivity solve consume values assembled away from the root.
    ``prec_rung`` is read only when the precision detectors are armed.
    """
    at_root = jnp.all(res.ee == 0.0)
    fresh_partials = res.njac > 0
    root_ok = jnp.asarray(res.converged) & fresh_partials & at_root
    if detect_precision:
        reduced = prec_rung < TOP_RUNG
        root_ok = jnp.logical_and(
            root_ok, jnp.logical_not(jnp.asarray(res.prec_false_conv) & reduced)
        )
    return root_ok


__all__ = [
    "IDANlsContext",
    "IDANlsResult",
    "ida_residual",
    "ida_state_jacobian",
    "ida_nls_setup",
    "ida_corrector",
    "ida_corrector_root",
    "MAX_NEWTON_ITERS",
    "MSBJ",
    "NEWTON_DIVERGENCE_RATE",
]
