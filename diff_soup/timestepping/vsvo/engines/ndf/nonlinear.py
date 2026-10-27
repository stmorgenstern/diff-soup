"""The correction-form Newton corrector for the NDF engine.

The corrector equation and its convergence law follow the QNDF solver of
OrdinaryDiffEq.jl and its shared simplified-Newton layer (MIT license; see
``THIRD_PARTY_NOTICES.md``). With predictor ``u0 = sum_j D[j]`` and history
term ``phi = sum_i gamma_i * D[i]``, the corrected state ``y = u0 + dd``
satisfies the NDF corrector equation

    M(t, y) @ (dd / beta0 + phi) / dt = f(t, y)

which this module solves in the scaled correction form

    res(dd) = M(t, y) * (dd * invgdt + phi/dt) - f(t, y),   invgdt = 1/(dt*beta0).

The exact Newton matrix of this residual is ``invgdt*M - J_f`` — composed
here as ``invgdt * (M - gdt*J_f)`` from the raw Jacobian the problem
supplies, with the bordered-global blocks negated into the same Schur
convention the sibling engines use. Rows with zero mass reduce to pure
algebraic roots (``-f_row``), which is what makes constant-singular-mass
index-1 systems integrable in this form. A state-dependent diagonal mass is
re-evaluated at the current iterate inside the residual and handled in weak
form: the pencil omits the ``dM/du`` term, a chord-Newton choice that
leaves the converged solution exact and only affects the iteration count.

Convergence follows the simplified-Newton rate law: with update norm
``ndz = wrms(delta, ewt) * error_const(k)`` (the NDF/BDF truncation error is
proportional to the total displacement, so the test norm carries the
local-error constant), the contraction estimate is

    theta_1 = theta_carried;   theta_m = max(0.3*theta_{m-1}, ndz_m/ndz_{m-1})

with ``theta`` carried across attempts and steps (reset to 1 after a
rejected step). The step converges when ``eta * ndz < kappa`` for
``eta = theta/(1 - theta)`` (or immediately when the first update is below
a small floor), diverges when ``theta > 2``, and a near-unit ``theta``
resolves by the raw ``ndz``. A diverged or unconverged sweep taken with a
stale Jacobian re-enters once with a fresh Jacobian and factorization; a
fresh-Jacobian failure ends the attempt (the step loop halves the step).

The two-tier reuse policy (a setup gate deciding whether to factor at all,
and a Jacobian gate deciding whether to re-evaluate ``J_f`` within a setup)
uses drift and step-count windows in the spirit of the reference's
W-reuse cutoffs; the window constants are this engine's own resource-bound
choices, documented on each constant.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.nonlinear.border_cache import border_cache_solve, refresh_border_cache
from diff_soup.nonlinear.generations import Generations, certify_exact_root
from diff_soup.nonlinear.result import LinearizationDirective
from diff_soup.nonlinear.solver import _stable_small_solve
from diff_soup.timestepping.vsvo.common.adaptive import (
    STAGNATION_LATCH,
    TOP_RUNG,
    stagnation_flag,
)
from diff_soup.timestepping.vsvo.common.contract import fy_reuse_window
from diff_soup.timestepping.vsvo.common.linear_ops import (
    empty_held,
    linearize,
)
from diff_soup.timestepping.vsvo.common.weights import wrms_norm
from diff_soup.utils import safe_div
from diff_soup.utils.lanes import lane_cond

# --- corrector constants ---
NEWTON_MAXITERS = 10
"""Maximum Newton iterations per sweep (the reference solver's default)."""

NEWTON_KAPPA = 0.01
"""Convergence-test tolerance: accept when ``eta * ndz < kappa``."""

RATE_LOWPASS = 0.3
"""Contraction-estimate low-pass: ``theta = max(0.3*theta_prev, ratio)``."""

DIVERGENCE_THETA = 2.0
"""Contraction estimate above which the sweep is declared divergent."""

FIRST_ITER_FAST_TOL = 1e-5
"""A first update below this floor converges immediately."""

W_DRIFT_CUTOFF = 0.2
"""Leading-coefficient drift ``|gdt/gdt_prev - 1|`` beyond which the setup
gate refactors (an own resource-bound choice in the spirit of the
reference's dt*gamma reuse cutoff)."""

MSBP = 20
"""Maximum steps between factorizations (own resource-bound choice)."""

MSBJ = 50
"""Default maximum steps between Jacobian evaluations.

A problem's ``max_fy_reuse_steps`` declaration may reduce this resource
window but never enlarge it.
"""

_SCHUR_EPS = 1e-12
"""Tikhonov regularization for the small Schur solve over the global block."""


class NDFNlsContext(NamedTuple):
    """The per-attempt inputs the corrector solves against.

    Attributes:
        t: The endpoint time of the step attempt.
        theta: The learnable-parameter pytree passed to the residual.
        invgdt: The pencil scalar ``1/(dt*beta0(k))``.
        gdt: ``dt*beta0(k)`` (the pencil composes ``M - gdt*J_f``).
        phi_dt: The history term ``phi/dt``, shape ``(n,)``.
        pred_y: The predictor ``u0 = sum_j D[j]``, shape ``(n,)``.
        ewt: The error-weight vector for the Newton norm, shape ``(n,)``.
        err_const_k: The order-``k`` local-error constant scaling the norm.
        theta_mem: The contraction estimate carried into this attempt.
        first_call: Whether this is the first attempt of a step.
        prev_conv_fail: Whether the prior attempt failed to converge.
        prev_err_fail: Whether the prior attempt failed the error test.
        prec_rung: The adaptive linear-precision rung active this attempt
            (traced int32), or ``None`` for concrete solve modes.
        rung_changed: Whether the rung changed on the previous attempt (the
            newly active rung's held factor must be rebuilt), or ``None``.
        control: The control-value pytree the residual reads, or
            ``None`` for a problem without a control channel.
    """

    t: Any
    theta: Any
    invgdt: jax.Array
    gdt: jax.Array
    phi_dt: jax.Array
    pred_y: jax.Array
    ewt: jax.Array
    err_const_k: jax.Array
    theta_mem: jax.Array
    first_call: jax.Array
    prev_conv_fail: jax.Array
    prev_err_fail: jax.Array
    prec_rung: Any = None
    rung_changed: Any = None
    control: Any = None


class ConvergenceUpdate(NamedTuple):
    """One convergence-law evaluation."""

    theta_next: jax.Array
    eta: jax.Array
    converged: jax.Array
    diverged: jax.Array


class NDFNlsResult(NamedTuple):
    """The outcome of one corrector attempt.

    Attributes:
        dd: The converged correction, shape ``(n,)``.
        y: The corrected state ``pred_y + dd``.
        converged: Whether the convergence test was met.
        diverged: Whether the contraction estimate declared divergence.
        iterations: The first-convergence iteration count (``int32``).
        ndz_last: The last update norm (for the trace).
        theta_out: The contraction memory to carry forward.
        call_setup: The setup-gate decision on the primary sweep.
        jbad: The Jacobian-recompute decision on the primary setup.
        jcur: Whether the Jacobian is current after this attempt.
        nsetups: Factorizations performed this attempt (``int32``).
        nje: Jacobian evaluations performed this attempt (``int32``).
        nni: Newton iterations performed this attempt (``int32``).
        nnf: Convergence failures counted this attempt (``int32``).
        gdt_prev_out: The post-setup leading-coefficient memory.
        nstlp_out: The step number of the most recent setup (``int32``).
        nstlj_out: The step number of the most recent Jacobian (``int32``).
        saved_jac_out: The ``(J_f, B, C, D)`` blocks in effect at exit.
        pencil_out: The ``(held, B, C, D)`` iteration matrix at exit: the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the factored state block and the global-coupling blocks.
        prec_stagnant: Whether the precision detectors latched increment
            stagnation on a failed sweep, or ``None`` when detection is off.
        prec_false_conv: Whether the sweep converged by increment while its
            residual never dropped below the first iterate's, or ``None``.
        lin_fallbacks: Accumulated linear-refinement fp64 fallbacks observed
            by the final sweep (fp64 scalar), or ``None``.
        lin_ir_steps: Accumulated mixed-precision refinement steps observed
            by the final sweep's linear solves (fp64 scalar), or ``None``.
        generations_out: The carry's generation record advanced by this
            attempt's rebuild events, or ``None`` when the caller supplied no
            record ([ndf_corrector_root][] only).
        certificate_out: The exact-root certificate issued at this attempt's
            root-acceptance decision, or ``None`` when the caller supplied no
            generation record ([ndf_corrector_root][] only).
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
            final sweep (``int32``) on an iterative route, else ``None``.
    """

    dd: jax.Array
    y: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array
    ndz_last: jax.Array
    theta_out: jax.Array
    call_setup: jax.Array
    jbad: jax.Array
    jcur: jax.Array
    nsetups: jax.Array
    nje: jax.Array
    nni: jax.Array
    nnf: jax.Array
    gdt_prev_out: jax.Array
    nstlp_out: jax.Array
    nstlj_out: jax.Array
    saved_jac_out: tuple
    pencil_out: tuple
    prec_stagnant: Any = None
    prec_false_conv: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    generations_out: Any = None
    certificate_out: Any = None
    border_cache_out: Any = None
    du_b_builds: Any = None
    schur_builds: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def ndf_residual(problem, ctx: NDFNlsContext, dd: jax.Array) -> jax.Array:
    """The correction-form residual through the solver problem.

    The method's discrete slope is affine in the correction,
    ``ydot = dd*invgdt + phi/dt``, so the correction root is exactly the
    residual ``F(t, pred + dd, ydot, theta)`` — the derivative
    operator is evaluated at the current iterate, and rows without
    derivative support reduce to algebraic roots.
    """
    y = ctx.pred_y + dd
    return problem.residual(ctx.t, y, dd * ctx.invgdt + ctx.phi_dt, ctx.theta, ctx.control)


def ndf_state_jacobian(problem, ctx: NDFNlsContext, values_dtype=None):
    """Assemble the fresh exact iteration matrix ``A(invgdt)`` at the predictor.

    Returns ``(A_aug, partials, B, C, D)``: the engine blocks of
    ``A(cj) = dF/dy + cj*dF/dydot`` with ``cj = invgdt`` read from the
    problem's linearization plan (the coupling blocks arrive already in the
    Schur convention), and the split partial values carried for a later
    coefficient-only recomposition. The predictor derivative is zero, so the
    state-action term of a state-dependent mass operator vanishes from
    the matrix exactly as this corrector's chord-Newton form requires.
    """
    ydot_pred = jnp.zeros_like(ctx.pred_y)
    fy, fydot = problem.state_partials(
        ctx.t, ctx.pred_y, ydot_pred, ctx.theta, values_dtype=values_dtype, control=ctx.control
    )
    a_aug, b, c, d = problem.recomposed_state_blocks(fy, fydot, ctx.invgdt)
    return a_aug, (fy, fydot), b, c, d


def _rebuild_pencil(problem, ctx: NDFNlsContext, partials_saved: tuple) -> jax.Array:
    """Recompose the iteration matrix from saved partials (no Jacobian evaluation).

    The stale ``F_y`` is kept; ``F_ydot`` alone is re-evaluated at the
    current predictor (the mass-operator coefficients — cheap), and the
    new leading coefficient recombines them.
    """
    return _rebuild_pencil_blocks(problem, ctx, partials_saved)[0]


def _rebuild_pencil_blocks(problem, ctx: NDFNlsContext, partials_saved: tuple):
    """Recompose all four engine blocks from saved partials at the current cj.

    A differential-global border makes ``D`` (and in general the coupling
    blocks) depend on the leading coefficient through ``cj * M_global``, so a
    setup that reuses the saved Jacobian must still recombine the coupling
    blocks at the current ``cj`` rather than keep the ones saved at an older
    leading coefficient.
    """
    fy_saved, _ = partials_saved
    ydot_pred = jnp.zeros_like(ctx.pred_y)
    fydot = problem.fydot_partial(ctx.t, ctx.pred_y, ydot_pred, ctx.theta, control=ctx.control)
    return problem.recomposed_state_blocks(fy_saved, fydot, ctx.invgdt)


def ndf_setup_gate(
    *,
    first_call,
    prev_conv_fail,
    prev_err_fail,
    nst,
    nstlp,
    gdt,
    gdt_prev,
    force,
):
    """Whether to assemble and factor a fresh iteration matrix this attempt.

    Fires on the first step, after any failure of the prior attempt, once
    the leading-coefficient drift leaves its deadband, once the step-count
    window is exhausted, or when the always-fresh policy forces it.
    """
    dtype = jnp.asarray(gdt).dtype
    drift = jnp.abs(gdt / gdt_prev - jnp.asarray(1.0, dtype))
    fired = (
        jnp.asarray(first_call) & (nst == 0)
        | jnp.asarray(prev_conv_fail)
        | jnp.asarray(prev_err_fail)
        | (nst == 0)
        | (nst >= nstlp + MSBP)
        | (drift > jnp.asarray(W_DRIFT_CUTOFF, dtype))
    )
    return jnp.logical_or(fired, jnp.asarray(bool(force)))


def convergence_update(
    *,
    m: jax.Array,
    ndz: jax.Array,
    ndz_prev: jax.Array,
    theta_run: jax.Array,
    tol_factor: float = 1.0,
) -> ConvergenceUpdate:
    """One evaluation of the simplified-Newton convergence law.

    On the first iteration the carried contraction estimate is used as-is
    (a fresh memory of 1 makes ``eta`` infinite, so only the fast floor can
    accept); later iterations low-pass the update-norm ratio into it. A
    near-unit estimate means the iteration is at the floating-point floor
    and resolves by the raw norm.

    ``tol_factor`` widens every acceptance threshold (the rate-law bound,
    the first-iteration fast floor, and the stagnated-lane norm bound) by
    the same scale, loosening the corrector band relative to the step
    tolerance; ``1.0`` leaves the law untouched.
    """
    accept_bound = NEWTON_KAPPA
    fast_floor = FIRST_ITER_FAST_TOL
    stag_bound = 1.0
    if tol_factor != 1.0:
        accept_bound = NEWTON_KAPPA * float(tol_factor)
        fast_floor = FIRST_ITER_FAST_TOL * float(tol_factor)
        stag_bound = float(tol_factor)
    dtype = jnp.asarray(ndz).dtype
    one = jnp.asarray(1.0, dtype)
    is_first = m == 0

    ratio = safe_div(ndz, ndz_prev)
    theta_next = jnp.where(
        is_first, theta_run, jnp.maximum(jnp.asarray(RATE_LOWPASS, dtype) * theta_run, ratio)
    )

    stag_band = 100.0 * jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype))
    stagnated = jnp.logical_not(is_first) & (jnp.abs(theta_next - one) <= stag_band)

    denom = one - theta_next
    eta = jnp.where(denom != 0, theta_next / jnp.where(denom != 0, denom, one), jnp.inf)
    eta_ok = eta >= 0
    # mask the product in stagnated/degenerate lanes so inf*0 never surfaces
    eta_ndz = jnp.where(eta_ok & jnp.logical_not(stagnated), eta * ndz, jnp.inf)

    fast_first = is_first & (ndz < jnp.asarray(fast_floor, dtype))
    conv_normal = fast_first | (eta_ndz < jnp.asarray(accept_bound, dtype))
    div_normal = jnp.logical_not(is_first) & (theta_next > jnp.asarray(DIVERGENCE_THETA, dtype))

    stag_bound_t = jnp.asarray(stag_bound, dtype)
    converged = jnp.where(stagnated, ndz <= stag_bound_t, conv_normal)
    diverged = jnp.where(stagnated, ndz > stag_bound_t, div_normal)
    return ConvergenceUpdate(theta_next=theta_next, eta=eta, converged=converged, diverged=diverged)


class _NewtonCarry(NamedTuple):
    """The loop state threaded through one Newton sweep.

    The trailing fields carry the adaptive precision policy's detector state
    and stay ``None`` (absent from the compiled program) for concrete solve
    modes.
    """

    m: jax.Array
    dd: jax.Array
    ndz_prev: jax.Array
    theta_run: jax.Array
    ndz: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array
    nni: jax.Array
    stag_count: Any = None
    resnrm_first: Any = None
    resnrm_last: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def ndf_corrector(
    problem,
    linear,
    ctx: NDFNlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_saved_jac: Optional[tuple] = None,
    frozen_pencil: Optional[tuple] = None,
    *,
    generations: Optional[Generations] = None,
    border_cache=None,
    nst,
    nstlp,
    nstlj,
    gdt_prev,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
) -> NDFNlsResult:
    """Solve the correction system for ``dd`` on one step attempt.

    The setup gate decides whether to factor a fresh iteration matrix or
    reuse the supplied ``frozen_pencil`` ``(held, B, C, D)`` (the
    [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix] of
    the factored state block and the coupling blocks); every setup refactors
    the held matrix in the storage the held factor names, and
    ``linear`` is the state block's
    [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear].
    Within a setup the Jacobian gate
    decides between a fresh ``J_f`` and a pencil rebuild from
    ``frozen_saved_jac``. A sweep that fails with a stale Jacobian re-enters
    exactly once with a fresh Jacobian; a fresh-Jacobian failure ends the
    attempt. Gate memory (``nst``/``nstlp``/``nstlj``/``gdt_prev``) is
    passed explicitly and the updated values are returned on the result.

    On a problem with bordered-global rows, ``border_cache`` (with
    ``generations``, the carry's counters at attempt entry) carries the
    border solve, the Schur complement, and its small dense factor across
    attempts: each sweep revalidates the cache against the generation
    counters in effect for that sweep and rebuilds exactly the stale tiers,
    and every Newton iteration solves the bordered system by triangular
    solves against the held factor. Without a cache each sweep rebuilds the
    border data and each iteration refactors the small system; the iterates
    are identical either way.
    ``newton_tol_factor`` widens the convergence law's acceptance thresholds
    by a constant scale (``1.0`` = the engine's standard band).
    ``jacobian_values_dtype`` computes fresh Jacobian values at that precision
    (``None`` = the state dtype).

    Under the adaptive precision policy ``linear`` holds one route per rung
    and ``detect_precision`` arms the failure
    detectors: increment stagnation and false convergence are reported on the
    result (with the accumulated linear fp64-fallback count) so the caller can
    attribute the failure to precision and promote its rung.
    """
    dtype = ctx.pred_y.dtype
    n = int(ctx.pred_y.shape[0])
    n_u = int(problem.n_u)
    n_lam = int(problem.n_lam)

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
    topology = problem.state_topology
    a_shape = (n_u, n_u) if topology is None else (int(topology.rows.shape[0]),)

    zero_i32 = jnp.asarray(0, jnp.int32)

    caching = border_cache is not None and n_lam > 0
    # A differential-global border row carries ``cj * M_global`` in ``D``:
    # its coupling blocks and cached Schur data track the factor (every
    # setup), not just fresh-Jacobian events. Pure-algebraic borders keep
    # the fresh-Jacobian reuse unchanged.
    border_tracks_factor = n_lam > 0 and bool(np.asarray(problem.id_mask)[n_u:].any())
    if border_cache is not None and generations is None:
        raise ValueError(
            "a border cache is generation-keyed: pass the carry's generations "
            "record alongside border_cache so the cache can be validated "
            "against the counters naming the carried factor and coupling "
            "blocks."
        )

    force_setup = bool(directive.refresh_jacobian) or (frozen_pencil is None)
    call_setup = ndf_setup_gate(
        first_call=ctx.first_call,
        prev_conv_fail=ctx.prev_conv_fail,
        prev_err_fail=ctx.prev_err_fail,
        nst=nst,
        nstlp=nstlp,
        gdt=ctx.gdt,
        gdt_prev=gdt_prev,
        force=force_setup,
    )
    # within a setup: fresh J on the first step, a current-Jacobian failure,
    # or an exhausted step window; otherwise rebuild from the saved Jacobian
    reuse_window = fy_reuse_window(problem.plan, MSBJ)
    jbad_primary = (nst == 0) | (nst >= nstlj + reuse_window) | jnp.asarray(ctx.prev_conv_fail)
    if inexact:
        # The Jacobian is evaluated at every attempt, so a sweep's matrix is
        # always current and only the preconditioner can be stale: the retry
        # below fires exactly when the preconditioner was not rebuilt.
        jbad_primary = jnp.asarray(True)
        if n_lam > 0:
            raise ConfigurationError(
                "the iterative (Krylov) corrector solves the state block of a problem without "
                f"bordered-global rows; this problem has {n_lam}. Use an exact linear backend "
                "(cudss, klu, dense) for problems with global variables."
            )

    gated = frozen_pencil is not None and frozen_saved_jac is not None
    if gated:
        partials_saved, b_saved, c_saved, d_saved = frozen_saved_jac
        saved_carried = (partials_saved, b_saved, c_saved, d_saved)
        held_frozen, b_frozen, c_frozen, d_frozen = frozen_pencil
        a_frozen = held_frozen.values.reshape(a_shape)

        def _fresh_setup(_op):
            a_f, parts_f, b_f, c_f, d_f = ndf_state_jacobian(problem, ctx, jacobian_values_dtype)
            return (a_f, b_f, c_f, d_f), (parts_f, b_f, c_f, d_f)

        def _saved_setup(_op):
            if border_tracks_factor:
                a_r, b_r, c_r, d_r = _rebuild_pencil_blocks(problem, ctx, partials_saved)
                return (a_r, b_r, c_r, d_r), saved_carried
            a_r = _rebuild_pencil(problem, ctx, partials_saved)
            return (a_r, b_saved, c_saved, d_saved), saved_carried

        def _setup_blocks(use_fresh_j):
            return lane_cond(use_fresh_j, _fresh_setup, _saved_setup, None)

        def _primary_setup(_op):
            pencil, saved = _setup_blocks(jbad_primary)
            return pencil, saved

        a_frozen_primary = a_frozen
        if inexact:
            # The operator is the current matrix at every attempt (the
            # assembly is a fraction of a preconditioner build); the setup gate
            # decides only the preconditioner's rebuild.
            pencil_primary, saved_after_primary = _fresh_setup(None)
        else:
            pencil_primary, saved_after_primary = lane_cond(
                call_setup,
                _primary_setup,
                lambda _op: ((a_frozen_primary, b_frozen, c_frozen, d_frozen), saved_carried),
                None,
                false_is_free=True,
            )
    else:
        a_fresh, parts_fresh, b_fresh, c_fresh, d_fresh = ndf_state_jacobian(
            problem, ctx, jacobian_values_dtype
        )
        a_reuse = _rebuild_pencil(problem, ctx, parts_fresh)
        saved_after_primary = (parts_fresh, b_fresh, c_fresh, d_fresh)

        def _setup_blocks(use_fresh_j):
            a_aug = jnp.where(use_fresh_j, a_fresh, a_reuse)
            return (a_aug, b_fresh, c_fresh, d_fresh), saved_after_primary

        pencil_primary, _ = _setup_blocks(jnp.asarray(True) if inexact else jbad_primary)
        held_frozen = empty_held(linear, int(np.prod(a_shape)), dtype)
    if inexact:
        # The Newton test's norm and threshold set the linear tolerance.
        linear_test = dict(
            weights=(ctx.ewt * ctx.err_const_k)[:n_u],
            newton_tol=NEWTON_KAPPA * float(newton_tol_factor),
        )
    else:
        linear_test = {}

    def _linearize(held, pencil, refresh, cache, factor_key, coupling_key):
        """Refactor the held state block where it was replaced; make the border/Schur data current.

        ``refresh`` is true exactly when ``pencil`` carries values replaced
        this sweep: the held factor is refactored from them, and otherwise
        the solves use the factor already held. With a cache,
        ``factor_key`` / ``coupling_key`` are the generation counters in
        effect for this sweep's factor and coupling blocks: the cache
        revalidates against them and rebuilds exactly the stale tiers.
        Without one, the border solve and Schur complement are rebuilt here
        and the small system is refactored on every iteration.
        """
        a_block, b_block, c_block, d_block = pencil
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
                    b_generation=coupling_key,
                    c_generation=coupling_key,
                    d_generation=coupling_key,
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
            du_b_built = schur_built = zero_i32

            def schur_solve(rhs):
                return rhs

        return held, solve, du_b, schur_solve, c_block, cache, du_b_built, schur_built

    def _run_sweep(solve, du_b, schur_solve, c_block, theta_mem, nni_init):
        def newton_delta(dd):
            resid = ndf_residual(problem, ctx, dd)
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
            return jnp.concatenate([du, dlam]), resnrm, lflag, lsteps

        def take_step(carry: _NewtonCarry) -> _NewtonCarry:
            delta, resnrm, lflag, lsteps = newton_delta(carry.dd)
            dd_new = carry.dd + delta
            ndz = jax.lax.stop_gradient(wrms_norm(delta, ctx.ewt) * ctx.err_const_k)
            upd = convergence_update(
                m=carry.m,
                ndz=ndz,
                ndz_prev=carry.ndz_prev,
                theta_run=carry.theta_run,
                tol_factor=newton_tol_factor,
            )
            m_next = carry.m + 1
            if detect_precision:
                stag = stagnation_flag(ndz, carry.ndz_prev, upd.converged)
                stag_count = jnp.where(stag, carry.stag_count + 1, 0).astype(jnp.int32)
                resnrm_first = jnp.where(carry.m == 0, resnrm, carry.resnrm_first)
                resnrm_last = resnrm
            else:
                stag_count = resnrm_first = resnrm_last = None
            converged, diverged = upd.converged, upd.diverged
            if inexact:
                # A linear solve that missed its tolerance ends the sweep as a
                # failure: the caller rebuilds a stale preconditioner and retries.
                missed = lflag > 0
                converged = converged & jnp.logical_not(missed)
                diverged = diverged | missed
                lin_failures = carry.lin_failures + missed.astype(jnp.int32)
                krylov_iters = jnp.maximum(carry.krylov_iters, lsteps.astype(jnp.int32))
                lin_fallbacks = lin_ir_steps = None
            else:
                lin_failures = krylov_iters = None
                lin_fallbacks = None if lflag is None else carry.lin_fallbacks + lflag
                lin_ir_steps = None if lsteps is None else carry.lin_ir_steps + lsteps
            return _NewtonCarry(
                m=m_next,
                dd=dd_new,
                ndz_prev=ndz,
                theta_run=upd.theta_next,
                ndz=ndz,
                converged=converged,
                diverged=diverged,
                iterations=jnp.where(converged, m_next, carry.iterations),
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
            m=jnp.asarray(0, jnp.int32),
            dd=jnp.zeros(n, dtype=dtype),
            ndz_prev=jnp.asarray(jnp.inf, dtype),
            theta_run=jnp.asarray(theta_mem, dtype),
            ndz=jnp.asarray(jnp.inf, dtype),
            converged=jnp.asarray(False),
            diverged=jnp.asarray(False),
            iterations=zero_i32,
            nni=nni_init,
            stag_count=zero_i32 if detect_precision else None,
            resnrm_first=jnp.asarray(0.0, dtype) if detect_precision else None,
            resnrm_last=jnp.asarray(0.0, dtype) if detect_precision else None,
            lin_fallbacks=jnp.zeros((), jnp.float64) if flagged else None,
            lin_ir_steps=jnp.zeros((), jnp.float64) if flagged else None,
            lin_failures=zero_i32 if inexact else None,
            krylov_iters=zero_i32 if inexact else None,
        )

        def done(carry: _NewtonCarry) -> jax.Array:
            return jnp.logical_or(carry.converged, carry.diverged)

        if loop_form == "fixed":

            def fori_body(_i, carry):
                return lane_cond(
                    jnp.logical_not(done(carry)), take_step, lambda c: c, carry, false_is_free=True
                )

            return jax.lax.fori_loop(0, NEWTON_MAXITERS, fori_body, init)
        if loop_form == "adaptive":

            def cond(carry: _NewtonCarry) -> jax.Array:
                return jnp.logical_and(
                    jnp.logical_not(done(carry)),
                    carry.m < jnp.asarray(NEWTON_MAXITERS, carry.m.dtype),
                )

            return jax.lax.while_loop(cond, take_step, init)
        raise ValueError(f"loop_form must be 'fixed' or 'adaptive', got {loop_form!r}.")

    # Generation counters in effect for the primary sweep: the primary setup
    # replaces the factor (and, with a fresh Jacobian, the coupling blocks)
    # BEFORE the sweep consumes them, so its keys see those bumps. They
    # mirror exactly the advancement `_advance_generations` applies after
    # the attempt, keeping cache keys and the carried record in lock-step.
    if caching:
        if border_tracks_factor:
            primary_fresh_key = call_setup
        else:
            primary_fresh_key = jnp.logical_and(call_setup, jbad_primary)
        factor_key0 = generations.factor + call_setup.astype(jnp.int32)
        coupling_key0 = generations.b + primary_fresh_key.astype(jnp.int32)
    else:
        factor_key0 = coupling_key0 = zero_i32
    cache_in = border_cache if caching else None

    held1, solve0, du_b0, schur_solve0, cblk0, cache1, dub0, sb0 = _linearize(
        held_frozen,
        pencil_primary,
        call_setup,
        cache_in,
        factor_key0,
        coupling_key0,
    )
    pencil_held1 = (held1,) + tuple(pencil_primary[1:])
    carry1 = _run_sweep(solve0, du_b0, schur_solve0, cblk0, ctx.theta_mem, zero_i32)

    # one fresh-Jacobian re-entry when the sweep failed on a stale Jacobian
    jcur_primary = jnp.logical_and(call_setup, jbad_primary)
    ran_retry = jnp.logical_and(jnp.logical_not(carry1.converged), jnp.logical_not(jcur_primary))

    def _retry(_op):
        if inexact:
            # The matrix is already current: the retry only rebuilds the
            # preconditioner from it.
            pencil_r, saved_r = pencil_primary, saved_after_primary
        else:
            pencil_r, saved_r = _setup_blocks(jnp.asarray(True))
        # The retry is always a fresh-Jacobian setup: factor and coupling
        # generations advance once more past the primary sweep's keys.
        if caching:
            factor_key_r = factor_key0 + 1
            coupling_key_r = coupling_key0 + 1
        else:
            factor_key_r = coupling_key_r = zero_i32
        # The refresh flag holds wherever this branch's result is kept; a lane
        # of a batch that runs the branch without needing it keeps its held
        # factor.
        held_r, solve_r, du_b_r, schur_solve_r, cblk_r, cache_r, dub_r, sb_r = _linearize(
            held1, pencil_r, ran_retry, cache1, factor_key_r, coupling_key_r
        )
        carry_r = _run_sweep(solve_r, du_b_r, schur_solve_r, cblk_r, carry1.theta_run, carry1.nni)
        return carry_r, (held_r,) + tuple(pencil_r[1:]), saved_r, cache_r, dub_r, sb_r

    carry2, pencil_out, saved_jac_out, cache_out, dub_retry, sb_retry = lane_cond(
        ran_retry,
        _retry,
        lambda _op: (carry1, pencil_held1, saved_after_primary, cache1, zero_i32, zero_i32),
        None,
        false_is_free=True,
    )
    du_b_builds = dub0 + dub_retry
    schur_builds = sb0 + sb_retry

    converged = carry2.converged
    nsetups = call_setup.astype(jnp.int32) + ran_retry.astype(jnp.int32)
    if inexact:
        nje = jnp.ones_like(nsetups)
    else:
        nje = jnp.logical_and(call_setup, jbad_primary).astype(jnp.int32) + ran_retry.astype(
            jnp.int32
        )
    nnf = ran_retry.astype(jnp.int32) + jnp.logical_not(converged).astype(jnp.int32)
    any_setup = jnp.logical_or(call_setup, ran_retry)
    jcur_out = jnp.where(ran_retry, jnp.asarray(True), jcur_primary)

    gdt_prev_out = jnp.where(any_setup, ctx.gdt, gdt_prev)
    nstlp_out = jnp.where(any_setup, nst, nstlp).astype(jnp.int32)
    any_fresh_j = jnp.logical_or(jnp.logical_and(call_setup, jbad_primary), ran_retry)
    nstlj_out = jnp.where(any_fresh_j, nst, nstlj).astype(jnp.int32)

    if detect_precision:
        prec_stagnant = (carry2.stag_count >= STAGNATION_LATCH) & jnp.logical_not(converged)
        prec_false_conv = (
            converged & (carry2.iterations >= 2) & (carry2.resnrm_last >= carry2.resnrm_first)
        )
    else:
        prec_stagnant = prec_false_conv = None

    return NDFNlsResult(
        dd=carry2.dd,
        y=ctx.pred_y + carry2.dd,
        converged=converged,
        diverged=carry2.diverged,
        iterations=carry2.iterations,
        ndz_last=carry2.ndz,
        theta_out=carry2.theta_run,
        call_setup=call_setup,
        jbad=jbad_primary,
        jcur=jcur_out,
        nsetups=nsetups,
        nje=nje,
        nni=carry2.nni,
        nnf=nnf,
        gdt_prev_out=gdt_prev_out,
        nstlp_out=nstlp_out,
        nstlj_out=nstlj_out,
        saved_jac_out=saved_jac_out,
        pencil_out=pencil_out,
        prec_stagnant=prec_stagnant,
        prec_false_conv=prec_false_conv,
        lin_fallbacks=carry2.lin_fallbacks,
        lin_ir_steps=carry2.lin_ir_steps,
        border_cache_out=cache_out,
        du_b_builds=du_b_builds,
        schur_builds=schur_builds,
        lin_failures=carry2.lin_failures,
        krylov_iters=carry2.krylov_iters,
    )


def ndf_corrector_root(
    problem,
    backend,
    linear,
    sensitivity_analysis,
    ctx: NDFNlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_saved_jac: Optional[tuple] = None,
    frozen_pencil: Optional[tuple] = None,
    *,
    reverse_analysis=None,
    generations: Optional[Generations] = None,
    border_cache=None,
    nst,
    nstlp,
    nstlj,
    gdt_prev,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
) -> NDFNlsResult:
    """One corrector attempt with the accepted root behind its derivative boundary.

    Runs [ndf_corrector][] as the primal solve and binds the returned
    corrected state to the implicit-function rule at that root: with
    ``cj = invgdt`` and the history aggregate ``q = phi/dt - cj * pred_y``,
    the accepted state solves ``F(t, y, cj*y + q, theta) = 0``, so the
    corrected state and correction carry the exact root derivative while
    every primal bookkeeping output (convergence flags, counters, contraction
    memory, setup decisions, carried matrix blocks) stays gradient-dead.
    ``linear`` is the state block's
    [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear],
    ``sensitivity_analysis`` the ``backend``'s forward-direction analysis of
    the problem's full-system pattern (``None`` on the dense path), which
    serves the tangent solve, and ``reverse_analysis`` its reverse-direction
    analysis, which serves the transposed solve of a reverse sweep (``None``
    reuses ``sensitivity_analysis``).

    ``generations`` names the caller's carried linearization artifacts: the
    returned ``generations_out`` advances its counters by this attempt's
    rebuild events, and — because this helper owns the root-acceptance
    decision — ``certificate_out`` is issued for those counters through
    [certify_exact_root][diff_soup.nonlinear.generations.certify_exact_root] with the
    exact-at-root predicate (see [_root_success][]): only a factor built
    this attempt from values evaluated exactly at the accepted root is
    certified, so downstream reuse decisions can trust an authorizing
    certificate unconditionally. The engine's own reuse windows are governed
    by its convergence monitoring, not by the certificate. With
    ``generations=None`` no record is kept and both outputs are ``None``.

    ``border_cache`` is the carry's generation-keyed border/Schur cache (see
    [ndf_corrector][]); it is consulted and refreshed only inside the
    primal solve, so the cached factors never reach the sensitivity rules —
    those rebuild exact values at the accepted root.

    Remaining arguments are those of [ndf_corrector][].
    """

    def corrector_at(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector at the reconstructed predictor (never differentiated).

        Every traced solver value (gate memory, error weights, carried matrix
        blocks, precision-rung state) arrives through ``primal_args`` so this
        closure captures only static configuration.
        """
        (
            ewt,
            err_const_k,
            gdt,
            theta_mem,
            first_call,
            prev_conv_fail,
            prev_err_fail,
            prec_rung,
            rung_changed,
            saved_jac_p,
            pencil_p,
            generations_p,
            border_cache_p,
            nst_p,
            nstlp_p,
            nstlj_p,
            gdt_prev_p,
        ) = primal_args
        ctx_p = NDFNlsContext(
            t=t,
            theta=theta,
            invgdt=cj,
            gdt=gdt,
            phi_dt=q + cj * guess,
            pred_y=guess,
            ewt=ewt,
            err_const_k=err_const_k,
            theta_mem=theta_mem,
            first_call=first_call,
            prev_conv_fail=prev_conv_fail,
            prev_err_fail=prev_err_fail,
            prec_rung=prec_rung,
            rung_changed=rung_changed,
            control=control,
        )
        return ndf_corrector(
            problem,
            linear,
            ctx_p,
            directive,
            loop_form,
            frozen_saved_jac=saved_jac_p,
            frozen_pencil=pencil_p,
            generations=generations_p,
            border_cache=border_cache_p,
            nst=nst_p,
            nstlp=nstlp_p,
            nstlj=nstlj_p,
            gdt_prev=gdt_prev_p,
            newton_tol_factor=newton_tol_factor,
            jacobian_values_dtype=jacobian_values_dtype,
            detect_precision=detect_precision,
        )

    def primal_fn(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector and expose its converged root to the derivative rule."""
        res = corrector_at(t, cj, q, theta, control, guess, primal_args)
        return res.y, res.converged, res

    q = ctx.phi_dt - ctx.invgdt * ctx.pred_y
    primal_args = (
        ctx.ewt,
        ctx.err_const_k,
        ctx.gdt,
        ctx.theta_mem,
        ctx.first_call,
        ctx.prev_conv_fail,
        ctx.prev_err_fail,
        ctx.prec_rung,
        ctx.rung_changed,
        frozen_saved_jac,
        frozen_pencil,
        generations,
        border_cache,
        nst,
        nstlp,
        nstlj,
        gdt_prev,
    )

    def offering_extractor(aux, t, cj, theta, control, primal_args):
        """Build the iteration-matrix offering from the corrector result.

        Recomposes ``A(cj) = F_y + cj * F_ydot`` from the corrector's saved
        split partials at the step's leading coefficient and gates it with the
        strict at-root predicate. The offering is assembled inside the
        derivative rule from the ``aux`` the primal already returned, so no
        second corrector run is needed to source it. It supplies only the
        sensitivity solve's matrix VALUE for a first derivative — the matrix's
        own tangent always rebuilds at the frozen root, keeping higher-order
        derivatives exact.
        """
        del t, theta, control
        prec_rung = primal_args[7]
        fy_saved, fydot_saved = aux.saved_jac_out[0]
        return problem.iteration_matrix_offering(
            fy_saved,
            fydot_saved,
            cj,
            _root_success(aux, prec_rung, detect_precision),
        )

    y, _ok, aux = problem.bind_step_root(
        backend,
        sensitivity_analysis,
        primal_fn,
        ctx.t,
        ctx.invgdt,
        q,
        ctx.theta,
        ctx.control,
        ctx.pred_y,
        primal_args,
        reverse_analysis=reverse_analysis,
        certificate_extractor=offering_extractor,
    )
    generations_out = certificate_out = None
    if generations is not None:
        border_tracks_factor = int(problem.n_lam) > 0 and bool(
            np.asarray(problem.id_mask)[int(problem.n_u) :].any()
        )
        generations_out = _advance_generations(
            generations, aux, border_tracks_factor=border_tracks_factor
        )
        certificate_out = certify_exact_root(
            generations_out, _root_success(aux, ctx.prec_rung, detect_precision)
        )
    return aux._replace(
        dd=y - ctx.pred_y,
        y=y,
        generations_out=generations_out,
        certificate_out=certificate_out,
    )


def _advance_generations(
    generations: Generations, res: NDFNlsResult, *, border_tracks_factor: bool = False
) -> Generations:
    """Advance the carry's generation counters by one attempt's rebuild events.

    The primary setup and the fresh-Jacobian retry each replace the
    iteration-matrix values, the point they were evaluated at, and the
    numeric factor built from them (held-factor and per-call solve styles
    alike rebuild on exactly these events), so each bumps ``point`` /
    ``matrix`` / ``factor`` once. On a pure-algebraic border the
    bordered-global coupling blocks are re-evaluated only by a fresh-Jacobian
    setup — a recomposition keeps the saved blocks — so ``b`` / ``c`` /
    ``d`` bump with the fresh-Jacobian events. When the border tracks the
    factor (a differential-global row carries ``cj * M_global`` in ``D``),
    every setup recombines the coupling blocks at the current leading
    coefficient, so ``b`` / ``c`` / ``d`` bump with every setup. Parameter
    and control identities never move within one solve.
    """

    def blend(cond, bumped: Generations, base: Generations) -> Generations:
        return jax.tree_util.tree_map(lambda a, b: jnp.where(cond, a, b), bumped, base)

    coupling = generations.b is not None
    if border_tracks_factor:
        primary_fresh = res.call_setup
    else:
        primary_fresh = jnp.logical_and(res.call_setup, res.jbad)
    ran_retry = (res.nsetups - res.call_setup.astype(res.nsetups.dtype)) > 0

    g = blend(res.call_setup, generations.bump("point", "matrix", "factor"), generations)
    if coupling:
        g = blend(primary_fresh, g.bump("b", "c", "d"), g)
    g = blend(ran_retry, g.bump("point", "matrix", "factor"), g)
    if coupling:
        g = blend(ran_retry, g.bump("b", "c", "d"), g)
    return g


def _root_success(res: NDFNlsResult, prec_rung, detect_precision: bool) -> jax.Array:
    """The exact-at-root predicate for certificate issuance.

    A certificate attests that the held factor was built from matrix values
    evaluated exactly at a converged root, so issuance requires all of:

    - the corrector converged (minus a reduced-precision false convergence
      when the precision detectors are armed — the step logic rejects those
      sweeps, so their roots are never accepted);
    - the primary setup evaluated a fresh Jacobian this attempt (a
      window-reused or recomposed matrix carries values from an earlier
      trajectory point);
    - the total applied correction is exactly zero, so the evaluation point
      provably IS the accepted root.

    The fresh-Jacobian retry is never certified: it evaluates at the same
    predictor, but proving the post-retry correction vanished would need
    plumbing that buys nothing — under-issuance is always safe (a
    certificate is an opportunity marker, not a liveness requirement),
    while over-issuance would let a downstream consumer trust a factor
    built away from the root. ``prec_rung`` is the attempt's precision rung,
    read only when the precision detectors are armed.
    """
    at_root = jnp.all(res.dd == 0.0)
    fresh_here = jnp.logical_and(res.call_setup, res.jbad)
    no_retry = (res.nsetups - res.call_setup.astype(res.nsetups.dtype)) == 0
    root_ok = jnp.asarray(res.converged) & fresh_here & no_retry & at_root
    if detect_precision:
        reduced = prec_rung < TOP_RUNG
        root_ok = jnp.logical_and(
            root_ok, jnp.logical_not(jnp.asarray(res.prec_false_conv) & reduced)
        )
    return root_ok


__all__ = [
    "NDFNlsContext",
    "NDFNlsResult",
    "ConvergenceUpdate",
    "convergence_update",
    "ndf_residual",
    "ndf_state_jacobian",
    "ndf_setup_gate",
    "ndf_corrector",
    "ndf_corrector_root",
    "NEWTON_MAXITERS",
    "NEWTON_KAPPA",
    "RATE_LOWPASS",
    "DIVERGENCE_THETA",
    "FIRST_ITER_FAST_TOL",
    "W_DRIFT_CUTOFF",
    "MSBP",
    "MSBJ",
]
