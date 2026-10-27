"""The correction-form corrector for the BDF-tracking engine.

This module implements the BDF-tracking engine's nonlinear step machinery: the
correction-form residual, the two-tier linear-solver setup, the saved-Jacobian
reuse channel, and a Newton corrector with a rate-memory (``crate``) convergence
test (``dcon``), distinct from the IDA-tracking engine's own memory.

A mass-free BDF corrector integrates ``y' = f`` with residual
``rl1*zn[1] + ycor - gamma*f`` and iteration matrix ``I - gamma*J_f``.
This engine supports a diagonal mass ``M`` restricted to the identity, a constant
matrix, or a purely time-dependent matrix. The residual generalizes to the
standard pencil ``M - gamma*J_f`` as follows. The corrected derivative satisfies
``ydot_new = ydot_pred + ycor/gamma`` (from ``zn[1]_new = zn[1] + l[1]*ycor`` and
``gamma = h/l[1]``), so the mass-form ODE ``M*ydot = f`` becomes ``M*(ydot_pred +
ycor/gamma) - f = 0``; multiplying by ``gamma`` and using ``gamma*ydot_pred =
rl1*zn[1]`` gives the raw residual ``M @ (rl1*zn[1] + ycor) - gamma*f`` (which
reduces exactly to the mass-free form for ``M = I``). Dividing by ``gamma``
(``cj = 1/gamma``) gives the scaled form this engine solves:

    res = cj * M @ (rl1*zn[1] + ycor) - f(t, pred_y + ycor).

Its Newton matrix is ``cj*M - J_f = cj*(M - gamma*J_f) = cj*A_pencil`` with
``A_pencil = M - gamma*J_f`` composed here from the raw ``J_f = df/du`` that
``jacobian_blocks`` returns and the mass diagonal -- identical in form to the
fully-implicit corrector's ``A_aug = cj*A_pencil``, with the global-coupling
blocks ``B/C/D`` negated into the Schur convention and used unscaled. The Newton
update and its norms are invariant to this overall scaling, so the iterates, the
convergence test, and the local-error estimate are the same as the mass-free
form's.

A row with zero mass is a pure algebraic root: its residual reduces to
``-f_row`` (root ``f_row = 0``) with pencil row ``-J_f_row``. A constant singular
mass ``M = diag(1, ..., 0)`` therefore yields an index-1 differential-algebraic
row handled correctly by the correction form -- which is why this engine can
integrate constant-mass differential-algebraic systems that a mass-free BDF
corrector cannot.
The mass is re-evaluated at the current time and iterate but carries no
state-derivative pencil term; only the identity, constant, and time-dependent
mass classes are supported.

The engine owns its Newton loop against the linear backend's factor / solve plus
a Schur reduction over the global block, so the engine's own iteration counts,
rate memory, and setup decisions are emitted natively and the shared nonlinear
layer is untouched.

Two decisions drive the linear work, both reproduced from the carried state. The
*setup gate* decides whether to assemble and factor a fresh iteration matrix at
all this attempt (a step-count / gamma-drift window). Within a setup, the
*Jacobian-recompute gate* decides whether to re-evaluate ``J_f`` or reuse a saved
copy: on a reuse, the pencil ``M(t_pred) - gamma*J_f_saved`` is rebuilt with the
new ``gamma`` and re-factored WITHOUT re-evaluating ``J_f``. The raw ``J_f`` the
problem supplies is carried directly for reuse, so a reuse setup does zero
Jacobian evaluations and a fresh setup does exactly one -- keeping the setup and
Jacobian-evaluation counters split, the same distinction a BDF corrector's
linear-algebra bookkeeping needs.

Two loop shapes are available through ``loop_form``. ``"fixed"`` runs a bounded
``fori_loop`` that freezes once the convergence test is met; ``"adaptive"`` runs
an early-exit ``while_loop``. Both use the identical convergence test, so the
choice never changes the accepted iterate or the reported iteration counts.
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
from diff_soup.timestepping.vsvo.common.contract import fy_reuse_window
from diff_soup.timestepping.vsvo.common.linear_ops import (
    empty_held,
    linearize,
)
from diff_soup.timestepping.vsvo.common.weights import wrms_norm
from diff_soup.utils import safe_div
from diff_soup.utils.lanes import lane_cond

from .flags import NonlinearFlag

# --- corrector constants ---
NLS_MAXCOR = 3
"""Maximum corrector (Newton) iterations per step attempt."""

CRDOWN = 0.3
"""Convergence-rate decay factor: ``crate = max(CRDOWN*crate, delnrm/delp)``."""

RDIV = 2.0
"""Divergence threshold: a recoverable failure is declared when ``delnrm >
RDIV*delp`` for ``m >= 1``."""

MSBP = 20
"""Maximum steps between linear-solver setups."""

DGMAX_LSETUP = 0.3
"""Gamma-ratio threshold that forces a linear-solver setup: setup fires when
``|gamrat - 1| > DGMAX_LSETUP``."""

MSBJ = 51
"""Default maximum steps between Jacobian evaluations.

A problem's ``max_fy_reuse_steps`` declaration may reduce this resource
window but never enlarge it.
"""

DGMAX_JBAD = 0.2
"""Gamma-drift threshold below which a bad-Jacobian setup re-evaluates ``J_f``."""

_SCHUR_EPS = 1e-12
"""Tikhonov regularization for the small Schur-complement solve over the global
block (matches the shared nonlinear layer's default)."""


class BDFNlsContext(NamedTuple):
    """The per-attempt inputs the corrector solves against.

    Built once per step attempt at the API boundary and held constant across
    every retry of the attempt (in particular ``ewt``, refreshed only after an
    accepted step, never on a rejected retry).

    Attributes:
        t: The endpoint time of the step attempt (after any endpoint clamp).
        theta: The learnable-parameter pytree passed to the residual.
        gamma: The pencil coefficient ``hinvGak = h/l[1]`` for this attempt.
        cj: The leading coefficient ``1/gamma`` (materialized, never inverted).
        gamrat: ``gamma/gammap``: the ratio that gates a fresh setup and drives
            the ``2/(1+gamrat)`` correction rescale.
        rl1_zn1: ``rl1 * zn[1]``, the predicted-derivative term, shape ``(n,)``.
        pred_y: The predicted state ``zn[0]``, shape ``(n,)``.
        ewt: The error-weight vector used for the Newton norm, shape ``(n,)``.
        tq4: ``tq[4] = nlscoef/tq[2]``: the convergence-test tolerance.
        nflag: The reason code for this corrector call
            ([NonlinearFlag][diff_soup.timestepping.vsvo.engines.bdf.flags.NonlinearFlag]).
        control: The control-value pytree the residual reads, or
            ``None`` for a problem without a control channel.
    """

    t: Any
    theta: Any
    gamma: jax.Array
    cj: jax.Array
    gamrat: jax.Array
    rl1_zn1: jax.Array
    pred_y: jax.Array
    ewt: jax.Array
    tq4: jax.Array
    nflag: jax.Array
    prec_rung: Any = None
    rung_changed: Any = None
    control: Any = None


class BDFNlsResult(NamedTuple):
    """The outcome of one correction-form corrector attempt.

    Attributes:
        acor: The converged correction ``ycor``, shape ``(n,)``.
        y: The corrected state ``pred_y + acor``, shape ``(n,)``.
        converged: Whether the convergence test was met.
        diverged: Whether the update norm exceeded the divergence threshold.
        iterations: The first-convergence iteration count (``int32``).
        delnrm: The last Newton-update weighted-RMS norm.
        crate: The convergence-rate memory carried to the next step.
        acnrm: The weighted-RMS norm of the correction (or the first update norm
            when convergence is reached on the first iteration).
        acnrmcur: Whether ``acnrm`` was set by the convergence test.
        dcon: The last convergence-test ratio (for the trace).
        call_setup: The setup-gate decision taken on the primary attempt.
        jbad: The Jacobian-recompute decision on the primary setup.
        jcur: Whether the Jacobian is current after this attempt.
        nsetups: Linear-solver setups performed this attempt (``int32``).
        nje: Jacobian evaluations performed this attempt (``int32``).
        nni: Newton iterations performed this attempt (``int32``).
        nnf: Newton convergence failures counted this attempt (``int32``): one
            per stale-Jacobian setup re-entry, plus one more if the attempt still
            failed.
        gamrat_out: The post-setup ``gamrat`` (``1`` after a setup, else carried).
        gammap_out: The post-setup ``gammap``.
        nstlp_out: The step number of the most recent setup (``int32``).
        nstlj_out: The step number of the most recent Jacobian update (``int32``).
        saved_jac_out: The ``(J_f, B, C, D)`` in effect at exit -- fresh where the
            Jacobian gate fired, carried otherwise.
        pencil_out: The ``(held, B, C, D)`` iteration matrix in effect at exit:
            the [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the state block (refactored where any setup fired, the supplied
            ``frozen_pencil``'s otherwise) and the global-coupling blocks.
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

    acor: jax.Array
    y: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array
    delnrm: jax.Array
    crate: jax.Array
    acnrm: jax.Array
    acnrmcur: jax.Array
    dcon: jax.Array
    call_setup: jax.Array
    jbad: jax.Array
    jcur: jax.Array
    nsetups: jax.Array
    nje: jax.Array
    nni: jax.Array
    nnf: jax.Array
    gamrat_out: jax.Array
    gammap_out: jax.Array
    nstlp_out: jax.Array
    nstlj_out: jax.Array
    saved_jac_out: tuple
    pencil_out: tuple
    prec_stagnant: Any = None
    prec_false_conv: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    border_cache_out: Any = None
    du_b_builds: Any = None
    schur_builds: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def bdf_residual(problem, ctx: BDFNlsContext, ycor: jax.Array) -> jax.Array:
    """The correction-form mass residual.

    Returns ``res = cj * M @ (rl1*zn[1] + ycor) - f(t, pred_y + ycor)`` with the
    diagonal mass ``M`` (generalized from a mass-free residual to the diagonal
    mass and divided by ``gamma``). A row with zero mass reduces to the pure
    algebraic root ``-f_row``.
    """
    y = ctx.pred_y + ycor
    return problem.residual(ctx.t, y, ctx.cj * (ctx.rl1_zn1 + ycor), ctx.theta, ctx.control)


def bdf_state_jacobian(problem, ctx: BDFNlsContext, values_dtype=None):
    """Assemble the fresh iteration matrix from the raw state Jacobian.

    Returns ``(A_aug, J_f, B, C, D)`` where ``A_aug = cj * (M - gamma*J_f)`` is
    the scaled state block, ``J_f`` is the raw right-hand-side Jacobian the
    problem supplies, and ``B, C, D`` are the global-coupling blocks.

    The problem's `jacobian_blocks` returns the raw ``J_f = df/du`` plus the
    raw ``(df/dlam, dg/du, dg/dlam)``; this engine composes the pencil ``M -
    gamma*J_f`` from ``J_f`` and the mass diagonal ``M(t_pred)`` and negates the
    coupling blocks into the Schur convention. The predicted derivative is passed
    as zeros because the pencil is derivative-independent for the identity,
    constant, and time-dependent mass classes this engine supports.
    """
    ydot_pred = jnp.zeros_like(ctx.pred_y)
    fy, fydot = problem.state_partials(
        ctx.t, ctx.pred_y, ydot_pred, ctx.theta, values_dtype=values_dtype, control=ctx.control
    )
    a_aug, b, c, d = problem.recomposed_state_blocks(fy, fydot, ctx.cj)
    return a_aug, (fy, fydot), b, c, d


def _bdf_rebuild_pencil(problem, ctx: BDFNlsContext, partials_saved: tuple) -> jax.Array:
    """Rebuild the scaled iteration matrix from a saved raw Jacobian.

    A reuse-Jacobian setup re-evaluates the mass diagonal at the predictor and
    combines it with the saved ``J_f`` and the new ``gamma``: ``A_pencil =
    diag(M(t_pred)) - gamma*J_f_saved``, ``A_aug = cj*A_pencil``. No
    ``jacobian_blocks`` call is made.
    """
    return _bdf_rebuild_pencil_blocks(problem, ctx, partials_saved)[0]


def _bdf_rebuild_pencil_blocks(problem, ctx: BDFNlsContext, partials_saved: tuple):
    """Rebuild all four engine blocks from a saved raw Jacobian at the current cj.

    A differential-global border makes ``D`` depend on the leading
    coefficient through ``cj * M_global``, so a saved-Jacobian setup must
    still recombine the coupling blocks at the current ``cj``.
    """
    fy_saved, _ = partials_saved
    ydot_pred = jnp.zeros_like(ctx.pred_y)
    fydot = problem.fydot_partial(ctx.t, ctx.pred_y, ydot_pred, ctx.theta, control=ctx.control)
    return problem.recomposed_state_blocks(fy_saved, fydot, ctx.cj)


def bdf_setup_gate(nflag, nst, nstlp, gamrat, refresh_jacobian, msbp=MSBP, dgmax=DGMAX_LSETUP):
    """The linear-solver setup gate.

    Returns whether to assemble and factor a fresh iteration matrix this attempt:
    a setup fires after a prior convergence or error-test failure, on the first
    step, once the step-count window ``nst >= nstlp + msbp`` is reached, or once
    the gamma drift ``|gamrat - 1| > dgmax`` leaves the deadband. The static
    ``refresh_jacobian`` flag forces the gate open (the always-fresh policy).
    """
    dtype = jnp.asarray(gamrat).dtype
    fired = (
        (nflag == NonlinearFlag.PREV_CONV_FAIL)
        | (nflag == NonlinearFlag.PREV_PREC_FAIL)
        | (nflag == NonlinearFlag.PREV_ERR_FAIL)
        | (nst == 0)
        | (nst >= nstlp + msbp)
        | (jnp.abs(gamrat - jnp.asarray(1.0, dtype)) > jnp.asarray(dgmax, dtype))
    )
    return jnp.logical_or(fired, jnp.asarray(bool(refresh_jacobian)))


def bdf_jbad_gate(
    nst,
    nstlj,
    gamma,
    gammap,
    convfail_other,
    convfail_bad_j,
    msbj=MSBJ,
    dgmax_jbad=DGMAX_JBAD,
):
    """The Jacobian-recompute gate.

    Returns whether to re-evaluate ``J_f`` within a setup. A fresh Jacobian is
    forced on the first step, once the step-count window ``nst >= nstlj + msbj``
    is reached, when a current-Jacobian failure occurred (``convfail_other``), or
    when a bad-Jacobian failure occurred with a small gamma drift
    (``convfail_bad_j`` and ``dgamma < dgmax_jbad``); a bad-Jacobian failure with
    a large gamma drift instead reuses the saved Jacobian, presuming gamma -- not
    the Jacobian -- was the problem. ``dgamma = |gamma/gammap - 1|``.
    """
    dtype = jnp.asarray(gamma).dtype
    dgamma = jnp.abs(gamma / gammap - jnp.asarray(1.0, dtype))
    return (
        (nst == 0)
        | (nst >= nstlj + msbj)
        | (jnp.asarray(convfail_bad_j) & (dgamma < jnp.asarray(dgmax_jbad, dtype)))
        | jnp.asarray(convfail_other)
    )


def bdf_conv_test(m, delnrm, delp, crate, tq4, tol_factor: float = 1.0):
    """One convergence-test evaluation.

    Returns ``(crate_new, dcon, converged, diverged)`` for a single Newton
    iteration. The rate memory ``crate`` is updated only for ``m > 0``; the
    convergence ratio is ``dcon = delnrm * min(1, crate) / tq4``; convergence is
    ``dcon <= 1``; a recoverable divergence is ``delnrm > RDIV*delp`` for
    ``m >= 1``. ``tol_factor`` widens the convergence-test tolerance ``tq4``
    by a constant scale (``1.0`` = the standard band); the divergence test is
    a norm ratio and is unaffected.
    """
    if tol_factor != 1.0:
        tq4 = tq4 * float(tol_factor)
    dtype = jnp.asarray(delnrm).dtype
    one = jnp.asarray(1.0, dtype)
    is_first = m == 0
    # ``safe_div``: on a trajectory the formula integrates exactly (an exact
    # polynomial fit) the Newton updates are exactly zero, so the unguarded
    # 0/0 here — reached only in the frozen/discarded lanes of the bounded
    # corrector — would poison reverse-mode cotangents of the live update.
    crate_updated = jnp.maximum(jnp.asarray(CRDOWN, dtype) * crate, safe_div(delnrm, delp))
    crate_new = jnp.where(is_first, crate, crate_updated)
    dcon = delnrm * jnp.minimum(one, crate_new) / tq4
    converged = dcon <= one
    diverged = jnp.logical_and(jnp.logical_not(is_first), delnrm > jnp.asarray(RDIV, dtype) * delp)
    return crate_new, dcon, converged, diverged


class _NewtonCarry(NamedTuple):
    """The loop state threaded through one Newton sweep."""

    m: jax.Array  # completed iterations (0 on entry)
    ycor: jax.Array
    delp: jax.Array
    crate: jax.Array
    delnrm: jax.Array
    dcon: jax.Array
    converged: jax.Array
    diverged: jax.Array
    iterations: jax.Array  # first-convergence count (m+1)
    acnrm: jax.Array
    acnrmcur: jax.Array
    nni: jax.Array
    stag_count: Any = None
    resnrm_first: Any = None
    resnrm_last: Any = None
    lin_fallbacks: Any = None
    lin_ir_steps: Any = None
    lin_failures: Any = None
    krylov_iters: Any = None


def bdf_corrector(
    problem,
    linear,
    state,
    ctx: BDFNlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_saved_jac: Optional[tuple] = None,
    frozen_pencil: Optional[tuple] = None,
    *,
    generations: Optional[Generations] = None,
    border_cache=None,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
) -> BDFNlsResult:
    """Solve the correction system for the correction ``acor``.

    Runs the BDF Newton corrector for one step attempt: the setup gate decides
    whether to assemble a fresh iteration matrix or reuse the supplied
    ``frozen_pencil``; within a setup the Jacobian-recompute gate decides whether
    to re-evaluate ``J_f`` or rebuild the pencil from the supplied
    ``frozen_saved_jac``; the Newton loop iterates the correction using the
    engine's own ``crate``/``dcon`` convergence test; and a bounded outer loop
    re-enters setup on a recoverable failure taken with a non-current Jacobian.
    Because any setup resets the gamma drift to zero and a fresh-Jacobian setup
    marks the Jacobian current, at most two such re-entries run before the
    attempt ends -- the first may only rebuild from the saved Jacobian (large
    carried gamma drift), the second always recomputes a fresh one.

    The saved Jacobian and the factored pencil persist across attempts and steps:
    they are re-evaluated only inside a setup, so between setups the same matrix
    and its factorization are reused. Every setup refactors the held pencil in
    the storage the held factor names, so the numeric factorizations equal the
    setups. Both are returned on
    the result so the caller can carry them into the next attempt, and neither
    is rolled back by a rejected attempt.

    Args:
        problem: The solver-contract problem (residual pieces, Jacobian blocks,
            mass).
        linear: The state block's
            [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear]
            (its precision-rung routes).
        state: The carried step state; the gate memory (``nst``, ``nstlp``,
            ``nstlj``, ``gammap``, ``crate``) is read from it.
        ctx: The per-attempt inputs (endpoint time, ``theta``, ``gamma``, ``cj``,
            ``gamrat``, predicted terms, error weights, tolerance, mass class,
            reason code).
        directive: The setup policy. Its static ``refresh_jacobian`` forces a
            fresh setup (the always-fresh policy).
        loop_form: ``"fixed"`` for a bounded freeze-on-done ``fori_loop`` or
            ``"adaptive"`` for an early-exit ``while_loop``. Both use the identical
            convergence test.
        frozen_saved_jac: A supplied ``(J_f, B, C, D)`` to reuse on a
            reuse-Jacobian setup; ``None`` forces a fresh evaluation.
        frozen_pencil: A supplied ``(held, B, C, D)`` to reuse when the setup gate
            stays closed, with ``held`` the
            [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix]
            of the factored state block; ``None`` forces a fresh factorization
            every attempt (the standalone corrector path).
        generations: The carry's generation record at attempt entry, required
            alongside ``border_cache`` (the cache keys validate against these
            counters plus this attempt's own setup events).
        border_cache: The carried generation-keyed border/Schur cache for a
            problem with bordered-global rows: each setup stage revalidates
            it and rebuilds exactly the stale tiers, and every Newton
            iteration solves the bordered system by triangular solves
            against the held small factor. ``None`` rebuilds the border data
            per sweep and refactors the small system per iteration; the
            iterates are identical either way.
        newton_tol_factor: Corrector-band scale — widens the convergence
            test's tolerance by a constant factor (``1.0`` = the engine's
            standard band). The divergence test is a norm ratio and is
            unaffected.
        jacobian_values_dtype: Optional dtype for fresh Jacobian-values
            evaluations (``None`` = the state dtype).
        detect_precision: Arm the adaptive policy's failure detectors
            (increment stagnation, false convergence) and report them on the
            result; ``False`` leaves the compiled program unchanged.

    Returns:
        The
        [BDFNlsResult][diff_soup.timestepping.vsvo.engines.bdf.nonlinear.BDFNlsResult]
        for the attempt.
    """
    dtype = ctx.pred_y.dtype
    ewt = ctx.ewt
    tq4 = ctx.tq4
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
    if inexact and n_lam > 0:
        raise ConfigurationError(
            "the iterative (Krylov) corrector solves the state block of a problem without "
            f"bordered-global rows; this problem has {n_lam}. Use an exact linear backend "
            "(cudss, klu, dense) for problems with global variables."
        )
    topology = problem.state_topology
    a_shape = (n_u, n_u) if topology is None else (int(topology.rows.shape[0]),)

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

    one_f = jnp.asarray(1.0, dtype)
    two_f = jnp.asarray(2.0, dtype)
    zero_i32 = jnp.asarray(0, dtype=jnp.int32)
    one_i32 = jnp.asarray(1, dtype=jnp.int32)

    # A missing pencil / saved Jacobian leaves nothing to reuse, so the attempt
    # factorizes fresh unconditionally (the always-fresh policy and the standalone
    # corrector path).
    force_setup = bool(directive.refresh_jacobian) or (frozen_pencil is None)

    # convfail on the primary attempt: a first call or a prior error-test failure
    # carries no failure; anything else is a current-Jacobian failure.
    no_failures = (ctx.nflag == NonlinearFlag.FIRST_CALL) | (
        ctx.nflag == NonlinearFlag.PREV_ERR_FAIL
    )
    convfail_other_primary = jnp.logical_not(no_failures)

    call_setup = bdf_setup_gate(
        ctx.nflag, state.nst, state.nstlp, ctx.gamrat, refresh_jacobian=force_setup
    )
    reuse_window = fy_reuse_window(problem.plan, MSBJ)
    jbad_primary = bdf_jbad_gate(
        state.nst,
        state.nstlj,
        ctx.gamma,
        state.gammap,
        convfail_other=convfail_other_primary,
        convfail_bad_j=jnp.asarray(False),
        msbj=reuse_window,
    )
    if inexact:
        # The Jacobian is evaluated at every attempt, so only the
        # preconditioner can be stale: one retry rebuilds it, never a second.
        jbad_primary = jnp.asarray(True)
    # A retry re-enters setup with a bad-Jacobian hint; its Jacobian-recompute
    # gate is decided the same way as the primary's, never forced fresh. The
    # gate's gamma drift is measured against the gammap the PRIMARY setup left
    # behind: any setup resets gammap to the current gamma. So once the primary
    # setup ran, the first retry sees zero drift and recomputes a fresh Jacobian;
    # if the primary gate stayed closed, the retry sees the carried drift and may
    # instead rebuild the pencil from the saved Jacobian.
    gammap_after_primary = jnp.where(call_setup, ctx.gamma, state.gammap)
    jbad_retry1 = bdf_jbad_gate(
        state.nst,
        state.nstlj,
        ctx.gamma,
        gammap_after_primary,
        convfail_other=jnp.asarray(False),
        convfail_bad_j=jnp.asarray(True),
        msbj=reuse_window,
    )
    # The first retry always runs a setup, which again resets gammap to gamma, so a
    # second retry sees zero drift and always recomputes a fresh Jacobian -- the
    # exact static bound on this corrector's outer Newton loop (a fresh-Jacobian
    # attempt that fails ends the step). The drift here is measured against gamma
    # itself.
    jbad_retry2 = bdf_jbad_gate(
        state.nst,
        state.nstlj,
        ctx.gamma,
        ctx.gamma,
        convfail_other=jnp.asarray(False),
        convfail_bad_j=jnp.asarray(True),
        msbj=reuse_window,
    )

    if inexact:
        jbad_retry1 = jbad_retry2 = jnp.asarray(True)

    # Assembly runs only where a setup consumes it: the gates BRANCH to the
    # fresh-Jacobian assembly (or the cheaper saved-Jacobian pencil rebuild)
    # instead of selecting between unconditionally computed sides, so a
    # gate-closed attempt does zero Jacobian evaluations. Each setup arm also
    # reports the saved Jacobian in effect after it, so the caller persists
    # exactly what a fresh or reused setup left behind.
    gated = frozen_pencil is not None and frozen_saved_jac is not None
    if gated:
        partials_saved, b_saved, c_saved, d_saved = frozen_saved_jac
        held_frozen, b_frozen, c_frozen, d_frozen = frozen_pencil
        a_aug_frozen = held_frozen.values.reshape(a_shape)
        saved_carried = (partials_saved, b_saved, c_saved, d_saved)

        def _fresh_setup(_operand):
            a_f, parts_f, b_f, c_f, d_f = bdf_state_jacobian(problem, ctx, jacobian_values_dtype)
            return (a_f, b_f, c_f, d_f), (parts_f, b_f, c_f, d_f)

        def _saved_setup(_operand):
            if border_tracks_factor:
                a_r, b_r, c_r, d_r = _bdf_rebuild_pencil_blocks(problem, ctx, partials_saved)
                return (a_r, b_r, c_r, d_r), saved_carried
            a_r = _bdf_rebuild_pencil(problem, ctx, partials_saved)
            return (a_r, b_saved, c_saved, d_saved), saved_carried

        def _setup_blocks(use_fresh_j):
            """One setup: fresh-J assembly when ``use_fresh_j``, else a saved-J rebuild."""
            return lane_cond(use_fresh_j, _fresh_setup, _saved_setup, None)

        def _primary_setup(_op):
            pencil, saved = _setup_blocks(jbad_primary)
            return pencil, saved

        # Primary pencil: a fresh/saved setup when the gate is open, else the
        # carried frozen pencil (and the carried saved Jacobian, untouched); the
        # held factor is refactored after the branch, at every attempt.
        a_frozen_primary = a_aug_frozen
        if inexact:
            # The operator is the current matrix at every attempt; the setup
            # gate decides only the preconditioner's rebuild.
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
        # The always-fresh policy and the standalone corrector path: the
        # assembly is consumed on every attempt, so it runs unconditionally,
        # and each setup selects the fresh pencil or the same-J rebuild by
        # its own recompute decision, exactly as before.
        a_aug_fresh, parts_fresh, b_fresh, c_fresh, d_fresh = bdf_state_jacobian(
            problem, ctx, jacobian_values_dtype
        )
        a_aug_reuse = _bdf_rebuild_pencil(problem, ctx, parts_fresh)
        saved_after_primary = (parts_fresh, b_fresh, c_fresh, d_fresh)

        def _setup_blocks(use_fresh_j):
            """One setup: the fresh pencil when ``use_fresh_j``, else the rebuild."""
            a_aug = jnp.where(use_fresh_j, a_aug_fresh, a_aug_reuse)
            return (a_aug, b_fresh, c_fresh, d_fresh), saved_after_primary

        # The gate is forced open on this path (``force_setup``), so the
        # primary pencil is always this setup's output, factored from empty.
        pencil_primary, _ = _setup_blocks(jbad_primary)
        held_frozen = empty_held(linear, int(np.prod(a_shape)), dtype)

    a_primary, b_primary, c_primary, d_primary = pencil_primary

    # The gamma-drift correction rescale is unit right after a setup (gamrat reset
    # to 1) and the carried 2/(1+gamrat) otherwise.
    gamrat_primary = jnp.where(call_setup, one_f, ctx.gamrat)
    step_scale_primary = two_f / (one_f + gamrat_primary)
    if inexact:
        # The operator is at the current gamma: no stale-matrix rescale.
        step_scale_primary = one_f
    step_scale_retry = one_f  # the retry always sets up, so gamrat resets to 1.
    if inexact:
        # The Newton test's norm and threshold set the linear tolerance:
        # converged when wrms(delta, ewt / tq4) is below the band scale.
        linear_test = dict(weights=(ewt / tq4)[:n_u], newton_tol=float(newton_tol_factor))
    else:
        linear_test = {}

    # crate is reset to 1 by a setup and carried otherwise.
    crate_primary = jnp.where(call_setup, one_f, state.crate)

    def _linearize(
        held, a_block, b_block, c_block, d_block, refresh, cache, factor_key, coupling_key
    ):
        """Refactor the held state block where it was replaced; make the border/Schur data current.

        ``refresh`` is true exactly when ``a_block`` carries values replaced
        this attempt: the held factor is refactored from them, and otherwise
        the solves use the factor already held. With a cache,
        ``factor_key`` / ``coupling_key`` are the generation counters in
        effect for this stage's factor and coupling blocks: the cache
        revalidates against them and rebuilds exactly the stale tiers.
        Without one, the border solve and Schur complement are rebuilt here
        and the small system is refactored on every iteration.
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
                du_b_built = schur_built = one_i32

                def schur_solve(rhs, _schur=schur):
                    return _stable_small_solve(_schur, rhs, _SCHUR_EPS)
        else:
            du_b = jnp.zeros((n_u, 0), dtype=dtype)
            du_b_built = schur_built = zero_i32

            def schur_solve(rhs):
                return rhs

        return held, solve, du_b, schur_solve, cache, du_b_built, schur_built

    def _run_loop(solve, du_b, c_block, schur_solve, crate_init, scale, nni_init):
        """One Newton sweep with the engine's crate/dcon convergence test.

        ``nni_init`` seeds the Newton-iteration counter so it accumulates across
        the primary attempt and each retry sweep.
        """

        def newton_delta(ycor):
            resid = bdf_residual(problem, ctx, ycor)
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
            delta, resnrm, lflag, lsteps = newton_delta(carry.ycor)
            ycor_new = carry.ycor + delta
            dn = wrms_norm(delta, ewt, mask=None)
            crate_new, dcon, converged_now, diverged_now = bdf_conv_test(
                carry.m, dn, carry.delp, carry.crate, tq4, tol_factor=newton_tol_factor
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
            is_first = carry.m == 0
            acnrm_step = jnp.where(is_first, dn, wrms_norm(ycor_new, ewt, mask=None))
            acnrm_new = jnp.where(converged_now, acnrm_step, carry.acnrm)
            acnrmcur_new = jnp.where(converged_now, jnp.asarray(True), carry.acnrmcur)
            m_next = carry.m + 1
            iters_new = jnp.where(converged_now, m_next, carry.iterations)
            if detect_precision:
                stag = stagnation_flag(dn, carry.delp, converged_now)
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
                ycor=ycor_new,
                delp=dn,
                crate=crate_new,
                delnrm=dn,
                dcon=dcon,
                converged=converged_now,
                diverged=diverged_now,
                iterations=iters_new,
                acnrm=acnrm_new,
                acnrmcur=acnrmcur_new,
                nni=carry.nni + one_i32,
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
            ycor=jnp.zeros(n, dtype=dtype),
            delp=jnp.asarray(jnp.inf, dtype=dtype),
            crate=jnp.asarray(crate_init, dtype=dtype),
            delnrm=jnp.asarray(jnp.inf, dtype=dtype),
            dcon=jnp.asarray(jnp.inf, dtype=dtype),
            converged=jnp.asarray(False),
            diverged=jnp.asarray(False),
            iterations=jnp.asarray(0, dtype=jnp.int32),
            acnrm=jnp.asarray(0.0, dtype=dtype),
            acnrmcur=jnp.asarray(False),
            nni=nni_init,
            stag_count=zero_i32 if detect_precision else None,
            resnrm_first=jnp.asarray(0.0, dtype=dtype) if detect_precision else None,
            resnrm_last=jnp.asarray(0.0, dtype=dtype) if detect_precision else None,
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

            return jax.lax.fori_loop(0, NLS_MAXCOR, fori_body, init)
        if loop_form == "adaptive":

            def cond(carry: _NewtonCarry) -> jax.Array:
                return jnp.logical_and(
                    jnp.logical_not(done(carry)),
                    carry.m < jnp.asarray(NLS_MAXCOR, dtype=carry.m.dtype),
                )

            return jax.lax.while_loop(cond, take_step, init)
        raise ValueError(f"loop_form must be 'fixed' or 'adaptive', got {loop_form!r}.")

    # Generation counters in effect for each setup stage. Every setup
    # replaces the factor once (a fresh Jacobian additionally replaces the
    # coupling blocks) BEFORE its sweep consumes them, and the caller
    # advances the carried record by exactly the setup / fresh-Jacobian
    # counts this attempt reports — so the keys stored in the cache equal
    # the record the next attempt validates against.
    if caching:
        if border_tracks_factor:
            coupling_bump0 = call_setup
            coupling_bump1 = jnp.asarray(True)
            coupling_bump2 = jnp.asarray(True)
        else:
            coupling_bump0 = jnp.logical_and(call_setup, jbad_primary)
            coupling_bump1 = jbad_retry1
            coupling_bump2 = jbad_retry2
        factor_key0 = generations.factor + call_setup.astype(jnp.int32)
        coupling_key0 = generations.b + coupling_bump0.astype(jnp.int32)
        factor_key1 = factor_key0 + one_i32
        coupling_key1 = coupling_key0 + coupling_bump1.astype(jnp.int32)
        factor_key2 = factor_key1 + one_i32
        coupling_key2 = coupling_key1 + coupling_bump2.astype(jnp.int32)
    else:
        factor_key0 = coupling_key0 = zero_i32
        factor_key1 = coupling_key1 = zero_i32
        factor_key2 = coupling_key2 = zero_i32
    cache_in = border_cache if caching else None

    held1, solve0, du_b0, schur_solve0, cache1, dub0, sb0 = _linearize(
        held_frozen,
        a_primary,
        b_primary,
        c_primary,
        d_primary,
        call_setup,
        cache_in,
        factor_key0,
        coupling_key0,
    )
    pencil_held1 = (held1, b_primary, c_primary, d_primary)
    carry1 = _run_loop(
        solve0, du_b0, c_primary, schur_solve0, crate_primary, step_scale_primary, zero_i32
    )

    # The Jacobian is current after the primary only if a setup ran and it
    # re-evaluated J. A recoverable failure taken with a non-current Jacobian
    # re-enters setup with a bad-Jacobian hint. Because a fresh-Jacobian setup
    # marks the Jacobian current, a
    # fresh-Jacobian retry that fails ends the attempt: at most two retries can
    # run (the outer loop's exact static bound).
    jcur_primary = jnp.logical_and(call_setup, jbad_primary)
    ran_retry1 = jnp.logical_and(jnp.logical_not(carry1.converged), jnp.logical_not(jcur_primary))

    # Each retry always runs a setup (fresh or saved rebuild) and reports the
    # pencil and saved Jacobian it left in effect, so the branch results carry
    # exactly what the caller must persist.
    def _retry1(_operand):
        if inexact:
            # The matrix is already current: the retry only rebuilds the
            # preconditioner from it.
            pencil_r, saved_r = pencil_primary, saved_after_primary
        else:
            pencil_r, saved_r = _setup_blocks(jbad_retry1)
        a_r, b_r, c_r, d_r = pencil_r
        # The refresh flag holds wherever this branch's result is kept; a lane
        # of a batch that runs the branch without needing it keeps its held
        # factor.
        held_r, solve_r, du_b_r, schur_solve_r, cache_r, dub_r, sb_r = _linearize(
            held1, a_r, b_r, c_r, d_r, ran_retry1, cache1, factor_key1, coupling_key1
        )
        carry_r = _run_loop(
            solve_r, du_b_r, c_r, schur_solve_r, one_f, step_scale_retry, carry1.nni
        )
        return carry_r, (held_r, b_r, c_r, d_r), saved_r, cache_r, dub_r, sb_r

    carry2, pencil_after1, saved_after1, cache_after1, dub1, sb1 = lane_cond(
        ran_retry1,
        _retry1,
        lambda _op: (carry1, pencil_held1, saved_after_primary, cache1, zero_i32, zero_i32),
        None,
        false_is_free=True,
    )

    # A second retry runs only when the first both failed recoverably and left the
    # Jacobian stale (a saved-Jacobian rebuild, jcur == FALSE). A retry always runs
    # a setup, so its Jacobian is current exactly when that setup re-evaluated J.
    jcur_retry1 = jbad_retry1
    ran_retry2 = jnp.logical_and(
        ran_retry1,
        jnp.logical_and(jnp.logical_not(carry2.converged), jnp.logical_not(jcur_retry1)),
    )

    def _retry2(_operand):
        pencil_r, saved_r = _setup_blocks(jbad_retry2)
        a_r, b_r, c_r, d_r = pencil_r
        # As the first retry: a lane that runs the branch without needing it
        # keeps its held factor, and the refactor takes over the held storage.
        held_r, solve_r, du_b_r, schur_solve_r, cache_r, dub_r, sb_r = _linearize(
            pencil_after1[0],
            a_r,
            b_r,
            c_r,
            d_r,
            ran_retry2,
            cache_after1,
            factor_key2,
            coupling_key2,
        )
        carry_r = _run_loop(
            solve_r, du_b_r, c_r, schur_solve_r, one_f, step_scale_retry, carry2.nni
        )
        return carry_r, (held_r, b_r, c_r, d_r), saved_r, cache_r, dub_r, sb_r

    carry3, pencil_after2, saved_after2, cache_after2, dub2, sb2 = lane_cond(
        ran_retry2,
        _retry2,
        lambda _op: (carry2, pencil_after1, saved_after1, cache_after1, zero_i32, zero_i32),
        None,
        false_is_free=True,
    )
    du_b_builds = dub0 + dub1 + dub2
    schur_builds = sb0 + sb1 + sb2

    # carry3 folds through unchanged when a retry did not run, so it already holds
    # the final iterate, the convergence flags, and the cumulative Newton count.
    converged = carry3.converged
    diverged = carry3.diverged
    ycor = carry3.ycor
    iterations = carry3.iterations
    delnrm = carry3.delnrm
    dcon = carry3.dcon
    crate_out = carry3.crate
    acnrmcur = carry3.acnrmcur
    acnrm_loop = carry3.acnrm
    nni = carry3.nni

    nsetups = (
        call_setup.astype(jnp.int32) + ran_retry1.astype(jnp.int32) + ran_retry2.astype(jnp.int32)
    )
    # A fresh Jacobian is evaluated by whichever setup recomputed it. At most one
    # of these fires: a fresh-J primary marks the Jacobian current and blocks both
    # retries, and a fresh-J first retry blocks the second.
    fresh_j_primary = jnp.logical_and(call_setup, jbad_primary)
    fresh_j_retry1 = jnp.logical_and(ran_retry1, jbad_retry1)
    fresh_j_retry2 = jnp.logical_and(ran_retry2, jbad_retry2)
    nje = (
        fresh_j_primary.astype(jnp.int32)
        + fresh_j_retry1.astype(jnp.int32)
        + fresh_j_retry2.astype(jnp.int32)
    )
    if inexact:
        # The Jacobian is evaluated once per attempt (a retry reuses it).
        nje = jnp.ones_like(nsetups)
    # Newton convergence failures counted this attempt: one per retry that ran,
    # plus one more if the whole attempt still failed.
    nnf = (
        ran_retry1.astype(jnp.int32)
        + ran_retry2.astype(jnp.int32)
        + jnp.logical_not(converged).astype(jnp.int32)
    )
    any_setup = jnp.logical_or(call_setup, jnp.logical_or(ran_retry1, ran_retry2))
    any_fresh_j = jnp.logical_or(fresh_j_primary, jnp.logical_or(fresh_j_retry1, fresh_j_retry2))
    jcur_out = jnp.where(ran_retry2, jbad_retry2, jnp.where(ran_retry1, jcur_retry1, jcur_primary))

    # The correction norm falls back to the full correction norm when the
    # convergence test did not set it.
    acnrm = jnp.where(acnrmcur, acnrm_loop, wrms_norm(ycor, ewt, mask=None))

    # A setup resets gamrat to 1, gammap to gamma, and nstlp to nst; a fresh
    # Jacobian additionally advances nstlj.
    gamrat_out = jnp.where(any_setup, one_f, ctx.gamrat)
    gammap_out = jnp.where(any_setup, ctx.gamma, state.gammap)
    nstlp_out = jnp.where(any_setup, state.nst, state.nstlp)
    nstlj_out = jnp.where(any_fresh_j, state.nst, state.nstlj)

    # The pencil and saved Jacobian in effect at exit: the last setup's output
    # when a retry fired, else the primary's — already threaded through the
    # branch results above (the saved Jacobian changes only on a fresh-J
    # setup; a saved-rebuild or gate-closed stage carries it through).
    pencil_out = pencil_after2
    saved_jac_out = saved_after2

    y = ctx.pred_y + ycor
    if detect_precision:
        prec_stagnant = (carry3.stag_count >= STAGNATION_LATCH) & jnp.logical_not(converged)
        prec_false_conv = (
            converged & (carry3.iterations >= 2) & (carry3.resnrm_last >= carry3.resnrm_first)
        )
    else:
        prec_stagnant = prec_false_conv = None

    return BDFNlsResult(
        acor=ycor,
        y=y,
        converged=converged,
        diverged=diverged,
        iterations=iterations,
        delnrm=delnrm,
        crate=crate_out,
        acnrm=acnrm,
        acnrmcur=acnrmcur,
        dcon=dcon,
        call_setup=call_setup,
        jbad=jbad_primary,
        jcur=jcur_out,
        nsetups=nsetups,
        nje=nje,
        nni=nni,
        nnf=nnf,
        gamrat_out=gamrat_out,
        gammap_out=gammap_out,
        nstlp_out=nstlp_out,
        nstlj_out=nstlj_out,
        saved_jac_out=saved_jac_out,
        pencil_out=pencil_out,
        prec_stagnant=prec_stagnant,
        prec_false_conv=prec_false_conv,
        lin_fallbacks=carry3.lin_fallbacks,
        lin_ir_steps=carry3.lin_ir_steps,
        border_cache_out=cache_after2,
        du_b_builds=du_b_builds,
        schur_builds=schur_builds,
        lin_failures=carry3.lin_failures,
        krylov_iters=carry3.krylov_iters,
    )


def bdf_corrector_root(
    problem,
    backend,
    linear,
    sensitivity_analysis,
    state,
    ctx: BDFNlsContext,
    directive: LinearizationDirective,
    loop_form: str,
    frozen_saved_jac: Optional[tuple] = None,
    frozen_pencil: Optional[tuple] = None,
    *,
    reverse_analysis=None,
    generations: Optional[Generations] = None,
    border_cache=None,
    newton_tol_factor: float = 1.0,
    jacobian_values_dtype=None,
    detect_precision: bool = False,
) -> BDFNlsResult:
    """One corrector attempt with the accepted root behind its derivative boundary.

    Runs
    [bdf_corrector][diff_soup.timestepping.vsvo.engines.bdf.nonlinear.bdf_corrector]
    as the primal solve and binds the corrected
    state to the implicit-function rule at that root: the attempt solves the
    residual ``F(t, y, cj*y + q, theta) = 0`` with the history
    aggregate ``q = cj * (rl1*zn[1] - pred_y)`` (the corrector's slope
    estimate ``cj*(rl1*zn[1] + ycor)`` written in terms of the corrected
    state), so the corrected state and correction carry the exact root
    derivative while every primal bookkeeping output (convergence flags,
    counters, rate memory, setup decisions, carried matrix blocks) stays
    gradient-dead. ``linear`` is the state block's
    [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear],
    ``sensitivity_analysis`` the ``backend``'s forward-direction analysis of
    the problem's full-system pattern (``None`` on the dense path), which
    serves the tangent solve, and ``reverse_analysis`` its reverse-direction
    analysis, which serves the transposed solve of a reverse sweep (``None``
    reuses ``sensitivity_analysis``). The primal iterates are
    bit-identical to a raw
    [bdf_corrector][diff_soup.timestepping.vsvo.engines.bdf.nonlinear.bdf_corrector]
    call, so a non-converged attempt reports exactly the raw corrector's
    failure bookkeeping.

    ``generations`` / ``border_cache`` carry the generation-keyed
    border/Schur cache into the primal solve (see
    [bdf_corrector][diff_soup.timestepping.vsvo.engines.bdf.nonlinear.bdf_corrector]);
    the cache never reaches the sensitivity rules, which rebuild exact
    values at the accepted root.
    """

    def corrector_at(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector at the reconstructed context (never differentiated).

        Every traced solver value (setup-gate memory, error weights, carried
        matrix blocks, gate scalars, precision-rung state) arrives through
        ``primal_args`` so this closure captures only static configuration.
        ``q`` seeds only the derivative rule: the exact predicted-derivative
        term rides ``primal_args``, keeping the primal bit-identical to the
        raw corrector.
        """
        del q
        (
            p_state,
            ewt,
            p_saved,
            p_pencil,
            p_generations,
            p_border_cache,
            prec_rung,
            rung_changed,
            gamma,
            gamrat,
            rl1_zn1,
            tq4,
            nflag,
        ) = primal_args
        ctx_p = BDFNlsContext(
            t=t,
            theta=theta,
            gamma=gamma,
            cj=cj,
            gamrat=gamrat,
            rl1_zn1=rl1_zn1,
            pred_y=guess,
            ewt=ewt,
            tq4=tq4,
            nflag=nflag,
            prec_rung=prec_rung,
            rung_changed=rung_changed,
            control=control,
        )
        return bdf_corrector(
            problem,
            linear,
            p_state,
            ctx_p,
            directive,
            loop_form,
            frozen_saved_jac=p_saved,
            frozen_pencil=p_pencil,
            generations=p_generations,
            border_cache=p_border_cache,
            newton_tol_factor=newton_tol_factor,
            jacobian_values_dtype=jacobian_values_dtype,
            detect_precision=detect_precision,
        )

    def primal_fn(t, cj, q, theta, control, guess, primal_args):
        """Run the corrector and expose its converged root to the derivative rule."""
        res = corrector_at(t, cj, q, theta, control, guess, primal_args)
        return res.y, res.converged, res

    q = ctx.cj * (ctx.rl1_zn1 - ctx.pred_y)
    primal_args = (
        state,
        ctx.ewt,
        frozen_saved_jac,
        frozen_pencil,
        generations,
        border_cache,
        ctx.prec_rung,
        ctx.rung_changed,
        ctx.gamma,
        ctx.gamrat,
        ctx.rl1_zn1,
        ctx.tq4,
        ctx.nflag,
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
        prec_rung = primal_args[6]
        fy_saved, fydot_saved = aux.saved_jac_out[0]
        return problem.iteration_matrix_offering(
            fy_saved,
            fydot_saved,
            cj,
            _bdf_root_success(aux, prec_rung, detect_precision),
        )

    y, _ok, aux = problem.bind_step_root(
        backend,
        sensitivity_analysis,
        primal_fn,
        ctx.t,
        ctx.cj,
        q,
        ctx.theta,
        ctx.control,
        ctx.pred_y,
        primal_args,
        reverse_analysis=reverse_analysis,
        certificate_extractor=offering_extractor,
    )
    return aux._replace(acor=y - ctx.pred_y, y=y)


def _bdf_root_success(res: BDFNlsResult, prec_rung, detect_precision: bool) -> jax.Array:
    """The exact-at-root predicate that gates the iteration-matrix offering.

    The offered values may be consumed only when they were evaluated exactly
    at the accepted root, so the predicate requires all of: the corrector
    converged (minus a reduced-precision false convergence when the detectors
    are armed); the primary setup evaluated a fresh Jacobian this attempt (a
    window-reused or recomposed matrix carries values from an earlier
    trajectory point); no fresh-Jacobian retry ran; and the total applied
    correction is exactly zero, so the evaluation point provably is the root.
    Under-issuance is always safe — a rejected offering rebuilds the exact
    matrix — while over-issuance would let the sensitivity solve consume values
    assembled away from the root. ``prec_rung`` is the attempt's precision rung,
    read only when the precision detectors are armed.
    """
    at_root = jnp.all(res.acor == 0.0)
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
    "BDFNlsContext",
    "BDFNlsResult",
    "bdf_residual",
    "bdf_state_jacobian",
    "bdf_setup_gate",
    "bdf_jbad_gate",
    "bdf_conv_test",
    "bdf_corrector",
    "bdf_corrector_root",
    "NLS_MAXCOR",
    "CRDOWN",
    "RDIV",
    "MSBP",
    "DGMAX_LSETUP",
    "MSBJ",
    "DGMAX_JBAD",
]
