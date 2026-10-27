"""Euler-Newton consistent initial conditions.

This is the minimal "Euler-Newton without line search" consistent-initialization
subset used by the IDA-class engine. Two modes:

``ALG_AND_YDOT``
    Solve the algebraic components of ``y`` and the differential components of
    ``ydot`` at leading coefficient ``cj = 1/hic``.
``ALL_Y_GIVEN_YDOT``
    Solve all of ``y`` at ``cj = 0`` given a fixed ``ydot``.

The step is one linear setup plus a damped-free Newton iteration: the Newton
matrix is the shared linearization ``F_y + cj * F_ydot`` with the
bordered-global rows routed through a Schur complement, and convergence is the
weighted-RMS norm of the Newton step ``J^{-1} F`` falling to or below
``ic_newton_tol``. Line search, fresh-Jacobian retry, step retry, two-pass weight
refresh, and constraint scaling are not part of this subset. The solve is
differentiable through the converged root by the implicit-function rule (see
[diff_soup.timestepping.consistent_ic.ift][]).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.nonlinear._kkt import _cast_like
from diff_soup.nonlinear.solver import _factor_state_block, _stable_small_solve
from diff_soup.timestepping._shared.robustness.derivative_ic import (
    recover_algebraic_derivative,
)
from diff_soup.timestepping.consistent_ic.ift import consistent_project
from diff_soup.timestepping.consistent_ic.pin_differential import (
    ContractOperator,
    ICConfig,
    ICMode,
    ICResult,
    ICStepContext,
    _alg_residual_norm,
    _ic_analysis,
    _ic_backend,
    _ic_topology,
    _nonlinear_config,
    _pin_operator,
    _pin_state_primal,
    _seed_ydot_diff,
    _structure,
)
from diff_soup.timestepping.vsvo.common.weights import error_weight, wrms_norm
from diff_soup.utils import safe_div

__all__ = ["euler_newton_ic", "euler_newton_setup", "ICConfig", "ICMode", "ICResult"]

# No first output time reaches this standalone entry, so the consistent-
# initialization time scale is a nominal unit interval. The converged root is
# independent of this scale (it only sets hic / cj, which drive the iteration,
# not the manifold it lands on).
_NOMINAL_TDIST = 1.0


class EulerSetup(NamedTuple):
    """The consistent-initialization setup scalars materialized for one solve.

    Attributes:
        tdist: The time scale ``|t1 - t0|`` (a nominal unit interval here).
        hic: The Euler step ``hic_frac * tdist`` after the ``ypnorm`` cap.
        cj: The leading coefficient ``1/hic`` (``ALG_AND_YDOT``) or ``0``
            (``ALL_Y_GIVEN_YDOT``).
        sysindex: ``0`` for a pure-differential ``ALG_AND_YDOT`` system (which
            triggers the ``tscale * |cj|`` norm rescale), else ``1``.
        tscale: The norm rescale time base (``tdist``).
        ewt: The weighted-RMS error-weight vector at the initial state.
        ypnorm: The masked weighted-RMS norm of the initial derivative.
    """

    tdist: Any
    hic: Any
    cj: Any
    sysindex: int
    tscale: Any
    ewt: Any
    ypnorm: Any


def _mode_is_alg(mode: ICMode) -> bool:
    """Whether ``mode`` is the algebraic-state / differential-derivative solve."""
    return mode is ICMode.ALG_AND_YDOT


def euler_newton_setup(problem, y0, ydot0, lam0, theta, mode, cfg, t0, t1=None):
    """Materialize the consistent-initialization setup scalars (``hic``, ``cj``, ``sysindex`` ...).

    The Euler step starts at ``hic_frac * tdist`` and is capped by the derivative
    norm (``if ypnorm > 0.5/hic: hic = 0.5/ypnorm``). The leading coefficient is
    ``cj = 1/hic`` for ``ALG_AND_YDOT`` and ``cj = 0`` for ``ALL_Y_GIVEN_YDOT``.

    ``tdist`` is the integration time scale ``|t1 - t0|``. Pass a finite ``t1`` to
    use the real interval; ``None`` (a standalone call with no target time) or a
    non-finite endpoint falls back to a nominal unit interval. Only ``hic``/``cj``
    depend on it, so the converged root is unchanged either way; the real scale
    sizes the Euler step to the actual integration interval.
    """
    n_u, n_lam, _diff_mask, _alg_index, _needs = _structure(problem)
    y0 = jnp.asarray(y0)
    ydot0 = jnp.asarray(ydot0)
    lam0 = jnp.asarray(lam0)
    y_aug = jnp.concatenate([y0, lam0])
    # Seed only: zeros over the full bordered partition are a Newton seed for
    # this solve, not a role statement — differential-global derivatives are
    # recovered by the role-masked slope fill downstream.
    ydot_aug = jnp.concatenate([ydot0, jnp.zeros((n_lam,), dtype=ydot0.dtype)])

    id_mask = jnp.asarray(problem.id_mask)
    ewt = error_weight(y_aug, cfg.rtol, cfg.atol)
    ypnorm = wrms_norm(ydot_aug, ewt, mask=id_mask)

    nominal = jnp.asarray(_NOMINAL_TDIST, dtype=y_aug.dtype)
    if t1 is None:
        tdist = nominal
    else:
        # A non-finite endpoint sentinel (``+inf``) means "no endpoint": fall
        # back to the nominal unit time scale. A finite endpoint sizes the
        # iteration to ``|t1 - t0|``. The reverse pass stays finite because the
        # ``abs`` branch's derivative is ``sign`` (bounded) and ``where`` zeros
        # that branch's cotangent on the sentinel lane.
        t1a = jnp.asarray(t1, dtype=y_aug.dtype)
        t0a = jnp.asarray(t0, dtype=y_aug.dtype)
        tdist = jnp.where(jnp.isfinite(t1a), jnp.abs(t1a - t0a), nominal)
    hic = jnp.asarray(cfg.hic_frac, dtype=y_aug.dtype) * tdist
    hic = jnp.where(ypnorm > 0.5 / hic, safe_div(jnp.asarray(0.5, dtype=y_aug.dtype), ypnorm), hic)

    is_alg = _mode_is_alg(mode)
    cj = 1.0 / hic if is_alg else jnp.asarray(0.0, dtype=y_aug.dtype)
    # Pure-differential ALG_AND_YDOT system (index 0): all rows differential.
    pure_diff = bool(np.min(np.asarray(problem.id_mask)) > 0.5)
    sysindex = 0 if (is_alg and pure_diff) else 1
    tscale = tdist
    return EulerSetup(
        tdist=tdist, hic=hic, cj=cj, sysindex=sysindex, tscale=tscale, ewt=ewt, ypnorm=ypnorm
    )


def _euler_state_jacobian(problem, u, lam, theta, ctx, cj, topology):
    """Assemble the fully-implicit IC Newton state block for the Euler step.

    The data-dependent ``cj`` is passed to the shared plan at the current
    iterate. Returns the state-block value vector over ``topology`` plus the
    ``(B, C, D)`` global coupling blocks in the Schur convention.
    """
    y = jnp.concatenate([u, lam])
    a, b, c, d = problem.state_jacobian_blocks(ctx.t0, y, ctx.ydot, theta, cj)
    a_values = jnp.reshape(a, (-1,)) if jnp.ndim(a) == 2 else jnp.asarray(a)
    return a_values, b, c, d


def _euler_state_solve(
    problem,
    y_aug0,
    ydot_aug0,
    theta,
    t0,
    cfg,
    setup,
    mode,
    topology,
    backend,
    *,
    layout=None,
    precision=None,
):
    """Run the Euler-Newton consistent-initialization loop; return the converged ``(u, lam)``.

    One linear setup builds the Schur-reduced ``F_y + cj F_ydot`` factor at the
    initial iterate; each Newton step recomputes the fully-implicit residual and
    solves ``delta = J^{-1} F``. The update follows the mode's row semantics
    (``ALG_AND_YDOT``: algebraic ``y`` and differential ``ydot``;
    ``ALL_Y_GIVEN_YDOT``: all ``y`` with ``ydot`` fixed), with a unit step (the
    line-search-off path). Convergence is ``wrms(delta) <= ic_newton_tol``. The
    differential derivative the iteration solves is not returned: at the converged
    root it equals ``M^{-1} f`` on the differential rows, which the caller
    recomputes from the projected state so the returned ``ydot`` differentiates
    through the root rather than through the loop. ``topology``/``backend`` are the
    state pattern and linear solver the caller factors the block on, and
    ``layout``/``precision`` are the placement and dtype policy that block is
    analyzed with (each ``None`` keeps the backend's own default).
    """
    is_alg = _mode_is_alg(mode)
    n_u, n_lam, _dm, _ai, _np = _structure(problem)
    analysis = _ic_analysis(backend, topology, layout=layout, precision=precision)

    u0 = y_aug0[:n_u]
    lam0 = y_aug0[n_u:]
    ctx0 = ICStepContext.make(t0=t0, ydot=ydot_aug0, n_u=n_u)
    a_vals, b, c, d = _euler_state_jacobian(problem, u0, lam0, theta, ctx0, setup.cj, topology)
    held = _factor_state_block(backend, analysis, a_vals)
    m = int(d.shape[0])
    if m > 0:
        du_b = _cast_like(held.solve(b.T), b).T
        schur = d - c @ du_b
    else:
        du_b = None
        schur = None

    id_mask = jnp.asarray(problem.id_mask)
    ewt = setup.ewt
    cj = setup.cj
    ic_newton_tol = jnp.asarray(cfg.ic_newton_tol, dtype=y_aug0.dtype)
    tscale_cj = setup.tscale * jnp.abs(cj)

    def _delta_and_fnorm(y_aug, ydot_aug):
        """Solve ``delta = J^{-1} F`` and its convergence norm at the iterate."""
        resid = problem.residual(t0, y_aug, ydot_aug, theta)
        r = resid[:n_u]
        g = resid[n_u:]
        du_r = _cast_like(held.solve(r), r)
        if m > 0:
            dlam = _stable_small_solve(schur, g - c @ du_r, cfg.schur_eps)
            du = du_r - du_b @ dlam
        else:
            dlam = jnp.zeros((0,), dtype=y_aug.dtype)
            du = du_r
        delta = jnp.concatenate([du, dlam])
        fnorm = wrms_norm(delta, ewt, mask=None)
        if setup.sysindex == 0:
            fnorm = fnorm * tscale_cj
        return delta, fnorm

    delta0, fnorm0 = _delta_and_fnorm(y_aug0, ydot_aug0)
    done0 = fnorm0 <= ic_newton_tol
    init = (y_aug0, ydot_aug0, delta0, fnorm0, done0, jnp.asarray(0, dtype=jnp.int32))

    def body(_i, carry):
        y_aug, ydot_aug, delta, fnorm, done, nit = carry
        if is_alg:
            y_new = y_aug - jnp.where(id_mask == 0, delta, 0.0)
            ydot_new = ydot_aug - jnp.where(id_mask == 1, cj * delta, 0.0)
        else:
            y_new = y_aug - delta
            ydot_new = ydot_aug
        delta_new, fnorm_new = _delta_and_fnorm(y_new, ydot_new)
        conv = fnorm_new <= ic_newton_tol
        y_out = jnp.where(done, y_aug, y_new)
        ydot_out = jnp.where(done, ydot_aug, ydot_new)
        delta_out = jnp.where(done, delta, delta_new)
        fnorm_out = jnp.where(done, fnorm, fnorm_new)
        nit_out = jnp.where(done, nit, nit + 1)
        done_out = jnp.logical_or(done, conv)
        return (y_out, ydot_out, delta_out, fnorm_out, done_out, nit_out)

    y_final, _ydot_final, _delta, _fnorm, _done, _nit = jax.lax.fori_loop(
        0, int(cfg.max_newton_iters), body, init
    )
    return y_final[:n_u], y_final[n_u:]


def _reproject_at(
    problem, theta, t_eps, n_u, n_lam, diff_mask, cfg, backend, *, layout=None, precision=None
):
    """Return a ``(u_seed, lam_seed) -> (u_eps, lam_eps)`` pin projection at ``t_eps``."""

    def reproject(u_seed, lam_seed):
        operator, be, analysis = _pin_operator(
            problem, n_u, n_lam, diff_mask, backend, layout=layout, precision=precision
        )
        ctx = ICStepContext.make(
            t0=t_eps, ydot=jnp.zeros((n_u + n_lam,), dtype=u_seed.dtype), n_u=n_u
        )
        ncfg = _nonlinear_config(cfg)
        return _pin_state_primal(
            operator, be, analysis, u_seed, lam_seed, theta, ctx, ncfg, diff_mask
        )

    return reproject


def _euler_ydot(
    problem,
    u,
    lam,
    ydot0,
    theta,
    t0,
    cfg,
    n_u,
    n_lam,
    diff_mask,
    mode,
    backend,
    *,
    layout=None,
    precision=None,
):
    """Assemble the returned ``ydot`` from the projected root per the row semantics.

    ``ALL_Y_GIVEN_YDOT`` returns the input derivative unchanged. ``ALG_AND_YDOT``
    returns the differential rows the iteration solves -- recomputed through the
    view's startup operation so the derivative rides
    the differentiable root -- with the algebraic rows passed through from the
    input, unless the finite-difference recovery is opted in (available only for
    ``ALG_AND_YDOT``; there is no derivative to recover when every ``y`` was solved
    against a fixed derivative).
    """
    if mode is ICMode.ALL_Y_GIVEN_YDOT:
        return jnp.asarray(ydot0)
    udot_diff_full = _seed_ydot_diff(
        problem,
        u,
        lam,
        theta,
        t0,
        n_u,
        tol=cfg.atol,
        backend=backend,
        layout=layout,
        precision=precision,
    )
    if not cfg.recover_algebraic_ydot:
        return jnp.where(diff_mask, udot_diff_full, jnp.asarray(ydot0))
    # ALG_AND_YDOT opt-in: fill the local- and global-algebraic derivative rows
    # by an Euler probe + finite difference across the manifold.
    #
    # Both finite-difference endpoints must come from the SAME projection so their
    # per-solve residuals cancel; the 1e-8 step otherwise amplifies the mismatch
    # between the euler-loop state and a differently-conditioned probe. Re-project
    # the baseline with the pin machinery the probe uses, then difference.
    eps_du = 1e-8
    reproject = _reproject_at(
        problem,
        theta,
        t0 + eps_du,
        n_u,
        n_lam,
        diff_mask,
        cfg,
        backend,
        layout=layout,
        precision=precision,
    )
    u_proj, lam_proj = _reproject_at(
        problem,
        theta,
        t0,
        n_u,
        n_lam,
        diff_mask,
        cfg,
        backend,
        layout=layout,
        precision=precision,
    )(u, lam)
    udot_diff_full = _seed_ydot_diff(
        problem,
        u_proj,
        lam_proj,
        theta,
        t0,
        n_u,
        tol=cfg.atol,
        backend=backend,
        layout=layout,
        precision=precision,
    )
    ydot, lamdot = recover_algebraic_derivative(
        u_proj=u_proj,
        lam_proj=lam_proj,
        udot_diff_full=udot_diff_full,
        alg_mask=jnp.logical_not(diff_mask),
        reproject=reproject,
        eps_du=eps_du,
    )
    del lamdot
    return ydot


def euler_newton_ic(
    problem,
    y0,
    ydot0,
    lam0,
    theta,
    mode,
    cfg,
    t0,
    *,
    t1=None,
    backend=None,
    layout=None,
    precision=None,
):
    """Solve consistent initial conditions by the Euler-Newton subset.

    In ``ALG_AND_YDOT`` mode the algebraic components of ``y0`` and the
    differential components of ``ydot0`` are solved; in ``ALL_Y_GIVEN_YDOT`` mode
    all of ``y0`` is solved with ``ydot0`` held fixed at ``cj = 0``. Every
    differentiation mode differentiates through the converged root by the
    implicit-function rule, never through the Newton loop.

    ``t1`` sets the integration time scale ``tdist = |t1 - t0|`` used to size the
    Euler step; ``None`` falls back to a nominal unit interval. The manifold the
    solve lands on is independent of ``tdist`` (it only shapes the iteration), so
    passing the real ``t1`` sizes the iteration to the actual integration interval
    without moving the root. ``backend`` is the linear solver the solve factors on
    (the engine passes its resolver-selected sparse backend); it defaults to the
    dense backend. ``layout`` and ``precision`` are the placement and dtype policy
    every linear analysis of this solve is built with -- an engine passes the
    layout its problem was lowered with and the precision it takes accepted-root
    derivatives at, so this projection factors where and at the dtype the rest of
    the solve does; each left ``None`` keeps the backend's own default for it.
    """
    if cfg.linesearch:
        raise ValueError(
            "line search is not available in the Euler-Newton consistent-IC solve; "
            "the damped-free Newton step (linesearch=False) is the available path."
        )

    n_u, n_lam, diff_mask, alg_u_index, _needs = _structure(problem)
    y0 = jnp.asarray(y0)
    ydot0 = jnp.asarray(ydot0)
    lam0 = jnp.asarray(lam0)
    t0 = jnp.asarray(t0, dtype=y0.dtype)
    is_alg = _mode_is_alg(mode)

    ydot_aug0 = jnp.concatenate([ydot0, jnp.zeros((n_lam,), dtype=ydot0.dtype)])
    # ``t1`` rides the context (a differentiable argument of the projection),
    # never the ``primal_fn`` closure: the closure crosses the projection's
    # non-differentiable argument seam, and a traced endpoint captured there
    # leaks across trace levels under ``jvp``-of-``jit``.
    ctx = ICStepContext.make(t0=t0, ydot=ydot_aug0, n_u=n_u, t1=t1)

    # The implicit-function boundary linearizes the fixed-derivative manifold
    # residual at the root (``cj = 0``). Differential rows are pinned in the
    # algebraic-state mode.
    topology = _ic_topology(problem, n_u)
    base_op = ContractOperator(
        problem=problem,
        n_u=n_u,
        n_lam=n_lam,
        topology=topology,
        cj=0.0,
    )
    backend = _ic_backend(backend)
    analysis = _ic_analysis(backend, topology, layout=layout, precision=precision)
    ncfg = _nonlinear_config(cfg)
    pin_kind = "differential" if is_alg else "none"

    def primal_fn(u0, lam0_, theta_, ctx_):
        """Forward value: the Euler-Newton consistent-initialization solve for ``(u, lam)``."""
        setup = euler_newton_setup(
            problem, u0, ctx_.ydot[:n_u], lam0_, theta_, mode, cfg, ctx_.t0, t1=ctx_.t1
        )
        y_aug0 = jnp.concatenate([u0, lam0_])
        return _euler_state_solve(
            problem,
            y_aug0,
            ctx_.ydot,
            theta_,
            ctx_.t0,
            cfg,
            setup,
            mode,
            topology,
            backend,
            layout=layout,
            precision=precision,
        )

    u, lam = consistent_project(
        base_op, backend, analysis, ncfg, pin_kind, primal_fn, y0, lam0, theta, ctx
    )

    ydot = _euler_ydot(
        problem,
        u,
        lam,
        ydot0,
        theta,
        t0,
        cfg,
        n_u,
        n_lam,
        diff_mask,
        mode,
        backend,
        layout=layout,
        precision=precision,
    )
    alg_norm = _alg_residual_norm(problem, u, lam, ydot, theta, t0, n_u, alg_u_index)
    return ICResult(y=u, lam=lam, ydot=ydot, alg_residual_norm=alg_norm)
