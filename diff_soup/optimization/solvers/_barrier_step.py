"""One step of the interior point, in the regular phase or the restoration phase.

The step solves the primal-dual KKT system of the phase's barrier problem, finds the fraction
to the boundary, runs the filter line search with second-order corrections, moves the
multipliers and evaluates the problem once at the accepted point. The phase flag selects the
restoration problem's system, merit function and multiplier steps, so the problem's closures
appear once in the loop body. A failed line search in the regular phase starts the
restoration phase; an iterate the original filter accepts ends it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.solvers import _filter, _hessian, _restoration
from diff_soup.optimization.solvers._barrier import (
    KAPPA_SIGMA,
    Bounds,
    bound_multiplier_steps,
    dual_fraction_to_boundary,
    fraction_to_boundary,
    primal_sigma,
    safeguard_multipliers,
)
from diff_soup.optimization.solvers._barrier_problem import (
    InteriorPointState,
    merit,
    phase_error,
    regular_error,
    terms,
)
from diff_soup.optimization.solvers._closures import Closures, evaluate, trial
from diff_soup.optimization.solvers._core import lagrangian_secant
from diff_soup.optimization.solvers._line_search import Correction, filter_line_search
from diff_soup.optimization.solvers._protocol import _int, _select
from diff_soup.utils.lanes import live_lane_cond

#: Ipopt's `jacobian_regularization_value` and `jacobian_regularization_exponent`:
#: `delta_c = DELTA_C * mu**KAPPA_C` for a rank-deficient constraint Jacobian.
DELTA_C = 1e-8
KAPPA_C = 0.25


def _hessian_of(solver, closures, state, args, restores):
    """The primal block's curvature: the Lagrangian Hessian, or in restoration that of the rows alone.

    The constraint curvature is the problem's `constraint_hessian` closure, else the difference
    of the Lagrangian Hessian at the multipliers and at zero, whose second term runs only in
    restoration. A quasi-Newton restoration phase learns it in its own matrix.
    """
    x, lam_e, lam_i, ev = state.x, state.lam_e, state.lam_i, state.evaluation
    mode = solver.hessian
    regular = lambda: _hessian.lagrangian_hessian(  # noqa: E731
        mode, closures, x, lam_e, lam_i, args, ev, state.quasi_newton
    )
    if not restores:
        return regular()
    on = state.restoration.on
    live = ~state.done
    if mode != "exact":
        curvature = state.quasi_newton if _hessian.uses_quasi_newton(mode) else 0.0
        return jnp.where(on, curvature, regular())
    if closures.constraint_hessian is not None:
        return live_lane_cond(
            on, lambda: closures.constraint_hessian(x, lam_e, lam_i, args), regular, live=live
        )
    zero_e, zero_i = jnp.zeros_like(lam_e), jnp.zeros_like(lam_i)
    objective = live_lane_cond(
        on,
        lambda: closures.lagrangian_hessian(x, zero_e, zero_i, args),
        lambda: jnp.zeros((x.shape[0], x.shape[0]), x.dtype),
        live=live,
        false_is_free=True,
    )
    return regular() - objective


def barrier_step(
    solver: Any,
    closures: Closures,
    bounds: Bounds,
    state: InteriorPointState,
    args: PyTree,
    restores: bool,
) -> InteriorPointState:
    """One step of the interior point's phase, ending the run on a failure.

    Args:
        solver: The [InteriorPoint][diff_soup.optimization.InteriorPoint].
        closures: The scaled closures.
        bounds: The bounds of `x`.
        state: The state, after the barrier update.
        args: `(args, scales)`.
        restores: Whether a failed line search starts the restoration phase.

    Returns:
        The state after the step and the phase change it makes.
    """
    x, s, ev, mu, tau = state.x, state.slacks, state.evaluation, state.mu, state.tau
    n_eq = closures.n_equality
    n_ineq = closures.n_inequality
    dtype = x.dtype
    phase = state.restoration
    on = phase.on if restores else jnp.asarray(False)
    gauss_newton = _hessian.uses_gauss_newton(solver.hessian, closures)

    parts = terms(state, bounds, restores)
    jacobian = jnp.concatenate([ev.jacobian_equality, ev.jacobian_inequality])
    lam = jnp.concatenate([state.lam_e, state.lam_i])
    hessian = _hessian_of(solver, closures, state, args, restores)
    sigma_x = primal_sigma(x, state.z_lower, state.z_upper, bounds)
    primal_block = hessian + jnp.diag(sigma_x + parts.primal_extra)
    dual_diagonal = jnp.concatenate([jnp.zeros(n_eq, dtype), parts.inv_sigma_s])
    slack_rhs = jnp.concatenate([jnp.zeros(n_eq, dtype), parts.r_s * parts.inv_sigma_s])
    if restores:
        dual_diagonal = dual_diagonal + parts.dual_extra
        slack_rhs = slack_rhs + parts.rhs_extra
    factor, delta_w = solver.kkt.factor_regularized(
        primal_block, jacobian, dual_diagonal, DELTA_C * mu**KAPPA_C, state.delta_w_last
    )

    def direction(constraint_rhs):
        dx, dlam = solver.kkt.solve_factored(factor, parts.r_x, -constraint_rhs - slack_rhs)
        ds = (parts.r_s - dlam[n_eq:]) * parts.inv_sigma_s
        if not restores:
            return dx, ds, dlam
        dp = jnp.where(on, (parts.r_p + dlam) * parts.inv_p, 0.0)
        dn = jnp.where(on, (parts.r_n - dlam) * parts.inv_n, 0.0)
        return dx, jnp.concatenate([ds, dp, dn]), dlam

    def split(slacks):
        if not restores:
            return slacks, phase.p, phase.n
        m = n_eq + n_ineq
        return slacks[:n_ineq], slacks[n_ineq : n_ineq + m], slacks[n_ineq + m :]

    def trial_point(x_trial, slacks_trial):
        value, equality, inequality, valid = trial(closures, x_trial, args, value_required=~on)
        s_t, p_t, n_t = split(slacks_trial)
        at = merit(state, bounds, restores, x_trial, s_t, p_t, n_t, value, equality, inequality)
        return (*at, valid)

    slacks = jnp.concatenate([s, phase.p, phase.n]) if restores else s
    theta, phi, constraint = merit(
        state, bounds, restores, x, s, phase.p, phase.n, ev.value, ev.equality, ev.inequality
    )
    dx, dslacks, dlam = direction(constraint)
    _, dp, dn = split(dslacks)
    alpha_max = fraction_to_boundary(x, dx, slacks, dslacks, tau, bounds)
    slope = parts.grad_x @ dx + parts.grad_s @ split(dslacks)[0]
    if restores:
        slope = slope + jnp.where(on, parts.grad_p @ dp + parts.grad_n @ dn, 0.0)
    eps = jnp.finfo(dtype).eps
    tiny = (
        jnp.maximum(
            jnp.max(jnp.abs(dx) / (1.0 + jnp.abs(x)), initial=0.0),
            jnp.max(jnp.abs(dslacks) / (1.0 + jnp.abs(slacks)), initial=0.0),
        )
        < 10.0 * eps
    )
    search = filter_line_search(
        trial_point,
        x,
        slacks,
        dx,
        dslacks,
        alpha_max=alpha_max,
        theta=theta,
        phi=phi,
        slope=slope,
        filt=state.filt,
        theta_min=state.theta_min,
        theta_max=state.theta_max,
        consecutive_failed=state.consecutive_failed,
        max_failed=solver.max_failed_evaluations,
        accept_full=tiny & factor.ok,
        correction=Correction(direction, constraint, dlam, tau, bounds),
    )
    searched = search.accepted & factor.ok
    exhausted = search.consecutive_failed >= solver.max_failed_evaluations
    filt_current = _filter.augment(
        state.filt, (1.0 - _filter.GAMMA_THETA) * theta, phi - _filter.GAMMA_PHI * theta
    )

    alpha = search.alpha
    ds, dp, dn = split(search.ds)
    dz_lower, dz_upper, dv = bound_multiplier_steps(
        x, search.dx, s, ds, state.z_lower, state.z_upper, state.v, mu, bounds
    )
    duals, dduals = state.v, dv
    if restores:
        dz_p = jnp.where(on, (mu - phase.z_p * phase.p - phase.z_p * dp) / phase.p, 0.0)
        dz_n = jnp.where(on, (mu - phase.z_n * phase.n - phase.z_n * dn) / phase.n, 0.0)
        duals = jnp.concatenate([state.v, phase.z_p, phase.z_n])
        dduals = jnp.concatenate([dv, dz_p, dz_n])
    alpha_dual = dual_fraction_to_boundary(
        state.z_lower, dz_lower, state.z_upper, dz_upper, duals, dduals, tau, bounds
    )
    x_new = x + alpha * search.dx
    s_new = s + alpha * ds
    lam_new = lam + alpha * search.dlam
    z_lower, z_upper, v = safeguard_multipliers(
        x_new,
        s_new,
        state.z_lower + alpha_dual * dz_lower,
        state.z_upper + alpha_dual * dz_upper,
        state.v + alpha_dual * dv,
        mu,
        bounds,
    )
    if restores:
        p_new, n_new = phase.p + alpha * dp, phase.n + alpha * dn
        z_p = _safeguard(phase.z_p + alpha_dual * dz_p, p_new, mu)
        z_n = _safeguard(phase.z_n + alpha_dual * dz_n, n_new, mu)
        moved_phase = dataclasses.replace(
            phase, p=p_new, n=n_new, z_p=z_p, z_n=z_n, steps=_int(phase.steps + on)
        )
    else:
        moved_phase = phase
    ev_new = live_lane_cond(
        searched,
        lambda: evaluate(closures, x_new, args, gauss_newton),
        lambda: ev,
        live=~state.done,
        false_is_free=True,
    )
    finite = jnp.where(on, ev_new.rows_finite, ev_new.finite)
    moved = searched & finite

    lam_e_new, lam_i_new = lam_new[:n_eq], lam_new[n_eq:]
    secant = lagrangian_secant(ev_new, ev, lam_e_new, lam_i_new)
    quasi_newton, updates = _hessian.update(
        solver.hessian,
        closures,
        state.quasi_newton,
        state.quasi_newton_updates,
        x_new - x,
        secant,
        ev_new,
    )
    if restores and _hessian.uses_quasi_newton(solver.hessian):
        rows_secant = (ev_new.jacobian_equality - ev.jacobian_equality).T @ lam_e_new + (
            ev_new.jacobian_inequality - ev.jacobian_inequality
        ).T @ lam_i_new
        resto = _hessian.damped_update(
            state.quasi_newton, state.quasi_newton_updates, x_new - x, rows_secant
        )
        quasi_newton, updates = _select(on, resto, (quasi_newton, updates))
    moved_state = dataclasses.replace(
        state,
        x=x_new,
        slacks=s_new,
        lam_e=lam_e_new,
        lam_i=lam_i_new,
        z_lower=z_lower,
        z_upper=z_upper,
        v=v,
        evaluation=ev_new,
        filt=_select(searched & search.f_type, state.filt, filt_current),
        quasi_newton=quasi_newton,
        quasi_newton_updates=updates,
        restoration=moved_phase,
    )
    stepped = _select(moved, moved_state, state)

    consecutive = search.consecutive_failed + (searched & ~finite)
    consecutive = jnp.where(moved, 0, consecutive)
    evaluations_failed = exhausted | (consecutive >= solver.max_failed_evaluations)
    delta_w_last = jnp.where(delta_w > 0.0, delta_w, state.delta_w_last)
    stepped = dataclasses.replace(
        stepped,
        step=_int(state.step + ~on),
        consecutive_failed=_int(consecutive),
        failed_evaluations=_int(state.failed_evaluations + search.failed),
        delta_w_last=delta_w_last,
        inner_steps=_int(state.inner_steps + on),
    )
    if not restores:
        failed = ~searched | evaluations_failed
        final, infeasible, error = stepped, jnp.asarray(False), regular_error(stepped, 0.0, bounds)
    else:
        tol = solver.tol
        accepted, theta_original = _restoration.accepts(
            stepped.saved,
            stepped.restoration,
            stepped.evaluation,
            stepped.x,
            stepped.slacks,
            bounds,
        )
        leave = on & moved & accepted
        converged = on & moved & ~leave & (phase_error(stepped, 0.0, bounds, True) <= tol)
        over = on & ~leave & (stepped.restoration.steps >= _restoration.MAX_STEPS)
        enter = ~on & ~searched & ~exhausted & (theta > tol)
        entered = _restoration.enter(stepped, theta, constraint, filt_current, solver.tau_min)
        left = _restoration.leave(stepped, bounds, solver.hessian, closures)
        final = _select(enter, entered, _select(leave, left, stepped))
        error = phase_error(final, 0.0, bounds, True)
        converged_at_entry = enter & (error <= tol)
        infeasible = (converged & (theta_original > tol)) | (converged_at_entry & (theta > tol))
        failed = (~searched & ~enter) | converged | over | converged_at_entry | evaluations_failed
    status = jnp.where(
        evaluations_failed,
        TerminationStatus.SOLVER_FAILED,
        jnp.where(infeasible, TerminationStatus.INFEASIBLE, TerminationStatus.RESTORATION_FAILED),
    )
    return dataclasses.replace(
        final,
        done=state.done | failed,
        status=_int(jnp.where(failed, status, state.status)),
        kkt_error=error,
    )


def _safeguard(z: Float[Array, "m"], distance: Float[Array, "m"], mu: Float[Array, ""]):
    """Ipopt's `kappa_sigma` safeguard on the multipliers of `p` or `n`."""
    target = mu / distance
    return jnp.clip(z, target / KAPPA_SIGMA, KAPPA_SIGMA * target)
