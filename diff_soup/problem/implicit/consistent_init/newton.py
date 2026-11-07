"""The initialization's Newton solve and its final gate.

Both routes run a chord phase over one frozen factor, then exact Newton with a fresh
matrix at every iteration, aiming at the residual 2-norm ``tol``. The joint route
iterates on the state-aligned unknowns with the exact matrix ``K`` (the algebraic
columns of ``F_y`` and the differential columns of ``F_ydot``); the reduced route
iterates on the algebraic values with ``G = F_y[a, a]``. The final gate adds the
relative part ``(rtol + γ_m) ‖|F_y| |y| + |F_ydot| |ydot|‖₂`` where the absolute
target was missed. These solves are never differentiated: the entries in
[diff_soup.problem.implicit.consistent_init.rules][] apply the implicit-function rule
at their root.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import SparseTopology
from diff_soup.nonlinear._kkt import RouteFactor, linear_route
from diff_soup.problem.implicit.consistent_init.plan import ConsistentInitPlan
from diff_soup.utils.lanes import lane_cond


def _embed(icp: ConsistentInitPlan, y_pinned, z):
    """Blend the state-aligned joint unknowns into the full ``(y, ydot)`` pair.

    Slot ``s`` of ``z`` is ``y[s]`` for an algebraic variable and ``ydot[s]``
    for a differential one; the algebraic slopes are structurally zero.
    """
    mask = jnp.asarray(icp.diff_mask)
    y = jnp.where(mask, y_pinned, z)
    ydot = jnp.where(mask, z, jnp.zeros_like(z))
    return y, ydot


def _k_matrix(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact dense ``K`` at a point (state-aligned columns).

    Column ``s`` is ``F_y[:, s]`` for an algebraic variable and
    ``F_ydot[:, s]`` for a differential one. The dense-route Jacobian and the
    validation/diagnostic reference; the sparse route uses
    [_k_values][diff_soup.problem.implicit.consistent_init.newton._k_values]
    and never calls this.
    """
    plan = icp.plan
    fy = plan.fy_values(t, y, ydot, theta, control)
    fydot = plan.fydot_values(t, y, ydot, theta, control)
    if plan.mode == "dense":
        return jnp.where(jnp.asarray(icp.diff_mask)[None, :], fydot, fy)
    k = jnp.zeros((icp.n, icp.n), dtype=jnp.result_type(fy, fydot))
    k = k.at[icp._fy_rows, icp._fy_cols].add(fy[icp._fy_keep], unique_indices=True)
    return k.at[icp._fyd_rows, icp._fyd_cols].add(fydot, unique_indices=True)


def _k_values(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact ``K`` values for the backend route, never an ``(n, n)`` array.

    Sparse-mode plans return values over
    [ConsistentInitPlan.k_topology][diff_soup.problem.implicit.consistent_init.ConsistentInitPlan.k_topology]
    (the kept ``F_y`` algebraic-column entries followed by the ``F_ydot``
    entries); dense-mode plans return the dense ``K`` flattened row-major
    over the full pattern.
    """
    plan = icp.plan
    if plan.mode == "dense":
        return _k_matrix(icp, t, y, ydot, theta, control).reshape(-1)
    fy = jnp.asarray(plan.fy_values(t, y, ydot, theta, control))
    fydot = jnp.asarray(plan.fydot_values(t, y, ydot, theta, control))
    dtype = jnp.result_type(fy, fydot)
    return jnp.concatenate([fy[icp._fy_keep].astype(dtype), fydot.astype(dtype)])


def _routes(icp: ConsistentInitPlan, values, n: int):
    """The forward and reverse routes of matrix values on the plan's linear route.

    The sparse route serves values over the plan's pattern with its backend
    (the reverse one analysed with the plan's reverse policy, at its adjoint
    dtype); the dense route serves a dense matrix (flattened row-major) with
    the dense backend.
    """
    values = jnp.asarray(values)
    reverse_dtype = values.dtype if icp.adjoint_dtype is None else jnp.dtype(icp.adjoint_dtype)
    if icp.linear_backend is None:
        topology = SparseTopology.full(n)
        return (
            linear_route(None, None, dtype=values.dtype, topology=topology),
            linear_route(None, None, dtype=reverse_dtype, topology=topology),
        )
    return (
        linear_route(icp.linear_backend, icp.linear_analysis, dtype=values.dtype),
        linear_route(icp.linear_backend, icp.reverse_linear_analysis, dtype=reverse_dtype),
    )


def _route_factor(icp: ConsistentInitPlan, values, n: int) -> RouteFactor:
    """Factor matrix values once on the plan's forward linear route."""
    route, _reverse = _routes(icp, values, n)
    return RouteFactor(route, jnp.asarray(values))


def _k_root_values(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact ``K`` values in the layout of the plan's linear route."""
    if icp.linear_backend is None:
        return _k_matrix(icp, t, y, ydot, theta, control).reshape(-1)
    return _k_values(icp, t, y, ydot, theta, control)


def _k_solver(icp, t, y, ydot, theta, control):
    """Factorize the exact ``K`` once at a point; return its solve.

    Routes through the plan's resolved linear backend when one is recorded
    (sparse values over the joint topology) and through the dense backend
    otherwise.
    """
    held = _route_factor(icp, _k_root_values(icp, t, y, ydot, theta, control), icp.n)
    return held.solve


#: Chord contraction gate: a chord step whose residual norm exceeds this
#: fraction of the previous norm hands the solve to fresh-Jacobian Newton.
_CHORD_CONTRACTION_LIMIT = 0.5


def _residual_scale(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """``‖|F_y| |y| + |F_ydot| |ydot|‖₂`` over the rows the solve drives to zero.

    The size of the residual's own terms at the iterate, the scale of the
    relative gate. The reduced route evaluates the algebraic rows of ``F_y``
    only. Sparse plans stay sparse.
    """
    plan = icp.plan
    rows = jnp.asarray(icp.a_idx) if icp.route == "reduced" else slice(None)
    scale = jnp.zeros(icp.n_a if icp.route == "reduced" else icp.n, dtype=y.dtype)
    for name, keep, row in icp._scale_terms:
        values = getattr(plan, f"{name}_values")(t, y, ydot, theta, control)
        state = y if name == "fy" else ydot
        if keep is None:
            scale = scale + jnp.abs(values[rows]) @ jnp.abs(state)
            continue
        cols = getattr(plan, f"{name}_topology").cols[keep]
        scale = scale.at[row].add(jnp.abs(values[keep]) * jnp.abs(state[cols]))
    return jnp.linalg.norm(scale)


def _rounding_bound(length: int, dtype) -> float:
    """``γ_m = m u / (1 - m u)``, the relative rounding bound of an ``m``-term sum.

    Evaluating ``Σ_j a_j x_j`` in floating point errs by at most
    ``γ_m Σ_j |a_j| |x_j|`` (Higham, Accuracy and Stability of Numerical
    Algorithms, 2nd ed., 2002, eq. 3.5), so a residual row with ``m``
    structural terms cannot be driven below ``γ_m`` times its scale.
    """
    unit = float(np.finfo(dtype).eps) / 2.0
    return length * unit / (1.0 - length * unit)


def _phased_newton(icp: ConsistentInitPlan, z0, residual_at, make_solver, scale_at):
    """Chord phase over one frozen factor, then fresh-Jacobian exact Newton.

    The chord phase factors the iteration matrix once at the seed and reuses
    that factor for up to ``chord_iters`` updates, watching per-iteration
    contraction and finiteness; poor contraction, a nonfinite update or
    residual, or an exhausted chord budget hands the best iterate so far to
    an exact-Newton phase that refreshes the matrix every iteration within
    the remaining ``max_iters`` budget. Convergence is always judged on a
    freshly evaluated residual, and an unconverged solve reports itself
    honestly. A seed that already satisfies the residual gate (or is
    nonfinite) returns without factoring anything.
    """
    r0 = residual_at(z0)
    norm0 = jnp.linalg.norm(r0)
    it0 = jnp.asarray(0, jnp.int32)
    chord_budget = min(icp.chord_iters, icp.max_iters)

    def solve_phases(_):
        """Run the chord phase, then exact Newton on whatever remains."""
        if chord_budget > 0:
            # One factorization before the chord loop serves every chord update.
            solve0 = make_solver(z0)

            def chord_going(carry):
                """Iterate while unconverged, in budget, finite, untriggered."""
                _z, r, it, stop = carry
                norm = jnp.linalg.norm(r)
                return (~stop) & (it < chord_budget) & (norm > icp.tol) & jnp.isfinite(norm)

            def chord_step(carry):
                """One frozen-factor update with contraction/finiteness guards."""
                z, r, it, _stop = carry
                z_new = z - solve0(r)
                r_new = residual_at(z_new)
                norm_old = jnp.linalg.norm(r)
                norm_new = jnp.linalg.norm(r_new)
                finite = jnp.isfinite(norm_new) & jnp.all(jnp.isfinite(z_new))
                improved = finite & (norm_new < norm_old)
                trigger = (~finite) | (norm_new > _CHORD_CONTRACTION_LIMIT * norm_old)
                z_keep = jnp.where(improved, z_new, z)
                r_keep = jnp.where(improved, r_new, r)
                return z_keep, r_keep, it + 1, trigger

            z, r, it, _ = jax.lax.while_loop(
                chord_going, chord_step, (z0, r0, it0, jnp.asarray(False))
            )
        else:
            z, r, it = z0, r0, it0

        def exact_going(carry):
            """Iterate while unconverged, within budget, and finite."""
            _z, r, it = carry
            norm = jnp.linalg.norm(r)
            return (it < icp.max_iters) & (norm > icp.tol) & jnp.isfinite(norm)

        def exact_step(carry):
            """One exact Newton update with a fresh iteration matrix."""
            z, r, it = carry
            solve = make_solver(z)
            z_new = z - solve(r)
            return z_new, residual_at(z_new), it + 1

        return jax.lax.while_loop(exact_going, exact_step, (z, r, it))

    def already_done(_):
        """The seed already meets the gate (or is nonfinite): no solve."""
        return z0, r0, it0

    z, r, iterations = jax.lax.cond(
        (norm0 > icp.tol) & jnp.isfinite(norm0), solve_phases, already_done, None
    )
    norm = jnp.linalg.norm(r)
    # Newton aims at the absolute target; only the final gate admits the
    # relative part, scaled by the residual's own terms at the final iterate
    # and evaluated only where the absolute target was missed.
    if icp.rtol > 0.0:
        scale = lane_cond(
            norm > icp.tol, scale_at, lambda _: jnp.zeros_like(norm), z, false_is_free=True
        )
        gate = icp.tol + (icp.rtol + _rounding_bound(icp._scale_length, norm.dtype)) * scale
    else:
        gate = jnp.asarray(icp.tol, norm.dtype)
    converged = (norm <= gate) & jnp.isfinite(norm) & jnp.isfinite(norm0) & jnp.isfinite(gate)
    return z, norm, iterations, converged


def _newton_primal(icp: ConsistentInitPlan, t0, y0, ydot_guess, theta, control):
    """Bounded chord-then-exact Newton solve for the joint unknowns.

    Never differentiated; the derivative entries apply the implicit-function
    rule at the converged root.
    """
    if icp.route != "joint":
        raise ValueError(
            "this entry runs the joint initialization solve but the plan was "
            "built for the reduced route; build the plan with route='joint', "
            "or solve through consistent_initial_state, which dispatches on "
            "the plan's own route."
        )
    z0 = jnp.where(jnp.asarray(icp.diff_mask), ydot_guess, y0)

    def residual_at(z):
        """The full residual at the embedded joint iterate."""
        y, ydot = _embed(icp, y0, z)
        return icp.plan._residual(t0, y, ydot, theta, control)

    def make_solver(z):
        """Factor the exact joint matrix at an iterate; return the solve."""
        y, ydot = _embed(icp, y0, z)
        return _k_solver(icp, t0, y, ydot, theta, control)

    def scale_at(z):
        """The relative gate's scale at the embedded joint iterate."""
        y, ydot = _embed(icp, y0, z)
        return _residual_scale(icp, t0, y, ydot, theta, control)

    z, norm, iterations, converged = _phased_newton(icp, z0, residual_at, make_solver, scale_at)
    y_star, ydot_star = _embed(icp, y0, z)
    return y_star, ydot_star, norm, iterations, converged


def _g_matrix(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact dense algebraic block ``G = F_y[a, a]`` at a point.

    The dense-route Jacobian of the reduced algebraic solve; the sparse
    route uses
    [_g_values][diff_soup.problem.implicit.consistent_init.newton._g_values]
    and never materializes this.
    """
    plan = icp.plan
    zeros = jnp.zeros_like(y)
    fy = plan.fy_values(t0, y, zeros, theta, control)
    if plan.mode == "dense":
        a = jnp.asarray(icp.a_idx)
        return fy[a][:, a]
    g = jnp.zeros((icp.n_a, icp.n_a), dtype=jnp.asarray(fy).dtype)
    return g.at[icp._ra_rows, icp._ra_cols].add(jnp.asarray(fy)[icp._ra_keep], unique_indices=True)


def _g_values(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact algebraic-block values for the backend route.

    Sparse-mode plans return values over
    [ConsistentInitPlan.reduced_topology][diff_soup.problem.implicit.consistent_init.ConsistentInitPlan.reduced_topology]; dense-mode plans return the
    dense block flattened row-major over the full pattern.
    """
    plan = icp.plan
    if plan.mode == "dense":
        return _g_matrix(icp, t0, y, theta, control).reshape(-1)
    zeros = jnp.zeros_like(y)
    fy = jnp.asarray(plan.fy_values(t0, y, zeros, theta, control))
    return fy[icp._ra_keep]


def _g_root_values(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact algebraic-block values in the layout of the plan's linear route."""
    if icp.linear_backend is None:
        return _g_matrix(icp, t0, y, theta, control).reshape(-1)
    return _g_values(icp, t0, y, theta, control)


def _g_solver(icp, t0, y, theta, control):
    """Factorize the exact algebraic block once at a point; return its solve.

    Routes through the plan's resolved linear backend when one is recorded
    and through the dense backend otherwise, mirroring
    [_k_solver][diff_soup.problem.implicit.consistent_init.newton._k_solver] on the
    compacted algebraic block.
    """
    held = _route_factor(icp, _g_root_values(icp, t0, y, theta, control), icp.n_a)
    return held.solve


def _reduced_newton_primal(icp: ConsistentInitPlan, t0, y0, theta, control):
    """Bounded chord-then-exact Newton solve on the algebraic rows.

    Never differentiated. The unknowns are the local- and global-algebraic values;
    the residual is the algebraic restriction of the full
    residual at zero slopes, and the iteration matrix is the exact algebraic
    block ``F_y[a, a]``.
    """
    if icp.route != "reduced":
        raise ValueError(
            "this entry runs the reduced initialization solve but the plan "
            "was built for the joint route; build the plan with "
            "route='reduced' (or 'auto' on an eligible problem), or solve "
            "through consistent_initial_state, which dispatches on the "
            "plan's own route."
        )
    y0 = jnp.asarray(y0)
    a_idx = jnp.asarray(icp.a_idx)
    zeros = jnp.zeros_like(y0)
    z0 = y0[a_idx]

    def residual_at(z):
        """The algebraic residual rows at the embedded iterate."""
        y = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
        return icp.plan._residual(t0, y, zeros, theta, control)[a_idx]

    def make_solver(z):
        """Factor the exact algebraic block at an iterate; return the solve."""
        y = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
        return _g_solver(icp, t0, y, theta, control)

    def scale_at(z):
        """The relative gate's scale at the embedded algebraic iterate."""
        y = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
        return _residual_scale(icp, t0, y, zeros, theta, control)

    z, _norm, iterations, converged = _phased_newton(icp, z0, residual_at, make_solver, scale_at)
    y_star = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
    return y_star, iterations, converged
