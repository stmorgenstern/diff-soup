"""The public initialization entries: the solve, its result and the eager diagnosis."""

from __future__ import annotations

import functools
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.problem.implicit.consistent_init.newton import _k_matrix, _newton_primal
from diff_soup.problem.implicit.consistent_init.plan import ConsistentInitPlan
from diff_soup.problem.implicit.consistent_init.rules import (
    consistent_ic_root,
    consistent_ic_root_reduced,
)


class ConsistentInitResult(eqx.Module):
    """The result of a consistent initialization.

    Attributes:
        y: The consistent state.
        ydot: The consistent derivative, zero on algebraic rows.
        residual_norm: The residual 2-norm at `y` and `ydot`.
        iterations: The Newton iterations used.
        converged: Whether the final residual passed the absolute plus relative gate.
    """

    y: jax.Array
    ydot: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array


def _plan_root_entry(icp: "ConsistentInitPlan"):
    """The plan's compiled root-solve entry.

    Dispatches on the plan's route (joint or reduced). Jitted once per plan
    and stored on it, so eager repeat calls replay one compiled program
    instead of re-tracing the Newton solve; inside an outer trace the jit
    inlines transparently. Living on the plan, the compiled entry shares the
    plan's own lifetime.
    """
    entry = getattr(icp, "_root_entry", None)
    if entry is None:
        root = consistent_ic_root_reduced if icp.route == "reduced" else consistent_ic_root
        entry = jax.jit(functools.partial(root, icp))
        icp._root_entry = entry
    return entry


def _solver_problem_and_theta(problem, theta):
    """Normalize an authored source/view and resolve its default parameters."""
    from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem

    if isinstance(problem, (MassProblem, ImplicitProblem)):
        default_theta = getattr(problem, "default_theta", None)
        return problem, default_theta() if theta is None and callable(default_theta) else theta
    from diff_soup.problem.implicit.solver_problem import solver_view

    view = solver_view(problem)
    return view.problem, view.default_theta() if theta is None else theta


def consistent_initial_state(
    problem,
    t0,
    y0,
    theta,
    control=None,
    *,
    ydot0=None,
    plan=None,
    init_plan: Optional[ConsistentInitPlan] = None,
    tol: float = 1e-12,
    rtol: float = 0.0,
    max_iters: int = 30,
    chord_iters: int = 8,
    backend="auto",
    analysis=None,
    route: str = "auto",
) -> ConsistentInitResult:
    """Solve for a consistent initial state of a DAE at `t0`.

    It holds the differential values of `y0` and solves for the algebraic values and the
    differential slopes with Newton's method. The result is differentiable with respect to
    `t0`, `y0`, `theta` and `control`.

    Args:
        problem: The problem from `spec.build()`, or a `MassProblem` or `ImplicitProblem`.
        t0: The initial time.
        y0: The initial state, whose algebraic entries are the starting guess.
        theta: The parameters, or `None` for the problem's defaults.
        control: The control values, or `None`.
        ydot0: A guess for the slopes, zeros by default.
        plan: A prebuilt linearization plan, or `None` to build one.
        init_plan: A prebuilt `ConsistentInitPlan`, which overrides every setting below it.
        tol: The residual 2-norm at which Newton's method stops.
        rtol: Relative part of the final gate: a positive value also accepts a
            residual 2-norm up to ``tol + (rtol + γ) S``, with
            ``S = ‖|F_y| |y| + |F_ydot| |ydot|‖₂`` at the final iterate and ``γ``
            the rounding bound of the longest residual row. Zero keeps the
            absolute gate only.
        max_iters: The total Newton iteration budget.
        chord_iters: The first iterations, which reuse one factorization before each later
            iteration refactors.
        backend: The linear backend, a name as on `linear=` or a `ds.linear` object.
        analysis: A prebuilt linear analysis of the Newton matrix, or `None`.
        route: `"reduced"` solves the algebraic block alone and needs a diagonal mass, `"joint"`
            the full system, and `"auto"` picks.

    Returns:
        A `ConsistentInitResult`.
    """
    problem, theta = _solver_problem_and_theta(problem, theta)
    icp = (
        init_plan
        if init_plan is not None
        else ConsistentInitPlan(
            problem,
            plan,
            tol=tol,
            rtol=rtol,
            max_iters=max_iters,
            chord_iters=chord_iters,
            backend=backend,
            analysis=analysis,
            route=route,
        )
    )
    y0 = jnp.asarray(y0)
    ydot_guess = jnp.zeros_like(y0) if ydot0 is None else jnp.asarray(ydot0)
    out = _plan_root_entry(icp)(t0, y0, ydot_guess, theta, control)
    y, ydot, norm, iterations, converged = out
    return ConsistentInitResult(
        y=y, ydot=ydot, residual_norm=norm, iterations=iterations, converged=converged
    )


def _partition_note(structure, index: int) -> str:
    """A ``" (partition 'name')"`` annotation for a state index, or ``""``."""
    names = [name for name, lo, hi in structure.partitions if lo <= index < hi]
    if not names:
        return ""
    return " (partition " + ", ".join(repr(name) for name in names) + ")"


def _describe_unknowns(icp: ConsistentInitPlan, z_indices) -> str:
    """Name joint unknowns: algebraic values ``y[i]`` and slopes ``ydot[i]``."""
    values = [int(j) for j in z_indices if not icp.diff_mask[int(j)]]
    slopes = [int(j) for j in z_indices if icp.diff_mask[int(j)]]
    parts = []
    if values:
        parts.append(
            "algebraic values "
            + ", ".join(f"y[{i}]{_partition_note(icp.structure, i)}" for i in values)
        )
    if slopes:
        parts.append(
            "differential slopes "
            + ", ".join(f"ydot[{i}]{_partition_note(icp.structure, i)}" for i in slopes)
        )
    return "; ".join(parts) if parts else "none identified"


def check_consistent_init(
    problem,
    t0,
    y0,
    theta,
    control=None,
    *,
    ydot0=None,
    plan=None,
    init_plan: Optional[ConsistentInitPlan] = None,
    tol: float = 1e-12,
    rtol: float = 0.0,
    max_iters: int = 30,
) -> ConsistentInitResult:
    """Solve for a consistent initial state, and raise with a diagnosis when there is none.

    It runs the joint route with a dense linear solve and then checks the rank of the joint
    matrix at the result. It runs on concrete arrays, outside `jax.jit`.

    Args:
        problem: The problem from `spec.build()`, or a `MassProblem` or `ImplicitProblem`.
        t0: The initial time.
        y0: The initial state, whose algebraic entries are the starting guess.
        theta: The parameters, or `None` for the problem's defaults.
        control: The control values, or `None`.
        ydot0: A guess for the slopes, zeros by default.
        plan: A prebuilt linearization plan, or `None` to build one.
        init_plan: A prebuilt `ConsistentInitPlan`, which overrides every setting below it.
        tol: The residual 2-norm at which Newton's method stops.
        rtol: Relative part of the final gate: a positive value also accepts a
            residual 2-norm up to ``tol + (rtol + γ) S``, with
            ``S = ‖|F_y| |y| + |F_ydot| |ydot|‖₂`` at the final iterate and ``γ``
            the rounding bound of the longest residual row. Zero keeps the
            absolute gate only.
        max_iters: The total Newton iteration budget.

    Returns:
        A converged `ConsistentInitResult`.

    Raises:
        ValueError: When the joint matrix is singular, naming the unknowns and rows involved, or
            when Newton's method does not converge.
    """
    problem, theta = _solver_problem_and_theta(problem, theta)
    icp = (
        init_plan
        if init_plan is not None
        else ConsistentInitPlan(
            problem, plan, tol=tol, rtol=rtol, max_iters=max_iters, backend="dense", route="joint"
        )
    )
    if icp.route != "joint":
        # The diagnostic always analyzes the joint matrix (it contains the
        # reduced solve's algebraic block); re-express the plan as its joint
        # equivalent.
        icp = ConsistentInitPlan(
            problem,
            icp.plan,
            tol=icp.tol,
            rtol=icp.rtol,
            max_iters=icp.max_iters,
            backend="dense",
            route="joint",
        )
    y0 = jnp.asarray(y0)
    ydot_guess = jnp.zeros_like(y0) if ydot0 is None else jnp.asarray(ydot0)
    y_star, ydot_star, norm, iterations, converged = _newton_primal(
        icp, t0, y0, ydot_guess, theta, control
    )
    k = np.asarray(_k_matrix(icp, t0, y_star, ydot_star, theta, control))
    n = icp.n
    if not np.all(np.isfinite(k)):
        raise ValueError(
            "consistent initialization cannot be analyzed: the joint matrix "
            "(the algebraic columns of F_y and the differential columns of "
            "F_ydot) has non-finite entries at the solved point: the "
            "residual's derivatives are not finite there. Supply seeds inside "
            "the residual's domain."
        )
    u, s, vh = np.linalg.svd(k)
    cutoff = (s[0] if s.size else 0.0) * n * np.finfo(k.dtype).eps
    rank = int(np.sum(s > cutoff))
    if rank < n:
        null_cols = vh[rank:]
        col_scale = np.max(np.abs(null_cols), axis=0)
        implicated_z = np.flatnonzero(col_scale > 1e-6 * max(col_scale.max(), 1e-300))
        row_scale = np.max(np.abs(u[:, rank:]), axis=1)
        implicated_rows = np.flatnonzero(row_scale > 1e-6 * max(row_scale.max(), 1e-300))
        rows_text = ", ".join(
            f"{int(i)}{_partition_note(icp.structure, int(i))}" for i in implicated_rows
        )
        raise ValueError(
            "consistent initialization is singular at the solved point: the "
            "joint matrix (the algebraic columns of F_y and the differential "
            f"columns of F_ydot) has rank {rank} of {n}. Implicated unknowns: "
            f"{_describe_unknowns(icp, implicated_z)}. Implicated equation "
            f"rows: [{rows_text}]. Every algebraic value and differential "
            "slope must be uniquely determined by the residual at t0: remove "
            "redundant equations, correct the declared variable roles, or add "
            "the missing coupling."
        )
    if not bool(converged):
        remedy = (
            "Raise max_iters, loosen tol, or supply closer algebraic-value and slope seeds."
            if jnp.asarray(y_star).dtype != jnp.float32
            else "In float32 the residual's own rounding can exceed tol; loosen tol, or "
            "initialize in float64 with Precision.fp32_ir() or Precision.fp64()."
        )
        raise ValueError(
            "consistent initialization did not converge: after "
            f"{int(iterations)} Newton iterations the residual 2-norm is "
            f"{float(norm):.3e} against the tolerance {icp.tol:.1e}, with a "
            f"nonsingular joint matrix at the last iterate. {remedy}"
        )
    return ConsistentInitResult(
        y=y_star,
        ydot=ydot_star,
        residual_norm=norm,
        iterations=iterations,
        converged=converged,
    )
