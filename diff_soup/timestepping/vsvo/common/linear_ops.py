"""The correctors' linear solves on the factor contract.

Every corrector of the vsvo engines solves its Newton systems with the
iteration matrix of the current setup: the matrix is rebuilt only when a setup
fires, and every Newton iteration, retry sweep and border (Schur) solve in
between reuses its factorization. This module holds that matrix between
attempts and builds the solve callables of one attempt:

* [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear]
  describes how the state block is factored: one
  [LinearRoute][diff_soup.nonlinear._kkt.LinearRoute] per precision rung (one
  for a concrete solve mode, two for the adaptive ladder), analysed once per
  pattern at trace time.
* [HeldMatrix][diff_soup.timestepping.vsvo.common.linear_ops.HeldMatrix] is the
  matrix an engine carries from attempt to attempt: the
  [Factor][diff_soup.linear.Factor] of each rung, whose values are the matrix.
* [refactor_held][diff_soup.timestepping.vsvo.common.linear_ops.refactor_held]
  refactors the held matrix where a setup replaced it,
  [held_solves][diff_soup.timestepping.vsvo.common.linear_ops.held_solves]
  builds the ``solve`` and ``solve_many`` callables against it, and
  [linearize][diff_soup.timestepping.vsvo.common.linear_ops.linearize] does
  both.

A backend that keeps its factors in native or host storage factors in the
process-wide default solve context: the engine's carry starts from
[empty_factor][diff_soup.linear.empty_factor] with the shared storage name
``"corrector"``, so every call site and every compiled program (the forward
solve, a gradient's forward sweep and its replays) refactors one set of factor
storage per pattern, precision and lane count, as the iteration matrix of one
integration. Every attempt refactors the held factor with its setup predicate;
the refactor makes the storage hold the carried matrix (it compares the values
it keeps with the values the storage holds and factors only where they differ
or a setup asked), so the solves of the attempt use the storage as it stands
without reading their handle. Under a checkpointed reverse pass a replayed
segment restores an older carry; its first attempt's refactor finds the
storage holding other values and factors the carried ones, so a replay never
solves with a factorization of another point. On a backend whose factors are
arrays (``Dense``) the factor itself rides the carry: it is refactored only
where a setup fires, and under ``CheckpointAdjoint`` every checkpoint holds a
copy of it, a memory the ``checkpoints`` count bounds. A layout that spreads
one factorization over the devices of a ``"model"`` mesh axis is held the same
way: on cuDSS its factor holds the matrix values and every solve factors them.

An iterative backend ([Krylov][diff_soup.linear.Krylov]) gives an iterative
route (``CorrectorLinear.inexact``): its held factor is a preconditioner of the
current matrix. The corrector passes the current matrix at every attempt, the
refactor takes it as the operator and rebuilds the preconditioner only where a
setup fires, and each solve runs GMRES to ``LINEAR_TOLERANCE_FRACTION`` (5 %)
of the Newton threshold in the Newton test's weighted norm, reporting whether
it got there (see
[held_solves][diff_soup.timestepping.vsvo.common.linear_ops.held_solves]).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._layout import Layout
from diff_soup.linear import (
    Factor,
    Tolerance,
    empty_factor,
    factor,
    refactor,
    solve,
    solve_checked,
)
from diff_soup.linear._context import DEFAULT_CONTEXT
from diff_soup.linear._refine import refine, refine_fixed
from diff_soup.nonlinear._kkt import LinearPlan, LinearRoute, linear_route
from diff_soup.timestepping.consistent_ic.pin_differential import _full_dense_topology
from diff_soup.timestepping.vsvo.common.backends import (
    adaptive_rung_precisions,
    linear_analyze_options,
)
from diff_soup.utils.lanes import lane_cond

SolveFn = Callable[[jax.Array], Any]

#: The shared storage name of the correctors' held factors.
_STORAGE = "corrector"


@dataclasses.dataclass(frozen=True)
class CorrectorLinear:
    """How a corrector factors and solves its state block.

    Built at trace time by
    [corrector_linear][diff_soup.timestepping.vsvo.common.linear_ops.corrector_linear];
    the analyses it holds are memoised by content, so every trace of the
    same problem shares them.

    Attributes:
        backend: The configured linear backend.
        routes: One [LinearRoute][diff_soup.nonlinear._kkt.LinearRoute] per
            precision rung, in rung order: one for a concrete solve mode,
            ``(fp32_ir, fp64)`` for the adaptive ladder.
    """

    backend: Any
    routes: tuple

    @property
    def adaptive(self) -> bool:
        """Whether the corrector switches between precision rungs."""
        return len(self.routes) > 1

    @property
    def reduced(self) -> bool:
        """Whether a rung factors in single precision below full-precision values."""
        return any(route.reduced is not None for route in self.routes)

    @property
    def inexact(self) -> bool:
        """Whether the solves are iterative: the operator is current, the preconditioner lagged.

        On such a route (a [Krylov][diff_soup.linear.Krylov] backend) the
        engine assembles the current matrix at every attempt, the held
        factor's preconditioner is rebuilt only where a setup fires, and every
        solve reports whether it met the linear tolerance.
        """
        return not self.routes[0].exact


#: The linear tolerance of an iterative corrector solve, as a fraction of the
#: Newton convergence threshold in the same weighted norm.
LINEAR_TOLERANCE_FRACTION = 0.05


def _working(route: LinearRoute) -> Any:
    """The analysis a route factors on: the single-precision one on a reduced route."""
    return route.full if route.reduced is None else route.reduced


def _context(analysis: Any) -> Any:
    """The solve context of a factor of ``analysis`` (none for a backend of array factors)."""
    if analysis.backend.factor_residency == "arrays":
        return None
    return DEFAULT_CONTEXT


def corrector_linear(
    backend: Any, topology: Any, *, layout: Any, precision: Any, dtype: Any
) -> CorrectorLinear:
    """The linear configuration of a corrector's state block.

    Args:
        backend: The resolved linear backend.
        topology: The state block's sparsity pattern (the full dense pattern
            for a dense problem).
        layout: The problem's [Layout][diff_soup.Layout].
        precision: The engine's [Precision][diff_soup.Precision]; an
            ``"adaptive"`` solve mode gives one route per rung of
            [adaptive_rung_precisions][diff_soup.timestepping.vsvo.common.backends.adaptive_rung_precisions].
        dtype: The dtype of the iteration-matrix values.

    Returns:
        The [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear].
    """
    if precision.solve_mode == "adaptive":
        rungs = adaptive_rung_precisions(precision)
    else:
        rungs = (precision,)
    routes = []
    for rung in rungs:
        plan = LinearPlan(
            topology=topology,
            layout=layout,
            options=tuple(sorted((linear_analyze_options(rung) or {}).items())),
            precision=rung,
        )
        routes.append(linear_route(backend, plan, dtype=dtype))
    return CorrectorLinear(backend=backend, routes=tuple(routes))


class HeldMatrix(eqx.Module):
    """The iteration matrix a corrector carries between attempts.

    On a backend that keeps its factors in native or host storage the matrix
    lives in its factors: ``factors`` holds one
    [Factor][diff_soup.linear.Factor] per precision rung, and the
    full-precision factor's values are the matrix. ``raw`` carries the
    full-precision values separately only where no factor holds them: with a
    single-precision rung (whose factor holds rounded values).

    Attributes:
        factors: The rung factors, in rung order.
        raw: The full-precision matrix values ``(nnz,)``, or ``None`` when
            the last factor's values are the matrix.
    """

    factors: tuple
    raw: Optional[jax.Array]

    @property
    def values(self) -> jax.Array:
        """The full-precision matrix values ``(nnz,)``."""
        if self.raw is not None:
            return self.raw
        return self.factors[-1].values


def problem_linear(backend: Any, problem: Any, *, precision: Any, dtype: Any) -> CorrectorLinear:
    """The linear configuration of an engine-facing problem's state block.

    The state block's pattern is the problem's ``state_topology`` (the full
    dense pattern for a dense problem), placed by the problem's lowered
    [Layout][diff_soup.Layout].

    Args:
        backend: The resolved linear backend.
        problem: The engine-facing problem view.
        precision: The engine's [Precision][diff_soup.Precision].
        dtype: The dtype of the iteration-matrix values.

    Returns:
        The [CorrectorLinear][diff_soup.timestepping.vsvo.common.linear_ops.CorrectorLinear].
    """
    topology = problem.state_topology
    if topology is None:
        topology = _full_dense_topology(int(problem.n_u))
    layout = getattr(problem, "layout", None)
    if layout is None:
        layout = Layout()
    return corrector_linear(backend, topology, layout=layout, precision=precision, dtype=dtype)


def state_block_size(problem: Any) -> int:
    """The number of values of a problem's state block (``n_u**2`` when it is dense)."""
    topology = problem.state_topology
    if topology is None:
        return int(problem.n_u) ** 2
    return int(topology.rows.shape[0])


def empty_held(linear: CorrectorLinear, nnz: int, dtype: Any) -> HeldMatrix:
    """The held matrix of a first attempt: no factorization yet, zero values.

    The first attempt's setup gate always fires, so its refactor fills the
    empty factors before any solve reads them. The factors name the shared
    storage ``"corrector"`` (see the module docstring); the storage is keyed by
    the rung's analysis, precision and lane count, so the rungs of the adaptive
    ladder do not share it.

    Args:
        linear: The corrector's linear configuration.
        nnz: The number of matrix values.
        dtype: The dtype of the matrix values.

    Returns:
        The held matrix to start the engine's carry with.
    """
    zeros = jnp.zeros((int(nnz),), dtype)
    factors = tuple(
        empty_factor(_working(route), ctx=_context(_working(route)), storage=_STORAGE)
        for route in linear.routes
    )
    keeps_raw = linear.reduced or linear.adaptive
    return HeldMatrix(factors=factors, raw=zeros if keeps_raw else None)


def _route_solves(
    f: Factor, route: LinearRoute, values: jax.Array, flagged: bool
) -> tuple[SolveFn, SolveFn]:
    """``(solve, solve_many)`` with one rung's factor, to the route's accuracy.

    A full-precision rung solves with its factor. A single-precision rung
    refines its solution against the full-precision ``values`` (``"fp32_ir"``)
    or checks its residual (``"fp32"``), and a solution that misses its target
    is replaced by a full-precision factorization and solve, under ``vmap``
    only when some lane misses. ``flagged`` makes ``solve`` return
    ``(x, fell_back, refinement_steps)`` as ``float64`` scalars.

    The callables return solutions only: the solves of a sweep need no order
    among themselves, and every later refactorization of the held matrix
    depends on the sweep's results (its setup predicate) or runs in a later
    iteration of the step loop, so it follows them.
    """
    zero = jnp.zeros((), jnp.float64)
    if route.reduced is None:

        def solve_full(rhs):
            x, _ = solve(f, rhs)
            return (x, zero, zero) if flagged else x

        return solve_full, lambda rhs_many: solve(f, rhs_many)[0]

    def solve_reduced(rhs):
        if route.mode == "fp32" and not route.residual_gate:
            return refine_fixed(f, rhs, values=values, steps=0)[0], zero, zero
        if route.mode == "fp32_ir":
            target, steps = route.residual_target, route.max_steps
        else:
            target, steps = route.direct_threshold, 0
        x, _, report = refine(f, rhs, values=values, residual_target=target, max_steps=steps)

        def full_precision(_):
            g = factor(route.full, values, ctx=_context(route.full))
            return solve(g, rhs)[0].astype(x.dtype)

        x = lane_cond(report.stalled, full_precision, lambda _: x, None, false_is_free=True)
        return x, report.stalled.astype(zero.dtype), report.steps.astype(zero.dtype)

    def solve_one(rhs):
        x, fell_back, steps = solve_reduced(rhs)
        return (x, fell_back, steps) if flagged else x

    return solve_one, lambda rhs_many: solve_reduced(rhs_many)[0]


def refactor_held(
    held: HeldMatrix,
    values: jax.Array,
    rebuild: Any,
    *,
    linear: CorrectorLinear,
    rung: Any = None,
    rung_changed: Any = None,
) -> HeldMatrix:
    """Refactor the held matrix where ``rebuild`` holds; keep it elsewhere.

    Where ``rebuild`` holds, the matrix becomes ``values`` and each held
    factor is refactored from them (under the adaptive ladder only the active
    rung's factor, and also when the rung was just entered, since an
    inactive rung's factor goes stale while the values evolve); elsewhere no
    factorization runs and the factor already held is kept. Under ``vmap``
    each lane decides for itself, and no factorization runs when no lane asks
    for one.

    Every refactor here runs in the shared storage the held factor names,
    whichever call site of whichever program runs it: the first sweep of an
    attempt, a retry sweep, or a replayed step of a checkpointed reverse
    pass. Each makes the storage hold the matrix it keeps (it compares the
    values it keeps with the values the storage holds and factors only where
    they differ: after a setup, or at the entry of a replayed segment whose
    restored matrix the storage no longer holds), so the solves after it use
    the storage as it stands. A backend whose factors are arrays carries them
    in the held factor and refactors them here only where ``rebuild`` holds.

    Args:
        held: The held matrix at attempt entry (or after an earlier sweep of
            the attempt).
        values: The candidate matrix values, any shape of ``nnz`` entries;
            where ``rebuild`` is false they must equal ``held.values``.
        rebuild: Boolean scalar (one per ``vmap`` lane): whether ``values``
            replaced the held matrix.
        linear: The corrector's linear configuration.
        rung: The active rung (``int32``) under the adaptive ladder, else
            ``None``.
        rung_changed: Whether the active rung was entered this attempt
            (adaptive ladder only).

    Returns:
        The held matrix in effect.
    """
    values = jnp.reshape(jnp.asarray(values), (-1,))
    rebuild = jnp.asarray(rebuild, jnp.bool_)
    if linear.adaptive:
        entered = jnp.asarray(rung_changed, jnp.bool_)
        asks = [(rung == i) & (rebuild | entered) for i in range(len(linear.routes))]
    else:
        asks = [rebuild]
    factors = tuple(
        refactor(f, values.astype(f.values.dtype), ask, adopt=True)
        for f, ask in zip(held.factors, asks)
    )
    return HeldMatrix(factors=factors, raw=None if held.raw is None else values)


def _iterative_solves(
    f: Factor, backend: Any, weights: Any, newton_tol: float
) -> tuple[SolveFn, SolveFn]:
    """``(solve, solve_many)`` of an iterative route's held factor.

    ``solve`` returns ``(x, failed, iterations)``: the solution, ``1.0``
    where the solve did not meet its tolerance (else ``0.0``) and the
    iteration count, both ``float64``. The tolerance is
    ``LINEAR_TOLERANCE_FRACTION * newton_tol`` in the root-mean-square norm
    weighted by ``weights`` (the Newton test's norm), within the backend's
    ``lagged_max_iter`` iterations. ``solve_many`` solves a block with the same
    rule and returns the solutions only.
    """
    default = backend.default_tolerance
    tol = Tolerance(
        0.0, LINEAR_TOLERANCE_FRACTION * float(newton_tol), backend.lagged_max_iter, default.restart
    )

    def solve_one(rhs):
        x, _, report = solve_checked(f, rhs, tol=tol, weights=weights)
        failed = jnp.logical_not(report.converged).astype(jnp.float64)
        return x, failed, report.iterations.astype(jnp.float64)

    def solve_many(rhs_many):
        return solve(f, rhs_many, tol=tol, weights=weights)[0]

    return solve_one, solve_many


def held_solves(
    held: HeldMatrix,
    *,
    linear: CorrectorLinear,
    rung: Any = None,
    flagged: bool = False,
    weights: Any = None,
    newton_tol: Optional[float] = None,
) -> tuple[SolveFn, SolveFn]:
    """The solve callables of a held matrix: the held factors solve.

    On an iterative route (``linear.inexact``) the held factor's operator is
    the current matrix and its preconditioner the one of the last setup:
    ``solve`` runs GMRES to ``0.05 * newton_tol`` in the root-mean-square norm
    weighted by ``weights`` (the Newton test's weights and threshold), within
    the backend's ``lagged_max_iter`` iterations, and returns
    ``(x, failed, iterations)`` whatever ``flagged`` asks (see
    ``_iterative_solves``); a failed solve is a recoverable nonlinear
    failure the corrector acts on. A corrector that passes no ``weights``
    solves with the held factor to the backend's default tolerance, as on an
    exact route.

    Args:
        held: The held matrix (after
            [refactor_held][diff_soup.timestepping.vsvo.common.linear_ops.refactor_held]).
        linear: The corrector's linear configuration.
        rung: The active rung (``int32``) under the adaptive ladder, else
            ``None``.
        flagged: Make ``solve`` return ``(x, fell_back, refinement_steps)``
            with ``float64`` counters of a single-precision rung's
            full-precision fallback and refinement steps (zeros on a
            full-precision rung).
        weights: The Newton test's error weights ``(n,)`` (iterative routes
            only; ``None`` elsewhere).
        newton_tol: The Newton test's threshold in the norm those weights
            define (iterative routes only).

    Returns:
        ``(solve, solve_many)`` for a right-hand side ``(n,)`` and a block
        ``(nrhs, n)``.

    Raises:
        ValueError: For ``weights`` without ``newton_tol``, or either on a
            route that is not iterative.
    """
    values = held.values
    if weights is not None or newton_tol is not None:
        if weights is None or newton_tol is None or not linear.inexact:
            raise ValueError(
                "weights= and newton_tol= set the linear tolerance of an iterative corrector "
                "route and go together: pass both on a route of an iterative backend, neither "
                "on an exact one."
            )
        return _iterative_solves(held.factors[0], linear.backend, weights, newton_tol)
    routes = linear.routes
    branches = [_route_solves(f, route, values, flagged) for f, route in zip(held.factors, routes)]
    if len(branches) == 1:
        return branches[0]

    def solve_active(rhs):
        return _lane_switch(rung, [branch[0] for branch in branches], rhs)

    def solve_many_active(rhs_many):
        return _lane_switch(rung, [branch[1] for branch in branches], rhs_many)

    return solve_active, solve_many_active


def _lane_switch(index: Any, branches: list, operand: Any) -> Any:
    """``lax.switch`` whose branches run under ``vmap`` only when some lane selects them.

    Under ``vmap`` with a per-lane ``index`` a ``lax.switch`` runs every
    branch on every lane; here a branch runs only when some lane selects it,
    so a batch whose lanes share one rung solves on that rung alone.
    """
    if len(branches) == 1:
        return branches[0](operand)
    return lane_cond(
        index == 0,
        branches[0],
        lambda op: _lane_switch(index - 1, branches[1:], op),
        operand,
    )


def linearize(
    held: HeldMatrix,
    values: jax.Array,
    rebuild: Any,
    *,
    linear: CorrectorLinear,
    rung: Any = None,
    rung_changed: Any = None,
    flagged: bool = False,
    weights: Any = None,
    newton_tol: Optional[float] = None,
) -> tuple[HeldMatrix, SolveFn, SolveFn]:
    """Refactor the held matrix where a setup replaced it and build its solves.

    [refactor_held][diff_soup.timestepping.vsvo.common.linear_ops.refactor_held]
    followed by [held_solves][diff_soup.timestepping.vsvo.common.linear_ops.held_solves];
    see them for the arguments. On an iterative route ``values`` is the
    current matrix at every attempt (the operator) and ``rebuild`` decides
    only whether the preconditioner is rebuilt from it.

    Returns:
        ``(held, solve, solve_many)``: the matrix in effect after this sweep,
        and callables solving with it for a right-hand side ``(n,)`` and a
        block ``(nrhs, n)``.
    """
    held = refactor_held(held, values, rebuild, linear=linear, rung=rung, rung_changed=rung_changed)
    solves = held_solves(
        held, linear=linear, rung=rung, flagged=flagged, weights=weights, newton_tol=newton_tol
    )
    return (held, *solves)


__all__ = [
    "CorrectorLinear",
    "HeldMatrix",
    "SolveFn",
    "corrector_linear",
    "empty_held",
    "held_solves",
    "linearize",
    "problem_linear",
    "refactor_held",
    "state_block_size",
]
