"""The skeleton every compiled constrained solver shares: the start, termination, record and run.

[ConstrainedSolver][] runs the interior point, SQP and the augmented Lagrangian. The start
evaluates the problem once at the first point, takes the scales from that evaluation and
scales it arithmetically. The loop is one `lax.while_loop` over the solver's own step and the
shared termination test; under `vmap` its batching rule keeps a finished lane's state while
the others step. The record unscales the multipliers and reads the active set. Each solver
supplies only its start state, its step, and the slacks and barrier parameter of its record.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree

from diff_soup._adjoints import AbstractAdjoint
from diff_soup.optimization._abc import OptimizationProblem
from diff_soup.optimization.inference.fitting.result import TerminationStatus
from diff_soup.optimization.kkt import DenseKKT, KKTSolve
from diff_soup.optimization.solvers import _hessian
from diff_soup.optimization.solvers._barrier import Bounds, bounds_from
from diff_soup.optimization.solvers._closures import (
    Closures,
    Evaluation,
    Scales,
    bounds_of,
    closures_of,
    evaluate,
    problem_scales,
    scaled_closures,
)
from diff_soup.optimization.solvers._implicit import active_set, implicit_run
from diff_soup.optimization.solvers._optimality import constraint_violation, optimality_error
from diff_soup.optimization.solvers._protocol import (
    CompiledSolver,
    SolveRecord,
    SolverState,
    _int,
    _raw,
)

#: Ipopt's `diverging_iterates_tol`: an iterate entry beyond it ends the run with `DIVERGED`.
DIVERGED_X = 1e20


class CoreState(SolverState):
    """The fields every compiled constrained solver carries, in the scaled problem.

    Attributes:
        lam_e: The equality multipliers at `x`.
        lam_i: The inequality multipliers at `x`.
        z_lower: The lower-bound multipliers, 0 where the bound is absent.
        z_upper: The upper-bound multipliers, 0 where the bound is absent.
        evaluation: The objective, constraints and first derivatives at `x`.
        scales: The objective and constraint scales.
        kkt_error: The scaled optimality error at `x`.
        quasi_newton: The BFGS matrix, or `(0, 0)` when not used.
        quasi_newton_updates: How many updates the BFGS matrix has had.
        consecutive_failed: The failed trial evaluations in a row.
        failed_evaluations: The failed trial evaluations in all.
        inner_steps: The steps of the inner solves so far.
        subproblem_status: The status of the subproblem whose failure ended the run, else `UNKNOWN`.
    """

    lam_e: Float[Array, "n_eq"]
    lam_i: Float[Array, "n_ineq"]
    z_lower: Float[Array, "n"]
    z_upper: Float[Array, "n"]
    evaluation: Evaluation
    scales: Scales
    kkt_error: Float[Array, ""]
    quasi_newton: Float[Array, "k k"]
    quasi_newton_updates: Int[Array, ""]
    consecutive_failed: Int[Array, ""]
    failed_evaluations: Int[Array, ""]
    inner_steps: Int[Array, ""]
    subproblem_status: Int[Array, ""]


class Start(NamedTuple):
    """What the shared start hands a solver: one scaled evaluation and the warm multipliers.

    Attributes:
        x: The first point.
        evaluation: The scaled evaluation at `x`.
        scales: The scales.
        warm: `(lam_e, lam_i, z_lower, z_upper)` of a warm start in the scaled problem, or `None`.
    """

    x: Float[Array, "n"]
    evaluation: Evaluation
    scales: Scales
    warm: tuple | None


def core_fields(start: Start, closures: Closures, hessian: str) -> dict:
    """The `CoreState` fields at the start, with the cold multipliers 0 and the counters 0."""
    x = start.x
    dtype = x.dtype
    n = x.shape[0]
    if start.warm is None:
        zeros = jnp.zeros((n,), dtype)
        multipliers = (
            jnp.zeros((closures.n_equality,), dtype),
            jnp.zeros((closures.n_inequality,), dtype),
            zeros,
            zeros,
        )
    else:
        multipliers = start.warm
    valid = start.evaluation.finite
    lam_e, lam_i, z_lower, z_upper = multipliers
    return dict(
        x=x,
        step=_int(0),
        done=~valid,
        status=_int(jnp.where(valid, TerminationStatus.MAX_STEPS, TerminationStatus.SOLVER_FAILED)),
        lam_e=lam_e,
        lam_i=lam_i,
        z_lower=z_lower,
        z_upper=z_upper,
        evaluation=start.evaluation,
        scales=start.scales,
        kkt_error=jnp.zeros((), dtype),
        quasi_newton=_hessian.initial_matrix(hessian, closures, n, dtype),
        quasi_newton_updates=_int(0),
        consecutive_failed=_int(0),
        failed_evaluations=_int(0),
        inner_steps=_int(0),
        subproblem_status=_int(TerminationStatus.UNKNOWN),
    )


def warm_multipliers(record: SolveRecord, scales: Scales, dtype: jnp.dtype) -> tuple:
    """A record's `(lam_e, lam_i, z_lower, z_upper)` in the scaled problem."""
    objective = scales.objective
    lower, upper = record.bound_multipliers
    return (
        jnp.asarray(record.multipliers_equality, dtype) * objective / scales.equality,
        jnp.asarray(record.multipliers_inequality, dtype) * objective / scales.inequality,
        jnp.asarray(lower, dtype) * objective,
        jnp.asarray(upper, dtype) * objective,
    )


def lagrangian_secant(
    new: Evaluation, old: Evaluation, lam_e: Float[Array, "n_eq"], lam_i: Float[Array, "n_ineq"]
) -> Float[Array, "n"]:
    """`grad L(x_new, lam) - grad L(x, lam)`: the gradient change a quasi-Newton update reads."""
    return (
        new.gradient
        - old.gradient
        + (new.jacobian_equality - old.jacobian_equality).T @ lam_e
        + (new.jacobian_inequality - old.jacobian_inequality).T @ lam_i
    )


def slack_free_error(
    evaluation: Evaluation,
    x: Float[Array, "n"],
    lam_e: Float[Array, "n_eq"],
    lam_i: Float[Array, "n_ineq"],
    z_lower: Float[Array, "n"],
    z_upper: Float[Array, "n"],
    bounds: Bounds,
) -> Float[Array, ""]:
    """The scaled optimality error of a solver without slacks: slacks `max(-c_I, 0)` at `mu = 0`."""
    return optimality_error(
        evaluation,
        x,
        jnp.maximum(-evaluation.inequality, 0.0),
        lam_e,
        lam_i,
        z_lower,
        z_upper,
        jnp.maximum(lam_i, 0.0),
        0.0,
        bounds,
    )


class ConstrainedSolver(CompiledSolver):
    """A compiled constrained solver: the shared start, termination, record and loop."""

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})
    tol: eqx.AbstractVar[float]
    max_steps: eqx.AbstractVar[int]
    hessian: eqx.AbstractVar[str]
    adjoint: eqx.AbstractVar[AbstractAdjoint]

    @abc.abstractmethod
    def _place(self, x0: Float[Array, "n"], bounds: Bounds, warm: bool) -> Float[Array, "n"]:
        """The first point: `x0` moved into its bounds."""
        raise NotImplementedError

    @abc.abstractmethod
    def _begin(self, closures: Closures, bounds: Bounds, start: Start) -> CoreState:
        """The solver's state at the start, before the termination test."""
        raise NotImplementedError

    @abc.abstractmethod
    def _iterate(
        self, closures: Closures, bounds: Bounds, state: CoreState, args: PyTree
    ) -> CoreState:
        """One step on prepared closures, ending the run on a failure but not testing convergence."""
        raise NotImplementedError

    @property
    def _kkt(self) -> KKTSolve:
        """The backend of the implicit derivative's KKT system."""
        return DenseKKT()

    def _check(self, raw: Closures) -> None:
        """Refuse options the problem cannot serve, before anything is traced."""
        _hessian.check_mode(self.hessian, raw)

    def _scales(
        self, problem: OptimizationProblem, raw: Closures, evaluation: Evaluation
    ) -> Scales:
        """The problem's own scales, else the gradient-based ones."""
        return problem_scales(problem, raw, evaluation)

    def _slacks(self, state: CoreState) -> Float[Array, "n_ineq"]:
        """The inequality slacks of the record: `max(-c_I, 0)`."""
        return jnp.maximum(-state.evaluation.inequality, 0.0)

    def _mu(self, state: CoreState) -> Float[Array, ""]:
        """The barrier parameter of the record: 0."""
        return jnp.zeros((), state.x.dtype)

    def _converged(self, state: CoreState) -> Bool[Array, ""]:
        """Whether the scaled optimality error is below `tol`."""
        return state.kkt_error <= self.tol

    def _prepare(
        self, problem: OptimizationProblem, like: Float[ArrayLike, "n"]
    ) -> tuple[Closures, Closures, Bounds]:
        """The raw closures, the scaled closures and the bounds of `problem`."""
        raw = closures_of(problem)
        self._check(raw)
        return raw, scaled_closures(raw), bounds_from(*bounds_of(problem, like))

    def _init(
        self,
        problem: OptimizationProblem,
        raw: Closures,
        closures: Closures,
        bounds: Bounds,
        x0: Float[ArrayLike, "n"],
        args: PyTree,
        warm_start: SolveRecord | None,
    ) -> CoreState:
        """The state at the first point, from one evaluation of the problem as given."""
        x = self._place(jnp.asarray(x0), bounds, warm_start is not None)
        unscaled = evaluate(raw, x, args, _hessian.uses_gauss_newton(self.hessian, raw))
        scales = self._scales(problem, raw, unscaled)
        warm = None if warm_start is None else warm_multipliers(warm_start, scales, x.dtype)
        state = self._begin(closures, bounds, Start(x, unscaled.scaled(scales), scales, warm))
        return self._with_termination(state)

    def init(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, "n"],
        args: PyTree,
        *,
        warm_start: SolveRecord | None = None,
    ) -> CoreState:
        """The state at `x0` moved into its bounds, or from a `warm_start` record's multipliers."""
        raw, closures, bounds = self._prepare(problem, x0)
        return self._init(problem, raw, closures, bounds, x0, args, warm_start)

    def step(self, problem: OptimizationProblem, state: CoreState, args: PyTree) -> CoreState:
        """One step, without the termination test."""
        _, closures, bounds = self._prepare(problem, state.x)
        return self._iterate(closures, bounds, state, (args, state.scales))

    def terminate(
        self, problem: OptimizationProblem | None, state: CoreState, args: PyTree
    ) -> tuple[Bool[Array, ""], Int[Array, ""]]:
        """End on the scaled optimality error, the budget, divergence, or an earlier failure."""
        converged = self._converged(state)
        finite = (
            jnp.all(jnp.isfinite(state.x))
            & jnp.all(jnp.isfinite(state.lam_e))
            & jnp.all(jnp.isfinite(state.lam_i))
            & jnp.isfinite(state.kkt_error)
        )
        diverged = ~finite | (jnp.max(jnp.abs(state.x), initial=0.0) > DIVERGED_X)
        over = state.step >= self.max_steps
        status = jnp.where(
            state.done,
            state.status,
            jnp.where(
                converged,
                TerminationStatus.KKT_TOL,
                jnp.where(diverged, TerminationStatus.DIVERGED, TerminationStatus.MAX_STEPS),
            ),
        )
        return state.done | converged | diverged | over, _int(status)

    def _with_termination(self, state: CoreState) -> CoreState:
        """`state` with `done` and `status` from `terminate`."""
        done, status = self.terminate(None, state, None)
        return eqx.tree_at(lambda s: (s.done, s.status), state, (done, status))

    def _advance(
        self, closures: Closures, bounds: Bounds, state: CoreState, args: PyTree
    ) -> CoreState:
        """One step on prepared closures, with the termination test."""
        return self._with_termination(self._iterate(closures, bounds, state, args))

    def run(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, "n"],
        args: PyTree,
        *,
        max_steps: int | None = None,
        warm_start: SolveRecord | None = None,
    ) -> tuple[Float[Array, "n"], SolveRecord]:
        """The compiled loop, as [CompiledSolver.run][diff_soup.optimization.solvers.CompiledSolver.run] describes."""
        solver = self if max_steps is None else dataclasses.replace(self, max_steps=max_steps)
        return implicit_run(
            solver._run_primal, _raw, self._kkt, problem, x0, args, warm_start, self.adjoint
        )

    def _run_primal(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, "n"],
        args: PyTree,
        warm_start: SolveRecord | None,
    ) -> tuple[Float[Array, "n"], SolveRecord]:
        """The loop itself, never differentiated."""
        raw, closures, bounds = self._prepare(problem, x0)
        state = self._init(problem, raw, closures, bounds, x0, args, warm_start)

        def body(current):
            return self._advance(closures, bounds, current, (args, current.scales))

        state = lax.while_loop(lambda current: ~current.done, body, state)
        return state.x, self._record(state, bounds)

    def _record(self, state: CoreState, bounds: Bounds) -> SolveRecord:
        """The `SolveRecord` of a finished state, unscaled."""
        scales = state.scales
        objective = scales.objective
        evaluation = state.evaluation
        record = SolveRecord(
            multipliers_equality=state.lam_e * scales.equality / objective,
            multipliers_inequality=state.lam_i * scales.inequality / objective,
            slacks=self._slacks(state) / scales.inequality,
            bound_multipliers=(state.z_lower / objective, state.z_upper / objective),
            mu=self._mu(state),
            value=evaluation.value / objective,
            steps=state.step,
            inner_steps=state.inner_steps,
            failed_evaluations=state.failed_evaluations,
            kkt_error=state.kkt_error,
            constraint_violation=constraint_violation(evaluation, scales),
            status=state.status,
            subproblem_status=state.subproblem_status,
            degenerate=jnp.asarray(False),
        )
        degenerate = active_set(
            state.x, record, bounds, evaluation.jacobian_equality, evaluation.jacobian_inequality
        ).degenerate
        return eqx.tree_at(lambda r: r.degenerate, record, degenerate)
