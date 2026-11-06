"""The protocol of a solver whose whole loop compiles: init, step, terminate, run."""

from __future__ import annotations

import abc
import time
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree

from diff_soup.optimization._abc import OptimizationProblem, OptSolver
from diff_soup.optimization.inference.fitting.result import OptResult, StepInfo, TerminationStatus
from diff_soup.optimization.solvers._barrier import Bounds, bounds_from
from diff_soup.optimization.solvers._closures import Closures, bounds_of, closures_of

#: Ipopt's `mu_init`, the barrier parameter a record of zeros carries.
MU_INIT = 0.1


def _int(value: ArrayLike) -> Int[Array, ""]:
    """`value` as an int32 array, the dtype of every counter and status in a state."""
    return jnp.asarray(value, dtype=jnp.int32)


def _select(pred: Bool[Array, ""], new: PyTree, old: PyTree) -> PyTree:
    """`new` where `pred`, else `old`, leaf by leaf."""
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), new, old)


def _raw(problem: OptimizationProblem, x: Float[Array, " n"]) -> tuple[Closures, Bounds]:
    """The raw closures and the bounds of `problem`, for the implicit derivative."""
    return closures_of(problem), bounds_from(*bounds_of(problem, x))


class SolverState(eqx.Module):
    """The state a compiled solver's loop carries, which each solver extends with its own fields.

    Attributes:
        x: The current point.
        step: The number of steps taken.
        done: Whether this problem's run has ended.
        status: The [TerminationStatus][diff_soup.optimization.TerminationStatus] code.
    """

    x: Float[Array, " n"]
    step: Int[Array, ""]
    done: Bool[Array, ""]
    status: Int[Array, ""]


class SolveRecord(eqx.Module):
    """What a compiled run returns beside the solution, every leaf an array.

    Attributes:
        multipliers_equality: The equality multipliers at the solution.
        multipliers_inequality: The inequality multipliers, at or above 0.
        slacks: The inequality slacks.
        bound_multipliers: `(lower, upper)` bound multipliers.
        mu: The final barrier parameter, 0 for a solver without one.
        value: The objective at the solution.
        steps: The number of steps taken.
        inner_steps: The steps of the inner solves: the restoration phase, SQP's QPs, the augmented Lagrangian's inner minimizations.
        failed_evaluations: The number of trial evaluations that failed.
        kkt_error: The scaled optimality error at the solution.
        constraint_violation: The largest constraint violation at the solution.
        status: The [TerminationStatus][diff_soup.optimization.TerminationStatus] code.
        subproblem_status: The status code of the subproblem whose failure ended the run with `SOLVER_FAILED` (SQP's QP), else `UNKNOWN`.
        degenerate: Whether a multiplier and its slack are both zero or the active rows are dependent, which makes the derivative the barrier solution's at `mu`.
    """

    multipliers_equality: Float[Array, " n_eq"]
    multipliers_inequality: Float[Array, " n_ineq"]
    slacks: Float[Array, " n_ineq"]
    bound_multipliers: tuple[Float[Array, " n"], Float[Array, " n"]]
    mu: Float[Array, ""]
    value: Float[Array, ""]
    steps: Int[Array, ""]
    inner_steps: Int[Array, ""]
    failed_evaluations: Int[Array, ""]
    kkt_error: Float[Array, ""]
    constraint_violation: Float[Array, ""]
    status: Int[Array, ""]
    subproblem_status: Int[Array, ""]
    degenerate: Bool[Array, ""]

    @classmethod
    def zeros(
        cls, n: int, n_eq: int, n_ineq: int, *, dtype: DTypeLike = jnp.float64
    ) -> SolveRecord:
        """A record with zero multipliers and slacks, `mu` at `MU_INIT` and the status `UNKNOWN`.

        It has the shapes and dtypes of a run's record, so a loop that hands records on can
        start from it.

        Args:
            n: The number of decision variables.
            n_eq: The number of equality rows.
            n_ineq: The number of inequality rows.
            dtype: The floating dtype of the problem.

        Returns:
            The record.
        """
        zero = jnp.zeros((), dtype)
        unknown = _int(TerminationStatus.UNKNOWN)
        return cls(
            multipliers_equality=jnp.zeros((n_eq,), dtype),
            multipliers_inequality=jnp.zeros((n_ineq,), dtype),
            slacks=jnp.zeros((n_ineq,), dtype),
            bound_multipliers=(jnp.zeros((n,), dtype), jnp.zeros((n,), dtype)),
            mu=jnp.asarray(MU_INIT, dtype),
            value=zero,
            steps=_int(0),
            inner_steps=_int(0),
            failed_evaluations=_int(0),
            kkt_error=zero,
            constraint_violation=zero,
            status=unknown,
            subproblem_status=unknown,
            degenerate=jnp.asarray(False),
        )


class CompiledSolver(OptSolver):
    """A solver whose whole loop compiles, so `run` composes with `jit`, `vmap` and `grad`.

    `run` is the pure entry and returns arrays only. Under `vmap` a finished problem keeps its
    state while the others step, until every problem is done or the budget ends. `solve` wraps
    `run` and returns the `OptResult` the other solvers return.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})
    max_steps: eqx.AbstractVar[int]

    @abc.abstractmethod
    def init(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, " n"],
        args: PyTree,
        *,
        warm_start: SolveRecord | None = None,
    ) -> SolverState:
        """The state before the first step, at `x0`, or from a `warm_start` record and point."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, problem: OptimizationProblem, state: SolverState, args: PyTree) -> SolverState:
        """One step of the loop."""
        raise NotImplementedError

    @abc.abstractmethod
    def terminate(
        self, problem: OptimizationProblem, state: SolverState, args: PyTree
    ) -> tuple[Bool[Array, ""], Int[Array, ""]]:
        """Whether the loop ends at `state`, and with which `TerminationStatus` code."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, " n"],
        args: PyTree,
        *,
        max_steps: int | None = None,
        warm_start: SolveRecord | None = None,
    ) -> tuple[Float[Array, " n"], SolveRecord]:
        """Minimize from `x0` and return the solution and its record, arrays only.

        The derivative in `args` is the implicit one at the solution, and `x0` and `warm_start`
        carry none. A run that did not end with `KKT_TOL` has NaN tangents.

        Args:
            problem: The numeric problem.
            x0: The starting point.
            args: The problem's arguments, which the derivative is taken in.
            max_steps: The step budget, or `None` for the solver's own.
            warm_start: The `SolveRecord` of an earlier run whose multipliers start this one, or `None`.

        Returns:
            `(x, record)`: the solution and its [SolveRecord][diff_soup.optimization.SolveRecord].
        """
        raise NotImplementedError

    def solve(self, problem: OptimizationProblem, *, callbacks: tuple = ()) -> OptResult:
        """Run on the host and return the `OptResult`, as `ds.optimization.optsolve` expects.

        The run compiles once and has no host step: each callback's `on_start` sees the
        problem, `on_step` the final step and `on_end` the result. The `SolveRecord` is
        `result.info["record"]` and the scaled optimality error `result.info["kkt_error"]`.
        """
        args = problem.args
        x0 = problem.x0
        for callback in callbacks:
            callback.on_start(int(np.size(x0)), 0, self.max_steps)
        start = time.perf_counter()
        x, record = eqx.filter_jit(self.run)(problem, x0, args)
        x = jax.block_until_ready(x)
        wall = time.perf_counter() - start
        record = jax.tree.map(np.asarray, record)
        status = TerminationStatus(int(record.status))
        cost = float(record.value)
        info = StepInfo(
            step=int(record.steps),
            cost=cost,
            grad_norm=float("nan"),
            params=np.asarray(x),
            accepted=True,
            delta_norm=float("nan"),
            wall_time=wall,
        )
        for callback in callbacks:
            callback.on_step(info)
        result = OptResult(
            opt_params=problem.fitted(x),
            params=np.asarray(x),
            initial_params=np.asarray(x0),
            cost=cost,
            grad_norm=float("nan"),
            status=status,
            message=status.name.lower(),
            n_steps=int(record.steps),
            max_steps=self.max_steps,
            wall_time=wall,
            info={"record": record, "kkt_error": float(record.kkt_error)},
        )
        for callback in callbacks:
            callback.on_end(result)
        return result
