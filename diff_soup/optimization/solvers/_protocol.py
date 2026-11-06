"""The protocol of a solver whose whole loop compiles: init, step, terminate, run."""

from __future__ import annotations

import abc
import enum
import time
from typing import Any, ClassVar

import equinox as eqx
import jax
import numpy as np

from diff_soup.inference._abc import OptSolver


class SolveStatus(enum.IntEnum):
    """How a compiled run ended, as the integer `SolveRecord.status` carries under `jit`.

    Attributes:
        KKT_TOL: The scaled optimality error fell below the tolerance.
        MAX_STEPS: The step budget ran out.
        INFEASIBLE: The restoration phase converged, or the augmented Lagrangian's penalty reached its cap, at a point that still violates the constraints.
        RESTORATION_FAILED: The line search failed, and the restoration phase, where there is one, did not recover.
        SOLVER_FAILED: Consecutive trial evaluations failed, a NaN or a DAE solve that did not finish.
        DIVERGED: An iterate or a multiplier left the finite range.
    """

    KKT_TOL = 0
    MAX_STEPS = 1
    INFEASIBLE = 2
    RESTORATION_FAILED = 3
    SOLVER_FAILED = 4
    DIVERGED = 5


class SolverState(eqx.Module):
    """The state a compiled solver's loop carries, which each solver extends with its own fields.

    Attributes:
        x: The current point, `(n,)`.
        step: The number of steps taken.
        done: Whether this problem's run has ended.
        status: The [SolveStatus][diff_soup.optimization.SolveStatus] as an integer array.
    """

    x: Any
    step: Any
    done: Any
    status: Any


class SolveRecord(eqx.Module):
    """What a compiled run returns beside the solution, every leaf an array.

    Attributes:
        multipliers_equality: The equality multipliers at the solution, `(n_eq,)`.
        multipliers_inequality: The inequality multipliers, `(n_ineq,)`, at or above 0.
        slacks: The inequality slacks, `(n_ineq,)`.
        bound_multipliers: `(lower, upper)` bound multipliers, `(n,)` each.
        mu: The final barrier parameter, 0 for a solver without one.
        steps: The number of steps taken.
        failed_evaluations: The number of trial evaluations that failed.
        kkt_error: The scaled optimality error at the solution.
        constraint_violation: The largest constraint violation at the solution.
        status: The [SolveStatus][diff_soup.optimization.SolveStatus] as an integer array.
        degenerate: Whether a multiplier and its slack are both zero or the active rows are dependent, which makes the derivative the barrier solution's at `mu`.
    """

    multipliers_equality: Any
    multipliers_inequality: Any
    slacks: Any
    bound_multipliers: Any
    mu: Any
    steps: Any
    failed_evaluations: Any
    kkt_error: Any
    constraint_violation: Any
    status: Any
    degenerate: Any


class CompiledSolver(OptSolver):
    """A solver whose whole loop compiles, so `run` composes with `jit`, `vmap` and `grad`.

    `run` is the pure entry and returns arrays only. Under `vmap` a finished problem keeps its
    state while the others step, until every problem is done or the budget ends. `solve` wraps
    `run` and returns the `OptResult` the other solvers return.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient", "constraint_jacobian"})

    @abc.abstractmethod
    def init(self, problem: Any, x0: Any, args: Any, *, warm_start: Any = None) -> SolverState:
        """The state before the first step, at `x0`, or from a `warm_start` record and point."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, problem: Any, state: SolverState, args: Any) -> SolverState:
        """One step of the loop."""
        raise NotImplementedError

    @abc.abstractmethod
    def terminate(self, problem: Any, state: SolverState, args: Any) -> tuple[Any, Any]:
        """Whether the loop ends at `state`, and with which `SolveStatus`, both scalar arrays."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(
        self,
        problem: Any,
        x0: Any,
        args: Any,
        *,
        max_steps: int | None = None,
        warm_start: Any = None,
    ) -> tuple[Any, SolveRecord]:
        """Minimize from `x0` and return the solution and its record, arrays only.

        The derivative in `args` is the implicit one at the solution, and `x0` and `warm_start`
        carry none. A run that did not end with `KKT_TOL` has NaN tangents.

        Args:
            problem: The numeric problem.
            x0: The starting point, `(n,)`.
            args: The problem's arguments, which the derivative is taken in.
            max_steps: The step budget, or `None` for the solver's own.
            warm_start: The `SolveRecord` of an earlier run whose multipliers start this one, or `None`.

        Returns:
            `(x, record)`: the solution and its [SolveRecord][diff_soup.optimization.SolveRecord].
        """
        raise NotImplementedError

    def solve(self, problem: Any, *, callbacks: tuple = ()) -> Any:
        """Run on the host and return the `OptResult`, as `ds.inference.optsolve` expects.

        The run compiles once and has no host step: each callback's `on_start` sees the
        problem, `on_step` the final step and `on_end` the result. The `SolveRecord` is
        `result.info["record"]`.
        """
        from diff_soup.inference.fitting.result import OptResult, StepInfo

        args = getattr(problem, "args", None)
        x0 = problem.x0
        for callback in callbacks:
            callback.on_start(int(np.size(x0)), 0, int(getattr(self, "max_steps", 0)))
        start = time.perf_counter()
        x, record = eqx.filter_jit(self.run)(problem, x0, args)
        x = jax.block_until_ready(x)
        wall = time.perf_counter() - start
        status = _termination_status(int(record.status))
        cost = float(_problem_value(problem)(x, args))
        info = StepInfo(
            step=int(record.steps),
            cost=cost,
            grad_norm=float(record.kkt_error),
            params=np.asarray(x),
            accepted=True,
            delta_norm=float("nan"),
            wall_time=wall,
        )
        for callback in callbacks:
            callback.on_step(info)
        result = OptResult(
            opt_params=problem.fitted(x) if hasattr(problem, "fitted") else None,
            params=np.asarray(x),
            initial_params=np.asarray(x0),
            cost=cost,
            grad_norm=float(record.kkt_error),
            status=status,
            message=SolveStatus(int(record.status)).name.lower(),
            n_steps=int(record.steps),
            max_steps=int(getattr(self, "max_steps", 0)),
            n_accepted=int(record.steps),
            wall_time=wall,
            info={"record": jax.tree.map(np.asarray, record)},
        )
        for callback in callbacks:
            callback.on_end(result)
        return result


# Until `TerminationStatus` gains the compiled solvers' statuses, the nearest existing one.
_INTERIM = {"KKT_TOL": "SUCCESS", "INFEASIBLE": "STALLED", "RESTORATION_FAILED": "STALLED"}


def _termination_status(code: int) -> Any:
    """The `TerminationStatus` of a `SolveStatus` code, by name."""
    from diff_soup.inference.fitting.result import TerminationStatus

    name = SolveStatus(code).name
    if name not in TerminationStatus.__members__:
        name = _INTERIM[name]
    return TerminationStatus[name]


def _problem_value(problem: Any) -> Any:
    """The raw `(x, args)` objective of a numeric problem."""
    return problem._value
