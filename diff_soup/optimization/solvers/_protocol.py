"""The protocol of a solver whose whole loop compiles: init, step, terminate, run."""

from __future__ import annotations

import abc
import enum
from typing import Any, ClassVar

import equinox as eqx

from diff_soup.optimization._abc import OptSolver


class SolveStatus(enum.IntEnum):
    """How a compiled run ended, as the integer `SolveRecord.status` carries under `jit`.

    Attributes:
        KKT_TOL: The scaled optimality error fell below the tolerance.
        MAX_STEPS: The step budget ran out.
        INFEASIBLE: The restoration phase converged to a point that still violates the constraints.
        RESTORATION_FAILED: The restoration phase ran out of its own budget.
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
    """The carry of a compiled solver's loop; each solver subclasses it with its own leaves.

    Attributes:
        x: The current point, `(n,)`.
        step: The number of steps taken.
        done: Whether this lane has finished.
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
        degenerate: Whether an active inequality had both its multiplier and its slack at zero, in which case the implicit derivative is that of the barrier solution at `mu`.
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
    """A solver whose loop is `lax.while_loop` over `step`, so `run` composes with `jit`, `vmap` and `grad`.

    `run` is the pure entry and returns arrays only. Under `vmap` a finished lane is masked:
    `step` applies where `done` is false and the state is kept where it is true, and the loop
    runs until every lane is done or the budget ends. `solve` wraps `run` on the host for
    callbacks, diagnostics and the `OptResult` the other solvers return. The derivative of
    `run` with respect to `args` is the implicit one at the solution, by one `custom_jvp`
    linear in its tangents.
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
        """Minimize from `x0` and return the solution and its record, arrays only."""
        raise NotImplementedError

    def solve(self, problem: Any, *, callbacks: tuple = ()) -> Any:
        """Run on the host and return the `OptResult`, as `ds.optimization.optsolve` expects."""
        raise NotImplementedError
