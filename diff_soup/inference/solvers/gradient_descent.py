"""gradient_descent.py — JAX-first gradient descent with momentum.

All parameter updates via ``jnp`` (stays on device).
Supports batched segment evaluation through the problem's batching policy.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import any_process, assert_decision_value, local_value
from diff_soup.inference._abc import OptSolver
from diff_soup.inference.fitting.bounds import projected_gradient_norm
from diff_soup.inference.fitting.callbacks import StopFit
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import OptResult, StepInfo, TerminationStatus
from diff_soup.inference.solvers._termination import gradient_tolerance, step_budget


class GradientDescent(OptSolver):
    """Gradient descent, steps along the negative gradient with a fixed learning rate and optional momentum.

    It needs the objective and its gradient. Box bounds are kept by projecting each iterate.

    Args:
        lr: The learning rate.
        momentum: The heavy-ball momentum coefficient, where `0` is plain gradient descent.
        max_steps: The step budget.
        grad_tol: The gradient-norm tolerance, or `None` for the dtype's default.
        ad: `"reverse"` takes the gradient from the reverse adjoint, `"forward"` from the forward-mode Jacobian.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.
    """

    # The least-squares gradient is J^T r, which a sharded joint reduces across the
    # device mesh without gathering the residual or Jacobian.
    requires: ClassVar[frozenset] = frozenset({"value_gradient"})

    lr: float = eqx.field(static=True, default=1e-3)
    momentum: float = eqx.field(static=True, default=0.0)
    max_steps: int = eqx.field(static=True, default=1000)
    grad_tol: float | None = eqx.field(static=True, default=None)
    ad: str = eqx.field(static=True, default="reverse")
    callbacks: tuple = eqx.field(static=True, default=())

    def solve(self, problem: Any, *, callbacks: tuple = ()) -> OptResult:
        """Run gradient descent (optionally with heavy-ball momentum).

        Args:
            problem: An [OptimizationProblem][diff_soup.inference._abc.OptimizationProblem]
                with ``"value_gradient"``: its ``value_and_gradient`` (the
                lowering differentiates it in this solver's ``ad`` mode), its
                ``regularization`` penalty, added to the objective, and its
                ``bounds``, which project every iterate.
            callbacks: Zero or more extra callback objects, merged with the
                solver's configured ``callbacks``.  Each callback's
                ``on_start``, ``on_step``, and ``on_end`` methods are invoked
                during the solve.  Must be passed as a keyword argument.

        Returns:
            A [OptResult][] with final parameters, cost/gradient
            histories, timing, and termination status.
        """
        cbs = tuple(self.callbacks) + tuple(callbacks)
        x0 = problem.x0
        n_params = len(x0)
        reg = NoReg() if problem.regularization is None else problem.regularization
        data_loss_and_grad = problem.value_and_gradient

        def loss_and_grad(x):
            loss, grad = data_loss_and_grad(x)
            reg_loss, reg_grad = jax.value_and_grad(reg.penalty)(x)
            return loss + reg_loss, grad + reg_grad

        # --- Box bounds ---------------------------------------------------
        # When the problem declares bounds, project every iterate onto the box
        # in this outer loop (the replicated flat x); the initial point is clipped
        # once so an infeasible x0 starts feasible.
        bnds = problem.bounds

        # --- Initialise ---------------------------------------------------
        for cb in cbs:
            cb.on_start(n_params, 0, self.max_steps)

        # no hard fp64 cast on the param path — ``x0``'s dtype flows.
        x = jnp.asarray(x0)
        max_steps = step_budget(self.max_steps, label="GradientDescent")
        grad_tolerance = gradient_tolerance(self.grad_tol, x.dtype, label="GradientDescent")
        grad_tol = None
        if bnds is not None:
            lb, ub = (jnp.asarray(side) for side in bnds)
            x = jnp.clip(x, lb, ub)
        v = jnp.zeros_like(x)
        t0_wall = time.perf_counter()
        cost_history, grad_history = [], []
        status = TerminationStatus.MAX_STEPS
        n_steps = 0

        # --- Main loop ----------------------------------------------------
        for step in range(max_steps):
            n_steps = step + 1
            loss, grad = loss_and_grad(x)
            grad_norm = float(local_value(jnp.linalg.norm(grad)))
            cost_history.append(float(local_value(loss)))
            grad_history.append(grad_norm)

            # At an active bound the ordinary gradient does not vanish; the
            # projected-gradient norm is the correct box-constrained stationarity
            # measure to stop on.
            if bnds is not None:
                stop_norm = float(local_value(projected_gradient_norm(x, grad, lb, ub)))
            else:
                stop_norm = grad_norm
            if grad_tol is None:
                grad_tol = grad_tolerance.threshold(stop_norm)
            if assert_decision_value(stop_norm < grad_tol, label="gradient descent"):
                status = TerminationStatus.GRAD_TOL
                break

            v = self.momentum * v - self.lr * grad
            x = x + v
            if bnds is not None:
                x = jnp.clip(x, lb, ub)

            info = StepInfo(
                step=step,
                cost=float(local_value(loss)),
                grad_norm=grad_norm,
                params=local_value(x),
                accepted=True,
                delta_norm=float(local_value(jnp.linalg.norm(v))),
                wall_time=time.perf_counter() - t0_wall,
            )
            stop_requested = False
            for cb in cbs:
                try:
                    cb.on_step(info)
                except StopFit:
                    stop_requested = True
            if any_process(stop_requested):
                status = TerminationStatus.STOPPED
                break

        if status in (TerminationStatus.MAX_STEPS, TerminationStatus.STOPPED):
            # The loop ended after a step it has not evaluated (or, with
            # ``max_steps=0``, before any): the cost and gradient of the point
            # returned.
            loss, grad = loss_and_grad(x)
            grad_norm = float(local_value(jnp.linalg.norm(grad)))
        wall_time = time.perf_counter() - t0_wall

        fitted = problem.fitted(x)

        result = OptResult(
            opt_params=fitted,
            params=local_value(x),
            initial_params=local_value(x0),
            cost=float(local_value(loss)),
            grad_norm=grad_norm,
            status=status,
            message=status.value,
            n_steps=n_steps,
            max_steps=self.max_steps,
            n_accepted=n_steps,
            cost_history=np.array(cost_history),
            grad_history=np.array(grad_history),
            wall_time=wall_time,
            info={"ad_mode": self.ad if problem.ad is None else problem.ad},
        )

        # Release callbacks so any open resources (e.g. a LogCallback's file
        # handle) are properly closed.
        for cb in cbs:
            cb.on_end(result)
        return result
