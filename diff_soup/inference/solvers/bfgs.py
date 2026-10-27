"""bfgs.py — JAX-first L-BFGS for PDE-constrained minimization.

Uses backtracking Armijo line search with two-loop L-BFGS recursion
(all operations via ``jnp``, stays on device between steps).
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
from diff_soup.inference.fitting.result import EvalCounts, OptResult, StepInfo, TerminationStatus
from diff_soup.inference.solvers._termination import (
    DEFAULT_STEP_TOL,
    gradient_tolerance,
    relative_tolerance,
    step_budget,
)


class BFGS(OptSolver):
    """The limited-memory BFGS method, quasi-Newton steps with a backtracking line search.

    It needs the objective and its gradient. Box bounds are kept by projecting each line-search
    trial. A failed line search at a non-finite objective or gradient ends the run with
    `DIVERGED`.

    Args:
        max_steps: The step budget.
        grad_tol: The gradient-norm tolerance, or `None` for the dtype's default.
        step_tol: The step-norm tolerance relative to the parameter norm, or `None` for the dtype's default.
        c1: The sufficient-decrease parameter of the line search.
        max_linesearch: The number of step halvings the line search tries per step.
        memory: The number of past steps the curvature estimate is built from.
        ad: `"reverse"` takes the gradient from the reverse adjoint, `"forward"` from the forward-mode Jacobian.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient"})

    max_steps: int = eqx.field(static=True, default=200)
    grad_tol: float | None = eqx.field(static=True, default=None)
    step_tol: float | None = eqx.field(static=True, default=None)
    c1: float = eqx.field(static=True, default=1e-4)
    max_linesearch: int = eqx.field(static=True, default=20)
    memory: int = eqx.field(static=True, default=10)
    ad: str = eqx.field(static=True, default="reverse")
    callbacks: tuple = eqx.field(static=True, default=())

    def solve(self, problem: Any, *, callbacks: tuple = ()) -> OptResult:
        """Run L-BFGS with Armijo line search on ``problem``.

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
            A [OptResult][diff_soup.inference.fitting.result.OptResult] populated with
            the final parameters, cost/gradient histories, timing, and termination
            status.
        """
        cbs = tuple(self.callbacks) + tuple(callbacks)
        x0 = problem.x0
        n_params = len(x0)
        reg = NoReg() if problem.regularization is None else problem.regularization
        n_seg = int(problem.n_segments)

        data_loss_and_grad = problem.value_and_gradient
        data_loss = problem.value

        def loss_and_grad(x):
            loss, grad = data_loss_and_grad(x)
            reg_loss, reg_grad = jax.value_and_grad(reg.penalty)(x)
            return loss + reg_loss, grad + reg_grad, jnp.isfinite(loss)

        def loss_only(x):
            return data_loss(x) + reg.penalty(x)

        # --- L-BFGS two-loop recursion (pure JAX) -------------------------
        def _lbfgs_direction(grad, S, Y, m):
            """Two-loop recursion.  S, Y: (memory, n), m: active count."""
            q = grad
            # the recursion scratch (alphas/rhos) follows ``grad``'s
            # dtype so an fp32 stream stays fp32 through the fori_loop carry
            # (a default jnp.zeros would be fp64 under x64 and corrupt the
            # carry-type match).
            alphas = jnp.zeros(self.memory, dtype=grad.dtype)
            rhos = jnp.zeros(self.memory, dtype=grad.dtype)

            def backward_body(i, carry):
                q, alphas, rhos = carry
                idx = m - 1 - i
                s_i, y_i = S[idx], Y[idx]
                rho_i = 1.0 / jnp.maximum(y_i @ s_i, 1e-30)
                alpha_i = rho_i * (s_i @ q)
                q = q - alpha_i * y_i
                rhos = rhos.at[idx].set(rho_i)
                alphas = alphas.at[idx].set(alpha_i)
                return q, alphas, rhos

            q, alphas, rhos = jax.lax.fori_loop(0, m, backward_body, (q, alphas, rhos))

            # Scale by most recent curvature
            gamma = jnp.where(
                m > 0,
                (S[m - 1] @ Y[m - 1]) / jnp.maximum(Y[m - 1] @ Y[m - 1], 1e-30),
                1.0,
            )
            r = gamma * q

            # Iterate over active entries (oldest → newest).
            def forward_body2(i, r):
                s_i, y_i = S[i], Y[i]
                beta = rhos[i] * (y_i @ r)
                return r + s_i * (alphas[i] - beta)

            start_idx = jnp.maximum(m - self.memory, 0)
            r = jax.lax.fori_loop(start_idx, m, forward_body2, r)
            return r

        # --- Box bounds ---------------------------------------------------
        # When the problem declares bounds, project each line-search trial onto
        # the box (projected-Armijo). The quasi-Newton curvature memory is left
        # untouched — standard projected L-BFGS accepts the (s, y) pairs implied by
        # the projected iterates. The initial point is clipped once.
        bnds = problem.bounds
        lb = ub = None
        if bnds is not None:
            lb, ub = (jnp.asarray(side) for side in bnds)

        # --- Initialise ---------------------------------------------------
        for cb in cbs:
            cb.on_start(n_params, 0, self.max_steps)

        # no hard fp64 cast on the param path — ``x0``'s dtype flows
        # through. The L-BFGS history (S, Y) follows ``x``'s dtype.
        x = jnp.asarray(x0)
        max_steps = step_budget(self.max_steps, label="BFGS")
        grad_tolerance = gradient_tolerance(self.grad_tol, x.dtype, label="BFGS")
        grad_tol = None
        step_tol = relative_tolerance(
            self.step_tol, DEFAULT_STEP_TOL, x.dtype, name="step_tol", label="BFGS"
        )
        if bnds is not None:
            x = jnp.clip(x, lb, ub)
        t0_wall = time.perf_counter()

        loss, grad, _ = loss_and_grad(x)
        cost_history, grad_history = [], []
        n_feval, n_geval = 1, 1

        S = jnp.zeros((self.memory, n_params), dtype=x.dtype)
        Y = jnp.zeros((self.memory, n_params), dtype=x.dtype)
        m_active = 0  # number of active (s,y) pairs

        status = TerminationStatus.MAX_STEPS
        n_accepted = 0
        # ``max_steps=0`` makes no step: the start point is returned.
        n_steps = 0

        # --- Main loop ----------------------------------------------------
        for step in range(max_steps):
            n_steps = step + 1
            grad_norm = float(local_value(jnp.linalg.norm(grad)))
            cost_history.append(float(local_value(loss)))
            grad_history.append(grad_norm)

            # At an active bound the ordinary gradient does not vanish; stop on the
            # projected-gradient norm (the box-constrained stationarity measure).
            if bnds is not None:
                stop_norm = float(local_value(projected_gradient_norm(x, grad, lb, ub)))
            else:
                stop_norm = grad_norm
            if grad_tol is None:
                grad_tol = grad_tolerance.threshold(stop_norm)
            if assert_decision_value(stop_norm < grad_tol, label="BFGS"):
                status = TerminationStatus.GRAD_TOL
                break

            # L-BFGS direction
            if m_active == 0:
                d = -grad
            else:
                d = -_lbfgs_direction(grad, S, Y, m_active)

            # A stale or ill-scaled curvature history can cease to be a
            # descent model. Reset it before line search rather than asking
            # Armijo to accept a direction whose directional derivative is
            # non-negative.
            dg = float(local_value(grad @ d))
            if not np.isfinite(local_value(dg)) or dg >= 0.0:
                d = -grad
                S = jnp.zeros_like(S)
                Y = jnp.zeros_like(Y)
                m_active = 0

            # Backtracking Armijo (projected trials when bounds are active)
            alpha = 1.0
            loss_new = loss
            line_search_succeeded = False
            for _ in range(self.max_linesearch):
                x_new = x + alpha * d
                if bnds is not None:
                    x_new = jnp.clip(x_new, lb, ub)
                loss_new = loss_only(x_new)
                n_feval += 1
                realized_slope = float(local_value(grad @ (x_new - x)))
                if np.isfinite(local_value(float(local_value(loss_new)))) and float(
                    local_value(loss_new)
                ) <= (float(local_value(loss)) + self.c1 * realized_slope):
                    line_search_succeeded = True
                    break
                alpha *= 0.5

            if not line_search_succeeded:
                # ``x_new`` holds the smallest trial the backtracking reached.
                # A non-finite trial loss or gradient is a genuine divergence.
                # A finite trial that only missed the sufficient-decrease test
                # is a stalled search: when its step is already below the step
                # tolerance, or the loss moved by no more than the evaluation's
                # own rounding, the run has reached its floor and terminates on
                # the step tolerance without taking the failed step. Otherwise
                # the smallest trial is taken and the tolerances end the run.
                trial_loss = float(local_value(loss_new))
                trial_step = float(local_value(jnp.linalg.norm(x_new - x)))
                if not (
                    np.isfinite(local_value(trial_loss)) and np.isfinite(local_value(grad_norm))
                ):
                    stalled_status = TerminationStatus.DIVERGED
                else:
                    roundoff = float(jnp.finfo(x.dtype).eps) * max(
                        1.0, abs(float(local_value(loss)))
                    )
                    at_floor = (
                        trial_step < step_tol * (1.0 + float(local_value(jnp.linalg.norm(x))))
                        or abs(trial_loss - float(local_value(loss))) <= roundoff
                    )
                    stalled_status = TerminationStatus.STEP_TOL if at_floor else None
                if assert_decision_value(stalled_status is not None, label="BFGS"):
                    status = stalled_status
                    info = StepInfo(
                        step=step,
                        cost=float(local_value(loss)),
                        grad_norm=grad_norm,
                        params=local_value(x),
                        accepted=False,
                        delta_norm=0.0,
                        wall_time=time.perf_counter() - t0_wall,
                    )
                    for cb in cbs:
                        cb.on_step(info)
                    break

            # The realized step is the displacement actually taken, which the
            # box projection and an exhausted backtracking both shorten.
            dnorm = float(local_value(jnp.linalg.norm(x_new - x)))
            accepted = assert_decision_value(
                float(local_value(loss_new)) < float(local_value(loss)), label="BFGS accept"
            )

            grad_prev = grad
            x_prev = x
            x = x_new
            loss, grad, _ = loss_and_grad(x)
            n_feval += 1
            n_geval += 1

            if accepted:
                s = x - x_prev
                y = grad - grad_prev
                ys = float(local_value(y @ s))
                if ys > 1e-10:
                    if m_active < self.memory:
                        S = S.at[m_active].set(s)
                        Y = Y.at[m_active].set(y)
                        m_active += 1
                    else:
                        S = jnp.concatenate((S[1:], s[None, :]), axis=0)
                        Y = jnp.concatenate((Y[1:], y[None, :]), axis=0)
                n_accepted += 1
            else:
                S = jnp.zeros_like(S)
                Y = jnp.zeros_like(Y)
                m_active = 0

            info = StepInfo(
                step=step,
                cost=float(local_value(loss)),
                grad_norm=grad_norm,
                params=local_value(x),
                accepted=accepted,
                delta_norm=dnorm,
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

            if assert_decision_value(
                dnorm < step_tol * (1.0 + float(local_value(jnp.linalg.norm(x)))), label="BFGS"
            ):
                status = TerminationStatus.STEP_TOL
                break

        wall_time = time.perf_counter() - t0_wall

        fitted = problem.fitted(x)

        final_grad_norm = float(local_value(jnp.linalg.norm(grad)))
        final_projected_grad_norm = (
            final_grad_norm
            if bnds is None
            else float(local_value(projected_gradient_norm(x, grad, lb, ub)))
        )

        result = OptResult(
            opt_params=fitted,
            params=local_value(x),
            initial_params=local_value(x0),
            cost=float(local_value(loss)),
            grad_norm=final_grad_norm,
            projected_grad_norm=final_projected_grad_norm,
            status=status,
            message=status.value,
            n_steps=n_steps,
            max_steps=self.max_steps,
            n_accepted=n_accepted,
            eval_counts=EvalCounts(
                residual_evals=n_feval,
                gradient_evals=n_geval,
                pde_solves=(n_feval + n_geval) * n_seg,
            ),
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
