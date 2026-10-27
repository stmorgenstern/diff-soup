"""optax_solver.py — Optax-based solver for neural-network / UDE parameters.

For problems with many parameters (1000s — e.g., MLP weights in a UDE),
gradient-based optimizers like Adam/AdamW are more practical than LM or BFGS.
This solver wraps optax's gradient transformations in the standard equinox
training loop pattern.

The ``ad`` option selects reverse- or forward-mode differentiation through the
same scalar loss used by the native first-order solvers.

Usage::

    import optax
    import diff_soup as ds

    problem = ds.inference.InferenceProblem(
        problem, series, solver=solver_config, parameters=groups
    )
    solver = ds.inference.Optax(optax.adamw(1e-3, weight_decay=1e-4), max_steps=5000)
    result = ds.inference.optsolve(problem, solver)
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._optional import import_optional
from diff_soup.inference._abc import OptSolver
from diff_soup.inference.fitting.bounds import projected_gradient_norm
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import OptResult, StepInfo, TerminationStatus
from diff_soup.inference.solvers._termination import gradient_tolerance


class Optax(OptSolver):
    """An [Optax](https://optax.readthedocs.io/) optimizer, first-order steps from a gradient transformation such as `optax.adamw`.

    It needs the objective and its gradient, and the optional `optax` dependency. Box bounds are
    kept by projection after each update. The result reports the evaluated point of lowest
    objective.

    Args:
        optimizer: The Optax gradient transformation, `optax.adam(1e-3)` when `None`.
        max_steps: The step budget.
        grad_tol: The gradient-norm tolerance, or `None` for the dtype's default.
        log_every: Ignored.
        ad: `"reverse"` takes the gradient from the reverse adjoint, `"forward"` from the forward-mode Jacobian.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.

    !!! example

        ```python
        import optax

        solver = ds.inference.Optax(optimizer=optax.adamw(1e-2), max_steps=200)
        result = ds.inference.optsolve(infproblem, solver)
        ```
    """

    requires: ClassVar[frozenset] = frozenset({"value_gradient"})

    optimizer: Any = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=5000)
    grad_tol: float | None = eqx.field(static=True, default=None)
    log_every: int = eqx.field(static=True, default=50)
    ad: str = eqx.field(static=True, default="reverse")
    callbacks: tuple = eqx.field(static=True, default=())

    def solve(self, problem, *, callbacks=()) -> OptResult:
        """Run Optax on an [InferenceProblem][] decision vector.

        Args:
            problem: An [OptimizationProblem][diff_soup.inference._abc.OptimizationProblem]
                with ``"value_gradient"``: its ``value_and_gradient`` (the
                lowering differentiates it in this solver's ``ad`` mode), its
                ``regularization`` penalty, added to the objective, and its
                ``bounds``, which project every iterate.
            callbacks: Additional callbacks for this solve call. These are
                merged with any callbacks stored on the solver instance and
                fired in the order ``(instance callbacks, call-site callbacks)``.
                Must be passed as a keyword argument.

        Returns:
            A [OptResult][diff_soup.inference.fitting.result.OptResult] describing
            the
            outcome of the training run.
        """
        optax = import_optional("optax")

        cbs = tuple(self.callbacks) + tuple(callbacks)

        if self.optimizer is None:
            optimizer = optax.adam(1e-3)
        else:
            optimizer = self.optimizer

        reg = NoReg() if problem.regularization is None else problem.regularization
        params = jnp.asarray(problem.x0)
        grad_tolerance = gradient_tolerance(self.grad_tol, params.dtype, label="Optax")
        grad_tol = None
        data_loss_and_grad = problem.value_and_gradient

        # --- Box bounds ---------------------------------------------------
        # When the problem declares bounds, project the parameters onto the box
        # after every optimizer update (inside the jitted step, closing over the
        # runtime lb/ub arrays). The optimizer moments are left untouched — this is
        # plain projected Adam/SGD. The initial point is clipped once.
        bnds = problem.bounds
        if bnds is not None:
            lb, ub = (jnp.asarray(side) for side in bnds)
            _flat0, _unravel0 = jax.flatten_util.ravel_pytree(params)
            params = _unravel0(jnp.clip(_flat0, lb, ub))

        initial_flat, _ = jax.flatten_util.ravel_pytree(params)
        n_params = int(initial_flat.shape[0])
        opt_state = optimizer.init(params)

        # The objective is evaluated outside the compiled update, so a problem whose
        # objective is a host function (the lanes of a lowered study) is fitted too.
        def step_fn(params, opt_state):
            loss, grads = data_loss_and_grad(params)
            reg_loss, reg_grad = jax.value_and_grad(reg.penalty)(params)
            loss = loss + reg_loss
            grads = grads + reg_grad
            new_params, new_opt_state = update_fn(params, opt_state, grads)
            return new_params, new_opt_state, loss, grads

        @eqx.filter_jit
        def update_fn(params, opt_state, grads):
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = eqx.apply_updates(params, updates)
            if bnds is not None:
                flat_new, unravel_new = jax.flatten_util.ravel_pytree(new_params)
                new_params = unravel_new(jnp.clip(flat_new, lb, ub))
            return new_params, new_opt_state

        for cb in cbs:
            cb.on_start(n_params, 0, self.max_steps)

        t0_wall = time.perf_counter()
        cost_history = []
        status = TerminationStatus.MAX_STEPS
        best_loss = float("inf")
        best_params = params
        best_grad_norm = float("inf")

        # The cost, the gradient, and the reported parameters all belong to the
        # same point: every step evaluates them at the current iterate, reports
        # them, tests convergence on them, and only then applies the update. The
        # returned parameters are the best point that was actually evaluated.
        for step in range(self.max_steps):
            new_params, new_opt_state, loss, grads = step_fn(params, opt_state)
            loss_val = float(loss)
            cost_history.append(loss_val)

            grad_flat, _ = jax.flatten_util.ravel_pytree(grads)
            param_flat, _ = jax.flatten_util.ravel_pytree(params)
            if bnds is not None:
                # At an active bound the ordinary gradient does not vanish; stop on
                # the projected-gradient norm instead.
                grad_norm = float(projected_gradient_norm(param_flat, grad_flat, lb, ub))
            else:
                grad_norm = float(jnp.linalg.norm(grad_flat))

            if loss_val < best_loss:
                best_loss = loss_val
                best_params = params
                best_grad_norm = grad_norm

            # The callback lifecycle reports every completed optimizer
            # step. Logging/throttling belongs inside the callback, not the solver.
            info = StepInfo(
                step=step,
                cost=loss_val,
                grad_norm=grad_norm,
                params=np.asarray(param_flat),
                accepted=True,
                delta_norm=0.0,
                wall_time=time.perf_counter() - t0_wall,
            )
            for cb in cbs:
                cb.on_step(info)

            if grad_tol is None:
                grad_tol = grad_tolerance.threshold(grad_norm)
            if grad_norm < grad_tol:
                status = TerminationStatus.GRAD_TOL
                break

            params, opt_state = new_params, new_opt_state

        wall_time = time.perf_counter() - t0_wall
        grad_norm = best_grad_norm
        param_flat, _ = jax.flatten_util.ravel_pytree(best_params)
        fitted = problem.fitted(param_flat)

        result = OptResult(
            opt_params=fitted,
            params=np.asarray(param_flat),
            initial_params=np.asarray(initial_flat),
            cost=best_loss,
            grad_norm=grad_norm,
            status=status,
            message=status.value,
            n_steps=step + 1,
            max_steps=self.max_steps,
            n_accepted=step + 1,
            cost_history=np.array(cost_history),
            wall_time=wall_time,
            info={
                "final_opt_state": opt_state,
                "ad_mode": self.ad if problem.ad is None else problem.ad,
            },
        )

        for cb in cbs:
            cb.on_end(result)

        return result
