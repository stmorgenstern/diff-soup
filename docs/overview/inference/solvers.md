# Optimization solvers

This page provides an overview of the optimizers behind
`ds.inference.optsolve`: the native Newton and quasi-Newton solvers, the
hierarchical trust-region solver, the Optax wrapper, and custom solvers. The examples reuse the recipe, data, and inference problem of
[fitting parameters](fitting.md#the-least-squares-problem):

```python
experiment = ds.inference.Experiment(problem, solver, (0.0, 0.1), y0, (measurement,))
infproblem = ds.inference.InferenceProblem(experiment, parameters=D)
```

## Recommended default

`ds.inference.NewtonTrustRegion()` at default settings is the recommended
optimizer for an optimization problem with fewer than 100 parameters. For
large optimization problems such as embedded neural networks, we
recommend `ds.inference.Optax` with `optax.adamw`
([external solvers](#external-solvers)).

## The native solvers

diff-soup implements several JAX-native optimization algorithms using
Newton and quasi-Newton methods that are specialized for PDE-constrained
optimization problems.

### Newton and quasi-Newton solvers

With \(\mathcal{L}(\theta)\) the objective, \(\nabla\mathcal{L}\) its
gradient, and \(H\) its Hessian, the three solvers differ in the
curvature they carry.

- `NewtonTrustRegion` (recommended) minimizes a quadratic model of
  \(\mathcal{L}\) inside a trust region, with the Gauss-Newton
  approximation \(J^\top J\) of the Hessian, with \(J\) the Jacobian of
  the residual, or the true Hessian \(H\) from the second-order
  sensitivities. It is the method
  of SciPy's `trust-exact` and `trust-ncg` (Nocedal and Wright, 2006).
- `LevenbergMarquardt` takes damped Gauss-Newton steps, with the
  curvature \(J^\top J\) from one forward-mode Jacobian per step
  (Marquardt, 1963).
- `BFGS` builds its curvature from the history of gradients, by the
  limited-memory BFGS update with a backtracking line search (Liu and
  Nocedal, 1989). The gradient comes from the reverse adjoint by
  default.

Each run ends with one of the status flags listed under
[analyze the result](fitting.md#analyze-the-result), and box bounds
declared on the recipe or passed as `ds.inference.Bounds` are kept by
every native solver ([box bounds](fitting.md#box-bounds)).

#### Newton trust region

```python
result = ds.inference.optsolve(infproblem, ds.inference.NewtonTrustRegion())
result[D]
result.info["subproblem"]                             # "exact", one parameter
result.trust_region.radius_history
```

The options, with their defaults:

- `max_steps=50` is the step budget.
- `hessian="auto"` uses the Gauss-Newton approximation \(J^\top J\),
  from one forward-mode Jacobian, whenever the objective is a sum of
  squared residual rows, as every `InferenceProblem` is. An
  `OptimizationProblem` without residual rows gets the true Hessian, and
  `"gauss-newton"` or `"true"` forces either. `hessian="true"` restores
  the exact Hessian of previous releases.
- `subproblem="auto"` solves the trust-region subproblem exactly, at
  every size under the default `x_scale`. It switches to conjugate
  gradients above `dense_cutoff=64` parameters only for a true Hessian
  with an `x_scale` other than `HessianScale`. `"exact"` and `"cg"`
  force either.
- `linear_solver="auto"` selects the factorization of the exact
  subproblem: `"eigh"`, a symmetric eigendecomposition, for a true
  Hessian. For the Gauss-Newton approximation it is `"cholesky"` under
  float64 and `"qr"` of the Jacobian under float32
  ([precision](../simulation/precision.md#inference)). `"eigh"`,
  `"cholesky"`, and `"qr"` force one.
- `radius_init=1.0` and `radius_max=1e3` are the initial and largest
  trust radius, in scaled parameter units.
- `eta=1e-4` is the ratio of achieved to predicted reduction a step
  must exceed to be accepted.
- `grad_tol=None`, `step_tol=None`, and `cost_tol=None` are the
  convergence tolerances on the gradient norm, the step norm, and the
  change in the objective. Under float64 each resolves to `1e-8`
  ([precision](../simulation/precision.md#inference)).
  `cost_atol=None` sets an absolute floor under the test on the change in
  the objective. Set it at the size of the noise in the objective when
  that noise does not shrink with the objective.
- `x_scale=` sets the parameter scaling. The default
  `ds.inference.HessianScale()` scales each parameter by the magnitude
  of its Hessian diagonal. `ds.inference.JacScale()` is refused with
  true curvature, and `HessianScale` is refused in `"cg"` mode.
- `composition="auto"` picks the second-order composition the
  time-stepping adjoint supports, forward over reverse where it can
  ([second order](../simulation/sensitivities.md#second-order)).
  `"forward-over-reverse"` and `"forward-over-forward"` force one.
- `hessian_refresh="adaptive"` decides how often the true Hessian is
  recomputed, with quasi-Newton (SR1) updates between refreshes.
  `"every"` recomputes it at every step, and an integer `k` every
  `k`-th step.
- `hessian_tile="auto"` sets how many directions of the true Hessian,
  or of the Gauss-Newton Jacobian, are formed at once, and with it the
  memory they take. `"auto"` forms the Hessian whole up to
  `dense_cutoff` parameters and the Jacobian whole when it fits the
  device, else either 32, 16 or 8 directions at a time. An integer from
  8 to 128, or `"full"`, fixes it.
- A true Hessian or a Jacobian that does not fit the device stops the
  fit before its first step. The error names the smaller tiles, and for
  the Hessian also `hessian="gauss-newton"` and `subproblem="cg"`.
- `cg_max_iter=None` is the conjugate-gradient budget, the parameter
  count when unset.
- `callbacks=()` attaches callbacks to the solver, merged with those
  passed to `ds.inference.optsolve` ([run the fit](fitting.md#run-the-fit)).

`result.info` records the `hessian`, `hessian_composition`,
`subproblem`, `hessian_tile`, and `linear_solver` settings the run used,
with `hessian_tile` the Jacobian's under Gauss-Newton. The
trust-region radius after every step is in
`result.trust_region.radius_history`.

A step or cost test that passes on a step the trust radius held short ends
the fit `STALLED`, not converged. `result.info["stall_reason"]` says why:
`"radius_collapsed"`, `"evaluation_noise"`, or `"invalid_trials"`.

#### Levenberg-Marquardt

```python
result = ds.inference.optsolve(infproblem, ds.inference.LevenbergMarquardt())
result[D]        # 0.2993
result.status    # TerminationStatus.COST_TOL
```

The options, with their defaults:

- `max_steps=30` is the step budget.
- `lam_init=1.0` is the initial damping, and `lam_max=1e6` is the
  damping above which it restarts from `lam_init`.
- `nu_init=2.0` is the factor the damping grows by on a rejected step.
- `accept_cutoff=1e-4` is the ratio of achieved to predicted reduction
  a step must exceed to be accepted.
- `grad_tol=None`, `step_tol=None`, and `cost_tol=None` are the
  convergence tolerances on the gradient norm, the step norm, and the
  change in the objective. Under float64 each resolves to `1e-8`
  ([precision](../simulation/precision.md#inference)).
  `cost_atol=None` sets an absolute floor under the test on the change in
  the objective. Set it at the size of the noise in the objective when
  that noise does not shrink with the objective.
- `max_step_norm=5.0` is the longest step allowed. `0` removes the limit.
- `x_scale=` sets the parameter scaling. The default
  `ds.inference.JacScale()` scales each parameter by the squared norm
  of its column of \(J\), and `ds.inference.ConstantScale(scale=s)`
  scales by \(1/s^2\), with `s` the typical magnitude of the
  parameters, one scalar or one entry per parameter.
- `jacobian_tile="auto"` sets how many directions of \(J\) are formed
  at once: whole when that fits the device, else 32, 16 or 8. An
  integer from 8 to 128, or `"full"`, fixes it, and
  `result.info["jacobian_tile"]` reports the choice.
- `linear_solver="auto"` factors the damped normal equations by
  Cholesky under float64, and solves the equivalent least-squares
  system by QR under float32. `"qr"` forces QR and is unavailable for a
  joint problem sharded across devices. `"solve"` uses a general dense
  solve. `result.info["linear_solver"]`
  reports the choice made.
- `callbacks=()` attaches callbacks to the solver.

A step or cost test that passes on a step the damping held short ends the
fit `STALLED`, with the reason in `result.info["stall_reason"]`.

```python
lm = ds.inference.LevenbergMarquardt(
    max_steps=100, lam_init=1e-2, x_scale=ds.inference.ConstantScale(scale=0.1))
result = ds.inference.optsolve(infproblem, lm)
result[D]        # 0.2993
```

#### BFGS

```python
result = ds.inference.optsolve(infproblem, ds.inference.BFGS())
result[D]        # 0.2993
result.status    # TerminationStatus.STEP_TOL
```

The options, with their defaults:

- `max_steps=200` is the step budget.
- `grad_tol=None` and `step_tol=None` are the convergence tolerances on
  the gradient norm and the step norm. Under float64 each resolves to
  `1e-8` ([precision](../simulation/precision.md#inference)).
- `c1=1e-4` is the sufficient-decrease parameter of the line search.
- `max_linesearch=20` is the number of step halvings the line search
  tries per step.
- `memory=10` is the number of past steps the curvature estimate is
  built from.
- `ad="reverse"` takes the gradient from the reverse adjoint, one
  adjoint pass per step. `"forward"` takes it from the forward-mode
  Jacobian, one sensitivity solve per parameter
  ([sensitivities and autodiff](../simulation/sensitivities.md)).
- `callbacks=()` attaches callbacks to the solver.

BFGS also ends with `DIVERGED` when the objective or its gradient is
not finite.

### Hierarchical solvers

`ds.inference.TrustRegion` fits a study whose parameters vary by cluster
([hierarchical fits](hierarchical.md)), where some parameters are shared
across the experiments and others take one value per cluster. It takes
the study lowered with `ds.lower(study)`, solves each trust-region step
cluster by cluster and the shared block last, and accepts a study split
across devices ([multi-GPU parallelism](multi-gpu.md)). Its options are
covered under [run the fit](hierarchical.md#run-the-fit) on the
hierarchical fits page.

## External solvers

`ds.inference.Optax` runs an Optax optimizer on the loss, with the
gradient from the reverse adjoint by default and from the forward-mode
Jacobian with `ad="forward"`.

### Optax

`ds.inference.Optax(optimizer=None, max_steps=5000, grad_tol=None)` runs
an [Optax](https://optax.readthedocs.io/) optimizer on \(\mathcal{L}\)
(install the optional dependency with `pip install optax`).

```python
import optax

result = ds.inference.optsolve(
    infproblem, ds.inference.Optax(optimizer=optax.adamw(1e-2), max_steps=200))
result[D]
```

- `optimizer=` is the Optax optimizer, `optax.adam(1e-3)` when omitted.
- `max_steps=5000` is the step budget.
- `grad_tol=None` is the gradient norm below which the run ends with
  `GRAD_TOL`, `1e-8` under float64.
- `ad="reverse"` and `callbacks=()` are as for `BFGS`, with the
  callbacks firing after every step.
- Box bounds are kept by projection after each step.
- The result reports the evaluated point of lowest loss, with its loss
  and gradient norm.

Optax ends with `GRAD_TOL` or `MAX_STEPS`.

## Least-squares problems without a model

`ds.inference.LeastSquaresProblem.from_residual(x0, residual)` poses a
least-squares problem from any JAX-traceable function, and
`ds.inference.optsolve` fits it with the native solvers:

```python
t = jnp.linspace(0.0, 2.0, 20)
y = 3.0 * jnp.exp(-1.3 * t)
problem = ds.inference.LeastSquaresProblem.from_residual(
    jnp.array([1.0, 1.0]), lambda x: x[0] * jnp.exp(-x[1] * t) - y)
result = ds.inference.optsolve(problem, ds.inference.LevenbergMarquardt())
result.params   # array([3. , 1.3])
```

- `jacobian=` supplies the derivative of the function, computed with
  `jax.jacfwd` by default.
- `bounds=(lb, ub)` bounds the fitted values.
- `is_valid=` is a check a trial point must pass before the function is
  evaluated there.

## Custom solvers

A custom solver is a subclass of `ds.inference.OptSolver` implementing
`solve`. For example, gradient descent with a backtracking line search,
taking the gradient from the reverse adjoint and projecting each step
onto the bounds:

```python
class LocalGradientDescent(ds.inference.OptSolver):
    requires = frozenset({"value_gradient"})
    step: float = 1.0
    max_steps: int = 100
    grad_tol: float = 1e-8

    def solve(self, problem, *, callbacks=()) -> ds.inference.OptResult:
        loss_and_grad = problem.value_and_gradient
        bounds = problem.bounds
        x = jnp.asarray(problem.x0)
        loss, grad = loss_and_grad(x)
        status = ds.inference.TerminationStatus.MAX_STEPS
        for _ in range(self.max_steps):
            if jnp.linalg.norm(grad) < self.grad_tol:
                status = ds.inference.TerminationStatus.GRAD_TOL
                break
            step = self.step
            while True:                                   # halve the step until the loss decreases
                x_new = x - step * grad
                if bounds is not None:
                    x_new = jnp.clip(x_new, *bounds)
                loss_new, grad_new = loss_and_grad(x_new)
                if loss_new < loss or step < 1e-12:
                    break
                step = step / 2
            x, loss, grad = x_new, loss_new, grad_new
        return ds.inference.OptResult(
            params=np.asarray(x), cost=float(loss), status=status)


result = ds.inference.optsolve(infproblem, LocalGradientDescent())
result[D]
```

- `solve(self, problem, *, callbacks=()) -> ds.inference.OptResult` is
  the one required method. `ds.inference.optsolve` calls it, then
  attaches the diagnostics and the failed-solve check to the result it
  returns.
- `requires` is a class variable naming what the solver needs from the
  problem, such as `"value_gradient"` for the loss and its gradient.
  `ds.inference.optsolve` refuses a problem that cannot provide it,
  naming the built-in solvers that suit the problem.

The problem passed to `solve` carries:

- `x0`, the starting values as one array.
- `bounds`, the lower and upper bound arrays, or `None` when the problem
  carries no bounds.
- `value_and_gradient(x)`, the loss and its gradient, and
  `value_gradient_hessian(x)`, the loss with its gradient and Hessian. A
  PDE solve that fails inside them returns a loss large enough for any
  solver to reject the point.
- `is_valid(x)`, whether the PDE solves at `x` succeeded.
- `regularization`, the penalty the solver adds to the loss, or `None`.

## References

- J. Nocedal and S. J. Wright. Numerical Optimization. Springer, second
  edition, 2006.
- D. W. Marquardt. An algorithm for least-squares estimation of
  nonlinear parameters. Journal of the Society for Industrial and
  Applied Mathematics, 11(2):431–441, 1963.
- D. C. Liu and J. Nocedal. On the limited memory BFGS method for large
  scale optimization. Mathematical Programming, 45(1):503–528, 1989.
