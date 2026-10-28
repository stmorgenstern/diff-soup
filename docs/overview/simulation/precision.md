# Precision

This page provides an overview of reduced- and mixed-precision solves
through the `ds.Precision` policy. Typical GPUs run float32
substantially faster than float64, so diff-soup provides reduced- and
mixed-precision strategies for systems that can benefit. A stiff
system can suffer under a reduced-precision factor, losing more to
corrector and step failures than the cheaper arithmetic saves, so
full float64 is the default. That default is the library's, not JAX's.
JAX computes in 32-bit unless its 64-bit mode is on, so a script enables
it before building a problem
([enable float64](../../installation/index.md#enable-float64)), and the
cuDSS backend the reduced modes run on refuses to work without it.

## The problem owns its precision

A `Precision` is declared once, on the problem, and every solve of that
problem runs under it. Each way of building a problem accepts one:

```python
precision = ds.Precision.fp32()

problem = spec.build(precision=precision)                 # Recipe API
problem = ds.lower(equation_set, precision=precision)     # lowered equation set
problem = ds.MassProblem(rhs=rhs, mass=mass, structure=structure,
                         precision=precision)
problem = ds.ImplicitProblem(residual=residual, structure=structure,
                             linearization=linearization, precision=precision)

sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 0.1), y0)     # runs in float32
```

The solver inherits the problem's policy, solve mode and
differentiation fields included, so `ds.JIDA()` needs no precision of
its own. The inputs of a solve are cast to the policy's
`compute_dtype` when it starts: the initial state, the bordered
globals, and the floating parameter leaves. The cast goes in both
directions, so a float64 `y0` handed to a float32 problem runs in
float32, and a float32 `y0` handed to a default problem runs in
float64. The dtype of an input never decides the precision of a solve.

The solvers still accept `precision=` on their constructors. A policy
given there is used when the problem declares none, or when it is the
same policy the problem declares. A solver policy that disagrees with
the problem's raises `ConfigurationError`, since precision belongs on
the problem.

## Choosing a mode

The presets bundle the common policies, and
`Precision(solve_mode=...)` sets the linear solve mode alone. On the
heat-equation solve from
[running a simulation](running-a-simulation.md):

```python
problem = spec.build()                                       # float64 (default)

problem = spec.build(precision=ds.Precision.fp32())          # end-to-end float32

# The recommended mixed-precision forward solve mode:
problem = spec.build(precision=ds.Precision(solve_mode="fp32_ir"))
```

- `Precision.fp64()`, the default policy: full double precision.
- `"fp32_ir"`, the recommended mixed-precision mode: float64 compute
  with fp32 factors refined toward a full-precision residual. The
  `Precision.fp32_ir()` preset bundles it with the sensitivity
  fields below.
- `Precision.fp32()`: end-to-end float32. States, times, parameters
  inside the solve, residual and Jacobian assembly, the fp32 factors
  and solves, and both derivative directions run in float32. Grid
  stencils cast the grid's geometry, which is stored in float64, to the
  field's dtype, so their arithmetic is float32 as well. A
  [custom operator](../modeling/operators.md#custom-operators) owns its
  own casts.
- `Precision.fp32_factor()`: float64 compute with direct fp32
  factors.
- `Precision(solve_mode="adaptive")`: the engine switches between the
  refined fp32 mode and full precision at runtime, promoting on
  failure. `Solution.stats.precision_promotions` counts the
  promotions. Adaptive mixed precision is experimental.

A float32 `compute_dtype` means float32 factors, so
`Precision(compute_dtype="float32")` is the same policy as
`Precision.fp32()`. Its default solve mode is stored as `"fp32"`, and
`Solution.stats.linear_solve_mode` reports it. The two refining modes,
`"fp32_ir"` and `"adaptive"`, measure a float64 residual and are
refused under a float32 compute dtype. The mixed modes keep a float64
compute dtype.

`Precision.jacobian_values_dtype` computes only the Jacobian entries
at a narrower dtype, for example float32 entries under float64
compute. The residual, the right-hand side, and all gradients stay at
`compute_dtype`. It cannot be wider than `compute_dtype`.

## Backends

The reduced modes run on the cuDSS backend, which the `"auto"`
backend resolution selects on a GPU ([linear solvers](linear.md)).
The dense backend serves the float64 policies and an end-to-end
float32 problem, whose single-precision LU it computes natively. KLU
factors in double precision only, so it refuses a float32 problem
rather than promoting it.
Float32 is meant for GPUs. A float32 problem solved on a CPU device
still runs in float32 on the dense backend, but a CPU gains little from
single precision, so the first such solve in a process warns and
recommends a float64 `Precision` there.

On cuDSS, a float32 problem factors and solves in float32, through its
derivatives and `jax.hessian` as well.

## Inference

An inference problem runs at the precision of its forward problem.
For a float32 problem the observed data, the likelihood and
observable parameters, the decision vector, the residual rows and
their Jacobian, the log-likelihood with its final sum, the MAP prior
rows, the optimizer's steps and cost tests, and the Laplace posterior
are all float32. The optimizer's parameter vector is float32 too, so
its trust-region or Levenberg–Marquardt linear algebra runs in
float32. Because forming $J^\top J$ squares the condition number,
`NewtonTrustRegion(hessian="gauss-newton")` and `LevenbergMarquardt`
factor the augmented Jacobian by QR under their default
`linear_solver="auto"` when the decision vector is float32, and use
Cholesky of the normal equations under float64.

The optimizers' termination tolerances follow the decision vector's
dtype when they are left unset. `step_tol` and `cost_tol` become
$\max(\text{default}, 10\,\varepsilon)$, with $\varepsilon$ the
machine epsilon of that dtype: under float64 this is the default
itself ($10^{-10}$ and $10^{-12}$), under float32 it is about
$1.2 \times 10^{-6}$. `grad_tol` stays the absolute $10^{-8}$ under
float64. Under float32 it is relative to the gradient norm at the start
of the fit, $\max(10^{-8}, 10\,\varepsilon)$ times that norm, because
a gradient taken through a float32 solve is only resolved relative to
its own size. This applies to `NewtonTrustRegion`,
`LevenbergMarquardt`, `BFGS`, `GradientDescent`, `Optax`, and the
block optimizers of hierarchical fits. A tolerance you set is used as
given, and a warning says so when it is below what the dtype resolves.

## Mixed-precision gradients

Differentiation through the solve inherits the precision of the
forward solve unless an override is provided
([the differentiation fields](#the-differentiation-fields)). For
example, to compute a mixed-precision gradient:

```python
problem = spec.build(precision=ds.Precision(solve_mode="fp32_ir"))

def loss(y0):
    sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 0.1), y0,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

g = jax.grad(loss)(y0)
```

To override the inheritance, computing the forward solve in full
precision and the transposed factorization of the gradient in fp32:

```python
precision = ds.Precision(reverse_sensitivity_solve_mode="fp32_ir")
problem = spec.build(precision=precision)
```

`jax.hessian` composes under every precision mode ([second
order](sensitivities.md#second-order)).

## The differentiation fields

- `forward_sensitivity_solve_mode=` lets forward sensitivities use a
  direction-specific solve mode. `None` inherits `solve_mode`, and
  `"fp64"` is the explicit full-precision escape under a reduced
  primal solve.
- `adjoint_compute_dtype=` and `reverse_sensitivity_solve_mode=` set
  the reverse sweep's dtype and solve mode, and `adjoint_ir_tol=`
  sets the refinement target of the transposed solves of the
  implicit-function derivatives (the backward solves of a root or a
  consistent initial state). The corrector's solves keep the forward
  target. A `None` reverse
  solve mode also inherits `solve_mode`. The dtype must be at least as
  wide as `compute_dtype`, and a narrower value is refused. Under a
  float32 compute dtype an explicit `"fp64"` reverse solve mode is an
  opt-in mixed policy with a double-precision transpose factor.
- Setting either reverse field, which the `fp32_ir()` and
  `fp32_factor()` presets do, leaves `jax.grad` and `jax.hessian`
  unchanged ([sensitivities and autodiff](sensitivities.md)).
  `Precision.fp64()` and `Precision.fp32()` leave the reverse fields
  unset.
