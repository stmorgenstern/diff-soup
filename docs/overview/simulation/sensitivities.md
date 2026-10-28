# Sensitivities and autodiff

This page provides an overview of differentiating a solve through
`ds.pdesolve`: the forward and reverse modes, parameter gradients, and
second-order composition. The model problem is the heat-equation solve
from [running a simulation](running-a-simulation.md).

A solve maps the initial state and the parameter values to the saved
states,

$$
(y_0, \theta) \mapsto u(t\,;\, y_0, \theta),
$$

and this map is differentiable.

- Forward mode (`jax.jvp`, `jax.jacfwd`) works at the default
  settings and runs in O(1) memory. It pushes one tangent through the
  solve per input direction, which suits few inputs and many outputs.
  The default `adjoint=None` selects `ds.ForwardAdjoint()`.
- Reverse mode (`jax.grad`, `jax.jacrev`, `jax.vjp`) is enabled by
  `adjoint=ds.CheckpointAdjoint()` and computes the gradient with
  respect to every input in one backward pass, which suits a scalar
  objective of many parameters.

Both modes are demonstrated on the initial state in
[running a simulation](running-a-simulation.md#differentiate-the-solve).

## Forward mode

Forward mode computes full parameter sensitivities at the default
settings. For example, to compute the sensitivity of the heat
equation's solution with respect to its diffusivity, the recipe
declares the diffusivity as a learnable parameter
([learnable parameters](../modeling/parameters.md)):

```python
D = spec.param("D", value=0.1, learnable=True)

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u)
```

`ds.default_theta(problem)` returns the problem's parameter pytree,
which holds every floating-point array of the problem: the declared
parameters and functions, the mass diagonal and the initial
multipliers. `jax.jacfwd` pushes one tangent per parameter direction
through the solve:

```python
theta = ds.default_theta(problem)

def final_state(theta):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta)
    return sol.ys[-1]

S = jax.jacfwd(final_state)(theta)   # d(final state)/d(theta), leaf for leaf
```

## Reverse mode and checkpointing

`adjoint=ds.CheckpointAdjoint()` switches the solve loop to a
checkpointed replay. The forward pass stores a bounded number of state
checkpoints, and the backward pass replays trajectory segments from
them while pulling the objective's cotangent back through each step.
The loop is [equinox](https://github.com/patrick-kidger/equinox)'s
checkpointed while loop, following the online checkpointing scheme of
Stumm and Walther (2010).

- The checkpoint count is set on the adjoint,
  `ds.CheckpointAdjoint(checkpoints=...)`. Reverse-mode memory is
  roughly the checkpoint count times the stored state, plus the save
  grid.
- The default `checkpoints=None` leaves the count to the Equinox
  loop's own default of roughly the square root of twice `max_steps`,
  140 checkpoints at the default step budget.
- The replay reuses the step decisions the forward pass recorded, so
  the backward pass integrates the same trajectory in every precision.
  A replay that reaches a different time or step size raises.
- Calling `jax.grad` on a solve without the adjoint option fails at
  trace time with JAX's error for reverse mode through a
  `lax.while_loop`.
- Reverse mode composes with `jax.vmap`. Batching solves is covered in
  [batched solves](batched-solves.md).
- Reverse mode differentiates through an affect event
  ([events](events.md)). The checkpointed gradient across a
  state-rewriting restart matches forward mode.

## Parameter gradients

`theta=` carries the parameter values a solve runs with and
differentiates against, and the gradient of a solve with respect to
`theta` is a pytree of the same structure. On the declared-diffusivity
recipe from [forward mode](#forward-mode), the loss solves with the
reverse adjoint and reduces the final state:

```python
def loss(theta):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

grads = jax.grad(loss)(theta)
```

`grads` mirrors `theta` leaf for leaf. To differentiate with respect to
one parameter, build `theta` inside the loss. `D` is the handle returned
by `spec.param`, and `ds.default_theta(problem, {D: value})` sets it:

```python
def loss_D(D_value):
    theta = ds.default_theta(problem, {D: D_value})
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

dloss_dD = jax.grad(loss_D)(0.1)
```

`theta[D]` reads the value `theta` holds for `D`, and
`theta.replace(D, value)` returns a copy with `D` set, the same as
`ds.default_theta(problem, {D: value})`. `grads.gradient(D)` reads
`dloss_dD` from `grads`.

An array parameter is set the same way, with an array as the value.
`default_theta` works under `jax.jacfwd`, `jax.grad` with
`ds.CheckpointAdjoint()` and `jax.jit`. When `theta` holds a network, an
`eqx.Module`, use `eqx.filter_jit` in place of `jax.jit`. `jax.grad` alone
is fine.

Named-parameter sensitivity runs, selected by handle, are provided by
the [inference](../inference/identifiability.md) layer.

## Second order

Second-order compositions work on the native engines, and the solve's
`adjoint=` selects the composition. The Hessian of the loss computes
three ways, and the fit layer reaches the same compositions through
`ds.inference.InferenceProblem.loss_grad_hessian_fn(composition=)`, which
`ds.inference.NewtonTrustRegion` and the observed information use
([optimization solvers](../inference/solvers.md)). Forward over reverse,
the `jax.hessian` route, is the most performant:

```python
def loss(theta):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

H = jax.hessian(loss)(theta)
```

Forward over forward runs at the default settings:

```python
def loss(theta):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta)
    return sol.ys[-1].sum()

H = jax.jacfwd(jax.jacfwd(loss))(theta)
```

Reverse over forward runs through the composable forward mode:

```python
def loss(theta):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, theta=theta,
                      adjoint=ds.ForwardAdjoint(composable=True))
    return sol.ys[-1].sum()

H = jax.jacrev(jax.jacfwd(loss))(theta)
```

Reverse over reverse is unsupported and raises.

The interaction of reduced-precision policies with first- and
second-order differentiation is covered in [precision](precision.md).

## Failed solves

Failure of a solve is reported through `sol.result`
([solutions and output](solutions.md)). Derivatives follow the same
contract: a failed solve yields NaN derivative entries.
