# Batched solves

This page provides an overview of running many independent solves as
one batched program and reading the batched results. The model problem
is the heat-equation solve from
[running a simulation](running-a-simulation.md).

## Batching a solve

`eqx.filter_vmap` maps a solve over a leading batch axis and compiles
one program that advances every lane together. The dynamic call
arguments batch, among them the initial state and the parameter
values.

```python
import equinox as eqx

y0_batch = jnp.stack([y0, 0.5 * y0, 0.25 * y0])

def solve_one(y0):
    return ds.pdesolve(problem, solver, (0.0, 0.1), y0,
                       saveat=ds.SaveAt(ts=np.linspace(0.01, 0.1, 10)))

sols = eqx.filter_vmap(solve_one)(y0_batch)
sols.ys        # (3, 10, 64), one leading lane axis
sols.ok        # (3,), per-lane outcomes
```

A function that returns a `Solution` batches under `eqx.filter_vmap`. The
solution carries the problem and other non-array fields, which
`eqx.filter_vmap` keeps static while batching the array fields. Plain
`jax.vmap` hands those fields to JAX as outputs and fails with a
`TypeError` naming the problem as an invalid JAX type. A function that
selects arrays from the solution before returning still runs under plain
`jax.vmap`: the saved states, a loss, or a gradient.

```python
def loss(y0):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

finals = jax.vmap(lambda y0: solve_one(y0).ys[-1])(y0_batch)
grads = jax.vmap(jax.grad(loss))(y0_batch)
```

A batch of implicit solves runs the Jacobian setup and the corrector's
retry only for the lanes that need them, so it costs about what its
solves cost one by one.

## Reading batched results

Every array field of the solution carries the lane axis, and each
lane reports its own outcome. Lanes terminate independently and trim
to the lane length. With the cooling stop event from
[events](events.md), hotter initial states run longer before
stopping:

```python
def solve_event(y0):
    return ds.pdesolve(problem, solver, (0.0, 0.2), y0,
                       saveat=ds.SaveAt(ts=np.linspace(0.01, 0.2, 20)),
                       event=cool)

sols = eqx.filter_vmap(solve_event)(jnp.stack([y0, 1.5 * y0, 2.0 * y0]))
sols.result == ds.RESULT.EVENT_STOP   # [True, True, True]
sols.valid_count                      # [6, 10, 13], filled rows per lane
sols.trimmed(lane=0).ts               # (6,), lane 0 without unfilled rows
```

A compiled JAX program needs fixed array shapes, so each lane writes
into the full requested buffers and `valid_mask` and `valid_count`
mark the rows it filled ([solutions and output](solutions.md)).
Reading through them is `jit` and `vmap` safe. `trimmed(lane=)`
performs the data-dependent slice on the host.

## Differentiation and backends

- Reverse mode composes with batching
  ([sensitivities and autodiff](sensitivities.md)).
- The dense and cuDSS backends batch independent matrices, and KLU is
  refused for them ([linear solvers](linear.md)).
- Projected dense output (`DenseOutputPolicy(mode="projected")`,
  [solutions and output](solutions.md)) batches on dense and cuDSS
  and runs lane by lane on KLU.
- The experiments of a fit are split across devices through
  `ds.Layout(mesh=...)` ([multi-GPU parallelism](../inference/multi-gpu.md)).
