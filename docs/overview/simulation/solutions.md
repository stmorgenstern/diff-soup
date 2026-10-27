# Solutions and output

This page provides an overview of declaring what a solve saves and
reading the returned `Solution`. The model problem is the heat-equation
solve from [running a simulation](running-a-simulation.md).

## Declare the output plan

`saveat=` takes a `ds.SaveAt`. The default plan saves the final state
only.

- `ts=` saves at the given times. The times are sorted, deduplicated,
  and must lie inside the integration interval.
- `t0=True` and `t1=True` add the endpoints, and combine with `ts=`.
- `steps=True` saves every accepted native or Diffrax step. On JNDF,
  JBDF, and JIDA, `steps=k` saves every `k`-th accepted step, and the
  final accepted state is always retained.
- `dense=True` records the interpolant for evaluation at any time in
  the interval.
- `outputs=` names what to store at every save time in place of the full
  state. Each name is a declared observation, a declared field,
  `"state"`, or `"lambdas"`, and `sol.outputs` maps each name to its
  samples.
- `ordered=True` uses the times in `ts=` exactly as given, which must
  already be increasing. The times can then differ between the solves of
  a `jax.vmap` batch. It cannot be combined with `t0=`, `t1=`, or
  `steps=`.

```python
import numpy as np

saveat = ds.SaveAt(ts=np.linspace(0.02, 0.08, 4), t1=True)
sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0, saveat=saveat)
sol.ts   # [0.02, 0.04, 0.06, 0.08, 0.1]
sol.ys   # (5, 64), one row per save time
```

Step samples are the accepted states themselves, recorded at their own
times, while a `ts` grid samples the interpolant between steps. For a DAE
the accepted states lie on the constraint manifold. `steps` combines with
`t0`, `t1`, and `dense`, and combining it with a `ts` grid is refused.
The output buffer is sized from `max_steps`, so choose `k` or `max_steps`
deliberately for a large state. Step output differentiates in forward
mode, and reverse mode (`CheckpointAdjoint`) keeps an explicit time grid.

## Dense output

A requested save time is evaluated in the loop by the engine's own
interpolating polynomial at the method's current order. `dense=True`
additionally stores the state and its derivative at every accepted
step, and `sol.evaluate(t)` serves a piecewise cubic through those
nodes at any time in the interval. `t` is a scalar or a
one-dimensional array of times, and a time outside the interval is
clamped to its nearest end.

```python
sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0,
                  saveat=ds.SaveAt(dense=True))
u_mid = sol.evaluate(0.05)   # (64,)
```

For a DAE, `SaveAt(output=ds.DenseOutputPolicy(mode=...))` selects how
the algebraic rows are sampled. The default `"fast"` mode samples every
local and bordered row through the step interpolant. Interior samples satisfy
constraints to interpolation accuracy, while samples at accepted step times
are exact accepted states. `"projected"` projects each sample's algebraic rows
back onto the constraint manifold. Projection is covered in
[consistent initialization](consistent-init.md).

## The outcome

`sol.ok` and `sol.result` report the outcome. A solve that does not
reach the end of the interval is never reported as success, and
`pdesolve` never raises on non-convergence. Failure is read from the
returned record.

```python
sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0,
                  saveat=ds.SaveAt(ts=np.linspace(0.01, 0.1, 10)),
                  max_steps=20)
sol.ok                               # False
sol.result == ds.RESULT.MAX_STEPS    # True
```

`sol.result` is a traced code from `ds.RESULT`. `SUCCESS` and
`EVENT_STOP` are the two successful outcomes, and every partial
terminal status is a failure.

- `SUCCESS`, the solve reached the end of the interval.
- `EVENT_STOP`, the solve was stopped by a terminating event
  ([events](events.md)).
- `MAX_STEPS`, the step budget ran out first.
- `STEP_COLLAPSE`, the step size collapsed below its lower bound.
- `NONLINEAR_DIVERGED`, the corrector diverged
  ([nonlinear solvers](nonlinear.md)).
- `LINEAR_FAIL`, a linear solve failed
  ([linear solvers](linear.md)).
- `SINGULAR`, the iteration matrix was detected singular.
- `INCONSISTENT_IC`, the initial state violates the algebraic
  constraints ([consistent initialization](consistent-init.md)).
- `ACCURACY_LIMIT`, the requested tolerances are tighter than double
  precision resolves at the solution's magnitude.
- `UNKNOWN`, the solve did not run to a terminal status.

## Fixed buffers and validity

A compiled JAX program needs fixed array shapes, so the output buffers
are allocated at their full requested size before the solve runs, and
rows fill as the solve passes each save time. A successful solve fills
every explicit-grid row. Accepted-step plans normally retain a NaN tail
because their capacity comes from the step budget. A solve that stops early leaves the remaining rows
unfilled: their `sol.ts` entries are `NaN`, `sol.valid_mask` is
`False` there, and `sol.valid_count` is the number of filled rows. The
host-side `sol.trimmed()` returns a copy with the unfilled rows
dropped. Reading batched results is covered in
[batched solves](batched-solves.md).

## Solver statistics

`sol.stats` carries the run's counters, among them `accepts`,
`rejects`, `newton_iters`, `factorizations`, `linear_solves`, and
`t_reached`, the time the solve actually reached. The full record is
documented in the
[solution reference](../../reference/simulation/solution.md).
