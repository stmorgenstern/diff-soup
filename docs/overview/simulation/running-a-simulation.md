# Running a simulation

This page provides an overview of running a built problem with
`ds.pdesolve`, diff-soup's forward-simulation entry point. Each option
links to the page covering it in detail.

## Run a solve

The model problem is the heat-equation recipe assembled on
[the Recipe API](../modeling/recipe-api.md) page, which ends with the
lowered `problem` and the assembled initial state `y0` in hand. The
full solve:

```python
import diff_soup as ds

solver = ds.JIDA()
sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0)
```

### Choose a solver

The native engines `JNDF`, `JBDF`, and `JIDA` implement the
variable-step, variable-order implicit multistep family in pure JAX. See
the [ODE/DAE solvers](solvers.md) page for more details on each engine,
the solver options, and recommendations for selecting a solver.

```python
solver = ds.JIDA()
```

### Call `pdesolve`

`ds.pdesolve(problem, solver, t_span, y0)` integrates the problem over
`t_span = (t0, t1)` from the initial state `y0`.

```python
sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0)
```

Under the hood, each solve is jit compiled, and repeated solves
execute through the fast compiled path. Dynamic arguments, such as the
time span, the initial state, and the parameter values, reuse the
compiled path. Changing a static option such as `max_steps`, or
rebuilding the problem, compiles a fresh program. See
[ODE/DAE solvers](solvers.md) for which options are static on each
engine.

### Read the solution

The result is a result-coded `Solution`. `sol.ys` holds the saved
states, one row per save time, and `sol.ok` / `sol.result` report the
outcome. The default output plan saves the final state only. Save
grids, dense output, and the solver statistics are covered in
[solutions and output](solutions.md).

## Differentiate the solve

The solve is differentiable. Forward mode works with the default
settings, so `jax.jacfwd` computes the sensitivity of the outputs with
respect to the initial state:

```python
import jax

def final_state(y0):
    return ds.pdesolve(problem, solver, (0.0, 0.1), y0).ys[-1]

J = jax.jacfwd(final_state)(y0)   # (64, 64), d(final state)/d(y0)
```

Reverse mode (`jax.grad`, `jax.vjp`) is enabled by passing
`adjoint=ds.CheckpointAdjoint()` on the call:

```python
def loss(y0):
    sol = ds.pdesolve(problem, solver, (0.0, 0.1), y0,
                      adjoint=ds.CheckpointAdjoint())
    return sol.ys[-1].sum()

g = jax.grad(loss)(y0)            # (64,), d(loss)/d(y0)
```

Differentiating with respect to parameters, the adjoint options, and
second derivatives are covered in
[sensitivities and autodiff](sensitivities.md).

## The options

Everything else on the call is keyword-only. Each option states its
default and links to the page that covers it.

- `saveat=` specifies the saved simulation outputs. The default option
  saves the output at the final timepoint only. `ds.SaveAt(ts=...)`
  specifies a grid of timepoints that outputs will be collected at via
  interpolation. `ds.SaveAt(steps=True)` saves the outputs at exactly
  the accepted steps. Covered in
  [solutions and output](solutions.md).
- `theta=` supplies the parameter values for this run, defaulting to
  the problem's declared values. Varying parameters through `theta`
  stays on the compiled program. The declared handles are covered in
  [learnable parameters](../modeling/parameters.md).
- `tolerances=` takes a `ds.ToleranceSpec` overriding the solver's
  scalar tolerances for one run, on the native engines. `atol=` accepts
  a scalar, a per-unknown vector, or a mapping keyed by field name, by
  role (`"differential"`, `"algebraic"`), or by partition (`"state"`,
  `"global"`), and the most specific spelling wins. `ic_rtol=` and
  `ic_atol=` set the consistent-initialization tolerances separately,
  defaulting to the integration values. On `JIDA`,
  `algebraic_error_control="exclude"` keeps the algebraic rows out of
  the local error test, and `JNDF` and `JBDF` refuse it. The solution
  records the resolved values as `sol.tolerances`, a
  `ds.ResolvedTolerances`.
- `adjoint=` selects the differentiation strategy. The default is
  forward-only, `jax.jvp` and `jax.jacfwd` in O(1) memory, and
  `adjoint=ds.CheckpointAdjoint()` enables reverse mode (`jax.grad`,
  `jax.vjp`). Covered in
  [sensitivities and autodiff](sensitivities.md).
- `event=` declares the events monitored during the solve, whether
  terminating or state-modifying. Covered in [events](events.md).
- `max_events=16` sets the capacity of the bounded event log on the
  native engines, and a value below one is refused. Covered in
  [events](events.md).
- `consistent_ic=` overrides the solver's initialization default. The
  projection pass runs automatically at the start of every solve and
  is a no-op for a pure ODE. Covered in
  [consistent initialization](consistent-init.md).
- `control_state0=` sets the initial operating point of a problem with
  declared controls. Declaring controls is covered in
  [controls](../modeling/controls.md) and switching them at runtime in
  [events](events.md).
- `lambda0=` passes the initial values of the bordered globals as
  their own block. The default `None` treats `y0` as the full
  augmented state, and a non-`None` value treats `y0` as the field
  block and packs the two. Covered in
  [fields and globals](../modeling/fields.md).
- `trace=True` attaches a per-step diagnostic payload to the returned
  solution.
- Engine keywords pass through unfiltered, among them `dt0`, `hmin`,
  `hmax`, `max_order`, `max_steps`, and the `linear` backend override.
  Covered in [ODE/DAE solvers](solvers.md) and
  [linear solvers](linear.md).
