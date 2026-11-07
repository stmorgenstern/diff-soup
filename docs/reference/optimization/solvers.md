# Constrained solvers

These solvers minimize a numeric problem under equality and inequality constraints and bounds, and their whole loop compiles. Each one's `run` composes with `jax.jit`, `jax.vmap` and `jax.grad` ([optimization solvers](../../overview/inference/solvers.md)).

!!! tip "Recommended defaults"
    `InteriorPoint()` is the default. `SQP()` takes fewer steps from a warm start, so it suits a sequence of nearby problems. `AugmentedLagrangian()` is the fallback for degenerate constraints.

---

### Solvers

::: diff_soup.optimization.InteriorPoint
    options:
      members: false

::: diff_soup.optimization.SQP
    options:
      members: false

::: diff_soup.optimization.AugmentedLagrangian
    options:
      members: false

---

### Derivatives

`run` differentiates its solution in `args` by the implicit function theorem at the solution, never through its iterations. The solver's `adjoint` picks the form:

- `ds.ForwardAdjoint()`, the default, is one forward rule. `jax.jvp` and `jax.jacfwd` cost one KKT solve per tangent, and `jax.grad` transposes the rule.
- `ds.nonlinear.ImplicitAdjoint()` is one reverse rule. `jax.grad` and `jax.vjp` cost one transposed KKT solve however large `args` is, and forward mode raises.

Choose the reverse rule for a large `args`, or for closures that run a solve under `CheckpointAdjoint(reverse_over_reverse=True)`: its backward pass differentiates the problem's gradient in `args` in reverse mode. Pose such a problem with the default `hessian_composition="forward-over-reverse"` and solve it with `hessian="exact"` or `"bfgs"`, since Gauss-Newton takes the residual's Jacobian in forward mode.

```python
solver = ds.optimization.InteriorPoint(hessian="exact", adjoint=ds.nonlinear.ImplicitAdjoint())
gradient = jax.grad(lambda a: loss(*solver.run(problem, problem.x0, a)))(args)
```

---

### Batches of problems

::: diff_soup.optimization.lanes

---

### Results

Every `run` returns the solution and a `SolveRecord`, whose `status` is a
[TerminationStatus][diff_soup.optimization.TerminationStatus] code.

::: diff_soup.optimization.SolveRecord
    options:
      members: false

---

??? abstract "Abstract base class"

    The details matter only when writing a compiled solver of your own.

    ::: diff_soup.optimization.solvers.CompiledSolver
        options:
          members:
            - init
            - step
            - terminate
            - run
            - solve

    ::: diff_soup.optimization.solvers.SolverState
        options:
          members: false
