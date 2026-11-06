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

    ::: diff_soup.optimization.CompiledSolver
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
