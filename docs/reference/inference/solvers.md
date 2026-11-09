# Optimization solvers

`ds.optimization.optsolve(problem, optimizer=...)` takes one of these solvers. See
[optimization solvers](../../overview/inference/solvers.md).

!!! tip "Recommended defaults"
    `NewtonTrustRegion()` is the default for fewer than about 100
    parameters, and `TrustRegion()` for a lowered hierarchical study. For
    large problems such as embedded neural networks, run `Optax` with
    `optax.adamw`. Bound every run with `max_steps`.

::: diff_soup.optimization.NewtonTrustRegion
    options:
      members: false

::: diff_soup.optimization.inference.TrustRegion
    options:
      members: false

::: diff_soup.optimization.inference.InequalityRows
    options:
      members: false

::: diff_soup.optimization.LevenbergMarquardt
    options:
      members: false

::: diff_soup.optimization.BFGS
    options:
      members: false

::: diff_soup.optimization.GradientDescent
    options:
      members: false

::: diff_soup.optimization.Optax
    options:
      members: false

---

### Scales

The `x_scale=` setting of a solver takes one of these.

::: diff_soup.optimization.inference.HessianScale
    options:
      members: false

::: diff_soup.optimization.inference.JacScale
    options:
      members: false

::: diff_soup.optimization.inference.ConstantScale
    options:
      members: false

---

### Gauges

::: diff_soup.optimization.inference.Gauge
    options:
      members:
        - pin
        - zero_mean
        - zero_mean_across
        - declared

---

??? abstract "Custom solvers and scales"

    A solver of your own implements `OptSolver` and receives an
    `OptimizationProblem` ([problems and constraints](../optimization/problems.md)).

    ::: diff_soup.optimization.inference.XScale
        options:
          members:
            - damping_diagonal
