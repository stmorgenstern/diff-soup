# Optimization solvers

`ds.inference.optsolve(problem, optimizer=...)` takes one of these solvers. See
[optimization solvers](../../overview/inference/solvers.md).

!!! tip "Recommended defaults"
    `NewtonTrustRegion()` is the default for fewer than about 100
    parameters, and `TrustRegion()` for a lowered hierarchical study. For
    large problems such as embedded neural networks, run `Optax` with
    `optax.adamw`. Bound every run with `max_steps`.

::: diff_soup.inference.NewtonTrustRegion
    options:
      members: false

::: diff_soup.inference.TrustRegion
    options:
      members: false

::: diff_soup.inference.LevenbergMarquardt
    options:
      members: false

::: diff_soup.inference.BFGS
    options:
      members: false

::: diff_soup.inference.GradientDescent
    options:
      members: false

::: diff_soup.inference.Optax
    options:
      members: false

---

### Scales

The `x_scale=` setting of a solver takes one of these.

::: diff_soup.inference.HessianScale
    options:
      members: false

::: diff_soup.inference.JacScale
    options:
      members: false

::: diff_soup.inference.ConstantScale
    options:
      members: false

---

### Constrained fits

::: diff_soup.inference.constrained_optsolve

::: diff_soup.inference.InequalityConstraints
    options:
      members: false

::: diff_soup.inference.Gauge
    options:
      members:
        - pin
        - zero_mean
        - zero_mean_across
        - declared

---

??? abstract "Custom solvers and problems"

    A solver of your own implements `OptSolver` and receives an
    `OptimizationProblem`, or a `LeastSquaresProblem` when the objective
    is half a squared residual norm.

    ::: diff_soup.inference.OptSolver
        options:
          members:
            - requires
            - solve

    ::: diff_soup.inference.OptimizationProblem
        options:
          members:
            - x0
            - value
            - value_and_gradient
            - gauss_newton
            - from_value

    ::: diff_soup.inference.LeastSquaresProblem
        options:
          members:
            - residual
            - residual_and_jacobian
            - from_residual

    ::: diff_soup.inference.XScale
        options:
          members:
            - damping_diagonal
