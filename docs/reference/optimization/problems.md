# Problems and constraints

Every problem `ds.optimization.optsolve` takes lowers to an `OptimizationProblem` of closures over one flat vector. Constraints are declared once, on the problem, and reach every solver that takes them.

### Problems

::: diff_soup.optimization.AbstractOptimizationProblem
    options:
      members:
        - numeric_problem
        - args

---

### Constraints

::: diff_soup.optimization.Constraints
    options:
      members: false

---

??? abstract "Numeric problems and custom solvers"

    The numeric layer matters only when writing a solver or posing a problem by hand.

    ::: diff_soup.optimization.OptSolver
        options:
          members:
            - requires
            - solve

    ::: diff_soup.optimization.OptimizationProblem
        options:
          members:
            - x0
            - args
            - value
            - value_and_gradient
            - gauss_newton
            - constraints
            - constraints_and_jacobians
            - lagrangian_hessian
            - closure
            - with_args
            - with_constraints
            - from_value

    ::: diff_soup.optimization.LeastSquaresProblem
        options:
          members:
            - residual
            - residual_and_jacobian
            - from_residual
