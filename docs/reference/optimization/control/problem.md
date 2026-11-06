# Control problems

A `ControlProblem` poses optimal control of a model's operational inputs over a horizon. The decision vector parameterizes the controls; the cost, the path rows and the terminal rows are functions of the solution over the horizon. It lowers to an `OptimizationProblem` like every other problem, so `optsolve` and every constrained solver take it.

::: diff_soup.optimization.control.ControlProblem
    options:
      members:
        - numeric_problem
        - args
        - closures

---

::: diff_soup.optimization.control.PiecewiseConstant
    options:
      members:
        - durations
        - horizon
        - interval
        - segments
