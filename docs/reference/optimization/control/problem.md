# Control problems

A `ControlProblem` poses optimal control of a model's operational inputs over a horizon. The decision vector parameterizes the controls; the cost, the path rows and the terminal rows are functions of the solution over the horizon. It lowers to an `OptimizationProblem` like every other problem, so `optsolve` and every constrained solver take it.

::: diff_soup.optimization.control.ControlProblem
    options:
      members:
        - numeric_problem
        - args

---

### Controls

`PiecewiseConstant` holds one target per interval of a grid. `SignalControls` holds signals of `diff_soup.optimization.inference.approx`, with free durations if asked ([controls](../../../overview/modeling/controls.md#control-signals)).

::: diff_soup.optimization.control.PiecewiseConstant
    options:
      members:
        - durations
        - horizon
        - interval
        - segments

::: diff_soup.optimization.control.SignalControls
    options:
      members:
        - durations
        - horizon
        - segments

---

??? abstract "Abstract base class"

    The details matter only when writing controls of your own.

    ::: diff_soup.optimization.control.AbstractControls
        options:
          members:
            - size
            - free_durations
            - durations
            - horizon
            - interval
            - interval_controls
            - segments
