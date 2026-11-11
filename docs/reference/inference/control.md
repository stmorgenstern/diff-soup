# Control

A `ControlProblem` designs a model's operating inputs over a horizon, and
`receding_horizon` applies it in closed loop.

::: diff_soup.optimization.control.ControlProblem
    options:
      members: false

---

### Controls

::: diff_soup.optimization.control.PiecewiseConstant
    options:
      members: false

::: diff_soup.optimization.control.SignalControls
    options:
      members: false

---

### Closed loop

::: diff_soup.optimization.control.receding_horizon
    options:
      members: false

::: diff_soup.optimization.control.mpc_step
    options:
      members: false

::: diff_soup.optimization.control.ClosedLoop
    options:
      members: false

::: diff_soup.optimization.control.HostLoop
    options:
      members: false

---

### Estimators

Each horizon starts from the estimator's state estimate.

::: diff_soup.optimization.control.FullState
    options:
      members: false

::: diff_soup.optimization.control.ExtendedKalmanFilter
    options:
      members: false

::: diff_soup.optimization.control.MovingHorizonEstimator
    options:
      members: false

---

??? abstract "Custom controls and estimators"

    The details matter only when writing controls or an estimator of your own.

    ::: diff_soup.optimization.control.AbstractControls
        options:
          members: false

    ::: diff_soup.optimization.control.Estimator
        options:
          members: false
