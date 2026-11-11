# Fitting

`ds.optimization.optsolve` fits an `InferenceProblem`, an experiment or a
study paired with an objective, and returns an `OptResult`. See [fitting
parameters](../../overview/inference/fitting.md).

!!! tip "Recommended defaults"
    Declare a `Measurement` and an `Experiment`, pose an `InferenceProblem`,
    and fit it with `ds.optimization.optsolve`. Start with `ds.optimization.inference.MLE()`
    and a `Gaussian` likelihood, and move to `MAP` when every fitted handle
    has a prior. Bound every run with the optimizer's `max_steps`.

::: diff_soup.optimization.optsolve

::: diff_soup.optimization.inference.InferenceProblem
    options:
      members:
        - placement
        - inventory
        - values
        - param_group_names

::: diff_soup.optimization.inference.MLE
    options:
      members: false

::: diff_soup.optimization.inference.MAP
    options:
      members: false

---

### Objective and trajectory

A trajectory sets how each experiment's solve is split in time ([multiple
shooting](../../overview/inference/multiple-shooting.md)).

::: diff_soup.optimization.inference.LeastSquares
    options:
      members: false

::: diff_soup.optimization.inference.SingleShoot
    options:
      members: false

::: diff_soup.optimization.inference.Multishoot
    options:
      members: false

::: diff_soup.optimization.inference.Windows
    options:
      members:
        - from_data
        - from_states
        - uniform
        - at

::: diff_soup.optimization.inference.Continuity
    options:
      members: false

::: diff_soup.optimization.inference.StateBasis
    options:
      members:
        - full
        - from_modes

::: diff_soup.optimization.inference.FixedInitialState
    options:
      members: false

::: diff_soup.optimization.inference.LearnedInitialState
    options:
      members: false

---

### Bounds and regularization

::: diff_soup.optimization.inference.Bounds
    options:
      members: false

::: diff_soup.optimization.inference.Tikhonov
    options:
      members: false

::: diff_soup.optimization.inference.NoReg
    options:
      members: false

---

### Results

::: diff_soup.optimization.OptResult
    options:
      members:
        - params
        - cost
        - status
        - message
        - converged
        - n_steps
        - diagnostics
        - log_likelihood
        - log_prior
        - save
        - load

::: diff_soup.optimization.TerminationStatus
    options:
      members: false

::: diff_soup.optimization.inference.OptDiagnostics
    options:
      members: false

---

### Callbacks

::: diff_soup.optimization.inference.Callback
    options:
      members:
        - bind
        - on_start
        - on_step
        - on_end

::: diff_soup.optimization.inference.LogCallback
    options:
      members: false

::: diff_soup.optimization.inference.CheckpointCallback
    options:
      members: false

::: diff_soup.optimization.inference.StopFit
    options:
      members: false
