# Fitting

`ds.inference.optsolve` fits an `InferenceProblem`, an experiment or a
study paired with an objective, and returns an `OptResult`. See [fitting
parameters](../../overview/inference/fitting.md).

!!! tip "Recommended defaults"
    Declare a `Measurement` and an `Experiment`, pose an `InferenceProblem`,
    and fit it with `ds.inference.optsolve`. Start with `ds.inference.MLE()`
    and a `Gaussian` likelihood, and move to `MAP` when every fitted handle
    has a prior. Bound every run with the optimizer's `max_steps`.

::: diff_soup.inference.optsolve

::: diff_soup.inference.InferenceProblem
    options:
      members:
        - placement
        - inventory
        - param_group_names

::: diff_soup.inference.MLE
    options:
      members: false

::: diff_soup.inference.MAP
    options:
      members: false

---

### Objective and trajectory

A trajectory sets how each experiment's solve is split in time ([multiple
shooting](../../overview/inference/multiple-shooting.md)).

::: diff_soup.inference.LeastSquares
    options:
      members: false

::: diff_soup.inference.SingleShoot
    options:
      members: false

::: diff_soup.inference.Multishoot
    options:
      members: false

::: diff_soup.inference.Windows
    options:
      members:
        - from_data
        - from_states
        - uniform
        - at

::: diff_soup.inference.Continuity
    options:
      members: false

::: diff_soup.inference.StateBasis
    options:
      members:
        - full
        - from_modes

::: diff_soup.inference.FixedInitialState
    options:
      members: false

::: diff_soup.inference.LearnedInitialState
    options:
      members: false

---

### Bounds and regularization

::: diff_soup.inference.Bounds
    options:
      members: false

::: diff_soup.inference.Tikhonov
    options:
      members: false

::: diff_soup.inference.NoReg
    options:
      members: false

---

### Results

::: diff_soup.inference.OptResult
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

::: diff_soup.inference.TerminationStatus
    options:
      members: false

::: diff_soup.inference.OptDiagnostics
    options:
      members: false

---

### Callbacks

::: diff_soup.inference.Callback
    options:
      members:
        - bind
        - on_start
        - on_step
        - on_end

::: diff_soup.inference.LogCallback
    options:
      members: false

::: diff_soup.inference.CheckpointCallback
    options:
      members: false

::: diff_soup.inference.StopFit
    options:
      members: false
