# Experiments and studies

A `Measurement` holds observed values, an `Experiment` binds a model and
its measurements, and a `Study` groups experiments with the parameters
they share or vary. See [fitting
parameters](../../overview/inference/fitting.md).

::: diff_soup.inference.Measurement
    options:
      members: false

::: diff_soup.inference.Experiment
    options:
      members: false

::: diff_soup.inference.Study
    options:
      members: false

::: diff_soup.inference.Hierarchy
    options:
      members: false

---

### Lowered studies

`ds.lower(study)` returns a `LoweredStudy` for a hierarchical fit ([hierarchical
fits](../../overview/inference/hierarchical.md)).

::: diff_soup.inference.LoweredStudy
    options:
      members:
        - prepare
        - evaluate
        - log_likelihood
        - value_and_gradient
        - observed_information
        - expected_information
        - plan
        - tables
        - for_population
        - retry_kinds
        - retry_first_step_factors
        - evaluation_noise
        - processes
        - close
