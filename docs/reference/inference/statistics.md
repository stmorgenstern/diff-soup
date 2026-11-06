# Statistics

These functions take an inference problem and its fit, and report the
uncertainty and identifiability of the fit. See
[statistical estimation](../../overview/inference/statistical-estimation.md)
and [identifiability](../../overview/inference/identifiability.md).

!!! tip "Recommended defaults"
    After a fit, call `information`, `laplace`, `identifiability` and
    `profile` on the problem and its fit, and `predict` for bands. Run
    `sample` only after these look sound.

### Local analysis

::: diff_soup.optimization.inference.information

::: diff_soup.optimization.inference.laplace

::: diff_soup.optimization.inference.evidence

::: diff_soup.optimization.inference.identifiability

::: diff_soup.optimization.inference.profile

::: diff_soup.optimization.inference.residual_report

::: diff_soup.optimization.inference.practical_report

::: diff_soup.optimization.inference.function_report

::: diff_soup.optimization.inference.cluster_covariance

---

### Bootstrap

::: diff_soup.optimization.inference.bootstrap

::: diff_soup.optimization.inference.BootstrapResult
    options:
      members:
        - values
        - interval
        - band
        - laplace
        - n_kept
        - n_failed

::: diff_soup.optimization.inference.BootstrapInterval
    options:
      members: false

::: diff_soup.optimization.inference.BootstrapBand
    options:
      members: false

---

### Prediction

::: diff_soup.optimization.inference.predict

::: diff_soup.optimization.inference.Prediction
    options:
      members: false

::: diff_soup.optimization.inference.PredictionBand
    options:
      members: false
