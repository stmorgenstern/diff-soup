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

::: diff_soup.inference.information

::: diff_soup.inference.laplace

::: diff_soup.inference.evidence

::: diff_soup.inference.identifiability

::: diff_soup.inference.profile

::: diff_soup.inference.residual_report

::: diff_soup.inference.practical_report

::: diff_soup.inference.function_report

::: diff_soup.inference.cluster_covariance

---

### Bootstrap

::: diff_soup.inference.bootstrap

::: diff_soup.inference.BootstrapResult
    options:
      members:
        - values
        - interval
        - band
        - laplace
        - n_kept
        - n_failed

::: diff_soup.inference.BootstrapInterval
    options:
      members: false

::: diff_soup.inference.BootstrapBand
    options:
      members: false

---

### Prediction

::: diff_soup.inference.predict

::: diff_soup.inference.Prediction
    options:
      members: false

::: diff_soup.inference.PredictionBand
    options:
      members: false
