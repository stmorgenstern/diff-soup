# Likelihoods and priors

A likelihood is the noise model of a measurement, and a prior states what
is known about a parameter before the data. See [statistical
estimation](../../overview/inference/statistical-estimation.md).

### Likelihoods

Pass one as `likelihood=` of a `Measurement`.

::: diff_soup.optimization.inference.Gaussian
    options:
      members: false

::: diff_soup.optimization.inference.CorrelatedGaussian
    options:
      members: false

::: diff_soup.optimization.inference.StudentT
    options:
      members: false

::: diff_soup.optimization.inference.Poisson
    options:
      members: false

::: diff_soup.optimization.inference.LogNormal
    options:
      members: false

::: diff_soup.optimization.inference.MultiplicativeGaussian
    options:
      members: false

---

### Priors

A `MAP` fit needs one on every fitted parameter handle.

::: diff_soup.optimization.inference.NormalPrior
    options:
      members: false

::: diff_soup.optimization.inference.LogNormalPrior
    options:
      members: false

::: diff_soup.optimization.inference.GaussianCoefficientPrior
    options:
      members: false
