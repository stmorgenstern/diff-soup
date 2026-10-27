# Likelihoods and priors

A likelihood is the noise model of a measurement, and a prior states what
is known about a parameter before the data. See [statistical
estimation](../../overview/inference/statistical-estimation.md).

### Likelihoods

Pass one as `likelihood=` of a `Measurement`.

::: diff_soup.inference.Gaussian
    options:
      members: false

::: diff_soup.inference.CorrelatedGaussian
    options:
      members: false

::: diff_soup.inference.StudentT
    options:
      members: false

::: diff_soup.inference.Poisson
    options:
      members: false

::: diff_soup.inference.LogNormal
    options:
      members: false

::: diff_soup.inference.MultiplicativeGaussian
    options:
      members: false

---

### Priors

A `MAP` fit needs one on every fitted parameter handle.

::: diff_soup.inference.NormalPrior
    options:
      members: false

::: diff_soup.inference.LogNormalPrior
    options:
      members: false

::: diff_soup.inference.GaussianCoefficientPrior
    options:
      members: false
