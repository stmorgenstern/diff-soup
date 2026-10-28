# Statistical estimation

This page provides an overview of fitting the learnable parameters of a
model to observed data as a statistical estimation problem with
`ds.inference.InferenceProblem`.

For maximum likelihood estimation, we
assume the data, \(u_{\text{data}}\), is distributed under a known
distribution, \(u_{\text{data}} \sim p(u_{\text{data}})\), such that our
loss function is replaced with a likelihood
\(p(u_{\text{data}} \mid \theta)\), where \(\theta\) are the model
parameters. For maximum a posteriori estimation, we also assume the model
parameters are probabilistic unknowns, distributed as
\(\theta \sim p(\theta)\). Throughout, we
fit the synthetic data of [fitting parameters](fitting.md), generated
with additive Gaussian noise by solving the heat equation

$$
\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial x^2}, \qquad x \in (0, 1),
$$

with \(u(0, t) = u(1, t) = 0\), \(u(x, 0) = x(1 - x)\), and \(D = 0.3\),
sampled at five times with noise of scale \(10^{-3}\):

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d

n = 32
spec = ds.problem("heat")
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))
spec.param("D", value=0.3)                                            # the generating diffusivity


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u)


spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)
problem_true = spec.build()
y0 = spec.assemble_state({})
solver = ds.JIDA()

ts = jnp.linspace(0.02, 0.1, 5)
sol_true = ds.pdesolve(problem_true, solver, (0.0, 0.1), y0, saveat=ds.SaveAt(ts=ts))
data = sol_true.ys + 1e-3 * np.random.default_rng(0).standard_normal(sol_true.ys.shape)
```

## The estimation problem

Here we parameterize the solution of the heat equation, \(\hat u(\theta)\),
with a learnable diffusivity denoted by the parameter \(\theta\), as in
[the least-squares problem](fitting.md#the-least-squares-problem). The
vanilla least-squares problem treats the data as exact and fits
\(\theta\) so that the solution matches the data as closely as possible.
In contrast, a statistical estimation problem treats the data as
uncertain, since every measurement carries noise, and under Bayesian
assumptions also treats the model as uncertain, since \(\theta\) is not
known before the data and what we know of it is incomplete. Both
uncertainties are stated as probability densities.

The uncertainty in the data is the likelihood
\(p(u_{\text{data}} \mid \theta)\), the probability of observing the data
if the diffusivity were \(\theta\). Under the assumption of additive
noise, it comes from the observation model

$$
u_{\text{data}} = \hat u(\theta) + \varepsilon,
$$

with \(\varepsilon\) the measurement noise, so the likelihood is the
density of \(\varepsilon\) evaluated at the misfit
\(u_{\text{data}} - \hat u(\theta)\). For independent Gaussian noise of
scale \(\sigma\) at every point,

$$
p(u_{\text{data}} \mid \theta) \propto \exp\!\Big(-\tfrac{1}{2\sigma^2}\,\big\|\hat u(\theta) - u_{\text{data}}\big\|_2^2\Big) = \exp\big(-\mathcal{L}(\theta)/\sigma^2\big),
$$

with \(\mathcal{L}\) the least-squares loss of
[fitting parameters](fitting.md#the-least-squares-problem).

In the Bayesian setting, we introduce the uncertainty in the model as
the distribution \(p(\theta)\), the prior, what we believe about the
diffusivity before the data, for example a normal density of mean
\(\theta_0\) and scale \(\tau\). We then use Bayes' rule to describe
the posterior as follows:

$$
p(\theta \mid u_{\text{data}}) = \frac{p(u_{\text{data}} \mid \theta)\,p(\theta)}{p(u_{\text{data}})} \propto p(u_{\text{data}} \mid \theta)\,p(\theta),
$$

where the posterior encodes how likely our model is given the measured
data. Maximum a posteriori estimation (MAP) then picks the most probable
diffusivity given the data,

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\theta \mid u_{\text{data}}) = \arg\min_\theta\,\big[-\log p(u_{\text{data}} \mid \theta) - \log p(\theta)\big].
$$

In a Bayesian setting, maximum likelihood estimation (MLE) can be
interpreted as the MAP under an uninformative prior, \(p(\theta) \propto 1\),

$$
\hat\theta_{\text{MLE}} = \arg\min_\theta\, \ell(\theta), \qquad \ell(\theta) = -\log p(u_{\text{data}} \mid \theta),
$$

with \(\ell\) the negative log likelihood. The posterior around the
estimate is what
[standard errors and covariance](#standard-errors-and-covariance),
[identifiability](identifiability.md), and
[posterior sampling](mcmc.md) describe. The full maximum-likelihood
fit:

```python
spec = ds.problem("heat")
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))
D = spec.param("D", value=0.1, learnable=True, bounds=(0.01, 2.0),
               prior=ds.inference.NormalPrior(mean=0.25, scale=0.1))       # the learnable diffusivity and its prior


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u)


spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)
problem = spec.build()
y0 = spec.assemble_state({})

measurement = ds.inference.Measurement("u", times=ts, values=data,
                                       likelihood=ds.inference.Gaussian(sigma=1e-3))   # declare the data and its noise model
experiment = ds.inference.Experiment(model=problem, solver=solver, t_span=(0.0, 0.1),
                                     initial_state=y0, measurements=(measurement,))     # bind the data to one solve
est = ds.inference.InferenceProblem(experiment, parameters=D, objective=ds.inference.MLE())   # pose the maximum-likelihood problem
fit = ds.inference.optsolve(est, ds.inference.LevenbergMarquardt())                     # solve it
fit[D]                                                                                  # the maximum-likelihood diffusivity
```

## Solving the estimation problem

The fit of [fitting parameters](fitting.md#solving-the-least-squares-problem)
carries over with two additions. `likelihood=` on the measurement states
the noise model of the data, and `objective=` on the inference problem
selects the maximum-likelihood or the maximum-a-posteriori estimate.
`spec.param` also takes `prior=`, the prior on the parameter
([priors and MAP](#priors-and-map)). `ds.inference.optsolve` and its
options are unchanged.

### Declare the measurement

`ds.inference.Measurement(name, times, values, *,
likelihood=ds.inference.Gaussian(sigma=1.0), observe=None, mask=None)`
declares one stream of observed values together with its noise model
([declare the data](fitting.md#declare-the-data)).

```python
measurement = ds.inference.Measurement("u", times=ts, values=data,
                                       likelihood=ds.inference.Gaussian(sigma=1e-3))
```

- `likelihood=` is the noise model of the measurement. The default is
  `ds.inference.Gaussian(sigma=1.0)`. The families are covered under
  [likelihoods](#likelihoods).

### Declare the experiment

`ds.inference.Experiment(model, solver, t_span, initial_state,
measurements, id=None, trajectory=None)` binds the measurements to one
solve of the model ([declare the experiment](fitting.md#declare-the-experiment)).

```python
experiment = ds.inference.Experiment(model=problem, solver=solver, t_span=(0.0, 0.1),
                                     initial_state=y0, measurements=(measurement,))
```

### Pose the estimation problem

`ds.inference.InferenceProblem(experiments, *, parameters,
objective=ds.inference.MLE(), ...)` pairs the experiment with its
objective and is the problem `ds.inference.optsolve` solves
([pose the inference problem](fitting.md#pose-the-inference-problem)).

```python
est = ds.inference.InferenceProblem(experiment, parameters=D, objective=ds.inference.MLE())
```

- `objective=` is `ds.inference.MLE()`, the default, or
  `ds.inference.MAP()`. `ds.inference.MAP()` requires a prior on every
  fitted parameter and is refused without one
  ([priors and MAP](#priors-and-map)).
- `est.with_start(fit)` returns the same problem started from an
  earlier fit of it.

### Analyze the result

The result is the `ds.inference.OptResult` of
[fitting parameters](fitting.md#analyze-the-result) with its statistical
fields filled. `fit[D]` returns the fitted value of the parameter `D`
by handle.

```python
fit[D]                  # the fitted value by handle
fit.objective           # MLE()
fit.log_likelihood      # the log likelihood at the fit
fit.log_prior           # None for a maximum-likelihood fit
```

The fit is the starting point of the analysis below,
[goodness of fit](#goodness-of-fit),
[standard errors and covariance](#standard-errors-and-covariance), and
[prediction bands](#prediction-bands), and of the
[identifiability](identifiability.md) analysis and
[posterior sampling](mcmc.md).

## Likelihoods

The likelihood is the noise model of a measurement, the density of
\(\varepsilon\) in the observation model. Independent Gaussian noise
of a known scale is the default and the textbook choice, and it is right
whenever the errors are additive, symmetric, and of a similar size
everywhere. The other families cover errors that are correlated across
the values of one time, errors with outliers, counts, and errors that
grow with the value. The families:

```python
ds.inference.Gaussian(sigma=1e-3)
ds.inference.CorrelatedGaussian(cholesky=1e-3 * jnp.eye(n))
ds.inference.StudentT(df=4.0, scale=1e-3)
ds.inference.Poisson(exposure=1.0)
ds.inference.LogNormal(scale=0.1)
ds.inference.MultiplicativeGaussian(scale=0.1)
```

- `ds.inference.Gaussian(sigma)` is independent Gaussian noise with
  standard deviation `sigma`, one value for every entry or one per
  entry. The fit is then the least-squares fit of
  [fitting parameters](fitting.md) with the weight \(1/\sigma\), and
  the scale sets the size of the standard errors.
- `ds.inference.CorrelatedGaussian(cholesky)` is Gaussian noise
  correlated across the values of one time, with covariance
  \(\Sigma = LL^\top\) and `cholesky` the lower factor \(L\). It is the
  choice when neighbouring sensors share their errors, at the price of
  one factor of the size of a frame.
- `ds.inference.StudentT(df, scale)` is heavy-tailed noise with `df`
  degrees of freedom and scale `scale`. It tolerates outliers, which a
  Gaussian fit follows, and approaches the Gaussian as `df` grows.
- `ds.inference.Poisson(exposure=1.0)` is the noise of counts, with the
  predicted value times `exposure` as the expected count. It is the
  choice for event counts, whose spread grows with the count.
- `ds.inference.LogNormal(scale)` is multiplicative noise, with the
  logarithm of each value Gaussian of standard deviation `scale` and the
  predicted value its mean. It is the choice for a positive quantity
  whose errors grow in proportion to it.
- `ds.inference.MultiplicativeGaussian(scale)` is Gaussian noise of
  standard deviation `scale` times the predicted value. An observed
  value of zero is refused under it, and such times are masked out.

Every family is a normalized density, so `fit.log_likelihood` is the
log probability of the data under the model and compares across
families. Whether the chosen family describes the data is checked after
the fit under [goodness of fit](#goodness-of-fit). For example, the
same fit under heavy-tailed noise:

```python
student_measurement = ds.inference.Measurement(
    "u", times=ts, values=data, likelihood=ds.inference.StudentT(df=4.0, scale=1e-3))
student_experiment = ds.inference.Experiment(
    model=problem, solver=solver, t_span=(0.0, 0.1), initial_state=y0,
    measurements=(student_measurement,))
student_fit = ds.inference.optsolve(
    ds.inference.InferenceProblem(student_experiment, parameters=D),
    ds.inference.LevenbergMarquardt())
student_fit[D]            # the heavy-tailed estimate
```

## Goodness of fit

`ds.inference.residual_report(problem, fit, *, experiment_id=None)`
checks the likelihood against the data after the fit. It standardizes
the residual of every measurement by its own likelihood, so that under
the declared noise model the standardized residuals have mean zero,
standard deviation one, and no correlation from one time to the next,
and it flags the departures in words:

```python
report = ds.inference.residual_report(est, fit)
report.channels["u"].std                     # near one when the noise scale is right
report.channels["u"].mean                    # near zero
report.channels["u"].lag1_autocorrelation    # near zero for independent errors
report.flags                                 # the departures from the noise model, empty here
```

- A standard deviation far from one says the declared scale is off by
  that factor, and the standard errors with it.
- A mean far from zero says the model misses the data on one side.
- A lag-1 autocorrelation far from zero says the errors are correlated
  in time.

Each entry of `channels` also carries `n_rows`, `cost`, `reduced_cost`,
and the `standardized` residuals themselves. `experiment_id=` selects the
experiment when several are fitted together.

## Priors and MAP

A prior states what we know about a parameter before the data, a
plausible value and how far from it we would be surprised. With a prior
declared on every fitted handle, `ds.inference.MAP()` minimizes the
negative log likelihood plus the negative log prior, and the fit is the
maximum-a-posteriori estimate. The prior matters when the data pin the
parameter down weakly, and it is negligible against informative data.
It is also what makes the posterior a distribution one can sample
([posterior sampling](mcmc.md)). The prior is declared on the handle,
`spec.param(..., prior=)` or `spec.function(..., prior=)`, and the
families:

```python
ds.inference.NormalPrior(mean=0.25, scale=0.1)
ds.inference.LogNormalPrior(median=0.25, log_sd=0.5)
ds.inference.GaussianCoefficientPrior(scale=1.0, smoothness=2.0)
```

- `ds.inference.NormalPrior(mean, scale)` is a normal prior with the
  given mean and standard deviation. It is the textbook choice, and with
  the Gaussian likelihood the MAP fit is the ridge-regularized
  least-squares fit of [regularization](fitting.md#regularization),
  centered on `mean`.
- `ds.inference.LogNormalPrior(median, log_sd)` is a log-normal prior on
  a positive parameter, with `median` its median and `log_sd` the
  standard deviation of its logarithm. It is the choice for a rate or a
  diffusivity that is positive and uncertain over orders of magnitude.
  It is refused on a function handle.
- `ds.inference.GaussianCoefficientPrior(scale, smoothness=2.0)`
  is a normal prior on the coefficients of a learnable function,
  centered on the declared coefficients, with a standard deviation that
  shrinks with the degree at the rate `smoothness`
  ([function approximation](function-approximation.md)).

The MAP fit of the same study uses the prior declared on `D`:

```python
est_map = ds.inference.InferenceProblem(experiment, parameters=D, objective=ds.inference.MAP())
fit_map = ds.inference.optsolve(est_map, ds.inference.LevenbergMarquardt())
fit_map[D]                # the maximum-a-posteriori diffusivity
fit_map.objective         # MAP()
fit_map.log_prior         # the log prior density at the fit
```

## Standard errors and covariance

Near the fit, the negative log likelihood is approximately quadratic,

$$
\ell(\theta) \approx \ell(\hat\theta) + \tfrac{1}{2}\,(\theta - \hat\theta)^\top I\,(\theta - \hat\theta),
$$

with \(I\) the information matrix, the curvature of the negative log
likelihood at the fit. It measures how sharply the data pin down each
parameter. Its inverse is the covariance of the estimate, and the
square roots of its diagonal are the standard errors. For a MAP fit the
curvature of the prior adds to \(I\).
`ds.inference.information(problem, *, at, kind="observed")`
assembles the information matrix at a point:

```python
info = ds.inference.information(est, at=fit)
info.observed            # the information matrix, here 1 by 1
info.rank()              # the number of directions the data determine
```

- `at=` is the point, a fit result or a mapping from handle to value,
  `{D: 0.3}`.
- `kind=` selects the curvature. `"observed"` is the curvature of the
  negative log likelihood at the data actually seen, and is the default.
  `"expected"` is the expected information, the curvature averaged over
  data drawn from the model, which for Gaussian noise is the Gauss-Newton
  curvature. Any other name is refused.
- `info.observed` and `info.expected` hold the matrix that was assembled.
  For a MAP problem, `info.prior_precision` is the curvature of the
  prior and `info.posterior_precision` the sum of both.
- `info.rank()` counts the directions the data determine. The full
  spectrum is the subject of [identifiability](identifiability.md).

`ds.inference.laplace(problem, fit, *, curvature="observed",
on_singular="raise")` builds the Laplace approximation, the Gaussian
centered on the fit with the inverse of the information matrix as its
covariance:

```python
posterior = ds.inference.laplace(est, fit)
posterior[D].sd                                  # the standard error of D
posterior[D].interval(0.95)                      # the 95% interval of the Gaussian
posterior.cov[D, D]                              # the variance of D
posterior.sample(jax.random.PRNGKey(0), 1000)    # draws from the Gaussian around the fit
```

- `fit` is the fit result the Gaussian is centered on, or a mapping
  from handle to value in its place.
- `curvature=` is `"observed"` or `"expected"`, as for `kind=` above.
  The default is `"observed"`. Any other name is refused.
- `on_singular=` says what happens when the data leave a direction
  undetermined. The default `"raise"` refuses the approximation.
  `"subspace"` returns it on the determined directions
  ([identifiability](identifiability.md)). Any other name is refused.
- `posterior[D]` is the marginal of one parameter, with `mean`, `sd`,
  and `interval(level)`. `posterior.cov[D, D]` is the covariance of two
  parameters. `posterior.marginal_sd()` gives every standard error at
  once, `posterior.covariance_dense()` the full covariance, and
  `posterior.sample(key, n)` draws.
- For a MAP fit `posterior.prior_spectrum` holds the eigenvalues of the
  prior curvature.
- A [multiple-shooting](multiple-shooting.md) fit that fits its window
  starts has them integrated out, so `information`, `laplace` and
  `identifiability` report the parameters alone.

For a MAP fit, `ds.inference.evidence(problem, fit, *,
curvature="observed")` uses the same Gaussian to approximate the log
evidence, the logarithm of the normalizer \(p(u_{\text{data}})\) of
Bayes' rule:

$$
\log p(u_{\text{data}}) \approx \log p(u_{\text{data}} \mid \hat\theta) + \log p(\hat\theta) + \tfrac{d}{2}\log 2\pi - \tfrac{1}{2}\log\det H,
$$

with \(d\) the number of parameters and \(H\) the information matrix plus
the curvature of the prior. Between priors or models fitted to the same
data, the larger evidence is preferred.

```python
ds.inference.evidence(est_map, fit_map).log_evidence    # the log evidence of the MAP fit
ds.inference.laplace(est_map, fit_map).log_evidence     # the same value
```

- The priors are the ones declared on the handles. A maximum-likelihood
  fit has no normalized prior and is refused.
- A fit that has not reached a stationary point is refused. The check
  tolerates a Newton decrement of 400 times the solver `rtol`, between
  0.1 and 0.5, and credits a float32 solve with an `rtol` of at least
  1e-3.

Uncertainty across the experiments of a population fit is estimated by
`ds.inference.cluster_covariance` and `ds.inference.bootstrap`
([hierarchical fits](hierarchical.md)).

## Prediction bands

`ds.inference.predict(problem, fit, *, experiment_id=None, kind="latent",
level=0.95)` carries the spread of the parameters through to the
predicted measurements. Each measurement is linearized at the fit and
the covariance of the Laplace approximation is propagated, giving one
band per measurement name:

```python
latent = ds.inference.predict(est, fit)["u"]                         # from the parameter uncertainty alone
predictive = ds.inference.predict(est, fit, kind="predictive")["u"]  # with the measurement noise added
latent.times, latent.mean, latent.sd, latent.lower, latent.upper    # over the observation times
```

- `kind="latent"` is the spread of the predicted values from the
  uncertainty of the parameters alone. `kind="predictive"` adds the
  noise of the measurement, the spread of a new observation.
- `level=` is the level of the interval `lower` to `upper`.
- `experiment_id=` selects the experiment when several are fitted
  together.
- The draws of [posterior sampling](mcmc.md) are accepted in place of
  the fit, and the band is then the spread over the draws.
