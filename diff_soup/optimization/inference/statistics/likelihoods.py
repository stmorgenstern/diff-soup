"""likelihoods.py — normalized observation models for fitting and inference.

A [Likelihood][diff_soup.inference.statistics.likelihoods.Likelihood] states the full probability of observed data given
predicted values — normalization constants included — so the same object can
drive weighted least squares, Fisher information, normalized log posteriors,
model comparison, and simulation. The supplied families cover fixed Gaussian
noise, counts, robust continuous measurements, and positive or multiplicative
errors:

* [Gaussian][diff_soup.inference.statistics.likelihoods.Gaussian] — independent errors with known (possibly per-observation)
  standard deviation.
* [CorrelatedGaussian][diff_soup.inference.statistics.likelihoods.CorrelatedGaussian] — one jointly Gaussian event with a known dense
  covariance, supplied through its lower Cholesky factor.
* [Poisson][diff_soup.inference.statistics.likelihoods.Poisson], [Multinomial][diff_soup.inference.statistics.likelihoods.Multinomial], and
  [DirichletMultinomial][diff_soup.inference.statistics.likelihoods.DirichletMultinomial] — normalized count events.
* [StudentT][diff_soup.inference.statistics.likelihoods.StudentT] — a heavy-tailed location likelihood.
* [LogNormal][diff_soup.inference.statistics.likelihoods.LogNormal] and [MultiplicativeGaussian][diff_soup.inference.statistics.likelihoods.MultiplicativeGaussian] — positive and
  explicitly parameter-dependent relative-error models.

The two Gaussian families have their usual weighted-least-squares rows.
`Poisson` and `StudentT` expose exact signed deviance rows as well, so their
point estimates can use the same least-squares engines without changing the
normalized likelihood being optimized. Other families are sampled or scored
through their scalar log probabilities.

All methods are pure and safe under ``jax.jit`` and ``jax.vmap``. Independent
channels add information only when the declared joint likelihood says they
are conditionally independent — combining likelihoods is the caller's
explicit declaration, never an implicit assumption.
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import special
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

__all__ = [
    "Likelihood",
    "Gaussian",
    "CorrelatedGaussian",
    "Poisson",
    "StudentT",
    "LogNormal",
    "MultiplicativeGaussian",
    "OutputInformation",
    "DiagonalOutputInformation",
    "CholeskyOutputInformation",
]

_LOG_2PI = math.log(2.0 * math.pi)


class OutputInformation(eqx.Module):
    """Expected Fisher information in prediction coordinates, as actions.

    Subclasses apply the Fisher metric without forming an inverse: ``apply``
    multiplies by ``I(mu)`` and ``whiten`` by a square root whose transpose
    product is that metric. For a fixed-covariance Gaussian this is the usual
    observation precision ``Σ⁻¹``; for count or parameter-dependent families
    it is the family-exact Fisher metric instead. ``log_det_covariance``
    returns the log determinant of the inverse metric retained under its
    original Gaussian-era name; likelihood density normalization always comes
    from
    [Likelihood.log_prob][diff_soup.inference.statistics.likelihoods.Likelihood.log_prob],
    never from this method.
    """

    def apply(self, v: Array) -> Array:
        """Multiply by the output-space precision ``Σ⁻¹``."""
        raise NotImplementedError

    def whiten(self, v: Array) -> Array:
        """Multiply by ``Σ^{-1/2}`` (unit-variance coordinates)."""
        raise NotImplementedError

    def log_det_covariance(self) -> Array:
        """Return ``log |Σ|`` for density normalization."""
        raise NotImplementedError


class DiagonalOutputInformation(OutputInformation):
    """Diagonal expected information ``diag(1/scale²)``.

    Attributes:
        scale: Inverse square roots of the diagonal Fisher entries. These are
            observation standard deviations only for fixed-scale Gaussians.
    """

    scale: Array

    def apply(self, v: Array) -> Array:
        """Multiply by the output-space precision ``Σ⁻¹``."""
        return v / self.scale**2

    def whiten(self, v: Array) -> Array:
        """Multiply by ``Σ^{-1/2}`` (unit-variance coordinates)."""
        return v / self.scale

    def log_det_covariance(self) -> Array:
        """Return ``log |Σ|`` for density normalization."""
        return 2.0 * jnp.sum(jnp.log(self.scale))


class CholeskyOutputInformation(OutputInformation):
    """Dense-covariance precision applied through triangular solves.

    Attributes:
        cholesky: Lower Cholesky factor ``L`` of the covariance ``Σ = L Lᵀ``.
    """

    cholesky: Array

    def _map_events(self, v: Array, *, transpose: bool) -> Array:
        """Solve one trailing-axis event at a time, preserving leading batches."""
        n = self.cholesky.shape[0]
        flat = jnp.reshape(v, (-1, n))
        matrix = self.cholesky.T if transpose else self.cholesky
        mapped = jax.vmap(lambda row: solve_triangular(matrix, row, lower=not transpose))(flat)
        return jnp.reshape(mapped, v.shape)

    def apply(self, v: Array) -> Array:
        """Multiply by the output-space precision ``Σ⁻¹``."""
        half = self._map_events(v, transpose=False)
        return self._map_events(half, transpose=True)

    def whiten(self, v: Array) -> Array:
        """Multiply by ``Σ^{-1/2}`` (unit-variance coordinates)."""
        return self._map_events(v, transpose=False)

    def log_det_covariance(self) -> Array:
        """Return ``log |Σ|`` for density normalization."""
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))


class Likelihood(eqx.Module):
    """Probability model of observed data given predicted values.

    Every implementation returns **normalized** log probabilities, can
    simulate observation replicates, exposes its prediction-coordinate score
    and expected Fisher information, and standardizes residuals for model
    checks. Families with
    an exact weighted-least-squares representation additionally define
    ``least_squares_rows(observed, predicted)``; its squared 2-norm equals
    ``-2 (log_prob - log_normalizer)``.

    The optional ``context`` keyword is reserved for observation metadata
    (exposures, totals, masks) that later families need; the Gaussian
    families ignore it.
    """

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized total log probability of the observed event(s)."""
        raise NotImplementedError

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate one observation replicate at the predicted values."""
        raise NotImplementedError

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return ``d log_prob / d predicted`` in prediction coordinates."""
        return jax.jacfwd(lambda mean: self.log_prob(observed, mean, context=context))(predicted)

    def expected_information(self, predicted: Array, *, context: Any = None) -> OutputInformation:
        """Return the output-space expected Fisher information as actions."""
        raise NotImplementedError

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return residuals scaled to unit variance under this model."""
        raise NotImplementedError

    def for_frames(
        self,
        keep: Array,
        *,
        full_prediction_shape: tuple,
        context: Any = None,
    ) -> tuple:
        """Return the likelihood/context view for statically kept frames.

        The base protocol has no frame-aligned fields, so it preserves both
        objects. Likelihoods owning pointwise array parameters override this
        method. Custom context-aware likelihoods can override it to select
        explicitly frame-aligned metadata without relying on ambiguous shape
        guessing.
        """
        del keep, full_prediction_shape
        return self, context

    def _saturated_prediction(self, observed: Array) -> Array:
        """Return the prediction whose expected observation is ``observed``.

        Callers that only hold data — the reference row scale a windowed
        problem needs, for instance — evaluate scale-bearing methods
        here. Location families predict the observation itself, which is what
        the base implementation returns; families predicting something else
        (a rate behind an exposure, say) override it.
        """
        return jnp.asarray(observed)


def _select_pointwise_field(
    value: Any,
    keep: Array,
    full_prediction_shape: tuple,
    *,
    who: str,
) -> Array:
    """Select a pointwise parameter while preserving per-output parameters."""
    value = jnp.asarray(value)
    trailing_shape = tuple(full_prediction_shape[1:])
    try:
        jnp.broadcast_to(value, trailing_shape)
    except ValueError:
        try:
            full = jnp.broadcast_to(value, full_prediction_shape)
        except ValueError as exc:
            raise ValueError(
                f"{who} with shape {value.shape} broadcasts to neither the "
                f"per-output shape {trailing_shape} nor the full unmasked "
                f"prediction shape {tuple(full_prediction_shape)}."
            ) from exc
        return full[jnp.asarray(keep)]
    return value


class Gaussian(Likelihood):
    """Independent Gaussian noise with standard deviation `sigma`.

    Pass `sigma`, or `scale` in its place, but not both.

    Args:
        sigma: The standard deviation, one value or one per entry of the measured values.
        scale: Another name for `sigma`.
    """

    scale: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, sigma=None, *, scale=None):
        """Declare a known standard deviation, scalar or per observation."""
        if sigma is not None and scale is not None:
            raise TypeError("Gaussian accepts sigma or scale, not both.")
        value = sigma if sigma is not None else scale
        if value is None:
            raise TypeError("Gaussian requires sigma=.")
        self.scale = jnp.asarray(value)

    @property
    def sigma(self):
        """Observation standard deviation."""
        return self.scale

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized total log probability of the observed event(s)."""
        z = (observed - predicted) / self.scale
        n = observed.size
        log_scales = jnp.sum(jnp.broadcast_to(jnp.log(self.scale), observed.shape))
        return -0.5 * jnp.sum(z**2) - log_scales - 0.5 * n * _LOG_2PI

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate one observation replicate at the predicted values."""
        return predicted + self.scale * jax.random.normal(key, predicted.shape)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the independent-Gaussian location score."""
        return (observed - predicted) / self.scale**2

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> DiagonalOutputInformation:
        """Return the output-space expected Fisher information as actions."""
        return DiagonalOutputInformation(scale=jnp.broadcast_to(self.scale, predicted.shape))

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return residuals scaled to unit variance under this model."""
        return (observed - predicted) / self.scale

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select a frame-aligned scale while preserving shared output scales."""
        scale = _select_pointwise_field(
            self.scale,
            keep,
            full_prediction_shape,
            who="Gaussian scale",
        )
        return eqx.tree_at(lambda likelihood: likelihood.scale, self, scale), context

    def least_squares_rows(
        self, observed: Array, predicted: Array, *, context: Any = None
    ) -> Array:
        """Whitened residual rows whose squared 2-norm is the data misfit."""
        return (observed - predicted) / self.scale

    def log_normalizer(self, predicted: Array, *, context: Any = None) -> Array:
        """Log of the density normalization at the predicted values."""
        log_scales = jnp.sum(jnp.broadcast_to(jnp.log(self.scale), predicted.shape))
        return -log_scales - 0.5 * predicted.size * _LOG_2PI


class CorrelatedGaussian(Likelihood):
    """Gaussian noise correlated across the values of one time, with covariance `cholesky @ cholesky.T`.

    For a measurement with one value per time, a factor with one row per time correlates the
    times instead.

    Args:
        cholesky: The lower Cholesky factor of the covariance, one row per value of a time.
    """

    cholesky: Array = eqx.field(converter=jnp.asarray)

    def _whiten_event(self, v: Array) -> Array:
        flat = v.reshape((-1, self.cholesky.shape[0]))
        z = jax.vmap(lambda row: solve_triangular(self.cholesky, row, lower=True))(flat)
        return z.reshape(v.shape)

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized total log probability of the observed event(s)."""
        z = self._whiten_event(observed - predicted)
        n_events = observed.size // self.cholesky.shape[0]
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))
        return -0.5 * jnp.sum(z**2) - 0.5 * n_events * (log_det + self.cholesky.shape[0] * _LOG_2PI)

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate one observation replicate at the predicted values."""
        eps = jax.random.normal(key, predicted.shape)
        flat = eps.reshape((-1, self.cholesky.shape[0]))
        colored = jax.vmap(lambda row: self.cholesky @ row)(flat)
        return predicted + colored.reshape(predicted.shape)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the joint-Gaussian location score."""
        residual = observed - predicted
        flat = residual.reshape((-1, self.cholesky.shape[0]))

        def precision(row):
            half = solve_triangular(self.cholesky, row, lower=True)
            return solve_triangular(self.cholesky.T, half, lower=False)

        return jax.vmap(precision)(flat).reshape(residual.shape)

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> CholeskyOutputInformation:
        """Return the output-space expected Fisher information as actions."""
        return CholeskyOutputInformation(cholesky=self.cholesky)

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return residuals scaled to unit variance under this model."""
        return self._whiten_event(observed - predicted)

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select a temporal covariance when scalar-output frames are masked."""
        n_frames = int(full_prediction_shape[0])
        temporal_event = len(full_prediction_shape) == 1 and self.cholesky.shape == (
            n_frames,
            n_frames,
        )
        if not temporal_event:
            return self, context
        covariance = self.cholesky @ self.cholesky.T
        keep = jnp.asarray(keep)
        selected = covariance[keep[:, None], keep[None, :]]
        cholesky = jnp.linalg.cholesky(selected)
        return (
            eqx.tree_at(lambda likelihood: likelihood.cholesky, self, cholesky),
            context,
        )

    def least_squares_rows(
        self, observed: Array, predicted: Array, *, context: Any = None
    ) -> Array:
        """Whitened residual rows whose squared 2-norm is the data misfit."""
        return self._whiten_event(observed - predicted)

    def log_normalizer(self, predicted: Array, *, context: Any = None) -> Array:
        """Log of the density normalization at the predicted values."""
        n_events = predicted.size // self.cholesky.shape[0]
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(self.cholesky)))
        return -0.5 * n_events * (log_det + self.cholesky.shape[0] * _LOG_2PI)


def _require_positive(name: str, value: Any) -> None:
    """Validate a likelihood constructor's positive scalar/array argument."""
    values = np.asarray(value)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must be finite and strictly positive; got {value!r}.")


def _valid_counts(observed: Array) -> Array:
    """Elementwise support check for non-negative integer-valued counts."""
    return (observed >= 0) & (observed == jnp.floor(observed))


class Poisson(Likelihood):
    """Independent Poisson noise of counts, with expected count `exposure` times the predicted value.

    The observed values must be counts, non-negative integers.

    Args:
        exposure: The positive factor from predicted value to expected count, one value or one
            per entry, by default 1.0.
    """

    exposure: Array = eqx.field(default=1.0, converter=jnp.asarray)

    def __post_init__(self):
        """Require a positive finite exposure."""
        _require_positive("Poisson exposure", self.exposure)

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized independent-Poisson log probability."""
        observed = jnp.asarray(observed)
        predicted = jnp.asarray(predicted)
        mean = self.exposure * predicted
        terms = special.xlogy(observed, mean) - mean - special.gammaln(observed + 1.0)
        valid = jnp.all(_valid_counts(observed)) & jnp.all(predicted >= 0)
        return jnp.where(valid, jnp.sum(terms), -jnp.inf)

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate independent counts at ``exposure * predicted``."""
        predicted = jnp.asarray(predicted)
        return jax.random.poisson(key, lam=self.exposure * predicted, shape=predicted.shape)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the rate score, using the finite zero-count limit."""
        observed = jnp.asarray(observed)
        predicted = jnp.asarray(predicted)
        return jnp.where(
            observed == 0,
            -self.exposure,
            observed / predicted - self.exposure,
        )

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> DiagonalOutputInformation:
        """Return exact rate-coordinate Fisher information ``exposure / rate``."""
        predicted = jnp.asarray(predicted)
        return DiagonalOutputInformation(scale=jnp.sqrt(predicted / self.exposure))

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return Pearson residuals, with degenerate zero events set to zero."""
        mean = self.exposure * jnp.asarray(predicted)
        residual = jnp.asarray(observed) - mean
        return jnp.where(mean == 0, jnp.zeros_like(residual), residual / jnp.sqrt(mean))

    def least_squares_rows(
        self, observed: Array, predicted: Array, *, context: Any = None
    ) -> Array:
        """Return signed Poisson deviance residuals.

        Half their squared norm is the negative log likelihood relative to
        the saturated model. The dimensionless form is evaluated with its
        series limit near ``observed == exposure * predicted`` so the row and
        its derivatives stay finite at a perfect fit, and its logarithm is
        taken at an argument bounded away from zero so a zero count also has
        a finite row and a finite derivative.
        """
        del context
        observed = jnp.asarray(observed)
        mean = self.exposure * jnp.asarray(predicted)
        safe_mean = jnp.where(mean > 0.0, mean, 1.0)
        relative = (observed - mean) / safe_mean
        # Below the switch the series is used, above it the closed form.
        # Truncating the series after its cubic term costs about
        # ``switch**4 / 15``, while the closed form loses about
        # ``2 * eps / switch`` to cancellation, and the two are equal at
        # ``switch**5 == 30 * eps``.
        eps = float(jnp.finfo(jnp.result_type(relative)).eps)
        switch = (30.0 * eps) ** 0.2
        use_closed = jnp.abs(relative) > switch
        # A zero count sits at ``relative == -1``, where the logarithm is
        # singular. Evaluating it at zero there leaves the closed form at its
        # limit ``-2 * relative`` — the factor ``1 + relative`` vanishes too —
        # with a finite derivative.
        safe_relative = jnp.where(relative > -1.0, relative, 0.0)
        deviance_over_mean = 2.0 * ((1.0 + relative) * jnp.log1p(safe_relative) - relative)
        nonzero_relative = jnp.where(use_closed, relative, 1.0)
        ratio = deviance_over_mean / nonzero_relative**2
        series = 1.0 - relative / 3.0 + relative**2 / 6.0 - relative**3 / 10.0
        ratio = jnp.where(use_closed, ratio, series)
        rows = relative * jnp.sqrt(safe_mean * jnp.maximum(ratio, 0.0))
        rows = jnp.where(
            mean > 0.0,
            rows,
            jnp.where(observed == 0.0, jnp.zeros_like(rows), jnp.full_like(rows, jnp.inf)),
        )
        # A count below zero is outside the support, so no deviance is defined.
        return jnp.where(observed < 0.0, jnp.full_like(rows, jnp.inf), rows)

    def _saturated_prediction(self, observed: Array) -> Array:
        """Return the rate whose expected count is ``observed``."""
        return jnp.asarray(observed) / self.exposure

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select frame-aligned exposures for kept predictions."""
        exposure = _select_pointwise_field(
            self.exposure,
            keep,
            full_prediction_shape,
            who="Poisson exposure",
        )
        return (
            eqx.tree_at(lambda likelihood: likelihood.exposure, self, exposure),
            context,
        )


class StudentT(Likelihood):
    """Independent Student-t noise with `df` degrees of freedom and scale `scale`.

    It tolerates outliers, and approaches Gaussian noise of standard deviation `scale` as `df`
    grows.

    Args:
        df: The degrees of freedom, positive, one value or one per entry of the measured values.
        scale: The scale of the noise, positive, one value or one per entry.
    """

    df: Array = eqx.field(converter=jnp.asarray)
    scale: Array = eqx.field(converter=jnp.asarray)

    def __post_init__(self):
        """Require positive degrees of freedom and scale."""
        _require_positive("StudentT df", self.df)
        _require_positive("StudentT scale", self.scale)

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized independent Student-t log probability."""
        residual = jnp.asarray(observed) - jnp.asarray(predicted)
        normalizer = (
            special.gammaln((self.df + 1.0) / 2.0)
            - special.gammaln(self.df / 2.0)
            - 0.5 * jnp.log(self.df * jnp.pi)
            - jnp.log(self.scale)
        )
        terms = normalizer - 0.5 * (self.df + 1.0) * jnp.log1p(
            residual**2 / (self.df * self.scale**2)
        )
        return jnp.sum(terms)

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate independent Student-t observation replicates."""
        predicted = jnp.asarray(predicted)
        return predicted + self.scale * jax.random.t(key, self.df, predicted.shape)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the bounded heavy-tail location score."""
        residual = jnp.asarray(observed) - jnp.asarray(predicted)
        return (self.df + 1.0) * residual / (self.df * self.scale**2 + residual**2)

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> DiagonalOutputInformation:
        """Return exact Student-t location Fisher information."""
        scale = self.scale * jnp.sqrt((self.df + 3.0) / (self.df + 1.0))
        return DiagonalOutputInformation(
            scale=jnp.broadcast_to(scale, jnp.asarray(predicted).shape)
        )

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the score residual standardized to unit model variance."""
        info_scale = self.scale * jnp.sqrt((self.df + 3.0) / (self.df + 1.0))
        return self.score(observed, predicted, context=context) * info_scale

    def least_squares_rows(
        self, observed: Array, predicted: Array, *, context: Any = None
    ) -> Array:
        """Return exact signed Student-t deviance residuals.

        Half their squared norm is the negative log likelihood relative to
        its location maximum. A series limit at zero keeps the residual
        differentiable when an observation is fitted exactly.
        """
        del context
        residual = jnp.asarray(observed) - jnp.asarray(predicted)
        standardized = residual / (jnp.sqrt(self.df) * self.scale)
        squared = standardized**2
        safe_squared = jnp.where(squared > 1.0e-8, squared, 1.0)
        ratio = jnp.log1p(squared) / safe_squared
        series = 1.0 - 0.5 * squared + squared**2 / 3.0
        ratio = jnp.where(squared > 1.0e-8, ratio, series)
        return standardized * jnp.sqrt((self.df + 1.0) * ratio)

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select frame-aligned degrees of freedom and scales."""
        df = _select_pointwise_field(self.df, keep, full_prediction_shape, who="StudentT df")
        scale = _select_pointwise_field(
            self.scale, keep, full_prediction_shape, who="StudentT scale"
        )
        view = eqx.tree_at(
            lambda likelihood: (likelihood.df, likelihood.scale),
            self,
            (df, scale),
        )
        return view, context


class LogNormal(Likelihood):
    """Independent log-normal noise, with the logarithm of each value Gaussian of standard deviation `scale`.

    The predicted value is the mean of the observed value, and both must be positive.

    Args:
        scale: The standard deviation of the logarithm, positive, one value or one per entry.
    """

    scale: Array = eqx.field(converter=jnp.asarray)

    def __post_init__(self):
        """Require a positive log-space standard deviation."""
        _require_positive("LogNormal scale", self.scale)

    def _log_residual(self, observed: Array, predicted: Array) -> Array:
        safe_observed = jnp.where(observed > 0, observed, 1.0)
        safe_predicted = jnp.where(predicted > 0, predicted, 1.0)
        return jnp.log(safe_observed) - jnp.log(safe_predicted) + 0.5 * self.scale**2

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized mean-preserving log-normal log probability."""
        observed = jnp.asarray(observed)
        predicted = jnp.asarray(predicted)
        residual = self._log_residual(observed, predicted)
        safe_observed = jnp.where(observed > 0, observed, 1.0)
        terms = (
            -0.5 * (residual / self.scale) ** 2
            - jnp.log(safe_observed)
            - jnp.log(self.scale)
            - 0.5 * _LOG_2PI
        )
        valid = jnp.all(observed > 0) & jnp.all(predicted > 0)
        return jnp.where(valid, jnp.sum(terms), -jnp.inf)

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate positive observations while preserving ``predicted`` as mean."""
        predicted = jnp.asarray(predicted)
        eps = jax.random.normal(key, predicted.shape)
        return predicted * jnp.exp(-0.5 * self.scale**2 + self.scale * eps)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the mean-coordinate log-normal score."""
        predicted = jnp.asarray(predicted)
        return self._log_residual(jnp.asarray(observed), predicted) / (self.scale**2 * predicted)

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> DiagonalOutputInformation:
        """Return exact mean-coordinate Fisher information."""
        predicted = jnp.asarray(predicted)
        return DiagonalOutputInformation(scale=self.scale * predicted)

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the unit-normal log-space residual."""
        return self._log_residual(jnp.asarray(observed), jnp.asarray(predicted)) / self.scale

    def least_squares_rows(self, observed, predicted, *, context=None):
        """Gaussian log-space rows for the mean-preserving observation model."""
        rows = self._log_residual(jnp.asarray(observed), jnp.asarray(predicted)) / self.scale
        return jnp.where((observed > 0) & (predicted > 0), rows, jnp.inf)

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select frame-aligned log-space scales."""
        scale = _select_pointwise_field(
            self.scale, keep, full_prediction_shape, who="LogNormal scale"
        )
        return eqx.tree_at(lambda likelihood: likelihood.scale, self, scale), context


class MultiplicativeGaussian(Likelihood):
    """Independent Gaussian noise with standard deviation `scale` times the absolute predicted value.

    An observed value of zero is refused unless its time is masked out.

    Args:
        scale: The relative standard deviation, positive, one value or one per entry.
    """

    scale: Array = eqx.field(converter=jnp.asarray)

    def __post_init__(self):
        """Require a positive relative standard deviation."""
        _require_positive("MultiplicativeGaussian scale", self.scale)

    def log_prob(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the normalized parameter-dependent Gaussian log probability."""
        observed = jnp.asarray(observed)
        predicted = jnp.asarray(predicted)
        safe = jnp.where(predicted != 0, predicted, 1.0)
        sd = self.scale * jnp.abs(safe)
        residual = (observed - safe) / sd
        terms = -0.5 * residual**2 - jnp.log(sd) - 0.5 * _LOG_2PI
        return jnp.where(jnp.all(predicted != 0), jnp.sum(terms), -jnp.inf)

    def sample(self, key: Array, predicted: Array, *, context: Any = None) -> Array:
        """Generate Gaussian observations with relative standard deviation."""
        predicted = jnp.asarray(predicted)
        return predicted + self.scale * jnp.abs(predicted) * jax.random.normal(key, predicted.shape)

    def score(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the full mean-and-variance score in prediction coordinates."""
        observed = jnp.asarray(observed)
        predicted = jnp.asarray(predicted)
        return observed * (observed - predicted) / (self.scale**2 * predicted**3) - 1.0 / predicted

    def expected_information(
        self, predicted: Array, *, context: Any = None
    ) -> DiagonalOutputInformation:
        """Return ``(scale**-2 + 2) / predicted**2`` exactly."""
        predicted = jnp.asarray(predicted)
        precision = (self.scale**-2 + 2.0) / predicted**2
        return DiagonalOutputInformation(scale=1.0 / jnp.sqrt(precision))

    def standardize(self, observed: Array, predicted: Array, *, context: Any = None) -> Array:
        """Return the unit-normal relative residual."""
        predicted = jnp.asarray(predicted)
        return (jnp.asarray(observed) - predicted) / (self.scale * jnp.abs(predicted))

    def least_squares_rows(self, observed, predicted, *, context=None):
        """Relative residual rows; the objective also includes the log determinant."""
        return (jnp.asarray(observed) - predicted) / (self.scale * predicted)

    def for_frames(
        self, keep: Array, *, full_prediction_shape: tuple, context: Any = None
    ) -> tuple:
        """Select frame-aligned relative-error scales."""
        scale = _select_pointwise_field(
            self.scale,
            keep,
            full_prediction_shape,
            who="MultiplicativeGaussian scale",
        )
        return eqx.tree_at(lambda likelihood: likelihood.scale, self, scale), context


class _MultiplicativeDeviance(MultiplicativeGaussian):
    """Exact deviance rows including the relative-noise normalization penalty."""

    def least_squares_rows(self, observed, predicted, *, context=None):
        observed, predicted = jnp.asarray(observed), jnp.asarray(predicted)
        a = (1 + jnp.sqrt(1 + 4 * self.scale**2)) / 2
        optimum = observed / a
        v = 1 - optimum / predicted
        small = jnp.abs(v) < 1e-4
        safe_v = jnp.where(small, 0.5, v)
        ratio = 2 * (-jnp.log(jnp.abs(1 - safe_v)) - safe_v) / safe_v**2
        ratio = jnp.where(small, 1 + 2 * v / 3 + v**2 / 2 + 2 * v**3 / 5 + v**4 / 3, ratio)
        return v * jnp.sqrt(jnp.maximum(a**2 / self.scale**2 + ratio, 0))


def __getattr__(name):
    if name in ("Multinomial", "DirichletMultinomial"):
        raise AttributeError(f"{name} was removed; use a supported measurement likelihood.")
    raise AttributeError(name)
