"""priors.py — normalized prior probability models for inference.

A [Prior][diff_soup.inference.statistics.priors.Prior] is a proper probability
model: its ``log_prob`` is a
normalized density in a declared coordinate, it can simulate draws for
prior-predictive checks, and Gaussian families expose their exact precision
for Laplace and design work. Because every density carries its coordinate
label, change-of-variables corrections are explicit rather than implicit:

* [GaussianPrior][diff_soup.inference.statistics.priors.GaussianPrior] — independent
  (diagonal-scale) or fully correlated
  (precision-matrix) Gaussian blocks.
* [TransformedPrior][diff_soup.inference.statistics.priors.TransformedPrior] — the
  exact pushforward of a base prior through a
  [Transform][diff_soup.inference.statistics.parameters.Transform].
* [IndependentProduct][diff_soup.inference.statistics.priors.IndependentProduct] — an
  explicit product of independent blocks.
* [pullback_log_prob][diff_soup.inference.statistics.priors.pullback_log_prob] —
  evaluate a physically-declared prior in the
  inference coordinate, including the ``log |dx/dz|`` correction.
* [prior_from_regularizer][diff_soup.inference.statistics.priors.prior_from_regularizer] —
  bridge a quadratic regularizer to the
  Gaussian prior it implies, but only when its mean, precision, and
  coordinate are all supplied and consistent; without them a regularizer
  stays regularization and cannot be used for evidence or normalized
  posterior densities.

* [NormalPrior][diff_soup.inference.statistics.priors.NormalPrior],
  [LogNormalPrior][diff_soup.inference.statistics.priors.LogNormalPrior], and
  [GaussianCoefficientPrior][diff_soup.inference.statistics.priors.GaussianCoefficientPrior]
  — the short physical-coordinate declarations an ``InferenceProblem`` with a
  ``MAP`` objective accepts per parameter handle; ``compile_prior`` lowers them
  (or a physical-coordinate ``GaussianPrior``) to normalized residual blocks.

All differentiable paths are pure ``jax.numpy`` and safe under ``jax.jit``
and ``jax.vmap``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

from diff_soup.inference.statistics.parameters import COORDINATES, Transform

__all__ = [
    "NormalPrior",
    "LogNormalPrior",
    "GaussianCoefficientPrior",
    "Prior",
    "GaussianPrior",
    "TransformedPrior",
    "IndependentProduct",
    "pullback_log_prob",
    "prior_from_regularizer",
]

_LOG_2PI = math.log(2.0 * math.pi)


class Prior(eqx.Module):
    """A proper probability model over a parameter block.

    Every implementation declares the coordinate its density lives in
    (``"x"`` physical or ``"z"`` inference), returns **normalized** log
    densities, and supports prior-predictive sampling. ``precision`` is
    optional: exact for Gaussian families, a local delta-method value for
    transformed ones.
    """

    def log_prob(self, value: Array) -> Array:
        """Return the normalized log density at ``value`` (declared coordinate)."""
        raise NotImplementedError

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw samples with shape ``sample_shape + (size,)``."""
        raise NotImplementedError

    def precision(self, at: Optional[Array] = None) -> Array:
        """Return the (n, n) precision — exact or local, see the subclass."""
        raise NotImplementedError

    @property
    def size(self) -> int:
        """Number of scalar parameters the prior covers."""
        raise NotImplementedError


def _check_coordinate(coordinate: str) -> str:
    if coordinate not in COORDINATES:
        raise ValueError(
            f"prior coordinate must be one of {COORDINATES} — 'x' for a density "
            f"on the physical parameters, 'z' for one on the inference "
            f"coordinate; got {coordinate!r}."
        )
    return coordinate


class GaussianPrior(Prior):
    """A proper Gaussian block prior, diagonal or fully correlated.

    Give either ``scale`` (independent per-scalar standard deviations) or
    ``precision`` (a full SPD precision matrix); exactly one must be
    supplied. The density is normalized in its declared coordinate.

    Attributes:
        mean: Prior mean, shape ``(n,)``.
        scale: Per-scalar standard deviations, shape ``(n,)``, or ``None``.
        precision_matrix: Full precision, shape ``(n, n)``, or ``None``.
        precision_cholesky: Lower Cholesky factor of ``precision_matrix``
            (``None`` for diagonal priors).
        coordinate: The coordinate of the density, ``"x"`` (default) or ``"z"``.
    """

    mean: Array
    scale: Optional[Array]
    precision_matrix: Optional[Array]
    precision_cholesky: Optional[Array]
    coordinate: str = eqx.field(static=True)

    def __init__(
        self,
        mean: Any,
        *,
        scale: Any = None,
        precision: Any = None,
        coordinate: str = "x",
    ):
        """Validate the exactly-one-of-scale/precision rule and factorize."""
        if (scale is None) == (precision is None):
            raise ValueError(
                "GaussianPrior needs exactly one of scale (independent standard "
                "deviations) and precision (a full SPD matrix)."
            )
        self.mean = jnp.asarray(mean)
        self.coordinate = _check_coordinate(coordinate)
        if scale is not None:
            self.scale = jnp.broadcast_to(jnp.asarray(scale), self.mean.shape)
            self.precision_matrix = None
            self.precision_cholesky = None
        else:
            self.scale = None
            self.precision_matrix = jnp.asarray(precision)
            self.precision_cholesky = jnp.linalg.cholesky(self.precision_matrix)

    @property
    def size(self) -> int:
        """Number of scalar parameters the prior covers."""
        return int(self.mean.shape[0])

    def log_prob(self, value: Array) -> Array:
        """Return the normalized log density at ``value`` (declared coordinate)."""
        dev = jnp.asarray(value) - self.mean
        n = self.mean.shape[0]
        if self.scale is not None:
            z = dev / self.scale
            return -0.5 * jnp.sum(z**2) - jnp.sum(jnp.log(self.scale)) - 0.5 * n * _LOG_2PI
        w = self.precision_cholesky.T @ dev
        log_det_precision = 2.0 * jnp.sum(jnp.log(jnp.diag(self.precision_cholesky)))
        return -0.5 * jnp.sum(w**2) + 0.5 * log_det_precision - 0.5 * n * _LOG_2PI

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw samples with shape ``sample_shape + (size,)``."""
        n = self.mean.shape[0]
        eps = jax.random.normal(key, tuple(sample_shape) + (n,))
        if self.scale is not None:
            return self.mean + self.scale * eps
        flat = eps.reshape((-1, n))
        colored = jax.vmap(lambda e: solve_triangular(self.precision_cholesky.T, e, lower=False))(
            flat
        )
        return self.mean + colored.reshape(eps.shape)

    def precision(self, at: Optional[Array] = None) -> Array:
        """Return the exact ``(n, n)`` precision (position-independent)."""
        if self.scale is not None:
            return jnp.diag(1.0 / self.scale**2)
        return self.precision_matrix


class TransformedPrior(Prior):
    """The exact pushforward of a base prior through a transform.

    If the base density lives in coordinate ``u`` and ``transform`` maps
    ``u`` to ``v = forward(u)``, this prior is the exact density of ``v``:
    ``log p(v) = base.log_prob(inverse(v)) - Σ log |d forward/du|`` at
    ``inverse(v)``. Its declared coordinate is the opposite of the base's.

    Attributes:
        base: The prior being transformed.
        transform: The elementwise change of variables.
    """

    base: Prior
    transform: Transform

    @property
    def size(self) -> int:
        """Number of scalar parameters the prior covers."""
        return self.base.size

    @property
    def coordinate(self) -> str:
        """The declared coordinate: the opposite side of the base's."""
        return "x" if self.base.coordinate == "z" else "z"

    def log_prob(self, value: Array) -> Array:
        """Return the normalized log density at ``value`` (declared coordinate)."""
        u = self.transform.inverse(jnp.asarray(value))
        return self.base.log_prob(u) - jnp.sum(self.transform.log_abs_det_jacobian(u))

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw samples with shape ``sample_shape + (size,)``."""
        return self.transform.forward(self.base.sample(key, sample_shape))

    def precision(self, at: Optional[Array] = None) -> Array:
        """Return the local delta-method precision at ``at``.

        The base precision is propagated through the Jacobian of the inverse
        map linearized at ``at`` (expressed in this prior's coordinate):
        ``Jᵀ P_base J`` with ``J = d inverse/d value``. This is a first-order
        local value, not the exact curvature of the transformed density.

        Raises:
            ValueError: If ``at`` is not given — a transformed precision has
                no position-independent value.
        """
        if at is None:
            raise ValueError(
                "TransformedPrior.precision needs the linearization point: pass "
                "precision(at=value) in this prior's declared coordinate."
            )
        jac = jax.jacfwd(self.transform.inverse)(jnp.asarray(at))
        return jac.T @ self.base.precision() @ jac


class IndependentProduct(Prior):
    """An explicit product of independent block priors.

    The blocks concatenate in order into one flat vector; independence across
    blocks is this object's declaration, not an assumption.

    Attributes:
        priors: The independent blocks, in concatenation order.
    """

    priors: tuple

    def __init__(self, priors: Any):
        """Validate a non-empty collection of same-coordinate priors."""
        priors = tuple(priors)
        if not priors:
            raise ValueError("IndependentProduct needs at least one prior block.")
        coords = {p.coordinate for p in priors}
        if len(coords) != 1:
            raise ValueError(
                f"IndependentProduct blocks must share one coordinate; got {sorted(coords)}."
            )
        self.priors = priors

    @property
    def size(self) -> int:
        """Number of scalar parameters the prior covers."""
        return sum(p.size for p in self.priors)

    @property
    def coordinate(self) -> str:
        """The shared declared coordinate of every block."""
        return self.priors[0].coordinate

    def _split(self, value: Array) -> list:
        parts = []
        start = 0
        for p in self.priors:
            parts.append(value[start : start + p.size])
            start += p.size
        return parts

    def log_prob(self, value: Array) -> Array:
        """Return the normalized log density at ``value`` (declared coordinate)."""
        value = jnp.asarray(value)
        parts = self._split(value)
        total = jnp.zeros(())
        for p, v in zip(self.priors, parts):
            total = total + p.log_prob(v)
        return total

    def sample(self, key: Array, sample_shape: tuple = ()) -> Array:
        """Draw samples with shape ``sample_shape + (size,)``."""
        keys = jax.random.split(key, len(self.priors))
        draws = [p.sample(k, sample_shape) for p, k in zip(self.priors, keys)]
        return jnp.concatenate(draws, axis=-1)

    def precision(self, at: Optional[Array] = None) -> Array:
        """Return the block-diagonal ``(n, n)`` precision."""
        if at is None:
            blocks = [p.precision() for p in self.priors]
        else:
            at = jnp.asarray(at)
            blocks = []
            for p, v in zip(self.priors, self._split(at)):
                try:
                    blocks.append(p.precision())
                except (ValueError, NotImplementedError):
                    blocks.append(p.precision(at=v))
        return jax.scipy.linalg.block_diag(*blocks)


def pullback_log_prob(prior: Prior, transform: Transform, z: Array) -> Array:
    """Evaluate a physically-declared prior in the inference coordinate.

    For a prior density on the physical value ``x`` and a coordinate map
    ``x = transform.forward(z)``, the induced density of ``z`` is
    ``log p(x(z)) + Σ log |dx/dz|`` — sampling or optimizing in ``z`` without
    the Jacobian term would target a different distribution.

    Args:
        prior: The prior whose density is declared on the transformed value.
        transform: The map from the evaluation coordinate to the prior's.
        z: The point in the evaluation coordinate.

    Returns:
        The normalized log density of ``z``.
    """
    z = jnp.asarray(z)
    x = transform.forward(z)
    return prior.log_prob(x) + jnp.sum(transform.log_abs_det_jacobian(z))


def prior_from_regularizer(
    regularizer: Any,
    *,
    mean: Any,
    precision: Any,
    coordinate: str,
) -> GaussianPrior:
    """Bridge a quadratic regularizer to the Gaussian prior it implies.

    A quadratic penalty only defines a normalized prior once its mean,
    precision, and coordinate are stated; this constructor requires all
    three and verifies on a deterministic probe set that the regularizer's
    penalty really is ``0.5 (v - mean)ᵀ P (v - mean)``. Any mismatch means
    the object is regularization, not a probability model, and it is
    rejected — evidence and normalized posteriors must not be built from it.

    Args:
        regularizer: An object with ``penalty(x) -> scalar`` (e.g.
            ``BlockRegularization``) acting on the same flat vector the
            prior covers.
        mean: The implied prior mean, shape ``(n,)``.
        precision: The implied SPD precision, shape ``(n, n)``.
        coordinate: The coordinate the density lives in (``"x"`` or ``"z"``).

    Returns:
        The verified [GaussianPrior][diff_soup.inference.statistics.priors.GaussianPrior].

    Raises:
        ValueError: If any metadata is missing or the penalty disagrees with
            the stated quadratic form.
    """
    if mean is None or precision is None or coordinate is None:
        raise ValueError(
            "prior_from_regularizer needs the full probability metadata: mean, "
            "precision, and coordinate. A quadratic penalty without them is "
            "regularization, not a normalized prior."
        )
    mean = np.asarray(mean, dtype=float)
    precision_np = np.asarray(precision, dtype=float)
    n = mean.shape[0]

    def quadratic(v: np.ndarray) -> float:
        dev = v - mean
        return float(0.5 * dev @ precision_np @ dev)

    probes = [mean]
    for i in range(n):
        e = mean.copy()
        e[i] += 1.0
        probes.append(e)
    for i in range(n - 1):
        e = mean.copy()
        e[i] += 1.0
        e[i + 1] -= 0.5
        probes.append(e)
    probes.append(mean + np.ones(n))
    for v in probes:
        stated = quadratic(v)
        actual = float(regularizer.penalty(jnp.asarray(v)))
        if not np.isclose(actual, stated, rtol=1e-8, atol=1e-12):
            raise ValueError(
                "prior_from_regularizer: the regularizer's penalty does not "
                f"match the stated Gaussian form (probe misfit {actual:.6e} vs "
                f"{stated:.6e}). State the regularizer's true mean/precision or "
                "keep it as regularization."
            )
    return GaussianPrior(mean=mean, precision=precision_np, coordinate=coordinate)


# ---------------------------------------------------------------------------
# Physical-coordinate prior declarations for EstimationProblem
# ---------------------------------------------------------------------------
# These are the short spellings an ``InferenceProblem`` with a ``MAP`` objective
# accepts per parameter handle. ``compile_prior`` lowers each declaration (or a
# physical-coordinate ``GaussianPrior``) to a residual block with a normalized
# log density.


@dataclasses.dataclass(frozen=True)
class NormalPrior:
    """A normal prior with mean `mean` and standard deviation `scale`.

    Declare it as `prior=` of `spec.param` or `spec.function`, for a `MAP` fit.

    Args:
        mean: The mean, one value or one per entry of the parameter.
        scale: The standard deviation, positive, one value or one per entry.
    """

    mean: Any
    scale: Any

    def __post_init__(self):
        """Reject non-positive standard deviations eagerly."""
        if not bool(np.all(np.asarray(self.scale) > 0.0)):
            raise ValueError("NormalPrior scale must be strictly positive.")


@dataclasses.dataclass(frozen=True)
class LogNormalPrior:
    """A log-normal prior with median `median` and standard deviation `log_sd` of the logarithm.

    Declare it as `prior=` of `spec.param`, for a `MAP` fit of a positive parameter.

    Args:
        median: The median, positive, one value or one per entry of the parameter.
        log_sd: The standard deviation of the parameter's logarithm, positive, one value or one
            per entry.
    """

    median: Any
    log_sd: Any

    def __post_init__(self):
        """Validate the positive median and log-space deviation."""
        if not bool(np.all(np.asarray(self.median) > 0.0)):
            raise ValueError("LogNormalPrior median must be strictly positive.")
        if not bool(np.all(np.asarray(self.log_sd) > 0.0)):
            raise ValueError("LogNormalPrior log_sd must be strictly positive.")


@dataclasses.dataclass(frozen=True)
class GaussianCoefficientPrior:
    """A normal prior on the coefficients of a function, centered on its declared coefficients.

    Declare it as `prior=` of `spec.function`, for a `MAP` fit. Coefficient `n`, counted from 0,
    has standard deviation `scale / (1 + n * (n + 1)) ** (smoothness / 2)`.

    Args:
        scale: The standard deviation of the first coefficient, positive.
        smoothness: How fast the standard deviation falls with the coefficient index,
            non-negative.
    """

    scale: float
    smoothness: float = 2.0

    def __post_init__(self):
        """Validate positive scale and non-negative smoothness."""
        if float(self.scale) <= 0.0:
            raise ValueError("GaussianCoefficientPrior scale must be strictly positive.")
        if float(self.smoothness) < 0.0:
            raise ValueError("GaussianCoefficientPrior smoothness must be non-negative.")


@dataclasses.dataclass(frozen=True)
class _NormalBlock:
    """Compiled diagonal normal block with residual and normalized density."""

    mean: Any
    scale: Any

    @property
    def size(self) -> int:
        return int(self.mean.size)

    def residual(self, value):
        """Rows whose half squared norm is the non-constant negative log density."""
        return (jnp.asarray(value) - self.mean) / self.scale

    def log_prob(self, value):
        """Normalized physical-coordinate log density."""
        z = self.residual(value)
        return -0.5 * jnp.sum(z**2) - jnp.sum(jnp.log(self.scale)) - 0.5 * self.size * _LOG_2PI


@dataclasses.dataclass(frozen=True)
class _LogNormalBlock:
    """Compiled independent log-normal block."""

    log_median: Any
    log_sd: Any

    @property
    def size(self) -> int:
        return int(self.log_median.size)

    def residual(self, value):
        """Exact physical-MAP rows, including the density's ``log(value)`` term."""
        log_value = jnp.log(jnp.asarray(value))
        mode_log = self.log_median - self.log_sd**2
        return (log_value - mode_log) / self.log_sd

    def log_prob(self, value):
        """Normalized physical-coordinate log density."""
        value = jnp.asarray(value)
        log_value = jnp.log(value)
        z = (log_value - self.log_median) / self.log_sd
        return (
            -0.5 * jnp.sum(z**2)
            - jnp.sum(log_value)
            - jnp.sum(jnp.log(self.log_sd))
            - 0.5 * self.size * _LOG_2PI
        )


@dataclasses.dataclass(frozen=True)
class _PrecisionNormalBlock:
    """Compiled correlated normal block represented by a precision Cholesky."""

    mean: Any
    precision_cholesky: Any

    @property
    def size(self) -> int:
        return int(self.mean.size)

    def residual(self, value):
        """Whiten deviations under the full precision matrix."""
        return self.precision_cholesky.T @ (jnp.asarray(value) - self.mean)

    def log_prob(self, value):
        """Normalized correlated normal density."""
        rows = self.residual(value)
        log_det_precision = 2.0 * jnp.sum(jnp.log(jnp.diag(self.precision_cholesky)))
        return -0.5 * jnp.sum(rows**2) + 0.5 * log_det_precision - 0.5 * self.size * _LOG_2PI


def _broadcast_flat(value: Any, size: int, *, name: str) -> Any:
    """Broadcast a declaration to one flat block with a clear shape error."""
    try:
        return jnp.ravel(jnp.broadcast_to(jnp.asarray(value), (size,)))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or broadcast to {size} parameter values.") from exc


def _initial_function_values(handle: Any, size: int) -> Any:
    """Flatten the declared differentiable leaves of a function handle."""
    dynamic, _static = eqx.partition(handle.fn, eqx.is_inexact_array)
    flat, _unravel = ravel_pytree(dynamic)
    if int(flat.size) != size:
        raise ValueError(
            f"function prior for {handle.name!r} expected {size} coefficients but "
            f"the declaration exposes {int(flat.size)}."
        )
    return flat


def compile_prior(declaration: Any, handle: Any, size: int) -> Any:
    """Compile one public or legacy prior declaration for one handle occurrence."""
    if isinstance(declaration, NormalPrior):
        return _NormalBlock(
            mean=_broadcast_flat(declaration.mean, size, name="NormalPrior mean"),
            scale=_broadcast_flat(declaration.scale, size, name="NormalPrior scale"),
        )
    if isinstance(declaration, LogNormalPrior):
        if getattr(handle, "kind", None) == "function":
            raise TypeError(
                "LogNormalPrior applies to physical scalar/array parameters, not functions."
            )
        return _LogNormalBlock(
            log_median=jnp.log(
                _broadcast_flat(declaration.median, size, name="LogNormalPrior median")
            ),
            log_sd=_broadcast_flat(declaration.log_sd, size, name="LogNormalPrior log_sd"),
        )
    if isinstance(declaration, GaussianCoefficientPrior):
        if getattr(handle, "kind", None) != "function":
            raise TypeError("GaussianCoefficientPrior requires a spec.function(...) handle.")
        degree = jnp.arange(size)
        weight = (1.0 + degree * (degree + 1.0)) ** (0.5 * declaration.smoothness)
        return _NormalBlock(
            mean=_initial_function_values(handle, size),
            scale=jnp.full((size,), declaration.scale) / weight,
        )

    # A normalized GaussianPrior on the physical coordinate is accepted as an
    # advanced declaration alongside the shorter NormalPrior spelling.
    if isinstance(declaration, GaussianPrior):
        if declaration.coordinate != "x":
            raise ValueError("MAP priors must be declared in physical coordinate 'x'.")
        if declaration.size != size:
            raise ValueError(
                f"prior for {handle.name!r} covers {declaration.size} values; expected {size}."
            )
        if declaration.scale is not None:
            return _NormalBlock(mean=declaration.mean, scale=declaration.scale)
        return _PrecisionNormalBlock(
            mean=declaration.mean,
            precision_cholesky=declaration.precision_cholesky,
        )
    raise TypeError(
        f"unsupported prior {type(declaration).__name__} on handle {handle.name!r}; "
        "use NormalPrior, LogNormalPrior, or GaussianCoefficientPrior."
    )
