"""Guarded Laplace approximation to a normalized marginal likelihood.

Marginal likelihood is sensitive to constants and local geometry that do not
matter for point estimation. [diff_soup.inference.statistics.evidence.laplace_evidence][] therefore accepts only
a posterior carrying explicit evidence provenance: a normalized likelihood at
a checked stationary mode, an unregularized positive-definite curvature, and
the exact prior-curvature contribution used in that factor. The normalized
prior supplied to the function is evaluated directly and its local curvature
is verified against the posterior before the approximation is reported.

The result preserves fit provenance and distinguishes directions informed by
the data from data-null directions made proper by the prior. These host-side
reports intentionally favor explicit diagnostics over a JIT-compatible API;
inference and design paths should continue to use posterior solve actions.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.statistics.laplace import LaplacePosterior
from diff_soup.inference.statistics.priors import Prior

__all__ = [
    "EvidenceResult",
    "PriorDominatedDirection",
    "PriorSensitivity",
    "laplace_evidence",
]

_CURVATURE_RTOL = 2e-7
_CURVATURE_ATOL = 2e-10
_DEFAULT_NULL_TOL = 1e-10


@dataclasses.dataclass(frozen=True)
class PriorDominatedDirection:
    """One local data-null direction whose finite scale comes from the prior."""

    index: int
    name: str
    classification: str
    data_information: float
    prior_information: float
    loadings: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class PriorSensitivity:
    """Local information split showing how strongly the result uses the prior.

    ``prior_effective_dimension`` is ``tr(H⁻¹ H_prior)`` and
    ``data_effective_dimension`` is ``tr(H⁻¹ H_data)`` at the fitted mode.
    Their sum is the integration dimension (up to roundoff). The trace
    fraction is a simpler coordinate-dependent diagnostic and is labeled as
    such; the effective-dimension split is invariant under affine coordinate
    changes.
    """

    prior_effective_dimension: float
    data_effective_dimension: float
    prior_information_fraction: float
    data_null_dimension: int
    classification: str


@dataclasses.dataclass(frozen=True)
class EvidenceResult:
    """A guarded Laplace log-evidence value and its honesty diagnostics."""

    log_evidence: float
    log_likelihood_at_mode: float
    log_prior_at_mode: float
    log_det_curvature: float
    dimension: int
    coordinate: str
    curvature_kind: str
    likelihood_normalized: bool
    curvature_regularization: float
    mode_gradient_norm: float
    stationarity_tolerance: float
    prior_sensitivity: PriorSensitivity
    prior_dominated_directions: tuple[PriorDominatedDirection, ...]
    provenance: Any
    warnings: tuple[str, ...]


def _scalar(value: Any, *, what: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"laplace_evidence: {what} must be a scalar; got shape {array.shape}.")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"laplace_evidence: {what} must be finite; got {result!r}.")
    return result


def _prior_declarations(prior: Any) -> None:
    """Refuse unknown declarations for duck priors; built-ins own the contract."""
    if isinstance(prior, Prior):
        return
    proper = getattr(prior, "proper", None)
    if proper is not True:
        status = "improper" if proper is False else "not declared proper"
        raise ValueError(
            "laplace_evidence requires a proper prior with a normalized density; "
            f"the supplied prior is {status}."
        )
    normalized = getattr(prior, "normalized", None)
    if normalized is not True:
        status = "unnormalized" if normalized is False else "not declared normalized"
        raise ValueError(
            f"laplace_evidence requires a normalized prior density; the supplied prior is {status}."
        )


def _prior_curvature(prior: Any, mode: np.ndarray) -> np.ndarray:
    """Return the observed negative-log-prior Hessian at the integration mode."""
    point = jnp.asarray(mode)
    try:
        curvature = jax.hessian(lambda value: -prior.log_prob(value))(point)
        result = np.asarray(curvature, dtype=float)
    except Exception as hessian_error:  # pragma: no cover - fallback is for duck priors
        try:
            try:
                result = np.asarray(prior.precision(at=point), dtype=float)
            except TypeError:
                result = np.asarray(prior.precision(), dtype=float)
        except Exception as precision_error:
            raise ValueError(
                "laplace_evidence could not evaluate the prior curvature at the mode; "
                "the prior must expose a differentiable scalar log_prob or precision(at=...)."
            ) from precision_error
        if result.size == 0:
            raise ValueError(
                "laplace_evidence received an empty prior curvature."
            ) from hessian_error
    n = mode.size
    if result.shape != (n, n):
        raise ValueError(
            "laplace_evidence: prior curvature must match the posterior dimension "
            f"({n}, {n}); got {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("laplace_evidence: prior curvature contains non-finite values.")
    asymmetry = np.max(np.abs(result - result.T)) if result.size else 0.0
    scale = np.max(np.abs(result)) if result.size else 0.0
    if asymmetry > _CURVATURE_RTOL * max(float(scale), 1.0):
        raise ValueError("laplace_evidence: prior curvature is not symmetric at the mode.")
    return 0.5 * (result + result.T)


def _validate_factor(
    posterior: LaplacePosterior,
    data_curvature: np.ndarray,
    stored_prior: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Validate the retained Cholesky factor and its curvature provenance."""
    n = posterior.dim
    factor = np.asarray(posterior.precision_factor, dtype=float)
    if factor.shape != (n, n) or not np.all(np.isfinite(factor)):
        raise ValueError(
            "laplace_evidence requires a finite square Cholesky factor matching the mode."
        )
    scale = float(np.max(np.abs(factor))) if factor.size else 0.0
    if np.max(np.abs(np.triu(factor, k=1)), initial=0.0) > _CURVATURE_RTOL * max(scale, 1.0):
        raise ValueError("laplace_evidence: precision_factor is not a lower Cholesky factor.")
    diagonal = np.diag(factor)
    if diagonal.size != n or np.any(diagonal <= 0.0):
        raise ValueError(
            "laplace_evidence requires a positive definite (SPD) curvature with a valid "
            "positive-diagonal Cholesky factor."
        )
    total = factor @ factor.T
    if not np.all(np.isfinite(total)):
        raise ValueError("laplace_evidence: Cholesky factor reconstructs non-finite curvature.")
    expected = data_curvature + stored_prior
    if not np.allclose(total, expected, rtol=_CURVATURE_RTOL, atol=_CURVATURE_ATOL):
        raise ValueError(
            "laplace_evidence: precision factor does not match the recorded data and prior "
            "curvature; a valid unmodified SPD Cholesky factor is required."
        )
    try:
        np.linalg.cholesky(0.5 * (total + total.T))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "laplace_evidence requires a positive definite (SPD) posterior curvature."
        ) from exc
    return total, 2.0 * float(np.sum(np.log(diagonal)))


def _direction_name(posterior: LaplacePosterior, loading: np.ndarray, index: int) -> str:
    names = posterior.names or tuple(f"parameter[{i}]" for i in range(posterior.dim))
    dominant = int(np.argmax(np.abs(loading)))
    if np.isclose(abs(float(loading[dominant])), 1.0, rtol=0.0, atol=1e-8):
        return str(names[dominant])
    terms = sorted(
        ((abs(float(weight)), str(name), float(weight)) for name, weight in zip(names, loading)),
        reverse=True,
    )
    visible = [f"{weight:+.3g}·{name}" for magnitude, name, weight in terms if magnitude > 1e-6]
    return f"data-null[{index}] " + " ".join(visible)


def _null_tolerance(provenance: Any) -> float:
    if not isinstance(provenance, Mapping):
        return _DEFAULT_NULL_TOL
    tolerances = provenance.get("tolerances", {})
    if not isinstance(tolerances, Mapping):
        return _DEFAULT_NULL_TOL
    try:
        value = float(tolerances.get("null_tol", _DEFAULT_NULL_TOL))
    except (TypeError, ValueError):
        return _DEFAULT_NULL_TOL
    return value if np.isfinite(value) and value >= 0.0 else _DEFAULT_NULL_TOL


def _prior_reports(
    posterior: LaplacePosterior,
    *,
    data_curvature: np.ndarray,
    prior_curvature: np.ndarray,
    total_curvature: np.ndarray,
) -> tuple[PriorSensitivity, tuple[PriorDominatedDirection, ...]]:
    null_tol = _null_tolerance(posterior.provenance)
    spectrum, loadings = np.linalg.eigh(data_curvature)
    largest = float(np.max(np.abs(spectrum))) if spectrum.size else 0.0
    threshold = null_tol * largest
    reports = []
    for index in np.flatnonzero(np.abs(spectrum) <= threshold):
        loading = loadings[:, index]
        data_information = float(loading @ data_curvature @ loading)
        if abs(data_information) <= max(threshold, _CURVATURE_ATOL):
            data_information = 0.0
        prior_information = float(loading @ prior_curvature @ loading)
        reports.append(
            PriorDominatedDirection(
                index=int(index),
                name=_direction_name(posterior, loading, int(index)),
                classification="prior-dominated",
                data_information=data_information,
                prior_information=prior_information,
                loadings=tuple(float(value) for value in loading),
            )
        )

    prior_effective = float(np.trace(np.linalg.solve(total_curvature, prior_curvature)))
    data_effective = float(np.trace(np.linalg.solve(total_curvature, data_curvature)))
    total_trace = float(np.trace(total_curvature))
    prior_fraction = (
        float(np.trace(prior_curvature)) / total_trace if abs(total_trace) > 0.0 else math.nan
    )
    sensitivity = PriorSensitivity(
        prior_effective_dimension=prior_effective,
        data_effective_dimension=data_effective,
        prior_information_fraction=prior_fraction,
        data_null_dimension=len(reports),
        classification="prior-dominated" if reports else "data-and-prior-informed",
    )
    return sensitivity, tuple(reports)


def _geometry_warnings(posterior: LaplacePosterior, prior_dominated: int) -> tuple[str, ...]:
    warnings = [str(item) for item in posterior.warnings]
    if posterior.curvature_kind != "observed":
        warnings.append(
            f"Evidence uses {posterior.curvature_kind!r} curvature rather than the observed "
            "negative-log-joint Hessian; this is an additional geometry approximation."
        )
    spectrum = np.linalg.eigvalsh(np.asarray(posterior._data_curvature, dtype=float))
    scale = float(np.max(np.abs(spectrum))) if spectrum.size else 0.0
    if np.any(spectrum < -_DEFAULT_NULL_TOL * scale):
        warnings.append(
            "The observed data curvature is locally indefinite; the total posterior is SPD "
            "only after adding prior curvature."
        )
    provenance = posterior.provenance
    flags = provenance.get("geometry_flags", ()) if isinstance(provenance, Mapping) else ()
    if isinstance(flags, str):
        flags = (flags,)
    for flag in flags:
        label = str(flag).replace("_", " ")
        warnings.append(f"Posterior geometry flag: {label}.")
    if prior_dominated:
        warnings.append(
            f"{prior_dominated} local data-null direction(s) are finite only through prior "
            "curvature and are labeled prior-dominated."
        )
    return tuple(dict.fromkeys(warnings))


def laplace_evidence(posterior: LaplacePosterior, prior: Any) -> EvidenceResult:
    """Return a guarded Laplace approximation to the normalized log evidence.

    The approximation is evaluated in ``posterior.coordinate`` as
    ``log p(y | mode) + log p(mode) + d/2 log(2π) - 1/2 log det(H)``.
    The function refuses missing normalizers, an unchecked mode, any numerical
    curvature ridge, a mismatched prior or coordinate, and an invalid factor.

    Args:
        posterior: Proper Laplace posterior carrying explicit evidence
            provenance from
            [laplace_posterior][diff_soup.inference.statistics.laplace.laplace_posterior].
        prior: Proper normalized prior density in the same integration
            coordinate. Its value and observed local curvature are evaluated
            directly at the mode.

    Returns:
        Frozen [EvidenceResult][diff_soup.inference.statistics.evidence.EvidenceResult]
        with the decomposition, provenance,
        prior-sensitivity summary, and prior-dominated direction reports.

    Raises:
        ValueError: If any density, stationarity, curvature, factor, prior, or
            coordinate honesty guard cannot be established.
    """
    if not isinstance(posterior, LaplacePosterior):
        raise TypeError(
            f"laplace_evidence requires a LaplacePosterior; got {type(posterior).__name__}."
        )
    if posterior.likelihood_normalized is not True:
        status = "unknown" if posterior.likelihood_normalized is None else "false"
        raise ValueError(
            "laplace_evidence requires a normalized likelihood density; its normalization "
            f"declaration is {status}."
        )
    if posterior.log_likelihood_at_mode is None:
        raise ValueError(
            "laplace_evidence requires the normalized log-likelihood at the mode; it is missing."
        )
    log_likelihood = _scalar(
        posterior.log_likelihood_at_mode,
        what="log-likelihood at the mode",
    )
    if posterior.curvature_regularization is None:
        raise ValueError(
            "laplace_evidence: curvature regularization is unknown; an unknown ridge status "
            "cannot support evidence."
        )
    regularization = _scalar(
        posterior.curvature_regularization,
        what="curvature regularization",
    )
    if regularization != 0.0:
        raise ValueError(
            "laplace_evidence refuses ridged or otherwise regularized curvature; rebuild the "
            "observed factor without a numerical ridge."
        )
    if posterior.mode_gradient_norm is None or posterior.stationarity_tolerance is None:
        raise ValueError(
            "laplace_evidence requires a checked stationary mode with both gradient norm and "
            "stationarity tolerance recorded."
        )
    gradient_norm = _scalar(posterior.mode_gradient_norm, what="mode gradient norm")
    tolerance = _scalar(
        posterior.stationarity_tolerance,
        what="stationarity tolerance",
    )
    if gradient_norm < 0.0 or tolerance <= 0.0 or gradient_norm > tolerance:
        raise ValueError(
            "laplace_evidence refuses a non-stationary mode: the recorded gradient norm "
            f"{gradient_norm:.3g} exceeds the stationarity tolerance {tolerance:.3g}."
        )

    _prior_declarations(prior)
    coordinate = getattr(prior, "coordinate", None)
    if coordinate != posterior.coordinate:
        raise ValueError(
            "laplace_evidence requires prior and posterior densities in one integration "
            f"coordinate; got prior {coordinate!r} and posterior {posterior.coordinate!r}."
        )
    size = getattr(prior, "size", None)
    if size is not None and int(size) != posterior.dim:
        raise ValueError(
            "laplace_evidence: prior dimension does not match the posterior mode; "
            f"got {int(size)} and {posterior.dim}."
        )
    mode = np.asarray(posterior.mode, dtype=float)
    log_prior = _scalar(prior.log_prob(jnp.asarray(mode)), what="log-prior at the mode")
    if posterior.prior_precision is None:
        raise ValueError(
            "laplace_evidence cannot verify prior curvature: the posterior did not retain the "
            "prior contribution used in its factor."
        )
    stored_prior = np.asarray(posterior.prior_precision, dtype=float)
    prior_curvature = _prior_curvature(prior, mode)
    if stored_prior.shape != prior_curvature.shape or not np.allclose(
        stored_prior,
        prior_curvature,
        rtol=_CURVATURE_RTOL,
        atol=_CURVATURE_ATOL,
    ):
        raise ValueError(
            "laplace_evidence: supplied prior curvature/precision mismatch; the normalized "
            "prior is not the prior contribution stored in the posterior factor."
        )

    data_curvature = np.asarray(posterior._data_curvature, dtype=float)
    if data_curvature.shape != (posterior.dim, posterior.dim):
        raise ValueError("laplace_evidence requires the posterior's stored data curvature.")
    data_curvature = 0.5 * (data_curvature + data_curvature.T)
    total, log_det = _validate_factor(posterior, data_curvature, stored_prior)
    sensitivity, directions = _prior_reports(
        posterior,
        data_curvature=data_curvature,
        prior_curvature=prior_curvature,
        total_curvature=total,
    )
    log_evidence = (
        log_likelihood + log_prior + 0.5 * posterior.dim * math.log(2.0 * math.pi) - 0.5 * log_det
    )
    warnings = _geometry_warnings(posterior, len(directions))
    return EvidenceResult(
        log_evidence=log_evidence,
        log_likelihood_at_mode=log_likelihood,
        log_prior_at_mode=log_prior,
        log_det_curvature=log_det,
        dimension=posterior.dim,
        coordinate=posterior.coordinate,
        curvature_kind=posterior.curvature_kind,
        likelihood_normalized=True,
        curvature_regularization=regularization,
        mode_gradient_norm=gradient_norm,
        stationarity_tolerance=tolerance,
        prior_sensitivity=sensitivity,
        prior_dominated_directions=directions,
        provenance=posterior.provenance,
        warnings=warnings,
    )
