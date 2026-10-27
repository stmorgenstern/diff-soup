"""Observable-compression comparisons under one declared generative model.

The full observation is the ordered product of an experiment's response
channels after their per-channel observables, kernels, and masks. A summary is
then applied to a draw from that *same* source event. This distinction keeps an
instrument kernel (noise declared after the kernel) separate from statistical
compression (noise induced by transforming an already noisy observation).

Two induced-likelihood modes are explicit:

* ``"analytic"`` supports
  [AffineSummary][diff_soup.inference.statistics.compression.AffineSummary] applied to fixed-covariance
  Gaussian source channels. The induced mean and covariance are exact.
* ``"simulate"`` fits a labeled Gaussian synthetic likelihood to paired source
  simulations. Its Fisher information includes derivatives of both the
  summary mean and covariance, and Monte Carlo standard errors are reported.

Exact comparisons enforce the data-processing inequality. Simulation-based
comparisons never clip apparent violations; they retain the estimate and flag
the violation alongside its uncertainty.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.inference.statistics.experiment import _channel_likelihood_view
from diff_soup.inference.statistics.information import (
    _expected_information_core,
    _resolve_at,
)
from diff_soup.inference.statistics.likelihoods import CorrelatedGaussian, Gaussian

__all__ = ["AffineSummary", "CompressionReport", "compare_compression"]

_DPI_RTOL = 2e-9
_PSD_RTOL = 1e-10


class AffineSummary(eqx.Module):
    """Explicit affine map of the flattened ordered full observation.

    The event is flattened by response-channel declaration order and native
    array order. Explicit structure is required in analytic mode so an
    arbitrary nonlinear callable is never silently treated with a delta-method
    Gaussian likelihood.

    Attributes:
        matrix: Dense map ``A``, shape ``(n_summary, n_full)``.
        offset: Optional offset ``b``, shape ``(n_summary,)``.
    """

    matrix: Array
    offset: Optional[Array]

    def __init__(self, matrix: Any, offset: Any = None):
        """Validate and store ``summary(y) = matrix @ vec(y) + offset``."""
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError(
                "AffineSummary matrix must have shape (n_summary, n_full) with "
                f"at least one summary row; got {matrix.shape}."
            )
        if not np.all(np.isfinite(np.asarray(matrix))):
            raise ValueError("AffineSummary matrix must contain only finite values.")
        if offset is None:
            offset_array = None
        else:
            offset_array = jnp.asarray(offset)
            if offset_array.shape != (matrix.shape[0],):
                raise ValueError(
                    f"AffineSummary offset must have shape ({matrix.shape[0]},); "
                    f"got {offset_array.shape}."
                )
            if not np.all(np.isfinite(np.asarray(offset_array))):
                raise ValueError("AffineSummary offset must contain only finite values.")
        self.matrix = matrix
        self.offset = offset_array

    def __call__(self, event: dict) -> Array:
        """Apply the affine map to one ordered response-event mapping."""
        flat = jnp.concatenate([jnp.ravel(value) for value in event.values()])
        result = self.matrix @ flat
        return result if self.offset is None else result + self.offset


@dataclasses.dataclass(frozen=True)
class CompressionReport:
    """Information retained by summaries of one common full observation.

    Array fields are host-side NumPy reports. ``per_direction_retention`` uses
    eigendirections of the full information (prior-whitened when a proper SPD
    prior is supplied). Source-null directions are ``NaN`` and identified by
    ``identified_directions``.
    """

    names: tuple
    full_information: np.ndarray
    summary_information: dict
    full_prior_relative_eigenvalues: Optional[np.ndarray]
    prior_relative_eigenvalues: dict
    posterior_trace_change: dict
    posterior_log_volume_change: dict
    full_directions: np.ndarray
    identified_directions: np.ndarray
    per_direction_retention: dict
    induced_kind: dict
    estimate_error: dict
    dpi: dict
    flags: tuple
    notes: tuple
    evaluated_at: dict

    @property
    def posterior_volume_change(self) -> dict:
        """Return posterior-volume fractional changes derived from log changes."""
        return {
            name: None if value is None else float(np.expm1(value))
            for name, value in self.posterior_log_volume_change.items()
        }

    @property
    def direction_retention(self) -> dict:
        """Backward-friendly alias of [per_direction_retention][diff_soup.inference.statistics.compression.CompressionReport.per_direction_retention]."""
        return self.per_direction_retention


def _symmetrize(matrix: Any) -> np.ndarray:
    """Return the symmetric part ``(A + Aᵀ) / 2`` of ``matrix`` as a float array."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _block_diagonal(blocks: list) -> np.ndarray:
    """Assemble the square ``blocks`` into one block-diagonal float matrix."""
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        result[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return result


def _psd_pseudoinverse(matrix: Any, *, who: str) -> np.ndarray:
    """Pseudoinverse of a symmetric positive-semidefinite matrix by eigendecomposition.

    Eigenvalues below a scale-relative tolerance are dropped; a negative eigenvalue
    beyond that tolerance raises ``ValueError`` naming ``who``.
    """
    matrix = _symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return np.zeros_like(matrix)
    tolerance = _PSD_RTOL * max(1, matrix.shape[0]) * scale
    if float(np.min(values, initial=0.0)) < -tolerance:
        raise ValueError(
            f"{who}: induced covariance is not positive semidefinite "
            f"(minimum eigenvalue {float(np.min(values)):.3e})."
        )
    keep = values > tolerance
    if not np.any(keep):
        return np.zeros_like(matrix)
    return (vectors[:, keep] / values[keep]) @ vectors[:, keep].T


def _gaussian_source_covariance(response: tuple, predictions: dict) -> np.ndarray:
    """Build the exact block covariance of fixed-Gaussian response channels."""
    blocks = []
    for channel in response:
        predicted = jnp.asarray(predictions[channel.name])
        likelihood, _ = _channel_likelihood_view(channel, predicted)
        if isinstance(likelihood, Gaussian):
            try:
                scale = jnp.broadcast_to(likelihood.scale, predicted.shape)
            except ValueError as exc:
                raise ValueError(
                    f"compare_compression: Gaussian scale for channel "
                    f"{channel.name!r} does not broadcast to its post-kernel, "
                    f"post-mask prediction shape {predicted.shape}."
                ) from exc
            blocks.append(np.diag(np.asarray(scale, dtype=float).reshape(-1) ** 2))
            continue
        if isinstance(likelihood, CorrelatedGaussian):
            event_size = int(likelihood.cholesky.shape[0])
            if predicted.size % event_size:
                raise ValueError(
                    f"compare_compression: channel {channel.name!r} prediction has "
                    f"{predicted.size} values, not a whole number of correlated "
                    f"events of size {event_size}."
                )
            covariance = np.asarray(likelihood.cholesky) @ np.asarray(likelihood.cholesky).T
            blocks.append(np.kron(np.eye(predicted.size // event_size), covariance))
            continue
        raise ValueError(
            "compare_compression induced='analytic' requires every response "
            "channel to use Gaussian or CorrelatedGaussian with fixed known "
            f"covariance; channel {channel.name!r} uses "
            f"{type(likelihood).__name__}. Use induced='simulate'."
        )
    return _block_diagonal(blocks)


def _flatten_prediction_jacobians(
    response: tuple, predictions: dict, jacobians: dict, n_parameters: int
) -> tuple:
    """Stack per-channel predictions and Jacobians into one mean vector and one Jacobian."""
    mean = np.concatenate(
        [np.asarray(predictions[channel.name]).reshape(-1) for channel in response]
    )
    jacobian = np.concatenate(
        [np.asarray(jacobians[channel.name]).reshape((-1, n_parameters)) for channel in response],
        axis=0,
    )
    return mean, jacobian


def _moments(samples: np.ndarray, *, who: str) -> tuple:
    """Sample mean and symmetrized unbiased covariance of a ``(replicate, value)`` array."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(
            f"{who}: summary simulations must form a 2-D (replicate, value) "
            f"array with at least two replicates; got {samples.shape}."
        )
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    return mean, _symmetrize(covariance)


def _strict_spd_solve(covariance: Any, rhs: Any, *, who: str) -> np.ndarray:
    """Solve ``covariance @ x = rhs`` by Cholesky, refusing a non-SPD matrix without a ridge."""
    covariance = _symmetrize(covariance)
    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        values = np.linalg.eigvalsh(covariance)
        raise ValueError(
            f"{who}: the simulation-estimated summary covariance is not SPD "
            f"(minimum eigenvalue {float(values.min()):.3e}). Increase "
            "n_simulations, remove redundant summary rows, or use an exact "
            "lower-dimensional summary; no ridge is added implicitly."
        ) from exc
    half = np.linalg.solve(lower, rhs)
    return np.linalg.solve(lower.T, half)


def _synthetic_information(
    covariance: np.ndarray,
    mean_derivative: np.ndarray,
    covariance_derivative: np.ndarray,
    *,
    who: str,
) -> np.ndarray:
    """Gaussian synthetic-likelihood Fisher including covariance derivatives."""
    solved_mean = _strict_spd_solve(covariance, mean_derivative, who=who)
    information = mean_derivative.T @ solved_mean
    solved_covariances = [
        _strict_spd_solve(covariance, derivative, who=who) for derivative in covariance_derivative
    ]
    n_parameters = mean_derivative.shape[1]
    for i in range(n_parameters):
        for j in range(i, n_parameters):
            value = 0.5 * np.trace(solved_covariances[i] @ solved_covariances[j])
            information[i, j] += value
            if i != j:
                information[j, i] += value
    return _symmetrize(information)


def _simulate_event_batch(response: tuple, predictions: dict, draw_keys: Array) -> dict:
    event = {}
    for index, channel in enumerate(response):
        channel_keys = jax.vmap(lambda key: jax.random.fold_in(key, index))(draw_keys)
        predicted = predictions[channel.name]
        likelihood, context = _channel_likelihood_view(channel, predicted)
        event[channel.name] = jax.vmap(
            lambda key: likelihood.sample(
                key,
                predicted,
                context=context,
            )
        )(channel_keys)
    return event


def _apply_summary_batch(summary: Any, event: dict, *, name: str, event_order: tuple) -> np.ndarray:
    if isinstance(summary, AffineSummary):
        flat = jnp.concatenate(
            [event[channel].reshape((event[channel].shape[0], -1)) for channel in event_order],
            axis=1,
        )
        if summary.matrix.shape[1] != flat.shape[1]:
            raise ValueError(
                f"compare_compression AffineSummary {name!r} matrix columns "
                f"must equal the full observation size {flat.shape[1]}; got "
                f"shape {summary.matrix.shape}."
            )
        values = flat @ summary.matrix.T
        if summary.offset is not None:
            values = values + summary.offset
        return np.asarray(values, dtype=float)
    try:
        values = jax.vmap(summary)(event)
    except Exception as exc:
        raise TypeError(
            f"compare_compression summary {name!r} must be a JAX-traceable "
            "callable accepting one dict of response-channel arrays."
        ) from exc
    values = np.asarray(values, dtype=float)
    return values.reshape((values.shape[0], -1))


def _source_parameter_scores(
    response: tuple,
    event: dict,
    predictions: dict,
    jacobians: dict,
    n_parameters: int,
) -> np.ndarray:
    """Return source-model parameter scores for a batch of full events."""
    n_replicates = next(iter(event.values())).shape[0]
    scores = np.zeros((n_replicates, n_parameters), dtype=float)
    for channel in response:
        likelihood, context = _channel_likelihood_view(channel, predictions[channel.name])
        output_score = jax.vmap(
            lambda observed: likelihood.score(
                observed,
                predictions[channel.name],
                context=context,
            )
        )(event[channel.name])
        output_score = np.asarray(output_score, dtype=float).reshape((n_replicates, -1))
        jacobian = np.asarray(jacobians[channel.name], dtype=float).reshape((-1, n_parameters))
        scores += output_score @ jacobian
    return scores


def _score_moment_information(
    samples: np.ndarray,
    source_scores: np.ndarray,
    *,
    who: str,
) -> np.ndarray:
    """Fit Gaussian moment derivatives with the likelihood-score identity."""
    _, covariance = _moments(samples, who=who)
    centered_samples = samples - np.mean(samples, axis=0)
    centered_scores = source_scores - np.mean(source_scores, axis=0)
    denominator = samples.shape[0] - 1
    mean_derivative = centered_samples.T @ centered_scores / denominator
    outer = np.einsum("ri,rj->rij", centered_samples, centered_samples, optimize=True)
    covariance_derivative = (
        np.einsum(
            "rij,rp->pij",
            outer - covariance[None, :, :],
            centered_scores,
            optimize=True,
        )
        / denominator
    )
    return _synthetic_information(
        covariance,
        mean_derivative,
        covariance_derivative,
        who=who,
    )


def _simulation_information(
    response: tuple,
    summaries: dict,
    *,
    key: Array,
    n_simulations: int,
    n_batches: int,
    base_predictions: dict,
    jacobians: dict,
    n_parameters: int,
) -> tuple:
    """Estimate Gaussian synthetic information using source score identities."""
    n_simulations = int(n_simulations)
    n_batches = int(n_batches)
    if n_simulations < 4 or n_batches < 2 or n_simulations % n_batches:
        raise ValueError(
            "compare_compression simulation mode needs n_simulations >= 4, "
            "n_batches >= 2, and n_simulations divisible by n_batches; got "
            f"{n_simulations} and {n_batches}."
        )
    draw_keys = jax.random.split(jnp.asarray(key), n_simulations)
    event = _simulate_event_batch(response, base_predictions, draw_keys)
    source_scores = _source_parameter_scores(
        response,
        event,
        base_predictions,
        jacobians,
        n_parameters,
    )
    samples = {
        name: _apply_summary_batch(
            summary,
            event,
            name=name,
            event_order=tuple(channel.name for channel in response),
        )
        for name, summary in summaries.items()
    }

    batch_size = n_simulations // n_batches
    information = {}
    errors = {}
    for name in summaries:
        summary_size = samples[name].shape[1]
        if batch_size <= summary_size:
            raise ValueError(
                f"compare_compression summary {name!r} has {summary_size} values "
                f"but only {batch_size} simulations per error-estimation batch. "
                "Increase n_simulations or reduce n_batches."
            )

        def estimate(selection: slice, label: str) -> np.ndarray:
            return _score_moment_information(
                samples[name][selection],
                source_scores[selection],
                who=f"summary {name!r} {label}",
            )

        information[name] = estimate(slice(None), "full simulation")
        batch_estimates = np.stack(
            [
                estimate(slice(i * batch_size, (i + 1) * batch_size), f"batch {i}")
                for i in range(n_batches)
            ]
        )
        errors[name] = np.std(batch_estimates, axis=0, ddof=1) / np.sqrt(n_batches)
    return information, errors


def _prior_geometry(prior: Any, at: np.ndarray, n_parameters: int) -> tuple:
    if getattr(prior, "coordinate", "x") != "x":
        raise ValueError(
            "compare_compression: prior-relative metrics require a prior in "
            f"physical coordinate 'x'; got {prior.coordinate!r}."
        )
    size = getattr(prior, "size", None)
    if size is not None and int(size) != n_parameters:
        raise ValueError(
            f"compare_compression: prior covers {int(size)} parameters but the "
            f"experiment predictor exposes {n_parameters}."
        )
    try:
        precision = _symmetrize(prior.precision(at=jnp.asarray(at)))
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            "compare_compression: prior-relative metrics need a prior exposing "
            "a proper SPD precision at the comparison point."
        ) from exc
    if precision.shape != (n_parameters, n_parameters):
        raise ValueError(
            f"compare_compression: prior precision must have shape "
            f"({n_parameters}, {n_parameters}); got {precision.shape}."
        )
    if not np.all(np.isfinite(precision)):
        raise ValueError("compare_compression: prior precision must contain only finite values.")
    try:
        lower = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "compare_compression: prior precision must be strictly positive "
            "definite; singular/ridged priors cannot define relative directions."
        ) from exc
    return precision, lower


def _prior_relative(matrix: np.ndarray, lower: np.ndarray) -> tuple:
    transformed = np.linalg.solve(lower, matrix)
    transformed = np.linalg.solve(lower, transformed.T).T
    values, vectors = np.linalg.eigh(_symmetrize(transformed))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    directions = np.linalg.solve(lower.T, vectors)
    return values, directions


def _posterior_metrics(
    prior_precision: np.ndarray,
    full_information: np.ndarray,
    summary_information: dict,
) -> tuple:
    def covariance_metrics(information: np.ndarray) -> tuple:
        precision = _symmetrize(prior_precision + information)
        lower = np.linalg.cholesky(precision)
        identity = np.eye(precision.shape[0])
        covariance = np.linalg.solve(lower.T, np.linalg.solve(lower, identity))
        log_det_covariance = -2.0 * np.sum(np.log(np.diag(lower)))
        return float(np.trace(covariance)), float(log_det_covariance)

    full_trace, full_log_det = covariance_metrics(full_information)
    trace_change = {}
    log_volume_change = {}
    for name, information in summary_information.items():
        trace, log_det = covariance_metrics(information)
        trace_change[name] = (trace - full_trace) / full_trace
        log_volume_change[name] = 0.5 * (log_det - full_log_det)
    return trace_change, log_volume_change


def _finalize_compression_report(
    *,
    names: tuple,
    at: Any,
    full_information: Any,
    summary_information: dict,
    induced_kind: dict,
    estimate_error: dict,
    prior: Any,
    notes: tuple = (),
) -> CompressionReport:
    """Validate DPI, compute common-direction/prior metrics, and freeze a report."""
    full = _symmetrize(full_information)
    n_parameters = full.shape[0]
    if full.shape != (len(names), len(names)):
        raise ValueError(
            "compare_compression: full information dimension disagrees with "
            f"parameter names ({full.shape} versus {len(names)} names)."
        )
    summary = {name: _symmetrize(value) for name, value in summary_information.items()}
    keys = set(summary)
    if keys != set(induced_kind) or keys != set(estimate_error):
        raise ValueError(
            "compare_compression: summary information, induced-kind, and "
            "estimate-error mappings must have identical names."
        )
    for name, matrix in summary.items():
        if matrix.shape != full.shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"compare_compression summary {name!r} information must be a "
                f"finite matrix of shape {full.shape}; got {matrix.shape}."
            )

    at_host = np.asarray(at, dtype=float)
    report_notes = list(notes)
    flags = []
    dpi = {}
    analytic_roundoff = set()
    for name, matrix in summary.items():
        difference = _symmetrize(full - matrix)
        minimum = float(np.linalg.eigvalsh(difference).min())
        tolerance = _DPI_RTOL * max(
            float(np.linalg.norm(full, 2)), float(np.linalg.norm(matrix, 2))
        )
        exact = induced_kind[name].startswith("analytic")
        if minimum < -tolerance and exact:
            raise ValueError(
                f"compare_compression: exact induced likelihood for summary "
                f"{name!r} violates the data-processing inequality "
                f"(min eig(I_full-I_summary)={minimum:.3e}, tolerance "
                f"{tolerance:.3e}). This indicates an incorrect induced "
                "likelihood or implementation defect."
            )
        if minimum < -tolerance:
            error = estimate_error[name]
            uncertainty = (
                0.0 if error is None else 2.0 * float(np.linalg.norm(np.asarray(error), 2))
            )
            clear = -minimum > uncertainty
            dpi[name] = {
                "status": "violated",
                "min_eigenvalue": minimum,
                "tolerance": tolerance,
                "uncertainty": uncertainty,
                "clear_beyond_estimation_error": clear,
            }
            qualification = (
                "clear beyond the reported estimation error"
                if clear
                else "within the reported estimation error"
            )
            flags.append(
                f"summary {name!r}: estimated data-processing violation "
                f"(min eigenvalue {minimum:.3e}); {qualification}; estimate "
                "is retained without clipping"
            )
        else:
            dpi[name] = {
                "status": "holds",
                "min_eigenvalue": minimum,
                "tolerance": tolerance,
                "uncertainty": None,
                "clear_beyond_estimation_error": None,
            }
            if minimum < 0 and exact:
                analytic_roundoff.add(name)
                report_notes.append(
                    f"summary {name!r}: data-processing difference had a tiny "
                    f"negative eigenvalue ({minimum:.3e}) within roundoff "
                    "tolerance; only derived retention ratios were clipped"
                )

    if prior is None:
        values, directions = np.linalg.eigh(full)
        order = np.argsort(values)[::-1]
        directions = directions[:, order]
        full_prior_relative = None
        relative = {name: None for name in summary}
        trace_change = {name: None for name in summary}
        log_volume_change = {name: None for name in summary}
        report_notes.append(
            "No prior was supplied: prior-relative eigenvalues and posterior "
            "trace/volume changes are undefined and remain None."
        )
    else:
        prior_precision, lower = _prior_geometry(prior, at_host, n_parameters)
        full_prior_relative, directions = _prior_relative(full, lower)
        relative = {name: _prior_relative(matrix, lower)[0] for name, matrix in summary.items()}
        trace_change, log_volume_change = _posterior_metrics(prior_precision, full, summary)

    denominator = np.einsum("ik,ij,jk->k", directions, full, directions)
    direction_tolerance = (
        _PSD_RTOL * max(1, n_parameters) * float(np.max(np.abs(denominator), initial=0.0))
    )
    identified = denominator > direction_tolerance
    retention = {}
    for name, matrix in summary.items():
        numerator = np.einsum("ik,ij,jk->k", directions, matrix, directions)
        values = np.full(denominator.shape, np.nan)
        values[identified] = numerator[identified] / denominator[identified]
        if induced_kind[name].startswith("analytic") and dpi[name]["status"] == "holds":
            values[identified] = np.clip(values[identified], 0.0, 1.0)
        retention[name] = values

    return CompressionReport(
        names=tuple(names),
        full_information=full,
        summary_information=summary,
        full_prior_relative_eigenvalues=full_prior_relative,
        prior_relative_eigenvalues=relative,
        posterior_trace_change=trace_change,
        posterior_log_volume_change=log_volume_change,
        full_directions=directions,
        identified_directions=identified,
        per_direction_retention=retention,
        induced_kind=dict(induced_kind),
        estimate_error={
            name: None if value is None else np.asarray(value, dtype=float)
            for name, value in estimate_error.items()
        },
        dpi=dpi,
        flags=tuple(flags),
        notes=tuple(report_notes),
        evaluated_at={
            "at": tuple(float(value) for value in at_host),
            "theta_names": tuple(names),
            "summaries": tuple(summary),
        },
    )


def compare_compression(
    experiment: Any,
    summaries: dict,
    *,
    at: Any,
    prior: Any = None,
    induced: str = "analytic",
    key: Any = None,
    n_simulations: int = 2048,
    n_batches: int = 8,
    parameters: Any = None,
) -> CompressionReport:
    """Compare deterministic summaries of one experiment's full observation.

    Args:
        experiment: Experiment declaration providing response channels and a
            shared predictor.
        summaries: Non-empty mapping of report names to summary maps. Analytic
            mode requires
            [AffineSummary][diff_soup.inference.statistics.compression.AffineSummary]; simulation mode accepts JAX-
            traceable callables receiving ``{channel_name: observation}``.
        at: Flat physical evaluation point, named overrides, or ``None`` for
            the predictor's declared point.
        prior: Optional proper prior with SPD precision in coordinate ``"x"``.
        induced: ``"analytic"`` or ``"simulate"``.
        key: Required PRNG key for simulation mode.
        n_simulations: Paired source replicates for the synthetic likelihood.
        n_batches: Deterministic batches used for Monte Carlo standard errors.
        parameters: Optional parameter groups passed to the experiment predictor.

    Returns:
        A frozen [CompressionReport][diff_soup.inference.statistics.compression.CompressionReport]
        with full/summary information,
        common-direction retention, optional prior/posterior metrics, DPI
        diagnostics, approximation errors, notes, and flags.
    """
    if induced not in ("analytic", "simulate"):
        raise ValueError(
            f"compare_compression induced must be 'analytic' or 'simulate'; got {induced!r}."
        )
    if not isinstance(summaries, dict) or not summaries:
        raise ValueError("compare_compression needs a non-empty summaries mapping.")
    if not all(isinstance(name, str) and name for name in summaries):
        raise ValueError("compare_compression summary names must be non-empty strings.")

    predictor = experiment.predictor(parameters=parameters)
    x = jnp.asarray(_resolve_at(predictor, at))
    response = tuple(channel for channel in experiment.channels if channel.role == "response")
    if not response:
        raise ValueError("compare_compression: experiment declares no response channel.")
    predictions, jacobians, _, full_information = _expected_information_core(predictor, x, response)
    n_parameters = int(x.shape[0])
    names = tuple(predictor.theta_names)

    if induced == "analytic":
        covariance = _gaussian_source_covariance(response, predictions)
        _, full_jacobian = _flatten_prediction_jacobians(
            response, predictions, jacobians, n_parameters
        )
        n_full = full_jacobian.shape[0]
        summary_information = {}
        for name, summary in summaries.items():
            if not isinstance(summary, AffineSummary):
                raise ValueError(
                    f"compare_compression analytic induced likelihood for "
                    f"summary {name!r} requires an explicit AffineSummary. A "
                    "bare/nonlinear callable cannot certify its induced law; "
                    "use induced='simulate'."
                )
            matrix = np.asarray(summary.matrix, dtype=float)
            if matrix.shape[1] != n_full:
                raise ValueError(
                    f"compare_compression AffineSummary {name!r} matrix columns "
                    f"must equal the full observation size {n_full}; got "
                    f"shape {matrix.shape}."
                )
            induced_covariance = matrix @ covariance @ matrix.T
            induced_jacobian = matrix @ full_jacobian
            inverse = _psd_pseudoinverse(
                induced_covariance, who=f"compare_compression summary {name!r}"
            )
            summary_information[name] = _symmetrize(induced_jacobian.T @ inverse @ induced_jacobian)
        induced_kind = {name: "analytic_affine_gaussian" for name in summaries}
        estimate_error = {name: None for name in summaries}
        notes = (
            "Analytic summaries use the exact affine image of the declared "
            "fixed-covariance Gaussian source likelihood.",
        )
    else:
        if key is None:
            raise ValueError("compare_compression induced='simulate' needs an explicit PRNG key.")
        summary_information, estimate_error = _simulation_information(
            response,
            summaries,
            key=jnp.asarray(key),
            n_simulations=n_simulations,
            n_batches=n_batches,
            base_predictions=predictions,
            jacobians=jacobians,
            n_parameters=n_parameters,
        )
        induced_kind = {name: "simulation_gaussian" for name in summaries}
        notes = (
            "Simulation summaries use a paired Gaussian synthetic likelihood; "
            "reported information includes summary-covariance derivatives and "
            "Monte Carlo standard errors. It is an approximation, not an exact "
            "induced density.",
        )

    return _finalize_compression_report(
        names=names,
        at=x,
        full_information=np.asarray(full_information),
        summary_information=summary_information,
        induced_kind=induced_kind,
        estimate_error=estimate_error,
        prior=prior,
        notes=notes,
    )
