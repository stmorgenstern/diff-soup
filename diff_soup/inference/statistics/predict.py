# LOC > 500 justification: predictive propagation and held-out comparison share
# the event/likelihood semantics defined here; keeping the audit layer beside
# the existing band and prior-predictive paths prevents a second scoring model.
"""predict.py — pointwise predictive bands and prior-predictive checks.

Carry a Gaussian (Laplace) posterior through to prediction space:

* [band_from_jacobian][] — the closed-form core: given the Jacobian of a
  channel prediction with respect to the posterior's parameters, produce the
  pointwise ±1σ band, either from parameter uncertainty alone (``"latent"``)
  or with the observation noise added on the diagonal (``"observation"``).
* [predict][] — propagate a fit or posterior draws to named measurement bands.
* [prior_predictive][] — the prior-predictive check: draw parameters
  from a proper prior, push every draw through the *same* prediction closure
  a fit would use (no second model path), and add one observation replicate
  per draw from each channel's declared likelihood.
* [compare_predictive][] — score competing fitted models on one audited
  held-out response manifest.  Conditioning inputs remain available, while
  response rows used for fitting are required to be disjoint from the rows
  used for comparison.

All bands are **pointwise**: each frame's interval covers that frame's value
marginally; the band is not a simultaneous confidence region for the whole
trajectory. Every variance here comes from covariance *actions* (solves
against the posterior's Cholesky factor) — no dense covariance and no matrix
inverse is ever formed on these paths.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.inference.statistics.experiment import _channel_likelihood_view
from diff_soup.inference.statistics.laplace import LaplacePosterior
from diff_soup.inference.statistics.likelihoods import CorrelatedGaussian, Gaussian

__all__ = [
    "ComparisonReport",
    "HeldOutEvent",
    "HeldOutPrediction",
    "ModelScore",
    "PredictiveBand",
    "band_from_jacobian",
    "compare_predictive",
    "Prediction",
    "PredictionBand",
    "predict",
    "prior_predictive",
]

#: Channel roles that describe measured predictions and therefore get bands.
#: The remaining roles are conditioning inputs, not model outputs.
BAND_ROLES = ("response", "reference")

BAND_KINDS = ("latent", "observation")


class PredictiveBand(eqx.Module):
    """A pointwise ±1σ prediction band for one channel. Frozen.

    An `equinox.Module` (immutable pytree), so bands pass cleanly
    through ``jax`` transformations; the string fields are static metadata.

    Attributes:
        channel: The channel the band belongs to.
        kind: ``"latent"`` — spread from parameter uncertainty alone — or
            ``"observation"`` — with the channel's observation-noise variance
            added, the spread of future *measurements*.
        mean: Predicted values at the posterior mode, kept frames in channel
            space.
        sd: Pointwise standard deviations, same shape as ``mean``.
        coordinate: The coordinate of the posterior the band came from
            (``"x"`` physical or ``"z"`` inference).
        label: Plain-words interpretation of the band. The bands are
            pointwise (per-frame marginal), not simultaneous over the
            trajectory.
    """

    channel: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    mean: Array
    sd: Array
    coordinate: str = eqx.field(static=True)
    label: str = eqx.field(static=True)


@dataclasses.dataclass(frozen=True)
class HeldOutEvent:
    """One complete held-out likelihood event and its stable row identities.

    A correlated image, histogram, or other grouped observation remains one
    event here.  ``row_ids`` identify its scalar entries in flattened order;
    they are audit metadata and never replace the event-shaped arrays passed
    to the likelihood.

    Attributes:
        experiment_id: Stable ID of the experiment that owns the event.
        channel_name: Declared response-channel name.
        source_id: Stable data-source identifier, when one was declared.
        row_ids: Scalar response identifiers in event-flattening order.
        observed: Complete held-out observation in its declared event shape.
        predicted: Model prediction with exactly the same shape.
        likelihood: Normalized observation model used for NLL scoring.
        context: Optional metadata forwarded unchanged to ``log_prob``.
    """

    experiment_id: str
    channel_name: str
    source_id: str | None
    row_ids: tuple[Any, ...]
    observed: Any
    predicted: Any
    likelihood: Any
    context: Any = None


@dataclasses.dataclass(frozen=True)
class HeldOutPrediction:
    """Predictions from one fitted model on an explicitly held-out split.

    ``training_score`` is retained as provenance only.  Ranking is always by
    the held-out metric, so a better training fit cannot win by itself.
    """

    events: tuple[HeldOutEvent, ...]
    conditioning_manifest: tuple[Any, ...]
    training_score: float


@dataclasses.dataclass(frozen=True)
class ModelScore:
    """One model's scalar held-out score and training-fit provenance."""

    value: float
    n_rows: int
    training_score: float


@dataclasses.dataclass(frozen=True)
class ComparisonReport:
    """Leakage-audited comparison of models on one common held-out cohort.

    Both supported metrics are minimized.  ``ranking`` is therefore ordered
    from the smallest held-out score to the largest, independently of the
    recorded training scores.
    """

    cohort: str
    split_id: str
    metric: str
    scores: dict[str, ModelScore]
    ranking: tuple[str, ...]
    winner: str
    held_out_response_manifest: tuple[Any, ...]
    conditioning_manifest: tuple[Any, ...]


def _record_field(record: Any, name: str) -> Any:
    """Read a manifest field from a dataclass/object or mapping."""
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"held-out response manifest entry is missing {name!r}.")
        return record[name]
    try:
        return getattr(record, name)
    except AttributeError as exc:
        raise ValueError(f"held-out response manifest entry is missing {name!r}.") from exc


def _freeze_manifest_value(value: Any) -> Any:
    """Return an equality-safe, immutable representation of manifest data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        items = (
            (_freeze_manifest_value(key), _freeze_manifest_value(item))
            for key, item in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_manifest_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            array = np.ascontiguousarray(np.asarray(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "held-out manifests must be concrete host values, not traced arrays."
            ) from exc
        return ("array", array.dtype.str, tuple(array.shape), array.tobytes())
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _host_event_array(value: Any, *, model: str, event: int, role: str) -> np.ndarray:
    """Materialize one held-out array with an event-specific error."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"model {model!r} held-out event {event} has a non-concrete {role} array."
        ) from exc
    if array.dtype == np.dtype("O"):
        raise TypeError(
            f"model {model!r} held-out event {event} has an object-valued {role} array."
        )
    return array


def _held_out_contract(split: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Validate the split and return its response and conditioning manifests."""
    try:
        response_manifest = tuple(split.held_out_response_manifest)
        conditioning_manifest = tuple(split.conditioning_manifest)
        train_rows = tuple(split.train_response_row_ids)
    except AttributeError as exc:
        raise TypeError(
            "compare_predictive requires a split with held-out response, "
            "conditioning, and training-row manifests."
        ) from exc
    if not response_manifest:
        raise ValueError("compare_predictive needs at least one held-out response event.")

    held_rows: list[Any] = []
    event_keys: list[tuple[Any, Any, Any]] = []
    for index, group in enumerate(response_manifest):
        key = (
            _record_field(group, "experiment_id"),
            _record_field(group, "channel_name"),
            _record_field(group, "source_id"),
        )
        if key in event_keys:
            raise ValueError(
                f"held-out response manifest contains duplicate likelihood-event identity {key!r}."
            )
        event_keys.append(key)

        row_ids = tuple(_record_field(group, "row_ids"))
        observed_shape = tuple(int(size) for size in _record_field(group, "observed_shape"))
        if any(size < 0 for size in observed_shape):
            raise ValueError(f"held-out response event {index} has invalid shape {observed_shape}.")
        expected_rows = int(np.prod(observed_shape, dtype=int))
        if not row_ids or len(row_ids) != expected_rows:
            raise ValueError(
                "held-out response row manifest must contain exactly one row ID "
                f"per scalar event value; event {index} has shape {observed_shape} "
                f"but {len(row_ids)} row IDs."
            )
        held_rows.extend(_freeze_manifest_value(row_id) for row_id in row_ids)

    if len(set(held_rows)) != len(held_rows):
        raise ValueError("held-out response row manifest contains duplicate row IDs.")
    train_row_set = {_freeze_manifest_value(row_id) for row_id in train_rows}
    overlap = train_row_set.intersection(held_rows)
    if overlap:
        raise ValueError(
            "response leakage: training and held-out response row manifests must "
            f"be disjoint; {len(overlap)} row(s) overlap."
        )
    return response_manifest, conditioning_manifest


def _validate_prediction(
    name: str,
    prediction: HeldOutPrediction,
    response_manifest: tuple[Any, ...],
    conditioning_manifest: tuple[Any, ...],
    reference_observed: list[np.ndarray] | None,
) -> tuple[list[tuple[HeldOutEvent, np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Validate one model against the split and common observed cohort."""
    if not isinstance(prediction, HeldOutPrediction):
        raise TypeError(
            f"model {name!r} must provide a HeldOutPrediction; got {type(prediction).__name__}."
        )
    events = tuple(prediction.events)
    if len(events) != len(response_manifest):
        raise ValueError(
            f"model {name!r} must predict every held-out response event in the "
            f"manifest: expected {len(response_manifest)}, got {len(events)}."
        )
    expected_conditioning = tuple(_freeze_manifest_value(entry) for entry in conditioning_manifest)
    actual_conditioning = tuple(
        _freeze_manifest_value(entry) for entry in prediction.conditioning_manifest
    )
    if actual_conditioning != expected_conditioning:
        raise ValueError(
            f"model {name!r} conditioning manifest must be identical to the "
            "split manifest, including initial_state and other declared inputs."
        )

    validated = []
    observed_arrays: list[np.ndarray] = []
    for index, (event, group) in enumerate(zip(events, response_manifest)):
        if not isinstance(event, HeldOutEvent):
            raise TypeError(
                f"model {name!r} event {index} must be a HeldOutEvent; got {type(event).__name__}."
            )
        expected_key = (
            _record_field(group, "experiment_id"),
            _record_field(group, "channel_name"),
            _record_field(group, "source_id"),
        )
        actual_key = (event.experiment_id, event.channel_name, event.source_id)
        expected_rows = tuple(
            _freeze_manifest_value(row_id) for row_id in _record_field(group, "row_ids")
        )
        actual_rows = tuple(_freeze_manifest_value(row_id) for row_id in event.row_ids)
        if actual_key != expected_key or actual_rows != expected_rows:
            raise ValueError(
                f"model {name!r} held-out row manifest for event {index} must be "
                "identical to the split (same event identity, rows, and order)."
            )

        observed = _host_event_array(event.observed, model=name, event=index, role="observed")
        predicted = _host_event_array(event.predicted, model=name, event=index, role="predicted")
        expected_shape = tuple(int(size) for size in _record_field(group, "observed_shape"))
        if observed.shape != expected_shape:
            raise ValueError(
                f"model {name!r} held-out event {index} observed shape "
                f"{observed.shape} does not match manifest shape {expected_shape}."
            )
        if predicted.shape != observed.shape:
            raise ValueError(
                f"model {name!r} held-out event {index} prediction shape "
                f"{predicted.shape} does not match observed event shape {observed.shape}."
            )
        if reference_observed is not None and not np.array_equal(
            observed, reference_observed[index], equal_nan=True
        ):
            raise ValueError(
                "all models must use identical observed values for every held-out row; "
                f"model {name!r} differs at event {index}."
            )
        observed_arrays.append(observed)
        validated.append((event, observed, predicted))

    try:
        training_score = float(prediction.training_score)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"model {name!r} training_score must be a scalar.") from exc
    if not np.isfinite(training_score):
        raise ValueError(f"model {name!r} training_score must be finite.")
    return validated, observed_arrays


def _event_nll(event: HeldOutEvent, observed: np.ndarray, predicted: np.ndarray) -> float:
    """Evaluate one normalized likelihood on its complete grouped event."""
    likelihood = event.likelihood
    if likelihood is None or not callable(getattr(likelihood, "log_prob", None)):
        raise TypeError("NLL comparison requires every held-out event to have a likelihood.")
    if getattr(likelihood, "normalized", True) is False:
        raise ValueError("NLL comparison requires normalized event likelihoods.")
    value = np.asarray(likelihood.log_prob(observed, predicted, context=event.context))
    if value.shape != ():
        raise ValueError(
            "a held-out likelihood log_prob must return one scalar for the whole event."
        )
    nll = -float(value)
    if not np.isfinite(nll):
        raise ValueError("held-out NLL must be finite for every likelihood event.")
    return nll


def compare_predictive(
    models: Mapping[str, HeldOutPrediction], split: Any, *, metric: str = "nll"
) -> ComparisonReport:
    """Rank fitted models on one common, leakage-audited held-out response set.

    The split is the source of truth for both response identities and allowed
    conditioning data.  Every model must predict exactly those likelihood
    events, with the same scalar rows in the same order and identical held-out
    observations.  Training response rows must be disjoint from them.  This
    permits initial conditions, controls, and boundary traces to condition a
    held-out prediction without allowing its response values into fitting.

    ``metric="nll"`` sums normalized ``-log_prob`` values, calling each
    likelihood once on its complete grouped event.  ``metric="rmse"`` takes
    the root mean squared error over all scalar held-out rows.  Both metrics
    are minimized; ``training_score`` is reported but never used for ranking.

    Args:
        models: Non-empty mapping from stable model name to held-out result.
        split: Split carrying ``cohort``, ``split_id``, held-out response and
            conditioning manifests, and ``train_response_row_ids``.
        metric: ``"nll"`` or ``"rmse"``.

    Returns:
        A [ComparisonReport][] ordered by held-out performance.

    Raises:
        TypeError: If model results or split metadata have the wrong form.
        ValueError: If manifests differ, response leakage is detected, event
            shapes disagree, or the requested score is not well-defined.
    """
    if metric not in ("nll", "rmse"):
        raise ValueError("compare_predictive metric must be 'nll' or 'rmse'.")
    if not isinstance(models, Mapping):
        raise TypeError("compare_predictive models must be a mapping by model name.")
    if not models:
        raise ValueError("compare_predictive needs at least one model; the mapping is empty.")
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError("compare_predictive model names must be non-empty strings.")

    response_manifest, conditioning_manifest = _held_out_contract(split)
    validated_models: dict[str, list[tuple[HeldOutEvent, np.ndarray, np.ndarray]]] = {}
    reference_observed: list[np.ndarray] | None = None
    for name, prediction in models.items():
        validated, observed = _validate_prediction(
            name,
            prediction,
            response_manifest,
            conditioning_manifest,
            reference_observed,
        )
        if reference_observed is None:
            reference_observed = observed
        validated_models[name] = validated

    scores: dict[str, ModelScore] = {}
    for name, events in validated_models.items():
        n_rows = sum(observed.size for _, observed, _ in events)
        if metric == "nll":
            value = sum(
                _event_nll(event, observed, predicted) for event, observed, predicted in events
            )
        else:
            squared_error = sum(
                float(np.sum((observed - predicted) ** 2)) for _, observed, predicted in events
            )
            value = float(np.sqrt(squared_error / n_rows))
            if not np.isfinite(value):
                raise ValueError(f"model {name!r} held-out RMSE must be finite.")
        scores[name] = ModelScore(
            value=float(value),
            n_rows=int(n_rows),
            training_score=float(models[name].training_score),
        )

    ranking = tuple(sorted(scores, key=lambda name: scores[name].value))
    try:
        cohort = split.cohort
        split_id = split.split_id
    except AttributeError as exc:
        raise TypeError("compare_predictive split must declare cohort and split_id.") from exc
    return ComparisonReport(
        cohort=cohort,
        split_id=split_id,
        metric=metric,
        scores=scores,
        ranking=ranking,
        winner=ranking[0],
        held_out_response_manifest=response_manifest,
        conditioning_manifest=conditioning_manifest,
    )


def _observation_noise_variance(likelihood: Any, mean: Array) -> Array:
    """Return the per-output observation-noise variance, broadcast to ``mean``."""
    if isinstance(likelihood, Gaussian):
        return jnp.broadcast_to(jnp.asarray(likelihood.scale) ** 2, mean.shape)
    if isinstance(likelihood, CorrelatedGaussian):
        diag = jnp.sum(jnp.asarray(likelihood.cholesky) ** 2, axis=1)
        return jnp.broadcast_to(diag, mean.shape)
    raise ValueError(
        "band_from_jacobian: an observation band needs a Gaussian observation "
        "noise model (Gaussian or CorrelatedGaussian) to supply the per-output "
        f"noise variance; got {type(likelihood).__name__}."
    )


def band_from_jacobian(
    posterior: LaplacePosterior,
    jacobian: Any,
    *,
    mean: Any,
    likelihood: Any = None,
    channel: str = "",
    kind: Optional[str] = None,
) -> PredictiveBand:
    """Build a pointwise prediction band from a channel Jacobian.

    The latent variance is the diagonal of ``J C Jᵀ`` with ``C`` the
    posterior covariance, evaluated through
    [covariance_action][diff_soup.inference.statistics.laplace.LaplacePosterior.covariance_action]
    on the columns of ``Jᵀ`` — the dense covariance is never formed. An
    observation band adds the likelihood's per-output noise variance on the
    diagonal.

    Args:
        posterior: The Gaussian posterior over the parameters.
        jacobian: Jacobian of the flattened channel prediction with respect
            to the posterior's coordinate at its mode, shape
            ``(n_out, n_params)``.
        mean: The predicted values the band is centered on (this function
            cannot compute them — it only sees the linearization). Any shape
            with ``n_out`` elements; the band's ``sd`` is returned in the
            same shape.
        likelihood: The channel's observation model ([Gaussian][] or
            [CorrelatedGaussian][]); required for an observation band,
            forbidden for a latent one.
        channel: Channel name recorded on the band.
        kind: ``"latent"`` or ``"observation"``. Default ``None`` infers the
            kind from ``likelihood``: absent → latent, present → observation.
            An explicit ``kind`` must agree with the likelihood's presence so
            a band can never be mislabeled.

    Returns:
        The [PredictiveBand][].

    Raises:
        ValueError: If shapes are inconsistent with the posterior, ``kind``
            is unknown or disagrees with ``likelihood``, or an observation
            band's likelihood has no Gaussian noise diagonal.
    """
    if kind is None:
        kind = "latent" if likelihood is None else "observation"
    if kind not in BAND_KINDS:
        raise ValueError(f"band_from_jacobian: kind must be one of {BAND_KINDS}; got {kind!r}.")
    if kind == "observation" and likelihood is None:
        raise ValueError(
            "band_from_jacobian: an observation band adds observation noise, so "
            "it needs the channel's likelihood; pass likelihood=... or build a "
            "latent band."
        )
    if kind == "latent" and likelihood is not None:
        raise ValueError(
            "band_from_jacobian: a latent band excludes observation noise; drop "
            "the likelihood or ask for kind='observation'."
        )

    j = jnp.asarray(jacobian)
    n_params = int(posterior.mode.shape[0])
    if j.ndim != 2 or j.shape[1] != n_params:
        raise ValueError(
            f"band_from_jacobian: jacobian must have shape (n_out, {n_params}) "
            f"matching the posterior's parameters; got {j.shape}."
        )
    mean = jnp.asarray(mean)
    if mean.size != j.shape[0]:
        raise ValueError(
            f"band_from_jacobian: mean has {mean.size} elements but the jacobian "
            f"predicts {j.shape[0]} outputs."
        )

    # diag(J C Jᵀ) via the covariance action on the columns of Jᵀ.
    cjt = posterior.covariance_action(j.T)
    variance = jnp.clip(jnp.sum(j.T * cjt, axis=0), 0.0, None)

    if kind == "observation":
        variance = variance + _observation_noise_variance(likelihood, mean).reshape(-1)
        spread = "parameter uncertainty plus observation noise"
    else:
        spread = "parameter uncertainty alone (no observation noise)"
    label = (
        f"pointwise (per-frame, not simultaneous) +/-1 sigma band of the "
        f"predicted {channel or 'channel'!r} from {spread}; posterior in "
        f"coordinate {posterior.coordinate!r} with {posterior.curvature_kind} "
        "curvature, linearized at its mode"
    )
    return PredictiveBand(
        channel=channel,
        kind=kind,
        mean=mean,
        sd=jnp.sqrt(variance).reshape(mean.shape),
        coordinate=posterior.coordinate,
        label=label,
    )


def _band_channels(experiment: Any) -> tuple:
    """Return the channels that describe predictions (response and reference)."""
    return tuple(c for c in experiment.channels if c.role in BAND_ROLES)


def _check_flat_coordinate(coordinate: str, *, who: str) -> None:
    """Reject posteriors/priors not expressed in the predictor's flat coordinate."""
    if coordinate != "x":
        raise ValueError(
            f"{who}: the experiment predictor takes the flat physical parameter "
            f"vector (coordinate 'x'), but the given coordinate is "
            f"{coordinate!r}. Express the distribution in 'x' first."
        )


def _linearized_predictions(posterior: LaplacePosterior, experiment: Any, parameters: Any):
    """Predict every band channel at the mode and linearize there.

    One shared solve evaluates the means; one ``jax.jacfwd`` pass through the
    same closure produces every channel's Jacobian (the prediction closure
    supports forward-mode differentiation only — do not reverse-differentiate
    through the solve).
    """
    _check_flat_coordinate(posterior.coordinate, who="predictive band")
    predictor = experiment.predictor(parameters=parameters)
    n_params = len(predictor.theta_names)
    if int(posterior.mode.shape[0]) != n_params:
        raise ValueError(
            f"predictive band: the posterior covers {int(posterior.mode.shape[0])} "
            f"parameters but the experiment predictor expects {n_params} "
            f"(order {predictor.theta_names})."
        )
    channels = _band_channels(experiment)
    names = tuple(c.name for c in channels)
    mode = jnp.asarray(posterior.mode)

    means, failed = predictor(mode)
    if bool(failed):
        raise RuntimeError(
            "predictive band: the forward solve failed at the posterior mode, so "
            "no prediction exists there. Loosen the solver tolerances or check "
            "that the mode is a physically valid parameter vector."
        )

    def flat_predictions(x_flat: Array) -> dict:
        preds, _ = predictor(x_flat)
        return {name: jnp.reshape(preds[name], (-1,)) for name in names}

    jacobians = jax.jacfwd(flat_predictions)(mode)
    return channels, means, jacobians


def _latent_band(posterior: LaplacePosterior, experiment: Any, *, parameters: Any = None) -> dict:
    """Return the parameter-uncertainty band of every predicted channel.

    Linearizes each response/reference channel of the experiment at the
    posterior mode and propagates the posterior covariance; observation noise
    is excluded; [predict][diff_soup.inference.predict] with ``kind="predictive"``
    adds the measurement spread.

    Args:
        posterior: A Gaussian posterior in the flat physical coordinate
            (``coordinate == "x"``) over the experiment's learnable
            parameters.
        experiment: The [Experiment][diff_soup.inference.statistics.measurement.Experiment].
        parameters: Optional parameter selection forwarded to
            ``experiment.predictor``; must match what the posterior covers.

    Returns:
        Dict mapping channel name to its ``"latent"``
        [PredictiveBand][]. Conditioning channels are absent.

    Raises:
        ValueError: If the posterior is not in coordinate ``"x"`` or its
            dimension disagrees with the predictor.
        RuntimeError: If the forward solve fails at the mode.
    """
    channels, means, jacobians = _linearized_predictions(posterior, experiment, parameters)
    return {
        c.name: band_from_jacobian(
            posterior,
            jacobians[c.name],
            mean=means[c.name],
            channel=c.name,
            kind="latent",
        )
        for c in channels
    }


def prior_predictive(
    prior: Any, experiment: Any, *, key: Array, n: int, parameters: Any = None
) -> dict:
    """Simulate prior-predictive observations through the experiment's model.

    Draws ``n`` parameter vectors from the prior, pushes each through the
    experiment's one shared prediction closure — the same declaration a fit
    uses, never a second model path — and adds one observation replicate per
    draw from each channel's declared likelihood. Compare the result against
    real data before fitting: data far outside the prior-predictive spread
    means the prior and the model disagree about the experiment.

    Draws run in a host loop, one solve per draw: each call reuses the one
    compiled predictor, whereas batching draws through ``vmap`` would couple
    the adaptive step-size control across draws and compile a separate
    batched stepper for no accuracy gain on a diagnostic check.

    Args:
        prior: A proper prior with ``coordinate == "x"`` whose ``sample``
            returns flat physical parameter vectors matching the predictor
            (e.g. [GaussianPrior][diff_soup.inference.statistics.priors.GaussianPrior]).
        experiment: The [Experiment][diff_soup.inference.statistics.measurement.Experiment].
        key: A ``jax.random`` PRNG key; the output is a deterministic
            function of it.
        n: Number of prior draws (and observation replicates).
        parameters: Optional parameter selection forwarded to
            ``experiment.predictor``; must match what the prior covers.

    Returns:
        Dict mapping each response/reference channel name to a
        ``(n, *channel_shape)`` array of simulated observations.

    Raises:
        ValueError: If the prior's coordinate is not ``"x"`` or its size
            disagrees with the predictor.
        RuntimeError: If the forward solve fails for a draw.
    """
    _check_flat_coordinate(prior.coordinate, who="prior_predictive")
    predictor = experiment.predictor(parameters=parameters)
    n_params = len(predictor.theta_names)
    if int(prior.size) != n_params:
        raise ValueError(
            f"prior_predictive: the prior covers {int(prior.size)} parameters but "
            f"the experiment predictor expects {n_params} "
            f"(order {predictor.theta_names})."
        )
    channels = _band_channels(experiment)

    key_draws, key_obs = jax.random.split(jnp.asarray(key))
    draws = prior.sample(key_draws, (int(n),))
    simulated = {c.name: [] for c in channels}
    for i in range(int(n)):
        preds, failed = predictor(jnp.asarray(draws[i]))
        if bool(failed):
            raise RuntimeError(
                f"prior_predictive: the forward solve failed for prior draw {i}. "
                "Tighten the prior around physically solvable parameters or "
                "loosen the solver tolerances."
            )
        for k, c in enumerate(channels):
            replicate_key = jax.random.fold_in(key_obs, i * len(channels) + k)
            likelihood, context = _channel_likelihood_view(c, preds[c.name])
            simulated[c.name].append(
                likelihood.sample(
                    replicate_key,
                    preds[c.name],
                    context=context,
                )
            )
    return {name: jnp.stack(values) for name, values in simulated.items()}


@dataclasses.dataclass(frozen=True)
class PredictionBand:
    """The prediction band of one measurement over its observation times.

    Attributes:
        times: The observation times, without the masked-out ones.
        mean: The predicted mean at each time.
        sd: The predicted standard deviation at each time.
        lower: The lower end, `mean - z * sd`, with `z` the normal quantile for the level.
        upper: The upper end, `mean + z * sd`.
    """

    times: Any
    mean: Any
    sd: Any
    lower: Any
    upper: Any


@dataclasses.dataclass(frozen=True)
class Prediction(Mapping):
    """The bands of `predict`, a mapping from measurement name to `PredictionBand`.

    `prediction["u"]` is the band of the measurement `"u"`, and `prediction.names` lists the
    measurement names.

    Attributes:
        bands: The bands, by measurement name.
    """

    bands: Mapping

    def __getitem__(self, name):
        """Return the named measurement band."""
        return self.bands[name]

    def __iter__(self):
        """Iterate over measurement names in declaration order."""
        return iter(self.bands)

    def __len__(self):
        """Return the number of measurement bands."""
        return len(self.bands)

    @property
    def names(self):
        """Measurement names in declaration order."""
        return tuple(self.bands)


def _conditional_moments(likelihood, prediction):
    from diff_soup.inference.statistics.likelihoods import (
        LogNormal,
        MultiplicativeGaussian,
        Poisson,
        StudentT,
    )

    if isinstance(likelihood, StudentT):
        if np.any(np.asarray(likelihood.df) <= 2):
            raise ValueError("predictive variance for StudentT requires df > 2.")
        return prediction, likelihood.scale**2 * likelihood.df / (likelihood.df - 2)
    if isinstance(likelihood, Poisson):
        mean = prediction * likelihood.exposure
        return mean, mean
    if isinstance(likelihood, LogNormal):
        return prediction, prediction**2 * jnp.expm1(likelihood.scale**2)
    if isinstance(likelihood, MultiplicativeGaussian):
        return prediction, (likelihood.scale * prediction) ** 2
    return prediction, _observation_noise_variance(likelihood, prediction)


def predict(problem, fit_or_draws, *, experiment_id=None, kind="latent", level=0.95):
    """Carry the parameter uncertainty of a fit or of posterior draws through to the predicted measurements.

    From a fit, each prediction is linearized at the fit and the Laplace covariance is
    propagated. From [`Draws`][diff_soup.inference.Draws], the band is the mean and spread of the
    predictions over the draws.

    Args:
        problem: The [`InferenceProblem`][diff_soup.inference.InferenceProblem] of the fit or
            the draws.
        fit_or_draws: A fit as `laplace` takes it, or the draws of `ds.inference.sample` on
            `problem`.
        experiment_id: The experiment to predict, required when the problem has several.
        kind: `"latent"` for the spread from the parameters alone, or `"predictive"` with the
            measurement noise added.
        level: The coverage of the band, strictly between 0 and 1.

    Returns:
        A [`Prediction`][diff_soup.inference.Prediction] mapping each measurement name to its
            [`PredictionBand`][diff_soup.inference.PredictionBand].
    """
    from statistics import NormalDist

    from diff_soup.inference.fitting._layout import require_layout
    from diff_soup.inference.statistics._analysis import experiment_view
    from diff_soup.inference.statistics.laplace import laplace
    from diff_soup.inference.statistics.sample import Draws

    if kind not in ("latent", "predictive"):
        raise ValueError('predict kind must be "latent" or "predictive".')
    if not 0 < level < 1:
        raise ValueError("predict level must lie between zero and one.")
    experiment, parameters, columns = experiment_view(problem, experiment_id)
    predictor = experiment.predictor(parameters=parameters)
    z = NormalDist().inv_cdf((1 + level) / 2)
    if isinstance(fit_or_draws, Draws):
        schema = fit_or_draws._parameter_schema
        if schema is None:
            raise ValueError(
                "predict draws carry no parameter schema; draw them with sample(problem, fit)."
            )
        require_layout(problem, schema, where="predict draws")
        values = fit_or_draws.values.reshape(-1, problem.n_params)
        predictions, failed = jax.jit(jax.vmap(lambda x: predictor(x[columns])))(values)
        if bool(jnp.any(failed)):
            raise RuntimeError("predict encountered a failed forward solve.")
        covariance = None
    else:
        post = laplace(problem, fit_or_draws)
        predictions, failed = predictor(post.mode[columns])
        if bool(failed):
            raise RuntimeError("predict encountered a failed forward solve.")
        jacobians = jax.jacfwd(lambda x: predictor(x[columns])[0])(post.mode)
        covariance = post.covariance_action
    bands = {}
    for measurement, channel in zip(experiment.measurements, experiment.channels):
        values = predictions[measurement.name]
        noise = 0.0
        likelihood, _ = _channel_likelihood_view(channel, values if covariance else values[0])
        if covariance is None:
            if kind == "predictive":
                values, variances = _conditional_moments(likelihood, values)
                noise = jnp.mean(jnp.broadcast_to(variances, values.shape), axis=0)
            mean = jnp.mean(values, axis=0)
            variance = jnp.var(values, axis=0) + noise
        else:
            mean = values
            # The posterior of a multiple-shooting fit covers the parameters alone.
            J = jacobians[measurement.name].reshape(-1, post.dim)
            variance = jnp.sum(J * covariance(J.T).T, axis=1).reshape(mean.shape)
            if kind == "predictive":
                mean, noise = _conditional_moments(likelihood, mean)
                from diff_soup.inference.statistics.likelihoods import Poisson

                if isinstance(likelihood, Poisson):
                    variance = variance * likelihood.exposure**2
                variance = variance + noise
        sd = jnp.sqrt(jnp.maximum(variance, 0.0))
        times = (
            measurement.times if measurement.mask is None else measurement.times[measurement.mask]
        )
        bands[measurement.name] = PredictionBand(times, mean, sd, mean - z * sd, mean + z * sd)
    return Prediction(bands)
