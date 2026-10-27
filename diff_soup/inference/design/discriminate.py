"""Robust T-optimal criteria for discriminating two predictive models."""

from __future__ import annotations

from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = ["DiscriminationEnsemble", "t_optimal"]


class DiscriminationEnsemble(eqx.Module):
    """Matched parameter draws and weights for robust model discrimination.

    Draw ``i`` from ``parameters_a`` is compared with draw ``i`` from
    ``parameters_b``.  The two models may have different parameter widths,
    but their ensemble axes must agree.  ``output_scale`` optionally puts
    heterogeneous outputs into common dimensionless units before distances
    are measured.
    """

    parameters_a: Array
    parameters_b: Array
    weights: Array
    output_scale: Optional[Array]

    def __init__(
        self,
        *,
        parameters_a: Any,
        parameters_b: Any,
        weights: Any,
        output_scale: Any = None,
    ):
        """Validate and store matched ensemble draws."""
        parameters_a = jnp.asarray(parameters_a)
        parameters_b = jnp.asarray(parameters_b)
        weights = jnp.asarray(weights)

        if parameters_a.ndim < 1 or parameters_b.ndim < 1:
            raise ValueError(
                "DiscriminationEnsemble: parameter arrays need a leading ensemble sample axis."
            )
        if parameters_a.shape[0] != parameters_b.shape[0]:
            raise ValueError(
                "DiscriminationEnsemble: both parameter ensembles must have "
                "the same number of leading samples; got "
                f"{parameters_a.shape[0]} and {parameters_b.shape[0]}."
            )
        n_samples = parameters_a.shape[0]
        if weights.ndim != 1 or weights.shape[0] != n_samples:
            raise ValueError(
                "DiscriminationEnsemble: weight length must equal the ensemble "
                f"sample count {n_samples}; got shape {weights.shape}."
            )
        host_weights = np.asarray(weights)
        if not np.all(np.isfinite(host_weights)):
            raise ValueError("DiscriminationEnsemble: weights must be finite.")
        if np.any(host_weights < 0.0):
            raise ValueError("DiscriminationEnsemble: weights must be nonnegative.")
        if float(np.sum(host_weights)) <= 0.0:
            raise ValueError("DiscriminationEnsemble: weights must have a positive sum.")

        if output_scale is not None:
            output_scale = jnp.asarray(output_scale)
            host_scale = np.asarray(output_scale)
            if not np.all(np.isfinite(host_scale)) or np.any(host_scale <= 0.0):
                raise ValueError(
                    "DiscriminationEnsemble: output_scale entries must be "
                    "finite and strictly positive."
                )

        self.parameters_a = parameters_a
        self.parameters_b = parameters_b
        self.weights = weights
        self.output_scale = output_scale


def t_optimal(
    model_a: Callable,
    model_b: Callable,
    design: Any,
    *,
    ensemble: DiscriminationEnsemble,
) -> Array:
    """Return the robust T-optimal squared predictive separation.

    The criterion is the normalized weighted mean

    ``sum_s w_s ||(f_b(theta_b[s], d) - f_a(theta_a[s], d)) / scale||²``.

    Larger values indicate designs at which the two competing models make
    predictions that are easier to distinguish.  Model evaluation and the
    reduction use only JAX operations, so the value supports ``jit``,
    ``vmap``, and differentiation with respect to ``design``.
    """
    if not isinstance(ensemble, DiscriminationEnsemble):
        raise TypeError("t_optimal: ensemble must be a DiscriminationEnsemble.")

    prediction_a = jax.vmap(lambda parameters: model_a(parameters, design))(ensemble.parameters_a)
    prediction_b = jax.vmap(lambda parameters: model_b(parameters, design))(ensemble.parameters_b)
    if prediction_a.shape != prediction_b.shape:
        raise ValueError(
            "t_optimal: model outputs must have the same shape for every "
            f"ensemble sample; got {prediction_a.shape} and {prediction_b.shape}."
        )

    difference = prediction_b - prediction_a
    if ensemble.output_scale is not None:
        try:
            broadcast_shape = jnp.broadcast_shapes(
                difference.shape[1:], ensemble.output_scale.shape
            )
        except ValueError as error:
            raise ValueError(
                "t_optimal: output_scale must broadcast to one model output; "
                f"got output shape {difference.shape[1:]} and scale shape "
                f"{ensemble.output_scale.shape}."
            ) from error
        if broadcast_shape != difference.shape[1:]:
            raise ValueError(
                "t_optimal: output_scale must broadcast to one model output "
                f"without expanding it; got output shape {difference.shape[1:]} "
                f"and scale shape {ensemble.output_scale.shape}."
            )
        difference = difference / ensemble.output_scale

    squared_distance = jnp.sum(
        jnp.reshape(difference, (difference.shape[0], -1)) ** 2,
        axis=1,
    )
    weights = ensemble.weights.astype(squared_distance.dtype)
    return jnp.sum(weights * squared_distance) / jnp.sum(weights)
