"""observation/base.py — Abstract observation operator protocol.

An observation operator maps FEM solution fields to observable quantities
for comparison with experimental data.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax


class ObservationOperator(eqx.Module):
    """Abstract base class for observation operators.

    Subclass this and implement [predict][].  The operator stores the
    static mapping geometry (element locations, quadrature weights, pixel
    masks) as frozen fields; the FEM state flows through as a JAX array.
    """

    @abstractmethod
    def predict(self, u: jax.Array, t: Any, theta: Any) -> jax.Array:
        """Map a nodal FEM field to observed quantities.

        Args:
            u: FEM nodal values, shape ``(n_nodes,)`` or ``(n_nodes, n_components)``.
            t: The observation time, or ``None`` when the caller has none to
                give; an operator that does not depend on time ignores it.
            theta: The current parameter pytree, or ``None`` when the caller
                has none to give; an operator that does not depend on
                parameters ignores it.

        Returns:
            observation:
                Predicted observation, shape depends on the operator
                (e.g. ``(n_pixels,)`` for image observations).
        """
        raise NotImplementedError

    def residual(
        self,
        nodal_field: jax.Array,
        obs_data: jax.Array,
        weight: float = 1.0,
        *,
        t: Any = None,
        theta: Any = None,
    ) -> jax.Array:
        """Compute weighted residual ``(predicted - observed) * weight``.

        Args:
            nodal_field: FEM nodal values.
            obs_data: Observed data, same shape as [predict][] output.
            weight: Scalar weight (e.g. ``1/sigma_epsilon``).
            t: Optional observation time, forwarded to [predict][].
            theta: Optional parameter pytree, forwarded to [predict][].

        Returns:
            residual:
                ``weight * (predict(nodal_field, t, theta) - obs_data)``.
        """
        return weight * (self.predict(nodal_field, t, theta) - obs_data)
