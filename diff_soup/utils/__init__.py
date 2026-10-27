"""Numerical helpers shared by every layer of diff_soup.

The modules here import JAX, NumPy and the root-private leaves
(``diff_soup._jax_compat``) and nothing else from diff_soup, so any package can
import them without an import cycle:

- ``sparse``: column coloring and forward-mode recovery of sparse Jacobian
  values on a fixed COO pattern.
- ``lanes``: branch selection that stays a branch under ``vmap``: a branch
  runs only when at least one lane takes it.
- ``placement``: the platform an array lives on and batch-tracer detection.
- ``precision``: warnings for a float32 default or float32 work on CPU.
- ``ad_guards``: trace-time recognition of the differentiation applied over a
  solve.
- ``_safe_math``: ``sqrt``, ``div`` and ``power`` with finite derivatives at
  their singular points.
- ``interpolation``: JIT-safe monotone cubic (PCHIP) interpolation.

The package logger is ``diff_soup._logging.log``; traced code uses
``jax.debug.print`` and ``jax.debug.callback`` directly.
"""

from ._safe_math import safe_div, safe_power, safe_sqrt
from .ad_guards import reject_forward_over_reverse
from .interpolation import (
    PchipCoeffs,
    pchip_derivative_coeffs,
    pchip_eval,
    pchip_eval_batch,
)
from .lanes import lane_cond, live_lane_cond
from .placement import array_platform, is_batch_tracer
from .precision import warn_if_float32_default, warn_if_float32_on_cpu
from .sparse import (
    build_padded_color_arrays,
    build_sparse_jvp_groups,
    sparse_jvp_jacobian_values,
    sparse_jvp_jacobian_values_scan,
)

__all__ = [
    # sparse Jacobian recovery
    "build_sparse_jvp_groups",
    "build_padded_color_arrays",
    "sparse_jvp_jacobian_values",
    "sparse_jvp_jacobian_values_scan",
    # batched-lane control flow
    "lane_cond",
    "live_lane_cond",
    # placement
    "array_platform",
    "is_batch_tracer",
    # precision warnings
    "warn_if_float32_default",
    "warn_if_float32_on_cpu",
    # differentiation guards
    "reject_forward_over_reverse",
    # safe math
    "safe_sqrt",
    "safe_div",
    "safe_power",
    # interpolation
    "PchipCoeffs",
    "pchip_eval",
    "pchip_eval_batch",
    "pchip_derivative_coeffs",
]
