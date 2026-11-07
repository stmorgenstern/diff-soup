"""The four phased primitives of a linear backend, and the constants their rules share."""

from __future__ import annotations

from typing import Any

import numpy as np

from diff_soup._jax_compat import (
    Primitive,
)

#: The dtype of a factor's order word.
ORDER_DTYPE = np.dtype(np.int32)

_ORIENTATIONS: tuple[str, ...] = ("A", "AT")
_VMAP_MODES: tuple[str, ...] = ("native", "sequential", "unsupported")
_HOOKS: dict[str, str] = {
    "factor": "lower_factor",
    "apply": "lower_apply",
    "refactor": "lower_refactor",
    "invalidate": "lower_invalidate",
}
_FUNS: dict[str, str] = {
    "factor": "factor_fun",
    "apply": "apply_fun",
    "refactor": "refactor_fun",
    "invalidate": "invalidation",
}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _backend_label(analysis: Any) -> str:
    """The name a backend is reported under in error messages."""
    name = getattr(analysis, "backend_name", None)
    if name:
        return str(name)
    return type(getattr(analysis, "backend", analysis)).__name__


phased_factor_p = Primitive("phased_factor")
"""Factor the assembled operator: ``(values, slot_token, context_id, order) -> (*handle, status, order)``."""

phased_solve_p = Primitive("phased_solve")
"""Solve with a factor: ``(values, *handle, order, b[, weights][, x0]) -> (x, status[, iterations, applications], order)``, linear in ``b``."""

phased_refactor_p = Primitive("phased_refactor")
"""Refactor where a predicate holds: ``(values, held_values, predicate, *handle, order) -> (kept_values, *handle, status, order)``."""

phased_invalidate_p = Primitive("phased_invalidate")
"""Mark the native storage a handle names stale: ``(*handle, context_id, order) -> (order,)``."""

for _primitive in (phased_factor_p, phased_solve_p, phased_refactor_p, phased_invalidate_p):
    _primitive.multiple_results = True
