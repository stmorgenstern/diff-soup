"""Optimizer termination tolerances resolved against the decision vector's dtype.

The built-in optimizers stop on a small gradient (``grad_tol``), a small
relative step (``step_tol``) or a small relative cost change (``cost_tol``).
Their fixed defaults suit a double-precision fit. A single-precision decision
vector resolves nothing below about ``1.2e-7`` relative, so a step or cost
change smaller than that is roundoff, and a gradient computed through a
single-precision solve carries noise relative to its own size. A tolerance
left at its default (``None``) is therefore resolved here from the dtype:

* ``step_tol`` and ``cost_tol`` become ``max(default, 10 * eps)``, where
  ``eps`` is the machine epsilon of the decision vector's dtype. In double
  precision that is the fixed default itself.
* ``grad_tol`` stays the absolute ``1e-8`` in double precision. In single (or
  narrower) precision it becomes relative to the gradient norm at the start of
  the fit: ``max(1e-8, 10 * eps)`` times that norm.

A tolerance the caller sets is used exactly as given. A warning says so when it
is below what the dtype can resolve.

The absolute cost-change tolerance ``cost_atol`` has no dtype floor: ``None``
is 0, and a finite value of at least 0 is used as given.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

__all__ = [
    "DEFAULT_COST_ATOL",
    "DEFAULT_COST_TOL",
    "DEFAULT_GRAD_TOL",
    "DEFAULT_STEP_TOL",
    "GradientTolerance",
    "absolute_tolerance",
    "gradient_tolerance",
    "relative_tolerance",
    "step_budget",
]

DEFAULT_GRAD_TOL = 1e-8
DEFAULT_STEP_TOL = 1e-8
DEFAULT_COST_TOL = 1e-8
DEFAULT_COST_ATOL = 0.0


def _resolution(dtype) -> float:
    """Ten machine epsilons of ``dtype``: the smallest relative change it resolves."""
    return 10.0 * float(jnp.finfo(dtype).eps)


def _narrower_than_double(dtype) -> bool:
    return jnp.finfo(dtype).bits < 64


def relative_tolerance(value, default: float, dtype, *, name: str, label: str) -> float:
    """Resolve a relative step or cost tolerance for a decision vector of ``dtype``.

    Args:
        value: The tolerance the caller set, or ``None`` for the default.
        default: The double-precision default.
        dtype: The decision vector's dtype.
        name: The tolerance's field name, for the warning.
        label: The optimizer's name, for the warning.

    Returns:
        ``max(default, 10 * eps)`` when ``value`` is ``None``, else ``value``.
    """
    floor = _resolution(dtype)
    if value is None:
        return max(default, floor)
    value = float(value)
    if 0.0 < value < floor:
        warnings.warn(
            f"{label} {name}={value:g} is below what {np.dtype(dtype).name} resolves "
            f"(10 * eps = {floor:.2g}); it is used as given, so this test can only "
            f"pass on an exactly zero change. Leave {name} unset for a "
            "dtype-aware default, or pass 0 to disable the test.",
            UserWarning,
            stacklevel=3,
        )
    return value


def absolute_tolerance(value, default: float, *, name: str, label: str) -> float:
    """Resolve an absolute tolerance, which has no dtype floor.

    Args:
        value: The tolerance the caller set, or ``None`` for the default.
        default: The default.
        name: The tolerance's field name, for the refusal.
        label: The optimizer's name, for the refusal.

    Returns:
        ``default`` when ``value`` is ``None``, else ``value`` as a float.

    Raises:
        ConfigurationError: If ``value`` is negative or not a finite number.
    """
    if value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not (math.isfinite(number) and number >= 0.0):
        raise ConfigurationError(
            f"{label} {name} must be a finite number >= 0, or None for the default {default}; "
            f"got {value!r}."
        )
    return number


def step_budget(value, *, label: str) -> int:
    """Resolve an optimizer's ``max_steps``: an integer of at least 0.

    ``max_steps=0`` makes no step: the optimizer evaluates its start point
    and returns it with ``n_steps`` 0 and the status ``max_steps_reached``.

    Args:
        value: The ``max_steps`` the caller set.
        label: The optimizer's name, for the refusal.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ConfigurationError: If ``value`` is not an integer of at least 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ConfigurationError(
            f"{label} max_steps must be an integer >= 0 (0 returns the start point); got {value!r}."
        )
    return int(value)


@dataclasses.dataclass(frozen=True)
class GradientTolerance:
    """A gradient tolerance, absolute or relative to the starting gradient norm.

    Attributes:
        value: The tolerance, or the relative factor when ``relative``.
        relative: Whether ``value`` scales the gradient norm at the start of
            the fit.
        given: Whether the caller set the tolerance.
        dtype_name: The decision vector's dtype name, for the warning.
        floor: Ten machine epsilons of that dtype.
        narrow: Whether that dtype is narrower than double precision.
        label: The optimizer's name, for the warning.
    """

    value: float
    relative: bool
    given: bool
    dtype_name: str
    floor: float
    narrow: bool
    label: str

    def threshold(self, initial_norm: float) -> float:
        """The absolute threshold for a fit whose first gradient has norm ``initial_norm``.

        Call once, with the gradient norm the optimizer tests at its first
        iteration.
        """
        if self.relative:
            return self.value * max(float(initial_norm), float(np.finfo(np.float64).tiny))
        if self.given and self.narrow and 0.0 < self.value < self.floor * float(initial_norm):
            warnings.warn(
                f"{self.label} grad_tol={self.value:g} is below what {self.dtype_name} "
                f"resolves for a gradient of norm {float(initial_norm):.3g} "
                f"(10 * eps relative, {self.floor * float(initial_norm):.2g}); it is used "
                "as given. Leave grad_tol unset for a tolerance relative to the "
                "starting gradient.",
                UserWarning,
                stacklevel=3,
            )
        return self.value


def gradient_tolerance(value, dtype, *, label: str) -> GradientTolerance:
    """Resolve ``grad_tol`` for a decision vector of ``dtype``.

    Args:
        value: The tolerance the caller set, or ``None`` for the default.
        dtype: The decision vector's dtype.
        label: The optimizer's name, for the warning.

    Returns:
        The absolute default ``1e-8`` in double precision; in narrower
        precision, ``max(1e-8, 10 * eps)`` relative to the starting gradient
        norm. A value the caller set is absolute and used as given.
    """
    floor = _resolution(dtype)
    narrow = _narrower_than_double(dtype)
    name = np.dtype(dtype).name
    if value is not None:
        return GradientTolerance(float(value), False, True, name, floor, narrow, label)
    if narrow:
        return GradientTolerance(
            max(DEFAULT_GRAD_TOL, floor), True, False, name, floor, True, label
        )
    return GradientTolerance(DEFAULT_GRAD_TOL, False, False, name, floor, False, label)
