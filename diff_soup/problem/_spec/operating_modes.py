"""The rows an operational control table holds, one entry per mode.

A table declared with ``spec.control.operational`` closes one current unknown
with the row of its active mode. Each entry here says, for one mode, what that
row is, how it reads the current, and how the current unknown starts when no
seed is given. The authoring namespace, the lowering of the rows, the start of
the current unknown and the schedule read this one table.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Optional

import jax.numpy as jnp

__all__ = ["OPERATING_MODES", "OperatingMode", "constant_current_slope", "operating_mode_names"]


class OperatingMode(NamedTuple):
    """One row of an operational control table.

    Attributes:
        row: ``(current, voltage, target) -> residual``, zero where the row holds.
        current_slope: The derivative of ``row`` in the current when it is a
            constant (``1`` or ``0``); ``None`` when it depends on the state or
            the target.
        start: ``(voltage, target, threshold) -> (current, usable, refused)``:
            the current unknown's start when no seed is given, whether that
            start is usable, and whether an unusable one is refused. ``None``
            keeps the start the model declares.
        refusal: Why an unusable start is refused, completed with the table's
            name; ``None`` when no start is refused.
    """

    row: Callable
    current_slope: Optional[float]
    start: Optional[Callable] = None
    refusal: Optional[str] = None


def _power_start(voltage, target, threshold):
    """``target / voltage``, usable above the voltage threshold; refused for a nonzero target."""
    usable = jnp.abs(voltage) > threshold
    current = target / jnp.where(usable, voltage, jnp.ones_like(voltage))
    return current, usable, jnp.logical_not(usable) & (target != 0)


def _resistance_start(voltage, target, threshold):
    """``voltage / target``, usable for a nonzero target and refused otherwise."""
    del threshold
    usable = target != 0
    current = voltage / jnp.where(usable, target, jnp.ones_like(target))
    return current, usable, jnp.logical_not(usable)


#: The supported modes, in their default order.
OPERATING_MODES: dict[str, OperatingMode] = {
    "current": OperatingMode(row=lambda i, v, t: i - t, current_slope=1.0),
    "voltage": OperatingMode(row=lambda i, v, t: v - t, current_slope=0.0),
    "power": OperatingMode(
        row=lambda i, v, t: v * i - t,
        current_slope=None,
        start=_power_start,
        refusal=(
            "power mode with a nonzero target starts at a terminal voltage at or below "
            "the seed threshold, so the current unknown has no usable default start"
        ),
    ),
    "resistance": OperatingMode(
        row=lambda i, v, t: v - t * i,
        current_slope=None,
        start=_resistance_start,
        refusal=(
            "resistance mode with a zero target is a short circuit with no usable "
            "default current start"
        ),
    ),
}


def operating_mode_names() -> tuple[str, ...]:
    """The names of the supported modes, in their default order."""
    return tuple(OPERATING_MODES)


def constant_current_slope(modes) -> bool:
    """Whether the row of every mode in ``modes`` has a constant derivative in the current."""
    return all(OPERATING_MODES[mode].current_slope is not None for mode in modes)
