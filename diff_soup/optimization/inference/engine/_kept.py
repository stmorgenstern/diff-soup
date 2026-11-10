"""The Jacobians a study's ``blocks`` evaluations keep on the device for its ``frozen`` requests.

A study that prepared the ``frozen`` mode runs its ``blocks`` requests through
the program that also returns each lane's Jacobian
([KEEPING_BLOCKS][diff_soup.optimization.inference.engine._kernel.KEEPING_BLOCKS]).
The Jacobians of a call never leave the device: the call is kept as it ran,
its Jacobians with the narrow points, first steps and lane inputs of its
units, and each unit (population member, lane) points at the call and slot of
its last ``blocks`` result that solved. A ``frozen`` request runs every kept call that
holds one of its units, in the call's own layout, so each unit is solved on
the device that holds its Jacobian; the call's other slots solve their kept
units again at their kept points and their results are dropped. A kept call no
unit points at any more is released.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import jax
import numpy as np


@dataclasses.dataclass(eq=False)
class KeptCall:
    """One call of a bucket's ``blocks`` program, kept for the ``frozen`` mode.

    Attributes:
        jacobian: The Jacobians of the call's units, ``(W, rows, narrow
            width)``, on the device (split over the mesh like the call).
        members: Per slot, the population member of its unit.
        positions: Per slot, the position of its unit's lane in the bucket.
        narrow: Per slot, the narrow point the unit was solved at.
        first: Per slot, the first step the unit's solve started from.
        inputs: The leaves of the lane inputs of the call, on the device, the
            first steps ``first`` among them.
    """

    jacobian: jax.Array
    members: np.ndarray
    positions: np.ndarray
    narrow: np.ndarray
    first: np.ndarray
    inputs: list


@dataclasses.dataclass(eq=False)
class KeptJacobians:
    """The kept calls of one bucket and the call and slot of every unit's Jacobian.

    Attributes:
        calls: Kept calls by their number, in the order they ran.
        where: ``(member, position)`` to ``(call number, slot)``.
    """

    calls: dict = dataclasses.field(default_factory=dict)
    where: dict = dataclasses.field(default_factory=dict)
    _held: dict = dataclasses.field(default_factory=dict)
    _next: int = 0

    def keep(self, call: KeptCall, slots: Iterable[int]) -> None:
        """Point the units of ``slots`` of ``call`` at it; release calls left without a unit."""
        slots = [int(slot) for slot in slots]
        if not slots:
            return
        number = self._next
        self._next += 1
        self.calls[number] = call
        self._held[number] = 0
        for slot in slots:
            unit = (int(call.members[slot]), int(call.positions[slot]))
            before = self.where.get(unit)
            self.where[unit] = (number, slot)
            self._held[number] += 1
            if before is not None:
                self._held[before[0]] -= 1
                if not self._held[before[0]]:
                    del self._held[before[0]], self.calls[before[0]]

    def nbytes(self) -> int:
        """The bytes of the kept Jacobians, over every device that holds a share."""
        return sum(int(call.jacobian.nbytes) for call in self.calls.values())


def kept_bytes(kept: dict) -> int:
    """The bytes of the Jacobians kept for every bucket of ``kept`` (``BatchedLanes.kept``)."""
    return sum(entry.nbytes() for entry in kept.values())


__all__ = ["KeptCall", "KeptJacobians", "kept_bytes"]
