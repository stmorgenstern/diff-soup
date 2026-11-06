"""Units dealt to the devices of a placement by their last trial-step counts, longest first.

A compiled program of a study placed over the devices of a mesh's data axis
runs each call split over those devices: the units at positions ``[d w, (d +
1) w)`` of a call of width ``W = n w`` run on device ``d``. The lanes on one
device advance in lockstep, so a device lasts as long as its slowest unit.
[deal][diff_soup.inference.engine._balance.deal] hands the units of a call to
the devices longest first, each to the device with the smallest load so far,
so the devices of a call finish together; it permutes units within a call and
never changes a program, so nothing compiles again.

[StepBalancer][diff_soup.inference.engine._balance.StepBalancer] uses the same
dealing to plan the experiment rows of a joint problem sharded over a mesh,
with fixed slot shapes, and accepts a new assignment only when its predicted
saving exceeds the measured time of moving the rows.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from diff_soup.inference.engine._order import UNKNOWN_STEPS


def deal(steps: Sequence[int], capacities: Sequence[int], *, lockstep: bool = True) -> tuple:
    """Deal items to bins longest first, each to the bin with the smallest load that has room.

    Args:
        steps: Per item, its trial-step count.
        capacities: Per bin, the number of items it takes; together at least
            the number of items.
        lockstep: ``True`` when a bin costs its largest step count (its items
            advance together); ``False`` when its items' step counts add up.

    Returns:
        Per bin, the indices of its items in the order they were dealt; ties
        go to the bin with fewer items, then the lower index, and items of
        equal step count are dealt by index.

    Raises:
        ValueError: If the capacities hold fewer items than ``steps``.
    """
    steps = [int(value) for value in steps]
    capacities = [int(value) for value in capacities]
    if sum(capacities) < len(steps):
        raise ValueError(
            f"{len(steps)} items are dealt to bins that take {sum(capacities)}; give capacities "
            "that hold every item."
        )
    bins = [[] for _ in capacities]
    loads = [0] * len(capacities)
    for index in sorted(range(len(steps)), key=lambda i: (-steps[i], i)):
        target = min(
            (b for b in range(len(bins)) if len(bins[b]) < capacities[b]),
            key=lambda b: (loads[b], len(bins[b]), b),
        )
        bins[target].append(index)
        loads[target] = (
            max(loads[target], steps[index]) if lockstep else loads[target] + steps[index]
        )
    return tuple(tuple(items) for items in bins)


def device_order(steps: np.ndarray, width: int, devices: int) -> np.ndarray:
    """The order of the units of a bucket's calls that deals each call's units to the devices.

    Args:
        steps: Per unit, in the order the units fill the calls, its trial
            steps when it was last evaluated, or ``UNKNOWN_STEPS``.
        width: ``W``, the units of one call, a multiple of ``devices``.
        devices: The devices a call is split over.

    Returns:
        A permutation of the units that keeps every unit in its call. Within a
        call whose step counts are all known, the units are dealt with
        [deal][diff_soup.inference.engine._balance.deal] to the call's
        devices (device ``d`` takes the positions ``[d w, (d + 1) w)``,
        ``w = W / devices``; the last call's padding fills its last
        positions); a call with a unit of unknown step count keeps its order.
    """
    steps = np.asarray(steps, dtype=np.int64)
    count = int(steps.shape[0])
    order = np.arange(count, dtype=np.int64)
    if devices < 2:
        return order
    share = width // devices
    for start in range(0, count, width):
        own = min(width, count - start)
        block = steps[start : start + own]
        if np.any(block == UNKNOWN_STEPS):
            continue
        capacities = [max(0, min(share, own - device * share)) for device in range(devices)]
        dealt = deal(block.tolist(), capacities)
        order[start : start + own] = start + np.concatenate(
            [np.asarray(items, dtype=np.int64) for items in dealt]
        )
    return order


@dataclasses.dataclass(frozen=True)
class StepAssignment:
    """One experiment order and step-class partition with fixed slot shapes."""

    lanes: tuple[tuple[int, ...], ...]
    buckets: tuple[tuple[tuple[int, ...], ...], ...]
    makespan_seconds: float
    changed: bool = False

    def row_order(self) -> np.ndarray:
        """Experiment index for every fixed slot, lane-major; ``-1`` marks an empty slot.

        Consumers that run one batch per lane permute their experiment rows by
        this order before sharding the leading axis over the lanes.
        """
        return np.asarray(
            [slot for lane in self.buckets for bucket in lane for slot in bucket], dtype=np.int32
        )

    def place(self, inputs, mesh):
        """Move experiment data onto this assignment using JAX device transfers.

        Returns a fixed ``(lane, step_class, slot, ...)`` pytree and its active
        mask. Consumers evaluate one vmap per step class, then use the same mask
        when reducing statistics. No experiment data is read by the host.
        """
        if mesh.shape.get("data") != len(self.lanes):
            raise ValueError("The assignment and mesh need the same number of experiment lanes.")
        n_experiments = sum(len(lane) for lane in self.lanes)
        if any(
            not value.ndim or value.shape[0] != n_experiments for value in jax.tree.leaves(inputs)
        ):
            raise ValueError("Every input must have one leading row per experiment.")
        slots = np.asarray(self.buckets, dtype=np.int32)
        active = slots >= 0
        indices = jnp.asarray(np.maximum(slots, 0))
        sharding = NamedSharding(mesh, PartitionSpec("data"))
        placed = jax.tree.map(
            lambda value: jax.device_put(jnp.take(value, indices, axis=0), sharding), inputs
        )
        return placed, jax.device_put(active, sharding)


class StepBalancer:
    """Greedy longest-first assignment gated by measured transfer cost.

    ``batch_step_seconds[k - 1]`` is the measured time for one step of a batch of
    ``k`` experiments. The integrator's previous trial-step counts determine
    the next plan, dealt by [deal][diff_soup.inference.engine._balance.deal].
    Reassignment occurs only when the predicted makespan saving exceeds the
    measured data-transfer time. Slot capacity and class count are fixed for
    the lifetime of a fit.

    ``step_classes`` fixes the number of step-class buckets per lane. The
    default partitions steps by powers of two, one batch per class; ``1`` runs
    every experiment of a lane in one batch, whose cost is the lane's largest
    step count times the full-width batch-step time.

    A sharded joint leaves this off by default: the integrator runs a lane's
    experiments as one vmapped step loop, and a vmapped loop charges every
    lane until the slowest experiment finishes, so a fixed-width batch costs
    its largest step count and reassigning experiments between lanes cannot
    shorten it. The per-class model above assumes a batch's cost follows its
    occupancy, which is not the case for that loop; the step counts are
    measured either way.
    """

    def __init__(
        self,
        n_experiments: int,
        n_lanes: int,
        *,
        max_steps: int,
        batch_step_seconds: Sequence[float],
        step_classes: int | None = None,
    ):
        """Fix the experiments, lanes, step bound, batch-step times and step classes."""
        if min(n_experiments, n_lanes, max_steps) < 1:
            raise ValueError("Experiment count, lane count, and step bound must be positive.")
        self.n_experiments = int(n_experiments)
        self.n_lanes = int(n_lanes)
        self.max_steps = int(max_steps)
        self.capacity = math.ceil(n_experiments / n_lanes)
        if step_classes is None:
            self.n_classes = (max_steps - 1).bit_length() + 1
        elif int(step_classes) == 1:
            self.n_classes = 1
        else:
            raise ValueError("step_classes accepts None (power-of-two classes) or 1 (one batch).")
        self.batch_step_seconds = tuple(float(value) for value in batch_step_seconds)
        if len(self.batch_step_seconds) != self.capacity or any(
            not math.isfinite(value) or value <= 0 for value in self.batch_step_seconds
        ):
            raise ValueError("Provide a positive measured batch-step time for each batch width.")
        self._lanes = tuple(
            tuple(range(lane * self.capacity, min((lane + 1) * self.capacity, n_experiments)))
            for lane in range(n_lanes)
        )

    def _plan(self, lanes, steps, *, changed=False):
        buckets, loads = [], []
        for lane in lanes:
            classes = [[] for _ in range(self.n_classes)]
            for index in lane:
                cls = (int(steps[index]) - 1).bit_length() if self.n_classes > 1 else 0
                classes[cls].append(index)
            seconds = sum(
                max(steps[index] for index in bucket)
                * self.batch_step_seconds[
                    (len(bucket) if self.n_classes > 1 else self.capacity) - 1
                ]
                for bucket in classes
                if bucket
            )
            loads.append(seconds)
            buckets.append(
                tuple(tuple(bucket) + (-1,) * (self.capacity - len(bucket)) for bucket in classes)
            )
        return StepAssignment(
            tuple(tuple(lane) for lane in lanes), tuple(buckets), max(loads), changed
        )

    def update(self, trial_steps: Sequence[int], *, transfer_seconds: float) -> StepAssignment:
        """Use observed steps and measured migration time to accept or retain a plan."""
        steps = tuple(trial_steps)
        if len(steps) != self.n_experiments or any(
            not isinstance(value, (int, np.integer)) or value < 1 or value > self.max_steps
            for value in steps
        ):
            raise ValueError("Trial-step counts must cover every experiment within max_steps.")
        if not math.isfinite(transfer_seconds) or transfer_seconds < 0:
            raise ValueError("Measured transfer time must be finite and nonnegative.")
        current = self._plan(self._lanes, steps)
        # One batch per lane costs its largest step count; step classes add up.
        lanes = deal(steps, (self.capacity,) * self.n_lanes, lockstep=self.n_classes == 1)
        candidate = self._plan(lanes, steps, changed=True)
        if current.makespan_seconds - candidate.makespan_seconds > transfer_seconds:
            self._lanes = candidate.lanes
            return candidate
        return current


__all__ = ["StepAssignment", "StepBalancer", "deal", "device_order"]
