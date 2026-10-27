"""Compact owner-local decision-vector plans.

An ownership-aware joint vector is laid out globally as shared coordinates,
owner coordinates, child-local coordinates, and child trajectory states.  A
heavy owner residual does not depend on coordinates owned by any other owner.
This module records that fact as immutable gather/scatter plans so compiled
owner kernels receive only the coordinates they can observe.

The plans are intentionally generic: owner identifiers are opaque and no
application-specific geometry or data concepts appear here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

__all__ = [
    "OwnerChildPointPlan",
    "OwnerPointPlan",
    "build_owner_point_plans",
    "gather_owner_points",
    "scatter_owner_points",
]


@dataclasses.dataclass(frozen=True)
class OwnerChildPointPlan:
    """Map one compact owner point to one child's flat decision vector.

    ``source_positions`` and ``target_positions`` are aligned: values gathered
    from the compact owner point at the former are scattered into the child
    point at the latter. ``local_positions`` identifies the non-shared source
    coordinates inside the owning plan and is used to assemble owner-local
    sufficient statistics.
    """

    child_index: int
    child_width: int
    source_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    local_positions: tuple[int, ...]
    owner_slice: tuple[int, int]
    child_local_slice: tuple[int, int]
    child_state_slice: tuple[int, int]

    @property
    def derivative_width(self) -> int:
        """Number of structurally active coordinates for this child."""
        return len(self.source_positions)

    def child_point(self, owner_point: Any) -> Array:
        """Reconstruct the child's flat point from a compact owner point."""
        owner_point = jnp.asarray(owner_point)
        values = owner_point[jnp.asarray(self.source_positions, dtype=jnp.int32)]
        out = jnp.zeros((self.child_width,), dtype=owner_point.dtype)
        return out.at[jnp.asarray(self.target_positions, dtype=jnp.int32)].set(values)

    def child_point_from_active(self, active_point: Any) -> Array:
        """Reconstruct the child directly from its active coordinate vector."""
        active_point = jnp.asarray(active_point)
        if active_point.shape != (self.derivative_width,):
            raise ValueError(
                f"active child point has shape {active_point.shape}; expected "
                f"{(self.derivative_width,)}."
            )
        out = jnp.zeros((self.child_width,), dtype=active_point.dtype)
        return out.at[jnp.asarray(self.target_positions, dtype=jnp.int32)].set(active_point)


@dataclasses.dataclass(frozen=True)
class OwnerPointPlan:
    """Static gather/scatter plan for one independent owner.

    Compact coordinates are ordered ``[shared | owner-local]``.  The local
    tail contains the owner's tied coordinate block (with groups tied at
    nested levels, the owner's own block followed by the blocks of the members
    below it, level by level) followed by every member child's local and state
    blocks in joint-vector order.  Consequently the compact input shape is
    independent of coordinates belonging to sibling owners and of the total
    cohort width.
    """

    owner_id: Any
    owner_index: int
    shared_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    child_plans: tuple[OwnerChildPointPlan, ...]
    n_parameters: int
    fingerprint: str

    @property
    def indices(self) -> tuple[int, ...]:
        """Global joint-vector columns in compact-point order."""
        return self.shared_indices + self.local_indices

    @property
    def shared_width(self) -> int:
        """Number of replicated shared coordinates."""
        return len(self.shared_indices)

    @property
    def local_width(self) -> int:
        """Number of coordinates exclusive to this owner."""
        return len(self.local_indices)

    @property
    def width(self) -> int:
        """Total compact input width."""
        return self.shared_width + self.local_width

    def gather(self, point: Any) -> Array:
        """Gather this owner's compact point without a host round trip."""
        point = jnp.asarray(point)
        if point.shape != (self.n_parameters,):
            raise ValueError(
                f"joint point has shape {point.shape}; expected {(self.n_parameters,)}."
            )
        return point[jnp.asarray(self.indices, dtype=jnp.int32)]

    def scatter_into(self, compact: Any, point: Any) -> Array:
        """Scatter one compact point into an existing full joint point."""
        compact = jnp.asarray(compact)
        point = jnp.asarray(point)
        if compact.shape != (self.width,):
            raise ValueError(
                f"compact owner point has shape {compact.shape}; expected {(self.width,)}."
            )
        if point.shape != (self.n_parameters,):
            raise ValueError(
                f"joint point has shape {point.shape}; expected {(self.n_parameters,)}."
            )
        return point.at[jnp.asarray(self.indices, dtype=jnp.int32)].set(compact)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_owner_point_plans(joint: Any) -> tuple[OwnerPointPlan, ...]:
    """Build compact point plans from an ownership-aware joint problem.

    A child's sources are the shared coordinates, then the blocks of its
    members level by level (its owner's block first), then its own local and
    state coordinates; its targets are the matching positions of its flat
    decision vector.
    """
    layout = joint.ownership_layout
    n_shared = int(layout.n_shared)
    shared_indices = tuple(range(n_shared))
    plans: list[OwnerPointPlan] = []
    for owner_index, owner_id in enumerate(layout.owner_ids):
        owner_columns = tuple(layout.owner_columns[owner_index])
        members = tuple(
            child_index
            for child_index, child_owner in enumerate(layout.child_owner_indices)
            if int(child_owner) == owner_index
        )
        local_indices = owner_columns
        for child_index in members:
            local_lo, local_hi = layout.local_slices[child_index]
            state_lo, state_hi = layout.state_slices[child_index]
            local_indices += tuple(range(local_lo, local_hi))
            local_indices += tuple(range(state_lo, state_hi))
        if len(set(local_indices)) != len(local_indices):
            raise ValueError(f"owner {owner_id!r} compact plan contains duplicate columns")
        local_position = {column: position for position, column in enumerate(local_indices)}
        child_plans: list[OwnerChildPointPlan] = []
        for child_index in members:
            child = joint.problems[child_index]
            child_shared = tuple(layout.shared_cols[child_index])
            child_owner = tuple(layout.owner_cols[child_index])
            child_local = tuple(layout.local_cols[child_index])
            local_lo, local_hi = layout.local_slices[child_index]
            state_lo, state_hi = layout.state_slices[child_index]
            n_theta = len(child_shared) + len(child_owner) + len(child_local)
            n_state = state_hi - state_lo

            sources = tuple(range(n_shared))
            targets = child_shared
            owner_sources = tuple(
                n_shared + local_position[column]
                for column in layout.child_tied_columns[child_index]
            )
            sources += owner_sources
            targets += child_owner
            local_sources = tuple(
                n_shared + local_position[column] for column in range(local_lo, local_hi)
            )
            sources += local_sources
            targets += child_local
            state_sources = tuple(
                n_shared + local_position[column] for column in range(state_lo, state_hi)
            )
            sources += state_sources
            targets += tuple(range(n_theta, n_theta + n_state))
            if len(sources) != len(targets):
                raise ValueError(
                    f"owner {owner_id!r} child {child_index} has an invalid compact map"
                )
            child_plans.append(
                OwnerChildPointPlan(
                    child_index=child_index,
                    child_width=int(np.asarray(child.x0).size),
                    source_positions=sources,
                    target_positions=targets,
                    local_positions=tuple(position - n_shared for position in sources[n_shared:]),
                    owner_slice=(
                        n_shared,
                        n_shared + len(owner_columns),
                    ),
                    child_local_slice=(
                        n_shared + local_position.get(local_lo, len(local_indices)),
                        n_shared + local_position.get(local_hi - 1, len(local_indices) - 1) + 1
                        if local_hi > local_lo
                        else n_shared + local_position.get(local_lo, len(local_indices)),
                    ),
                    child_state_slice=(
                        n_shared + local_position.get(state_lo, len(local_indices)),
                        n_shared + local_position.get(state_hi - 1, len(local_indices) - 1) + 1
                        if state_hi > state_lo
                        else n_shared + local_position.get(state_lo, len(local_indices)),
                    ),
                )
            )
        payload = {
            "schema": 1,
            # Fingerprint only the compact executable structure. Raw joint
            # columns and owner position deliberately stay out: the same owner
            # kernel must be reusable when unrelated sibling owners are added
            # to or removed from a cohort.
            "shared_width": len(shared_indices),
            "local_width": len(local_indices),
            "children": [
                {
                    "child_index": child.child_index,
                    "child_width": child.child_width,
                    "sources": child.source_positions,
                    "targets": child.target_positions,
                }
                for child in child_plans
            ],
        }
        plans.append(
            OwnerPointPlan(
                owner_id=owner_id,
                owner_index=owner_index,
                shared_indices=shared_indices,
                local_indices=local_indices,
                child_plans=tuple(child_plans),
                n_parameters=int(layout.n_joint),
                fingerprint=_fingerprint(payload),
            )
        )
    return tuple(plans)


def gather_owner_points(plans: Iterable[OwnerPointPlan], point: Any) -> tuple[Array, ...]:
    """Gather compact points in deterministic plan order."""
    return tuple(plan.gather(point) for plan in plans)


def scatter_owner_points(
    plans: Sequence[OwnerPointPlan],
    compact_points: Sequence[Any],
    *,
    check_shared: bool = True,
) -> Array:
    """Reconstruct one full joint point from every owner's compact point.

    Shared values appear in each compact point.  They are written once and,
    outside tracing, checked for exact agreement so a stale owner input cannot
    silently overwrite a newer global point.
    """
    plans = tuple(plans)
    compact_points = tuple(compact_points)
    if not plans:
        raise ValueError("scatter_owner_points requires at least one owner plan")
    if len(plans) != len(compact_points):
        raise ValueError(f"received {len(compact_points)} compact points for {len(plans)} plans")
    n_parameters = plans[0].n_parameters
    if any(plan.n_parameters != n_parameters for plan in plans):
        raise ValueError("owner plans do not describe the same joint width")
    first = jnp.asarray(compact_points[0])
    out = jnp.zeros((n_parameters,), dtype=first.dtype)
    shared_width = plans[0].shared_width
    out = out.at[jnp.asarray(plans[0].shared_indices, dtype=jnp.int32)].set(first[:shared_width])
    shared_host = None
    if check_shared:
        try:
            shared_host = np.asarray(first[:shared_width])
        except (jax.errors.TracerArrayConversionError, TypeError):
            shared_host = None
    covered: set[int] = set(plans[0].shared_indices)
    for plan, compact in zip(plans, compact_points):
        compact = jnp.asarray(compact)
        if compact.shape != (plan.width,):
            raise ValueError(
                f"owner {plan.owner_id!r} compact point has shape {compact.shape}; "
                f"expected {(plan.width,)}."
            )
        if plan.shared_indices != plans[0].shared_indices:
            raise ValueError("owner plans disagree about shared coordinates")
        if shared_host is not None and not np.array_equal(
            np.asarray(compact[:shared_width]), shared_host
        ):
            raise ValueError(f"owner {plan.owner_id!r} has stale shared coordinates")
        overlap = covered.intersection(plan.local_indices)
        if overlap:
            raise ValueError(f"owner plans overlap on local columns {sorted(overlap)}")
        covered.update(plan.local_indices)
        out = out.at[jnp.asarray(plan.local_indices, dtype=jnp.int32)].set(compact[shared_width:])
    if len(covered) != n_parameters:
        missing = sorted(set(range(n_parameters)) - covered)
        raise ValueError(f"owner plans do not cover joint columns {missing}")
    return out
