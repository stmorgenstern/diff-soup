"""Reductions of lane results to the blocks an optimizer works with.

A lane returns its normal matrix and its gradient in its own narrow
coordinates. A reducer sums the lanes of each owner into that owner's blocks
of the bordered system

.. code-block:: text

    [ A    B_1  ...  B_K ] [ s_g ]     [ g_g ]
    [ B_1' C_1          ] [ s_1 ]     [ g_1 ]
    [ ...       ...     ] [ ... ] = - [ ... ]
    [ B_K'          C_K ] [ s_K ]     [ g_K ]

where ``g`` are the shared coordinates and ``1 .. K`` the owners. The weights
of the owners belong to the reduction, not to the lanes: a lane result is the
same for every weighting, so one evaluation serves every bootstrap replicate
that shares its point.

Every sum runs over a table that lists the lanes of an owner in a fixed order,
so its value does not depend on which device produced a lane or on the order
in which lanes arrived. A reduction refuses a set of lanes that contains a
failed one of nonzero weight. A lane of weight zero is outside the sum: a
failed lane's cost and derivatives are zero, so its failure is not refused.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.engine._status import ABSENT, LaneStatus, failures

Array = jax.Array


class LaneFailure(RuntimeError):
    """A reduction was asked to sum a set of lanes that contains a failed lane.

    Attributes:
        failures: One record per failed lane: ``lane``, ``kind``,
            ``t_reached``, ``rung``.
    """

    def __init__(self, records: tuple):
        """Name the failed lanes."""
        self.failures = tuple(records)
        listed = "; ".join(
            f"{record['lane']}: {record['kind']} at t = {record['t_reached']:.6g}"
            for record in self.failures[:8]
        )
        more = "" if len(self.failures) <= 8 else f"; and {len(self.failures) - 8} more"
        super().__init__(
            f"{len(self.failures)} lane(s) failed, so their sum is not defined ({listed}{more}). "
            "Every lane has to solve at the point, or the caller has to state what a failed "
            "lane contributes."
        )


class TreeTables(eqx.Module):
    """Where the narrow coordinates of every lane sit in the blocks of its owner.

    Attributes:
        lanes: Per owner, the indices of its lanes, padded to the largest
            lane count of an owner.
        present: Whether an entry of ``lanes`` names a lane.
        local: Per lane and local narrow coordinate, its position in the
            local block of the lane's owner.
        mask: Per owner, which positions of its local block are coordinates;
            the others pad an owner narrower than the widest.
        n_shared: The number of shared coordinates, which lead every narrow
            point.
    """

    lanes: Array
    present: Array
    local: Array
    mask: Array
    n_shared: int = eqx.field(static=True)


class OwnerBlocks(eqx.Module):
    """The blocks of the bordered system, one leading entry per owner.

    Attributes:
        shared: Per owner, its contribution ``A_o`` to the shared block.
        cross: Per owner, the coupling ``B_o`` of shared and local coordinates.
        local: Per owner, its local block ``C_o``; a padded position carries
            one on the diagonal.
        gradient_shared: Per owner, its contribution to the shared gradient.
        gradient_local: Per owner, its local gradient.
        cost: Per owner, its weighted cost.
        rows: Per owner, its number of residual rows.
        mask: Per owner, which positions of its local block are coordinates.
    """

    shared: Array
    cross: Array
    local: Array
    gradient_shared: Array
    gradient_local: Array
    cost: Array
    rows: Array
    mask: Array


class OwnerGradients(eqx.Module):
    """The gradient tier by tier, one leading entry per owner.

    Attributes:
        shared: Per owner, its contribution to the gradient of the shared
            coordinates; their sum over the owners is that gradient.
        local: Per owner, the gradient of its local coordinates.
        cost: Per owner, its weighted cost.
        rows: Per owner, its number of residual rows.
        mask: Per owner, which positions of its local block are coordinates.
    """

    shared: Array
    local: Array
    cost: Array
    rows: Array
    mask: Array


def tree_tables(
    owner_of_lane: Any, local_positions: Any, local_widths: Any, n_shared: int, device: Any = None
):
    """Build the tables of a two-tier tree on the host and place them on ``device``.

    Args:
        owner_of_lane: Per lane, the index of its owner.
        local_positions: Per lane, the position of each of its local narrow
            coordinates in the local block of its owner.
        local_widths: Per owner, the width of its local block.
        n_shared: The number of shared coordinates.
        device: The JAX device of the tables; ``None`` for the default device.

    Returns:
        The [TreeTables][diff_soup.inference.engine._reducers.TreeTables].
    """
    owner_of_lane = np.asarray(owner_of_lane, dtype=np.int64).reshape(-1)
    local_positions = np.asarray(local_positions, dtype=np.int32)
    local_widths = np.asarray(local_widths, dtype=np.int64).reshape(-1)
    n_owners = int(local_widths.shape[0])
    if local_positions.ndim != 2 or local_positions.shape[0] != owner_of_lane.shape[0]:
        raise ValueError(
            "local_positions needs one row per lane; got shape "
            f"{local_positions.shape} for {owner_of_lane.shape[0]} lanes."
        )
    if owner_of_lane.size and (owner_of_lane.min() < 0 or owner_of_lane.max() >= n_owners):
        raise ValueError(f"a lane names an owner outside 0 .. {n_owners - 1}.")
    widest = int(local_widths.max()) if n_owners else 0
    for lane, owner in enumerate(owner_of_lane):
        positions = local_positions[lane]
        if positions.size and (positions.min() < 0 or positions.max() >= local_widths[owner]):
            raise ValueError(
                f"lane {lane} places a coordinate outside the local block of its owner, "
                f"which has {int(local_widths[owner])} coordinates."
            )
        if len(set(positions.tolist())) != positions.size:
            raise ValueError(f"lane {lane} places two coordinates at one position.")
    members = [np.flatnonzero(owner_of_lane == owner) for owner in range(n_owners)]
    most = max((len(member) for member in members), default=0)
    lanes = np.zeros((n_owners, most), dtype=np.int32)
    present = np.zeros((n_owners, most), dtype=bool)
    for owner, member in enumerate(members):
        lanes[owner, : len(member)] = member
        present[owner, : len(member)] = True
    mask = np.arange(widest)[None, :] < local_widths[:, None]
    lanes, present, local_positions, mask = jax.device_put(
        (lanes, present, local_positions, mask), device
    )
    return TreeTables(
        lanes=lanes, present=present, local=local_positions, mask=mask, n_shared=int(n_shared)
    )


def refuse_failed(status: LaneStatus, lane_ids: tuple, weights: Any = None) -> None:
    """Raise [LaneFailure][diff_soup.inference.engine._reducers.LaneFailure] if a lane failed.

    Args:
        status: The status of the lanes, one entry per lane.
        lane_ids: The identifier of each lane, in the same order.
        weights: ``None``, or one weight per lane in the same order: a failed
            lane of weight zero is outside the sum and is not refused.

    Raises:
        LaneFailure: If a lane failed (with ``weights``, a lane of nonzero
            weight).
        ValueError: If ``weights`` does not give one value per lane, or a
            unit that was not evaluated (status ``absent``) would weigh more
            than zero.
    """
    records = failures(status, lane_ids)
    absent = np.asarray(status.kind).reshape(-1) == ABSENT
    if (records or np.any(absent)) and weights is not None:
        weights = np.asarray(weights).reshape(-1)
        if weights.shape[0] != len(lane_ids):
            raise ValueError(
                f"the refusal of failed lanes takes one weight per lane, {len(lane_ids)}; got "
                f"{weights.shape[0]}."
            )
        position = {lane: index for index, lane in enumerate(lane_ids)}
        records = tuple(record for record in records if weights[position[record["lane"]]] != 0)
        absent = absent & (weights != 0)
    if np.any(absent):
        named = [lane_ids[int(index)] for index in np.flatnonzero(absent)]
        listed = ", ".join(str(lane) for lane in named[:8])
        more = "" if len(named) <= 8 else f" and {len(named) - 8} more"
        raise ValueError(
            f"the lanes {listed}{more} were not evaluated (status absent: the request's units= "
            "left them out) and would weigh more than zero in this sum; evaluate them, or give "
            "them the weight zero."
        )
    if records:
        raise LaneFailure(records)


def _lane_weights_of_owners(tables: TreeTables, owner_weights: Any) -> np.ndarray:
    """Each lane's weight on the host: that of its owner."""
    owner_weights = np.asarray(owner_weights).reshape(-1)
    lanes = np.asarray(tables.lanes)
    present = np.asarray(tables.present)
    per_lane = np.ones((int(tables.local.shape[0]),), dtype=owner_weights.dtype)
    for owner in range(lanes.shape[0]):
        per_lane[lanes[owner][present[owner]]] = owner_weights[owner]
    return per_lane


def _owner_sum(tables: TreeTables, per_lane: Array) -> Array:
    """Sum a per-lane array over the lanes of every owner, in table order."""
    gathered = per_lane[tables.lanes]
    keep = jnp.reshape(tables.present, tables.present.shape + (1,) * (per_lane.ndim - 1))
    return jnp.sum(jnp.where(keep, gathered, jnp.zeros((), per_lane.dtype)), axis=1)


def _weighted(weights: Array, per_owner: Array) -> Array:
    return jnp.reshape(weights, weights.shape + (1,) * (per_owner.ndim - 1)) * per_owner


@jax.jit
def _reduce_blocks(tables: TreeTables, normal, gradient, cost, rows, weights) -> OwnerBlocks:
    g = tables.n_shared
    width = tables.mask.shape[1]
    dtype = normal.dtype

    def in_owner_coordinates(normal_lane, gradient_lane, positions):
        cross = jnp.zeros((g, width), dtype).at[:, positions].add(normal_lane[:g, g:])
        local = (
            jnp.zeros((width, width), dtype)
            .at[positions[:, None], positions[None, :]]
            .add(normal_lane[g:, g:])
        )
        local_gradient = jnp.zeros((width,), dtype).at[positions].add(gradient_lane[g:])
        return normal_lane[:g, :g], cross, local, gradient_lane[:g], local_gradient

    shared, cross, local, gradient_shared, gradient_local = jax.vmap(in_owner_coordinates)(
        normal, gradient, tables.local
    )
    weights = jnp.asarray(weights, dtype)
    padded = jnp.logical_not(tables.mask).astype(dtype)
    return OwnerBlocks(
        shared=_weighted(weights, _owner_sum(tables, shared)),
        cross=_weighted(weights, _owner_sum(tables, cross)),
        local=_weighted(weights, _owner_sum(tables, local)) + jax.vmap(jnp.diag)(padded),
        gradient_shared=_weighted(weights, _owner_sum(tables, gradient_shared)),
        gradient_local=_weighted(weights, _owner_sum(tables, gradient_local)),
        cost=weights * _owner_sum(tables, cost),
        rows=_owner_sum(tables, rows),
        mask=tables.mask,
    )


@jax.jit
def _reduce_gradient(tables: TreeTables, gradient, cost, rows, weights) -> OwnerGradients:
    g = tables.n_shared
    width = tables.mask.shape[1]
    dtype = gradient.dtype

    def in_owner_coordinates(gradient_lane, positions):
        local = jnp.zeros((width,), dtype).at[positions].add(gradient_lane[g:])
        return gradient_lane[:g], local

    shared, local = jax.vmap(in_owner_coordinates)(gradient, tables.local)
    weights = jnp.asarray(weights, dtype)
    return OwnerGradients(
        shared=_weighted(weights, _owner_sum(tables, shared)),
        local=_weighted(weights, _owner_sum(tables, local)),
        cost=weights * _owner_sum(tables, cost),
        rows=_owner_sum(tables, rows),
        mask=tables.mask,
    )


def _owner_weights(weights: Any, tables: TreeTables, dtype: Any) -> np.ndarray:
    """One weight per owner, on the host; the compiled reduction takes them where it runs."""
    n_owners = int(tables.mask.shape[0])
    if weights is None:
        return np.ones((n_owners,), dtype)
    weights = np.asarray(weights, dtype)
    if weights.shape != (n_owners,):
        raise ValueError(
            f"the reduction takes one weight per owner, shape {(n_owners,)}; got {weights.shape}."
        )
    return weights


def reduce_blocks(tables: TreeTables, results: Any, lane_ids: tuple, weights: Any = None):
    """Sum the blocks of a set of lanes into the blocks of their owners.

    Args:
        tables: The tables of the tree.
        results: The results of the lanes in the ``blocks`` mode, in the lane
            order the tables were built for.
        lane_ids: The identifier of each lane, in the same order.
        weights: One weight per owner; one when omitted.

    Returns:
        The [OwnerBlocks][diff_soup.inference.engine._reducers.OwnerBlocks].

    Raises:
        LaneFailure: If a lane of the set failed whose owner's weight is not
            zero.
        ValueError: If the results carry no blocks.
    """
    if results.normal is None or results.gradient is None:
        raise ValueError(
            "the blocks of the owners are summed from lane results of the blocks mode; "
            "these results carry no normal matrix."
        )
    owner_weights = _owner_weights(weights, tables, results.normal.dtype)
    refuse_failed(
        results.status,
        lane_ids,
        None if weights is None else _lane_weights_of_owners(tables, owner_weights),
    )
    return _reduce_blocks(
        tables, results.normal, results.gradient, results.cost, results.rows, owner_weights
    )


def reduce_gradient(tables: TreeTables, results: Any, lane_ids: tuple, weights: Any = None):
    """Sum the gradients of a set of lanes into the gradient of every tier.

    Args:
        tables: The tables of the tree.
        results: The results of the lanes in the ``gradient`` or ``blocks``
            mode, in the lane order the tables were built for.
        lane_ids: The identifier of each lane, in the same order.
        weights: One weight per owner; one when omitted.

    Returns:
        The [OwnerGradients][diff_soup.inference.engine._reducers.OwnerGradients].

    Raises:
        LaneFailure: If a lane of the set failed whose owner's weight is not
            zero.
        ValueError: If the results carry no gradient.
    """
    if results.gradient is None:
        raise ValueError(
            "the gradient is summed from lane results of the gradient or the blocks mode; "
            "these results carry no gradient."
        )
    owner_weights = _owner_weights(weights, tables, results.gradient.dtype)
    refuse_failed(
        results.status,
        lane_ids,
        None if weights is None else _lane_weights_of_owners(tables, owner_weights),
    )
    return _reduce_gradient(tables, results.gradient, results.cost, results.rows, owner_weights)


__all__ = [
    "LaneFailure",
    "OwnerBlocks",
    "OwnerGradients",
    "TreeTables",
    "reduce_blocks",
    "reduce_gradient",
    "refuse_failed",
    "tree_tables",
]
