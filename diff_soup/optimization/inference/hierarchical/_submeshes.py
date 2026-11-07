"""Replicated placement on a one-lane submesh, and common-shape sums over the lanes of a mesh.

A joint problem placed on a mesh whose structural groups run on disjoint
submeshes ([mesh_groups][diff_soup.mesh_groups]) moves its point onto each
group's submesh and sums the groups' statistics over the lanes of the mesh.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from diff_soup._distributed import mesh_groups, owns


def on_submesh(value, mesh: Mesh):
    """Place one local replica on a submesh without a device-to-host transfer."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if not value.is_fully_replicated:
            raise ValueError("Owner inputs must be replicated before submesh placement.")
        value = value.addressable_shards[0].data
    return jax.device_put(value, NamedSharding(mesh, P()))


def sum_lanes(mesh: Mesh, contributions, *, template):
    """Sum common-shape device buffers with one full-mesh experiment collective.

    Contributions are ``(lane, pytree)`` pairs produced only by the processes
    owning that lane. Missing lanes contribute zeros; ragged owner factors stay
    on their original submeshes.
    """
    submeshes = mesh_groups(mesh, (1,) * int(mesh.shape["data"]))
    local = {}
    for lane, value in contributions:
        local.setdefault(lane, []).append(value)
    buffers = {}
    for lane, submesh in enumerate(submeshes):
        if not owns(submesh):
            continue
        values = local.get(lane, ())
        if values:
            value = jax.tree.map(lambda *items: sum(items[1:], items[0]), *values)
        else:
            value = jax.tree.map(
                lambda shape: on_submesh(np.zeros(shape.shape, shape.dtype), submesh), template
            )
        expanded = jax.tree.map(lambda item: jnp.expand_dims(item, 0), value)
        buffers[lane] = jax.tree.map(
            lambda item: {shard.device: shard.data for shard in item.addressable_shards},
            expanded,
        )

    # Dictionaries of device buffers are kept as leaves, independent of their
    # process-local device keys.
    def is_buffers(item):
        return isinstance(item, dict)

    def assemble(shape, *parts):
        by_device = {device: value for part in parts for device, value in part.items()}
        sharding = NamedSharding(mesh, P("data"))
        arrays = [by_device[device] for device in sharding.addressable_devices]
        return jax.make_array_from_single_device_arrays(
            (len(submeshes), *shape.shape), sharding, arrays
        )

    stacked = jax.tree.map(assemble, template, *buffers.values(), is_leaf=is_buffers)
    specs = jax.tree.map(lambda _: P("data"), template)
    replicated = jax.tree.map(lambda _: P(), template)

    @jax.jit
    @functools.partial(
        jax.shard_map, mesh=mesh, in_specs=(specs,), out_specs=replicated, check_vma=True
    )
    def reduce(values):
        return jax.tree.map(lambda value: jax.lax.psum(jnp.sum(value, axis=0), "data"), values)

    return reduce(stacked)
