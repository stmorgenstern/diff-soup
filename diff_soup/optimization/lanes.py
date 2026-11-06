"""A batch of independent problems as lanes: one device, several devices, several processes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec


def _pad(tree: Any, count: int, padded: int) -> Any:
    """Every leaf's lane axis grown from `count` to `padded` by repeating its last lane."""
    if padded == count:
        return tree

    def grow(leaf):
        leaf = jnp.asarray(leaf)
        tail = jnp.repeat(leaf[-1:], padded - count, axis=0)
        return jnp.concatenate([leaf, tail], axis=0)

    return jax.tree.map(grow, tree)


def _global(tree: Any, sharding: NamedSharding) -> Any:
    """Host leaves, the same full batch on every process, as global arrays sharded by lane."""

    def place(leaf):
        if isinstance(leaf, jax.core.Tracer):
            return leaf
        if isinstance(leaf, jax.Array) and leaf.sharding == sharding:
            return leaf
        host = np.asarray(leaf)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])

    return jax.tree.map(place, tree)


def lanes(
    solver: Any,
    problem: Any,
    args_batch: Any,
    x0_batch: Any,
    *,
    layout: Any = None,
    max_steps: int | None = None,
) -> tuple[Any, Any]:
    """Solve a batch of problems that differ in `args` and `x0`.

    On one device this is `jax.vmap` of `solver.run`. With a `Layout` whose mesh has a `"data"`
    axis, each device solves its share and the results are gathered once. Every process passes
    the same full batch.

    Args:
        solver: A [CompiledSolver][diff_soup.optimization.CompiledSolver].
        problem: The numeric problem every member of the batch solves.
        args_batch: The `args` of each problem, batched along the leading axis.
        x0_batch: The starting point of each problem, `(count, n)`.
        layout: The placement, or `None` for one device.
        max_steps: The step budget, or `None` for the solver's own.

    Returns:
        `(x, record)`, each batched along the leading axis, as `vmap` of `run` returns them.
    """

    def run(x0, args):
        return solver.run(problem, x0, args, max_steps=max_steps)

    mesh = None if layout is None else layout.mesh
    if mesh is None or "data" not in mesh.axis_names:
        return jax.vmap(run)(x0_batch, args_batch)

    count = int(np.shape(x0_batch)[0])
    devices = int(mesh.shape["data"])
    padded = -(-count // devices) * devices
    x0_batch, args_batch = _pad((x0_batch, args_batch), count, padded)
    lane = PartitionSpec("data")

    def per_device(x0_local, args_local):
        out = jax.vmap(run)(x0_local, args_local)
        return jax.tree.map(lambda leaf: jax.lax.all_gather(leaf, "data", tiled=True), out)

    mapped = jax.shard_map(
        per_device, mesh=mesh, in_specs=(lane, lane), out_specs=PartitionSpec(), check_vma=False
    )
    if jax.process_count() > 1:
        x0_batch, args_batch = _global((x0_batch, args_batch), NamedSharding(mesh, lane))
    out = jax.jit(mapped)(x0_batch, args_batch)
    return jax.tree.map(lambda leaf: leaf[:count], out)
