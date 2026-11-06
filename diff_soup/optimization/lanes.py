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
    """Run `solver` on a batch of problems that differ only in `args` and `x0`.

    On one device this is `vmap` of the solver's `run` with masked termination. With a `Layout`
    whose mesh has a `data` axis, the lane axis is sharded over it with `shard_map` and a loop
    per device, so devices never synchronize per iteration and gather at the end. Across
    processes the same program runs on `ds.global_mesh(("data",))` with each process's local
    shard.

    A lane count that the `data` axis does not divide is padded with copies of the last lane,
    which are dropped from the result. Across processes, every process passes the same full
    batch (or global arrays already sharded over `data`), and every process receives the whole
    result. The derivative in `args` is the solver's implicit one, lane by lane.

    Args:
        solver: A [CompiledSolver][diff_soup.optimization.CompiledSolver].
        problem: The numeric problem the lanes share.
        args_batch: The `args` of each lane, batched along the leading axis.
        x0_batch: The starting point of each lane, `(lanes, n)`.
        layout: The placement of the lane axis, or `None` for one device.
        max_steps: The step budget, or the solver's own.

    Returns:
        `(x, record)`: the solutions, `(lanes, n)`, and the batched
        [SolveRecord][diff_soup.optimization.SolveRecord].
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
