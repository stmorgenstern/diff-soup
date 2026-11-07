"""A batch of independent problems as lanes: one device, several devices, several processes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike, Float, PyTree

from diff_soup.optimization._abc import OptimizationProblem
from diff_soup.optimization.solvers._protocol import CompiledSolver, SolveRecord


def _pad(tree: PyTree, count: int, padded: int) -> PyTree:
    """Every leaf's lane axis grown from `count` to `padded` by repeating its last lane."""
    if padded == count:
        return tree

    def grow(leaf):
        leaf = jnp.asarray(leaf)
        tail = jnp.repeat(leaf[-1:], padded - count, axis=0)
        return jnp.concatenate([leaf, tail], axis=0)

    return jax.tree.map(grow, tree)


def _global(tree: PyTree, sharding: NamedSharding) -> PyTree:
    """Host leaves, the same full batch on every process, as global arrays sharded by lane."""

    def place(leaf):
        if isinstance(leaf, jax.core.Tracer):
            return leaf
        if isinstance(leaf, jax.Array) and leaf.sharding == sharding:
            return leaf
        host = np.asarray(leaf)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])

    return jax.tree.map(place, tree)


def _automatic(mesh: Mesh) -> Mesh:
    """`mesh` with every axis `Auto`, as diff-soup places arrays (`jax.make_mesh` makes them `Explicit`)."""
    if all(kind == AxisType.Auto for kind in mesh.axis_types):
        return mesh
    return Mesh(mesh.devices, mesh.axis_names, axis_types=(AxisType.Auto,) * len(mesh.axis_names))


@eqx.filter_jit
def _program(solver, problem, args_batch, x0_batch, mesh, max_steps):
    """The lanes' one compiled program, cached on the solver, the problem's static part and the mesh.

    With a mesh, each device runs `vmap` of `run` on its share and the results are gathered
    once. `check_vma=False`: the solver's loops carry values that are the same on every device
    beside the per-device ones, which the varying-axes check would need marked with `pvary`;
    every output is gathered over the axis, so it is the same on every device.
    """

    def run(x0, args):
        return solver.run(problem, x0, args, max_steps=max_steps)

    if mesh is None:
        return jax.vmap(run)(x0_batch, args_batch)
    lane = PartitionSpec("data")

    def per_device(x0_local, args_local):
        out = jax.vmap(run)(x0_local, args_local)
        return jax.tree.map(lambda leaf: jax.lax.all_gather(leaf, "data", tiled=True), out)

    mapped = jax.shard_map(
        per_device, mesh=mesh, in_specs=(lane, lane), out_specs=PartitionSpec(), check_vma=False
    )
    return mapped(x0_batch, args_batch)


def lanes(
    solver: CompiledSolver,
    problem: OptimizationProblem,
    args_batch: PyTree,
    x0_batch: Float[ArrayLike, "count n"],
    *,
    layout: object = None,
    max_steps: int | None = None,
) -> tuple[Float[Array, "count n"], SolveRecord]:
    """Solve a batch of problems that differ in `args` and `x0`.

    On one device this is `jax.vmap` of `solver.run`. With a `Layout` whose mesh has a `"data"`
    axis, each device solves its share and the results are gathered once. Every process passes
    the same full batch. The program compiles once per solver, problem and mesh: a later call
    with a new batch of the same shapes runs it again.

    Args:
        solver: A [CompiledSolver][diff_soup.optimization.solvers.CompiledSolver].
        problem: The numeric problem every member of the batch solves.
        args_batch: The `args` of each problem, batched along the leading axis.
        x0_batch: The starting point of each problem.
        layout: The placement, or `None` for one device.
        max_steps: The step budget, or `None` for the solver's own.

    Returns:
        `(x, record)`, each batched along the leading axis, as `vmap` of `run` returns them.
    """
    mesh = None if layout is None else layout.mesh
    if mesh is None or "data" not in mesh.axis_names:
        return _program(solver, problem, args_batch, x0_batch, None, max_steps)
    mesh = _automatic(mesh)
    count = int(np.shape(x0_batch)[0])
    devices = int(mesh.shape["data"])
    padded = -(-count // devices) * devices
    x0_batch, args_batch = _pad((x0_batch, args_batch), count, padded)
    if jax.process_count() > 1:
        sharding = NamedSharding(mesh, PartitionSpec("data"))
        x0_batch, args_batch = _global((x0_batch, args_batch), sharding)
    out = _program(solver, problem, args_batch, x0_batch, mesh, max_steps)
    return jax.tree.map(lambda leaf: leaf[:count], out)
