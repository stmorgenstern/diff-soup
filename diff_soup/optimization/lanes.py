"""A batch of independent problems as lanes: one device, several devices, several processes."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from diff_soup._layout import Layout
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


def map_lanes(fn: Callable, batch: tuple, mesh: Mesh | None) -> PyTree:
    """`vmap(fn)(*batch)`, or with a mesh each device's share of the lanes, gathered once.

    `check_vma=False`: the solvers' loops carry values that are the same on every device beside
    the per-device ones, which the varying-axes check would need marked with `pvary`; every
    output is gathered over the axis, so it is the same on every device. The lane count must be
    a multiple of the `"data"` axis.
    """
    if mesh is None:
        return jax.vmap(fn)(*batch)
    lane = PartitionSpec("data")

    def per_device(*local):
        out = jax.vmap(fn)(*local)
        return jax.tree.map(lambda leaf: jax.lax.all_gather(leaf, "data", tiled=True), out)

    specs = tuple(lane for _ in batch)
    mapped = jax.shard_map(
        per_device, mesh=mesh, in_specs=specs, out_specs=PartitionSpec(), check_vma=False
    )
    return mapped(*batch)


def placement(layout: Layout | None) -> Mesh | None:
    """The `"data"` mesh of `layout` with every axis `Auto`, or `None` for one device."""
    mesh = None if layout is None else layout.mesh
    if mesh is None or "data" not in mesh.axis_names:
        return None
    return _automatic(mesh)


def per_lane(value: ArrayLike | None, count: int) -> Array | None:
    """`value` broadcast to one entry per lane, or `None`."""
    return None if value is None else jnp.broadcast_to(jnp.asarray(value), (count,))


@eqx.filter_jit
def _program(solver, problem, batch, mesh):
    """The lanes' one compiled program, cached on the solver, the problem's static part and the mesh.

    The budgets and tolerances are lanes of the batch, runtime values of the program.
    """

    def run(x0, args, warm, max_steps, tol):
        return solver.run(problem, x0, args, max_steps=max_steps, warm_start=warm, tol=tol)

    return map_lanes(run, batch, mesh)


def lanes(
    solver: CompiledSolver,
    problem: OptimizationProblem,
    args_batch: PyTree,
    x0_batch: Float[ArrayLike, "count n"],
    *,
    layout: Layout | None = None,
    max_steps: Int[ArrayLike, "..."] | None = None,
    tol: Float[ArrayLike, "..."] | None = None,
    warm_start: SolveRecord | None = None,
) -> tuple[Float[Array, "count n"], SolveRecord]:
    """Solve a batch of problems that differ in `args` and `x0`.

    On one device this is `jax.vmap` of `solver.run`. With a `Layout` whose mesh has a `"data"`
    axis, each device solves its share and the results are gathered once. Every process passes
    the same full batch. The program compiles once per solver, problem and mesh: a later call
    with a new batch of the same shapes runs it again, and so does one with new budgets or
    tolerances, which are runtime values of the program.

    Args:
        solver: A [CompiledSolver][diff_soup.optimization.solvers.CompiledSolver].
        problem: The numeric problem every member of the batch solves.
        args_batch: The `args` of each problem, batched along the leading axis.
        x0_batch: The starting point of each problem.
        layout: The placement, or `None` for one device.
        max_steps: The step budget, one for every problem or one per problem, or `None` for the solver's own.
        tol: The scaled optimality error at which each run ends, one or one per problem, or `None` for the solver's own.
        warm_start: The `SolveRecord`s of earlier runs, batched along the leading axis, or `None`.

    Returns:
        `(x, record)`, each batched along the leading axis, as `vmap` of `run` returns them.
    """
    mesh = placement(layout)
    count = int(np.shape(x0_batch)[0])
    batch = (x0_batch, args_batch, warm_start, per_lane(max_steps, count), per_lane(tol, count))
    if mesh is None:
        return _program(solver, problem, batch, None)
    devices = int(mesh.shape["data"])
    padded = -(-count // devices) * devices
    batch = _pad(batch, count, padded)
    if jax.process_count() > 1:
        batch = _global(batch, NamedSharding(mesh, PartitionSpec("data")))
    out = _program(solver, problem, batch, mesh)
    return jax.tree.map(lambda leaf: leaf[:count], out)
