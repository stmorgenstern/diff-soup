"""A batch of independent problems as lanes: one device, several devices, several processes."""

from __future__ import annotations

from typing import Any


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
    raise NotImplementedError
