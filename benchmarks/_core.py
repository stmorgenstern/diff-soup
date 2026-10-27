"""Shared declarations and measurement helpers for benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Workload:
    """One already-constructed JAX workload and its qualification hooks."""

    function: Callable
    arguments: tuple
    check: Callable[[Any], tuple[bool, float, float]]
    telemetry: Callable[[], dict[str, Any]] = field(default=lambda: {})


@dataclass(frozen=True)
class Case:
    """Static identity and deferred construction for one benchmark case."""

    name: str
    version: str
    suites: tuple[str, ...]
    build: Callable[[], Workload]
    solver: str
    backend: str
    transform: str
    problem: str
    dimensions: dict[str, int]
    sparsity: dict[str, Any]
    batch_width: int
    direction_count: int
    seed: int = 0
    expected_targets: tuple[str, ...] = ()
    forbidden_targets: tuple[str, ...] = ("diffsoup_solve_batched",)
    required_device: str | None = None
    # The same workload built at batch width 1 with a single direction. The
    # structure check traces it to learn which scans belong to the solver itself,
    # so they are not mistaken for batch-lane scans when their length happens to
    # equal the batch width. ``None`` counts every batch-length scan.
    single_lane_build: Callable[[], Workload] | None = None

    def metadata(self) -> dict[str, Any]:
        """Return schema-ready static metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "solver": self.solver,
            "backend": self.backend,
            "transform": self.transform,
            "problem": self.problem,
            "dimensions": self.dimensions,
            "sparsity": self.sparsity,
            "batch_width": self.batch_width,
            "direction_count": self.direction_count,
            "seed": self.seed,
        }
