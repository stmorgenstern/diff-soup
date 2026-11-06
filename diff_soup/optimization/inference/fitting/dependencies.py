"""Static residual-block dependency declarations.

Large least-squares residuals often decompose into blocks that observe only a
small subset of a problem's decision vector.  ``ResidualDependencyBlock``
records that subset without attaching application meaning to it.  Statistics
assemblers may differentiate the block in its compact coordinate and scatter
the exact normal contribution back into a larger owner layout.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ResidualDependencyBlock:
    """One residual callable and the decision columns it can observe.

    ``residual_fn`` accepts the problem's complete flat point.  Consumers
    are expected to reconstruct that point from ``columns`` before
    differentiation, which makes excluded columns structurally absent from the
    tangent program rather than merely zero-valued.
    """

    name: str
    columns: tuple[int, ...]
    row_count: int
    residual_fn: Callable[[Any], Any] = dataclasses.field(
        repr=False,
        compare=False,
    )
    elimination_groups: tuple[tuple[int, ...], ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate the immutable structural declaration."""
        columns = tuple(int(column) for column in self.columns)
        if len(set(columns)) != len(columns):
            raise ValueError(f"residual dependency block {self.name!r} repeats columns")
        if any(column < 0 for column in columns):
            raise ValueError(f"residual dependency block {self.name!r} has negative columns")
        if int(self.row_count) < 0:
            raise ValueError(f"residual dependency block {self.name!r} has a negative row count")
        if not callable(self.residual_fn):
            raise TypeError("ResidualDependencyBlock residual_fn must be callable")
        elimination_groups = tuple(
            tuple(int(column) for column in group) for group in self.elimination_groups
        )
        flattened = tuple(column for group in elimination_groups for column in group)
        if any(not group for group in elimination_groups):
            raise ValueError(
                f"residual dependency block {self.name!r} has an empty elimination group"
            )
        if len(set(flattened)) != len(flattened):
            raise ValueError(
                f"residual dependency block {self.name!r} repeats an eliminable column"
            )
        if any(column not in columns for column in flattened):
            raise ValueError(
                f"residual dependency block {self.name!r} declares an "
                "eliminable column outside its dependency columns"
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "elimination_groups", elimination_groups)


@dataclasses.dataclass(frozen=True)
class ResidualDependencyBucket:
    """Same-shape residual blocks evaluated by one indexed callable.

    ``residual_fn(active_point, lane_index)`` consumes only one lane's declared
    columns. Every lane has the same active width and row count, allowing a
    statistics kernel to map a single traced solver body over heterogeneous
    numeric data without unrolling one copy per block.
    """

    name: str
    columns: tuple[tuple[int, ...], ...]
    row_count: int
    residual_fn: Callable[[Any, Any], Any] = dataclasses.field(
        repr=False,
        compare=False,
    )
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate lane widths and the indexed callable."""
        columns = tuple(tuple(int(column) for column in lane) for lane in self.columns)
        if not columns:
            raise ValueError("ResidualDependencyBucket requires at least one lane")
        widths = {len(lane) for lane in columns}
        if len(widths) != 1:
            raise ValueError(f"residual dependency bucket {self.name!r} has ragged lane widths")
        for lane in columns:
            if len(set(lane)) != len(lane) or any(column < 0 for column in lane):
                raise ValueError(f"residual dependency bucket {self.name!r} has invalid columns")
        if int(self.row_count) < 0:
            raise ValueError(f"residual dependency bucket {self.name!r} has a negative row count")
        if not callable(self.residual_fn):
            raise TypeError("ResidualDependencyBucket residual_fn must be callable")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "row_count", int(self.row_count))

    @property
    def width(self) -> int:
        """Active derivative width shared by every lane."""
        return len(self.columns[0])

    @property
    def block_count(self) -> int:
        """Number of residual blocks represented by the bucket."""
        return len(self.columns)


def audit_residual_dependencies(
    blocks: tuple[ResidualDependencyBlock, ...],
    point: Any,
    *,
    atol: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    """Differentiate excluded columns and fail if a declaration is incomplete.

    This is intentionally a debug/validation utility: it forms each block's
    full Jacobian to prove that every excluded directional derivative is zero.
    Production statistics consume the compact declarations and never pay this
    audit cost.
    """
    point = jnp.asarray(point)
    if point.ndim != 1:
        raise ValueError(f"dependency-audit point must be one-dimensional; got {point.shape}")
    width = int(point.shape[0])
    records = []
    for block in tuple(blocks):
        invalid = [column for column in block.columns if column >= width]
        if invalid:
            raise ValueError(
                f"residual dependency block {block.name!r} declares columns "
                f"outside width {width}: {invalid}"
            )
        jacobian = jax.jacfwd(lambda value, _fn=block.residual_fn: jnp.reshape(_fn(value), (-1,)))(
            point
        )
        excluded = tuple(column for column in range(width) if column not in block.columns)
        if excluded:
            excluded_values = np.asarray(jacobian)[:, np.asarray(excluded)]
            maximum = float(np.max(np.abs(excluded_values), initial=0.0))
        else:
            maximum = 0.0
        passed = bool(np.isfinite(maximum) and maximum <= float(atol))
        records.append(
            {
                "name": block.name,
                "row_count": block.row_count,
                "declared_width": len(block.columns),
                "excluded_width": len(excluded),
                "max_abs_excluded_derivative": maximum,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                f"residual dependency block {block.name!r} has excluded "
                f"derivative {maximum:.6g} above atol={float(atol):.6g}"
            )
    return tuple(records)


__all__ = [
    "ResidualDependencyBlock",
    "ResidualDependencyBucket",
    "audit_residual_dependencies",
]
