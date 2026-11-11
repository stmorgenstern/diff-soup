"""The Gauss-Newton system of a parameter hierarchy, solved along its tree.

The system of a hierarchy is stored per node of every level, as the level
reducer sums it ([LevelBlocks][diff_soup.optimization.inference.engine.LevelBlocks]): per
node its diagonal block and its coupling to each of its ancestors, and at the
root the shared block. Joint columns are laid out level by level: the shared
coordinates, then every node of level 1, and so on down to the experiments.

A damped solve eliminates the levels from the experiments up. Every node of a
level is factored at once, and the Schur complement of its block is folded into
the blocks of its ancestors, so the stored set of blocks never grows. The
shared block is solved last, and the steps follow top-down. A member's rows
are eliminated with its node, so they cost in proportion to the members; the
other equality rows are imposed by one small projection, and the factors give
the inverse-norm ``s^T (H + lambda I)^-1 s`` a secular search on ``lambda``
needs.

Every array computation is compiled once per shape; the damping, the scale,
the diagonal added to the scaled matrix, the point, and which coordinates are
fixed and which nodes are held enter as arguments. The secular search runs on
the host.
"""

from __future__ import annotations

from diff_soup.optimization.inference.hierarchical._level_inner import (
    inner_added,
    inner_predicted,
    inner_programs,
    inner_quadratic,
    inner_row_multipliers,
    inner_secular_step,
    inner_solve_scaled,
)
from diff_soup.optimization.inference.hierarchical._level_solve import (
    AttemptSolve,
    attempt_solve,
    level_quadratic,
    level_secular_step,
    level_vectors,
    row_multipliers,
    scaled_system,
    solve_damped,
    solve_scaled,
)
from diff_soup.optimization.inference.hierarchical._level_system import (
    LevelLayout,
    LevelSystem,
    MemberRows,
    level_dense,
    level_layout,
    level_system,
    node_of_columns,
    regularizer_blocks_at,
)

__all__ = [
    "AttemptSolve",
    "LevelLayout",
    "LevelSystem",
    "MemberRows",
    "attempt_solve",
    "inner_added",
    "inner_predicted",
    "inner_programs",
    "inner_quadratic",
    "inner_row_multipliers",
    "inner_secular_step",
    "inner_solve_scaled",
    "level_dense",
    "level_layout",
    "level_quadratic",
    "level_secular_step",
    "level_system",
    "level_vectors",
    "node_of_columns",
    "regularizer_blocks_at",
    "row_multipliers",
    "scaled_system",
    "solve_damped",
    "solve_scaled",
]
