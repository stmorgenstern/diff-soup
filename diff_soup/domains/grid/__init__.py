"""Structured-grid domains for FVM/FDM (1D and N-D Cartesian).

Re-exports the ``Grid1D`` / ``GridND`` pytrees and the coordinate-system
factories.
"""

from diff_soup.domains.grid.core import (
    CoordSystem,
    Grid1D,
    GridND,
    cartesian_grid_1d,
    cartesian_grid_2d,
    cartesian_grid_3d,
    cylindrical_grid_1d,
    spherical_grid_1d,
)

__all__ = [
    "CoordSystem",
    "Grid1D",
    "GridND",
    "cartesian_grid_1d",
    "cartesian_grid_2d",
    "cartesian_grid_3d",
    "cylindrical_grid_1d",
    "spherical_grid_1d",
]
