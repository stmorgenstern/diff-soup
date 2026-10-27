"""diff_soup.domains: meshes, grids, fields, and the operator bank.

Exports: ``Domain``, ``Grid1D``, ``GridND``, ``Mesh``, ``TaggedMesh``,
``StaggeredGrid``, ``DiscreteField``, ``Operators``, ``AbstractOperator``,
and mesh/grid factory functions.

``TaggedMesh`` is the public surface for meshes with boundary tags.

AMR helpers (``kelly_error_estimate``, ``refine_adaptive``,
``remesh_adaptive``) are not exposed; use [refine_uniform][diff_soup.domains.geometry.refine_uniform]
for uniform red refinement.

Raw stencil kernels (``fvm_*`` / ``fdm_*``) are private and not re-exported.
"""

from diff_soup.domains.base import Domain
from diff_soup.domains.fields import DiscreteField
from diff_soup.domains.grid import (
    CoordSystem,
    Grid1D,
    GridND,
    cartesian_grid_1d,
    cartesian_grid_2d,
    cartesian_grid_3d,
    cylindrical_grid_1d,
    spherical_grid_1d,
)
from diff_soup.domains.mesh import (
    Mesh,
    TaggedMesh,
    box_mesh,
    cylindrical_interval_mesh,
    interval_mesh,
    rectangle_mesh,
    spherical_interval_mesh,
)
from diff_soup.domains.operators import AbstractOperator, Operators
from diff_soup.domains.staggered_grid import StaggeredGrid, staggered_grid_2d, staggered_grid_3d

__all__ = [
    "Domain",
    "DiscreteField",
    "Mesh",
    "TaggedMesh",
    "rectangle_mesh",
    "box_mesh",
    "interval_mesh",
    "cylindrical_interval_mesh",
    "spherical_interval_mesh",
    "CoordSystem",
    "Grid1D",
    "GridND",
    "StaggeredGrid",
    "cartesian_grid_1d",
    "cartesian_grid_2d",
    "cartesian_grid_3d",
    "cylindrical_grid_1d",
    "spherical_grid_1d",
    "staggered_grid_2d",
    "staggered_grid_3d",
    "Operators",
    # Extension contract ABC: escape hatch for custom discretizations
    # (DG, spectral, particle); not concrete — users subclass this, not
    # Operators directly.
    "AbstractOperator",
]
