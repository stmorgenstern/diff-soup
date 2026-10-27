"""StaggeredGrid: N-D Cartesian grid with first-class face-centered storage.

A [StaggeredGrid][diff_soup.domains.staggered_grid.StaggeredGrid] wraps an underlying [GridND][diff_soup.domains.grid.GridND]
(the cell-center grid) and exposes the array shapes of values that live on
**faces** rather than cell centers.  It is the domain type for MAC
(marker-and-cell) layouts, such as the velocity/pressure staggering of
incompressible Navier-Stokes.

Conventions (2D example, generalized to N-D)
--------------------------------------------

For an ``(nx, ny)`` cell grid:

* **Cell-centered scalars**  (e.g. pressure, density, scalar tracer)
  live on an ``(nx, ny)`` array at [Location.CELL_CENTER][diff_soup.domains.fields.Location.CELL_CENTER].
* **x-face values**  (e.g. x-component velocity ``u_x``, x-direction flux)
  live on ``(nx + 1, ny)`` at [Location.FACE_X][diff_soup.domains.fields.Location.FACE_X] — one extra slot along
  axis 0 because there are ``nx + 1`` x-faces.
* **y-face values**  live on ``(nx, ny + 1)`` at [Location.FACE_Y][diff_soup.domains.fields.Location.FACE_Y].
* **z-face values**  (3D only) on ``(nx, ny, nz + 1)`` at
  [Location.FACE_Z][diff_soup.domains.fields.Location.FACE_Z].

No "half-cell" or "ghost" slots are added: boundary handling belongs to the
operators that act on the face values.

Design
------
[StaggeredGrid][diff_soup.domains.staggered_grid.StaggeredGrid] is additive: it **wraps** an existing ``GridND``
rather than replacing it, so the cell-center metric factors
(``cell_volumes``, ``face_areas``, ``cell_widths``) are reused verbatim from
the underlying ``Grid1D`` axes.  Existing physics modules that accept
``GridND`` continue to work unchanged when passed ``staggered.cell_grid``.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Array

from diff_soup.domains.base import Domain
from diff_soup.domains.fields import Location
from diff_soup.domains.grid import GridND, cartesian_grid_2d, cartesian_grid_3d

__all__ = [
    "StaggeredGrid",
    "staggered_grid_2d",
    "staggered_grid_3d",
]


class StaggeredGrid(Domain):
    """A marker-and-cell (MAC) grid, with scalars at cell centers and velocities on cell faces.

    Each velocity component lives on the faces normal to its axis, so its array has one extra entry along that axis. The built-in operator bank has no staggered stencils, so apply its operators on `cell_grid` or register a bank that implements them.

    Attributes:
        cell_grid: The [GridND][diff_soup.domains.GridND] of cell centers.
    """

    cell_grid: GridND  # traced pytree child — carries cell-center metric factors

    # ------------------------------------------------------------------
    # Domain interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Spatial dimension inherited from the underlying cell grid."""
        return self.cell_grid.dim

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of cells along each axis."""
        return self.cell_grid.shape

    @property
    def num_nodes(self) -> int:
        """Number of cell centers (conventional 'nodes' for staggered layouts).

        Face-centered variables have their own, larger, array shapes: see
        [face_shape][diff_soup.domains.staggered_grid.StaggeredGrid.face_shape].
        """
        return self.cell_grid.num_nodes

    @property
    def discretization_type(self) -> str:
        """Always ``"fvm"`` — staggered grids are a MAC-style FVM layout.

        The tag names the layout family, not a set of available operators:
        the built-in operator bank carries no MAC stencils, so ``laplacian``,
        ``div_flux``, ``gradient``, ``divergence``, ``lumped_mass``, and
        ``advect_upwind`` all refuse a ``StaggeredGrid`` and point at
        ``cell_grid`` (the collocated cell-center grid) instead.  Pointwise
        operators and [face_shape][diff_soup.domains.staggered_grid.StaggeredGrid.face_shape]
        work as usual.
        """
        return "fvm"

    def node_coordinates(self) -> Array:
        """Cell-center coordinates flattened to ``(num_nodes, dim)``."""
        return self.cell_grid.node_coordinates()

    # ------------------------------------------------------------------
    # Face shape queries
    # ------------------------------------------------------------------

    def face_shape(self, location: Location) -> tuple[int, ...]:
        """Return the array shape of the values stored at `location`.

        Args:
            location: A `Location` from `diff_soup.domains.fields`, either `CELL_CENTER` or a face location `FACE_X`, `FACE_Y` or `FACE_Z` within the grid's dimension.

        Returns:
            `shape` at cell centers, or `shape` with one more entry along the face's axis.
        """
        if location is Location.CELL_CENTER:
            return self.shape
        if location.is_face:
            axis = location.face_axis
            if axis >= self.dim:
                raise ValueError(
                    f"{location} invalid on {self.dim}-D StaggeredGrid "
                    f"(face axis {axis} >= dim {self.dim})"
                )
            s = list(self.shape)
            s[axis] += 1
            return tuple(s)
        raise ValueError(
            f"StaggeredGrid does not support location {location}; "
            f"use CELL_CENTER or FACE_X/FACE_Y/FACE_Z."
        )

    # ------------------------------------------------------------------
    # Sparsity pattern
    # ------------------------------------------------------------------

    def sparsity_pattern(self, n_vars: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centered sparsity pattern compatible with [Domain][diff_soup.domains.base.Domain].

        For the simple case where all ``n_vars`` variables live at
        [Location.CELL_CENTER][diff_soup.domains.fields.Location.CELL_CENTER],
        this delegates to the underlying
        ``GridND.sparsity_pattern`` and gives the standard 5-point (2D) /
        7-point (3D) structured-grid stencil.

        A mixed cell/face pattern (MAC layouts) is a function of the stencils
        that couple the two locations, which this library does not ship:
        build that COO pattern from your own operators' stencil footprint and
        pass it to
        [SparseJacobian][diff_soup.problem._ad.sparse_jac.SparseJacobian].
        """
        return self.cell_grid.sparsity_pattern(n_vars)


# =========================================================================
# Factory functions
# =========================================================================


def staggered_grid_2d(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    method: str = "fvm",
) -> StaggeredGrid:
    """Build a two-dimensional staggered (MAC) grid on `[0, lx] x [0, ly]`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        lx: The length along x.
        ly: The length along y.
        method: The discretization of the underlying `cell_grid`, `"fvm"` or `"fdm"`.

    Returns:
        A [StaggeredGrid][diff_soup.domains.StaggeredGrid].
    """
    return StaggeredGrid(cell_grid=cartesian_grid_2d(nx, ny, lx, ly, method=method))


def staggered_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    *,
    method: str = "fvm",
) -> StaggeredGrid:
    """Build a three-dimensional staggered (MAC) grid on `[0, lx] x [0, ly] x [0, lz]`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        nz: The number of cells along z.
        lx: The length along x.
        ly: The length along y.
        lz: The length along z.
        method: The discretization of the underlying `cell_grid`, `"fvm"` or `"fdm"`.

    Returns:
        A [StaggeredGrid][diff_soup.domains.StaggeredGrid].
    """
    return StaggeredGrid(
        cell_grid=cartesian_grid_3d(nx, ny, nz, lx, ly, lz, method=method),
    )
