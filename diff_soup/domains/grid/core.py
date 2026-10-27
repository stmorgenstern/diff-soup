# 685 LOC -- grid primitives (CoordSystem + Grid1D + GridND + factories for cartesian/cylindrical/spherical 1D-3D); kept colocated to share constructor invariants; splitting deferred.
"""Grid construction for FVM/FDM operators on structured domains.

Provides 1D grid classes with coordinate-aware metric factors
(Cartesian, cylindrical, spherical) and N-D Cartesian tensor-product
grids.  Metric factors are pre-computed at construction time so that
operators never branch on coordinate system at runtime.

All domain types are ``eqx.Module`` subtypes of [Domain][diff_soup.domains.base.Domain] so they
participate in JAX pytree transforms and can be carried as fields of
other modules.

Numeric fields are JAX arrays for JIT compatibility.
Static fields (coord_system, n, _method) are frozen for tracing.
"""

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from diff_soup.domains.base import Domain
from diff_soup.domains.grid._dtype import grid_in_field_dtype

__all__ = [
    # Enums / classes
    "CoordSystem",
    "Grid1D",
    "GridND",
    # Cartesian factories
    "cartesian_grid_1d",
    "cartesian_grid_2d",
    "cartesian_grid_3d",
    # Curvilinear factories
    "cylindrical_grid_1d",
    "spherical_grid_1d",
]


# =========================================================================
# Coordinate system enum
# =========================================================================


class CoordSystem(str, Enum):
    """The coordinate system of a grid or mesh.

    Its members `CARTESIAN`, `CYLINDRICAL` and `SPHERICAL` compare equal to their lowercase names. Cylindrical and spherical domains are one-dimensional, with the radius as their coordinate.
    """

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"


# Exponent ``d`` of each coordinate system's metric: under a change of length
# scale, a face area scales as ``length**d`` and a cell volume as
# ``length**(d + 1)``.
_METRIC_DIMENSION = {
    CoordSystem.CARTESIAN: 0,
    CoordSystem.CYLINDRICAL: 1,
    CoordSystem.SPHERICAL: 2,
}


# =========================================================================
# Grid1D
# =========================================================================


class Grid1D(Domain):
    """A one-dimensional grid of `n` cells, each holding one node.

    Build it with [cartesian_grid_1d][diff_soup.domains.cartesian_grid_1d], [cylindrical_grid_1d][diff_soup.domains.cylindrical_grid_1d] or [spherical_grid_1d][diff_soup.domains.spherical_grid_1d].

    Attributes:
        nodes: The node positions, one per cell.
        faces: The face positions, `n + 1` of them.
        cell_widths: The width of each cell.
        node_distances: The distances between neighbouring nodes, `n - 1` of them.
        cell_volumes: The volume of each cell, per radian or per steradian on a curvilinear grid.
        face_areas: The area of each face, per radian or per steradian on a curvilinear grid.
        coord_system: The [CoordSystem][diff_soup.domains.CoordSystem] of the grid.
        n: The number of cells.
    """

    # Core geometry
    nodes: Float[Array, "n"]
    faces: Float[Array, "n_plus_1"]
    cell_widths: Float[Array, "n"]
    node_distances: Float[Array, "n_minus_1"]

    # Coordinate-aware metric factors
    cell_volumes: Float[Array, "n"]
    face_areas: Float[Array, "n_plus_1"]

    # FDM support -----------------------------------------------------------
    metric_gradient_correction: Float[Array, "n"]
    lhopital_extra_d2: Float[Array, "n"]

    # Metadata (static for JIT)
    coord_system: CoordSystem = eqx.field(static=True)
    n: int = eqx.field(static=True)
    _method: str = eqx.field(static=True, default="fvm")

    # ------------------------------------------------------------------
    # Domain interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """[Grid1D][diff_soup.domains.grid.core.Grid1D] spatial dimension: always ``1``."""
        return 1

    @property
    def num_nodes(self) -> int:
        """Number of cell centers on the 1D grid."""
        return self.n

    @property
    def discretization_type(self) -> str:
        """``"fvm"`` or ``"fdm"`` (set at construction via ``method``)."""
        return self._method

    def node_coordinates(self) -> Array:
        """Reshape node coordinates for [Domain][diff_soup.domains.base.Domain] uniformity.

        Returns an array of shape ``(n, 1)``.
        """
        return self.nodes[:, None]  # (n, 1)

    def in_field_dtype(self, field: Float[Array, "..."]) -> "Grid1D":
        """This grid with its floating-point geometry in the dtype of ``field``.

        The stored geometry is double precision; a single-precision field is
        discretized, and handed its coordinates, in its own dtype
        ([grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype]).
        """
        return grid_in_field_dtype(self, field)

    def scaled(self, factor: ArrayLike) -> "Grid1D":
        """Return the grid with every length multiplied by `factor`.

        Face areas and cell volumes scale with the metric, so `spherical_grid_1d(n, 1.0).scaled(R)` matches `spherical_grid_1d(n, R)`. `factor` may be a traced value, which is how `spec.geometry` makes the size of a grid a parameter.

        Args:
            factor: The length scale.

        Returns:
            A [Grid1D][diff_soup.domains.Grid1D] with the same cells, coordinate system and discretization.
        """
        factor = jnp.asarray(factor)
        d = _METRIC_DIMENSION[self.coord_system]
        return Grid1D(
            nodes=self.nodes * factor,
            faces=self.faces * factor,
            cell_widths=self.cell_widths * factor,
            node_distances=self.node_distances * factor,
            cell_volumes=self.cell_volumes * factor ** (d + 1),
            face_areas=self.face_areas * factor**d if d else self.face_areas,
            metric_gradient_correction=self.metric_gradient_correction / factor,
            lhopital_extra_d2=self.lhopital_extra_d2,
            coord_system=self.coord_system,
            n=self.n,
            _method=self._method,
        )

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Declared spatial extent, including boundary half-cells for FVM."""
        coordinates = self.faces if self._method == "fvm" else self.nodes
        return ((float(coordinates[0]), float(coordinates[-1])),)

    def sparsity_pattern(
        self, n_vars: int = 1, *, periodic: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """COO tridiagonal sparsity pattern, block-expanded for ``n_vars`` DOFs.

        Args:
            n_vars: DOFs per cell; for ``n_vars > 1`` the pattern is expanded
                to the dense cross-variable coupling with lexicographic
                ``(row, col)`` ordering.
            periodic: If ``True``, add the wrap coupling between the first and
                last node/cell (the grid is periodic on its one axis).
        """
        return _radius_one_pattern((self.n,), (0,) if periodic else (), n_vars)

    # ------------------------------------------------------------------
    # Family API
    # ------------------------------------------------------------------

    def lumped_mass(self) -> Array:
        """Return the lumped mass (cell-volume) weights for this grid.

        These are the per-cell finite-volume weights ``cell_volumes`` (for a
        spherical grid the shell weights ``(r_{i+1}^3 - r_i^3) / 3``).  They are
        used for member mass-diagonal weighting so that a lumped mass matrix is
        simply ``diag(lumped_mass())``.

        Returns:
            The ``(n,)`` array of positive cell-volume weights.
        """
        return self.cell_volumes

    def nodes_for_tag(self, tag: str) -> list[int]:
        """Return the node indices belonging to a named boundary tag.

        Args:
            tag: Boundary tag name.  Only ``"surface"`` (the outer ``r = R``
                node) is supported on a 1D grid.

        Returns:
            A list of node indices for the tag; ``[n - 1]`` for ``"surface"``.

        Raises:
            ValueError: If ``tag`` is not a supported boundary tag.
        """
        if tag == "surface":
            return [self.n - 1]
        raise ValueError(
            f"Grid1D.nodes_for_tag: unknown boundary tag {tag!r}; supported tags: 'surface'"
        )


# =========================================================================
# Grid1D internal helpers
# =========================================================================

_EPS = 1e-30  # guard for 1/r at origin


def _build_faces_and_nodes(
    n: int,
    x_min: float,
    x_max: float,
    half_cell_left: bool,
    half_cell_right: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (faces, nodes) arrays for a 1D grid."""
    if not half_cell_left and not half_cell_right:
        dx = (x_max - x_min) / n
        faces = jnp.linspace(x_min, x_max, n + 1)
        nodes = x_min + dx * (jnp.arange(n) + 0.5)

    elif not half_cell_left and half_cell_right:
        dx = (x_max - x_min) / (n - 0.5)
        interior = x_min + dx * jnp.arange(n)
        faces = jnp.concatenate([interior, jnp.array([x_max])])
        centers = 0.5 * (faces[:-1] + faces[1:])
        nodes = centers.at[n - 1].set(x_max)

    elif half_cell_left and not half_cell_right:
        dx = (x_max - x_min) / (n - 0.5)
        right_faces = x_min + dx / 2 + dx * jnp.arange(n - 1)
        faces = jnp.concatenate([jnp.array([x_min]), right_faces, jnp.array([x_max])])
        centers = 0.5 * (faces[:-1] + faces[1:])
        nodes = centers.at[0].set(x_min)

    else:
        dx = (x_max - x_min) / (n - 1)
        inner = x_min + dx / 2 + dx * jnp.arange(n - 1)
        faces = jnp.concatenate([jnp.array([x_min]), inner, jnp.array([x_max])])
        centers = 0.5 * (faces[:-1] + faces[1:])
        nodes = centers.at[0].set(x_min).at[n - 1].set(x_max)

    return faces, nodes


# =========================================================================
# Cartesian factory
# =========================================================================


def cartesian_grid_1d(
    n: int | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    *,
    faces: ArrayLike | None = None,
    half_cell_left: bool = False,
    half_cell_right: bool = False,
    method: str = "fvm",
) -> Grid1D:
    """Build a one-dimensional Cartesian grid on `[x_min, x_max]`.

    Pass `n`, `x_min` and `x_max` for a uniform grid, or `faces=` alone for explicit face positions. Each node sits at the center of its cell. A half cell, on the uniform grid only, puts its node on the boundary.

    Args:
        n: The number of cells.
        x_min: The left end.
        x_max: The right end, greater than `x_min`.
        faces: Strictly increasing face positions, at least two, in place of `n`, `x_min` and `x_max`.
        half_cell_left: If `True`, the first cell is a half cell with its node at `x_min`.
        half_cell_right: If `True`, the last cell is a half cell with its node at `x_max`.
        method: The discretization, `"fvm"` or `"fdm"`.

    Returns:
        A [Grid1D][diff_soup.domains.Grid1D].
    """
    if method not in ("fvm", "fdm"):
        raise ValueError(f"cartesian_grid_1d: method must be 'fvm' or 'fdm', got {method!r}")
    if faces is not None:
        if n is not None or x_min is not None or x_max is not None:
            raise ValueError(
                "cartesian_grid_1d: faces= fixes the cell count and the domain "
                "extent, so n, x_min and x_max must be omitted; pass either faces= "
                "alone or n, x_min, x_max"
            )
        if half_cell_left or half_cell_right:
            raise ValueError(
                "cartesian_grid_1d: half_cell_left / half_cell_right apply only to "
                "the uniform (n, x_min, x_max) grid; with faces= place the face "
                "positions directly"
            )
        faces_np = np.asarray(faces, dtype=np.float64)
        if faces_np.ndim != 1 or faces_np.shape[0] < 2:
            raise ValueError(
                "cartesian_grid_1d: faces must be a 1-D array of at least two face "
                f"positions, got shape {faces_np.shape}"
            )
        if not np.all(np.diff(faces_np) > 0.0):
            raise ValueError(
                "cartesian_grid_1d: faces must be strictly increasing (every cell "
                f"width positive), got faces={faces_np}"
            )
        n = faces_np.shape[0] - 1
        nodes_np = 0.5 * (faces_np[:-1] + faces_np[1:])
        # Geometry is formed in double precision, then stored as JAX arrays.
        faces, nodes = jnp.asarray(faces_np), jnp.asarray(nodes_np)
        cell_widths = jnp.asarray(faces_np[1:] - faces_np[:-1])
        node_distances = jnp.asarray(nodes_np[1:] - nodes_np[:-1])
    else:
        if n is None or x_min is None or x_max is None:
            raise ValueError(
                "cartesian_grid_1d: expected n, x_min and x_max for a uniform grid "
                "(or faces= alone for explicit face positions)"
            )
        if int(n) < 1:
            raise ValueError(f"cartesian_grid_1d: n must be a positive integer, got n={n}")
        if float(x_max) <= float(x_min):
            raise ValueError(
                f"cartesian_grid_1d: expected x_max > x_min, got x_min={x_min}, x_max={x_max}"
            )
        faces, nodes = _build_faces_and_nodes(n, x_min, x_max, half_cell_left, half_cell_right)
        cell_widths = faces[1:] - faces[:-1]
        node_distances = nodes[1:] - nodes[:-1]

    return Grid1D(
        nodes=nodes,
        faces=faces,
        cell_widths=cell_widths,
        node_distances=node_distances,
        cell_volumes=cell_widths,
        face_areas=jnp.ones(n + 1),
        metric_gradient_correction=jnp.zeros(n),
        lhopital_extra_d2=jnp.zeros(n),
        coord_system=CoordSystem.CARTESIAN,
        n=n,
        _method=method,
    )


# =========================================================================
# Cylindrical factory
# =========================================================================


def cylindrical_grid_1d(
    n: int,
    r_max: float,
    *,
    r_min: float = 0.0,
    half_cell_right: bool = True,
    method: str = "fvm",
) -> Grid1D:
    """Build a radial grid in cylindrical coordinates on `[r_min, r_max]`.

    Cell volumes and face areas carry the metric, per radian, so the axis at `r = 0` needs no special treatment. With `r_min > 0` the first node sits at `r_min`.

    Args:
        n: The number of cells.
        r_max: The outer radius.
        r_min: The inner radius.
        half_cell_right: If `True`, the last cell is a half cell with its node at `r_max`, the surface.
        method: The discretization, `"fvm"` or `"fdm"`.

    Returns:
        A [Grid1D][diff_soup.domains.Grid1D].
    """
    half_cell_left = r_min > 0.0
    faces, nodes = _build_faces_and_nodes(
        n,
        r_min,
        r_max,
        half_cell_left=half_cell_left,
        half_cell_right=half_cell_right,
    )
    cell_widths = faces[1:] - faces[:-1]
    node_distances = nodes[1:] - nodes[:-1]

    cell_volumes = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    face_areas = faces

    safe_nodes = jnp.where(nodes > _EPS, nodes, 1.0)
    mgc = jnp.where(nodes > _EPS, 1.0 / safe_nodes, 0.0)
    lhd = jnp.where(nodes > _EPS, 0.0, 1.0)

    return Grid1D(
        nodes=nodes,
        faces=faces,
        cell_widths=cell_widths,
        node_distances=node_distances,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        metric_gradient_correction=mgc,
        lhopital_extra_d2=lhd,
        coord_system=CoordSystem.CYLINDRICAL,
        n=n,
        _method=method,
    )


# =========================================================================
# Spherical factory
# =========================================================================


def spherical_grid_1d(
    n: int,
    r_max: float,
    *,
    r_min: float = 0.0,
    half_cell_right: bool = True,
    method: str = "fvm",
) -> Grid1D:
    """Build a radial grid in spherical coordinates on `[r_min, r_max]`.

    Cell volumes and face areas carry the metric, per steradian, so the center at `r = 0` needs no special treatment. With `r_min > 0` the first node sits at `r_min`.

    Args:
        n: The number of cells.
        r_max: The outer radius.
        r_min: The inner radius.
        half_cell_right: If `True`, the last cell is a half cell with its node at `r_max`, the surface.
        method: The discretization, `"fvm"` or `"fdm"`.

    Returns:
        A [Grid1D][diff_soup.domains.Grid1D].
    """
    half_cell_left = r_min > 0.0
    faces, nodes = _build_faces_and_nodes(
        n,
        r_min,
        r_max,
        half_cell_left=half_cell_left,
        half_cell_right=half_cell_right,
    )
    cell_widths = faces[1:] - faces[:-1]
    node_distances = nodes[1:] - nodes[:-1]

    cell_volumes = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
    face_areas = faces**2

    safe_nodes = jnp.where(nodes > _EPS, nodes, 1.0)
    mgc = jnp.where(nodes > _EPS, 2.0 / safe_nodes, 0.0)
    lhd = jnp.where(nodes > _EPS, 0.0, 2.0)

    return Grid1D(
        nodes=nodes,
        faces=faces,
        cell_widths=cell_widths,
        node_distances=node_distances,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
        metric_gradient_correction=mgc,
        lhopital_extra_d2=lhd,
        coord_system=CoordSystem.SPHERICAL,
        n=n,
        _method=method,
    )


# =========================================================================
# GridND (tensor-product Cartesian)
# =========================================================================


class GridND(Domain):
    """A tensor-product Cartesian grid in two or three dimensions.

    Build it with [cartesian_grid_2d][diff_soup.domains.cartesian_grid_2d] or [cartesian_grid_3d][diff_soup.domains.cartesian_grid_3d]. A field on it is a flat array of `num_nodes` values in row-major order over `shape`.

    Attributes:
        axes: One [Grid1D][diff_soup.domains.Grid1D] per axis.
    """

    axes: tuple  # tuple[Grid1D, ...] — pytree children (Grid1D is eqx.Module)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Declared lower and upper extent along each coordinate axis."""
        return tuple(axis.bounds[0] for axis in self.axes)

    @property
    def dim(self) -> int:
        """Number of grid axes."""
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of cells along each axis."""
        return tuple(ax.n for ax in self.axes)

    @property
    def num_nodes(self) -> int:
        """Total cell count (product of per-axis cell counts)."""
        r = 1
        for ax in self.axes:
            r *= ax.n
        return r

    @property
    def discretization_type(self) -> str:
        """Inherited from the first axis (all axes share ``method``)."""
        return self.axes[0]._method

    @property
    def cell_volumes(self) -> Array:
        """Per-cell volumes, flattened row-major: shape ``(num_nodes,)``.

        The grid is a tensor product, so a cell's volume is the product of its
        axes' cell volumes at that cell's per-axis index: the cell widths on a
        Cartesian axis, and the coordinate-aware shell weights on a cylindrical
        or spherical one.
        """
        volume = self.axes[0].cell_volumes
        for ax in self.axes[1:]:
            volume = volume[..., None] * ax.cell_volumes
        return volume.reshape(-1)

    @property
    def outer_face_areas(self) -> Array:
        """Boundary-face area exposed by each cell, flattened row-major.

        A cell exposes one face for every axis on which it is the first or the
        last cell; that face's area is the axis' own face area there (``1`` on a
        Cartesian axis, the coordinate-aware factor on a curvilinear one) times
        the cell volumes of the remaining axes.  Cells that expose several faces
        — corners and edges, and a cell on a one-cell axis, which is both first
        and last — report the sum of their exposed areas, and interior cells
        report ``0``.

        Dividing this by [cell_volumes][diff_soup.domains.grid.core.GridND.cell_volumes]
        gives the finite-volume factor that turns a flux density through the
        boundary into the contribution to a cell's residual row; the areas
        themselves are the surface measure a boundary integral weights by.

        Returns:
            The ``(num_nodes,)`` array of exposed boundary-face areas.
        """
        ndim = len(self.axes)
        total = None
        for d, axis in enumerate(self.axes):
            exposed = (
                jnp.zeros_like(axis.cell_volumes)
                .at[0]
                .add(axis.face_areas[0])
                .at[-1]
                .add(axis.face_areas[-1])
            )
            term = None
            for e, other in enumerate(self.axes):
                factor = exposed if e == d else other.cell_volumes
                factor = factor.reshape(tuple(-1 if k == e else 1 for k in range(ndim)))
                term = factor if term is None else term * factor
            total = term if total is None else total + term
        return total.reshape(-1)

    def node_coordinates(self) -> Array:
        """Return the coordinates of every node, in row-major order over `shape`.

        Returns:
            An array of shape `(num_nodes, dim)`.
        """
        grids = jnp.meshgrid(*[ax.nodes for ax in self.axes], indexing="ij")
        return jnp.stack([g.ravel() for g in grids], axis=-1)

    def in_field_dtype(self, field: Float[Array, "..."]) -> "GridND":
        """This grid with the geometry of every axis in the dtype of ``field``.

        The stored geometry is double precision; a single-precision field is
        discretized, and handed its coordinates, in its own dtype
        ([grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype]).
        """
        return grid_in_field_dtype(self, field)

    def sparsity_pattern(
        self, n_vars: int = 1, *, periodic_axes: tuple[int, ...] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """COO pattern for the structured-grid stencil, block-expanded for ``n_vars``.

        Each cell couples to itself and to its two neighbours along every axis
        (5-point in 2D, 7-point in 3D); the pattern is replicated across the
        variable-variable coupling blocks.

        Args:
            n_vars: DOFs per node; see
                [Grid1D.sparsity_pattern]
                [diff_soup.domains.grid.core.Grid1D.sparsity_pattern].
            periodic_axes: Axes on which the grid is periodic; each gets the
                wrap coupling between its first and last hyperplane.
        """
        return _radius_one_pattern(self.shape, tuple(periodic_axes), n_vars)


def cartesian_grid_2d(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    method: str = "fvm",
) -> GridND:
    """Build a two-dimensional Cartesian grid on `[0, lx] x [0, ly]`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        lx: The length along x.
        ly: The length along y.
        method: The discretization, `"fvm"` or `"fdm"`.

    Returns:
        A [GridND][diff_soup.domains.GridND].
    """
    return GridND(
        axes=(
            cartesian_grid_1d(nx, 0.0, lx, method=method),
            cartesian_grid_1d(ny, 0.0, ly, method=method),
        ),
    )


def cartesian_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    *,
    method: str = "fvm",
) -> GridND:
    """Build a three-dimensional Cartesian grid on `[0, lx] x [0, ly] x [0, lz]`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        nz: The number of cells along z.
        lx: The length along x.
        ly: The length along y.
        lz: The length along z.
        method: The discretization, `"fvm"` or `"fdm"`.

    Returns:
        A [GridND][diff_soup.domains.GridND].
    """
    return GridND(
        axes=(
            cartesian_grid_1d(nx, 0.0, lx, method=method),
            cartesian_grid_1d(ny, 0.0, ly, method=method),
            cartesian_grid_1d(nz, 0.0, lz, method=method),
        ),
    )


# =========================================================================
# Sparsity helpers (host-side NumPy, not traced)
# =========================================================================


def stencil_sparsity_nd(
    shape: tuple[int, ...],
    offsets: np.ndarray,
    periodic_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """COO sparsity pattern generated from an explicit stencil offset set.

    Each grid cell couples to ``cell + offset`` for every row of ``offsets``,
    so an operator whose true dependency reach is wider than the built-in
    radius-1 stencil (e.g. a composed laplacian) can declare its exact
    topology.  On axes listed in ``periodic_axes`` the neighbour index wraps
    around the domain edge; on all other axes couplings that fall outside the
    grid are dropped.

    Args:
        shape: Per-axis cell counts of the structured grid (row-major cell
            flattening, matching the field layout).
        offsets: Integer array of shape ``(n_offsets, ndim)``; each row is a
            per-axis cell offset of one stencil point.
        periodic_axes: Axes whose offsets wrap around the domain edge.

    Returns:
        ``(rows, cols)`` int32 COO arrays.  Entries may contain duplicates
        (e.g. distinct offsets wrapping onto the same cell on a small periodic
        grid); callers dedup with their own pass.
    """
    shape_arr = np.asarray(shape, dtype=np.int64)
    ndim = int(shape_arr.size)
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, ndim)
    total = int(np.prod(shape_arr))
    if offsets.shape[0] == 0 or total == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    strides = np.ones(ndim, dtype=np.int64)
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape_arr[d + 1]

    multi = np.empty((total, ndim), dtype=np.int64)
    tmp = np.arange(total, dtype=np.int64)
    for d in range(ndim):
        multi[:, d] = tmp // strides[d]
        tmp = tmp % strides[d]

    periodic = np.zeros(ndim, dtype=bool)
    for d in periodic_axes:
        periodic[int(d)] = True

    # (n_offsets, total, ndim) neighbour multi-indices: wrap periodic axes,
    # mask out-of-range cells on the others.
    nbr = multi[None, :, :] + offsets[:, None, :]
    valid = np.ones(nbr.shape[:2], dtype=bool)
    for d in range(ndim):
        if periodic[d]:
            nbr[:, :, d] %= shape_arr[d]
        else:
            in_range = (nbr[:, :, d] >= 0) & (nbr[:, :, d] < shape_arr[d])
            valid &= in_range
            nbr[:, :, d] = np.clip(nbr[:, :, d], 0, shape_arr[d] - 1)

    rows = np.broadcast_to(np.arange(total, dtype=np.int64)[None, :], valid.shape)[valid]
    cols = (nbr @ strides)[valid]
    return rows.astype(np.int32), cols.astype(np.int32)


def _radius_one_pattern(
    shape: tuple[int, ...], periodic_axes: tuple[int, ...], n_vars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Radius-one structured-grid pattern, deduplicated, block-expanded for ``n_vars``.

    Every cell couples to itself and to ``cell +- 1`` along each axis; on the
    axes in ``periodic_axes`` the neighbour wraps around the domain edge. For
    ``n_vars > 1`` each cell pair ``(i, j)`` expands to every variable pair,
    DOF ``n_vars * cell + var``. The result is sorted by ``(row, col)``.
    """
    ndim = len(shape)
    eye = np.eye(ndim, dtype=np.int64)
    offsets = np.concatenate([np.zeros((1, ndim), dtype=np.int64), eye, -eye])
    rows_1, cols_1 = stencil_sparsity_nd(shape, offsets, periodic_axes=periodic_axes)
    pairs = np.unique(np.stack((rows_1, cols_1), axis=1), axis=0)
    rows_1, cols_1 = pairs[:, 0], pairs[:, 1]
    if n_vars == 1:
        return rows_1, cols_1
    var = np.arange(n_vars, dtype=np.int32)
    rows = n_vars * rows_1[:, None, None] + var[None, :, None]
    cols = n_vars * cols_1[:, None, None] + var[None, None, :]
    shape_3 = (rows_1.size, n_vars, n_vars)
    rows = np.broadcast_to(rows, shape_3).reshape(-1)
    cols = np.broadcast_to(cols, shape_3).reshape(-1)
    order = np.lexsort((cols, rows))
    return rows[order], cols[order]
