"""domains/mesh.py — Unstructured mesh domains for FEM.

Provides [Mesh][diff_soup.domains.mesh.core.Mesh] and
[TaggedMesh][diff_soup.domains.mesh.core.TaggedMesh] as ``eqx.Module`` subtypes
of [Domain][diff_soup.domains.base.Domain].  All fields are static (topology is
never differentiated).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import equinox as eqx
import numpy as np

from diff_soup.domains.base import Domain
from diff_soup.domains.grid.core import CoordSystem

# Power of the radius in the volume element ``r**p dr`` of each coordinate
# system: Cartesian ``dr``, cylindrical ``r dr`` (per radian), spherical
# ``r**2 dr`` (per steradian).  The same convention as the radial grid
# factories, so a finite-volume and a finite-element mesh of one domain
# measure the same total.
_METRIC_EXPONENT = {
    CoordSystem.CARTESIAN: 0,
    CoordSystem.CYLINDRICAL: 1,
    CoordSystem.SPHERICAL: 2,
}


def metric_exponent(coord_system) -> int:
    """Return ``p`` in the volume element ``r**p dr`` for a coordinate system.

    Args:
        coord_system: A [CoordSystem][diff_soup.domains.grid.core.CoordSystem],
            or ``None`` for a domain that carries no coordinate system.

    Returns:
        ``0`` for Cartesian (and for ``None``), ``1`` for cylindrical, ``2``
        for spherical.
    """
    if coord_system is None:
        return 0
    return _METRIC_EXPONENT[CoordSystem(coord_system)]


class Mesh(Domain):
    """A finite-element mesh given by its node coordinates and element connectivity.

    Build one from your own points and cells, or with a mesh factory. A mesh always discretizes by finite elements.

    Args:
        points: The node coordinates, shape `(num_nodes, dim)`.
        cells: The node indices of each element, one row per element.
        ele_type: The element type, such as `"LINE2"`, `"TRI3"`, `"QUAD4"`, `"TET4"` or `"HEX8"`.
        periodic_axes: Axes whose opposite boundaries are identified as periodic, in two or more dimensions.
        coord_system: The [CoordSystem][diff_soup.domains.CoordSystem], which must be Cartesian unless the mesh is one-dimensional.
    """

    points: np.ndarray
    cells: np.ndarray
    ele_type: str = eqx.field(static=True)
    periodic_axes: tuple = eqx.field(static=True, default=())
    coord_system: CoordSystem = eqx.field(static=True, default=CoordSystem.CARTESIAN)
    # Derived finite-element spaces are constructed outside this immutable
    # topology container: a mutable cache on a frozen pytree silently goes
    # stale. Build spaces via make_fe_space instead.

    def __init__(
        self,
        points,
        cells,
        ele_type: str,
        periodic_axes=(),
        coord_system: CoordSystem = CoordSystem.CARTESIAN,
    ):
        """Build a mesh from node coordinates and cell connectivity.

        Inputs are coerced to ``numpy`` arrays so the topology is carried on
        the host and never traced by JAX; JIT keys on shape/dtype only.

        Raises:
            ValueError: If a curvilinear ``coord_system`` is given for a mesh
                that is not one-dimensional.
        """
        # Coerce to numpy so mesh topology lives on the host and is never
        # traced.  JIT keys on shape/dtype, not values.
        self.points = np.asarray(points)
        self.cells = np.asarray(cells)
        self.ele_type = ele_type
        self.periodic_axes = tuple(periodic_axes)
        self.coord_system = CoordSystem(coord_system)
        if self.coord_system is not CoordSystem.CARTESIAN and self.points.shape[1] != 1:
            raise ValueError(
                f"a {self.coord_system.value} mesh must be one-dimensional, so that its "
                f"one coordinate is the radius; got points of dimension "
                f"{self.points.shape[1]}. Use a Cartesian mesh, or mesh the radial "
                f"coordinate alone."
            )

    # ------------------------------------------------------------------
    # Domain interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Spatial dimension (second axis of ``points``)."""
        return int(self.points.shape[1])

    @property
    def num_nodes(self) -> int:
        """Number of mesh nodes (first axis of ``points``)."""
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of elements (first axis of ``cells``)."""
        return int(self.cells.shape[0])

    @property
    def discretization_type(self) -> str:
        """Always ``"fem"`` — unstructured meshes route to FEM backends."""
        return "fem"

    def node_coordinates(self) -> np.ndarray:
        """Return node coordinates as a ``(num_nodes, dim)`` array."""
        return self.points

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Minimum and maximum node coordinate along each spatial dimension."""
        return tuple((float(axis.min()), float(axis.max())) for axis in self.points.T)

    def sparsity_pattern(self, n_vars: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return COO ``(rows, cols)`` for the DOF stiffness sparsity.

        Args:
            n_vars: Number of DOFs per node (1 for scalar problems, ``dim``
                for vector-valued elasticity, etc.).
        """
        from diff_soup.domains.mesh.fem.assembly import build_dof_sparsity

        return build_dof_sparsity(
            np.asarray(self.cells, dtype=np.int32),
            self.num_nodes,
            n_vars,
        )

    def _edges_in_one_element(self) -> np.ndarray:
        """Edges belonging to exactly one element, as sorted node pairs."""
        cells_np = np.asarray(self.cells, dtype=np.int64)
        et = self.ele_type.upper()
        if et in ("TRI3", "TRI6"):
            n_edges = 3
        elif et in ("QUAD4", "QUAD8", "QUAD9"):
            n_edges = 4
        else:
            raise NotImplementedError(f"boundary_edges not implemented for element type {et!r}")
        edge_count: dict = {}
        for cell in cells_np:
            nodes = cell.tolist()
            for i in range(n_edges):
                j = (i + 1) % n_edges
                edge = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        boundary = [e for e, cnt in edge_count.items() if cnt == 1]
        if not boundary:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(sorted(boundary), dtype=np.int32)

    def _vertices_in_one_element(self) -> np.ndarray:
        """Vertices belonging to exactly one element, one per row."""
        nodes, counts = np.unique(np.asarray(self.cells, dtype=np.int64), return_counts=True)
        return nodes[counts == 1].astype(np.int32).reshape(-1, 1)

    def boundary_edges(self) -> np.ndarray:
        """Detect boundary edges (edges in exactly one element).

        Returns:
            Shape ``(n_boundary_edges, 2)`` int32 array of node index
            pairs.  Works for 2D elements (TRI3, TRI6, QUAD4, QUAD8,
            QUAD9).
        """
        return self._edges_in_one_element()

    def boundary_facets(self) -> np.ndarray:
        """Detect the facets on the mesh boundary, one facet per row.

        A facet is one dimension below the element, so its width follows the
        mesh's dimension: a single node in 1D, a node pair in 2D.

        Returns:
            Shape ``(n_boundary_facets, n_nodes_per_facet)`` int32 array.

        Raises:
            NotImplementedError: If the element type has no facet enumeration
                (3D element types are enumerated by the mesh generator and
                carried on a
                [TaggedMesh][diff_soup.domains.mesh.core.TaggedMesh] instead).
        """
        if self.dim == 1:
            return self._vertices_in_one_element()
        return self._edges_in_one_element()

    def boundary_nodes(self) -> np.ndarray:
        """Return unique sorted node indices on the mesh boundary.

        Boundary nodes are identified by facets appearing in exactly one
        element.
        """
        facets = self.boundary_facets()
        if facets.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(facets.ravel()).astype(np.int32)


def _line_mesh(
    n_cells: int,
    lo: float,
    hi: float,
    *,
    with_boundary_tags: bool,
    tag_names: Dict[int, str],
    coord_system: CoordSystem,
    factory: str,
) -> Mesh:
    """Build the shared LINE2 node grid, cells, and end-point facet table."""
    if int(n_cells) < 1:
        raise ValueError(f"{factory}: n_cells must be a positive integer, got {n_cells}")

    n_cells = int(n_cells)
    points = np.linspace(float(lo), float(hi), n_cells + 1).reshape(-1, 1)
    idx = np.arange(n_cells + 1)
    cells = np.stack((idx[:-1], idx[1:]), axis=-1).astype(np.int32)

    if not with_boundary_tags:
        return Mesh(points=points, cells=cells, ele_type="LINE2", coord_system=coord_system)

    return TaggedMesh(
        points=points,
        cells=cells,
        ele_type="LINE2",
        boundary_faces=np.array([[0], [n_cells]], dtype=np.int32),
        boundary_tags=np.array([1, 2], dtype=np.int32),
        tag_names=tag_names,
        coord_system=coord_system,
    )


def cylindrical_interval_mesh(
    n_cells: int,
    r_max: float,
    *,
    r_min: float = 0.0,
    with_boundary_tags: bool = True,
) -> Mesh:
    """Build a radial line mesh in cylindrical coordinates on `[r_min, r_max]`.

    Its element measure carries the metric `r dr` per radian, the same metric as [cylindrical_grid_1d][diff_soup.domains.cylindrical_grid_1d]. With tags, the inner and outer ends are named `"center"` and `"surface"`.

    Args:
        n_cells: The number of elements.
        r_max: The outer radius, greater than `r_min`.
        r_min: The inner radius, zero or greater.
        with_boundary_tags: If `True`, return a [TaggedMesh][diff_soup.domains.TaggedMesh] with both ends tagged.

    Returns:
        A [Mesh][diff_soup.domains.Mesh], or a `TaggedMesh` when tagged.
    """
    _validate_radii("cylindrical_interval_mesh", r_min, r_max)
    return _line_mesh(
        n_cells,
        r_min,
        r_max,
        with_boundary_tags=with_boundary_tags,
        tag_names={1: "center", 2: "surface"},
        coord_system=CoordSystem.CYLINDRICAL,
        factory="cylindrical_interval_mesh",
    )


def spherical_interval_mesh(
    n_cells: int,
    r_max: float,
    *,
    r_min: float = 0.0,
    with_boundary_tags: bool = True,
) -> Mesh:
    """Build a radial line mesh in spherical coordinates on `[r_min, r_max]`.

    Its element measure carries the metric `r**2 dr` per steradian, the same metric as [spherical_grid_1d][diff_soup.domains.spherical_grid_1d]. With tags, the inner and outer ends are named `"center"` and `"surface"`.

    Args:
        n_cells: The number of elements.
        r_max: The outer radius, greater than `r_min`.
        r_min: The inner radius, zero or greater.
        with_boundary_tags: If `True`, return a [TaggedMesh][diff_soup.domains.TaggedMesh] with both ends tagged.

    Returns:
        A [Mesh][diff_soup.domains.Mesh], or a `TaggedMesh` when tagged.
    """
    _validate_radii("spherical_interval_mesh", r_min, r_max)
    return _line_mesh(
        n_cells,
        r_min,
        r_max,
        with_boundary_tags=with_boundary_tags,
        tag_names={1: "center", 2: "surface"},
        coord_system=CoordSystem.SPHERICAL,
        factory="spherical_interval_mesh",
    )


def _validate_radii(factory: str, r_min: float, r_max: float) -> None:
    """Reject a negative inner radius or a non-positive radial extent."""
    if float(r_min) < 0.0:
        raise ValueError(f"{factory}: r_min must be zero or greater, got r_min={r_min}")
    if float(r_max) <= float(r_min):
        raise ValueError(f"{factory}: expected r_max > r_min, got r_min={r_min}, r_max={r_max}")


def interval_mesh(
    n_cells: int,
    x_min: float,
    x_max: float,
    *,
    with_boundary_tags: bool = True,
) -> Mesh:
    """Build a one-dimensional mesh of two-node line elements on `[x_min, x_max]`.

    It is the finite-element counterpart of [cartesian_grid_1d][diff_soup.domains.cartesian_grid_1d]. With tags, its end nodes are named `"left"` and `"right"`.

    Args:
        n_cells: The number of elements.
        x_min: The left end.
        x_max: The right end, greater than `x_min`.
        with_boundary_tags: If `True`, return a [TaggedMesh][diff_soup.domains.TaggedMesh] with both ends tagged.

    Returns:
        A [Mesh][diff_soup.domains.Mesh], or a `TaggedMesh` when tagged.
    """
    if float(x_max) <= float(x_min):
        raise ValueError(f"interval_mesh: expected x_max > x_min, got x_min={x_min}, x_max={x_max}")
    return _line_mesh(
        n_cells,
        x_min,
        x_max,
        with_boundary_tags=with_boundary_tags,
        tag_names={1: "left", 2: "right"},
        coord_system=CoordSystem.CARTESIAN,
        factory="interval_mesh",
    )


def rectangle_mesh(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    periodic_axes: tuple[int, ...] = (),
    ele_type: str = "QUAD4",
    with_boundary_tags: bool = False,
) -> Mesh:
    """Build a structured mesh of the rectangle `[0, lx] x [0, ly]`.

    `ele_type="TRI3"` splits each cell into two triangles. With tags, the sides are named `"left"`, `"right"`, `"bottom"` and `"top"`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        lx: The length along x.
        ly: The length along y.
        periodic_axes: Axes whose opposite sides are identified as periodic.
        ele_type: The element type, `"QUAD4"` or `"TRI3"`.
        with_boundary_tags: If `True`, return a [TaggedMesh][diff_soup.domains.TaggedMesh] with the four sides tagged.

    Returns:
        A [Mesh][diff_soup.domains.Mesh], or a `TaggedMesh` when tagged.
    """
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError(
            f"rectangle_mesh: nx and ny must be positive integers, got nx={nx}, ny={ny}"
        )
    if float(lx) <= 0.0 or float(ly) <= 0.0:
        raise ValueError(f"rectangle_mesh: lx and ly must be positive, got lx={lx}, ly={ly}")
    if ele_type not in ("QUAD4", "TRI3"):
        raise ValueError(f"rectangle_mesh: ele_type must be 'QUAD4' or 'TRI3', got {ele_type!r}")

    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    points = np.stack((xv, yv), axis=-1).reshape(-1, 2)
    idx = np.arange(points.shape[0]).reshape(nx + 1, ny + 1)

    quads = np.stack(
        (idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]),
        axis=-1,
    ).reshape(-1, 4)

    if ele_type == "QUAD4":
        cells = quads
    else:  # TRI3 — split each QUAD into 2 triangles
        n0, n1, n2, n3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
        tris_a = np.stack((n0, n1, n2), axis=-1)
        tris_b = np.stack((n0, n2, n3), axis=-1)
        cells = np.concatenate((tris_a, tris_b), axis=0)

    if not with_boundary_tags:
        return Mesh(points=points, cells=cells, ele_type=ele_type, periodic_axes=periodic_axes)

    # Tag boundary edges by axis-aligned position.
    tag_names = {1: "left", 2: "right", 3: "bottom", 4: "top"}
    edges_list = []
    tags_list = []
    # left: x = 0  -> nodes idx[0, :]; edges between successive y nodes
    for j in range(ny):
        edges_list.append((int(idx[0, j]), int(idx[0, j + 1])))
        tags_list.append(1)
    # right: x = lx -> nodes idx[nx, :]
    for j in range(ny):
        edges_list.append((int(idx[nx, j]), int(idx[nx, j + 1])))
        tags_list.append(2)
    # bottom: y = 0 -> nodes idx[:, 0]
    for i in range(nx):
        edges_list.append((int(idx[i, 0]), int(idx[i + 1, 0])))
        tags_list.append(3)
    # top: y = ly -> nodes idx[:, ny]
    for i in range(nx):
        edges_list.append((int(idx[i, ny]), int(idx[i + 1, ny])))
        tags_list.append(4)
    boundary_edges = np.asarray(edges_list, dtype=np.int32)
    boundary_tags = np.asarray(tags_list, dtype=np.int32)
    return TaggedMesh(
        points=points,
        cells=cells,
        ele_type=ele_type,
        boundary_edges=boundary_edges,
        boundary_tags=boundary_tags,
        tag_names=tag_names,
        periodic_axes=periodic_axes,
    )


def box_mesh(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    *,
    periodic_axes: tuple[int, ...] = (),
    ele_type: str = "HEX8",
    with_boundary_tags: bool = False,
) -> Mesh:
    """Build a structured mesh of the box `[0, lx] x [0, ly] x [0, lz]`.

    `ele_type="TET4"` splits each cell into six tetrahedra. Boundary tags, named `"x_min"`, `"x_max"`, `"y_min"`, `"y_max"`, `"z_min"` and `"z_max"`, need `"TET4"`.

    Args:
        nx: The number of cells along x.
        ny: The number of cells along y.
        nz: The number of cells along z.
        lx: The length along x.
        ly: The length along y.
        lz: The length along z.
        periodic_axes: Axes whose opposite faces are identified as periodic.
        ele_type: The element type, `"HEX8"` or `"TET4"`.
        with_boundary_tags: If `True`, return a [TaggedMesh][diff_soup.domains.TaggedMesh] with the six faces tagged.

    Returns:
        A [Mesh][diff_soup.domains.Mesh], or a `TaggedMesh` when tagged.

    Raises:
        ValueError: If `with_boundary_tags=True` with `ele_type="HEX8"`.
    """
    if int(nx) < 1 or int(ny) < 1 or int(nz) < 1:
        raise ValueError(
            f"box_mesh: nx, ny, nz must be positive integers, got nx={nx}, ny={ny}, nz={nz}"
        )
    if float(lx) <= 0.0 or float(ly) <= 0.0 or float(lz) <= 0.0:
        raise ValueError(f"box_mesh: lx, ly, lz must be positive, got lx={lx}, ly={ly}, lz={lz}")
    if ele_type not in ("HEX8", "TET4"):
        raise ValueError(f"box_mesh: ele_type must be 'HEX8' or 'TET4', got {ele_type!r}")

    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    z = np.linspace(0.0, lz, nz + 1)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)
    idx = np.arange(points.shape[0]).reshape(nx + 1, ny + 1, nz + 1)

    hexes = np.stack(
        (
            idx[:-1, :-1, :-1],
            idx[1:, :-1, :-1],
            idx[1:, 1:, :-1],
            idx[:-1, 1:, :-1],
            idx[:-1, :-1, 1:],
            idx[1:, :-1, 1:],
            idx[1:, 1:, 1:],
            idx[:-1, 1:, 1:],
        ),
        axis=-1,
    ).reshape(-1, 8)

    if ele_type == "HEX8":
        cells = hexes
    else:  # TET4 — standard 6-tet decomposition of each hex
        # Hex node ordering: 0=(0,0,0), 1=(1,0,0), 2=(1,1,0), 3=(0,1,0),
        #                    4=(0,0,1), 5=(1,0,1), 6=(1,1,1), 7=(0,1,1)
        # 6-tet split using diagonal 0-6:
        tet_offsets = np.array(
            [
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 1, 6, 5],
                [0, 4, 5, 6],
                [0, 4, 6, 7],
                [0, 6, 3, 7],
            ],
            dtype=np.int64,
        )
        tets = np.take_along_axis(
            hexes[:, None, :].repeat(6, axis=1),
            tet_offsets[None, :, :].repeat(hexes.shape[0], axis=0),
            axis=2,
        ).reshape(-1, 4)
        cells = tets

    if not with_boundary_tags:
        return Mesh(points=points, cells=cells, ele_type=ele_type, periodic_axes=periodic_axes)

    # Tag boundary faces by axis-aligned position.
    tag_names = {1: "x_min", 2: "x_max", 3: "y_min", 4: "y_max", 5: "z_min", 6: "z_max"}
    faces_list = []
    tags_list = []

    def _tri_faces_x(i_const: int, tag: int):
        """Boundary triangles on the x = const face (i fixed)."""
        for j in range(ny):
            for k in range(nz):
                # Quad nodes at this face: idx[i_const, j..j+1, k..k+1]
                a = int(idx[i_const, j, k])
                b = int(idx[i_const, j + 1, k])
                c = int(idx[i_const, j + 1, k + 1])
                d = int(idx[i_const, j, k + 1])
                faces_list.append((a, b, c))
                tags_list.append(tag)
                faces_list.append((a, c, d))
                tags_list.append(tag)

    def _tri_faces_y(j_const: int, tag: int):
        for i in range(nx):
            for k in range(nz):
                a = int(idx[i, j_const, k])
                b = int(idx[i + 1, j_const, k])
                c = int(idx[i + 1, j_const, k + 1])
                d = int(idx[i, j_const, k + 1])
                faces_list.append((a, b, c))
                tags_list.append(tag)
                faces_list.append((a, c, d))
                tags_list.append(tag)

    def _tri_faces_z(k_const: int, tag: int):
        for i in range(nx):
            for j in range(ny):
                a = int(idx[i, j, k_const])
                b = int(idx[i + 1, j, k_const])
                c = int(idx[i + 1, j + 1, k_const])
                d = int(idx[i, j + 1, k_const])
                faces_list.append((a, b, c))
                tags_list.append(tag)
                faces_list.append((a, c, d))
                tags_list.append(tag)

    if ele_type != "TET4":
        raise ValueError(
            "box_mesh: with_boundary_tags is available for ele_type='TET4'. A "
            "HEX8 box would carry an empty boundary-facet table, so no boundary "
            "could be selected by name; build it with ele_type='TET4', or with "
            "with_boundary_tags=False."
        )
    _tri_faces_x(0, 1)  # x_min
    _tri_faces_x(nx, 2)  # x_max
    _tri_faces_y(0, 3)  # y_min
    _tri_faces_y(ny, 4)  # y_max
    _tri_faces_z(0, 5)  # z_min
    _tri_faces_z(nz, 6)  # z_max
    boundary_faces = np.asarray(faces_list, dtype=np.int32)
    boundary_tags = np.asarray(tags_list, dtype=np.int32)
    return TaggedMesh(
        points=points,
        cells=cells,
        ele_type=ele_type,
        boundary_faces=boundary_faces,
        boundary_tags=boundary_tags,
        tag_names=tag_names,
        periodic_axes=periodic_axes,
    )


class TaggedMesh(Mesh):
    """A [Mesh][diff_soup.domains.Mesh] whose boundary facets carry integer tags, optionally named.

    Boundary conditions select facets by name, as `tag="rim"` or with [`ds.tags`][diff_soup.problem.tags]. Facet integrals, such as flux conditions and boundary operators, support `LINE2`, `TRI3` and `TET4` elements.

    Args:
        points: The node coordinates, shape `(num_nodes, dim)`.
        cells: The node indices of each element, one row per element.
        ele_type: The element type.
        boundary_edges: The boundary edges of a two-dimensional mesh, as pairs of node indices.
        boundary_faces: The boundary faces of a three-dimensional mesh as node triples, or the boundary nodes of a one-dimensional mesh.
        boundary_tags: One integer tag per boundary edge or face, in the same order.
        tag_names: The name of each tag, such as `{1: "left"}`.
        tag_groups: Group names that each select several tags, such as `{"boundary": (1, 2)}`.
        periodic_axes: As for [Mesh][diff_soup.domains.Mesh].
        coord_system: As for [Mesh][diff_soup.domains.Mesh].
    """

    boundary_edges: np.ndarray
    boundary_faces: np.ndarray
    boundary_tags: np.ndarray
    tag_names: Optional[Dict[int, str]] = eqx.field(static=True, default=None)
    tag_groups: Optional[Dict[str, Tuple[int, ...]]] = eqx.field(static=True, default=None)

    def __init__(
        self,
        points,
        cells,
        ele_type: str,
        boundary_edges=None,
        boundary_faces=None,
        boundary_tags=None,
        tag_names=None,
        tag_groups=None,
        periodic_axes=(),
        coord_system: CoordSystem = CoordSystem.CARTESIAN,
    ):
        """Build a mesh with physical-group tags on boundary entities.

        Missing ``boundary_edges`` / ``boundary_faces`` / ``boundary_tags`` default
        to empty int32 arrays so the mesh is still valid (just with no tagged
        boundaries).
        """
        super().__init__(
            points, cells, ele_type, periodic_axes=periodic_axes, coord_system=coord_system
        )
        if boundary_edges is None:
            boundary_edges = np.empty((0, 2), dtype=np.int32)
        if boundary_faces is None:
            boundary_faces = np.empty((0, 3), dtype=np.int32)
        if boundary_tags is None:
            boundary_tags = np.empty((0,), dtype=np.int32)
        self.boundary_edges = np.asarray(boundary_edges, dtype=np.int32)
        self.boundary_faces = np.asarray(boundary_faces, dtype=np.int32)
        self.boundary_tags = np.asarray(boundary_tags, dtype=np.int32)
        self.tag_names = tag_names
        self.tag_groups = (
            None
            if tag_groups is None
            else {str(k): tuple(int(t) for t in v) for k, v in tag_groups.items()}
        )

    def _entities_for_ids(self, entities: np.ndarray, ids) -> np.ndarray:
        """Return the rows of *entities* whose tag id is one of *ids*.

        A mesh tags either its edges (2D) or its faces (3D), so the other table
        is empty and carries no entity for any id; selecting from it returns an
        empty array of the same width rather than indexing it with a
        full-length mask.
        """
        if entities.shape[0] != self.boundary_tags.shape[0]:
            return entities[:0]
        return entities[np.isin(self.boundary_tags, list(ids))]

    def boundary_facets(self) -> np.ndarray:
        """Return the whole tagged boundary-facet table, one facet per row.

        A mesh tags either its edges (2D) or its faces (3D and 1D), so this
        returns whichever of the two tables the tags index.

        Returns:
            Shape ``(n_boundary_facets, n_nodes_per_facet)`` int32 array; empty
            when no boundary entity is tagged.
        """
        n_tags = int(self.boundary_tags.shape[0])
        if self.boundary_edges.shape[0] == n_tags and self.boundary_edges.size > 0:
            return self.boundary_edges
        if self.boundary_faces.shape[0] == n_tags and self.boundary_faces.size > 0:
            return self.boundary_faces
        return self.boundary_edges[:0]

    def facets_for_tag(self, tag: int) -> np.ndarray:
        """Return the boundary facets that carry an integer tag.

        Args:
            tag: The integer tag.

        Returns:
            The matching edges, or faces if no edge matches, as rows of node indices, possibly empty.
        """
        return self._facets_for_ids([int(tag)])

    def facets_for_name(self, name: str) -> np.ndarray:
        """Return the boundary facets of a named tag or tag group.

        Args:
            name: A name in `tag_names`, or a key of `tag_groups`.

        Returns:
            The facets of every tag with that name or in that group, as rows of node indices.

        Raises:
            KeyError: If the mesh names no tags, or no tag or group is called `name`.
        """
        return self._facets_for_ids(self._tag_ids_for_name(name))

    def _facets_for_ids(self, ids) -> np.ndarray:
        """Select facets by tag id from whichever facet table is populated."""
        ids = list(ids)
        edges = self._entities_for_ids(self.boundary_edges, ids)
        if edges.shape[0] > 0:
            return edges
        return self._entities_for_ids(self.boundary_faces, ids)

    def boundary_nodes(self) -> np.ndarray:
        """Return unique sorted node indices that lie on tagged boundaries.

        Collects node indices from both ``boundary_edges`` (2D meshes) and
        ``boundary_faces`` (3D meshes), deduplicates, and returns them sorted.

        Returns:
            Shape ``(n_nodes,)`` int32 array.  Empty if no boundary entities
            are tagged.
        """
        parts = []
        if self.boundary_edges.size > 0:
            parts.append(self.boundary_edges.ravel())
        if self.boundary_faces.size > 0:
            parts.append(self.boundary_faces.ravel())
        if not parts:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(parts)).astype(np.int32)

    def _tag_ids_for_name(self, name: str) -> list:
        """Return ALL tag ids named *name*, per-entity names first, then groups.

        Args:
            name: The physical-group name or ``tag_groups`` group name to look
                up.

        Returns:
            List of integer tag ids (one or more).

        Raises:
            KeyError: If the mesh carries no names at all, or *name* is neither
                a tag name nor a group name.
        """
        if self.tag_names is None and self.tag_groups is None:
            raise KeyError(
                f"this mesh names no boundary tags, so {name!r} cannot be resolved; "
                "build it with tag_names (and tag_groups) to select boundaries by name"
            )
        names = self.tag_names or {}
        groups = self.tag_groups or {}
        matching = [tid for tid, nm in names.items() if nm == name]
        if matching:
            return matching
        if name in groups:
            return list(groups[name])
        available = sorted(set(names.values()) | set(groups))
        raise KeyError(f"No boundary tag named {name!r}; available names: {available}")
