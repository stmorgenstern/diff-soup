# 1177 LOC -- image-mask-to-mesh pipeline (skimage shapely loaders + vtk export); kept as one file because callers pull the full pipeline at once; splitting deferred.
"""geometry/meshing.py — Triangulate polygons and masks with gmsh.

Converts boundary polygons (from [mask_to_polygon][diff_soup.domains.geometry.mask.mask_to_polygon])
to triangulated [Mesh][diff_soup.domains.mesh.Mesh] objects ready for FEM
assembly.

Meshing runs on ``gmsh``, which installs with diff_soup.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from diff_soup._optional import import_optional
from diff_soup.domains.geometry.mask import mask_to_polygon
from diff_soup.domains.mesh import Mesh, TaggedMesh


def polygon_to_mesh(
    polygon: List[Tuple[float, float]],
    *,
    mesh_size: float = 1.0,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Triangulate a 2-D boundary polygon using gmsh.

    Args:
        polygon: Ordered list of ``(x, y)`` vertices defining the domain boundary.
            The polygon is automatically closed (last vertex connected to first).
            Must have at least 3 vertices.
        mesh_size: Target element size in physical units.  Smaller values produce finer
            meshes with more elements.
        order: Mesh polynomial order.  ``1`` = linear triangles (TRI3),
            ``2`` = quadratic (TRI6).  Only ``1`` is currently supported by
            the FEM assembly in diff_soup.
        verbose: If ``False`` (default), suppress all gmsh terminal output.

    Returns:
        [Mesh][diff_soup.domains.mesh.Mesh] with ``ele_type="TRI3"`` and:
        ``points`` — shape ``(n_nodes, 2)`` float64 array of node coordinates
        in physical units; ``cells`` — shape ``(n_elem, 3)`` int32 array of
        0-based node indices for each triangle.

    Raises:
        ValueError: if the polygon has fewer than 3 vertices or gmsh fails to mesh.

    Example:
        ::

            from diff_soup.domains.geometry import polygon_to_mesh

            square = [(0,0), (1,0), (1,1), (0,1)]
            mesh = polygon_to_mesh(square, mesh_size=0.1)
            print(mesh.points.shape, mesh.cells.shape)
    """
    gmsh = import_optional("gmsh")

    if len(polygon) < 3:
        raise ValueError(f"polygon must have at least 3 vertices, got {len(polygon)}")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("diff_soup_domain")

        # Add points
        point_tags = []
        for i, (x, y) in enumerate(polygon):
            tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size, i + 1)
            point_tags.append(tag)

        # Add lines
        n = len(point_tags)
        line_tags = []
        for i in range(n):
            tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n])
            line_tags.append(tag)

        # Curve loop + plane surface
        loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
        gmsh.model.geo.addPlaneSurface([loop_tag])

        gmsh.model.geo.synchronize()

        # Set mesh order and generate
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        # Extract nodes and triangles
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # node_coords is flat [x0,y0,z0, x1,y1,z1, ...]
        coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # keep x, y only

        if order == 1:
            elem_type = 2  # TRI3
            ele_type_str = "TRI3"
            nodes_per_elem = 3
        else:
            elem_type = 9  # TRI6
            ele_type_str = "TRI6"
            nodes_per_elem = 6

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_mask = np.array(elem_types) == elem_type
        if not np.any(tri_mask):
            raise ValueError(
                f"gmsh produced no elements of type {elem_type} (TRI{nodes_per_elem}). "
                "Check that the polygon is valid and non-self-intersecting."
            )

        idx = int(np.argwhere(tri_mask).flatten()[0])
        connectivity = np.array(elem_node_tags[idx], dtype=np.int64).reshape(-1, nodes_per_elem)

        # gmsh node tags are 1-based; build a compact 0-based index
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        cells = np.array(
            [[tag_to_idx[int(c)] for c in row] for row in connectivity],
            dtype=np.int32,
        )

        # Reorder points so index matches tag_to_idx
        order_arr = [tag_to_idx[int(t)] for t in node_tags]
        points = np.empty_like(coords)
        for src_i, dst_i in enumerate(order_arr):
            points[dst_i] = coords[src_i]

    finally:
        gmsh.finalize()

    return Mesh(points=points, cells=cells, ele_type=ele_type_str)


def mask_to_mesh(
    mask: np.ndarray,
    *,
    mesh_size: float = 1.0,
    pixel_size: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    epsilon_factor: float = 0.02,
    order: int = 1,
    contour_index: int = 0,
    verbose: bool = False,
) -> Mesh:
    """Convert a binary mask directly to a triangulated mesh.

    Convenience wrapper: mask → polygon ([mask_to_polygon][]) →
    mesh ([polygon_to_mesh][]).

    Args:
        mask: 2-D NumPy array (uint8 or bool), ``True``/non-zero = inside.
        mesh_size: Target triangle size in physical units.
        pixel_size: Physical size of one pixel (scales polygon coordinates).
        origin: Physical-coordinate offset ``(x0, y0)`` for the polygon vertices.
        epsilon_factor: Douglas-Peucker simplification factor (see [mask_to_polygon][]).
        order: Element order: 1 = TRI3, 2 = TRI6.
        contour_index: Which contour to use when the mask has multiple components.
        verbose: Forward gmsh terminal output if ``True``.

    Returns:
        Triangulated mesh in physical coordinates.

    Example:
        ::

            import numpy as np
            from diff_soup.domains.geometry import mask_to_mesh

            mask = np.zeros((100, 100), dtype=np.uint8)
            # Circular particle
            cx, cy, r = 50, 50, 30
            yy, xx = np.ogrid[:100, :100]
            mask[(xx - cx)**2 + (yy - cy)**2 < r**2] = 1

            mesh = mask_to_mesh(mask, mesh_size=3.0, pixel_size=0.5)
    """
    polygon = mask_to_polygon(
        mask,
        epsilon_factor=epsilon_factor,
        pixel_size=pixel_size,
        origin=origin,
        contour_index=contour_index,
    )
    return polygon_to_mesh(
        polygon,
        mesh_size=mesh_size,
        order=order,
        verbose=verbose,
    )


# ======================================================================
# Boundary edge extraction
# ======================================================================


def extract_boundary_edges(
    cells: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find boundary edges of a TRI3 mesh (edges belonging to exactly one element).

    Args:
        cells: Shape ``(n_elem, 3)`` int32 array of triangle connectivity.

    Returns:
        boundary_edges:
            Shape ``(n_bnd, 2)`` int32 array of boundary edge node pairs.
            Each edge is stored with the smaller node index first.
        boundary_elements:
            Shape ``(n_bnd,)`` int32 array mapping each boundary edge to the
            index of its parent element in ``cells``.
    """
    # Build all edges: for each triangle, 3 edges
    # Edge convention: (min(a,b), max(a,b)) for canonical form
    local_edges = [(0, 1), (1, 2), (2, 0)]
    all_edges = []  # (canonical_edge, element_index)
    for elem_idx in range(cells.shape[0]):
        tri = cells[elem_idx]
        for li, lj in local_edges:
            a, b = int(tri[li]), int(tri[lj])
            canonical = (min(a, b), max(a, b))
            all_edges.append((canonical, elem_idx))

    # Count occurrences of each edge
    edge_count: dict = {}
    edge_elem: dict = {}
    for edge, elem_idx in all_edges:
        if edge in edge_count:
            edge_count[edge] += 1
        else:
            edge_count[edge] = 1
            edge_elem[edge] = elem_idx

    # Boundary edges appear exactly once
    bnd_edges = []
    bnd_elems = []
    for edge, count in edge_count.items():
        if count == 1:
            bnd_edges.append(edge)
            bnd_elems.append(edge_elem[edge])

    if len(bnd_edges) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    boundary_edges = np.array(bnd_edges, dtype=np.int32)
    boundary_elements = np.array(bnd_elems, dtype=np.int32)
    return boundary_edges, boundary_elements


# ======================================================================
# Element quality metrics
# ======================================================================


def mesh_quality(
    points: np.ndarray,
    cells: np.ndarray,
) -> dict:
    """Compute element quality metrics for a TRI3 mesh.

    Args:
        points: Shape ``(n_nodes, 2)`` float64 node coordinates.
        cells: Shape ``(n_elem, 3)`` int32 triangle connectivity.

    Returns:
        dict
            Per-element arrays (each shape ``(n_elem,)``):

            - ``'min_angle'`` — minimum interior angle in degrees.
            - ``'max_angle'`` — maximum interior angle in degrees.
            - ``'aspect_ratio'`` — longest edge / shortest edge.
            - ``'area'`` — signed area (positive for CCW orientation).

            And a ``'summary'`` dict with min/max/mean of each metric.
    """
    p0 = points[cells[:, 0]]  # (n_elem, 2)
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]

    # Edge vectors
    e0 = p1 - p0  # edge opposite vertex 2
    e1 = p2 - p1  # edge opposite vertex 0
    e2 = p0 - p2  # edge opposite vertex 1

    # Edge lengths
    l0 = np.linalg.norm(e0, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)

    # Interior angles via dot products at each vertex
    # Angle at vertex 0: between edges e0 and -e2
    cos_a0 = np.sum(e0 * (-e2), axis=1) / (l0 * l2 + 1e-300)
    cos_a1 = np.sum(e1 * (-e0), axis=1) / (l1 * l0 + 1e-300)
    cos_a2 = np.sum(e2 * (-e1), axis=1) / (l2 * l1 + 1e-300)

    # Clamp for numerical safety
    cos_a0 = np.clip(cos_a0, -1.0, 1.0)
    cos_a1 = np.clip(cos_a1, -1.0, 1.0)
    cos_a2 = np.clip(cos_a2, -1.0, 1.0)

    a0 = np.degrees(np.arccos(cos_a0))
    a1 = np.degrees(np.arccos(cos_a1))
    a2 = np.degrees(np.arccos(cos_a2))

    angles = np.stack([a0, a1, a2], axis=1)  # (n_elem, 3)
    min_angle = angles.min(axis=1)
    max_angle = angles.max(axis=1)

    # Aspect ratio: longest / shortest edge
    lengths = np.stack([l0, l1, l2], axis=1)
    aspect_ratio = lengths.max(axis=1) / (lengths.min(axis=1) + 1e-300)

    # Signed area via cross product
    area = 0.5 * (e0[:, 0] * (-e2[:, 1]) - e0[:, 1] * (-e2[:, 0]))

    # Summary statistics
    summary = {}
    for name, arr in [
        ("min_angle", min_angle),
        ("max_angle", max_angle),
        ("aspect_ratio", aspect_ratio),
        ("area", area),
    ]:
        summary[name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    return {
        "min_angle": min_angle,
        "max_angle": max_angle,
        "aspect_ratio": aspect_ratio,
        "area": area,
        "summary": summary,
    }


# ======================================================================
# Tagged polygon meshing
# ======================================================================


def polygon_to_tagged_mesh(
    polygon: List[Tuple[float, float]],
    *,
    mesh_size: float = 1.0,
    segment_tags: Optional[List[int]] = None,
    segment_names: Optional[dict] = None,
    order: int = 1,
    verbose: bool = False,
) -> TaggedMesh:
    """Triangulate a polygon and tag each boundary segment.

    Like [polygon_to_mesh][], but assigns physical group tags to each
    polygon edge so that boundary conditions can be applied per-segment.

    Several polygon edges may share one tag id: they then form a single named
    region, which is how a named stretch of the boundary spanning more than one
    polygon edge is written (``segment_tags=[1, 2, 2, 1]`` with
    ``segment_names={1: "rim", 2: "bowl"}``).

    Args:
        polygon: Ordered ``(x, y)`` vertices defining the domain boundary (auto-closed).
        mesh_size: Target element size.
        segment_tags: Optional list of integer tags, one per polygon edge.  Ids may
            repeat to group edges into one region.  If ``None``, edges are tagged
            sequentially starting from 1.
        segment_names: Optional ``{tag: name}`` dict.  If ``None`` and ``segment_tags`` is
            also ``None``, names default to ``"seg_0"``, ``"seg_1"``, etc.
            A ``"boundary"`` name covering every tagged edge is always available
            as well, unless a segment already carries that name.
        order: Mesh polynomial order (1 = TRI3).
        verbose: Forward gmsh terminal output.

    Returns:
        Mesh with ``boundary_edges``, ``boundary_tags``, ``tag_names``, and the
        whole-boundary entry of ``tag_groups``.
    """
    gmsh = import_optional("gmsh")

    n_verts = len(polygon)
    if n_verts < 3:
        raise ValueError(f"polygon must have at least 3 vertices, got {n_verts}")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    n_segs = n_verts  # number of polygon edges (auto-closed)
    if segment_tags is not None and len(segment_tags) != n_segs:
        raise ValueError(
            f"segment_tags length {len(segment_tags)} != number of polygon edges {n_segs}"
        )

    # Build default tags / names
    if segment_tags is None:
        seg_tags = list(range(1, n_segs + 1))
    else:
        seg_tags = list(segment_tags)

    if segment_names is None:
        tag_names: dict = {}
        for i, t in enumerate(seg_tags):
            if t not in tag_names:
                tag_names[t] = f"seg_{i}"
    else:
        tag_names = dict(segment_names)

    # Every distinct id, in the order the polygon first uses it.  One physical
    # group per id, so ids repeated across polygon edges group those edges.
    group_ids: List[int] = []
    for t in seg_tags:
        if t not in group_ids:
            group_ids.append(int(t))

    # The whole boundary, addressable by name.  It unions the per-segment ids
    # rather than tagging the edges a second time, so selecting it never
    # double-counts an edge.  A segment already named "boundary" keeps the name.
    tag_groups = None if "boundary" in tag_names.values() else {"boundary": tuple(group_ids)}

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("diff_soup_tagged")

        # Add points
        point_tags = []
        for i, (x, y) in enumerate(polygon):
            tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size, i + 1)
            point_tags.append(tag)

        # Add lines and assign physical groups per segment
        line_tags = []
        for i in range(n_verts):
            lt = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n_verts])
            line_tags.append(lt)

        # Curve loop + surface
        loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
        gmsh.model.geo.addPlaneSurface([loop_tag])
        gmsh.model.geo.synchronize()

        # One physical group per distinct tag id, holding every line that
        # carries it.  gmsh rejects a second group with an id it already has,
        # so the lines are grouped first and each id is added exactly once.
        lines_for_id: dict = {tag_id: [] for tag_id in group_ids}
        for lt, tag_id in zip(line_tags, seg_tags):
            lines_for_id[int(tag_id)].append(lt)
        for pg_tag in group_ids:
            gmsh.model.addPhysicalGroup(1, lines_for_id[pg_tag], tag=pg_tag)
            if pg_tag in tag_names:
                gmsh.model.setPhysicalName(1, pg_tag, tag_names[pg_tag])

        # Physical group for the surface (needed for gmsh to mesh it)
        gmsh.model.addPhysicalGroup(2, [1], tag=999)

        # Generate
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        # ---- Extract volume mesh ----
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        if order == 1:
            elem_type = 2
            ele_type_str = "TRI3"
            nodes_per_elem = 3
        else:
            elem_type = 9
            ele_type_str = "TRI6"
            nodes_per_elem = 6

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_mask = np.array(elem_types) == elem_type
        if not np.any(tri_mask):
            raise ValueError("gmsh produced no triangle elements.")

        idx = int(np.argwhere(tri_mask).flatten()[0])
        connectivity = np.array(elem_node_tags[idx], dtype=np.int64).reshape(-1, nodes_per_elem)

        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        cells = np.array(
            [[tag_to_idx[int(c)] for c in row] for row in connectivity],
            dtype=np.int32,
        )

        points = np.empty_like(coords)
        for src_i, dst_i in enumerate([tag_to_idx[int(t)] for t in node_tags]):
            points[dst_i] = coords[src_i]

        # ---- Extract tagged boundary edges ----
        bnd_edges_list = []
        bnd_tags_list = []

        # Iterate over each distinct tag id's physical group, so an id shared
        # by several polygon edges contributes each of its edges exactly once.
        for pg_tag in group_ids:
            # Get line elements for this physical group
            entities = gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag)
            for ent in entities:
                et, _, ent_node_tags = gmsh.model.mesh.getElements(1, ent)
                if len(et) == 0:
                    continue
                # Line elements (type 1) have 2 nodes each
                for ti in range(len(et)):
                    if et[ti] == 1:  # 2-node line
                        line_nodes = np.array(ent_node_tags[ti], dtype=np.int64).reshape(-1, 2)
                        for row in line_nodes:
                            a = tag_to_idx[int(row[0])]
                            b = tag_to_idx[int(row[1])]
                            bnd_edges_list.append((a, b))
                            bnd_tags_list.append(pg_tag)

        if len(bnd_edges_list) > 0:
            boundary_edges = np.array(bnd_edges_list, dtype=np.int32)
            boundary_tags = np.array(bnd_tags_list, dtype=np.int32)
        else:
            boundary_edges = np.empty((0, 2), dtype=np.int32)
            boundary_tags = np.empty((0,), dtype=np.int32)

    finally:
        gmsh.finalize()

    return TaggedMesh(
        points=points,
        cells=cells,
        ele_type=ele_type_str,
        boundary_edges=boundary_edges,
        boundary_tags=boundary_tags,
        tag_names=tag_names,
        tag_groups=tag_groups,
    )


# ======================================================================
# 3-D meshing via gmsh OCC kernel
# ======================================================================


def _extract_tet_mesh(gmsh, order: int) -> Mesh:
    """Extract a TET4/TET10 mesh from the current gmsh model.

    Shared helper for the 3-D meshing functions.  Assumes ``gmsh`` has
    already been initialised and ``mesh.generate(3)`` has been called.
    """
    if order == 1:
        elem_type = 4  # TET4
        ele_type_str = "TET4"
        nodes_per_elem = 4
    else:
        elem_type = 11  # TET10
        ele_type_str = "TET10"
        nodes_per_elem = 10

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(node_coords).reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_mask = np.array(elem_types) == elem_type
    if not np.any(tet_mask):
        raise ValueError(
            f"gmsh produced no elements of type {elem_type} ({ele_type_str}). "
            "Check that the geometry and mesh size are valid."
        )

    idx = int(np.argwhere(tet_mask).flatten()[0])
    connectivity = np.array(elem_node_tags[idx], dtype=np.int64).reshape(-1, nodes_per_elem)

    # gmsh node tags are 1-based; build compact 0-based index
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    cells = np.array(
        [[tag_to_idx[int(c)] for c in row] for row in connectivity],
        dtype=np.int32,
    )

    # Reorder points so index matches tag_to_idx
    points = np.empty_like(coords)
    for src_i, dst_i in enumerate([tag_to_idx[int(t)] for t in node_tags]):
        points[dst_i] = coords[src_i]

    return Mesh(points=points, cells=cells, ele_type=ele_type_str)


def rectangle_mesh_quad(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    *,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Generate a structured QUAD4 or QUAD8 rectangular mesh via gmsh.

    Uses gmsh's point-line-surface extrusion with ``recombine=True`` to
    produce a tensor-product quadrilateral mesh directly (no triangulation
    + recombine).  For ``order=2``, enables ``Mesh.SecondOrderIncomplete``
    to emit 8-node serendipity (QUAD8) elements rather than 9-node
    full-Lagrange (QUAD9).

    For simple QUAD4 meshes without gmsh, use
    [rectangle_mesh][diff_soup.domains.mesh.rectangle_mesh].

    Args:
        Nx: Number of cells along the x axis.
        Ny: Number of cells along the y axis.
        Lx: Domain side length along x.  Mesh covers ``[0, Lx]``.
        Ly: Domain side length along y.  Mesh covers ``[0, Ly]``.
        order: ``1`` = QUAD4 (4-node linear), ``2`` = QUAD8 (8-node serendipity).
        verbose: Forward gmsh terminal output.

    Returns:
        [Mesh][diff_soup.domains.mesh.Mesh] with ``ele_type="QUAD4"`` or
        ``"QUAD8"``.
    """
    gmsh = import_optional("gmsh")

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if Nx < 1 or Ny < 1:
        raise ValueError(f"Nx, Ny must be >= 1, got ({Nx}, {Ny})")

    if order == 1:
        elem_type = 3  # gmsh: 4-node quadrangle
        ele_type_str = "QUAD4"
        nodes_per_elem = 4
    else:
        elem_type = 16  # gmsh: 8-node quadrangle (serendipity)
        ele_type_str = "QUAD8"
        nodes_per_elem = 8

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        if order == 2:
            # Serendipity (8-node) instead of full Lagrange (9-node)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

        gmsh.model.add("diff_soup_rectangle_quad")

        # Structural extrusion: point -> line -> surface (quads via recombine)
        p = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        extruded_line = gmsh.model.geo.extrude([(0, p)], Lx, 0, 0, [Nx], [1])
        gmsh.model.geo.extrude([extruded_line[1]], 0, Ly, 0, [Ny], [1], recombine=True)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)[:, :2]

        # Extract quad elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        mask = np.array(elem_types) == elem_type
        if not np.any(mask):
            raise ValueError(
                f"gmsh produced no elements of type {elem_type} ({ele_type_str}). "
                "Check that Nx, Ny >= 1 and the extrusion succeeded."
            )

        idx = int(np.argwhere(mask).flatten()[0])
        connectivity = np.array(elem_node_tags[idx], dtype=np.int64).reshape(-1, nodes_per_elem)

        # gmsh tags are 1-based; build compact 0-based index
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        cells = np.array(
            [[tag_to_idx[int(c)] for c in row] for row in connectivity],
            dtype=np.int32,
        )

        # Reorder points so row index matches tag_to_idx
        points = np.empty_like(coords)
        for src_i, dst_i in enumerate([tag_to_idx[int(t)] for t in node_tags]):
            points[dst_i] = coords[src_i]
    finally:
        gmsh.finalize()

    return Mesh(points=points, cells=cells, ele_type=ele_type_str)


def box_mesh_tet(
    lx: float,
    ly: float,
    lz: float,
    mesh_size: float,
    *,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Generate a box domain meshed with tetrahedra via gmsh OCC.

    Creates a ``[0, lx] x [0, ly] x [0, lz]`` box and fills it with
    TET4 (linear) or TET10 (quadratic) tetrahedra.

    Args:
        lx: Box dimension along x.
        ly: Box dimension along y.
        lz: Box dimension along z.
        mesh_size: Target element size.
        order: Mesh polynomial order.  ``1`` = TET4, ``2`` = TET10.
        verbose: Forward gmsh terminal output.

    Returns:
        [Mesh][diff_soup.domains.mesh.Mesh] with ``ele_type="TET4"``
        or ``"TET10"``.
    """
    gmsh = import_optional("gmsh")

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("diff_soup_box_tet")
        gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        result = _extract_tet_mesh(gmsh, order)
    finally:
        gmsh.finalize()

    return result


def cylinder_mesh(
    radius: float,
    height: float,
    mesh_size: float,
    *,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Generate a cylinder domain meshed with tetrahedra via gmsh OCC.

    The cylinder axis is along z, starting at the origin and extending
    to ``(0, 0, height)``.

    Args:
        radius: Cylinder radius.
        height: Cylinder height along z.
        mesh_size: Target element size.
        order: Mesh polynomial order.  ``1`` = TET4, ``2`` = TET10.
        verbose: Forward gmsh terminal output.

    Returns:
        [Mesh][diff_soup.domains.mesh.Mesh] with ``ele_type="TET4"``
        or ``"TET10"``.
    """
    gmsh = import_optional("gmsh")

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("diff_soup_cylinder")
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        result = _extract_tet_mesh(gmsh, order)
    finally:
        gmsh.finalize()

    return result


def sphere_mesh(
    radius: float,
    mesh_size: float,
    *,
    order: int = 1,
    verbose: bool = False,
) -> Mesh:
    """Generate a sphere domain meshed with tetrahedra via gmsh OCC.

    The sphere is centered at the origin.

    Args:
        radius: Sphere radius.
        mesh_size: Target element size.
        order: Mesh polynomial order.  ``1`` = TET4, ``2`` = TET10.
        verbose: Forward gmsh terminal output.

    Returns:
        [Mesh][diff_soup.domains.mesh.Mesh] with ``ele_type="TET4"``
        or ``"TET10"``.
    """
    gmsh = import_optional("gmsh")

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("diff_soup_sphere")
        gmsh.model.occ.addSphere(0, 0, 0, radius)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)

        result = _extract_tet_mesh(gmsh, order)
    finally:
        gmsh.finalize()

    return result
