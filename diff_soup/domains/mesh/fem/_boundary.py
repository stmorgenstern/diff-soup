"""Boundary face / edge enumeration for FEM meshes.

Single source of truth used by [`diff_soup.problem.ir.lower`][diff_soup.problem.ir.lower]
(FluxBC, RobinBC, TractionBC, BoundaryIntegralConstraint) and by
``q.ops.boundary_integral`` / ``boundary_trace`` / ``surface_flux``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
    from diff_soup.domains.mesh.core import TaggedMesh


def boundary_face_assembly(
    mesh: "TaggedMesh", tag: Union[str, int]
) -> tuple[
    Int[Array, "n_faces vertices_per_face"], Float[Array, " n_faces"], Float[Array, "n_faces dim"]
]:
    """Return ``(face_indices, face_areas_or_lengths, outward_normals)`` for a tagged boundary.

    Args:
        mesh: A [TaggedMesh][diff_soup.domains.TaggedMesh] with ``boundary_edges`` (2D) or
            ``boundary_faces`` (3D faces, 1D points) populated and ``boundary_tags``
            mapping each entry to a tag id, plus ``tag_names`` for the
            string-name reverse lookup.
        tag: A tag name (``str``) present in ``mesh.tag_names`` or an
            integer tag id present in ``mesh.boundary_tags``.

    Returns:
        ``face_indices``: ``(n_faces, vertices_per_face)`` integer array of
            node indices into ``mesh.points`` for each boundary face / edge.
        ``face_areas_or_lengths``: ``(n_faces,)`` float array.  Point measure in
            1D, edge length in 2D, face area in 3D.
        ``outward_normals``: ``(n_faces, dim)`` float array, unit length.

    Raises:
        KeyError: If ``tag`` (string) is not in ``mesh.tag_names`` or
            (int) is not present in ``mesh.boundary_tags``.
        NotImplementedError: If ``mesh.ele_type`` is not LINE2, TRI3, or TET4.
    """
    if isinstance(tag, str):
        ids = mesh._tag_ids_for_name(tag)
    else:
        ids = [int(tag)]
    return boundary_face_assembly_for_ids(mesh, ids)


def boundary_face_assembly_for_ids(
    mesh: "TaggedMesh", ids: Iterable[int]
) -> tuple[
    Int[Array, "n_faces vertices_per_face"], Float[Array, " n_faces"], Float[Array, "n_faces dim"]
]:
    """Return ``(face_indices, face_areas_or_lengths, outward_normals)`` for explicit tag ids.

    Like [boundary_face_assembly][diff_soup.domains.mesh.fem._boundary.boundary_face_assembly], but takes an already-resolved iterable
    of integer tag ids rather than a single name or id.  The boundary entries
    whose tag id is in ``ids`` form the selected facet set (their union); use
    this when several names have been pre-resolved to ids and combined.

    Args:
        mesh: A [TaggedMesh][diff_soup.domains.TaggedMesh] with ``boundary_edges`` (2D) or
            ``boundary_faces`` (3D faces, 1D points) populated and ``boundary_tags``
            mapping each entry to a tag id.
        ids: An iterable of integer tag ids to select.

    Returns:
        The same ``(face_indices, face_areas_or_lengths, outward_normals)``
        triple as [boundary_face_assembly][diff_soup.domains.mesh.fem._boundary.boundary_face_assembly].

    Raises:
        KeyError: If none of ``ids`` is present in ``mesh.boundary_tags``.
        NotImplementedError: If ``mesh.ele_type`` is not LINE2, TRI3, or TET4.
    """
    ids = list(ids)
    et = mesh.ele_type.upper()
    if et == "LINE2":
        return _line2_vertices(mesh, ids)
    if et == "TRI3":
        return _tri3_edges(mesh, ids)
    if et == "TET4":
        return _tet4_faces(mesh, ids)
    raise NotImplementedError(
        f"boundary_face_assembly: element type {mesh.ele_type} not supported "
        f"(supported: LINE2, TRI3, TET4; QUAD4/HEX8/higher-order not yet implemented)"
    )


def _line2_vertices(
    mesh: "TaggedMesh", ids: list
) -> tuple[Int[Array, "n_points 1"], Float[Array, " n_points"], Float[Array, "n_points 1"]]:
    """LINE2 boundary points: 1 node/facet, measure is ``r**p``, normal is +-1.

    Integrating over a point is evaluating at it, so a Cartesian point facet
    carries measure ``1`` — the same number a Cartesian ``Grid1D`` records as
    its face area.  A curvilinear mesh weighs the point by the surface metric
    ``r**p`` (``p = 1`` cylindrical, ``2`` spherical), matching the radial
    grids' face areas; at ``r = 0`` that measure is zero, which is how the
    boundary term at the axis vanishes without a special case.  The outward
    normal has no tangent to rotate and no pair of edges to cross, so it is the
    sign of the offset from the mesh midpoint.
    """
    points_all = np.asarray(mesh.boundary_faces, dtype=np.int32)
    tags_all = np.asarray(mesh.boundary_tags, dtype=np.int32)
    if points_all.size == 0 or tags_all.size != points_all.shape[0]:
        raise KeyError(f"Boundary tag ids {ids} not present (no 1D boundary points tagged)")
    mask = np.isin(tags_all, ids)
    facets = points_all[mask]
    if facets.shape[0] == 0:
        raise KeyError(f"Boundary tag ids {ids} not present in mesh.boundary_tags")

    from diff_soup.domains.mesh.core import metric_exponent

    points = np.asarray(mesh.points, dtype=np.float64)
    x = points[facets[:, 0], 0]
    p = metric_exponent(getattr(mesh, "coord_system", None))
    measure = np.ones_like(x) if p == 0 else x**p
    midpoint = float(np.mean(points[:, 0]))
    sign = np.sign(x - midpoint)
    sign = np.where(sign == 0.0, 1.0, sign)

    return (
        jnp.asarray(facets, dtype=jnp.int32),
        jnp.asarray(measure),
        jnp.asarray(sign[:, None]),
    )


def _tri3_edges(
    mesh: "TaggedMesh", ids: list
) -> tuple[Int[Array, "n_edges 2"], Float[Array, " n_edges"], Float[Array, "n_edges 2"]]:
    """TRI3 boundary edges: 2 nodes/edge, length is Euclidean distance, normal is outward unit."""
    edges_all = np.asarray(mesh.boundary_edges, dtype=np.int32)
    tags_all = np.asarray(mesh.boundary_tags, dtype=np.int32)
    if edges_all.size == 0 or tags_all.size != edges_all.shape[0]:
        raise KeyError(f"Boundary tag ids {ids} not present (no 2D boundary edges tagged)")
    mask = np.isin(tags_all, ids)
    edges = edges_all[mask]
    if edges.shape[0] == 0:
        raise KeyError(f"Boundary tag ids {ids} not present in mesh.boundary_tags")

    points = np.asarray(mesh.points, dtype=np.float64)
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    tangent = p1 - p0
    lengths = np.linalg.norm(tangent, axis=1)
    # Outward normal: rotate tangent by -90deg via (ty, -tx).
    # For axis-aligned rectangle boundaries this gives the correct outward
    # direction provided the edges are consistently oriented.  We orient
    # each edge so the interior is to the LEFT of (p0->p1) by inspecting
    # the centroid of the boundary loop.
    raw_normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    raw_normal = raw_normal / lengths[:, None]
    # Make outward by checking against the mesh centroid.
    centroid = np.mean(points, axis=0)
    edge_mid = 0.5 * (p0 + p1)
    outward_dir = edge_mid - centroid
    outward_dir = outward_dir / (np.linalg.norm(outward_dir, axis=1, keepdims=True) + 1e-30)
    sign = np.sign(np.sum(raw_normal * outward_dir, axis=1))
    sign = np.where(sign == 0, 1.0, sign)
    normals = raw_normal * sign[:, None]

    return (
        jnp.asarray(edges, dtype=jnp.int32),
        jnp.asarray(lengths),
        jnp.asarray(normals),
    )


def _tet4_faces(
    mesh: "TaggedMesh", ids: list
) -> tuple[Int[Array, "n_faces 3"], Float[Array, " n_faces"], Float[Array, "n_faces 3"]]:
    """TET4 boundary faces: 3 nodes/face, area is half cross-product magnitude, normal is outward unit."""
    faces_all = np.asarray(mesh.boundary_faces, dtype=np.int32)
    tags_all = np.asarray(mesh.boundary_tags, dtype=np.int32)
    if faces_all.size == 0 or tags_all.size != faces_all.shape[0]:
        raise KeyError(f"Boundary tag ids {ids} not present (no 3D boundary faces tagged)")
    mask = np.isin(tags_all, ids)
    faces = faces_all[mask]
    if faces.shape[0] == 0:
        raise KeyError(f"Boundary tag ids {ids} not present in mesh.boundary_tags")

    points = np.asarray(mesh.points, dtype=np.float64)
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    cross_mag = np.linalg.norm(cross, axis=1)
    areas = 0.5 * cross_mag
    raw_normal = cross / cross_mag[:, None]
    # Orient outward via centroid test.
    centroid = np.mean(points, axis=0)
    face_mid = (p0 + p1 + p2) / 3.0
    outward_dir = face_mid - centroid
    outward_dir = outward_dir / (np.linalg.norm(outward_dir, axis=1, keepdims=True) + 1e-30)
    sign = np.sign(np.sum(raw_normal * outward_dir, axis=1))
    sign = np.where(sign == 0, 1.0, sign)
    normals = raw_normal * sign[:, None]

    return (
        jnp.asarray(faces, dtype=jnp.int32),
        jnp.asarray(areas),
        jnp.asarray(normals),
    )


__all__ = ["boundary_face_assembly", "boundary_face_assembly_for_ids"]
