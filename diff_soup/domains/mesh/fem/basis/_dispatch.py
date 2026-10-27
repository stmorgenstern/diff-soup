# Single dispatch surface; volume + face branches both fan out over every
# element type, and splitting per-type harms the at-a-glance overview of
# which gauss order, normals, and face_inds map to which element.
"""Reference-element shape functions and quadrature dispatch.

Each supported element type lives in its own module
([line2][], [tri3][], [tri6][], [quad4][], [quad8][], [tet4][],
[tet10][], [hex8][], [hex20][], [hex27][]); shared
quadrature rules live in [_common][].

The two public entry points [get_shape_vals_and_grads][] and
[get_face_shape_vals_and_grads][] dispatch on the ``ele_type`` string
("TRI3", "QUAD8", ...) and tensor-product the relevant 1D / 2D / 3D
quadrature with the matching shape functions.

A facet is one dimension below its element: a face in 3D, an edge in 2D, and
a single point in 1D.  ``LINE2`` therefore reports one node and one quadrature
point per facet, with weight ``1`` — integrating over a point is evaluating at
it.
"""

from __future__ import annotations

import numpy as np

from diff_soup.domains.mesh.fem.basis._common import (
    _gauss_1d,
    _gauss_1d_3,
    _gauss_tet,
    _gauss_tri,
)
from diff_soup.domains.mesh.fem.basis.hex8 import _hex8_shape
from diff_soup.domains.mesh.fem.basis.hex20 import _hex20_shape
from diff_soup.domains.mesh.fem.basis.hex27 import _hex27_shape
from diff_soup.domains.mesh.fem.basis.line2 import _line2_shape
from diff_soup.domains.mesh.fem.basis.quad4 import _quad4_shape
from diff_soup.domains.mesh.fem.basis.quad8 import _quad8_shape
from diff_soup.domains.mesh.fem.basis.tet4 import _tet4_shape
from diff_soup.domains.mesh.fem.basis.tet10 import _tet10_shape
from diff_soup.domains.mesh.fem.basis.tri3 import _tri3_shape
from diff_soup.domains.mesh.fem.basis.tri6 import _tri6_shape

__all__ = [
    "get_shape_vals_and_grads",
    "get_face_shape_vals_and_grads",
]


def get_shape_vals_and_grads(
    ele_type: str,
    gauss_order: int | None = None,
    *,
    metric_degree: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return volume-quadrature shape values, gradients, and weights.

    Dispatches on ``ele_type`` (e.g. ``"QUAD4"``, ``"HEX8"``, ``"TRI3"``) and
    tensor-products the appropriate 1D / 2D / 3D quadrature rule with the
    matching reference-element shape functions.

    Args:
        ele_type: Element type tag.
        gauss_order: Quadrature order; element-specific default when ``None``.
        metric_degree: Polynomial degree the coordinate metric adds to every
            integrand — ``0`` on a Cartesian domain, ``1`` for a cylindrical
            radial one (weight ``r``), ``2`` for a spherical one (weight
            ``r**2``).  The default quadrature order accounts for it, because a
            rule chosen from the basis alone is no longer exact once the
            measure multiplies the integrand.  Ignored when ``gauss_order`` is
            given explicitly.

    Returns:
        Tuple ``(vals, grads, weights)`` with shapes ``(n_q, n_nodes)``,
        ``(n_q, n_nodes, dim)``, and ``(n_q,)`` respectively.

    Raises:
        NotImplementedError: If ``ele_type`` is not supported.
    """
    if ele_type == "LINE2":
        # A mass integrand is N_i N_j r**p: degree 2 + metric_degree.  Two-point
        # Gauss is exact through degree 3, three-point through degree 5.
        o = (3 if metric_degree >= 2 else 2) if gauss_order is None else gauss_order
        pts_1d, w = _gauss_1d_3(o)
        vals, grads = _line2_shape(pts_1d[:, None])
        return vals, grads, w

    if ele_type == "QUAD4":
        o = 2 if gauss_order is None else gauss_order
        pts_1d, w_1d = _gauss_1d(o)
        rr, ss = np.meshgrid(pts_1d, pts_1d, indexing="ij")
        wr, ws = np.meshgrid(w_1d, w_1d, indexing="ij")
        pts = np.stack((rr, ss), axis=-1).reshape(-1, 2)
        w = (wr * ws).reshape(-1)
        vals, grads = _quad4_shape(pts)
        return vals, grads, w

    if ele_type == "HEX8":
        o = 2 if gauss_order is None else gauss_order
        pts_1d, w_1d = _gauss_1d(o)
        rr, ss, tt = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing="ij")
        wr, ws, wt = np.meshgrid(w_1d, w_1d, w_1d, indexing="ij")
        pts = np.stack((rr, ss, tt), axis=-1).reshape(-1, 3)
        w = (wr * ws * wt).reshape(-1)
        vals, grads = _hex8_shape(pts)
        return vals, grads, w

    if ele_type == "TRI3":
        o = 2 if gauss_order is None else gauss_order
        pts, w = _gauss_tri(o)
        vals, grads = _tri3_shape(pts)
        return vals, grads, w

    if ele_type == "TRI6":
        o = 2 if gauss_order is None else gauss_order
        pts, w = _gauss_tri(o)
        vals, grads = _tri6_shape(pts)
        return vals, grads, w

    if ele_type == "QUAD8":
        o = 3 if gauss_order is None else gauss_order
        pts_1d, w_1d = _gauss_1d_3(o)
        rr, ss = np.meshgrid(pts_1d, pts_1d, indexing="ij")
        wr, ws = np.meshgrid(w_1d, w_1d, indexing="ij")
        pts = np.stack((rr, ss), axis=-1).reshape(-1, 2)
        w = (wr * ws).reshape(-1)
        vals, grads = _quad8_shape(pts)
        return vals, grads, w

    if ele_type == "TET4":
        o = 2 if gauss_order is None else gauss_order
        pts, w = _gauss_tet(o)
        vals, grads = _tet4_shape(pts)
        return vals, grads, w

    if ele_type == "TET10":
        o = 2 if gauss_order is None else gauss_order
        pts, w = _gauss_tet(o)
        vals, grads = _tet10_shape(pts)
        return vals, grads, w

    if ele_type == "HEX20":
        o = 3 if gauss_order is None else gauss_order
        pts_1d, w_1d = _gauss_1d_3(o)
        rr, ss, tt = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing="ij")
        wr, ws, wt = np.meshgrid(w_1d, w_1d, w_1d, indexing="ij")
        pts = np.stack((rr, ss, tt), axis=-1).reshape(-1, 3)
        w = (wr * ws * wt).reshape(-1)
        vals, grads = _hex20_shape(pts)
        return vals, grads, w

    if ele_type == "HEX27":
        o = 3 if gauss_order is None else gauss_order
        pts_1d, w_1d = _gauss_1d_3(o)
        rr, ss, tt = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing="ij")
        wr, ws, wt = np.meshgrid(w_1d, w_1d, w_1d, indexing="ij")
        pts = np.stack((rr, ss, tt), axis=-1).reshape(-1, 3)
        w = (wr * ws * wt).reshape(-1)
        vals, grads = _hex27_shape(pts)
        return vals, grads, w

    raise NotImplementedError(f"Unsupported element type: {ele_type}")


def get_face_shape_vals_and_grads(
    ele_type: str,
    gauss_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return face-quadrature shape data for boundary integrals.

    For every face of the reference element, provides shape values, reference
    gradients, quadrature weights, outward unit normals, and face-local node
    index tables.  Used by FEM flux / traction boundary assembly.

    Args:
        ele_type: Element type tag.
        gauss_order: Quadrature order; element-specific default when ``None``.

    Returns:
        Tuple ``(vals, grads, weights, normals, face_inds)``.
        ``vals`` has shape ``(n_faces, n_q, n_nodes)``; ``grads`` has shape
        ``(n_faces, n_q, n_nodes, dim)``; ``weights`` has shape
        ``(n_faces, n_q)``; ``normals`` has shape ``(n_faces, dim)``;
        ``face_inds`` has shape ``(n_faces, n_face_nodes)``.

    Raises:
        NotImplementedError: If ``ele_type`` is not supported.
    """
    if ele_type == "LINE2":
        # The facets of a line element are its two end points.  A point carries
        # no reference length, so its single quadrature weight is 1 and the
        # facet integral is a plain evaluation; the outward normals are the
        # scalars -1 and +1.
        face_pts = np.array([[[-1.0]], [[1.0]]])  # (2 faces, 1 quad point, 1 coord)
        vals, grads = _line2_shape(face_pts.reshape(-1, 1))
        return (
            vals.reshape(2, 1, 2),
            grads.reshape(2, 1, 2, 1),
            np.ones((2, 1)),
            np.array([[-1.0], [1.0]]),
            np.array([[0], [1]], dtype=np.int32),
        )

    if ele_type == "QUAD4":
        o = 2 if gauss_order is None else gauss_order
        p, w = _gauss_1d(o)
        pts = []
        normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        face_inds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        for face in range(4):
            if face == 0:
                pts.append(np.stack((p, -np.ones_like(p)), axis=-1))
            elif face == 1:
                pts.append(np.stack((np.ones_like(p), p), axis=-1))
            elif face == 2:
                pts.append(np.stack((p, np.ones_like(p)), axis=-1))
            else:
                pts.append(np.stack((-np.ones_like(p), p), axis=-1))
        face_pts = np.stack(pts)
        vals, grads = _quad4_shape(face_pts.reshape(-1, 2))
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 2),
            np.tile(w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "HEX8":
        o = 2 if gauss_order is None else gauss_order
        p, w = _gauss_1d(o)
        uu, vv = np.meshgrid(p, p, indexing="ij")
        ww = np.outer(w, w).reshape(-1)
        uv = np.stack((uu, vv), axis=-1).reshape(-1, 2)
        normals = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        face_inds = np.array(
            [
                [0, 3, 7, 4],
                [1, 2, 6, 5],
                [0, 1, 5, 4],
                [3, 2, 6, 7],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            dtype=np.int32,
        )
        face_pts = []
        for face in range(6):
            if face == 0:
                xyz = np.column_stack((-np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 1:
                xyz = np.column_stack((np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 2:
                xyz = np.column_stack((uv[:, 0], -np.ones(len(uv)), uv[:, 1]))
            elif face == 3:
                xyz = np.column_stack((uv[:, 0], np.ones(len(uv)), uv[:, 1]))
            elif face == 4:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], -np.ones(len(uv))))
            else:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], np.ones(len(uv))))
            face_pts.append(xyz)
        face_pts = np.stack(face_pts)
        vals, grads = _hex8_shape(face_pts.reshape(-1, 3))
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 3),
            np.tile(ww[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "TRI3":
        # Parametrise each edge with t ∈ [−1, 1] (same convention as QUAD4 faces).
        # Face 0: nodes [0,1], r=(1+t)/2, s=0       outward normal (0,−1)
        # Face 1: nodes [1,2], r=(1−t)/2, s=(1+t)/2 outward normal (1,1)/√2
        # Face 2: nodes [2,0], r=0,        s=(1−t)/2 outward normal (−1,0)
        o = 2 if gauss_order is None else gauss_order
        p, w = _gauss_1d(o)
        t = p
        face_pts = [
            np.stack((0.5 * (1.0 + t), np.zeros_like(t)), axis=-1),
            np.stack((0.5 * (1.0 - t), 0.5 * (1.0 + t)), axis=-1),
            np.stack((np.zeros_like(t), 0.5 * (1.0 - t)), axis=-1),
        ]
        face_pts = np.stack(face_pts)  # (3, n_q, 2)
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        vals, grads = _tri3_shape(face_pts.reshape(-1, 2))
        s2 = 1.0 / np.sqrt(2.0)
        normals = np.array([[0.0, -1.0], [s2, s2], [-1.0, 0.0]])
        face_inds = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 2),
            np.tile(w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "TRI6":
        # 3 edges, each with 3 nodes (quadratic).
        # Parametrise with t ∈ [−1, 1], same convention as TRI3.
        # Edge 0: nodes [0,3,1]  r=(1+t)/2, s=0
        # Edge 1: nodes [1,4,2]  r=(1-t)/2, s=(1+t)/2
        # Edge 2: nodes [2,5,0]  r=0, s=(1-t)/2
        o = 3 if gauss_order is None else gauss_order
        p, w = _gauss_1d_3(o)
        t = p
        face_pts = [
            np.stack((0.5 * (1.0 + t), np.zeros_like(t)), axis=-1),
            np.stack((0.5 * (1.0 - t), 0.5 * (1.0 + t)), axis=-1),
            np.stack((np.zeros_like(t), 0.5 * (1.0 - t)), axis=-1),
        ]
        face_pts = np.stack(face_pts)
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        vals, grads = _tri6_shape(face_pts.reshape(-1, 2))
        s2 = 1.0 / np.sqrt(2.0)
        normals = np.array([[0.0, -1.0], [s2, s2], [-1.0, 0.0]])
        face_inds = np.array([[0, 3, 1], [1, 4, 2], [2, 5, 0]], dtype=np.int32)
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 2),
            np.tile(w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "QUAD8":
        # 4 edges, each with 3 nodes (quadratic).
        # Same edge parametrisation as QUAD4 but with quadratic shape functions.
        o = 3 if gauss_order is None else gauss_order
        p, w = _gauss_1d_3(o)
        pts = []
        normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        # Edge 0: nodes [0,4,1] (bottom), Edge 1: nodes [1,5,2] (right),
        # Edge 2: nodes [2,6,3] (top),    Edge 3: nodes [3,7,0] (left)
        face_inds = np.array([[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]], dtype=np.int32)
        for face in range(4):
            if face == 0:
                pts.append(np.stack((p, -np.ones_like(p)), axis=-1))
            elif face == 1:
                pts.append(np.stack((np.ones_like(p), p), axis=-1))
            elif face == 2:
                pts.append(np.stack((p, np.ones_like(p)), axis=-1))
            else:
                pts.append(np.stack((-np.ones_like(p), p), axis=-1))
        face_pts = np.stack(pts)
        vals, grads = _quad8_shape(face_pts.reshape(-1, 2))
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 2),
            np.tile(w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "TET4":
        # 4 triangular faces. Use TRI3 face shape functions via _gauss_tri.
        # Face 0: nodes [1,2,3] (opposite vertex 0)
        # Face 1: nodes [0,3,2] (opposite vertex 1)
        # Face 2: nodes [0,1,3] (opposite vertex 2)
        # Face 3: nodes [0,2,1] (opposite vertex 3)
        o = 2 if gauss_order is None else gauss_order
        tri_pts, tri_w = _gauss_tri(o)  # (n_q, 2) on ref triangle
        n_q = len(tri_w)

        # Reference tet vertices
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        face_verts = [
            (v1, v2, v3),  # face 0
            (v0, v3, v2),  # face 1
            (v0, v1, v3),  # face 2
            (v0, v2, v1),  # face 3
        ]
        face_inds = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)

        face_pts_list = []
        normals_list = []
        for va, vb, vc in face_verts:
            # Map (u, v) on ref triangle to 3D: P = va + u*(vb - va) + v*(vc - va)
            u = tri_pts[:, 0:1]  # (n_q, 1)
            v = tri_pts[:, 1:2]
            pts_3d = va[None, :] + u * (vb - va)[None, :] + v * (vc - va)[None, :]
            face_pts_list.append(pts_3d)
            # Outward normal: cross(vb-va, vc-va), normalised
            e1 = vb - va
            e2 = vc - va
            n = np.cross(e1, e2)
            n = n / np.linalg.norm(n)
            normals_list.append(n)

        face_pts = np.stack(face_pts_list)  # (4, n_q, 3)
        normals = np.stack(normals_list)  # (4, 3)

        vals, grads = _tet4_shape(face_pts.reshape(-1, 3))
        n_faces = 4
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 3),
            np.tile(tri_w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "TET10":
        # 4 faces, each TRI6 (6 nodes per face).
        # Face 0: nodes [1,2,3, 7,8,9] (opposite vertex 0)
        # Face 1: nodes [0,3,2, 6,8,5] (opposite vertex 1)
        # Face 2: nodes [0,1,3, 4,9,6] (opposite vertex 2)
        # Face 3: nodes [0,2,1, 5,7,4] (opposite vertex 3)
        o = 2 if gauss_order is None else gauss_order
        tri_pts, tri_w = _gauss_tri(o)
        n_q = len(tri_w)

        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        face_verts = [
            (v1, v2, v3),
            (v0, v3, v2),
            (v0, v1, v3),
            (v0, v2, v1),
        ]
        face_inds = np.array(
            [
                [1, 2, 3, 7, 8, 9],
                [0, 3, 2, 6, 8, 5],
                [0, 1, 3, 4, 9, 6],
                [0, 2, 1, 5, 7, 4],
            ],
            dtype=np.int32,
        )

        face_pts_list = []
        normals_list = []
        for va, vb, vc in face_verts:
            u = tri_pts[:, 0:1]
            v = tri_pts[:, 1:2]
            pts_3d = va[None, :] + u * (vb - va)[None, :] + v * (vc - va)[None, :]
            face_pts_list.append(pts_3d)
            e1 = vb - va
            e2 = vc - va
            n = np.cross(e1, e2)
            n = n / np.linalg.norm(n)
            normals_list.append(n)

        face_pts = np.stack(face_pts_list)
        normals = np.stack(normals_list)

        vals, grads = _tet10_shape(face_pts.reshape(-1, 3))
        n_faces = 4
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 3),
            np.tile(tri_w[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "HEX20":
        # 6 faces, each QUAD8 (8 nodes per face).
        o = 3 if gauss_order is None else gauss_order
        p, w = _gauss_1d_3(o)
        uu, vv = np.meshgrid(p, p, indexing="ij")
        ww = np.outer(w, w).reshape(-1)
        uv = np.stack((uu, vv), axis=-1).reshape(-1, 2)
        normals = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Face node indices (8 nodes each: 4 corners + 4 midedges of face)
        face_inds = np.array(
            [
                [0, 3, 7, 4, 11, 19, 15, 16],  # r=-1
                [1, 2, 6, 5, 9, 18, 13, 17],  # r=+1
                [0, 1, 5, 4, 8, 17, 12, 16],  # s=-1
                [3, 2, 6, 7, 10, 18, 14, 19],  # s=+1
                [0, 1, 2, 3, 8, 9, 10, 11],  # t=-1
                [4, 5, 6, 7, 12, 13, 14, 15],  # t=+1
            ],
            dtype=np.int32,
        )
        face_pts = []
        for face in range(6):
            if face == 0:
                xyz = np.column_stack((-np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 1:
                xyz = np.column_stack((np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 2:
                xyz = np.column_stack((uv[:, 0], -np.ones(len(uv)), uv[:, 1]))
            elif face == 3:
                xyz = np.column_stack((uv[:, 0], np.ones(len(uv)), uv[:, 1]))
            elif face == 4:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], -np.ones(len(uv))))
            else:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], np.ones(len(uv))))
            face_pts.append(xyz)
        face_pts = np.stack(face_pts)
        vals, grads = _hex20_shape(face_pts.reshape(-1, 3))
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 3),
            np.tile(ww[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    if ele_type == "HEX27":
        # 6 faces, each 9 nodes (4 corners + 4 midedges + 1 face center).
        o = 3 if gauss_order is None else gauss_order
        p, w = _gauss_1d_3(o)
        uu, vv = np.meshgrid(p, p, indexing="ij")
        ww = np.outer(w, w).reshape(-1)
        uv = np.stack((uu, vv), axis=-1).reshape(-1, 2)
        normals = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Face node indices (9 nodes each: 4 corners + 4 midedges + 1 face center)
        face_inds = np.array(
            [
                [0, 3, 7, 4, 11, 19, 15, 16, 25],  # r=-1
                [1, 2, 6, 5, 9, 18, 13, 17, 23],  # r=+1
                [0, 1, 5, 4, 8, 17, 12, 16, 22],  # s=-1
                [3, 2, 6, 7, 10, 18, 14, 19, 24],  # s=+1
                [0, 1, 2, 3, 8, 9, 10, 11, 20],  # t=-1
                [4, 5, 6, 7, 12, 13, 14, 15, 21],  # t=+1
            ],
            dtype=np.int32,
        )
        face_pts = []
        for face in range(6):
            if face == 0:
                xyz = np.column_stack((-np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 1:
                xyz = np.column_stack((np.ones(len(uv)), uv[:, 0], uv[:, 1]))
            elif face == 2:
                xyz = np.column_stack((uv[:, 0], -np.ones(len(uv)), uv[:, 1]))
            elif face == 3:
                xyz = np.column_stack((uv[:, 0], np.ones(len(uv)), uv[:, 1]))
            elif face == 4:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], -np.ones(len(uv))))
            else:
                xyz = np.column_stack((uv[:, 0], uv[:, 1], np.ones(len(uv))))
            face_pts.append(xyz)
        face_pts = np.stack(face_pts)
        vals, grads = _hex27_shape(face_pts.reshape(-1, 3))
        n_faces, n_q = face_pts.shape[0], face_pts.shape[1]
        return (
            vals.reshape(n_faces, n_q, -1),
            grads.reshape(n_faces, n_q, -1, 3),
            np.tile(ww[None, :], (n_faces, 1)),
            normals,
            face_inds,
        )

    raise NotImplementedError(f"Unsupported element type: {ele_type}")
