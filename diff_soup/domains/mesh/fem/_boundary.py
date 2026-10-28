"""Boundary facets of a finite-element mesh: their nodes, measures and outward normals.

One facet table serves the flux, Robin and traction conditions, the boundary
integral constraints, ``q.ops.boundary_integral`` / ``boundary_trace`` /
``surface_flux`` and the surface measures.

Each returned row lists one facet's nodes so that an equal share per entry is
that node's weight ``int_f N_i dS / |f|``, which is the consistent load of a
constant flux density:

* a point (``LINE2``): ``[a]``;
* a linear edge (``TRI3``, ``QUAD4``): ``[a, b]``;
* a quadratic edge (``TRI6``, ``QUAD8``): ``[a, m, m, m, m, b]``, the Simpson
  weights ``1/6, 2/3, 1/6``;
* a triangle (``TET4``): ``[a, b, c]``;
* a quadrilateral (``HEX8``): ``[a, b, c, d]``, exact on a parallelogram.

The row mean of a nodal field is then the facet's average of it, and the
consumers (which split each facet's measure equally over its row) need no
per-element case.  The outward normal of each facet points away from the
element that owns it; on an interface facet shared by two tagged regions it
points out of the region whose name sorts first.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

if TYPE_CHECKING:
    from diff_soup.domains.mesh.core import TaggedMesh


# Local facets of each element type, in the node order of
# ``basis/_dispatch.py``'s face tables.  An edge lists its two end nodes, a
# face its corners in cyclic order; a quadratic edge's midside node is in
# ``_EDGE_MIDS`` (same edge order).
_LOCAL_EDGES = {
    "TRI3": ((0, 1), (1, 2), (2, 0)),
    "TRI6": ((0, 1), (1, 2), (2, 0)),
    "QUAD4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "QUAD8": ((0, 1), (1, 2), (2, 3), (3, 0)),
}
_EDGE_MIDS = {"TRI6": (3, 4, 5), "QUAD8": (4, 5, 6, 7)}
_LOCAL_FACES = {
    "TET4": ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
    "HEX8": (
        (0, 3, 7, 4),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 2, 6, 7),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ),
}
SUPPORTED_ELEMENT_TYPES = ("LINE2", "TRI3", "TRI6", "QUAD4", "QUAD8", "TET4", "HEX8")

_Triple = tuple[
    Int[Array, "n_faces row_width"], Float[Array, " n_faces"], Float[Array, "n_faces dim"]
]


def _element_type(mesh) -> str:
    et = str(mesh.ele_type).upper()
    if et not in SUPPORTED_ELEMENT_TYPES:
        raise NotImplementedError(
            f"boundary facets of {mesh.ele_type} elements are not supported; facet "
            f"integrals (flux, Robin and traction conditions, boundary operators) "
            f"support {', '.join(SUPPORTED_ELEMENT_TYPES)}"
        )
    return et


def _local_facets(et: str) -> np.ndarray:
    return np.asarray(_LOCAL_EDGES.get(et) or _LOCAL_FACES[et], dtype=np.int64)


def _facet_table(mesh) -> np.ndarray:
    """The tagged facet table, one facet per row (empty when nothing is tagged)."""
    return np.asarray(mesh.boundary_facets(), dtype=np.int64)


def _facet_owners(mesh, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The element facets that the table rows name.

    A row is matched by its first ``k`` nodes (the edge ends or the face
    corners) as a set.  Returns ``(cell, local, count)``: for each row the
    owning cell and its local facet index (two columns, the second equal to the
    first unless the facet is shared by two elements) and the number of owners.

    Raises:
        ValueError: If a row is not a facet of any element.
    """
    et = _element_type(mesh)
    local = _local_facets(et)
    k = local.shape[1]
    n_rows = rows.shape[0]
    if rows.ndim != 2 or rows.shape[1] < k:
        raise ValueError(
            f"a {et} mesh lists each boundary facet by at least its {k} corner nodes; "
            f"the facet table has rows of width {rows.shape[1] if rows.ndim == 2 else 0}"
        )
    cells = np.asarray(mesh.cells, dtype=np.int64)
    n_local = local.shape[0]
    cell_keys = np.sort(cells[:, local].reshape(-1, k), axis=1)
    row_keys = np.sort(rows[:, :k], axis=1)
    _, inverse = np.unique(np.concatenate([row_keys, cell_keys]), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    row_group, cell_group = inverse[:n_rows], inverse[n_rows:]
    order = np.argsort(cell_group, kind="stable")
    sorted_groups = cell_group[order]
    start = np.searchsorted(sorted_groups, row_group, side="left")
    count = np.searchsorted(sorted_groups, row_group, side="right") - start
    missing = np.flatnonzero(count == 0)
    if missing.size:
        i = int(missing[0])
        raise ValueError(
            f"boundary facet row {i} (nodes {rows[i].tolist()}) is not a facet of any "
            f"{et} element; list a facet by its end nodes or corners first, then any "
            "midside node"
        )
    last = max(len(order) - 1, 0)
    first = order[np.minimum(start, last)]
    second = np.where(count > 1, order[np.minimum(start + 1, last)], first)
    owners = np.stack([first, second], axis=1)
    return owners // n_local, owners % n_local, count


def _choose_owner(mesh, cell: np.ndarray, local: np.ndarray, count: np.ndarray):
    """One owner per facet: the cell whose region name sorts first on an interface."""
    chosen_cell, chosen_local = cell[:, 0].copy(), local[:, 0].copy()
    shared = np.flatnonzero(count > 1)
    cell_tags = getattr(mesh, "cell_tags", None)
    if shared.size == 0 or cell_tags is None:
        return chosen_cell, chosen_local
    names = {v: k for k, v in (getattr(mesh, "cell_tag_names", None) or {}).items()}
    tags = np.asarray(cell_tags)
    for i in shared.tolist():
        a, b = int(cell[i, 0]), int(cell[i, 1])
        ka = names.get(int(tags[a]), f"~{int(tags[a]):012d}")
        kb = names.get(int(tags[b]), f"~{int(tags[b]):012d}")
        if kb < ka:
            chosen_cell[i], chosen_local[i] = b, local[i, 1]
    return chosen_cell, chosen_local


def _edge_nodes(et, cells, cell, local) -> np.ndarray:
    """``(n, 2)`` end nodes, or ``(n, 3)`` ``[a, b, m]`` on a quadratic edge."""
    ends = np.asarray(_LOCAL_EDGES[et], dtype=np.int64)[local]
    nodes = np.stack([cells[cell, ends[:, 0]], cells[cell, ends[:, 1]]], axis=1)
    if et in _EDGE_MIDS:
        mids = np.asarray(_EDGE_MIDS[et], dtype=np.int64)[local]
        nodes = np.concatenate([nodes, cells[cell, mids][:, None]], axis=1)
    return nodes


def _face_nodes(et, cells, cell, local) -> np.ndarray:
    corners = np.asarray(_LOCAL_FACES[et], dtype=np.int64)[local]
    return np.take_along_axis(cells[cell], corners, axis=1)


def _orient(raw_normal, facet_points, cell_points) -> np.ndarray:
    """Flip each unit normal to point away from its owning element."""
    away = facet_points.mean(axis=1) - cell_points.mean(axis=1)
    sign = np.sign(np.sum(raw_normal * away, axis=1))
    return raw_normal * np.where(sign == 0.0, 1.0, sign)[:, None]


def _quadratic_edge_length(p: np.ndarray) -> np.ndarray:
    """Arc length of ``[a, b, m]`` edges, three-point Gauss on the LINE3 map."""
    a, b, m = p[:, 0], p[:, 1], p[:, 2]
    length = np.zeros(p.shape[0])
    for t, w in zip((-np.sqrt(0.6), 0.0, np.sqrt(0.6)), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)):
        tangent = a * (t - 0.5) + b * (t + 0.5) - 2.0 * t * m
        length += w * np.linalg.norm(tangent, axis=1)
    return length


def _bilinear_face_area(p: np.ndarray) -> np.ndarray:
    """Area of cyclic ``[a, b, c, d]`` faces, 2x2 Gauss on the bilinear map."""
    g = 1.0 / np.sqrt(3.0)
    area = np.zeros(p.shape[0])
    for u in (-g, g):
        for v in (-g, g):
            x_u = 0.25 * ((1 - v) * (p[:, 1] - p[:, 0]) + (1 + v) * (p[:, 2] - p[:, 3]))
            x_v = 0.25 * ((1 - u) * (p[:, 3] - p[:, 0]) + (1 + u) * (p[:, 2] - p[:, 1]))
            area += np.linalg.norm(np.cross(x_u, x_v), axis=1)
    return area


class _Facets:
    """Geometry of a selection of boundary facets (host arrays)."""

    def __init__(self, mesh, mask: np.ndarray):
        et = _element_type(mesh)
        self.ele_type = et
        table = _facet_table(mesh)
        rows = table[mask]
        points = np.asarray(mesh.points, dtype=np.float64)
        self.n_nodes = int(points.shape[0])
        if et == "LINE2":
            self._init_points(mesh, rows, points)
            return
        cell, local, count = _facet_owners(mesh, rows)
        cell, local = _choose_owner(mesh, cell, local, count)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        self.cell = cell
        cell_points = points[cells[cell]]
        if et in _LOCAL_EDGES:
            nodes = _edge_nodes(et, cells, cell, local)
            p = points[nodes]
            tangent = p[:, 1] - p[:, 0]
            chord = np.linalg.norm(tangent, axis=1)
            raw = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1) / chord[:, None]
            if nodes.shape[1] == 3:
                self.kind = "edge3"
                self.measure = _quadratic_edge_length(p)
                a, b, m = nodes[:, 0:1], nodes[:, 1:2], nodes[:, 2:3]
                self.rows = np.concatenate([a, m, m, m, m, b], axis=1)
            else:
                self.kind = "edge2"
                self.measure = chord
                self.rows = nodes
        else:
            nodes = _face_nodes(et, cells, cell, local)
            p = points[nodes]
            if nodes.shape[1] == 3:
                self.kind = "tri"
                cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                self.measure = 0.5 * np.linalg.norm(cross, axis=1)
            else:
                self.kind = "quad"
                cross = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
                self.measure = _bilinear_face_area(p)
            raw = cross / np.linalg.norm(cross, axis=1)[:, None]
            self.rows = nodes
        self.nodes = nodes
        self.normals = _orient(raw, p, cell_points)

    def _init_points(self, mesh, rows, points):
        """LINE2 boundary points: measure ``r**p``, normal the sign of the offset.

        Integrating over a point is evaluating at it, so a Cartesian point carries
        measure ``1``, the face area a Cartesian ``Grid1D`` records.  A curvilinear
        mesh weighs the point by ``r**p`` (``p = 1`` cylindrical, ``2`` spherical),
        so the boundary term at the axis vanishes without a special case.
        """
        from diff_soup.domains.mesh.core import metric_exponent

        x = points[rows[:, 0], 0]
        p = metric_exponent(getattr(mesh, "coord_system", None))
        sign = np.sign(x - float(np.mean(points[:, 0])))
        cells = np.asarray(mesh.cells, dtype=np.int64)
        self.kind = "point"
        self.rows = self.nodes = rows[:, :1]
        self.measure = np.ones_like(x) if p == 0 else x**p
        self.normals = np.where(sign == 0.0, 1.0, sign)[:, None]
        touches = (cells[None, :, :] == rows[:, :1, None]).any(axis=2)  # (n_points, n_cells)
        if not touches.any(axis=1).all():
            raise ValueError("a LINE2 boundary point is not a node of any element")
        self.cell = np.argmax(touches, axis=1)

    def triple(self) -> _Triple:
        return (
            jnp.asarray(self.rows, dtype=jnp.int32),
            jnp.asarray(self.measure),
            jnp.asarray(self.normals),
        )


def boundary_facet_mask_for_ids(mesh: "TaggedMesh", ids: Iterable[int]) -> np.ndarray:
    """Boolean mask over ``mesh.boundary_facets()`` of the rows tagged with one of ``ids``."""
    table = _facet_table(mesh)
    tags = np.asarray(mesh.boundary_tags, dtype=np.int64)
    if table.shape[0] == 0 or tags.shape[0] != table.shape[0]:
        return np.zeros(table.shape[0], dtype=bool)
    return np.isin(tags, [int(i) for i in ids])


def boundary_facet_mask_for_nodes(mesh: "TaggedMesh", nodes: Any) -> np.ndarray:
    """Boolean mask of the outer boundary facets whose vertices all lie in ``nodes``.

    A facet shared by two elements (an interface) is never selected by nodes.
    Negative indices count from the end, as in a NumPy index.
    """
    table = _facet_table(mesh)
    if table.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    n = int(np.asarray(mesh.points).shape[0])
    chosen = np.asarray(nodes, dtype=np.int64).reshape(-1)
    chosen = np.where(chosen < 0, chosen + n, chosen)
    in_set = np.zeros(n, dtype=bool)
    in_set[chosen[(chosen >= 0) & (chosen < n)]] = True
    et = _element_type(mesh)
    k = 1 if et == "LINE2" else _local_facets(et).shape[1]
    mask = in_set[table[:, :k]].all(axis=1)
    if et != "LINE2" and mask.any():
        _, _, count = _facet_owners(mesh, table)
        mask &= count == 1
    return mask


def boundary_face_assembly_for_mask(mesh: "TaggedMesh", mask: Any) -> _Triple:
    """Return ``(face_indices, face_measures, outward_normals)`` for a facet mask.

    Args:
        mesh: A [TaggedMesh][diff_soup.domains.TaggedMesh].
        mask: A boolean array with one entry per row of ``mesh.boundary_facets()``.

    Returns:
        The rows described in the module docstring, ``(n_faces, row_width)``;
        the facet measures, ``(n_faces,)`` (a point measure in 1D, a length in 2D,
        an area in 3D); and the outward unit normals, ``(n_faces, dim)``.

    Raises:
        KeyError: If the mask selects no facet.
        NotImplementedError: If the element type has no facet support.
    """
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    table = _facet_table(mesh)
    if mask.shape[0] != table.shape[0]:
        raise ValueError(
            f"a facet mask has one entry per tagged boundary facet ({table.shape[0]}); "
            f"got {mask.shape[0]}"
        )
    if not mask.any():
        raise KeyError("the selection names no tagged boundary facet")
    return _Facets(mesh, mask).triple()


def boundary_face_assembly(mesh: "TaggedMesh", tag: Union[str, int]) -> _Triple:
    """Return ``(face_indices, face_measures, outward_normals)`` for a tagged boundary.

    Args:
        mesh: A [TaggedMesh][diff_soup.domains.TaggedMesh].
        tag: A tag name or tag-group name, or an integer tag id.

    Returns:
        The triple of ``boundary_face_assembly_for_mask``.

    Raises:
        KeyError: If the name or id tags no facet.
        NotImplementedError: If the element type has no facet support.
    """
    ids = mesh._tag_ids_for_name(tag) if isinstance(tag, str) else [int(tag)]
    return boundary_face_assembly_for_ids(mesh, ids)


def boundary_face_assembly_for_ids(mesh: "TaggedMesh", ids: Iterable[int]) -> _Triple:
    """Return ``(face_indices, face_measures, outward_normals)`` for explicit tag ids.

    The facets whose tag id is in ``ids`` form the selection (their union).

    Raises:
        KeyError: If none of ``ids`` tags a facet.
        NotImplementedError: If the element type has no facet support.
    """
    ids = [int(i) for i in ids]
    _element_type(mesh)
    mask = boundary_facet_mask_for_ids(mesh, ids)
    if not mask.any():
        raise KeyError(f"Boundary tag ids {ids} not present in mesh.boundary_tags")
    return _Facets(mesh, mask).triple()


# ---------------------------------------------------------------------------
# Natural boundary load int_f D s N_i dS (the FEM side of spec.bc.neumann)
# ---------------------------------------------------------------------------


def _face_rule(kind: str):
    """Reference points, weights and face shape values/derivatives for one facet kind."""
    if kind in ("edge2", "edge3"):
        t = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
        w = np.array([5.0, 8.0, 5.0]) / 9.0
        if kind == "edge2":
            n = np.stack([(1 - t) / 2, (1 + t) / 2], axis=1)
            dn = np.stack([-0.5 * np.ones_like(t), 0.5 * np.ones_like(t)], axis=1)
        else:  # nodes [a, b, m] at t = -1, 1, 0
            n = np.stack([t * (t - 1) / 2, t * (t + 1) / 2, 1 - t**2], axis=1)
            dn = np.stack([t - 0.5, t + 0.5, -2 * t], axis=1)
        return w, n, dn[:, :, None]
    if kind == "tri":
        r = np.array([1.0 / 6, 2.0 / 3, 1.0 / 6])
        s = np.array([1.0 / 6, 1.0 / 6, 2.0 / 3])
        w = np.full(3, 1.0 / 6)
        n = np.stack([1 - r - s, r, s], axis=1)
        dn = np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (3, 3, 2))
        return w, n, dn
    g = 1.0 / np.sqrt(3.0)  # quad, cyclic corners at (-1,-1), (1,-1), (1,1), (-1,1)
    u = np.array([-g, g, g, -g])
    v = np.array([-g, -g, g, g])
    su, sv = np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0])
    n = 0.25 * (1 + u[:, None] * su) * (1 + v[:, None] * sv)
    dn = np.stack([0.25 * su * (1 + v[:, None] * sv), 0.25 * sv * (1 + u[:, None] * su)], axis=-1)
    return np.ones(4), n, dn


def _surface_jacobian(dn: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``(n_faces, n_q)`` surface measure ``|dx/dt|`` or ``|x_u x x_v|``."""
    jac = np.einsum("qkr,fkd->fqdr", dn, p)
    if jac.shape[-1] == 1:
        return np.linalg.norm(jac[..., 0], axis=-1)
    return np.linalg.norm(np.cross(jac[..., 0], jac[..., 1]), axis=-1)


def _node_to_dof(mesh) -> np.ndarray | None:
    from diff_soup.domains.operators.dispatch import _get_fe_space

    fe = _get_fe_space(mesh)
    if int(fe.num_dofs) == int(np.asarray(mesh.points).shape[0]):
        return None
    mapping = np.arange(np.asarray(mesh.points).shape[0])
    mapping[np.asarray(mesh.cells).reshape(-1)] = np.asarray(fe.dof_cells).reshape(-1)
    return mapping


def boundary_load(
    mesh: "TaggedMesh",
    facets: Any,
    value: Any,
    coefficient: Any = 1.0,
    *,
    op_name: str = "boundary_load",
) -> Float[Array, " num_nodes"]:
    """Nodal load vector ``int_Gamma D s N_i dS`` over a set of boundary facets.

    This is the natural boundary term of ``-div(D grad u)`` with ``du/dn = s``
    on the selected facets (``spec.bc.neumann`` on a mesh).  It is traceable in
    ``value`` and ``coefficient``; the geometry is fixed by ``mesh`` and ``facets``.

    Args:
        mesh: A [TaggedMesh][diff_soup.domains.TaggedMesh].
        facets: A tag name, an integer tag id, or a boolean mask with one entry per
            row of ``mesh.boundary_facets()``.
        value: The normal derivative ``s``, a scalar or one value per selected facet
            in table order.
        coefficient: The diffusivity ``D``, a scalar, one value per node, one per
            cell, or one per quadrature point ``(n_cells, n_q)`` (each cell's mean is used).
        op_name: The caller, named in error messages.

    Returns:
        The load at every mesh node, ``(num_nodes,)``, zero off the selected facets.

    Raises:
        KeyError: If the selection names no facet.
        ValueError: If ``value`` or ``coefficient`` has a shape that fits no reading.
    """
    from diff_soup.domains.operators.dispatch import _get_fe_space, _read_fem_operand

    if isinstance(facets, (str, int, np.integer)):
        ids = mesh._tag_ids_for_name(facets) if isinstance(facets, str) else [int(facets)]
        mask = boundary_facet_mask_for_ids(mesh, ids)
    else:
        mask = np.asarray(facets, dtype=bool).reshape(-1)
    if not mask.any():
        raise KeyError(f"{op_name}: the selection names no tagged boundary facet")
    f = _Facets(mesh, mask)
    n_faces = int(f.rows.shape[0])

    s = jnp.asarray(value)
    if s.ndim == 1 and s.shape[0] == n_faces:
        s = s[:, None]
    elif s.ndim != 0:
        raise ValueError(
            f"{op_name}: the boundary value is a scalar or one value per selected facet "
            f"({n_faces}); got shape {tuple(s.shape)}"
        )

    d, layout = _read_fem_operand(mesh, coefficient, op_name, ambiguous="quad")
    if layout == "quad":
        jxw = _get_fe_space(mesh).jxw
        d_cell = jnp.sum(d * jxw, axis=1) / jnp.sum(jxw, axis=1)
        d_face = d_cell[jnp.asarray(f.cell)][:, None]
    elif layout == "nodal":
        to_dof = _node_to_dof(mesh) if d.shape[0] != f.n_nodes else None
        index = f.nodes if to_dof is None else to_dof[f.nodes]
        d_nodes = d[jnp.asarray(index)]  # (n_faces, n_face_nodes)
    else:
        d_face = d

    if f.kind == "point":
        if layout == "nodal":
            d_face = d_nodes
        load = jnp.broadcast_to(d_face * s * jnp.asarray(f.measure)[:, None], (n_faces, 1))
    else:
        w, n, dn = _face_rule(f.kind)
        p = np.asarray(mesh.points, dtype=np.float64)[f.nodes]
        weight = jnp.asarray(w[None, :] * _surface_jacobian(dn, p))  # (n_faces, n_q)
        if layout == "nodal":
            d_face = jnp.einsum("qk,fk->fq", jnp.asarray(n), d_nodes)
        integrand = jnp.broadcast_to(weight * d_face * s, weight.shape)
        load = jnp.einsum("fq,qk->fk", integrand, jnp.asarray(n))
    out = jnp.zeros((f.n_nodes,), dtype=load.dtype)
    return out.at[jnp.asarray(f.nodes).reshape(-1)].add(load.reshape(-1))


__all__ = [
    "SUPPORTED_ELEMENT_TYPES",
    "boundary_face_assembly",
    "boundary_face_assembly_for_ids",
    "boundary_face_assembly_for_mask",
    "boundary_facet_mask_for_ids",
    "boundary_facet_mask_for_nodes",
    "boundary_load",
]
