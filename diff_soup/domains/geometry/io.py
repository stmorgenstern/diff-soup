"""geometry/io.py — Read/write meshes via meshio.

Supports any format meshio handles (.msh, .vtk, .vtu, .inp, .xdmf, etc.).
Requires ``meshio``, from the ``mesh`` extra::

    pip install "diff-soup[mesh]"
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from diff_soup._optional import import_optional
from diff_soup.domains.mesh import Mesh, TaggedMesh

# meshio cell type -> diff_soup ele_type
_MESHIO_TO_DS: Dict[str, str] = {
    "line": "LINE2",
    "triangle": "TRI3",
    "triangle6": "TRI6",
    "quad": "QUAD4",
    "quad8": "QUAD8",
    "tetra": "TET4",
    "tetra10": "TET10",
    "hexahedron": "HEX8",
    "hexahedron20": "HEX20",
    "hexahedron27": "HEX27",
}

# Reverse mapping: diff_soup ele_type -> meshio cell type
_DS_TO_MESHIO: Dict[str, str] = {v: k for k, v in _MESHIO_TO_DS.items()}

# Topological dimension of each meshio cell type read as cells or facets.
_CELL_DIM: Dict[str, int] = {
    "vertex": 0,
    "line": 1,
    "line3": 1,
    "triangle": 2,
    "triangle6": 2,
    "quad": 2,
    "quad8": 2,
    "quad9": 2,
    "tetra": 3,
    "tetra10": 3,
    "hexahedron": 3,
    "hexahedron20": 3,
    "hexahedron27": 3,
}

# The meshio facet type of each diff_soup element type.
_FACET_TYPES: Dict[str, tuple] = {
    "LINE2": ("vertex",),
    "TRI3": ("line",),
    "QUAD4": ("line",),
    "TRI6": ("line3",),
    "QUAD8": ("line3",),
    "TET4": ("triangle",),
    "HEX8": ("quad",),
    "TET10": ("triangle6",),
    "HEX20": ("quad8",),
    "HEX27": ("quad9",),
}


# ======================================================================
# Reading
# ======================================================================


def read_mesh(
    path: Union[str, Path],
    *,
    ele_type: Optional[str] = None,
) -> Union[Mesh, TaggedMesh]:
    """Read a mesh file using meshio.

    Supports any format meshio handles (``.msh``, ``.vtk``, ``.vtu``,
    ``.inp``, ``.xdmf``, etc.).  Every cell block of the element type is read.
    A gmsh ``.msh`` file with physical groups gives a
    [TaggedMesh][diff_soup.domains.mesh.TaggedMesh]: the groups one dimension
    below the cells tag the boundary facets, and the groups of the cells'
    own dimension become ``cell_tags`` and ``cell_tag_names``.

    Args:
        path: Path to the mesh file.
        ele_type: diff_soup element type to read (e.g. ``"TRI3"``).  If ``None``,
            the cells of the highest dimension in the file.

    Returns:
        Mesh or TaggedMesh
            A [TaggedMesh][diff_soup.domains.mesh.TaggedMesh] when the file is
            a gmsh ``.msh`` with physical groups; otherwise a plain
            [Mesh][diff_soup.domains.mesh.Mesh].

    Raises:
        ValueError: if the file contains no supported cell types, the
            specified ``ele_type`` has no matching cells, or the highest
            dimension mixes two element types and ``ele_type`` is not given.
    """
    meshio = import_optional("meshio")

    path = Path(path)
    mesh_data = meshio.read(path)

    points = np.asarray(mesh_data.points, dtype=np.float64)

    # Drop z coordinate if all zeros (2D mesh stored with 3D points)
    if points.shape[1] == 3 and np.allclose(points[:, 2], 0.0):
        points = points[:, :2]

    blocks = list(mesh_data.cells)
    if ele_type is not None:
        meshio_type = _DS_TO_MESHIO.get(ele_type)
        if meshio_type is None:
            raise ValueError(
                f"Unknown ele_type {ele_type!r}. Supported: {list(_DS_TO_MESHIO.keys())}"
            )
        if not any(cb.type == meshio_type for cb in blocks):
            raise ValueError(
                f"No cell block of type {meshio_type!r} found in {path}. "
                f"Available: {[cb.type for cb in blocks]}"
            )
        ds_ele_type = ele_type
    else:
        supported = [cb.type for cb in blocks if cb.type in _MESHIO_TO_DS]
        if not supported:
            raise ValueError(
                f"No supported cell types found in {path}. "
                f"Available: {[cb.type for cb in blocks]}. "
                f"Supported: {list(_MESHIO_TO_DS.keys())}"
            )
        top = max(_CELL_DIM[t] for t in supported)
        top_types = sorted({t for t in supported if _CELL_DIM[t] == top})
        if len(top_types) > 1:
            raise ValueError(f"{path} mixes {top_types} cells; pass ele_type= to read one kind")
        meshio_type = top_types[0]
        ds_ele_type = _MESHIO_TO_DS[meshio_type]

    physical = mesh_data.cell_data.get("gmsh:physical")

    def gather(types):
        index = [i for i, cb in enumerate(blocks) if cb.type in types]
        if not index:
            return None, None
        data = np.concatenate([np.asarray(blocks[i].data, dtype=np.int64) for i in index])
        if "tetra10" in types:
            # meshio hands tetra10 in VTK order (node 8 on edge 1-3, node 9 on edge 2-3);
            # the TET10 basis uses the gmsh order, which swaps the two.
            data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
        if physical is None:
            return data, None
        tags = np.concatenate([np.asarray(physical[i], dtype=np.int64) for i in index])
        return data, tags

    cells, cell_tags = gather((meshio_type,))
    facets, facet_tags = gather(_FACET_TYPES.get(ds_ele_type, ()))

    is_gmsh = path.suffix.lower() == ".msh"
    if not is_gmsh or physical is None:
        return Mesh(points=points, cells=cells.astype(np.int32), ele_type=ds_ele_type)

    # field_data[name] = [physical_tag, topological_dim]; gmsh numbers the
    # physical groups of each dimension separately.
    dim = _CELL_DIM[meshio_type]
    facet_names: Dict[int, str] = {}
    region_names: Dict[str, int] = {}
    present = set(np.unique(cell_tags).tolist())
    for name, (phys_tag, topo_dim) in mesh_data.field_data.items():
        if int(topo_dim) == dim - 1:
            facet_names[int(phys_tag)] = name
        elif int(topo_dim) == dim and int(phys_tag) in present:
            region_names[name] = int(phys_tag)
    if facets is None:
        facets = np.empty((0, dim if dim > 1 else 1), dtype=np.int64)
        facet_tags = np.empty((0,), dtype=np.int64)
    table = {"boundary_edges" if dim == 2 else "boundary_faces": facets.astype(np.int32)}
    return TaggedMesh(
        points=points,
        cells=cells.astype(np.int32),
        ele_type=ds_ele_type,
        boundary_tags=facet_tags.astype(np.int32),
        tag_names=facet_names or None,
        cell_tags=cell_tags.astype(np.int32),
        cell_tag_names=region_names or None,
        **table,
    )


# ======================================================================
# Writing
# ======================================================================


def write_mesh(
    path: Union[str, Path],
    mesh: Mesh,
    *,
    point_data: Optional[Dict[str, np.ndarray]] = None,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a mesh to file using meshio.

    The output format is inferred from the file extension.

    Args:
        path: Output file path (e.g. ``"out.vtu"``, ``"out.msh"``).
        mesh: [Mesh][diff_soup.domains.mesh.Mesh] to write.
        point_data: Optional dict mapping field names to per-node arrays of shape
            ``(n_nodes,)`` or ``(n_nodes, dim)``.
        cell_data: Optional dict mapping field names to per-cell arrays of shape
            ``(n_cells,)``.  Each value is wrapped in a single-element list
            as required by meshio.
    """
    meshio = import_optional("meshio")

    meshio_type = _DS_TO_MESHIO.get(mesh.ele_type)
    if meshio_type is None:
        raise ValueError(
            f"Cannot map ele_type {mesh.ele_type!r} to a meshio cell type. "
            f"Supported: {list(_DS_TO_MESHIO.keys())}"
        )

    pts = np.asarray(mesh.points, dtype=np.float64)
    # meshio expects 3D points
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])

    cells_list = [(meshio_type, np.asarray(mesh.cells))]

    # Wrap cell_data values in a list (meshio convention: one entry per block)
    cd = None
    if cell_data is not None:
        cd = {name: [np.asarray(arr)] for name, arr in cell_data.items()}

    pd = None
    if point_data is not None:
        pd = {name: np.asarray(arr) for name, arr in point_data.items()}

    meshio_mesh = meshio.Mesh(
        points=pts,
        cells=cells_list,
        point_data=pd,
        cell_data=cd,
    )
    meshio_mesh.write(str(path))


# ======================================================================
# VTU convenience
# ======================================================================


def write_vtu(
    path: Union[str, Path],
    mesh: Mesh,
    fields: Dict[str, np.ndarray],
) -> None:
    """Write mesh + fields to ParaView-compatible VTU format.

    Args:
        path: Output file path.  Should end in ``.vtu``.
        mesh: [Mesh][diff_soup.domains.mesh.Mesh] to write.
        fields: Dict mapping names to per-node arrays.  Scalar fields have shape
            ``(n_nodes,)``; vector fields have shape ``(n_nodes, vec)``.
            Multi-component fields (``vec > 1``) are split into named
            components: e.g. ``"u"`` with shape ``(n, 2)`` becomes
            ``"u_0"`` and ``"u_1"``.
    """
    path = Path(path)
    if path.suffix.lower() != ".vtu":
        path = path.with_suffix(".vtu")

    point_data: Dict[str, np.ndarray] = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            point_data[name] = arr
        elif arr.ndim == 2 and arr.shape[1] > 1:
            for comp in range(arr.shape[1]):
                point_data[f"{name}_{comp}"] = arr[:, comp]
        else:
            # shape (n, 1) — squeeze to scalar
            point_data[name] = arr.ravel()

    write_mesh(path, mesh, point_data=point_data)
