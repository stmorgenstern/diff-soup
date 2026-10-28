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
    ``.inp``, ``.xdmf``, etc.).

    Args:
        path: Path to the mesh file.
        ele_type: diff_soup element type override (e.g. ``"TRI3"``).  If ``None``,
            auto-detected from the first cell block in the file.

    Returns:
        Mesh or TaggedMesh
            A [TaggedMesh][diff_soup.domains.mesh.TaggedMesh] when the file is
            a gmsh ``.msh`` with physical groups; otherwise a plain
            [Mesh][diff_soup.domains.mesh.Mesh].

    Raises:
        ValueError: if the file contains no supported cell types, or the
            specified ``ele_type`` has no matching cells.
    """
    meshio = import_optional("meshio")

    path = Path(path)
    mesh_data = meshio.read(path)

    points = np.asarray(mesh_data.points, dtype=np.float64)

    # Drop z coordinate if all zeros (2D mesh stored with 3D points)
    if points.shape[1] == 3 and np.allclose(points[:, 2], 0.0):
        points = points[:, :2]

    # --- Determine cell block to use ---
    if ele_type is not None:
        # User-specified: find matching cell block
        meshio_type = _DS_TO_MESHIO.get(ele_type)
        if meshio_type is None:
            raise ValueError(
                f"Unknown ele_type {ele_type!r}. Supported: {list(_DS_TO_MESHIO.keys())}"
            )
        block = None
        for cb in mesh_data.cells:
            if cb.type == meshio_type:
                block = cb
                break
        if block is None:
            raise ValueError(
                f"No cell block of type {meshio_type!r} found in {path}. "
                f"Available: {[cb.type for cb in mesh_data.cells]}"
            )
        ds_ele_type = ele_type
    else:
        # Auto-detect from first supported cell block
        block = None
        ds_ele_type = None
        for cb in mesh_data.cells:
            if cb.type in _MESHIO_TO_DS:
                block = cb
                ds_ele_type = _MESHIO_TO_DS[cb.type]
                break
        if block is None:
            raise ValueError(
                f"No supported cell types found in {path}. "
                f"Available: {[cb.type for cb in mesh_data.cells]}. "
                f"Supported: {list(_MESHIO_TO_DS.keys())}"
            )

    cells = np.asarray(block.data, dtype=np.int32)

    # --- Check for gmsh physical groups (tagged mesh) ---
    is_gmsh = path.suffix.lower() == ".msh"
    if is_gmsh and mesh_data.field_data:
        # meshio stores physical groups in field_data:
        #   field_data[name] = [physical_tag, topological_dim]
        # and cell_sets or cell_sets_dict for group membership.
        #
        # Extract 1D (edge) physical groups for boundary tagging.
        tag_names: Dict[int, str] = {}
        for name, (phys_tag, topo_dim) in mesh_data.field_data.items():
            tag_names[int(phys_tag)] = name

        # Find boundary edge cells (1D line elements)
        edge_cells = None
        edge_block_idx = None
        for idx, cb in enumerate(mesh_data.cells):
            if cb.type == "line":
                edge_cells = np.asarray(cb.data, dtype=np.int32)
                edge_block_idx = idx
                break

        if edge_cells is not None and edge_block_idx is not None:
            # Get per-cell physical tags for the edge block
            # meshio stores this in cell_data["gmsh:physical"]
            edge_tags = None
            if "gmsh:physical" in mesh_data.cell_data:
                phys_data = mesh_data.cell_data["gmsh:physical"]
                if edge_block_idx < len(phys_data):
                    edge_tags = np.asarray(phys_data[edge_block_idx], dtype=np.int32)

            if edge_tags is None:
                edge_tags = np.zeros(edge_cells.shape[0], dtype=np.int32)

            return TaggedMesh(
                points=points,
                cells=cells,
                ele_type=ds_ele_type,
                boundary_edges=edge_cells,
                boundary_tags=edge_tags,
                tag_names=tag_names,
            )

    return Mesh(points=points, cells=cells, ele_type=ds_ele_type)


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
