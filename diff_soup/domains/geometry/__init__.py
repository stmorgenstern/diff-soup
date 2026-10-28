"""diff_soup.domains.geometry — Complex geometry support via OpenCV + gmsh.

``opencv-python`` installs with diff_soup. Meshing with ``gmsh`` and reading
or writing meshes with ``meshio`` need the ``mesh`` extra,
``pip install "diff-soup[mesh]"``.

Typical workflow::

    from diff_soup.domains.geometry import mask_to_mesh

    mesh = mask_to_mesh(binary_mask_np, mesh_size=0.5)
    # mesh is a diff_soup.domains.Mesh with triangular elements
"""

from .io import read_mesh, write_mesh, write_vtu
from .mask import mask_to_polygon
from .meshing import (
    box_mesh_tet,
    cylinder_mesh,
    extract_boundary_edges,
    mask_to_mesh,
    mesh_quality,
    polygon_to_mesh,
    polygon_to_tagged_mesh,
    rectangle_mesh_quad,
    sphere_mesh,
)
from .refinement import (
    refine_uniform,
    transfer_solution_refined,
)

# Not exported: csg_mesh, surface_to_tet_mesh (CSG path removed — no live callers);
# kelly_error_estimate, refine_adaptive, remesh_adaptive (AMR path removed — no live callers).

__all__ = [
    "mask_to_polygon",
    "polygon_to_mesh",
    "polygon_to_tagged_mesh",
    "rectangle_mesh_quad",
    "mask_to_mesh",
    "extract_boundary_edges",
    "mesh_quality",
    "read_mesh",
    "write_mesh",
    "write_vtu",
    "box_mesh_tet",
    "cylinder_mesh",
    "sphere_mesh",
    "refine_uniform",
    "transfer_solution_refined",
]
