"""Mesh primitives, factories, and finite-element infrastructure.

The [core][diff_soup.domains.mesh.core] module owns the
[Mesh][diff_soup.domains.mesh.Mesh] and
[TaggedMesh][diff_soup.domains.mesh.TaggedMesh] pytree types;
[fem][diff_soup.domains.mesh.fem] owns finite-element spaces, basis
evaluation, and assembly helpers.

Only the lightweight pytree types and the pure-NumPy factories are re-exported
here.  The FEM subpackage remains importable by its qualified name
(``diff_soup.domains.mesh.fem``); it pulls in heavier dependencies, so it
stays out of the package import.  The gmsh-based
mesh generators live in [diff_soup.domains.geometry][] and are surfaced
through the [diff_soup.meshes][] facade.
"""

from diff_soup.domains.mesh.core import (
    Mesh,
    TaggedMesh,
    box_mesh,
    cylindrical_interval_mesh,
    interval_mesh,
    rectangle_mesh,
    spherical_interval_mesh,
)
from diff_soup.domains.mesh.helpers import select_rigid_body_pins

__all__ = [
    "Mesh",
    "TaggedMesh",
    "box_mesh",
    "cylindrical_interval_mesh",
    "interval_mesh",
    "rectangle_mesh",
    "spherical_interval_mesh",
    "select_rigid_body_pins",
]
