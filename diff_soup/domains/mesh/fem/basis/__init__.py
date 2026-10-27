"""Reference-element shape functions and quadrature dispatch.

Each supported element type lives in its own module
(`line2`, `tri3`, `tri6`, `quad4`, `quad8`, `tet4`,
`tet10`, `hex8`, `hex20`, `hex27`); shared
quadrature rules live in `_common`.

The two public entry points [get_shape_vals_and_grads][diff_soup.domains.mesh.fem.basis.get_shape_vals_and_grads] and
[get_face_shape_vals_and_grads][diff_soup.domains.mesh.fem.basis.get_face_shape_vals_and_grads] dispatch on the ``ele_type`` string
("TRI3", "QUAD8", ...) and tensor-product the relevant 1D / 2D / 3D
quadrature with the matching shape functions.  They live in
`._dispatch` so this package init stays a thin re-export layer.
"""

from diff_soup.domains.mesh.fem.basis._dispatch import (
    get_face_shape_vals_and_grads,
    get_shape_vals_and_grads,
)

# Re-export the per-element shape functions at the package level so callers
# that previously reached into the flat module via attribute access (e.g.
# ``basis._tri3_shape``) keep working.  These names are still considered
# private (underscore prefix) — the public surface is __all__ above.
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
    # Per-element shape functions, reachable as attributes of this package
    "_hex8_shape",
    "_hex20_shape",
    "_hex27_shape",
    "_line2_shape",
    "_quad4_shape",
    "_quad8_shape",
    "_tet4_shape",
    "_tet10_shape",
    "_tri3_shape",
    "_tri6_shape",
]
