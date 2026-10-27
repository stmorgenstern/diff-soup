"""DiscreteField: array + location tag, pytree-registered.

A [DiscreteField][diff_soup.domains.fields.field.DiscreteField] bundles an
array of values with a [Location][diff_soup.domains.fields.field.Location]
tag indicating *where* on a domain those values live.  It is an ``eqx.Module``
with a single traced leaf (the array) and one static metadata field
(``location``); the domain is not stored inside the field and is always
carried separately by the operator or problem context that consumes / produces
the field, following diff_soup's existing ``Operators(domain)`` pattern.

Locations
---------

========================  ============================================
``Location.NODE``         nodal values (FEM, or FD node-based storage)
``Location.CELL_CENTER``  cell-centered values (FVM scalar variables)
``Location.FACE_X``       x-face centered (staggered velocity u_x, x-flux)
``Location.FACE_Y``       y-face centered (staggered velocity u_y, y-flux)
``Location.FACE_Z``       z-face centered (staggered velocity u_z, z-flux)
``Location.QUAD_POINT``   values at FEM quadrature points
========================  ============================================

Component / tensor fields
-------------------------
Vector- and tensor-valued fields are represented by giving the ``array`` an
extra trailing axis.  The component count is implicit in the array shape; it
is not stored as static metadata.  For a 2D staggered velocity this means you
typically carry two separate scalar
[DiscreteField][diff_soup.domains.fields.field.DiscreteField] instances
``(u_x @ FACE_X, u_y @ FACE_Y)`` rather than one vector field, because in a
MAC discretization the components live at *different* locations.
"""

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["Location", "DiscreteField"]


class Location(str, Enum):
    """Where a [DiscreteField][diff_soup.domains.fields.field.DiscreteField] value lives on its domain.

    Inherits from ``str`` so it is hashable and usable as a JIT static arg
    (same pattern as
    [diff_soup.domains.grid.core.CoordSystem][]).
    """

    NODE = "node"
    CELL_CENTER = "cell_center"
    FACE_X = "face_x"
    FACE_Y = "face_y"
    FACE_Z = "face_z"
    QUAD_POINT = "quad_point"

    @property
    def is_face(self) -> bool:
        """True if this location lives on a face of a structured grid."""
        return self in (Location.FACE_X, Location.FACE_Y, Location.FACE_Z)

    @property
    def face_axis(self) -> int:
        """Axis index (0, 1, 2) for face locations; raises for non-face tags."""
        if self is Location.FACE_X:
            return 0
        if self is Location.FACE_Y:
            return 1
        if self is Location.FACE_Z:
            return 2
        raise ValueError(f"{self} is not a face location")


class DiscreteField(eqx.Module):
    """An array of values paired with the location they live at.

    The location is a `Location` from `diff_soup.domains.fields`: `NODE`, `CELL_CENTER`, `FACE_X`, `FACE_Y`, `FACE_Z` or `QUAD_POINT`. The field is a pytree whose one leaf is the array. `replace(array)` returns a copy with new values, and `at(location)` a copy at a new location.

    Args:
        array: The values.
        location: A `Location`, or its string value such as `"face_x"`.
    """

    array: Array
    location: Location = eqx.field(static=True)

    def __init__(self, array, location: Location):
        """Wrap ``array`` as a field at ``location`` (string tags are coerced)."""
        self.array = jnp.asarray(array)
        # Coerce string → enum so callers may pass either form.
        self.location = Location(location) if not isinstance(location, Location) else location

    # ------------------------------------------------------------------
    # Shape / dtype convenience (all derived from ``array``)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying values array."""
        return self.array.shape

    @property
    def dtype(self):
        """Dtype of the underlying values array."""
        return self.array.dtype

    @property
    def ndim(self) -> int:
        """Number of array axes (includes any trailing component axis)."""
        return self.array.ndim

    # ------------------------------------------------------------------
    # Functional updates
    # ------------------------------------------------------------------

    def replace(self, array: Array) -> "DiscreteField":
        """Return a new DiscreteField with the same location and a new array.

        Convenience for operators that transform values but preserve where
        they live (e.g. applying a pointwise nonlinearity).
        """
        return DiscreteField(array, self.location)

    def at(self, location: Location) -> "DiscreteField":
        """Return a new [DiscreteField][diff_soup.domains.fields.field.DiscreteField].

        Relabelled to a different location. This is a pure metadata relabel; no values are moved.  Used by
        reconstruction / interpolation operators that produce a new array
        *and* want to assert a new location.  The usual pattern is:

        >>> face_values = _reconstruct(cell_field.array)  # compute array
        >>> DiscreteField(face_values, Location.FACE_X)   # build directly

        ``at`` is the degenerate case where the array is already correct
        and only the tag needs to change, which is rare but occasionally
        useful for adapters.
        """
        return DiscreteField(self.array, location)


# =========================================================================
# Pytree registration
# =========================================================================
#
# ``eqx.Module`` already registers the class as a pytree.  The array field
# is a traced leaf and the ``location`` field is static (hashed into
# treedef) by virtue of the ``eqx.field(static=True)`` marker above — no
# further registration is required. The one-traced-leaf regression guard
# lives in ``tests/core/domains/test_field_pytree.py`` — import-time array
# creation is forbidden here (it forces device init on bare import; a broken
# CUDA stack then breaks ``import diff_soup.domains`` entirely).
