"""Run grid stencils in the dtype of the fields on the grid.

A :class:`~diff_soup.domains.grid.Grid1D` stores its geometry -- node
positions, spacings, cell volumes, face areas, and the curvilinear metric
corrections -- in double precision, whatever dtype the fields living on it use.
The stencils do their arithmetic in the field's dtype instead: every kernel
first views the grid through :func:`grid_in_field_dtype`, so a single-precision
field is discretized with single-precision geometry and single-precision
arithmetic, and a double-precision field sees the stored geometry unchanged.

:func:`to_field_dtype` covers the remaining inputs a stencil term is built
from -- a diffusivity or boundary value that arrives in a wider dtype than the
field -- so a term always lands in the field's dtype. Writing a wider term into
a narrower array would be a narrowing scatter, which JAX reports today and will
reject in a future release.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

__all__ = ["grid_in_field_dtype", "narrow_to_field_dtype", "to_field_dtype"]


def grid_in_field_dtype(grid, field: Float[Array, "..."]):
    """View a grid with its geometry in the dtype of ``field``.

    Every floating-point array of the grid (for a tensor-product grid, of each
    axis) is cast to ``field``'s dtype. Arrays already in that dtype are
    returned as they are, so a double-precision field on a double-precision
    grid uses the stored geometry bit for bit. A field that is not floating
    point leaves the grid unchanged.

    Args:
        grid: A [Grid1D][diff_soup.domains.grid.Grid1D] or
            [GridND][diff_soup.domains.grid.GridND].
        field: The field the stencil acts on. Only its dtype is read.

    Returns:
        The grid, with its floating-point geometry in ``field``'s dtype.
    """
    dtype = jnp.result_type(field)
    if not jnp.issubdtype(dtype, jnp.floating):
        return grid

    def cast(leaf):
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is None or leaf_dtype == dtype:
            return leaf
        if not jnp.issubdtype(leaf_dtype, jnp.floating):
            return leaf
        return jnp.asarray(leaf).astype(dtype)

    return jax.tree_util.tree_map(cast, grid)


def narrow_to_field_dtype(value, field: Float[Array, "..."]) -> Float[Array, "..."]:
    """Cast a stencil coefficient to ``field``'s dtype only if it is wider.

    A coefficient narrower than the field (a single-precision diffusivity on a
    double-precision field) is left as it is, and the arithmetic promotes it to
    the field's dtype; a wider one (a double-precision velocity on a
    single-precision field) is cast so the stencil stays in the field's dtype.

    Args:
        value: A stencil coefficient, scalar or array.
        field: The field the stencil acts on. Only its dtype is read.

    Returns:
        ``value`` as an array, in ``field``'s dtype when it was wider.
    """
    value = jnp.asarray(value)
    if jnp.promote_types(value.dtype, field.dtype) != field.dtype:
        return value.astype(field.dtype)
    return value


def to_field_dtype(
    value: Float[ArrayLike, "..."],
    field: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Cast a stencil input or term into the dtype of the field it belongs to.

    Stencils cast their coefficient inputs (a diffusivity, a boundary flux)
    with this before the arithmetic, and the terms they write back into a
    field array after it, so an input supplied in a wider dtype neither widens
    the computation nor makes the write a narrowing scatter. When the value is
    already in the field's dtype the cast is a no-op.

    Args:
        value: An input or term of a stencil.
        field: The array whose dtype the value must take -- the array the
            term is scattered into, or the operator's own input field.

    Returns:
        ``value`` in ``field``'s dtype.
    """
    return jnp.asarray(value).astype(field.dtype)
