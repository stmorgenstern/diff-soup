"""Apply a one-dimensional column kernel along one axis of an N-D grid field.

The N-D grid operators are sums of one-dimensional operators, one per axis
(dimension splitting). [sweep_axis][diff_soup.domains.grid._sweep.sweep_axis]
moves the swept axis to the front, flattens the others into one batch axis and
vmaps the column kernel over it, slicing any grid-shaped coefficient alongside.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


def sweep_axis(kernel: Callable, c: Array, axis: int, *coefficients) -> Array:
    """Apply ``kernel(column, *coefficient_columns)`` along ``axis`` of ``c``.

    Args:
        kernel: A function of one column of ``c`` (and one column of each
            grid-shaped coefficient) returning a column of the same length.
        c: A grid-shaped field.
        axis: The axis the columns run along.
        *coefficients: Each a scalar, passed to every column unchanged, or an
            array shaped like ``c``, sliced into columns with it.

    Returns:
        The kernel's columns reassembled into ``c``'s shape.
    """
    moved = jnp.moveaxis(c, axis, 0)
    shape = moved.shape
    batched = tuple(jnp.ndim(k) == c.ndim for k in coefficients)
    columns = [moved.reshape(shape[0], -1)]
    for k, is_field in zip(coefficients, batched):
        columns.append(jnp.moveaxis(jnp.asarray(k), axis, 0).reshape(shape[0], -1) if is_field else k)
    in_axes = (1,) + tuple(1 if is_field else None for is_field in batched)
    out = jax.vmap(kernel, in_axes=in_axes, out_axes=1)(*columns)
    return jnp.moveaxis(out.reshape(shape), 0, axis)
