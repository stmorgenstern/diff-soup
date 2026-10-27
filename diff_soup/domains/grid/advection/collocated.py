r"""First-order upwind advection on collocated (cell/node-centered) cartesian grids.

These kernels approximate the advective term :math:`\mathbf{v}\cdot\nabla u`
directly on a collocated grid — the field and every velocity component live at
the same cell centers / nodes.  The value is differenced in place, one axis at
a time.

Because the advective (non-conservative) form reads only nodal values and the
distance between adjacent collocation points, the numerics are identical for the
finite-volume and finite-difference dispatches — there is no separate flux vs.
stencil form as there is for diffusion.  A single kernel therefore serves both
``method="fvm"`` and ``method="fdm"`` cartesian grids.

Upwind rule (per axis, for the term :math:`v\,\partial u/\partial x`):

* ``v >= 0`` uses the backward difference ``(u[i] - u[i-1]) / h`` (information
  travels from the low side);
* ``v < 0`` uses the forward difference ``(u[i+1] - u[i]) / h``.

The two sides are blended branchlessly through ``v_+ = max(v, 0)`` and
``v_- = min(v, 0)`` so a per-axis sign flip needs no control flow.

Boundary handling:

* **periodic** axes take neighbours across the domain edge (``jnp.roll``
  semantics), with the wrap distance equal to the two half-cells that meet at the
  boundary;
* **non-periodic** axes use a zero-gradient (outflow / copy-out) condition — the
  ghost value equals the boundary value, so the one-sided difference that would
  reach outside the domain is exactly zero there.

The grid stores its collocation-point spacings in double precision; the
operators cast them, and a velocity wider than the field, to the field's dtype
before differencing -- see
[grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype] -- so a
single-precision field is advected in single-precision arithmetic.

All functions are pure, JIT-, grad-, and vmap-safe.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diff_soup.domains.grid import Grid1D, GridND
from diff_soup.domains.grid._dtype import (
    grid_in_field_dtype,
    narrow_to_field_dtype,
    to_field_dtype,
)

__all__ = [
    "advect_upwind_1d",
    "advect_upwind_nd",
    "normalize_velocity",
]


def normalize_velocity(velocity: Any, dim: int) -> tuple[Array, ...]:
    """Resolve a velocity argument to one constant speed per grid axis.

    Accepts a single scalar (the same speed on every axis) or a length-``dim``
    sequence / 1-D array of per-axis constant speeds.  Spatially varying
    velocity fields are not supported.

    Args:
        velocity: A python/JAX scalar, or a length-``dim`` sequence / 1-D array
            of per-axis scalars.
        dim: Number of grid axes.

    Returns:
        A ``dim``-tuple of 0-D JAX arrays, one speed per axis.

    Raises:
        ValueError: if ``velocity`` is a sequence of the wrong length, carries a
            non-scalar (spatially varying) per-axis entry, or is an array whose
            shape is neither scalar nor ``(dim,)``.
    """
    if isinstance(velocity, (list, tuple)):
        if len(velocity) != dim:
            raise ValueError(
                "advect_upwind: a per-axis velocity sequence must give one speed "
                f"per grid axis (length {dim} for this {dim}-D grid); got length "
                f"{len(velocity)}."
            )
        comps = []
        for vk in velocity:
            vk_arr = jnp.asarray(vk)
            if vk_arr.ndim != 0:
                raise ValueError(
                    "advect_upwind: each per-axis velocity must be a constant scalar; "
                    f"got an array of shape {tuple(vk_arr.shape)}. Spatially varying "
                    "velocity fields are not supported — pass one constant speed per axis."
                )
            comps.append(vk_arr)
        return tuple(comps)

    v_arr = jnp.asarray(velocity)
    if v_arr.ndim == 0:
        return tuple(v_arr for _ in range(dim))
    if v_arr.ndim == 1 and v_arr.shape[0] == dim:
        return tuple(v_arr[k] for k in range(dim))
    raise ValueError(
        "advect_upwind: velocity must be a constant scalar (the same speed on every "
        f"axis) or a length-{dim} sequence of per-axis constant speeds for this "
        f"{dim}-D grid; got an array of shape {tuple(v_arr.shape)}. Spatially varying "
        "velocity fields are not supported."
    )


def advect_upwind_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    velocity: Float[Array, ""],
    *,
    periodic: bool = False,
) -> Float[Array, "n"]:
    r"""First-order upwind approximation of :math:`v\,\partial c/\partial x`.

    Args:
        grid: 1D collocated grid (``method`` ``"fvm"`` or ``"fdm"``).
        c: Field values at the collocation points, shape ``(n,)``.
        velocity: Constant scalar advection speed ``v`` (may be signed).
        periodic: If ``True``, neighbours wrap across the domain edge; otherwise
            a zero-gradient (outflow) condition is applied at both boundaries.

    Returns:
        The upwind ``v * dc/dx`` at each collocation point, shape ``(n,)``, in
        ``c``'s dtype.  Note this is the advective operator itself, *not* the
        ``-v * dc/dx`` right-hand-side contribution.
    """
    grid = grid_in_field_dtype(grid, c)
    n = grid.n
    if n < 2:
        return jnp.zeros_like(c)

    a = narrow_to_field_dtype(velocity, c)
    a_pos = 0.5 * (a + jnp.abs(a))  # >= 0 part -> backward difference
    a_neg = 0.5 * (a - jnp.abs(a))  # <= 0 part -> forward difference

    nd = grid.node_distances  # (n-1,) distance between adjacent collocation points

    if periodic:
        # Distance across the periodic seam: the two half-cells that meet there.
        wrap = 0.5 * grid.cell_widths[0] + 0.5 * grid.cell_widths[-1]
        h_back = jnp.concatenate([wrap[None], nd])  # gap from i-1 to i
        h_fwd = jnp.concatenate([nd, wrap[None]])  # gap from i to i+1
        c_back = jnp.roll(c, 1)  # c[i-1], wraps at i=0
        c_fwd = jnp.roll(c, -1)  # c[i+1], wraps at i=n-1
    else:
        # Zero-gradient boundaries: ghost = boundary value, so the outward-
        # reaching one-sided difference vanishes there.  The dummy boundary
        # spacings never matter (their numerators are zero).
        h_back = jnp.concatenate([nd[:1], nd])
        h_fwd = jnp.concatenate([nd, nd[-1:]])
        c_back = jnp.concatenate([c[:1], c[:-1]])
        c_fwd = jnp.concatenate([c[1:], c[-1:]])

    back_diff = (c - c_back) / h_back
    fwd_diff = (c_fwd - c) / h_fwd
    return to_field_dtype(a_pos * back_diff + a_neg * fwd_diff, c)


def advect_upwind_nd(
    grid: GridND,
    c: Float[Array, "..."],
    velocity_per_axis: tuple[Array, ...],
    periodic_axes: tuple[int, ...] = (),
) -> Float[Array, "..."]:
    r"""First-order upwind approximation of :math:`\mathbf{v}\cdot\nabla c` on an N-D grid.

    The advective operator is a sum of per-axis terms
    :math:`\sum_d v_d\,\partial c/\partial x_d`, so each axis is differenced by
    [advect_upwind_1d][diff_soup.domains.grid.advection.collocated.advect_upwind_1d]
    (moved to the front and vmapped over the remaining
    axes) and the contributions are accumulated.

    Args:
        grid: N-D cartesian grid.
        c: Field values, grid-shaped ``(nx, ny[, nz])``.
        velocity_per_axis: One constant scalar speed per axis (see
            [normalize_velocity][diff_soup.domains.grid.advection.collocated.normalize_velocity]).
        periodic_axes: Axes whose neighbours wrap across the domain edge; every
            other axis uses the zero-gradient boundary.

    Returns:
        The upwind ``v . grad c`` at each collocation point, grid-shaped, in
        ``c``'s dtype.
    """
    out = jnp.zeros_like(c)
    for d in range(grid.dim):
        ax = grid.axes[d]
        vd = velocity_per_axis[d]
        per = d in periodic_axes
        c_m = jnp.moveaxis(c, d, 0)
        sh = c_m.shape
        c_f = c_m.reshape(sh[0], -1)
        contrib_f = jax.vmap(
            lambda col, _vd=vd, _ax=ax, _per=per: advect_upwind_1d(_ax, col, _vd, periodic=_per),
            in_axes=1,
            out_axes=1,
        )(c_f)
        out = out + jnp.moveaxis(contrib_f.reshape(sh), 0, d)
    return out
