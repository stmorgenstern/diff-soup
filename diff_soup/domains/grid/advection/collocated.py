r"""Upwind, central, TVD and WENO advection on collocated (cell/node-centered) cartesian grids.

These kernels approximate the advective term :math:`\mathbf{v}\cdot\nabla u`
directly on a collocated grid — the field and every velocity component live at
the same cell centers / nodes.  The value is differenced in place, one axis at
a time.  [div_advect_1d][diff_soup.domains.grid.advection.collocated.div_advect_1d]
and [div_advect_nd][diff_soup.domains.grid.advection.collocated.div_advect_nd]
give the conservative form :math:`\nabla\cdot(\mathbf{v} u)` from flux-limited
(TVD), WENO3, central or upwind face fluxes.  A velocity is a constant per axis
or a cell-centred field per axis.

Because the advective (non-conservative) form reads only nodal values and the
distance between adjacent collocation points, the numerics are identical for the
finite-volume and finite-difference dispatches — there is no separate flux vs.
stencil form as there is for diffusion.  A single kernel therefore serves both
``method="fvm"`` and ``method="fdm"`` cartesian grids.

Upwind rule of the advective form (per axis, for the term :math:`v\,\partial u/\partial x`):

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

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from diff_soup.domains.grid import Grid1D, GridND
from diff_soup.domains.grid._dtype import (
    grid_in_field_dtype,
    narrow_to_field_dtype,
    to_field_dtype,
)
from diff_soup.domains.grid._sweep import sweep_axis

__all__ = [
    "SCHEMES",
    "LIMITERS",
    "advect_upwind_1d",
    "advect_upwind_nd",
    "div_advect_1d",
    "div_advect_nd",
    "normalize_velocity",
]


SCHEMES = ("tvd", "weno3", "central", "upwind")
LIMITERS = ("van_leer", "koren", "minmod")

# The van Leer slope divides by |back| + |fwd|; this guard keeps a flat region
# (0/0) at a zero slope with finite derivatives, its square a normal float32.
_VAN_LEER_EPS = 1e-18
# The WENO3 smoothness guard; Jiang and Shu used 1e-6, and 1e-10 lets the weights
# see jumps in small fields too.
_WENO_EPS = 1e-10


def _limited_change(limiter: str, back: Array, fwd: Array) -> Array:
    """``phi(r) * fwd`` with ``r = back / fwd``, the limited change from the upwind cell.

    ``back`` is the difference from the cell upstream of the upwind cell and
    ``fwd`` the difference to the downwind cell.  Koren and minmod are written
    without a division; van Leer's harmonic mean carries ``_VAN_LEER_EPS``.
    """
    if limiter == "van_leer":
        return (back * jnp.abs(fwd) + jnp.abs(back) * fwd) / (
            jnp.abs(back) + jnp.abs(fwd) + _VAN_LEER_EPS
        )
    s = jnp.sign(fwd)
    a, b = s * back, jnp.abs(fwd)  # r = a / b
    if limiter == "koren":
        return s * jnp.maximum(0.0, jnp.minimum(jnp.minimum(2.0 * a, (b + 2.0 * a) / 3.0), 2.0 * b))
    return s * jnp.maximum(0.0, jnp.minimum(a, b))


def _face_value(scheme: str, limiter: str, far: Array, up: Array, down: Array) -> Array:
    """The upwind-biased face value from the cells ``far``, ``up`` and ``down`` of the flow."""
    if scheme == "tvd":
        return up + 0.5 * _limited_change(limiter, up - far, down - up)
    # WENO3-JS weights d_k / (eps + beta_k)**2, multiplied through by both squares
    # and by 1 / (sum of both) so a flat float32 region keeps finite derivatives.
    beta_far, beta_near = _WENO_EPS + (up - far) ** 2, _WENO_EPS + (down - up) ** 2
    total = beta_far + beta_near
    w_far = (1.0 / 3.0) * (beta_near / total) ** 2
    w_near = (2.0 / 3.0) * (beta_far / total) ** 2
    return (w_far * (1.5 * up - 0.5 * far) + w_near * 0.5 * (up + down)) / (w_far + w_near)


def normalize_velocity(
    velocity: Any,
    dim: int,
    shape: "tuple[int, ...] | None" = None,
    *,
    op_name: str = "advect_upwind",
) -> tuple[Array, ...]:
    """Resolve a velocity argument to one entry per grid axis.

    Accepts a single scalar (the same speed on every axis), a length-``dim``
    sequence or 1-D array of per-axis scalars, or a length-``dim`` sequence
    whose entries are scalars or cell-centred arrays (a velocity field, one
    value per cell, flat or grid-shaped).  On a one-dimensional grid a bare
    array of one value per cell is the field itself.

    Args:
        velocity: The velocity argument as passed to the operator.
        dim: Number of grid axes.
        shape: The grid shape, which a velocity field must match; ``None``
            refuses fields.
        op_name: The operator name used in error messages.

    Returns:
        A ``dim``-tuple: a 0-D array per constant axis, a grid-shaped array per
        field axis.

    Raises:
        ValueError: if ``velocity`` is a sequence of the wrong length, or an
            entry or array whose shape fits neither form.
    """
    n_cells = None if shape is None else int(np.prod(shape))

    def as_field(v):
        if shape is not None and v.ndim == 1 and v.shape[0] == n_cells:
            return v.reshape(shape)
        if shape is not None and tuple(v.shape) == tuple(shape):
            return v
        return None

    if isinstance(velocity, (list, tuple)):
        if len(velocity) != dim:
            raise ValueError(
                f"{op_name}: a per-axis velocity sequence must give one entry per grid "
                f"axis (length {dim} for this {dim}-D grid); got length {len(velocity)}."
            )
        comps = []
        for vk in velocity:
            vk_arr = jnp.asarray(vk)
            if vk_arr.ndim == 0:
                comps.append(vk_arr)
                continue
            field = as_field(vk_arr)
            if field is None:
                raise ValueError(
                    f"{op_name}: each per-axis velocity must be a constant or one value "
                    f"per cell (shape {tuple(shape) if shape else '(num_nodes,)'}, or "
                    f"({n_cells},) flat); got an array of shape {tuple(vk_arr.shape)}."
                )
            comps.append(field)
        return tuple(comps)

    v_arr = jnp.asarray(velocity)
    if v_arr.ndim == 0:
        return tuple(v_arr for _ in range(dim))
    if dim == 1 and as_field(v_arr) is not None and v_arr.shape[0] != 1:
        return (as_field(v_arr),)
    if v_arr.ndim == 1 and v_arr.shape[0] == dim:
        return tuple(v_arr[k] for k in range(dim))
    raise ValueError(
        f"{op_name}: velocity must be a constant (the same speed on every axis) or a "
        f"length-{dim} sequence with one entry per axis, each a constant or one value "
        f"per cell; got an array of shape {tuple(v_arr.shape)}. Pass a velocity field "
        "as a tuple, one array per axis."
    )


def advect_upwind_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    velocity: Float[Array, "..."],
    *,
    periodic: bool = False,
) -> Float[Array, "n"]:
    r"""First-order upwind approximation of :math:`v\,\partial c/\partial x`.

    Args:
        grid: 1D collocated grid (``method`` ``"fvm"`` or ``"fdm"``).
        c: Field values at the collocation points, shape ``(n,)``.
        velocity: Advection speed ``v`` (may be signed), a constant or one value
            per cell.
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
        velocity_per_axis: One entry per axis, a constant or a grid-shaped field
            (see
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
        per = d in periodic_axes
        out = out + sweep_axis(
            lambda col, vcol, _ax=ax, _per=per: advect_upwind_1d(_ax, col, vcol, periodic=_per),
            c,
            d,
            velocity_per_axis[d],
        )
    return out


def div_advect_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    velocity: Float[Array, "..."],
    *,
    periodic: bool = False,
    scheme: str = "tvd",
    limiter: str = "van_leer",
) -> Float[Array, "n"]:
    r"""Conservative approximation of :math:`\partial (v c)/\partial x`.

    Each interior face carries ``v_f c_f``: ``v_f`` is the velocity at the face
    (the mean of the two neighbouring cell velocities for a velocity field) and
    ``c_f`` a value reconstructed from the side the flow comes from, by the sign
    of ``v_f``.  Under ``"upwind"`` ``c_f`` is the upstream cell value.  Under
    ``"tvd"`` it is the upstream value plus half the limited change
    ``phi(r) (c_down - c_up)``, with ``r`` the ratio of the upstream difference
    to the downstream one (van Leer, Koren or minmod ``phi``).  Under
    ``"weno3"`` it is the third-order WENO-JS blend of the two two-cell
    reconstructions, with smoothness guard ``1e-10``.  Under ``"central"`` the
    face carries the mean of the two neighbouring cell fluxes ``v c`` instead.
    An interior face whose upstream cell is a boundary cell has no second
    upstream cell, and ``"tvd"`` and ``"weno3"`` take the upstream value there.

    Under every scheme a boundary face carries the outgoing flux ``v_b c_b``
    when the flow leaves through it and nothing when the flow enters, so an
    inflow comes only from a boundary condition. The cell value is the net
    outflow times ``face_area / cell_volume``, which conserves the integral up
    to the boundary fluxes.

    Args:
        grid: 1D collocated grid.
        c: Field values, shape ``(n,)``.
        velocity: Advection speed, a constant or one value per cell.
        periodic: If ``True``, the two end faces are one face, wrapped.
        scheme: ``"tvd"``, ``"weno3"``, ``"central"`` or ``"upwind"``, the interior
            face flux.
        limiter: ``"van_leer"``, ``"koren"`` or ``"minmod"``, the ``"tvd"`` limiter.

    Returns:
        ``d(v c)/dx`` at each cell, shape ``(n,)``, in ``c``'s dtype.
    """
    grid = grid_in_field_dtype(grid, c)
    n = grid.n
    v = narrow_to_field_dtype(velocity, c)
    if v.ndim == 0:
        v_inner = jnp.broadcast_to(v, (n - 1,))
        v_low, v_high, v_wrap = v, v, v
    else:
        v_inner = 0.5 * (v[:-1] + v[1:])
        v_low, v_high, v_wrap = v[0], v[-1], 0.5 * (v[-1] + v[0])
    v_face = jnp.concatenate([v_inner, v_wrap[None]])  # face k joins cells k and k + 1 (mod n)

    if scheme == "central":
        vc = v * c
        face_flux = 0.5 * (vc + jnp.roll(vc, -1))
    else:
        right, right2, left = jnp.roll(c, -1), jnp.roll(c, -2), jnp.roll(c, 1)
        if scheme == "upwind":
            from_low, from_high = c, right
        else:
            from_low = _face_value(scheme, limiter, left, c, right)
            from_high = _face_value(scheme, limiter, right2, right, c)
            if not periodic:
                k = np.arange(n)
                from_low = jnp.where(k >= 1, from_low, c)
                from_high = jnp.where(k <= n - 3, from_high, right)
        face_flux = jnp.maximum(v_face, 0.0) * from_low + jnp.minimum(v_face, 0.0) * from_high
    inner, wrap = face_flux[:-1], face_flux[-1]
    if periodic:
        low, high = wrap, wrap
    else:
        low = jnp.minimum(v_low, 0.0) * c[0]
        high = jnp.maximum(v_high, 0.0) * c[-1]
    flux = jnp.concatenate([low[None], inner, high[None]])
    area_flux = grid.face_areas * flux
    return to_field_dtype((area_flux[1:] - area_flux[:-1]) / grid.cell_volumes, c)


def div_advect_nd(
    grid: GridND,
    c: Float[Array, "..."],
    velocity_per_axis: tuple[Array, ...],
    periodic_axes: tuple[int, ...] = (),
    scheme: str = "tvd",
    limiter: str = "van_leer",
) -> Float[Array, "..."]:
    r"""Conservative :math:`\nabla\cdot(\mathbf{v} c)` on an N-D grid.

    The sum over axes of
    [div_advect_1d][diff_soup.domains.grid.advection.collocated.div_advect_1d],
    each axis with its own velocity entry.

    Args:
        grid: N-D cartesian grid.
        c: Field values, grid-shaped.
        velocity_per_axis: One entry per axis, a constant or a grid-shaped field.
        periodic_axes: Axes whose end faces wrap.
        scheme: ``"tvd"``, ``"weno3"``, ``"central"`` or ``"upwind"``, the interior
            face flux.
        limiter: ``"van_leer"``, ``"koren"`` or ``"minmod"``, the ``"tvd"`` limiter.

    Returns:
        ``div(v c)`` at each cell, grid-shaped, in ``c``'s dtype.
    """
    out = jnp.zeros_like(c)
    for d in range(grid.dim):
        ax = grid.axes[d]
        per = d in periodic_axes
        out = out + sweep_axis(
            lambda col, vcol, _ax=ax, _per=per: div_advect_1d(
                _ax, col, vcol, periodic=_per, scheme=scheme, limiter=limiter
            ),
            c,
            d,
            velocity_per_axis[d],
        )
    return out
