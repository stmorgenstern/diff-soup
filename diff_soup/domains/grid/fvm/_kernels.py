# 657 LOC -- FVM finite-volume kernels for 1D + 2D + 3D in cartesian + cylindrical + spherical coords; symmetric variants kept colocated for invariance audit; splitting deferred.
"""Finite volume method operators on structured grids.

Provides FVM discretizations of diffusion (div(D grad c)) and related
operators for 1D curvilinear grids and 2D/3D Cartesian tensor-product
grids.  Coordinate metric factors (cell_volumes, face_areas) are encoded
in the [Grid1D][] object, so all operators are coordinate-system
agnostic at runtime.  The grid stores those factors in double precision; each
operator casts them, and its coefficient inputs, to the field's dtype before any
arithmetic (see
[grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype]), so a
single-precision field is discretized in single-precision arithmetic and a
double-precision field uses the stored geometry unchanged.  Operator outputs are
field-shaped and come back in the field's dtype.

All functions are pure, JIT-compatible, AD-compatible, and vmap-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Scalar

from diff_soup.domains.grid import Grid1D, GridND
from diff_soup.domains.grid._dtype import (
    grid_in_field_dtype,
    narrow_to_field_dtype,
    to_field_dtype,
)
from diff_soup.domains.grid._sweep import sweep_axis

__all__ = [
    # 1D operators
    "fvm_face_flux_1d",
    "fvm_div_flux_1d",
    "fvm_laplacian_1d",
    "fvm_gradient_1d",
    "fvm_apply_neumann_flux_1d",
    # 2D/3D Cartesian operators
    "fvm_div_flux_2d",
    "fvm_div_flux_3d",
    "fvm_laplacian_2d",
    "fvm_laplacian_3d",
    "fvm_div_flux_axes",
    "fvm_apply_normal_derivative",
    "fvm_apply_periodic_bc_nd",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _harmonic_mean(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Unweighted harmonic mean with safe division.

    Equal to [_harmonic_mean_weighted][] with equal widths.  Kept
    for callers that don't have width context.
    """
    return jnp.where(a + b > 0, 2.0 * a * b / (a + b), 0.0)


def _harmonic_mean_weighted(
    D_left: Float[Array, "..."],
    D_right: Float[Array, "..."],
    w_left: Float[Array, "..."],
    w_right: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Width-weighted harmonic mean of nodal diffusivities at a face.

    Formula (Recktenwald 2012; LIONSIMBA / PyBaMM convention)::

        D_eff = (w_l + w_r) * D_l * D_r / (w_l * D_r + w_r * D_l)

    Reduces to ``2 D_l D_r / (D_l + D_r)`` when ``w_l == w_r``, so
    uniform-mesh callers see no change.  On non-uniform meshes this is
    the only face-value choice consistent with continuous flux at the
    interface (no spurious mass transfer).
    """
    num = (w_left + w_right) * D_left * D_right
    den = w_left * D_right + w_right * D_left
    return jnp.where(den > 0, num / den, 0.0)


def _diffusivity_at_faces(
    grid: Grid1D,
    D: Float[ArrayLike, "..."],
    field: Float[Array, "n"],
) -> Float[Array, "n_plus_1"]:
    """Map diffusivity to face values, in the dtype of the field it multiplies.

    * Scalar D: broadcast directly.
    * Shape ``(n,)``: nodal values -> width-weighted harmonic mean at
      interior faces, zero at boundaries.
    * Shape ``(n+1,)``: already at faces, use as-is.

    The diffusivity and the cell widths are cast to the field dtype first, so
    the width-weighted mean is computed in that dtype and the flux stays a
    single-precision computation for a single-precision field.

    Args:
        grid: 1D grid with ``n`` nodes.
        D: Diffusivity — scalar, shape ``(n,)`` (nodal), or ``(n+1,)`` (faces).
        field: Cell-centered field the face diffusivity will multiply.  Only
            its dtype is read.

    Returns:
        Face diffusivities, shape ``(n+1,)``, in ``field``'s dtype.
    """
    grid = grid_in_field_dtype(grid, field)
    D = to_field_dtype(D, field)
    if D.ndim == 0:
        return jnp.broadcast_to(D, (grid.n + 1,))
    if D.shape == (grid.n + 1,):
        return D
    # Nodal values (n,) -> faces via width-weighted harmonic mean at
    # interior faces.  This matches PyBaMM's FVM and is required for
    # bit-faithful parity on non-uniform meshes (e.g. multi-region cells
    # in DFN); on uniform meshes it reduces to the unweighted form.
    w = grid.cell_widths
    D_face = jnp.zeros(grid.n + 1, dtype=field.dtype)
    D_face = D_face.at[1 : grid.n].set(_harmonic_mean_weighted(D[:-1], D[1:], w[:-1], w[1:]))
    return D_face


# ---------------------------------------------------------------------------
# 1D operators
# ---------------------------------------------------------------------------


def fvm_face_flux_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    D: Float[ArrayLike, "..."],
) -> Float[Array, "n_plus_1"]:
    """Diffusive flux at cell faces: F[j] = -D_face[j] * (c[j] - c[j-1]) / h[j-1].

    Boundary faces (j=0 and j=n) have zero flux (natural Neumann).

    The flux is computed and returned in ``c``'s dtype; the node distances are
    cast to it first -- see
    [grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype].

    Args:
        grid: 1D grid with ``n`` nodes.
        c: Scalar field at cell centers, shape ``(n,)``.
        D: Diffusivity — scalar, shape ``(n,)`` (nodal), or ``(n+1,)`` (faces).

    Returns:
        Face fluxes, shape ``(n+1,)``, in ``c``'s dtype.
    """
    grid = grid_in_field_dtype(grid, c)
    D_face = _diffusivity_at_faces(grid, D, c)
    # Interior fluxes at faces 1..n-1
    dc = c[1:] - c[:-1]  # (n-1,)
    flux_interior = -D_face[1 : grid.n] * dc / grid.node_distances  # (n-1,)
    # Assemble full flux vector with zero-flux boundaries
    flux = jnp.zeros(grid.n + 1, dtype=c.dtype)
    flux = flux.at[1 : grid.n].set(flux_interior)
    return flux


def fvm_div_flux_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    D: Float[ArrayLike, "..."],
) -> Float[Array, "n"]:
    r"""FVM discretization of :math:`\nabla \cdot (D \nabla c)`.

    Computes the net area-weighted flux into each cell divided by cell
    volume.  Coordinate metric factors are encoded in ``grid.face_areas``
    and ``grid.cell_volumes``, so this works for Cartesian, cylindrical,
    and spherical grids without branching.  The factors are cast to ``c``'s
    dtype first, so the balance is computed in that dtype -- see
    [grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype].

    Args:
        grid: 1D grid with ``n`` nodes.
        c: Scalar field at cell centers, shape ``(n,)``.
        D: Diffusivity — scalar, shape ``(n,)`` (nodal), or ``(n+1,)`` (faces).

    Returns:
        Divergence of diffusive flux at cell centers, shape ``(n,)``, in
        ``c``'s dtype.  Positive for concave-up profiles (diffusion smooths).
    """
    grid = grid_in_field_dtype(grid, c)
    flux = fvm_face_flux_1d(grid, c, D)  # (n+1,) -- Fick's flux (-D dc/dx)
    # div(D grad c) = -div(Fick flux), so negate the net flux
    net_flux = grid.face_areas[:-1] * flux[:-1] - grid.face_areas[1:] * flux[1:]
    return net_flux / grid.cell_volumes


def fvm_laplacian_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
) -> Float[Array, "n"]:
    r"""FVM Laplacian :math:`\nabla^2 c` (i.e. ``fvm_div_flux_1d`` with D=1).

    Args:
        grid: 1D grid with ``n`` nodes.
        c: Scalar field at cell centers, shape ``(n,)``.

    Returns:
        Laplacian at cell centers, shape ``(n,)``, in ``c``'s dtype.
    """
    return fvm_div_flux_1d(grid, c, jnp.asarray(1.0, c.dtype))


def fvm_gradient_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
) -> Float[Array, "n"]:
    """Central-difference gradient at cell centers (for post-processing).

    Uses one-sided differences at the boundary nodes.

    Node coordinates and spacings are cast to ``c``'s dtype first, so the
    difference quotients are computed in that dtype -- see
    [grid_in_field_dtype][diff_soup.domains.grid._dtype.grid_in_field_dtype].

    Args:
        grid: 1D grid with ``n`` nodes.
        c: Scalar field at cell centers, shape ``(n,)``.

    Returns:
        Gradient at cell centers, shape ``(n,)``, in ``c``'s dtype.
    """
    grid = grid_in_field_dtype(grid, c)
    n = grid.n
    # Interior: central difference
    grad_interior = (c[2:] - c[:-2]) / (grid.nodes[2:] - grid.nodes[:-2])  # (n-2,)
    # Boundary: one-sided
    grad_left = (c[1] - c[0]) / grid.node_distances[0]
    grad_right = (c[n - 1] - c[n - 2]) / grid.node_distances[n - 2]
    return jnp.concatenate([grad_left[None], grad_interior, grad_right[None]])


def fvm_apply_neumann_flux_1d(
    grid: Grid1D,
    rhs: Float[Array, "n"],
    *,
    flux_left: Float[Scalar, ""] | None = None,
    flux_right: Float[Scalar, ""] | None = None,
) -> Float[Array, "n"]:
    r"""Add prescribed Neumann flux contributions to an existing RHS vector.

    The sign convention matches [fvm_div_flux_1d][]: a *positive*
    ``flux_right`` means flux *into* the domain at the right boundary
    (adds positive contribution to ``rhs[-1]``).

    For curvilinear grids, ``face_areas[0] = 0`` at r=0, so a left flux
    has no effect there (symmetry BC is automatically satisfied).

    Args:
        grid: 1D grid with ``n`` nodes.
        rhs: Right-hand side vector to modify, shape ``(n,)``.
        flux_left: Prescribed flux at the left boundary (face 0).  ``None`` = no change.
        flux_right: Prescribed flux at the right boundary (face n).  ``None`` = no change.

    Returns:
        Modified RHS vector, shape ``(n,)``.
    """
    grid = grid_in_field_dtype(grid, rhs)
    if flux_left is not None:
        rhs = rhs.at[0].add(
            to_field_dtype(-grid.face_areas[0] * flux_left / grid.cell_volumes[0], rhs)
        )
    if flux_right is not None:
        rhs = rhs.at[-1].add(
            to_field_dtype(grid.face_areas[-1] * flux_right / grid.cell_volumes[-1], rhs)
        )
    return rhs


def fvm_apply_robin_1d(
    grid: Grid1D,
    rhs: Float[Array, "n"],
    u: Float[Array, "n"],
    *,
    side: str,
    alpha: Float[Scalar, ""],
    beta: Float[Scalar, ""],
    g: Float[Scalar, ""],
) -> Float[Array, "n"]:
    r"""Apply Robin BC to a 1D FVM RHS vector.

    Robin condition: :math:`\alpha\, u + \beta\, \partial u / \partial n = g`
    on the boundary.  The outward normal flux at the boundary face becomes
    :math:`(g - \alpha\, u_\text{boundary}) / \beta`, which replaces the
    default zero-flux assumption.

    The sign convention matches [fvm_div_flux_1d][]: the Robin flux
    contribution is added to the RHS of the boundary cell.

    Args:
        grid: 1D grid with ``n`` nodes.
        rhs: Right-hand side vector to modify, shape ``(n,)``.
        u: Current solution field, shape ``(n,)``.
        side: ``'low'`` for the left boundary, ``'high'`` for the right boundary.
        alpha: Coefficient of ``u`` in the Robin condition.
        beta: Coefficient of ``du/dn``.  Must be nonzero.
        g: RHS scalar of the Robin condition.

    Returns:
        Modified RHS vector, shape ``(n,)``.
    """
    grid = grid_in_field_dtype(grid, rhs)
    if side == "low":
        robin_flux = (g - alpha * u[0]) / beta
        rhs = rhs.at[0].add(
            to_field_dtype(-grid.face_areas[0] * robin_flux / grid.cell_volumes[0], rhs)
        )
    elif side == "high":
        robin_flux = (g - alpha * u[-1]) / beta
        rhs = rhs.at[-1].add(
            to_field_dtype(grid.face_areas[-1] * robin_flux / grid.cell_volumes[-1], rhs)
        )
    else:
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")
    return rhs


# N-dimensional Neumann and Robin helpers are omitted; callers compose the
# dimension-split operators below.

# ---------------------------------------------------------------------------
# 2D/3D Cartesian operators (dimension splitting via vmap)
# ---------------------------------------------------------------------------


def fvm_div_flux_2d(
    grid: GridND,
    c: Float[Array, "nx ny"],
    D: Float[ArrayLike, "..."],
) -> Float[Array, "nx ny"]:
    r"""FVM discretization of :math:`\nabla \cdot (D \nabla c)` on a 2D Cartesian grid.

    Uses dimension splitting: the 2D operator
    :math:`\partial_x(D\,\partial_x c) + \partial_y(D\,\partial_y c)` decomposes
    as the sum of 1D operators applied along each axis independently and
    vmapped over the other axis.

    Args:
        grid: 2D Cartesian grid (``GridND`` with 2 axes).
        c: Scalar field, shape ``(nx, ny)``.
        D: Diffusivity — scalar (broadcast), or grid-shaped ``(nx, ny)`` for a
            state-dependent / spatially varying field. Per-axis face arrays
            are not supported in the multi-D path; reshape upstream if needed.

    Returns:
        Divergence of diffusive flux, shape ``(nx, ny)``, in ``c``'s dtype.
    """
    D = jnp.asarray(D)
    if D.ndim == c.ndim:
        # Spatially varying / state-dependent field: vmap c and D together.
        contrib_x = jax.vmap(
            lambda col, dcol: fvm_div_flux_1d(grid.axes[0], col, dcol),
            in_axes=(1, 1),
            out_axes=1,
        )(c, D)
        contrib_y = jax.vmap(
            lambda row, drow: fvm_div_flux_1d(grid.axes[1], row, drow),
            in_axes=(0, 0),
            out_axes=0,
        )(c, D)
    else:
        # Scalar (or otherwise broadcastable) D: pass through unbatched.
        contrib_x = jax.vmap(
            lambda col: fvm_div_flux_1d(grid.axes[0], col, D),
            in_axes=1,
            out_axes=1,
        )(c)
        contrib_y = jax.vmap(
            lambda row: fvm_div_flux_1d(grid.axes[1], row, D),
            in_axes=0,
            out_axes=0,
        )(c)
    return contrib_x + contrib_y


def fvm_div_flux_3d(
    grid: GridND,
    c: Float[Array, "nx ny nz"],
    D: Float[ArrayLike, "..."],
) -> Float[Array, "nx ny nz"]:
    r"""FVM discretization of :math:`\nabla \cdot (D \nabla c)` on a 3D Cartesian grid.

    Dimension-split extension of [fvm_div_flux_2d][]. For state-dependent /
    spatially varying ``D``, both ``c`` and ``D`` are reshaped together along the
    swept axis so the underlying 1D dispatch sees a matched ``(c, D)`` pair.

    Args:
        grid: 3D Cartesian grid (``GridND`` with 3 axes).
        c: Scalar field, shape ``(nx, ny, nz)``.
        D: Diffusivity — scalar (broadcast), or grid-shaped ``(nx, ny, nz)``.

    Returns:
        Divergence of diffusive flux, shape ``(nx, ny, nz)``, in ``c``'s dtype.
    """
    nx, ny, nz = grid.shape
    D = jnp.asarray(D)
    field_D = D.ndim == c.ndim

    # Along x-axis: flatten (ny, nz) into a single column-major axis
    c_reshaped_x = c.reshape(nx, ny * nz)
    if field_D:
        D_reshaped_x = D.reshape(nx, ny * nz)
        contrib_x = jax.vmap(
            lambda col, dcol: fvm_div_flux_1d(grid.axes[0], col, dcol),
            in_axes=(1, 1),
            out_axes=1,
        )(c_reshaped_x, D_reshaped_x)
    else:
        contrib_x = jax.vmap(
            lambda col: fvm_div_flux_1d(grid.axes[0], col, D),
            in_axes=1,
            out_axes=1,
        )(c_reshaped_x)
    contrib_x = contrib_x.reshape(nx, ny, nz)

    # Along y-axis: bring axis 1 to the front
    c_reshaped_y = c.transpose(1, 0, 2).reshape(ny, nx * nz)
    if field_D:
        D_reshaped_y = D.transpose(1, 0, 2).reshape(ny, nx * nz)
        contrib_y = jax.vmap(
            lambda col, dcol: fvm_div_flux_1d(grid.axes[1], col, dcol),
            in_axes=(1, 1),
            out_axes=1,
        )(c_reshaped_y, D_reshaped_y)
    else:
        contrib_y = jax.vmap(
            lambda col: fvm_div_flux_1d(grid.axes[1], col, D),
            in_axes=1,
            out_axes=1,
        )(c_reshaped_y)
    contrib_y = contrib_y.reshape(ny, nx, nz).transpose(1, 0, 2)

    # Along z-axis: bring axis 2 to the front
    c_reshaped_z = c.transpose(2, 0, 1).reshape(nz, nx * ny)
    if field_D:
        D_reshaped_z = D.transpose(2, 0, 1).reshape(nz, nx * ny)
        contrib_z = jax.vmap(
            lambda col, dcol: fvm_div_flux_1d(grid.axes[2], col, dcol),
            in_axes=(1, 1),
            out_axes=1,
        )(c_reshaped_z, D_reshaped_z)
    else:
        contrib_z = jax.vmap(
            lambda col: fvm_div_flux_1d(grid.axes[2], col, D),
            in_axes=1,
            out_axes=1,
        )(c_reshaped_z)
    contrib_z = contrib_z.reshape(nz, nx, ny).transpose(1, 2, 0)

    return contrib_x + contrib_y + contrib_z


def fvm_laplacian_2d(
    grid: GridND,
    c: Float[Array, "nx ny"],
) -> Float[Array, "nx ny"]:
    r"""FVM Laplacian :math:`\nabla^2 c` on a 2D Cartesian grid (D=1).

    Args:
        grid: 2D Cartesian grid.
        c: Scalar field, shape ``(nx, ny)``.

    Returns:
        Laplacian, shape ``(nx, ny)``, in ``c``'s dtype.
    """
    return fvm_div_flux_2d(grid, c, jnp.asarray(1.0, c.dtype))


def fvm_laplacian_3d(
    grid: GridND,
    c: Float[Array, "nx ny nz"],
) -> Float[Array, "nx ny nz"]:
    r"""FVM Laplacian :math:`\nabla^2 c` on a 3D Cartesian grid (D=1).

    Args:
        grid: 3D Cartesian grid.
        c: Scalar field, shape ``(nx, ny, nz)``.

    Returns:
        Laplacian, shape ``(nx, ny, nz)``, in ``c``'s dtype.
    """
    return fvm_div_flux_3d(grid, c, jnp.asarray(1.0, c.dtype))


def fvm_div_flux_axes(
    grid: GridND,
    c: Float[Array, "..."],
    D_axes: tuple,
) -> Float[Array, "..."]:
    r"""FVM :math:`\sum_d \partial_d (D_d\, \partial_d c)` with one diffusivity per axis.

    The diagonal-tensor form of [fvm_div_flux_2d][] and [fvm_div_flux_3d][]:
    axis ``d`` is swept with its own ``D_axes[d]``.

    Args:
        grid: 2D or 3D Cartesian grid.
        c: Scalar field, grid-shaped.
        D_axes: One diffusivity per axis, each a scalar or grid-shaped.

    Returns:
        Divergence of diffusive flux, grid-shaped, in ``c``'s dtype.
    """
    out = jnp.zeros_like(c)
    for d, D in enumerate(D_axes):
        ax = grid.axes[d]
        out = out + sweep_axis(
            lambda col, dcol, _ax=ax: fvm_div_flux_1d(_ax, col, dcol), c, d, jnp.asarray(D)
        )
    return out


# ---------------------------------------------------------------------------
# FluxBC / PeriodicBC application (called from dispatch_apply_bcs)
# ---------------------------------------------------------------------------


def fvm_apply_normal_derivative(
    grid: "Grid1D | GridND",
    result: Float[Array, "..."],
    D: Float[ArrayLike, "..."],
    axis: int,
    side: str,
    du_dn: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """Add the flux of a prescribed outward normal derivative through one boundary face.

    A ghost value ``u_cell + h du/dn`` beyond the face gives the face gradient
    ``du/dn``, so the face carries ``D_b du/dn`` into the domain, with ``D_b``
    the boundary cell's diffusivity (a face value on a ``Grid1D`` given face
    diffusivities). The boundary cell gains that flux times
    ``face_area / cell_volume``.

    Args:
        grid: A ``Grid1D`` (``axis`` 0) or a ``GridND``.
        result: Operator result with zero-flux boundaries, grid-shaped.
        D: Diffusivity along ``axis``, a scalar or grid-shaped (``(n + 1,)``
            face values on a ``Grid1D`` also accepted).
        axis: Axis normal to the face.
        side: ``"low"`` or ``"high"``.
        du_dn: Outward normal derivative, a scalar or one value per face cell.

    Returns:
        ``result`` with the boundary flux added.
    """
    ax = grid if isinstance(grid, Grid1D) else grid.axes[axis]
    ax = grid_in_field_dtype(ax, result)
    n = ax.n
    end = 0 if side == "low" else n - 1
    D = jnp.asarray(D)
    if D.ndim == 0:
        D_b = D
    elif isinstance(grid, Grid1D) and D.shape == (n + 1,):
        D_b = D[0] if side == "low" else D[-1]
    else:
        D_b = jnp.take(D, end, axis=axis)
    area = ax.face_areas[0] if side == "low" else ax.face_areas[-1]
    flux = area * D_b * du_dn / ax.cell_volumes[end]
    index = [slice(None)] * result.ndim
    index[axis] = end
    face = result[tuple(index)]
    return result.at[tuple(index)].add(to_field_dtype(jnp.broadcast_to(flux, face.shape), result))


def fvm_apply_flux_bc_1d(
    grid: Grid1D,
    result: Float[Array, "n"],
    u: Float[Array, "n"],
    D: Float[ArrayLike, "..."],
    side: str,
    du_dn: Float[Array, ""],
) -> Float[Array, "n"]:
    """Apply FluxBC to a 1D FVM operator result.

    Adds ``D * du_dn * face_area / cell_volume`` to the boundary cell, with
    ``du_dn`` the outward normal derivative.
    """
    del u
    return fvm_apply_normal_derivative(grid, result, D, 0, side, du_dn)


def fvm_apply_flux_bc_nd(
    domain: "GridND",
    result: Float[Array, "..."],
    u: Float[Array, "..."],
    D: Float[ArrayLike, "..."],
    axis: int,
    side: str,
    du_dn_fn,
    t: "Array",
) -> Float[Array, "..."]:
    """Apply FluxBC to an N-D FVM operator result.

    Args:
        domain: Grid on which the operator is assembled.
        result: Existing operator result to which the BC correction is added.
        u: Current state array.
        D: Diffusivity, a scalar or grid-shaped.
        axis: Axis along which the boundary lies.
        side: ``"low"`` or ``"high"``.
        du_dn_fn: Callable ``(u_boundary, t) -> du/dn``.
        t: Current time (passed to ``du_dn_fn``).
    """
    end = 0 if side == "low" else domain.shape[axis] - 1
    u_b = jnp.take(u, end, axis=axis)
    du_dn = jax.vmap(lambda ub: du_dn_fn(ub, t))(u_b.ravel()).reshape(u_b.shape)
    return fvm_apply_normal_derivative(domain, result, D, axis, side, du_dn)


def fvm_apply_periodic_bc_1d(
    grid: Grid1D,
    result: Float[Array, "n"],
    u: Float[Array, "n"],
    D: Float[ArrayLike, "..."],
) -> Float[Array, "n"]:
    """Apply periodic BC wrap-around flux correction for 1D FVM.

    The wrap distance and metric factors, and a diffusivity wider than
    ``result``, are cast to ``result``'s dtype, so the correction is computed
    in that dtype.
    """
    grid = grid_in_field_dtype(grid, result)
    D_val = narrow_to_field_dtype(D, result)
    wrap_dist = grid.cell_widths[0] / 2.0 + grid.cell_widths[-1] / 2.0

    if D_val.ndim == 0:
        D_wrap = D_val
    else:
        D_wrap = _harmonic_mean(D_val[-1], D_val[0])

    flux_wrap = -D_wrap * (u[0] - u[-1]) / wrap_dist
    result = result.at[0].add(
        to_field_dtype(grid.face_areas[0] * flux_wrap / grid.cell_volumes[0], result)
    )
    result = result.at[-1].add(
        to_field_dtype(-grid.face_areas[-1] * flux_wrap / grid.cell_volumes[-1], result)
    )
    return result


def fvm_apply_periodic_bc_nd(
    grid: GridND,
    result: Float[Array, "..."],
    u: Float[Array, "..."],
    D: Float[ArrayLike, "..."],
    axes: tuple[int, ...],
) -> Float[Array, "..."]:
    """Apply the periodic wrap-flux correction along each listed axis of an N-D FVM grid.

    The N-D FVM operator is a sum of 1D operators applied along each axis
    (dimension splitting), so the periodic correction is likewise per-axis
    and additive: for each axis in ``axes``, the 1D wrap-flux correction
    ([fvm_apply_periodic_bc_1d][]) is evaluated on a zero-filled slice
    (its corrections are scatter-adds onto the two boundary cells, so
    starting from zeros isolates the pure wrap contribution) and vmapped
    over every other axis, then accumulated onto ``result``.

    Args:
        grid: N-D Cartesian grid.
        result: Existing operator result (natural zero-flux boundaries) to
            correct, grid-shaped.
        u: Current state, grid-shaped.
        D: Diffusivity — scalar, or grid-shaped for a spatially varying
            field. At each wrap face the nodal values on either side are
            combined with the unweighted harmonic mean, matching
            [fvm_apply_periodic_bc_1d][].
        axes: Axes to wrap.

    Returns:
        ``result`` plus the periodic wrap correction on the listed axes.
    """
    D = jnp.asarray(D)
    field_D = D.ndim == u.ndim
    out = result
    for d in axes:
        ax_grid = grid.axes[d]
        u_m = jnp.moveaxis(u, d, 0)
        sh = u_m.shape
        u_f = u_m.reshape(sh[0], -1)
        zeros_f = jnp.zeros_like(u_f)
        if field_D:
            D_f = jnp.moveaxis(D, d, 0).reshape(sh[0], -1)
            corr_f = jax.vmap(
                lambda zc, uc, dc: fvm_apply_periodic_bc_1d(ax_grid, zc, uc, dc),
                in_axes=1,
                out_axes=1,
            )(zeros_f, u_f, D_f)
        else:
            corr_f = jax.vmap(
                lambda zc, uc: fvm_apply_periodic_bc_1d(ax_grid, zc, uc, D),
                in_axes=1,
                out_axes=1,
            )(zeros_f, u_f)
        out = out + jnp.moveaxis(corr_f.reshape(sh), 0, d)
    return out
