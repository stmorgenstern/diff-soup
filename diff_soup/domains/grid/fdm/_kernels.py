# 529 LOC -- FDM finite-difference kernels for 1D + 2D + 3D; mirror of fvm/_kernels.py grouping; splitting deferred.
"""Finite difference method operators on structured grids.

Provides node-based FDM operators (gradient, second derivative, Laplacian,
divergence of flux) for 1D grids with coordinate-aware metric corrections
(Cartesian, cylindrical, spherical), plus dimension-splitting extensions
for 2D/3D Cartesian tensor-product grids.

The grid stores its node spacings and metric corrections in double
precision; each operator casts them to the field's dtype before any arithmetic
(see :func:`~diff_soup.domains.grid._dtype.grid_in_field_dtype`). An operator
applied to a single-precision field therefore computes and returns in single
precision, and applied to a double-precision field is unchanged.

All functions are pure, JIT-compatible, AD-compatible, and vmap-compatible.
They work with [Grid1D][diff_soup.domains.grid.Grid1D] and
[GridND][diff_soup.domains.grid.GridND] objects.
"""

from __future__ import annotations

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
    # 1D operators
    "fdm_gradient_1d",
    "fdm_second_derivative_1d",
    "fdm_laplacian_1d",
    "fdm_div_flux_1d",
    # 2D operators
    "fdm_laplacian_2d",
    "fdm_div_flux_2d",
    # 3D operators
    "fdm_laplacian_3d",
    "fdm_div_flux_3d",
    "fdm_apply_periodic_bc_nd",
]


# ---------------------------------------------------------------------------
# 1D operators
# ---------------------------------------------------------------------------


def fdm_gradient_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
) -> Float[Array, "n"]:
    """Central-difference gradient at grid nodes.

    Interior nodes use the centered formula; boundary nodes use one-sided
    differences.  Returns zeros for grids with ``n <= 1``.  The spacings are
    cast to ``c``'s dtype, so the result is computed and returned in it.
    """
    return _central_gradient(grid_in_field_dtype(grid, c), c)


def _central_gradient(grid: Grid1D, c: Float[Array, "n"]) -> Float[Array, "n"]:
    """The gradient stencil of [fdm_gradient_1d][], on the grid exactly as given."""
    n = grid.n

    # Degenerate case: single node
    if n <= 1:
        return jnp.zeros_like(c)

    # Boundary (one-sided)
    grad_left = (c[1] - c[0]) / grid.node_distances[0]
    grad_right = (c[-1] - c[-2]) / grid.node_distances[-1]

    # n == 2: no interior nodes
    if n == 2:
        return jnp.stack([grad_left, grad_right])

    # Interior: centered formula  (c[j+1] - c[j-1]) / (x[j+1] - x[j-1])
    grad_interior = (c[2:] - c[:-2]) / (grid.nodes[2:] - grid.nodes[:-2])

    return jnp.concatenate([grad_left[None], grad_interior, grad_right[None]])


def fdm_second_derivative_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
) -> Float[Array, "n"]:
    """Second derivative at grid nodes using non-uniform spacing.

    Interior nodes use the standard three-point finite difference.
    Boundary nodes use a shifted three-point stencil (forward at left,
    backward at right).  Returns zeros for grids with ``n < 3``.  The spacings
    are cast to ``c``'s dtype, so the result is computed and returned in it.
    """
    grid = grid_in_field_dtype(grid, c)
    n = grid.n

    if n < 3:
        return jnp.zeros_like(c)

    h = grid.node_distances  # (n-1,)

    # Interior (j = 1 .. n-2):
    #   d2c[j] = 2*[(c[j+1]-c[j])/h_p - (c[j]-c[j-1])/h_m] / (h_m + h_p)
    h_m = h[:-1]  # (n-2,)  h[j-1]
    h_p = h[1:]  # (n-2,)  h[j]
    d2_interior = 2.0 * ((c[2:] - c[1:-1]) / h_p - (c[1:-1] - c[:-2]) / h_m) / (h_m + h_p)

    # Boundary: forward stencil at j=0, backward at j=n-1
    # Forward: nodes 0,1,2
    h0, h1 = h[0], h[1]
    d2_left = 2.0 * ((c[2] - c[1]) / h1 - (c[1] - c[0]) / h0) / (h0 + h1)

    # Backward: nodes n-3, n-2, n-1
    hm2, hm1 = h[-2], h[-1]
    d2_right = 2.0 * ((c[-1] - c[-2]) / hm1 - (c[-2] - c[-3]) / hm2) / (hm2 + hm1)

    return jnp.concatenate([d2_left[None], d2_interior, d2_right[None]])


def fdm_laplacian_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
) -> Float[Array, "n"]:
    r"""Coordinate-aware Laplacian at grid nodes.

    Combines second derivative and gradient using the grid's pre-computed
    metric fields:

    .. math::

        \nabla^2 c[j] = (1 + p_j)\, d^2c[j] + \frac{p}{r_j}\, dc[j]

    where :math:`p_j` is ``lhopital_extra_d2[j]`` (non-zero only at
    :math:`r = 0`) and :math:`p/r_j` is ``metric_gradient_correction[j]``
    (zero at :math:`r = 0`).

    Cartesian
        :math:`\nabla^2 c = d^2c/dx^2`
    Cylindrical
        :math:`\nabla^2 c = d^2c/dr^2 + (1/r)\,dc/dr`,
        with L'Hopital at :math:`r = 0`: :math:`2\,d^2c/dr^2`
    Spherical
        :math:`\nabla^2 c = d^2c/dr^2 + (2/r)\,dc/dr`,
        with L'Hopital at :math:`r = 0`: :math:`3\,d^2c/dr^2`

    The metric fields are cast to ``c``'s dtype, so the combination above is
    computed in that dtype.
    """
    grid = grid_in_field_dtype(grid, c)
    d2c = fdm_second_derivative_1d(grid, c)
    dc = fdm_gradient_1d(grid, c)
    return (1.0 + grid.lhopital_extra_d2) * d2c + grid.metric_gradient_correction * dc


def fdm_div_flux_1d(
    grid: Grid1D,
    c: Float[Array, "n"],
    D: Float[Array, "..."],
) -> Float[Array, "n"]:
    r"""Divergence of diffusive flux :math:`\nabla \cdot (D \nabla c)`.

    Uses the product-rule identity, which holds in all 1D curvilinear
    coordinate systems:

    .. math::

        \nabla \cdot (D \nabla c) = D\,\nabla^2 c
                                    + \frac{\partial D}{\partial r}
                                      \frac{\partial c}{\partial r}

    When *D* is a scalar (constant), the second term vanishes.

    Args:
        grid: 1D grid.
        c: Nodal field values, ``(n,)``.
        D: Diffusivity -- scalar or ``(n,)`` array. A diffusivity wider than
            ``c``'s dtype is cast to it first, so the result stays in that dtype.
    """
    raw_grid = grid
    grid = grid_in_field_dtype(grid, c)
    D = narrow_to_field_dtype(D, c)
    lap_c = fdm_laplacian_1d(grid, c)

    if jnp.ndim(D) == 0:
        return D * lap_c

    grad_c = fdm_gradient_1d(grid, c)
    if D.dtype == c.dtype:
        grad_D = fdm_gradient_1d(grid, D)
    else:
        # A diffusivity narrower than the field is differenced against the
        # stored geometry and kept in its own dtype, so a double-precision
        # field reproduces its result whatever dtype D arrives in.
        grad_D = _central_gradient(raw_grid, D).astype(D.dtype)
    return D * lap_c + grad_D * grad_c


# ---------------------------------------------------------------------------
# 2D operators (Cartesian tensor-product via dimension splitting)
# ---------------------------------------------------------------------------


def fdm_laplacian_2d(
    grid: GridND,
    c: Float[Array, "nx ny"],
) -> Float[Array, "nx ny"]:
    r"""2D Cartesian Laplacian via dimension splitting.

    Sums 1D Laplacians along each axis of a tensor-product grid.
    """
    # Along axis 0: vmap over columns (axis-1 index)
    contrib_x = jax.vmap(
        lambda col: fdm_laplacian_1d(grid.axes[0], col),
        in_axes=1,
        out_axes=1,
    )(c)
    # Along axis 1: vmap over rows (axis-0 index)
    contrib_y = jax.vmap(
        lambda row: fdm_laplacian_1d(grid.axes[1], row),
        in_axes=0,
        out_axes=0,
    )(c)
    return contrib_x + contrib_y


def fdm_div_flux_2d(
    grid: GridND,
    c: Float[Array, "nx ny"],
    D: Float[Array, "..."],
) -> Float[Array, "nx ny"]:
    r"""2D Cartesian :math:`\nabla \cdot (D \nabla c)` via dimension splitting.

    When *D* is a scalar, each axis contribution uses constant-D.
    When *D* is ``(nx, ny)``, each 1D slice gets the matching 1D slice of D.
    """
    if jnp.ndim(D) == 0:
        contrib_x = jax.vmap(
            lambda col: fdm_div_flux_1d(grid.axes[0], col, D),
            in_axes=1,
            out_axes=1,
        )(c)
        contrib_y = jax.vmap(
            lambda row: fdm_div_flux_1d(grid.axes[1], row, D),
            in_axes=0,
            out_axes=0,
        )(c)
    else:
        contrib_x = jax.vmap(
            lambda col, d_col: fdm_div_flux_1d(grid.axes[0], col, d_col),
            in_axes=(1, 1),
            out_axes=1,
        )(c, D)
        contrib_y = jax.vmap(
            lambda row, d_row: fdm_div_flux_1d(grid.axes[1], row, d_row),
            in_axes=(0, 0),
            out_axes=0,
        )(c, D)
    return contrib_x + contrib_y


# ---------------------------------------------------------------------------
# 3D operators (Cartesian tensor-product via dimension splitting)
# ---------------------------------------------------------------------------


def fdm_laplacian_3d(
    grid: GridND,
    c: Float[Array, "nx ny nz"],
) -> Float[Array, "nx ny nz"]:
    r"""3D Cartesian Laplacian via dimension splitting.

    Sums 1D Laplacians along each axis of a tensor-product grid.
    """
    nx, ny, nz = grid.shape

    # Axis 0: flatten axes 1,2, vmap over combined index
    c_flat = c.reshape(nx, ny * nz)
    lap0_flat = jax.vmap(
        lambda col: fdm_laplacian_1d(grid.axes[0], col),
        in_axes=1,
        out_axes=1,
    )(c_flat)
    contrib_x = lap0_flat.reshape(nx, ny, nz)

    # Axis 1: flatten axes 0,2, vmap over combined index
    c_t1 = c.transpose(1, 0, 2).reshape(ny, nx * nz)
    lap1_flat = jax.vmap(
        lambda col: fdm_laplacian_1d(grid.axes[1], col),
        in_axes=1,
        out_axes=1,
    )(c_t1)
    contrib_y = lap1_flat.reshape(ny, nx, nz).transpose(1, 0, 2)

    # Axis 2: flatten axes 0,1, vmap over combined index
    c_t2 = c.transpose(2, 0, 1).reshape(nz, nx * ny)
    lap2_flat = jax.vmap(
        lambda col: fdm_laplacian_1d(grid.axes[2], col),
        in_axes=1,
        out_axes=1,
    )(c_t2)
    contrib_z = lap2_flat.reshape(nz, nx, ny).transpose(1, 2, 0)

    return contrib_x + contrib_y + contrib_z


def fdm_div_flux_3d(
    grid: GridND,
    c: Float[Array, "nx ny nz"],
    D: Float[Array, "..."],
) -> Float[Array, "nx ny nz"]:
    r"""3D Cartesian :math:`\nabla \cdot (D \nabla c)` via dimension splitting.

    When *D* is a scalar, each axis uses constant-D.
    When *D* is ``(nx, ny, nz)``, each 1D slice gets the matching slice of D.
    """
    nx, ny, nz = grid.shape

    if jnp.ndim(D) == 0:
        # -- Axis 0 --
        c0 = c.reshape(nx, ny * nz)
        r0 = jax.vmap(
            lambda col: fdm_div_flux_1d(grid.axes[0], col, D),
            in_axes=1,
            out_axes=1,
        )(c0).reshape(nx, ny, nz)

        # -- Axis 1 --
        c1 = c.transpose(1, 0, 2).reshape(ny, nx * nz)
        r1 = (
            jax.vmap(
                lambda col: fdm_div_flux_1d(grid.axes[1], col, D),
                in_axes=1,
                out_axes=1,
            )(c1)
            .reshape(ny, nx, nz)
            .transpose(1, 0, 2)
        )

        # -- Axis 2 --
        c2 = c.transpose(2, 0, 1).reshape(nz, nx * ny)
        r2 = (
            jax.vmap(
                lambda col: fdm_div_flux_1d(grid.axes[2], col, D),
                in_axes=1,
                out_axes=1,
            )(c2)
            .reshape(nz, nx, ny)
            .transpose(1, 2, 0)
        )

    else:
        # -- Axis 0 --
        c0 = c.reshape(nx, ny * nz)
        d0 = D.reshape(nx, ny * nz)
        r0 = jax.vmap(
            lambda col, dcol: fdm_div_flux_1d(grid.axes[0], col, dcol),
            in_axes=(1, 1),
            out_axes=1,
        )(c0, d0).reshape(nx, ny, nz)

        # -- Axis 1 --
        c1 = c.transpose(1, 0, 2).reshape(ny, nx * nz)
        d1 = D.transpose(1, 0, 2).reshape(ny, nx * nz)
        r1 = (
            jax.vmap(
                lambda col, dcol: fdm_div_flux_1d(grid.axes[1], col, dcol),
                in_axes=(1, 1),
                out_axes=1,
            )(c1, d1)
            .reshape(ny, nx, nz)
            .transpose(1, 0, 2)
        )

        # -- Axis 2 --
        c2 = c.transpose(2, 0, 1).reshape(nz, nx * ny)
        d2 = D.transpose(2, 0, 1).reshape(nz, nx * ny)
        r2 = (
            jax.vmap(
                lambda col, dcol: fdm_div_flux_1d(grid.axes[2], col, dcol),
                in_axes=(1, 1),
                out_axes=1,
            )(c2, d2)
            .reshape(nz, nx, ny)
            .transpose(1, 2, 0)
        )

    return r0 + r1 + r2


# ---------------------------------------------------------------------------
# FluxBC / PeriodicBC application (called from dispatch_apply_bcs)
# ---------------------------------------------------------------------------


def fdm_apply_flux_bc_1d(
    grid: Grid1D,
    result: "Array",
    u: "Array",
    D: "Array",
    side: str,
    du_dn: "Array",
) -> "Array":
    """Apply FluxBC to a 1D FDM operator result via ghost-node correction.

    The boundary row is rebuilt from the ghost-node stencil with the grid's
    metric fields cast to ``result``'s dtype, and the row is cast to that dtype
    before the scatter, so a wider boundary value never narrows on write.
    """
    grid = grid_in_field_dtype(grid, result)
    n = grid.n
    if n < 3:
        return result
    D_b = D if jnp.ndim(D) == 0 else (D[0] if side == "low" else D[-1])
    dx = grid.node_distances

    if side == "low":
        u_ghost = u[1] + 2.0 * dx[0] * du_dn
        d2_ghost = (u[1] - 2.0 * u[0] + u_ghost) / dx[0] ** 2
        grad_at_0 = -du_dn  # du/dx = -du/dn at left boundary
        corrected = D_b * (
            (1.0 + grid.lhopital_extra_d2[0]) * d2_ghost
            + grid.metric_gradient_correction[0] * grad_at_0
        )
        result = result.at[0].set(to_field_dtype(corrected, result))
    else:
        u_ghost = u[n - 2] + 2.0 * dx[-1] * du_dn
        d2_ghost = (u_ghost - 2.0 * u[n - 1] + u[n - 2]) / dx[-1] ** 2
        grad_at_n = du_dn  # du/dx = du/dn at right boundary
        corrected = D_b * (
            (1.0 + grid.lhopital_extra_d2[n - 1]) * d2_ghost
            + grid.metric_gradient_correction[n - 1] * grad_at_n
        )
        result = result.at[n - 1].set(to_field_dtype(corrected, result))
    return result


def fdm_apply_periodic_bc_1d(
    grid: Grid1D,
    result: "Array",
    u: "Array",
    D: "Array",
) -> "Array":
    """Apply periodic BC wrap-around stencil correction for 1D FDM.

    Each wrapped end row is rebuilt from the ghost-node stencil with the grid's
    metric fields cast to ``result``'s dtype, and the row is cast to that dtype
    before the scatter, so a wider diffusivity never narrows on write.
    """
    grid = grid_in_field_dtype(grid, result)
    D_val = jnp.asarray(D)
    dx_left = grid.node_distances[0]
    dx_right = grid.node_distances[-1]

    if D_val.ndim == 0:
        D_wrap = D_val
    else:
        # Harmonic mean for interface diffusivity
        D_wrap = jnp.where(
            D_val[-1] + D_val[0] > 0,
            2.0 * D_val[-1] * D_val[0] / (D_val[-1] + D_val[0]),
            0.0,
        )

    # Node 0: ghost left = u[-1]
    d2_0 = (u[1] - 2.0 * u[0] + u[-1]) / (dx_left**2)
    grad_0 = (u[1] - u[-1]) / (2.0 * dx_left)
    corrected_0 = D_wrap * (
        (1.0 + grid.lhopital_extra_d2[0]) * d2_0 + grid.metric_gradient_correction[0] * grad_0
    )
    result = result.at[0].set(to_field_dtype(corrected_0, result))

    # Node n-1: ghost right = u[0]
    d2_n = (u[0] - 2.0 * u[-1] + u[-2]) / (dx_right**2)
    grad_n = (u[0] - u[-2]) / (2.0 * dx_right)
    corrected_n = D_wrap * (
        (1.0 + grid.lhopital_extra_d2[-1]) * d2_n + grid.metric_gradient_correction[-1] * grad_n
    )
    result = result.at[-1].set(to_field_dtype(corrected_n, result))
    return result


def fdm_apply_periodic_bc_nd(
    grid: GridND,
    result: Float[Array, "..."],
    u: Float[Array, "..."],
    D: Float[Array, "..."],
    axes: tuple[int, ...],
) -> Float[Array, "..."]:
    """Apply the periodic wrap-stencil correction along each listed axis of an N-D FDM grid.

    The N-D FDM operator is a sum of 1D operators applied along each axis
    (dimension splitting), so the periodic correction is likewise per-axis and
    additive. Unlike the FVM wrap flux, the 1D FDM periodic kernel
    ([fdm_apply_periodic_bc_1d][diff_soup.domains.grid.fdm._kernels.fdm_apply_periodic_bc_1d])
    *overwrites* its two boundary rows rather
    than adding to them, so applying it directly per axis would clobber the
    other axes' contributions at corner nodes. Instead, for each axis this
    computes the delta between the periodic and the natural-boundary 1D
    result -- zero on interior rows, nonzero only on the two boundary rows of
    that axis -- and accumulates the deltas onto ``result``, which composes
    correctly at corners.

    Args:
        grid: N-D Cartesian grid.
        result: Existing operator result (natural one-sided boundaries) to
            correct, grid-shaped.
        u: Current state, grid-shaped.
        D: Diffusivity -- scalar, or grid-shaped for a spatially varying
            field. At each wrap face the nodal values on either side are
            combined with the unweighted harmonic mean, matching
            [fdm_apply_periodic_bc_1d][diff_soup.domains.grid.fdm._kernels.fdm_apply_periodic_bc_1d].
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
        if field_D:
            D_f = jnp.moveaxis(D, d, 0).reshape(sh[0], -1)

            def _delta(uc, dc):
                plain = fdm_div_flux_1d(ax_grid, uc, dc)
                periodic = fdm_apply_periodic_bc_1d(ax_grid, plain, uc, dc)
                return periodic - plain

            delta_f = jax.vmap(_delta, in_axes=1, out_axes=1)(u_f, D_f)
        else:

            def _delta(uc):
                plain = fdm_div_flux_1d(ax_grid, uc, D)
                periodic = fdm_apply_periodic_bc_1d(ax_grid, plain, uc, D)
                return periodic - plain

            delta_f = jax.vmap(_delta, in_axes=1, out_axes=1)(u_f)
        out = out + jnp.moveaxis(delta_f.reshape(sh), 0, d)
    return out
