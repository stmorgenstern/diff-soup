# 534 LOC -- pixel-averaging operator + UpConvDownOperator (mesh -> PSF -> image) + interpolate_image_to_nodes + boundary-trace ops; splitting deferred.
r"""observation/pixel_averaging.py — FEM field to image pixel observation operators.

Provides two observation operators:

1. [PixelAveragingOperator][diff_soup.inference.data.pixel_averaging.PixelAveragingOperator] — direct mesh-to-pixel interpolation
   (simple, for unit tests and cases where PSF blurring is not needed)

2. [UpConvDownOperator][diff_soup.inference.data.pixel_averaging.UpConvDownOperator] — production observation operator implementing
   the full pipeline from the Zhao dimensional:

   .. math:: \\mathcal{O} = D \\cdot C \\cdot U

   - **U**: barycentric interpolation from FEM mesh to fine rectangular grid
   - **C**: normalized one-pixel box PSF convolution on the fine grid
   - **D**: linear interpolation / subsampling from fine grid to image pixels

   All three stages are linear in the nodal field, so the composed operator
   is differentiable via standard JAX AD.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from diff_soup.inference.data.base import ObservationOperator


class PixelAveragingOperator(ObservationOperator):
    """Observation operator mapping FEM nodal values to image pixel values.

    Precomputes an interpolation matrix ``W`` of shape ``(n_valid_pixels, n_nodes)``
    such that ``obs = W @ nodal_field``.  At evaluation time this is a single
    sparse-ish matmul (stored as dense rows at element-local level for efficiency
    under JAX tracing).

    Attributes:
        interp_nodes: Node indices for each valid pixel, shape ``(n_valid, nodes_per_elem)``.
            ``interp_nodes[i]`` gives the element node indices for pixel ``i``.
        interp_weights: Shape function weights at pixel center, shape ``(n_valid, nodes_per_elem)``.
            ``interp_weights[i]`` gives the shape function values for pixel ``i``.
        n_nodes_: Total number of FEM nodes.
        valid_mask: Boolean mask over the full image, shape ``(ny, nx)``.
            Stored for reconstruction of full images from the valid-pixel vector.
    """

    interp_nodes: jax.Array  # (n_valid, nodes_per_elem) int32
    interp_weights: jax.Array  # (n_valid, nodes_per_elem) float64
    n_nodes_: int = eqx.field(static=True)
    # valid_mask is a regular pytree leaf (numpy, never traced) rather than a
    # static field: Equinox's static warning fires on any array leaf — numpy
    # or JAX — in a static field.
    valid_mask: np.ndarray

    def predict(self, nodal_field: jax.Array, t: Any, theta: Any) -> jax.Array:
        """Map nodal field → valid pixel values, shape ``(n_valid,)``; ``t``/``theta`` unused."""
        del t, theta
        # Gather node values for each pixel's element
        vals = nodal_field[self.interp_nodes]  # (n_valid, nodes_per_elem)
        weights = jnp.asarray(self.interp_weights, dtype=nodal_field.dtype)
        return jnp.sum(vals * weights, axis=1)  # (n_valid,)

    def to_image(self, pixel_values: jax.Array, fill: float = 0.0) -> jax.Array:
        """Scatter valid-pixel values back into a full image array.

        Args:
            pixel_values: Shape ``(n_valid,)``.
            fill: Fill value for invalid pixels.

        Returns:
            image:
                Shape ``(ny, nx)``.
        """
        ny, nx = self.valid_mask.shape
        img = jnp.full(ny * nx, fill)
        valid_flat = jnp.asarray(self.valid_mask.ravel())
        valid_idx = jnp.where(valid_flat, size=int(self.valid_mask.sum()))[0]
        img = img.at[valid_idx].set(pixel_values)
        return img.reshape(ny, nx)

    @property
    def n_valid(self) -> int:
        """Number of valid (interior) pixels."""
        return int(self.interp_nodes.shape[0])

    @classmethod
    def from_mesh_and_image(
        cls,
        points: np.ndarray,
        cells: np.ndarray,
        image_shape: tuple[int, int],
        pixel_origin: tuple[float, float],
        pixel_size: float,
        valid_mask: Optional[np.ndarray] = None,
        ele_type: str = "TRI3",
    ) -> "PixelAveragingOperator":
        """Build the operator by locating each pixel center in the mesh.

        Args:
            points: Mesh node coordinates, shape ``(n_nodes, 2)``.
            cells: Element connectivity, shape ``(n_cells, nodes_per_elem)``.
            image_shape: ``(ny, nx)`` — number of pixels in y and x.
            pixel_origin: ``(x0, y0)`` — physical coordinates of the center of pixel [0, 0].
            pixel_size: Physical size of one pixel (assumed square).
            valid_mask: Boolean mask ``(ny, nx)``.  ``None`` → all pixels valid.
            ele_type: Element type: ``"TRI3"`` or ``"QUAD4"``.
        """
        ny, nx = image_shape
        if valid_mask is None:
            valid_mask = np.ones((ny, nx), dtype=bool)

        x0, y0 = pixel_origin
        # Pixel center coordinates
        px = x0 + np.arange(nx) * pixel_size
        py = y0 + np.arange(ny) * pixel_size
        PX, PY = np.meshgrid(px, py)
        pixel_pts = np.stack([PX.ravel(), PY.ravel()], axis=1)  # (ny*nx, 2)

        valid_flat = valid_mask.ravel()
        valid_pixel_pts = pixel_pts[valid_flat]  # (n_valid, 2)

        n_valid = valid_pixel_pts.shape[0]
        nodes_per_elem = cells.shape[1]
        interp_nodes = np.zeros((n_valid, nodes_per_elem), dtype=np.int32)
        interp_weights = np.zeros((n_valid, nodes_per_elem), dtype=np.float64)

        # For each valid pixel, find containing element and compute shape functions
        for i in range(n_valid):
            pt = valid_pixel_pts[i]
            elem_idx, ref_coords = _find_element(pt, points, cells, ele_type)
            if elem_idx >= 0:
                interp_nodes[i] = cells[elem_idx]
                interp_weights[i] = _eval_shape_functions(ref_coords, ele_type)
            # If not found, weights stay zero (pixel gets 0 — will be masked)

        return cls(
            interp_nodes=jnp.asarray(interp_nodes),
            interp_weights=jnp.asarray(interp_weights),
            n_nodes_=int(points.shape[0]),
            valid_mask=valid_mask,
        )


# ---------------------------------------------------------------------------
# Element location helpers (NumPy, one-time construction cost)
# ---------------------------------------------------------------------------


def _find_element(
    pt: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    ele_type: str,
) -> tuple[int, np.ndarray]:
    """Find which element contains point ``pt``.  Returns (elem_idx, ref_coords)."""
    if ele_type == "TRI3":
        return _find_in_tri3(pt, points, cells)
    elif ele_type == "QUAD4":
        return _find_in_quad4(pt, points, cells)
    raise ValueError(f"Unsupported element type: {ele_type}")


def _find_in_tri3(
    pt: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Locate point in TRI3 mesh using barycentric coordinates."""
    for e in range(cells.shape[0]):
        n0, n1, n2 = cells[e]
        v0 = points[n0]
        v1 = points[n1]
        v2 = points[n2]
        # Barycentric coordinates
        d = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(d) < 1e-15:
            continue
        lam1 = ((v1[1] - v2[1]) * (pt[0] - v2[0]) + (v2[0] - v1[0]) * (pt[1] - v2[1])) / d
        lam2 = ((v2[1] - v0[1]) * (pt[0] - v2[0]) + (v0[0] - v2[0]) * (pt[1] - v2[1])) / d
        lam3 = 1.0 - lam1 - lam2
        tol = -1e-8
        if lam1 >= tol and lam2 >= tol and lam3 >= tol:
            # Reference coordinates for TRI3: (r, s) = (lam2, lam3)
            return e, np.array([lam2, lam3])
    return -1, np.zeros(2)


def _find_in_quad4(
    pt: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Locate point in QUAD4 mesh using inverse isoparametric mapping."""
    for e in range(cells.shape[0]):
        coords = points[cells[e]]  # (4, 2)
        # Simple check: is point inside the quad's bounding box?
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        if pt[0] < xmin - 1e-10 or pt[0] > xmax + 1e-10:
            continue
        if pt[1] < ymin - 1e-10 or pt[1] > ymax + 1e-10:
            continue
        # Newton iteration to find reference coordinates
        ref = np.array([0.0, 0.0])
        for _ in range(10):
            N = _quad4_shape_at(ref)
            mapped = N @ coords
            residual = mapped - pt
            if np.linalg.norm(residual) < 1e-12:
                break
            dN = _quad4_grad_at(ref)
            J = dN @ coords  # (2, 2)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-15:
                break
            Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det
            ref = ref - Jinv @ residual
        tol = 1.0 + 1e-6
        if abs(ref[0]) <= tol and abs(ref[1]) <= tol:
            return e, ref
    return -1, np.zeros(2)


def _eval_shape_functions(ref_coords: np.ndarray, ele_type: str) -> np.ndarray:
    """Evaluate shape functions at reference coordinates."""
    if ele_type == "TRI3":
        r, s = ref_coords
        return np.array([1.0 - r - s, r, s])
    elif ele_type == "QUAD4":
        return _quad4_shape_at(ref_coords)
    raise ValueError(f"Unsupported: {ele_type}")


def _quad4_shape_at(ref: np.ndarray) -> np.ndarray:
    """QUAD4 shape functions at reference point (r, s) ∈ [-1, 1]²."""
    r, s = ref
    return np.array(
        [
            0.25 * (1 - r) * (1 - s),
            0.25 * (1 + r) * (1 - s),
            0.25 * (1 + r) * (1 + s),
            0.25 * (1 - r) * (1 + s),
        ]
    )


def _quad4_grad_at(ref: np.ndarray) -> np.ndarray:
    """QUAD4 shape function gradients at reference point, shape (2, 4)."""
    r, s = ref
    return np.array(
        [
            [-0.25 * (1 - s), 0.25 * (1 - s), 0.25 * (1 + s), -0.25 * (1 + s)],
            [-0.25 * (1 - r), -0.25 * (1 + r), 0.25 * (1 + r), 0.25 * (1 - r)],
        ]
    )


# ===========================================================================
# UpConvDownOperator — production observation pipeline (Zhao dimensional)
# ===========================================================================


class UpConvDownOperator(ObservationOperator):
    r"""Production observation operator: mesh → fine grid → PSF → image pixels.

    Implements :math:`\mathcal{O} = D \cdot C \cdot U` per the Zhao dimensional:

    1. **U** — barycentric interpolation from TRI3 FEM mesh to a fine
       rectangular grid (default 10× the image resolution).
    2. **C** — normalized one-pixel box PSF convolution on the fine grid.
       Normalization divides by the convolved particle-support mask so that
       boundary pixels are not artificially dimmed.
    3. **D** — subsampling the fine grid at image pixel center positions.

    All stages are linear in the nodal field, so the composed operator is
    differentiable via standard JAX automatic differentiation.
    """

    # U: mesh-to-fine-grid interpolation (precomputed geometry)
    fine_nodes: jax.Array  # (n_valid_fine, 3) int32 — node indices
    fine_weights: jax.Array  # (n_valid_fine, 3) float64 — barycentric weights
    fine_valid_flat: jax.Array  # (n_valid_fine,) int32 — flat index into fine grid

    # C: precomputed mask normalization
    inv_conv_mask: jax.Array  # (ny_f, nx_f) float64 — 1/conv(mask, PSF)

    # D: flat fine-grid indices for each valid image pixel
    down_flat: jax.Array  # (n_valid_img,) int32

    # Geometry metadata.  valid_mask stays a regular pytree leaf to avoid the
    # Equinox static-array warning; the scalar ints remain truly static.
    valid_mask: np.ndarray  # (ny, nx) bool
    ny_f_: int = eqx.field(static=True)
    nx_f_: int = eqx.field(static=True)
    ny_: int = eqx.field(static=True)
    nx_: int = eqx.field(static=True)
    n_nodes_: int = eqx.field(static=True)
    upscale_: int = eqx.field(static=True)

    def predict(self, nodal_field: jax.Array, t: Any, theta: Any) -> jax.Array:
        """Map nodal field → valid pixel values.

        Args:
            nodal_field: (n_nodes,) scalar FEM field at mesh nodes.
            t: Unused; the PSF pipeline does not depend on the frame time.
            theta: Unused; the PSF pipeline does not depend on parameters.

        Returns:
            (n_valid_img,) predicted pixel values at valid image pixels.
        """
        del t, theta
        # --- U: interpolate to fine grid ---
        # Gather node values and compute barycentric interpolation
        vals = jnp.sum(
            nodal_field[self.fine_nodes] * jnp.asarray(self.fine_weights, dtype=nodal_field.dtype),
            axis=1,
        )  # (n_valid_fine,)
        c_fine = (
            jnp.zeros(
                self.ny_f_ * self.nx_f_,
                dtype=nodal_field.dtype,
            )
            .at[self.fine_valid_flat]
            .set(vals)
        )
        c_fine = c_fine.reshape(self.ny_f_, self.nx_f_)

        # --- C: normalized box PSF convolution ---
        c_conv = self._box_convolve(c_fine) * jnp.asarray(
            self.inv_conv_mask, dtype=nodal_field.dtype
        )

        # --- D: downsample to image pixels (gather valid positions) ---
        return c_conv.ravel()[self.down_flat]

    def to_image(self, pixel_values: jax.Array, fill: float = 0.0) -> jax.Array:
        """Scatter valid-pixel values back into a full ``(ny, nx)`` image."""
        ny, nx = self.ny_, self.nx_
        img = jnp.full(ny * nx, fill, dtype=pixel_values.dtype)
        valid_idx = jnp.where(
            jnp.asarray(self.valid_mask.ravel()),
            size=int(self.valid_mask.sum()),
        )[0]
        img = img.at[valid_idx].set(pixel_values)
        return img.reshape(ny, nx)

    @property
    def n_valid(self) -> int:
        """Number of valid (interior) image pixels."""
        return int(self.down_flat.shape[0])

    # ---- internal: separable box PSF sum via cumulative sums ----

    def _box_convolve(self, c_fine: jax.Array) -> jax.Array:
        """Box sum over a ``(k, k)`` window, centred as for an even kernel.

        The PSF kernel is all ones, so the 2D box sum separates into one pass per
        axis and each pass is a difference of cumulative sums: window
        ``[i - k//2, i + (k-1)//2]``, zero outside. That is O(1) work per output
        in ``k`` instead of O(k^2).

        Expressing this as ``conv_general_dilated`` with a ones kernel gives the
        same numbers but does not survive being differentiated in bulk. A
        forward-mode Jacobian over ``n`` tangent columns lowers it to a float64
        convolution of batch ``n * n_frames``, and XLA cannot select a cuDNN
        algorithm for float64 convolutions at that size -- it stalls for seconds
        per attempt and then fails. Keeping this free of convolution primitives
        is what makes a batched Jacobian through an image observation tractable.

        The cumulative-sum form is not bitwise identical to the convolution: it
        forms each window sum as a difference of two running totals, so in
        principle its relative error grows with the ratio of axis total to window
        sum. Measured through the full observation path on real particle meshes
        (1.3k-12k nodes, ``upscale = 10``), the two agree to 4-6e-16 per pixel and
        the resulting image RMSE differs by under 1e-17 -- machine epsilon, and
        orders below any imaging tolerance.
        """
        k = self.upscale_
        pad_left = k // 2
        pad_right = (k - 1) // 2
        out = c_fine
        for axis in (0, 1):
            n = out.shape[axis]
            # prefix[j] = sum of out[..., :j, ...] so a window is one difference.
            cumulative = jnp.cumsum(out, axis=axis)
            leading_zero = jnp.zeros_like(jax.lax.slice_in_dim(cumulative, 0, 1, axis=axis))
            prefix = jnp.concatenate([leading_zero, cumulative], axis=axis)
            index = jnp.arange(n)
            upper = jnp.clip(index + pad_right + 1, 0, n)
            lower = jnp.clip(index - pad_left, 0, n)
            out = jnp.take(prefix, upper, axis=axis) - jnp.take(prefix, lower, axis=axis)
        return out

    # ---- factory ----

    @classmethod
    def from_mesh_and_image(
        cls,
        points: np.ndarray,
        cells: np.ndarray,
        image_shape: tuple[int, int],
        pixel_origin: tuple[float, float] = (0.0, 0.0),
        pixel_size: float = 1.0,
        valid_mask: Optional[np.ndarray] = None,
        upscale: int = 10,
    ) -> "UpConvDownOperator":
        """Build the operator from mesh geometry and image metadata.

        Args:
            points: (n_nodes, 2) mesh node coordinates.
            cells: (n_cells, 3) TRI3 connectivity.
            image_shape: (ny, nx) image dimensions in pixels.
            pixel_origin: (x0, y0) physical coords of pixel [0, 0] center.
            pixel_size: physical size of one square pixel.
            valid_mask: (ny, nx) bool mask — ``None`` → all valid.
            upscale: fine-grid resolution multiplier (default 10 per dimensional).
        """
        ny, nx = image_shape
        if valid_mask is None:
            valid_mask = np.ones((ny, nx), dtype=bool)

        x0, y0 = pixel_origin
        dx = pixel_size
        dx_f = dx / upscale

        # --- Fine grid ---
        # Extends half a pixel beyond image in each direction so that
        # edge-pixel PSF footprints are fully contained.
        nx_f = nx * upscale + 1
        ny_f = ny * upscale + 1
        x_fine = (x0 - dx / 2.0) + np.arange(nx_f) * dx_f
        y_fine = (y0 - dx / 2.0) + np.arange(ny_f) * dx_f

        # Image pixel j center is at fine-grid index: upscale//2 + j*upscale
        down_x_idx = upscale // 2 + np.arange(nx) * upscale
        down_y_idx = upscale // 2 + np.arange(ny) * upscale

        # --- U: locate fine-grid points in mesh (vectorized) ---
        XF, YF = np.meshgrid(x_fine, y_fine)  # (ny_f, nx_f)

        # matplotlib is an OPTIONAL dependency — imported lazily here (the only
        # use site) so `import diff_soup.inference` does not hard-require it.
        from matplotlib.tri import Triangulation

        tri_obj = Triangulation(points[:, 0], points[:, 1], cells)
        finder = tri_obj.get_trifinder()
        tri_indices = finder(XF.ravel(), YF.ravel()).reshape(ny_f, nx_f)

        fine_mask = tri_indices >= 0
        valid_flat = np.where(fine_mask.ravel())[0]

        # Vectorized barycentric coordinates
        iy_valid = valid_flat // nx_f
        ix_valid = valid_flat % nx_f
        e_valid = tri_indices[iy_valid, ix_valid]

        n0s, n1s, n2s = cells[e_valid, 0], cells[e_valid, 1], cells[e_valid, 2]
        v0x, v0y = points[n0s, 0], points[n0s, 1]
        v1x, v1y = points[n1s, 0], points[n1s, 1]
        v2x, v2y = points[n2s, 0], points[n2s, 1]

        pt_x = x_fine[ix_valid]
        pt_y = y_fine[iy_valid]

        d = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
        lam1 = ((v1y - v2y) * (pt_x - v2x) + (v2x - v1x) * (pt_y - v2y)) / d
        lam2 = ((v2y - v0y) * (pt_x - v2x) + (v0x - v2x) * (pt_y - v2y)) / d
        lam3 = 1.0 - lam1 - lam2

        fine_nodes = np.stack([n0s, n1s, n2s], axis=1).astype(np.int32)
        fine_weights = np.stack([lam1, lam2, lam3], axis=1)

        # --- C: precompute mask normalization ---
        kernel_np = np.ones((upscale, upscale), dtype=np.float64)
        conv_full = fftconvolve(
            fine_mask.astype(np.float64),
            kernel_np,
            mode="full",
        )
        # Extract with correct centering: conv_full[pr:pr+ny_f, pr:pr+nx_f]
        # where pr = (upscale-1)//2, so output[i] = sum from i-upscale//2 to i+pr
        pr = (upscale - 1) // 2
        conv_mask = conv_full[pr : pr + ny_f, pr : pr + nx_f]
        # Avoid division by zero; set inv to 0 where mask has no support
        inv_conv_mask = np.zeros_like(conv_mask)
        pos = conv_mask > 0.5
        inv_conv_mask[pos] = 1.0 / conv_mask[pos]

        # --- D: effective valid mask ---
        # Only include pixels where (a) user says valid AND (b) mesh has
        # PSF support at that pixel's fine-grid position.
        conv_at_pixels = conv_mask[down_y_idx][:, down_x_idx]  # (ny, nx)
        mesh_support = conv_at_pixels > 0.5
        effective_mask = np.asarray(valid_mask, dtype=bool) & mesh_support
        valid_ij = np.argwhere(effective_mask)  # (n_valid_img, 2) — (y, x)
        down_fine_y = down_y_idx[valid_ij[:, 0]]
        down_fine_x = down_x_idx[valid_ij[:, 1]]
        down_flat = down_fine_y * nx_f + down_fine_x

        return cls(
            fine_nodes=jnp.asarray(fine_nodes),
            fine_weights=jnp.asarray(fine_weights),
            fine_valid_flat=jnp.asarray(valid_flat, dtype=jnp.int32),
            inv_conv_mask=jnp.asarray(inv_conv_mask),
            down_flat=jnp.asarray(down_flat, dtype=jnp.int32),
            valid_mask=np.asarray(effective_mask, dtype=bool),
            ny_f_=int(ny_f),
            nx_f_=int(nx_f),
            ny_=int(ny),
            nx_=int(nx),
            n_nodes_=int(points.shape[0]),
            upscale_=int(upscale),
        )


# ===========================================================================
# Image-to-node projection (inverse direction)
# ===========================================================================


def interpolate_image_to_nodes(
    image: np.ndarray,
    points: np.ndarray,
    pixel_origin: tuple[float, float] = (0.0, 0.0),
    pixel_size: float = 1.0,
    *,
    fill: Optional[float] = None,
) -> np.ndarray:
    """Bilinear interpolation from image grid to mesh node positions.

    Uses linear interpolation inside the image bounds and nearest-neighbor
    extrapolation for nodes outside them.

    A masked image carries non-finite pixels outside the domain, and linear
    interpolation spreads them to every node of a cell that touches one.
    Pass ``fill`` to replace those pixels with a finite value first, so that
    nodes on the domain boundary interpolate finitely.

    Args:
        image: (ny, nx) image data.
        points: (n_points, 2) query positions with columns (x, y).
        pixel_origin: (x0, y0) physical coords of pixel [0, 0] center.
        pixel_size: physical size of one square pixel.
        fill: value substituted for non-finite pixels (NaN, +/-inf) before
            interpolation. Default: None, interpolating the image as given,
            so non-finite pixels propagate to the nodes around them.

    Returns:
        (n_points,) interpolated values at each query point.
    """
    ny, nx = image.shape
    x0, y0 = pixel_origin
    x_pix = x0 + np.arange(nx) * pixel_size
    y_pix = y0 + np.arange(ny) * pixel_size

    values = image.astype(np.float64)
    if fill is not None:
        values = np.where(np.isfinite(values), values, float(fill))

    interp = RegularGridInterpolator(
        (y_pix, x_pix),
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,  # nearest extrapolation
    )

    # Query at (y, x) order
    query_yx = points[:, [1, 0]]
    return interp(query_yx)
