"""observation/boundary_smoothing.py — Boundary trace smoothing utilities.

Provides three strategies for smoothing concentration traces along a closed
FEM boundary contour before imposing them as Dirichlet boundary conditions:

1. **Raw interpolation** (``smooth_boundary_raw``): No smoothing.  Bilinear
   interpolation from image pixels to FEM boundary nodes.  Faithful to
   experimental data.

2. **Savitzky-Golay** (``smooth_boundary_savgol``): periodic Savitzky-Golay
   filter along the contour, following the preprocessing approach of
   Zhao et al.

3. **Smoothing spline** (``smooth_boundary_spline``): Principled smoothing
   spline with noise parameter σ_ε.  Uses ``UnivariateSpline`` with
   ``s = 3N`` (Wahba/Craven criterion for 3× periodic wrapping) and
   weights ``w = 1/σ_ε``.

All functions accept ``(n_frames, n_bnd)`` raw boundary values and return
``(n_frames, n_bnd)`` smoothed values clipped to ``[c_lo, c_hi]``.

Usage::

    from diff_soup.inference.data import interpolate_image_to_nodes
    from diff_soup.inference.data.boundary_smoothing import (
        extract_boundary_traces,
        smooth_boundary_spline,
    )

    bc_raw = extract_boundary_traces(c_images, mesh_points, bnd_indices)
    bc_smooth = smooth_boundary_spline(bc_raw, mesh_points, bnd_indices, sigma_eps=0.01)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from diff_soup.inference.data.pixel_averaging import interpolate_image_to_nodes


def extract_boundary_traces(
    c_images: np.ndarray,
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
    pixel_origin: tuple[float, float] = (0.0, 0.0),
    pixel_size: float = 1.0,
    fill_nan: Optional[float] = None,
) -> np.ndarray:
    """Bilinear interpolation from image frames to boundary mesh nodes.

    Args:
        c_images: (n_frames, ny, nx) concentration images.  NaN outside particle.
        mesh_points: (n_nodes, 2) mesh node coordinates (x, y).
        boundary_indices: (n_bnd,) integer indices of boundary nodes.
        pixel_origin: (x0, y0) physical coords of pixel [0, 0] center.
        pixel_size: physical size of one square pixel.
        fill_nan: value to replace NaN pixels before interpolation.
            Default: None (use minimum finite value in each frame).

    Returns:
        bc_raw: (n_frames, n_bnd) boundary concentrations (unsmoothed).
    """
    n_frames = c_images.shape[0]
    bnd_pts = mesh_points[boundary_indices]
    n_bnd = len(boundary_indices)
    bc_raw = np.zeros((n_frames, n_bnd), dtype=np.float64)

    for f in range(n_frames):
        frame = c_images[f].copy()
        if fill_nan is not None:
            frame[np.isnan(frame)] = fill_nan
        else:
            fmin = np.nanmin(frame) if np.any(np.isfinite(frame)) else 0.0
            frame[np.isnan(frame)] = fmin
        bc_raw[f] = interpolate_image_to_nodes(frame, bnd_pts, pixel_origin, pixel_size)

    return bc_raw


def _contour_sort(
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort boundary nodes by angle around centroid.

    Returns (angles_sorted, sort_idx, unsort_idx).
    """
    bnd_pts = mesh_points[boundary_indices]
    centroid = bnd_pts.mean(axis=0)
    angles = np.arctan2(bnd_pts[:, 1] - centroid[1], bnd_pts[:, 0] - centroid[0])
    sort_idx = np.argsort(angles)
    unsort_idx = np.argsort(sort_idx)
    return angles[sort_idx], sort_idx, unsort_idx


def smooth_boundary_raw(
    bc_raw: np.ndarray,
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
    c_lo: float = 0.0126,
    c_hi: float = 0.9874,
) -> np.ndarray:
    """No smoothing — just clip to admissible interval.

    Args:
        bc_raw: (n_frames, n_bnd) raw boundary values.
        mesh_points: used only for API consistency.
        boundary_indices: used only for API consistency.
        c_lo: lower clipping bound.
        c_hi: upper clipping bound.

    Returns:
        bc: (n_frames, n_bnd) clipped boundary values.
    """
    return np.clip(bc_raw, c_lo, c_hi)


def smooth_boundary_savgol(
    bc_raw: np.ndarray,
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
    c_lo: float = 0.0126,
    c_hi: float = 0.9874,
    window: int = 7,
    order: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing along the boundary contour.

    Sorts nodes by angle, applies periodic wrap-around padding, filters with
    ``scipy.signal.savgol_filter``, and unscrambles back to original ordering.

    Args:
        bc_raw: (n_frames, n_bnd) raw boundary values.
        mesh_points: (n_nodes, 2) mesh coordinates.
        boundary_indices: (n_bnd,) boundary node indices.
        c_lo: lower clipping bound.
        c_hi: upper clipping bound.
        window: filter window length (must be odd).
        order: polynomial order.

    Returns:
        bc: (n_frames, n_bnd) smoothed, clipped boundary values.
    """
    _, sort_idx, unsort_idx = _contour_sort(mesh_points, boundary_indices)
    n_frames, n_bnd = bc_raw.shape

    win = min(window, n_bnd)
    if win % 2 == 0:
        win -= 1
    win = max(win, order + 2)
    if win > n_bnd:
        win = n_bnd if n_bnd % 2 == 1 else n_bnd - 1

    bc_smooth = np.zeros_like(bc_raw)
    for f in range(n_frames):
        sv = bc_raw[f, sort_idx]
        if n_bnd >= win >= order + 2:
            padded = np.concatenate([sv[-win:], sv, sv[:win]])
            filtered = savgol_filter(padded, win, order)
            bc_smooth[f] = filtered[win : win + n_bnd][unsort_idx]
        else:
            bc_smooth[f] = bc_raw[f]

    return np.clip(bc_smooth, c_lo, c_hi)


def smooth_boundary_spline(
    bc_raw: np.ndarray,
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
    c_lo: float = 0.0126,
    c_hi: float = 0.9874,
    sigma_eps: float = 0.01,
) -> np.ndarray:
    """Smoothing spline with noise parameter σ_ε (principled, periodic).

    Uses ``UnivariateSpline`` with periodic 3× wrapping and Wahba/Craven
    smoothing parameter ``s = 3N`` (expected weighted residual sum over
    the tripled domain).  Weights ``w = 1/σ_ε``.

    Larger σ_ε → more smoothing.  Typical values:
    - σ_ε = 0.01 (Degnan-Morgenstern): nearly interpolating, preserves features
    - σ_ε = 0.07 (Zhao et al.): moderate smoothing, removes ~1-pixel noise

    Args:
        bc_raw: (n_frames, n_bnd) raw boundary values.
        mesh_points: (n_nodes, 2) mesh coordinates.
        boundary_indices: (n_bnd,) boundary node indices.
        c_lo: lower clipping bound.
        c_hi: upper clipping bound.
        sigma_eps: per-pixel noise standard deviation.

    Returns:
        bc: (n_frames, n_bnd) smoothed, clipped boundary values.
    """
    angles_sorted, sort_idx, unsort_idx = _contour_sort(mesh_points, boundary_indices)
    n_frames, n_bnd = bc_raw.shape

    w = np.full(n_bnd, 1.0 / sigma_eps)
    theta_ext = np.concatenate(
        [angles_sorted - 2 * np.pi, angles_sorted, angles_sorted + 2 * np.pi]
    )
    w_ext = np.tile(w, 3)
    s_val = 3 * n_bnd  # Wahba criterion

    bc_smooth = np.zeros_like(bc_raw)
    for f in range(n_frames):
        c_sorted = bc_raw[f, sort_idx]
        c_ext = np.tile(c_sorted, 3)
        try:
            spl = UnivariateSpline(theta_ext, c_ext, w=w_ext, s=s_val)
            bc_smooth[f, sort_idx] = spl(angles_sorted)
        except Exception:
            # Fallback to raw if spline fitting fails
            bc_smooth[f] = bc_raw[f]

    return np.clip(bc_smooth, c_lo, c_hi)


def smooth_boundary(
    bc_raw: np.ndarray,
    mesh_points: np.ndarray,
    boundary_indices: np.ndarray,
    method: Literal["raw", "savgol", "spline"] = "spline",
    c_lo: float = 0.0126,
    c_hi: float = 0.9874,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to a boundary smoothing method.

    Args:
        bc_raw: Raw boundary values, shape ``(n_frames, n_bnd)``.
        mesh_points: FEM node coordinates, shape ``(n_nodes, 2)``.
        boundary_indices: Indices of the boundary nodes into ``mesh_points``.
        method: one of ``"raw"``, ``"savgol"``, ``"spline"``.
        c_lo: Lower clip bound applied to the smoothed values.
        c_hi: Upper clip bound applied to the smoothed values.
        **kwargs: passed to the chosen method.
            For ``"savgol"``: ``window=7``, ``order=3``.
            For ``"spline"``: ``sigma_eps=0.01``.

    Returns:
        bc: (n_frames, n_bnd) smoothed, clipped boundary values.
    """
    if method == "raw":
        return smooth_boundary_raw(bc_raw, mesh_points, boundary_indices, c_lo, c_hi, **kwargs)
    elif method == "savgol":
        return smooth_boundary_savgol(bc_raw, mesh_points, boundary_indices, c_lo, c_hi, **kwargs)
    elif method == "spline":
        return smooth_boundary_spline(bc_raw, mesh_points, boundary_indices, c_lo, c_hi, **kwargs)
    else:
        raise ValueError(
            f"Unknown boundary smoothing method: {method!r}. Expected 'raw', 'savgol', or 'spline'."
        )
