"""geometry/mask.py — Extract low-poly boundary polygons from binary masks.

Uses OpenCV's contour detection and the Douglas-Peucker polygon approximation
to convert raster binary masks (e.g. particle cross-sections) into compact
polygons suitable for gmsh triangulation.

Contour tracing runs on ``opencv-python``, which installs with diff_soup.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from diff_soup._optional import import_optional


def mask_to_polygon(
    mask: np.ndarray,
    *,
    epsilon_factor: float = 0.02,
    pixel_size: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    contour_index: int = 0,
) -> List[Tuple[float, float]]:
    """Convert a binary raster mask to a low-poly boundary polygon.

    Uses ``cv2.findContours`` to extract the outer boundary and
    ``cv2.approxPolyDP`` (Douglas-Peucker) to reduce it to a compact
    polygon with straight edges.

    Args:
        mask: 2-D NumPy array (uint8 or bool).  Non-zero pixels are treated as
            "inside".  Shape ``(H, W)``.
        epsilon_factor: Controls the aggressiveness of polygon simplification.  The Douglas-
            Peucker epsilon is ``epsilon_factor * perimeter``.  Larger values
            produce fewer vertices but a coarser shape.  Typical range 0.005–0.05.
        pixel_size: Physical size of one pixel.  All returned coordinates are multiplied
            by this value so the polygon is in physical units.
        origin: ``(x0, y0)`` offset added to all polygon vertices (physical units).
        contour_index: Which contour to return when the mask has multiple disconnected
            regions.  The contours are sorted by area (largest first).

    Returns:
        Ordered list of ``(x, y)`` vertices in physical coordinates.  The
        polygon is *not* closed (first vertex ≠ last vertex).

    Raises:
        ValueError: if no contour is found (mask is all-zero).
        IndexError: if ``contour_index`` exceeds the number of found contours.

    Example:
        ::

            import numpy as np
            from diff_soup.domains.geometry import mask_to_polygon

            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 16:48] = 1          # square region

            polygon = mask_to_polygon(mask, epsilon_factor=0.01, pixel_size=0.1)
            # → 4 corner vertices for a square
    """
    cv2 = import_optional("cv2")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.max() > 1:
        # Already uint8 image-range; binarise
        _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    else:
        mask_u8 = mask_u8 * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in mask — check that the mask has non-zero pixels.")

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[contour_index]  # shape (N, 1, 2)

    # Douglas-Peucker approximation
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)  # shape (M, 1, 2)
    pts = approx.reshape(-1, 2).astype(np.float64)  # (M, 2) in pixel coords

    # Convert to physical coordinates
    # OpenCV uses (col, row) = (x, y) convention already
    x0, y0 = origin
    polygon = [(float(p[0]) * pixel_size + x0, float(p[1]) * pixel_size + y0) for p in pts]
    return polygon
