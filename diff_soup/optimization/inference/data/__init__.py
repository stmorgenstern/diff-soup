"""Differentiable observation operators for ``diff_soup.inference``.

Maps PDE solution fields to observable quantities (images, point measurements, etc.)
for comparison with experimental data.  Observation operators are composable with
the PDE solver in the loss function.

Modules:

* [base][diff_soup.inference.data.base] — Abstract protocol
* [pixel_averaging][diff_soup.inference.data.pixel_averaging] — FEM field → image pixel comparison
* [distribution][diff_soup.inference.data.distribution] — population (distribution-space)
  observable + kernels
"""

from diff_soup.inference.data.base import ObservationOperator
from diff_soup.inference.data.boundary_smoothing import (
    extract_boundary_traces,
    smooth_boundary,
    smooth_boundary_raw,
    smooth_boundary_savgol,
    smooth_boundary_spline,
)
from diff_soup.inference.data.distribution import (
    CDFBins,
    Moments,
    PopulationObservable,
    SoftHistogram,
)
from diff_soup.inference.data.pixel_averaging import (
    PixelAveragingOperator,
    UpConvDownOperator,
    interpolate_image_to_nodes,
)

__all__ = [
    "ObservationOperator",
    "PixelAveragingOperator",
    "UpConvDownOperator",
    "interpolate_image_to_nodes",
    "extract_boundary_traces",
    "smooth_boundary",
    "smooth_boundary_raw",
    "smooth_boundary_savgol",
    "smooth_boundary_spline",
    # Population / distribution observable
    "PopulationObservable",
    "SoftHistogram",
    "Moments",
    "CDFBins",
]
