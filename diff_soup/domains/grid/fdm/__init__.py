"""Curated public re-exports of low-level FDM operators.

Escape hatch for direct access to FDM stencil operators on
[Grid1D][diff_soup.domains.grid.Grid1D] / [GridND][diff_soup.domains.grid.GridND]. For most users, the
high-level [Operators][diff_soup.domains.Operators] dispatch class is the recommended
entry point.
"""

from __future__ import annotations

from ._kernels import (
    fdm_div_flux_1d,
    fdm_div_flux_2d,
    fdm_div_flux_3d,
    fdm_gradient_1d,
    fdm_laplacian_1d,
    fdm_laplacian_2d,
    fdm_laplacian_3d,
    fdm_second_derivative_1d,
)

__all__ = [
    "fdm_gradient_1d",
    "fdm_second_derivative_1d",
    "fdm_laplacian_1d",
    "fdm_div_flux_1d",
    "fdm_laplacian_2d",
    "fdm_div_flux_2d",
    "fdm_laplacian_3d",
    "fdm_div_flux_3d",
]
