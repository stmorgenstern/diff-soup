"""Curated public re-exports of low-level FVM operators.

Escape hatch for direct access to FVM stencil operators on
[Grid1D][diff_soup.domains.grid.Grid1D] / [GridND][diff_soup.domains.grid.GridND]. For most users, the
high-level [Operators][diff_soup.Operators] dispatch class is the recommended
entry point.
"""

from __future__ import annotations

from ._kernels import (
    fvm_apply_neumann_flux_1d,
    fvm_div_flux_1d,
    fvm_div_flux_2d,
    fvm_div_flux_3d,
    fvm_face_flux_1d,
    fvm_gradient_1d,
    fvm_laplacian_1d,
    fvm_laplacian_2d,
    fvm_laplacian_3d,
)

__all__ = [
    "fvm_face_flux_1d",
    "fvm_div_flux_1d",
    "fvm_laplacian_1d",
    "fvm_gradient_1d",
    "fvm_apply_neumann_flux_1d",
    "fvm_div_flux_2d",
    "fvm_div_flux_3d",
    "fvm_laplacian_2d",
    "fvm_laplacian_3d",
]
