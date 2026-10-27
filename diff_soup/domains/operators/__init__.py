"""Spatial operator dispatch layer.

[Operators][diff_soup.domains.operators.base.Operators] binds to a
[Domain][diff_soup.domains.base.Domain] and provides a unified API for common
PDE operators (Laplacian, div-flux, gradient) that dispatches to the correct
backend (FVM, FDM, or FEM).

Sub-namespaces for escape-hatch access to low-level discretization-specific
operators:

- [diff_soup.domains.operators.fem][] -- FEM element kernels
- [diff_soup.domains.grid.fvm][] -- FVM stencil operators
- [diff_soup.domains.grid.fdm][] -- FDM stencil operators
- [diff_soup.domains.grid][] -- structured grid constructors
"""

from . import fem
from .base import AbstractOperator, Operators

__all__ = [
    # Extension contract ABC — escape hatch for custom discretizations (DG,
    # spectral, particle). Users subclass this, not Operators directly.
    # Operators is duck-type compatible but does NOT inherit.
    "AbstractOperator",
    "Operators",
    "fem",
]
