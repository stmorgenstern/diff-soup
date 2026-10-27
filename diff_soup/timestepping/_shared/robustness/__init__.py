"""Robustness helper shared by the consistent-initialization machinery.

Exposes ``recover_algebraic_derivative`` — the algebraic-row derivative
recovery used by the DAE consistent-IC projection.
"""

from diff_soup.timestepping._shared.robustness.derivative_ic import (  # noqa: F401
    recover_algebraic_derivative,
)
