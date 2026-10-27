"""Quadrature rules shared by every element type.

These helpers return reference-element quadrature points and weights for
the reference shapes used throughout [`diff_soup.domains.mesh.fem.basis`][diff_soup.domains.mesh.fem.basis]:

* ``_gauss_1d`` / ``_gauss_1d_3`` — Gauss-Legendre on ``[-1, 1]``.
* ``_gauss_tri`` — symmetric quadrature on the unit triangle ``(0,0)-(1,0)-(0,1)``.
* ``_gauss_tet`` — symmetric quadrature on the unit tetrahedron
  ``(0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)``.
"""

from __future__ import annotations

import numpy as np


def _gauss_tri(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss quadrature for the reference triangle (0,0)-(1,0)-(0,1)."""
    if order == 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5])
    if order == 2:
        pts = np.array([[1.0 / 6, 1.0 / 6], [2.0 / 3, 1.0 / 6], [1.0 / 6, 2.0 / 3]])
        return pts, np.array([1.0 / 6, 1.0 / 6, 1.0 / 6])
    raise NotImplementedError("Only order=1 or 2 is supported for TRI3 quadrature")


def _gauss_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature on ``[-1, 1]`` (order 1 or 2)."""
    if order == 1:
        return np.array([0.0]), np.array([2.0])
    if order == 2:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    raise NotImplementedError("Only order=1 or 2 is supported in native basis backend")


def _gauss_tet(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss quadrature for the reference tetrahedron.

    Vertices ``(0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)``; weights sum to ``1/6``.
    """
    if order == 1:
        return (
            np.array([[0.25, 0.25, 0.25]]),
            np.array([1.0 / 6.0]),
        )
    if order == 2:
        a = 0.1381966011250105
        b = 0.5854101966249685
        pts = np.array([[a, a, a], [b, a, a], [a, b, a], [a, a, b]])
        return pts, np.full(4, 1.0 / 24.0)
    raise NotImplementedError("Only order=1 or 2 is supported for TET quadrature")


def _gauss_1d_3(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss quadrature on [-1,1] up to order 3."""
    if order <= 1:
        return np.array([0.0]), np.array([2.0])
    if order <= 2:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    if order <= 3:
        a = np.sqrt(3.0 / 5.0)
        return np.array([-a, 0.0, a]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    raise NotImplementedError("Only order<=3 is supported for 1D quadrature")
