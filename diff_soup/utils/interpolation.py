"""JIT-safe PCHIP interpolation for use inside JAX-traced code.

The standard ``interpax.PchipInterpolator`` performs input validation
(boolean checks on finiteness) during construction, which fails when called
inside ``jax.lax.cond`` or other tracing contexts.

This module provides [PchipCoeffs][diff_soup.utils.interpolation.PchipCoeffs] which precomputes PCHIP spline
coefficients eagerly (outside JIT), and [pchip_eval][diff_soup.utils.interpolation.pchip_eval] which evaluates
the spline using only trace-safe JAX operations.

Usage::

    # Build coefficients eagerly (outside JIT)
    coeffs = PchipCoeffs.from_data(ts_numpy, vs_numpy)

    # Evaluate inside JIT / lax.cond / vmap / grad
    val = pchip_eval(coeffs, t)        # scalar t → scalar or (n_cols,)
    vals = pchip_eval_batch(coeffs, t) # scalar t → (n_cols,) for 2D vs

Both ``pchip_eval`` and ``pchip_eval_batch`` are differentiable w.r.t. ``t``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PchipCoeffs(NamedTuple):
    """Precomputed PCHIP spline coefficients for JIT-safe evaluation.

    Stores the breakpoints and Hermite basis coefficients so that evaluation
    only requires ``jnp.searchsorted`` plus polynomial evaluation — no boolean
    checks, no object construction inside JIT.

    Fields:
        x: Breakpoints, shape ``(n,)``, strictly increasing.
        c: Hermite polynomial coefficients, shape ``(4, n-1)`` for scalar
            data or ``(4, n-1, m)`` for multi-column data. ``c[0]`` is the
            cubic term; ``c[3]`` is the constant term.
    """

    x: jax.Array  # (n,)
    c: jax.Array  # (4, n-1) or (4, n-1, m)

    @classmethod
    def from_data(
        cls,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "PchipCoeffs":
        """Build PCHIP coefficients eagerly from NumPy data.

        Args:
            x: Strictly increasing breakpoints, shape ``(n,)``.
            y: Function values at breakpoints, shape ``(n,)`` or ``(n, m)``.

        Returns:
            Precomputed [PchipCoeffs][diff_soup.utils.interpolation.PchipCoeffs] ready for JIT-safe evaluation.

        Raises:
            ValueError: If fewer than 2 data points are provided.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64)
        n = len(x)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        multi = y.ndim == 2
        if not multi:
            y = y[:, None]  # (n, 1)

        m = y.shape[1]
        h = np.diff(x)  # (n-1,)
        delta = np.diff(y, axis=0) / h[:, None]  # (n-1, m)

        # PCHIP slopes via Fritsch-Carlson method
        d = np.zeros_like(y)  # (n, m)
        for col in range(m):
            d[:, col] = _pchip_slopes(x, y[:, col], h, delta[:, col])

        # Hermite basis coefficients for each interval.
        # On interval [x_i, x_{i+1}], the polynomial is:
        #   p(s) = c0*s^3 + c1*s^2 + c2*s + c3  where s = x - x_i
        # p(0) = yi, p(hi) = yi1, p'(0) = di, p'(hi) = di1
        c = np.zeros((4, n - 1, m), dtype=np.float64)
        for i in range(n - 1):
            hi = h[i]
            for col in range(m):
                yi = y[i, col]
                yi1 = y[i + 1, col]
                di = d[i, col]
                di1 = d[i + 1, col]
                c[3, i, col] = yi
                c[2, i, col] = di
                c[1, i, col] = (3.0 * (yi1 - yi) / hi - 2.0 * di - di1) / hi
                c[0, i, col] = (di + di1 - 2.0 * (yi1 - yi) / hi) / (hi * hi)

        if not multi:
            c = c[:, :, 0]  # (4, n-1)

        return cls(
            x=jnp.asarray(x),
            c=jnp.asarray(c),
        )


def pchip_eval(coeffs: PchipCoeffs, t: jax.Array) -> jax.Array:
    """Evaluate precomputed PCHIP spline at scalar ``t``.

    JIT-safe and differentiable w.r.t. ``t``. Clamps ``t`` to the data range.

    Args:
        coeffs: Precomputed spline coefficients from [PchipCoeffs.from_data][diff_soup.utils.interpolation.PchipCoeffs.from_data].
        t: Scalar evaluation point.

    Returns:
        Scalar if ``coeffs`` was built from 1-D data; shape ``(m,)`` for 2-D
        data.
    """
    x = coeffs.x
    c = coeffs.c
    n = x.shape[0]

    # Clamp t to data range
    t = jnp.clip(t, x[0], x[-1])

    # Find interval: i such that x[i] <= t < x[i+1]
    i = jnp.clip(jnp.searchsorted(x, t, side="right") - 1, 0, n - 2)

    # Local coordinate
    s = t - x[i]

    # Evaluate cubic: c[0]*s^3 + c[1]*s^2 + c[2]*s + c[3]
    if c.ndim == 2:
        # 1D data: c is (4, n-1)
        return c[0, i] * s**3 + c[1, i] * s**2 + c[2, i] * s + c[3, i]
    else:
        # 2D data: c is (4, n-1, m)
        return c[0, i] * s**3 + c[1, i] * s**2 + c[2, i] * s + c[3, i]


def pchip_eval_batch(coeffs: PchipCoeffs, t: jax.Array) -> jax.Array:
    """Evaluate precomputed PCHIP at scalar ``t``, returning ``(m,)`` for multi-column data.

    Alias for [pchip_eval][diff_soup.utils.interpolation.pchip_eval]; provided so callers that always work with
    multi-column data can use a descriptive name.

    Args:
        coeffs: Precomputed spline coefficients from [PchipCoeffs.from_data][diff_soup.utils.interpolation.PchipCoeffs.from_data].
        t: Scalar evaluation point.

    Returns:
        Shape ``(m,)`` array of interpolated values.
    """
    return pchip_eval(coeffs, t)


def pchip_derivative_coeffs(coeffs: PchipCoeffs) -> PchipCoeffs:
    """Analytically differentiate PCHIP coefficients.

    Given ``p(s) = c0*s³ + c1*s² + c2*s + c3``, the derivative is
    ``p'(s) = 3*c0*s² + 2*c1*s + c2``.

    Returns a new [PchipCoeffs][diff_soup.utils.interpolation.PchipCoeffs] with cubic degree reduced to quadratic,
    stored in the same ``(4, n-1)`` layout with ``c[0]=0`` padding so that
    [pchip_eval][diff_soup.utils.interpolation.pchip_eval] can be reused directly.

    Args:
        coeffs: Precomputed spline coefficients to differentiate.

    Returns:
        New [PchipCoeffs][diff_soup.utils.interpolation.PchipCoeffs] whose [pchip_eval][diff_soup.utils.interpolation.pchip_eval] gives the derivative.
    """
    c = coeffs.c
    if c.ndim == 2:
        # (4, n-1): c[0]=cubic, c[1]=quadratic, c[2]=linear, c[3]=constant
        dc = jnp.stack(
            [
                jnp.zeros_like(c[0]),  # pad: no cubic term in derivative
                3.0 * c[0],  # s² coefficient
                2.0 * c[1],  # s  coefficient
                c[2],  # constant coefficient
            ]
        )
    else:
        # (4, n-1, m)
        dc = jnp.stack(
            [
                jnp.zeros_like(c[0]),
                3.0 * c[0],
                2.0 * c[1],
                c[2],
            ]
        )
    return PchipCoeffs(x=coeffs.x, c=dc)


def _pchip_slopes(x, y, h, delta):
    """Compute PCHIP slopes using Fritsch-Carlson method.

    Args:
        x: Breakpoints, shape ``(n,)``.
        y: Function values, shape ``(n,)``.
        h: Interval widths, shape ``(n-1,)`` (``np.diff(x)``).
        delta: Divided differences, shape ``(n-1,)`` (``np.diff(y) / h``).

    Returns:
        Slope array, shape ``(n,)``.
    """
    n = len(x)
    d = np.zeros(n)

    if n == 2:
        d[0] = d[1] = delta[0]
        return d

    # Interior points: weighted harmonic mean
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0

    # Endpoints: one-sided shape-preserving formula
    d[0] = _pchip_endpoint(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_endpoint(h[-1], h[-2], delta[-1], delta[-2])

    return d


def _pchip_endpoint(h1, h2, del1, del2):
    """PCHIP endpoint slope (non-centered shape-preserving).

    Args:
        h1: Width of the adjacent interval.
        h2: Width of the next-adjacent interval.
        del1: Divided difference on the adjacent interval.
        del2: Divided difference on the next-adjacent interval.

    Returns:
        Endpoint slope scalar.
    """
    d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
    if np.sign(d) != np.sign(del1):
        d = 0.0
    elif np.sign(del1) != np.sign(del2) and abs(d) > abs(3 * del1):
        d = 3 * del1
    return d
