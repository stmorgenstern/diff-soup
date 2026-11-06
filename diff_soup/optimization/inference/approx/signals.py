"""signals.py — Learnable time-signal basis functions.

Provides four small ``eqx.Module`` callables for representing a
time-dependent quantity — a boundary condition, a forcing term, a control
input — as a differentiable function of a scalar time ``t``:

* [PiecewiseConstant][diff_soup.inference.approx.signals.PiecewiseConstant] — a
  step function on fixed segments.
* [PiecewiseLinear][diff_soup.inference.approx.signals.PiecewiseLinear] — a
  continuous, piecewise-linear interpolant.
* [CubicHermite][diff_soup.inference.approx.signals.CubicHermite] — a smooth,
  per-segment cubic interpolant with user-supplied slopes at each knot.
* [Fourier][diff_soup.inference.approx.signals.Fourier] — a truncated Fourier
  series, periodic by construction.

Every class exposes ``__call__(t) -> value`` (a scalar, or an array with a
trailing channel dimension for multi-component signals) plus a uniform
``.knots`` attribute — the segment edges for the three piecewise bases, and
an empty array for [Fourier][diff_soup.inference.approx.signals.Fourier]
(which has none). All coefficient and
knot leaves are ordinary ``jax.Array`` values (shifting them never triggers a
recompile), and none of the evaluation logic uses Python control flow that
would need to trace differently for different inputs — every method is safe
to wrap in ``jax.vmap`` or differentiate with ``jax.grad`` / ``jax.jacfwd``.
When a signal is selected as a fit learnable by name, its ``learnable_paths``
pick only the coefficient leaves (values / slopes / harmonic amplitudes) —
the knots and the Fourier period define the basis and are not fit
coefficients (a knot's gradient is zero almost everywhere).

Usage example::

    knots = jnp.array([0.0, 1.0, 2.0, 3.0])
    forcing = PiecewiseConstant(knots, jnp.array([1.0, -1.0, 2.0]))
    # Inside a spec equation: q.fn.forcing(q.t)

Segment convention
------------------
The three piecewise bases
([PiecewiseConstant][diff_soup.inference.approx.signals.PiecewiseConstant],
[PiecewiseLinear][diff_soup.inference.approx.signals.PiecewiseLinear],
[CubicHermite][diff_soup.inference.approx.signals.CubicHermite]) all select
the active segment the same way: a query
time that lands exactly on an interior knot is treated as belonging to the
segment *to its left*. Concretely, a numerical solver that has just
integrated a step ending precisely at knot ``t_k`` and then evaluates the
signal at ``t == t_k`` sees the value the step was actually integrated
against — the left segment — not the one that is about to start.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _inexact(x) -> jax.Array:
    """Convert ``x`` to a ``jax.Array`` with an inexact (floating-point) dtype.

    Integer or boolean input is promoted to the platform's default float
    dtype (``float32`` or ``float64``, depending on whether 64-bit JAX is
    enabled); an array that is already floating-point is returned unchanged
    (its dtype is preserved).
    """
    arr = jnp.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.inexact):
        return arr
    return arr.astype(jnp.result_type(float))


class PiecewiseConstant(eqx.Module):
    """A step function that is constant on each of ``K`` segments.

    Attributes:
        knots: Segment edges, sorted ascending, shape ``(K + 1,)``. Includes
            both the start and end of the domain the signal is defined over.
        values: The constant value on each segment, shape ``(K,)`` for a
            scalar signal or ``(K, m)`` for an ``m``-component signal.

    A query time outside ``[knots[0], knots[-1]]`` clamps to the nearest
    edge segment rather than raising or extrapolating.
    """

    knots: jax.Array
    values: jax.Array

    def __init__(self, knots, values):
        """Build a step signal from segment edges and per-segment values."""
        self.knots = _inexact(knots)
        self.values = _inexact(values)

    def __call__(self, t) -> jax.Array:
        """Return the value of the active segment at time ``t``.

        Left-continuous: a ``t`` exactly equal to an interior knot returns
        the value of the segment ending at that knot, not the one starting
        there.
        """
        idx = jnp.clip(
            jnp.searchsorted(self.knots, t, side="left") - 1,
            0,
            self.values.shape[0] - 1,
        )
        return self.values[idx]

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the learnable-leaf path of this signal (``"values"``).

        The knots stay runtime arrays (shifting them never recompiles) but are
        structural breakpoints, not fit coefficients: a knot has zero gradient
        almost everywhere, so selecting a signal by name for a fit picks only
        its per-segment values.
        """
        return ("values",)


class PiecewiseLinear(eqx.Module):
    """A continuous signal that interpolates linearly between knot values.

    Attributes:
        knots: Knot locations, sorted ascending, shape ``(K + 1,)``.
        values: The signal value at each knot, shape ``(K + 1,)`` for a
            scalar signal or ``(K + 1, m)`` for an ``m``-component signal.

    A query time outside ``[knots[0], knots[-1]]`` clamps to the boundary
    value rather than extrapolating.
    """

    knots: jax.Array
    values: jax.Array

    def __init__(self, knots, values):
        """Build a piecewise-linear signal from knot locations and values."""
        self.knots = _inexact(knots)
        self.values = _inexact(values)

    def __call__(self, t) -> jax.Array:
        """Return the linearly-interpolated value at time ``t``."""
        n_segments = self.knots.shape[0] - 1
        idx = jnp.clip(
            jnp.searchsorted(self.knots, t, side="right") - 1,
            0,
            n_segments - 1,
        )
        t0 = self.knots[idx]
        t1 = self.knots[idx + 1]
        frac = jnp.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        v0 = self.values[idx]
        v1 = self.values[idx + 1]
        if self.values.ndim > 1:
            frac = frac[..., None]
        return v0 + frac * (v1 - v0)

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the learnable-leaf path of this signal (``"values"``).

        The knots stay runtime arrays but are structural breakpoints, not fit
        coefficients; selecting the signal by name picks only its knot values.
        """
        return ("values",)


class CubicHermite(eqx.Module):
    """A smooth signal built from per-segment cubic Hermite interpolants.

    Each segment ``[knots[k], knots[k + 1]]`` is a cubic polynomial matched
    to the value and slope supplied at both of its endpoints, so the signal
    is continuous everywhere; it is also differentiable within a segment,
    with the derivative free to change from one segment to the next only if
    the supplied slopes themselves differ across the shared knot.

    Attributes:
        knots: Knot locations, sorted ascending, shape ``(K + 1,)``.
        values: The signal value at each knot, shape ``(K + 1,)`` or
            ``(K + 1, m)``.
        slopes: The signal's time-derivative at each knot, same shape as
            ``values``.

    Uses the same left-continuous segment-selection convention as
    [PiecewiseConstant][diff_soup.inference.approx.signals.PiecewiseConstant]:
    a query exactly on an interior knot is evaluated as the end of the left
    segment.
    """

    knots: jax.Array
    values: jax.Array
    slopes: jax.Array

    def __init__(self, knots, values, slopes):
        """Build a cubic Hermite signal from knots, values, and slopes."""
        self.knots = _inexact(knots)
        self.values = _inexact(values)
        self.slopes = _inexact(slopes)

    def __call__(self, t) -> jax.Array:
        """Return the interpolated value at time ``t``."""
        n_segments = self.knots.shape[0] - 1
        idx = jnp.clip(
            jnp.searchsorted(self.knots, t, side="left") - 1,
            0,
            n_segments - 1,
        )
        t0 = self.knots[idx]
        t1 = self.knots[idx + 1]
        dt = t1 - t0
        s = jnp.clip((t - t0) / dt, 0.0, 1.0)

        v0 = self.values[idx]
        v1 = self.values[idx + 1]
        m0 = self.slopes[idx]
        m1 = self.slopes[idx + 1]

        if self.values.ndim > 1:
            s_b = s[..., None]
            dt_b = dt[..., None]
        else:
            s_b = s
            dt_b = dt

        s2 = s_b * s_b
        s3 = s2 * s_b
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s_b
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        return h00 * v0 + h10 * dt_b * m0 + h01 * v1 + h11 * dt_b * m1

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the learnable-leaf paths (``"values"``, ``"slopes"``).

        The knots stay runtime arrays but are structural breakpoints, not fit
        coefficients; selecting the signal by name picks the knot values and
        slopes.
        """
        return ("values", "slopes")


class Fourier(eqx.Module):
    """A truncated Fourier series, periodic by construction.

    ``value(t) = mean + sum_k cos_coeffs[k] * cos(2*pi*(k+1)*t/period)
                      + sin_coeffs[k] * sin(2*pi*(k+1)*t/period)``

    Attributes:
        period: The signal's period.
        cos_coeffs: Cosine-mode amplitudes, shape ``(n_modes,)`` for a
            scalar signal or ``(n_modes, m)`` for an ``m``-component signal.
        sin_coeffs: Sine-mode amplitudes, same shape as ``cos_coeffs``.
        mean: The signal's time-average (scalar, or shape ``(m,)``).
        knots: Always an empty array —
            [Fourier][diff_soup.inference.approx.signals.Fourier] has no
            segment structure. Present only so all four bases share the same
            ``.knots`` attribute.
    """

    period: jax.Array
    cos_coeffs: jax.Array
    sin_coeffs: jax.Array
    mean: jax.Array
    knots: jax.Array

    def __init__(self, period, cos_coeffs, sin_coeffs, mean):
        """Build a Fourier signal from a period, mode coefficients, and mean."""
        self.period = _inexact(period)
        self.cos_coeffs = _inexact(cos_coeffs)
        self.sin_coeffs = _inexact(sin_coeffs)
        self.mean = _inexact(mean)
        self.knots = jnp.zeros((0,), dtype=self.cos_coeffs.dtype)

    def __call__(self, t) -> jax.Array:
        """Return the Fourier-series value at time ``t``."""
        n_modes = self.cos_coeffs.shape[0]
        k = jnp.arange(1, n_modes + 1, dtype=self.cos_coeffs.dtype)
        angle = 2.0 * jnp.pi * k * t / self.period
        if self.cos_coeffs.ndim > 1:
            angle = angle[..., None]
        terms = self.cos_coeffs * jnp.cos(angle) + self.sin_coeffs * jnp.sin(angle)
        return self.mean + jnp.sum(terms, axis=0)

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the learnable-leaf paths (``"cos_coeffs"``, ``"sin_coeffs"``, ``"mean"``).

        The period stays a runtime array but is the basis definition, not a
        fit coefficient; selecting the signal by name picks the harmonic
        coefficients and the mean.
        """
        return ("cos_coeffs", "sin_coeffs", "mean")
