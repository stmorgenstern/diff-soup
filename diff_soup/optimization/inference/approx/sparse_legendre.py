"""Sparse products of Legendre polynomials on physical input intervals."""

from __future__ import annotations

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class SparseLegendre(eqx.Module):
    """A multivariate Legendre expansion with a sparse degree index set.

    ``truncation="total"`` retains sum(k) <= degree. ``"hyperbolic"``
    retains product(k + 1) <= degree + 1. ``indices`` lists the terms in
    graded lexicographic order, matching ``coeffs``. Each argument is
    normalized from its interval to [-1, 1] and broadcasts with the others.
    An optional ``transform(*inputs, raw)`` is applied after expansion.
    """

    coeffs: jax.Array
    intervals: tuple = eqx.field(static=True)
    indices: tuple = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    truncation: str = eqx.field(static=True)
    transform: Any = eqx.field(static=True)

    def __init__(
        self,
        *,
        intervals,
        degree,
        truncation="total",
        coeffs=None,
        transform=None,
        key=None,
        init_std=0.0,
    ):
        """Choose intervals and degree, optionally supplying coefficients or a key."""
        intervals = tuple(tuple(map(float, interval)) for interval in intervals)
        if not intervals or any(
            len(b) != 2 or not np.isfinite(b).all() or b[0] >= b[1] for b in intervals
        ):
            raise ValueError(
                "SparseLegendre intervals must be nonempty finite (lo, hi) pairs with lo < hi"
            )
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("SparseLegendre degree must be a nonnegative integer")
        if truncation not in ("total", "hyperbolic"):
            raise ValueError("SparseLegendre truncation must be 'total' or 'hyperbolic'")
        candidates = itertools.product(range(degree + 1), repeat=len(intervals))
        indices = tuple(
            sorted(
                (
                    k
                    for k in candidates
                    if (
                        sum(k) <= degree
                        if truncation == "total"
                        else np.prod(np.asarray(k) + 1) <= degree + 1
                    )
                ),
                key=lambda k: (sum(k), k),
            )
        )
        if coeffs is None:
            coeffs = (
                jnp.zeros(len(indices))
                if key is None
                else init_std * jax.random.normal(key, (len(indices),))
            )
        coeffs = jnp.asarray(coeffs, dtype=float)
        if coeffs.shape != (len(indices),):
            raise ValueError(
                f"SparseLegendre coeffs must have shape {(len(indices),)} for its index set; got {coeffs.shape}"
            )
        if transform is not None and not callable(transform):
            raise TypeError("SparseLegendre transform must be callable or None")
        self.coeffs, self.intervals, self.indices = coeffs, intervals, indices
        self.degree, self.truncation, self.transform = degree, truncation, transform

    def __call__(self, *inputs):
        """Evaluate at one broadcast-compatible argument per input dimension."""
        if len(inputs) != len(self.intervals):
            raise ValueError(
                f"SparseLegendre expects {len(self.intervals)} input arguments; got {len(inputs)}"
            )
        values = jnp.broadcast_arrays(*[jnp.asarray(value) for value in inputs])
        tables = []
        for value, (lo, hi) in zip(values, self.intervals):
            x = 2 * (value - lo) / (hi - lo) - 1
            table = [jnp.ones_like(x)]
            if self.degree:
                table.append(x)
            for k in range(2, self.degree + 1):
                table.append(((2 * k - 1) * x * table[-1] - (k - 1) * table[-2]) / k)
            tables.append(table)
        raw = jnp.zeros_like(values[0], dtype=self.coeffs.dtype)
        for coefficient, index in zip(self.coeffs, self.indices):
            term = coefficient
            for table, k in zip(tables, index):
                term = term * table[k]
            raw = raw + term
        return raw if self.transform is None else self.transform(*values, raw)

    def learnable_paths(self) -> tuple[str, ...]:
        """Fit the coefficients while keeping intervals and index set fixed."""
        return ("coeffs",)


def sparse_legendre_weights(basis: SparseLegendre) -> np.ndarray:
    """Return 1 + |k|(|k| + 1) in the basis's graded index order."""
    degree = np.asarray([sum(k) for k in basis.indices])
    return 1.0 + degree * (degree + 1)
