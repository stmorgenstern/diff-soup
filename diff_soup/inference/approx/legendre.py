"""legendre.py — Legendre polynomial function approximation.

Provides [LegendreExpansion][diff_soup.inference.approx.legendre.LegendreExpansion], the core generic building block for
Legendre polynomial expansions.  Users compose their own physics-specific
functions from this base::

    expansion = LegendreExpansion(params)
    # Chemical potential: mu(c) = log(c/(1-c)) + expansion(2c-1)
    mu = lambda c: jnp.log(c/(1-c)) + expansion(2*c - 1)
    # Positive function: D(c) = exp(expansion(2c-1))
    D = lambda c: jnp.exp(expansion(2*c - 1))

The coefficients are ``jax.Array`` leaves and thus learnable via
``eqx.filter_grad``.

Convention: concentration inputs c ∈ [0, 1] are mapped to x = 2c − 1 ∈ [−1, 1]
before evaluating the Legendre basis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class LegendreExpansion(eqx.Module):
    """Legendre polynomial expansion evaluated at x ∈ [−1, 1].

    Uses the three-term recurrence for numerical stability.

    Attributes:
        params: Coefficients (a_0, a_1, …, a_n), shape ``(degree+1,)``.
            These are JAX array leaves and are the learnable parameters.
    """

    params: jax.Array  # (degree+1,)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate Σ_k params[k] * P_k(x) for x ∈ [−1, 1]."""
        n = len(self.params)
        result = self.params[0] * jnp.ones_like(x)
        if n >= 2:
            result = result + self.params[1] * x
        p_prev = jnp.ones_like(x)
        p_curr = x
        for k in range(2, n):
            p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
            result = result + self.params[k] * p_next
            p_prev, p_curr = p_curr, p_next
        return result


class Legendre(eqx.Module):
    """Named function approximator on a physical input interval.

    ``LegendreExpansion`` expects callers to provide the already-scaled
    coordinate ``x in [-1, 1]``.  ``Legendre`` is the user-facing wrapper for
    spec-authored learnable functions: it owns the coefficients, maps
    physical inputs onto ``[-1, 1]``, and optionally applies a transform to
    the raw expansion value.

    The transform signature is ``transform(c, p)`` where ``p`` is the raw
    Legendre expansion.  For example:

    ``mu_h = Legendre(theta, transform=lambda c, p: log(c/(1-c)) + p)``
    ``D = Legendre(theta, transform=lambda c, p: c*(1-c)*exp(p))``
    """

    coeffs: jax.Array
    interval: tuple[float, float] = eqx.field(static=True, default=(0.0, 1.0))
    transform: Optional[Callable] = eqx.field(static=True, default=None)
    clip: Optional[tuple[float, float]] = eqx.field(static=True, default=None)

    def __call__(self, c: jax.Array) -> jax.Array:
        """Map ``c`` onto ``[-1, 1]``, evaluate the expansion, apply the transform."""
        c_eval = jnp.asarray(c)
        if self.clip is not None:
            c_eval = jnp.clip(c_eval, self.clip[0], self.clip[1])
        lo, hi = self.interval
        x = 2.0 * (c_eval - lo) / (hi - lo) - 1.0
        raw = LegendreExpansion(self.coeffs)(x)
        if self.transform is None:
            return raw
        return self.transform(c_eval, raw)

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the learnable-leaf path(s) of this approximator (``"coeffs"``)."""
        return ("coeffs",)

    def gram(self, *, quadrature: Optional[int] = None) -> jax.Array:
        """Gram matrix of the scaled Legendre basis over this interval.

        Returns ``G`` with ``G[i, j] = ∫ φ_i(c) φ_j(c) dc`` integrated over
        ``interval``, where ``φ_k(c) = P_k(x(c))`` is the k-th Legendre
        polynomial composed with the affine map onto ``[-1, 1]`` used by
        ``__call__``.  The Gram matrix describes the raw basis functions;
        any ``transform`` or ``clip`` on this approximator plays no role.

        By default the exact closed form is used: Legendre orthogonality on
        the mapped interval gives a diagonal matrix with entries
        ``(hi - lo) / (2k + 1)``.  Passing ``quadrature=n`` computes the same
        integrals with n-point Gauss-Legendre quadrature instead (exact for
        polynomial integrands up to degree ``2n - 1``); use it to cross-check
        the closed form or as a template for weighted metrics.

        Args:
            quadrature: Number of Gauss-Legendre points, or ``None`` for the
                exact closed form.

        Returns:
            ``jax.Array`` of shape ``(n_coeffs, n_coeffs)``, symmetric
            positive definite.
        """
        lo, hi = self.interval
        n = int(self.coeffs.shape[0])
        if quadrature is None:
            k = np.arange(n)
            gram = np.diag((hi - lo) / (2.0 * k + 1.0))
        else:
            n_quad = int(quadrature)
            if n_quad < 1:
                raise ValueError(
                    f"quadrature must be a positive number of Gauss-Legendre "
                    f"points; got {quadrature!r}."
                )
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            vander = np.polynomial.legendre.legvander(nodes, n - 1)
            gram = 0.5 * (hi - lo) * (vander.T * weights) @ vander
        return jnp.asarray(gram)

    def constraint_rows(
        self,
        kind: str,
        *,
        at,
        equals,
        order: int = 1,
    ) -> tuple[jax.Array, jax.Array]:
        """Linear equality rows in coefficient space, as an ``(A, b)`` pair.

        Each row ``a`` satisfies ``a @ coeffs == b`` exactly when the raw
        Legendre expansion (before any ``transform``/``clip``) meets the
        requested condition; transforms are nonlinear and are not
        represented.

        Supported kinds:

        * ``"value"`` — the expansion evaluated at the physical point(s)
          ``at`` equals ``equals``; each row is
          ``[φ_0(c), …, φ_K(c)]``.
        * ``"derivative"`` — the ``order``-th derivative with respect to the
          physical coordinate at ``at`` equals ``equals`` (the affine map onto
          ``[-1, 1]`` is accounted for by the chain rule).

        ``at`` may be a scalar or a 1-D array of points; ``equals`` may be a
        scalar (broadcast over the points) or an array matching ``at``.
        Multiple points produce stacked rows.

        Args:
            kind: ``"value"`` or ``"derivative"``.
            at: Physical evaluation point(s) in the class's interval
                coordinates.
            equals: Target value(s) for the right-hand side ``b``.
            order: Derivative order (``kind="derivative"`` only; ignored for
                ``kind="value"``).

        Returns:
            ``(A, b)`` with ``A`` of shape ``(n_points, n_coeffs)`` and ``b``
            of shape ``(n_points,)``, both ``jax.Array``.
        """
        if kind not in ("value", "derivative"):
            raise ValueError(
                f"unknown constraint kind {kind!r}; valid kinds are 'value' and 'derivative'."
            )
        lo, hi = self.interval
        n = int(self.coeffs.shape[0])
        at_arr = np.atleast_1d(np.asarray(at, dtype=float))
        if at_arr.ndim != 1:
            raise ValueError(
                f"`at` must be a scalar or a 1-D array of points; got an "
                f"array of shape {at_arr.shape}."
            )
        try:
            b = np.broadcast_to(np.asarray(equals, dtype=float), at_arr.shape)
        except ValueError as exc:
            raise ValueError(
                f"`equals` must be a scalar or match the shape of `at`; got "
                f"`equals` of shape {np.shape(equals)} for {at_arr.shape[0]} "
                f"point(s)."
            ) from exc
        x = 2.0 * (at_arr - lo) / (hi - lo) - 1.0
        if kind == "value":
            rows = np.polynomial.legendre.legvander(x, n - 1)
        else:
            order = int(order)
            if order < 1:
                raise ValueError(
                    f"derivative constraint order must be a positive integer; got {order}."
                )
            # Column k holds the Legendre coefficients of d^order P_k / dx^order.
            der_coeffs = np.polynomial.legendre.legder(np.eye(n), m=order, axis=0)
            # legval maps coefficient columns to values: shape (n, n_points).
            values = np.atleast_2d(np.polynomial.legendre.legval(x, der_coeffs))
            rows = ((2.0 / (hi - lo)) ** order) * values.T
        return jnp.asarray(rows), jnp.asarray(np.array(b, dtype=float))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def mono_to_legendre(
    mono_coeffs: np.ndarray,
    n_leg_terms: int,
    *,
    offset: float = 0.0,
) -> np.ndarray:
    """Convert monomial coefficients to Legendre basis via least-squares fit.

    Evaluates the monomial polynomial on a dense grid in [0.01, 0.99],
    maps to x = 2c − 1 ∈ [−1, 1], and fits a Legendre expansion of
    the given degree.

    Args:
        mono_coeffs: Monomial coefficients ``[a_0, a_1, …, a_n]`` for
            ``Σ a_k c^k`` (array-like, shape ``(n+1,)``).
        n_leg_terms: Number of Legendre terms to fit (degree = n_leg_terms).
        offset: Additive constant (e.g. ``log(4)`` for exchange current
            convention).

    Returns:
        Legendre coefficients.
    """
    from numpy.polynomial.legendre import legfit

    mono_coeffs = np.asarray(mono_coeffs)
    c = np.linspace(0.01, 0.99, 2000)
    x = 2.0 * c - 1.0
    v = sum(float(a) * c**i for i, a in enumerate(mono_coeffs))
    if offset != 0.0:
        v = v + offset
    return legfit(x, v, deg=n_leg_terms)
