"""regularization.py — Composable regularization for PDE parameter estimation.

Regularization modifies the objective function to prevent overfitting and
improve conditioning.  It is part of the **problem** (not the solver),
because it changes what is being optimized.

For least-squares problems, Tikhonov regularization appends extra rows
to the residual vector: ``r_aug = [r; sqrt(λ·w) * θ]``, which naturally
adds ``λ·w·θ²`` to the Gauss-Newton normal equations.

For scalar minimization, it adds ``0.5 * λ * Σ w_k θ_k²`` to the loss.

``penalty`` / ``penalty_grad`` are JAX-native (``jnp``), so a traced/JIT'd
input does not raise ``ConcretizationTypeError`` and the parameter dtype
stream is preserved.  The host-side weight builder ``legendre_weights`` stays
NumPy (it is a compile-time config array).

The protocol every regularizer follows is:

- ``residual(x)`` — returns a 1-D JAX array of extra least-squares rows
  (length 0 for no regularization).
- ``penalty(x)`` — returns ``0.5 * ||residual(x)||²`` as a scalar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np


def legendre_weights(n_params: int) -> np.ndarray:
    """Default regularisation weights: ``1 + k(k+1)`` per Legendre degree.

    The returned vector has exactly ``n_params`` entries. The weight for
    coefficient *k* is ``1 + k(k+1)``; the base weight of 1 prevents the
    zeroth coefficient from running away. Separate expansion blocks should
    use separate regularizer blocks when each block should restart at degree
    zero.
    """
    return np.array([1.0 + k * (k + 1) for k in range(int(n_params))], dtype=np.float64)


class Regularizer(ABC):
    """Composable regularization for least-squares parameter estimation.

    All regularizers implement ``residual(x)``, which returns the extra
    least-squares rows appended to the data residual before solving.
    The default ``penalty(x)`` is ``0.5 * ||residual(x)||²``.

    Subclasses must implement [residual][diff_soup.inference.fitting.regularization.Regularizer.residual].  They may override
    [penalty][diff_soup.inference.fitting.regularization.Regularizer.penalty] when a cheaper closed-form expression exists, provided
    the result is numerically identical to the default.
    """

    @abstractmethod
    def residual(self, x):
        """Return the extra regularization rows as a 1-D JAX array.

        The returned array has length equal to the number of regularization
        constraints; an unconstrained regularizer returns ``jnp.zeros((0,))``.

        Args:
            x: Parameter vector (JAX array).

        Returns:
            1-D JAX array of extra residual rows.
        """
        ...

    def penalty(self, x):
        """Scalar regularization penalty ``0.5 * ||residual(x)||²``.

        Args:
            x: Parameter vector (JAX array).

        Returns:
            Scalar JAX array equal to ``0.5 * jnp.sum(residual(x) ** 2)``.
        """
        r = self.residual(x)
        return 0.5 * jnp.sum(r**2)


class NoReg(Regularizer):
    """No penalty, used by the solvers when a problem has no regularization."""

    lambda_: float = 0.0

    def residual(self, x):
        """Return an empty 1-D array — no regularization rows are added.

        Args:
            x: Parameter vector (JAX array).

        Returns:
            Zero-length JAX array with the same dtype as ``x``.
        """
        return jnp.zeros((0,), dtype=jnp.result_type(x))

    def augment_residual(self, r: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``r`` unchanged (no extra rows)."""
        return r

    def augment_jacobian(self, J: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``J`` unchanged (no extra rows)."""
        return J

    def penalty(self, x):
        """Return ``0.0`` — no penalty is applied (JAX-native, JIT-safe)."""
        return jnp.zeros((), dtype=jnp.result_type(x))

    def penalty_grad(self, x):
        """Return a zero gradient matching ``x``'s shape (JAX-native)."""
        return jnp.zeros_like(x)


@dataclass(frozen=True)
class BlockRegularization(Regularizer):
    """Composable block-structured regularization for hierarchical inverse problems.

    Applies separate Tikhonov-style penalties to different parameter blocks.
    Each block is identified by a slice or index array into the flat parameter
    vector, with its own weight and per-element weights.

    Attributes:
        blocks: List of ``(indices, lambda_, weights)`` tuples.  Each block
            is a tuple of: ``indices`` (slice, ``np.ndarray`` of int, or
            ``(start, stop)`` tuple), ``lambda_`` (regularization strength
            for this block), and ``weights`` (per-element weights as
            ``np.ndarray`` or ``None`` for uniform).

    Example::

        reg = BlockRegularization([
            (slice(0, 5),  0.01, legendre_weights(5)),    # global μ_h coeffs
            (slice(5, 11), 0.01, legendre_weights(6)),    # global j₀ coeffs
            (slice(11, 31), 1.0, None),                    # heterogeneity Z
        ])
    """

    blocks: list  # list of (indices, lambda_, weights)

    def _iter_blocks(self, x):
        """Yield ``(x_block, lambda_, weights)`` for each configured block."""
        for indices, lam, w in self.blocks:
            if isinstance(indices, tuple) and len(indices) == 2:
                indices = slice(indices[0], indices[1])
            xi = x[indices]
            wi = w if w is not None else np.ones(len(xi))
            yield xi, lam, wi

    def residual(self, x):
        """Concatenate ``sqrt(lambda_ * w) * x_block`` rows for all blocks.

        Returns only the regularization rows as a 1-D array.  An empty block
        list returns a length-0 array.

        Args:
            x: Parameter vector (JAX array).

        Returns:
            1-D JAX array containing the stacked regularization rows.
        """
        dt = jnp.result_type(x)
        parts = []
        for xi, lam, wi in self._iter_blocks(x):
            sqrt_w = jnp.asarray(np.sqrt(lam * np.asarray(wi)), dtype=dt)
            parts.append(sqrt_w * xi)
        if not parts:
            return jnp.zeros((0,), dtype=dt)
        return jnp.concatenate(parts)

    def block_rows(self, x) -> tuple:
        """Column-local ``(columns, residual, jacobian)`` triples, one per block.

        A fit over a joint problem adds each block's Gauss-Newton statistics
        to the owner or global tier that holds its columns.
        """
        x = jnp.asarray(x)
        dt = jnp.result_type(x)
        rows = []
        for indices, lam, w in self.blocks:
            if isinstance(indices, tuple) and len(indices) == 2:
                indices = slice(indices[0], indices[1])
            columns = np.arange(x.shape[0])[indices]
            weights = np.ones(columns.size) if w is None else np.asarray(w, dtype=float)
            sqrt_w = jnp.asarray(np.sqrt(lam * weights), dtype=dt)
            rows.append(
                (
                    tuple(int(column) for column in columns),
                    sqrt_w * x[columns],
                    jnp.diag(sqrt_w),
                )
            )
        return tuple(rows)

    def augment_residual(self, r: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Append ``sqrt(lambda_ * w) * x_block`` rows per block to ``r``."""
        parts = [r]
        for xi, lam, wi in self._iter_blocks(x):
            parts.append(np.sqrt(lam * wi) * xi)
        return np.concatenate(parts)

    def augment_jacobian(self, J: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Append per-block diagonal-ish rows aligning with each block's slice."""
        parts = [J]
        n = len(x)
        for indices, lam, w in self.blocks:
            if isinstance(indices, tuple) and len(indices) == 2:
                indices = slice(indices[0], indices[1])
            xi = x[indices]
            wi = w if w is not None else np.ones(len(xi))
            block = np.zeros((len(xi), n))
            idx = np.arange(n)[indices]
            for i, j in enumerate(idx):
                block[i, j] = np.sqrt(lam * wi[i])
            parts.append(block)
        return np.concatenate(parts, axis=0)

    def penalty(self, x):
        """Scalar penalty ``0.5 * Σ_block lambda_ * Σ w_k x_k²`` (JAX-native).

        Weights are cast to ``x``'s dtype so an fp32 param stream is not
        silently promoted to fp64.
        """
        dt = jnp.result_type(x)
        total = jnp.zeros((), dtype=dt)
        for xi, lam, wi in self._iter_blocks(x):
            total = total + 0.5 * lam * jnp.sum(jnp.asarray(wi, dtype=dt) * xi**2)
        return total

    def penalty_grad(self, x):
        """Gradient of `penalty`: ``lambda_ * w_k * x_k`` on each block."""
        dt = jnp.result_type(x)
        g = jnp.zeros_like(x)
        for indices, lam, w in self.blocks:
            if isinstance(indices, tuple) and len(indices) == 2:
                indices = slice(indices[0], indices[1])
            xi = x[indices]
            wi = w if w is not None else np.ones(len(xi))
            g = g.at[indices].set(lam * jnp.asarray(wi, dtype=dt) * xi)
        return g


@dataclass(frozen=True)
class Tikhonov(Regularizer):
    """Adds the penalty `0.5 * lambda_ * sum(weights * x**2)` on the fitted values to the objective.

    Pass it as `regularization=` of an `InferenceProblem`.

    Args:
        lambda_: The strength of the penalty.
        weights: One weight per penalized value, or `None` to weight the `k`-th value by `1 + k(k+1)`, one for a single parameter.
        on: A handle whose values alone are penalized, or `None` to penalize every fitted parameter.
    """

    lambda_: float = 0.01
    weights: Optional[np.ndarray] = None
    on: Any = None

    def _needs_columns(self) -> None:
        """Refuse to evaluate rows for a handle whose columns were never resolved."""
        if self.on is not None:
            raise ValueError(
                "Tikhonov(on=handle) regularizes the handle's columns and needs the "
                "problem's parameter layout to find them: pass it as "
                "InferenceProblem(regularization=...) instead of evaluating its rows directly."
            )

    def on_columns(self, blocks: Sequence[tuple]) -> "BlockRegularization":
        """The same strength and weights applied to explicit column blocks.

        Args:
            blocks: Column index tuples, one per occurrence of the handle.

        Returns:
            A [BlockRegularization][diff_soup.inference.fitting.regularization.BlockRegularization]
            with one block per entry.
        """
        resolved = []
        for columns in blocks:
            columns = np.asarray(columns, dtype=int)
            weights = self.weights if self.weights is not None else legendre_weights(columns.size)
            if np.asarray(weights).shape != (columns.size,):
                raise ValueError(
                    f"Tikhonov weights have shape {np.asarray(weights).shape}; the handle "
                    f"occupies {columns.size} columns."
                )
            resolved.append((columns, float(self.lambda_), np.asarray(weights, dtype=float)))
        return BlockRegularization(resolved)

    def _sqrt_w(self, n_params: int) -> np.ndarray:
        """Return ``sqrt(lambda_ * weights)``, defaulting to Legendre weights."""
        w = self.weights if self.weights is not None else legendre_weights(n_params)
        return np.sqrt(self.lambda_ * w)

    def residual(self, x):
        """Return ``sqrt(lambda_ * w) * x`` as a 1-D JAX array.

        The returned array has the same length as ``x`` and the same dtype.
        Appending this to the data residual reproduces Tikhonov regularization
        in a Gauss-Newton / Levenberg-Marquardt solve.

        Args:
            x: Parameter vector (JAX array).

        Returns:
            1-D JAX array of shape ``(len(x),)``.
        """
        self._needs_columns()
        n = x.shape[0] if hasattr(x, "shape") and x.shape else len(x)
        w = self.weights if self.weights is not None else legendre_weights(n)
        sqrt_w = jnp.asarray(np.sqrt(self.lambda_ * np.asarray(w)), dtype=jnp.result_type(x))
        return sqrt_w * x

    def augment_residual(self, r: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Append ``sqrt(lambda_ * w) * x`` rows to the residual vector."""
        sw = self._sqrt_w(len(x))
        return np.concatenate([r, sw * x])

    def augment_jacobian(self, J: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Append ``diag(sqrt(lambda_ * w))`` rows to the Jacobian."""
        sw = self._sqrt_w(len(x))
        return np.concatenate([J, np.diag(sw)], axis=0)

    def penalty(self, x):
        """Scalar Tikhonov penalty ``0.5 * lambda_ * Σ w_k x_k²`` (JAX-native).

        Weights are cast to ``x``'s dtype so an fp32 param stream is not
        silently promoted to fp64.
        """
        dt = jnp.result_type(x)
        n = x.shape[0] if hasattr(x, "shape") and x.shape else len(x)
        w = self.weights if self.weights is not None else legendre_weights(n)
        return 0.5 * self.lambda_ * jnp.sum(jnp.asarray(w, dtype=dt) * x**2)

    def penalty_grad(self, x):
        """Gradient of `penalty`: ``lambda_ * w_k * x_k`` (JAX-native)."""
        dt = jnp.result_type(x)
        n = x.shape[0] if hasattr(x, "shape") and x.shape else len(x)
        w = self.weights if self.weights is not None else legendre_weights(n)
        return self.lambda_ * jnp.asarray(w, dtype=dt) * x
