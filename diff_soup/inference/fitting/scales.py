"""scales.py — Parameter-scaling strategy objects for damped Gauss-Newton.

Each concrete subclass computes the positive diagonal used to scale the
Levenberg-Marquardt normal equations.  Pass an instance (e.g.
``JacScale()``) to a solver; do not pass a string flag.

Protocols
---------
``damping_diagonal(J, x) -> positive 1-D array of length n = x.shape[0]``
    Returns a strictly positive diagonal whose entries reflect the current
    estimate of appropriate curvature scaling for each parameter.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp


class XScale(eqx.Module):
    """Base class for the parameter scaling of the optimizers' damping and trust regions.

    A subclass implements `damping_diagonal`. Levenberg-Marquardt damps by that diagonal, and
    the trust-region solvers measure steps in its square root.
    """

    @abstractmethod
    def damping_diagonal(self, J: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return the damping diagonal, one positive entry per parameter.

        Args:
            J: The residual Jacobian, or the curvature matrix of a trust-region solver.
            x: The current parameters.

        Returns:
            A 1-D array with the length of `x`.
        """

    def damping_from_gram(self, JtJ: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return the damping diagonal from the Gauss-Newton matrix ``J^T J``.

        A solver that reduces ``J^T J`` across a device mesh never materializes the
        full Jacobian ``J``, so it scales the normal equations from the Gram matrix
        directly.  The default reproduces the Jacobian-column squared-norm scaling
        (``max(diag(J^T J), 1e-10)``); override it for a strategy that does not
        derive from the Gram diagonal.

        Args:
            JtJ: The Gauss-Newton matrix of shape ``(n, n)``.
            x: Current parameter vector of shape ``(n,)``.

        Returns:
            A strictly positive 1-D array of length ``n``.
        """
        return jnp.maximum(jnp.diag(JtJ), 1e-10)


class JacScale(XScale):
    """Scales each parameter by the squared norm of its Jacobian column, the default of `LevenbergMarquardt`.

    The trust-region solvers refuse it with `hessian="true"`.
    """

    def damping_diagonal(self, J: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``max(diag(J^T J), 1e-10)`` element-wise.

        Args:
            J: Jacobian matrix of shape ``(m, n)``.
            x: Current parameter vector of shape ``(n,)`` (used for shape only).

        Returns:
            Positive 1-D array of length ``n``.
        """
        return jnp.maximum(jnp.diag(J.T @ J), 1e-10)


class HessianScale(XScale):
    """Scales each parameter by the magnitude of its curvature diagonal, the default of the trust-region solvers.

    `NewtonTrustRegion` refuses it with a CG subproblem.
    """

    def damping_diagonal(self, H: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``max(abs(diag(H)), eps)`` element-wise.

        Args:
            H: Symmetric Hessian matrix of shape ``(n, n)``.
            x: Current parameter vector of shape ``(n,)`` (used for shape only).

        Returns:
            Positive 1-D array of length ``n``.
        """
        del x
        diagonal = jnp.diag(H)
        return jnp.maximum(jnp.abs(diagonal), jnp.finfo(diagonal.dtype).eps)


class ConstantScale(XScale):
    """Scales each parameter by a fixed typical magnitude, with the damping diagonal `1 / scale**2`.

    Args:
        scale: The typical magnitude of the parameters, one scalar or one entry per parameter.
    """

    scale: Any

    def damping_diagonal(self, J: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``1 / scale**2`` broadcast to length ``x.shape[0]``.

        Args:
            J: Jacobian matrix of shape ``(m, n)`` (unused; present for
                protocol compatibility).
            x: Current parameter vector of shape ``(n,)``; determines the
                output length.

        Returns:
            Positive 1-D array of length ``n``.
        """
        d = 1.0 / jnp.asarray(self.scale) ** 2
        return jnp.broadcast_to(d, (x.shape[0],))

    def damping_from_gram(self, JtJ: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``1 / scale**2`` broadcast to length ``x.shape[0]`` (ignores ``JtJ``)."""
        d = 1.0 / jnp.asarray(self.scale) ** 2
        return jnp.broadcast_to(d, (x.shape[0],))
