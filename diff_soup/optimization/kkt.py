"""The solve of one primal-dual KKT system, behind an interface its factorization can change.

The interior point and SQP steps solve

    [ W + Sigma + delta_w I    A^T       ] [dx  ]   [r_primal]
    [ A                        -delta_c I] [dlam] = [r_dual  ]

The dense backend takes the condensed route: a Cholesky factorization of the primal block,
`delta_w` raised until it succeeds, then the Schur complement on the constraint rows. A banded
backend on cuDSS replaces only this class when multiple shooting or transcription needs it.
"""

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx


class KKTSolve(eqx.Module):
    """One KKT solve, returning the step and whether the inertia came out right."""

    @abc.abstractmethod
    def solve(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        rhs_primal: Any,
        rhs_dual: Any,
    ) -> tuple[Any, Any, Any]:
        """Solve the KKT system at the given regularization.

        Args:
            primal_block: `W + Sigma`, the Lagrangian Hessian plus the barrier diagonal, `(n, n)`.
            constraint_jacobian: `A`, the rows of every active constraint, `(m, n)`.
            delta_w: The primal regularization added to the diagonal.
            delta_c: The dual regularization.
            rhs_primal: The primal right-hand side, `(n,)`.
            rhs_dual: The dual right-hand side, `(m,)`.

        Returns:
            `(dx, dlam, inertia_ok)`: the primal and dual steps, and whether the factorization
            certified the inertia `(n, m, 0)` at this `delta_w`.
        """
        raise NotImplementedError


class DenseKKT(KKTSolve):
    """The condensed dense route: Cholesky of the primal block and the Schur complement.

    Args:
        delta_w_init: The first primal regularization tried when the factorization fails.
        delta_w_max: The largest primal regularization before the step is declared failed.
        delta_w_factor: The factor the regularization grows by.
    """

    delta_w_init: float = eqx.field(static=True, default=1e-4)
    delta_w_max: float = eqx.field(static=True, default=1e40)
    delta_w_factor: float = eqx.field(static=True, default=8.0)

    def solve(
        self,
        primal_block: Any,
        constraint_jacobian: Any,
        delta_w: Any,
        delta_c: Any,
        rhs_primal: Any,
        rhs_dual: Any,
    ) -> tuple[Any, Any, Any]:
        """Solve by the condensed route; see [KKTSolve.solve][diff_soup.optimization.KKTSolve.solve]."""
        raise NotImplementedError
