"""The solver view surface a retained consistent-initialization solve consumes.

A differential-algebraic solve seeds its history from a state on the algebraic
manifold. The consistent-initialization kernels
([diff_soup.timestepping.consistent_ic][]) compute that state from the
residual, exact iteration-matrix plan, and startup slope.
[ConsistentInitProblem][diff_soup.problem.consistent_init.ConsistentInitProblem] documents that retained kernel boundary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConsistentInitProblem(Protocol):
    """The members every retained consistent-initialization solve reads.

    The state topology is the shared plan's state block; ``None`` selects a
    dense block. Numerical operations never fall back to authored ``mass`` /
    ``f`` or a raw RHS pencil.

    Attributes:
        n_u: The state length (the differential plus local-algebraic rows).
        n_lam: The bordered-global length (``0`` with no global unknowns).
        id_mask: The differential/algebraic indicator over the augmented state; a
            positive entry marks a differential row, a zero entry an algebraic
            row.
    """

    n_u: int
    n_lam: int
    id_mask: Any
    state_topology: Any

    def residual(self, t: Any, y: Any, ydot: Any, theta: Any, control: Any = None) -> Any:
        """Return the residual over the augmented state."""
        ...

    def state_jacobian_blocks(
        self, t: Any, y: Any, ydot: Any, theta: Any, cj: Any, **kwargs: Any
    ) -> Any:
        """Return the ``F_y + cj F_ydot`` Schur blocks."""
        ...

    def startup_slope(self, t: Any, y: Any, theta: Any, **kwargs: Any) -> Any:
        """Return the flow slope at ``(t, y)``."""
        ...
