"""Consistent initialization on the full implicit residual, with exact derivatives.

Given pinned differential values ``y_d`` at ``t0``, the solver finds the
algebraic values ``y_a`` and differential slopes ``ydot_d`` — with ``ydot_a``
structurally zero — as the joint root of the full residual
``F(t0, y, ydot, theta, control) = 0`` in the unknowns ``(y_a, ydot_d)``.
The joint unknown vector is laid out state-aligned: slot ``s`` holds ``y[s]``
when variable ``s`` is algebraic and ``ydot[s]`` when it is differential, so
the exact Jacobian in the unknowns is the column blend

    K[:, s] = F_y[:, s]  (s algebraic)   |   K[:, s] = F_ydot[:, s]  (s differential),

which drives both the Newton iteration and the implicit-function derivatives
at the converged root. The state-aligned layout keeps the residual's own
structural diagonal on ``K``'s diagonal, which static-pivoting sparse
factorizations rely on for accuracy. Working on the full residual handles
derivative couplings in any equation row. Supplied algebraic values and
slope seeds are initial guesses only: they may accelerate the solve but
never define it, and they carry no derivative. Pinned differential values,
``theta``, and ``control`` differentiate through the full residual.

``K`` is solved through a linear backend selected by size and platform (the
``backend`` argument): tiny systems use a dense LU, larger ones keep ``K``
as sparse values over its static column-selected topology and factorize
through a sparse-direct backend (KLU on CPU, cuDSS on GPU) in both the
primal orientation and, for a reverse sweep, the transposed one — the sparse
path never materializes an ``(n, n)`` matrix. The resolved route is recorded
on the plan as ``backend_name``.

The modules: ``plan`` (the route, the backend and the index maps), ``newton``
(the solve and its gate), ``rules`` (the implicit-function derivatives),
``solve`` (the public entries). The Euler-Newton initialization that
[JIDA][diff_soup.JIDA] runs for ``ic_mode="ya_ydp_euler"`` and
``ic_mode="y_fixed_ydot"`` is in ``euler_newton``, over the records and pinned
operators of ``pinned`` and the implicit-function boundary of ``pinned_root``.
"""

from diff_soup.problem.implicit.consistent_init.plan import (
    ConsistentInitPlan,
    build_consistent_init_plan,
    initialization_route,
    reduced_route_eligible,
    resolve_backend_route,
)
from diff_soup.problem.implicit.consistent_init.rules import (
    consistent_ic_root,
    consistent_ic_root_reduced,
)
from diff_soup.problem.implicit.consistent_init.solve import (
    ConsistentInitResult,
    check_consistent_init,
    consistent_initial_state,
)

__all__ = [
    "ConsistentInitPlan",
    "ConsistentInitResult",
    "build_consistent_init_plan",
    "check_consistent_init",
    "consistent_ic_root",
    "consistent_ic_root_reduced",
    "consistent_initial_state",
    "initialization_route",
    "reduced_route_eligible",
    "resolve_backend_route",
]
