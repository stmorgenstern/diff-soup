"""One mode-parameterized entry for every consistent-initialization solve.

A differential-algebraic solve must start from a state on the algebraic
manifold. Which unknowns the projection determines depends on the integrator:
the Euler-Newton kernels solve the algebraic state and the differential
derivative, or all of the state given a fixed derivative.
[consistent_init][diff_soup.timestepping.consistent_ic.api.consistent_init] selects the kernel from an [InitMode][diff_soup.timestepping.consistent_ic.api.InitMode] so every
engine reaches the same entry:

``ALG_AND_YDOT`` / ``ALL_Y_GIVEN_YDOT``
    [euler_newton_ic][diff_soup.timestepping.consistent_ic.euler_newton.euler_newton_ic].
``ADJOINT``
    reserved for a terminal-cotangent surface and rejected by this entry.

The default startup projection — pin the differential rows, solve the local
and global algebraic values — is not served here: it runs the
capability-routed initialization solve on the solver problem
([consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state]).

The dispatch is pure plumbing: each mode's result equals the direct kernel call,
which differentiates through the converged root by the implicit-function rule,
never through the Newton iteration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, NamedTuple

from diff_soup.timestepping.consistent_ic.euler_newton import euler_newton_ic
from diff_soup.timestepping.consistent_ic.pin_differential import (
    ICConfig,
    ICMode,
)


class InitMode(enum.Enum):
    """Which consistent-initialization solve an [ICRequest][diff_soup.timestepping.consistent_ic.api.ICRequest] selects.

    ``ALG_AND_YDOT`` solves the algebraic components of the state and the
    differential components of the derivative; ``ALL_Y_GIVEN_YDOT`` solves all of
    the state given a fixed derivative. ``ADJOINT`` is reserved and is not
    served by [consistent_init][diff_soup.timestepping.consistent_ic.api.consistent_init].
    """

    ALG_AND_YDOT = enum.auto()
    ALL_Y_GIVEN_YDOT = enum.auto()
    ADJOINT = enum.auto()


class ICState(NamedTuple):
    """The seed the solve starts from: state, bordered globals, derivative.

    ``ydot`` seeds the Euler-Newton modes; it defaults to ``None`` for
    callers that supply no derivative seed.
    """

    y: Any
    lam: Any
    ydot: Any = None


@dataclass(frozen=True)
class ICRequest:
    """Mode, time, and tolerances for a consistent-init solve.

    ``cfg`` carries the full [ICConfig][diff_soup.timestepping.consistent_ic.pin_differential.ICConfig] (rtol and atol are distinct there,
    so the request must too). ``t1`` sizes the Euler-Newton time scale. ``t0`` and ``t1`` are typed ``Any`` because an engine
    passes them as traced arrays under ``jax.jit``.
    """

    mode: InitMode
    t0: Any
    cfg: ICConfig
    t1: Any = None


# The two Euler-Newton entry modes map onto the kernel's own subset selector.
_EULER_MODES = {
    InitMode.ALG_AND_YDOT: ICMode.ALG_AND_YDOT,
    InitMode.ALL_Y_GIVEN_YDOT: ICMode.ALL_Y_GIVEN_YDOT,
}


def consistent_init(problem, state, theta, request, backend, *, layout=None, precision=None):
    """Project ``state`` onto the algebraic manifold for ``problem`` at ``request.t0``.

    Dispatches to the kernel named by ``request.mode`` and returns its result
    unchanged (an
    [ICResult][diff_soup.timestepping.consistent_ic.pin_differential.ICResult]
    for every forward mode). ``backend`` is the linear solver the projection
    factors on (an engine passes its resolver-selected sparse backend); ``None``
    defaults each kernel to the dense backend. ``layout`` and ``precision`` are
    the placement and dtype policy the projection's linear analyses are built
    with -- an engine passes the policy the rest of its solve runs under, so the
    projection never factors at the backend's defaults behind it; each left
    ``None`` keeps the backend's own default for it.
    """
    if request.mode in _EULER_MODES:
        return euler_newton_ic(
            problem,
            state.y,
            state.ydot,
            state.lam,
            theta,
            _EULER_MODES[request.mode],
            request.cfg,
            request.t0,
            t1=request.t1,
            backend=backend,
            layout=layout,
            precision=precision,
        )
    if request.mode is InitMode.ADJOINT:
        raise ValueError(
            "adjoint consistent-initialization projects a terminal cotangent onto the "
            "adjoint manifold -- a different input surface (a cotangent plus the terminal "
            "forward state) than the state-seed projection this entry serves. "
            "Native IDA performs this projection through its own AD bridge."
        )
    raise ValueError(f"unknown consistent-initialization mode: {request.mode!r}")
