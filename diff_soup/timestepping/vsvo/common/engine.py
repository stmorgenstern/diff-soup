"""The engine the driver runs, and the report of each step it takes."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import equinox as eqx
import jax

from .contract import (
    EngineCapabilities,
)


# ---------------------------------------------------------------------------
# The engine protocol and the driver/engine per-attempt data contract
# ---------------------------------------------------------------------------
@runtime_checkable
class Engine(Protocol):
    """The step kernel the driver runs. Opaque behind three methods + a flag.

    An engine owns one internal step: it builds its own carry (history,
    coefficients, step and order state), advances one attempt, and evaluates its
    dense output on the last accepted interval. The driver never inspects the
    carry; it reads only the `StepReport` each attempt returns.

    Attributes:
        capabilities: The `EngineCapabilities` the driver validates the
            problem against before the loop.

    An engine may also expose a ``trace_kind`` string (``"ndf"`` / ``"bdf"`` /
    ``"ida"``) naming its per-step trace family; the driver reads it only when a
    trace is requested and defaults to ``"ndf"`` when it is absent. An engine
    whose ``init_carry`` consistent-initializes the state may expose
    ``state_at_entry(carry)`` returning the augmented ``[u; lam]`` vector the
    solve starts from; the driver records it as the ``t0`` output sample and
    falls back to the raw input state when the method is absent. An engine
    that exposes ``decision_template()`` (the zero-valued decision inputs of
    one attempt) and accepts ``attempt_step(carry, decision=(replay,
    recorded))`` has its decisions recorded under a checkpointed reverse solve,
    so the backward replay decides as the forward sweep did. Such an engine also
    exposes ``step_at_entry(carry)`` and accepts ``init_carry(..., step=(replay,
    recorded))``, so a replayed event restart enters with the step the forward
    sweep's restart chose instead of searching for it again. An engine whose
    ``attempt_step`` takes a ``live`` keyword receives the loop's condition for
    the lane: under ``vmap`` the loop discards the attempt of a lane where it is
    false, so the engine may leave that lane out of any decision it takes for
    the whole batch, provided the lanes it keeps compute the same values. An
    engine that exposes ``history_at_exit(carry)`` (the record a later solve
    continues from) and accepts ``init_carry(..., history=...)`` (a
    [SolveHistory][diff_soup.timestepping.vsvo.solution.SolveHistory] to
    continue from) serves a solve's ``history=`` request.
    """

    capabilities: EngineCapabilities

    def init_carry(self, problem, t0, t1, y0, ydot0, lam0, theta, cfg):
        """Build the engine's initial per-step carry.

        ``y0``/``ydot0``/``lam0`` are the driver's consistent initial
        differential state, its derivative, and the initial multipliers.
        """
        ...

    def attempt_step(self, carry):
        """Advance one internal step attempt (accepted or rejected).

        Returns ``(carry, report)`` where ``report`` is a `StepReport`.
        Pure of the driver's save-grid, event, and statistics bookkeeping.
        """
        ...

    def dense(self, carry, t):
        """Evaluate ``(y(t), ydot(t))`` on the last accepted step's interval."""
        ...


class StepReport(eqx.Module):
    """What one step attempt reports back to the driver.

    All leaves are fixed-shape arrays. The adaptive controls (``accepted``,
    ``h_new``, ``k_new``, the counters) are integer/boolean state the driver
    accumulates and threads but never differentiates.

    Attributes:
        accepted: Whether the attempt was accepted.
        t_new: Time after the attempt (equal to the prior time if rejected).
        y_new: Accepted differential state (unchanged if rejected).
        ydot_new: Accepted state derivative.
        lam_new: Multipliers after the attempt.
        h_new: Step size chosen for the next attempt.
        k_new: Order chosen for the next attempt.
        err_estimate: The accept/reject error estimate.
        result: Termination code for this attempt (the code space
            [`result_from_code`][diff_soup._solution.result_from_code] consumes).
        newton_iters: Corrector-iteration delta for the shared statistics.
        jac_refreshes: Delta of fresh state-Jacobian evaluations. An attempt
            that factorizes a matrix recomposed from a saved state Jacobian
            contributes to ``factorizations`` but not to this.
        factorizations: Iteration-matrix-factorization delta.
        linear_solves: Linear-solve delta.
        rhs_evals: Right-hand-side-evaluation delta.
        residual_evals: Residual-evaluation delta.
        trace_record: The engine's per-attempt trace record, or ``None`` when
            the solve is not tracing (the driver supplies the shared header).
        precision_promotions: Per-cause promotion-counter delta of the
            adaptive linear-precision policy (``int32[4]``), or ``None`` for
            concrete solve modes.
        ir_refinement_steps: Mixed-precision refinement-step delta of this
            attempt's held-factor linear solves (fp64 scalar), or ``None``
            for an engine/configuration that does not report it.
        fp64_fallbacks: Reduced-precision fp64-fallback delta of this
            attempt's held-factor linear solves (fp64 scalar), or ``None``.
        multirhs_solves: Multi-right-hand-side linear-solve delta (the
            Schur-complement pre-builds this attempt ran), or ``None`` for an
            engine that does not report it.
        newton_failures: Nonlinear-corrector convergence-failure delta, or
            ``None`` for an engine that does not report it.
        multirhs_columns: Right-hand-side column delta across this attempt's
            multi-RHS solves, or ``None``.
        du_b_builds: Coupling-block border-solve construction delta, or
            ``None``.
        schur_builds: Schur-complement construction delta, or ``None``.
        decisions: The decision inputs this attempt used (the engine's
            ``decision_template`` shape), recorded by a checkpointed reverse
            solve so its replay decides as the forward sweep did, or ``None``
            for an engine that does not record them.
    """

    accepted: jax.Array
    t_new: jax.Array
    y_new: jax.Array
    ydot_new: jax.Array
    lam_new: jax.Array
    h_new: jax.Array
    k_new: jax.Array
    err_estimate: jax.Array
    result: jax.Array
    newton_iters: jax.Array
    jac_refreshes: jax.Array
    factorizations: jax.Array
    linear_solves: jax.Array
    rhs_evals: jax.Array
    residual_evals: jax.Array
    trace_record: Optional[Any] = None
    precision_promotions: Optional[jax.Array] = None
    ir_refinement_steps: Optional[jax.Array] = None
    fp64_fallbacks: Optional[jax.Array] = None
    multirhs_solves: Optional[jax.Array] = None
    newton_failures: Optional[jax.Array] = None
    multirhs_columns: Optional[jax.Array] = None
    du_b_builds: Optional[jax.Array] = None
    schur_builds: Optional[jax.Array] = None
    decisions: Optional[Any] = None


__all__ = ["Engine", "StepReport"]
