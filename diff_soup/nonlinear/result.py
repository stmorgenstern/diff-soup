"""Structured result contract shared by the Newton engines.

The nonlinear-solver engines report one structured [NewtonResult][diff_soup.nonlinear.result.NewtonResult] that
carries the converged state, the accumulated Newton correction, convergence /
diagnostic scalars, a typed [FailureKind][diff_soup.nonlinear.result.FailureKind] code, and per-solve linear-work
counters. The engine also accepts one [LinearizationDirective][diff_soup.nonlinear.result.LinearizationDirective] per solve
that declares whether to reuse a supplied Jacobian / factor and how to rescale
the Newton update for a stale linearization coefficient. Keeping the reuse
decision in an explicit directive means the nonlinear layer never decides
Jacobian reuse on its own — each timestepper hands down the directive it wants.

All runtime result scalars (``converged``, ``iterations``, ``rate``,
``failure``) are traced arrays in the hot path; [FailureKind][diff_soup.nonlinear.result.FailureKind] exists only
so the ``failure`` code reads as a name at rest. The failure code is carried as
an ``int32`` array holding ``int(FailureKind.X)``.
"""

from __future__ import annotations

import enum
from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax


class FailureKind(enum.IntEnum):
    """Typed reason a Newton solve did not reach a clean converged root.

    ``NONE`` is the success sentinel (value ``0``). The runtime
    ``NewtonResult.failure`` field carries ``int(FailureKind.X)`` as a traced
    ``int32`` array; this enum is for readability only and never appears in the
    hot path.

    The generic corrector reports ``NONE`` (converged), ``CONVERGENCE`` (the
    iteration cap was reached without meeting the convergence test), and
    ``TOO_SLOW`` (the rate guard latched on a stalling / diverging sweep). The
    ``RESIDUAL_RECOVERABLE`` / ``RESIDUAL_FATAL`` and ``LINEAR_SETUP`` /
    ``LINEAR_SOLVE`` members are set by the per-source engine kernels and the
    linear layer, which report a failed residual gate or a failed linear
    factorisation / solve.

    Members:
        NONE: The solve converged; no failure.
        CONVERGENCE: The iteration cap was hit without convergence.
        RESIDUAL_RECOVERABLE: The residual gate failed but a retry may recover.
        RESIDUAL_FATAL: The residual gate failed and no retry can recover.
        LINEAR_SETUP: A Jacobian factorisation / setup failed.
        LINEAR_SOLVE: A linear system solve failed.
        TOO_SLOW: The convergence-rate guard latched on a stall.
    """

    NONE = 0
    CONVERGENCE = 1
    RESIDUAL_RECOVERABLE = 2
    RESIDUAL_FATAL = 3
    LINEAR_SETUP = 4
    LINEAR_SOLVE = 5
    TOO_SLOW = 6


class LinearizationDirective(eqx.Module):
    """Per-solve instruction from a timestepper to the Newton layer.

    Declares how the engine sets up the linear system for one solve. There are
    three source directives:

    * ``fresh`` — re-evaluate the Jacobian and refit the factor (a full fresh
      Newton setup). Built by [fresh][].
    * ``refit_factor`` — keep the supplied Jacobian but refit its numeric factor.
    * ``reuse`` — reuse both the supplied Jacobian and its factor (a chord /
      simplified-Newton sweep). Built by [reuse][].

    Attributes:
        refresh_jacobian: When ``True``, re-evaluate the Jacobian for this solve
            rather than reuse the supplied blocks. A static Python bool: it gates
            a trace-time structural choice (whether the engine hands the
            corrector fresh blocks or reused ones).
        refit_factor: When ``True``, refit the numeric factor. A static Python
            bool for the same trace-time reason.
        step_scale: The stale-coefficient rescale applied to the Newton update.
            The sources apply ``2/(1+cjratio)`` (IDA) or ``2/(1+gamrat)``
            (CVODE) to correct a chord step taken under a drifted linearization
            coefficient. Default ``1.0`` is a no-op. A runtime leaf (a scalar or
            a 0-d array), so a traced per-step value is accepted.
    """

    refresh_jacobian: bool = eqx.field(static=True)
    refit_factor: bool = eqx.field(static=True)
    step_scale: Any

    @classmethod
    def fresh(cls, step_scale: Any = 1.0) -> "LinearizationDirective":
        """A fresh-Jacobian, refit-factor directive (full Newton setup)."""
        return cls(refresh_jacobian=True, refit_factor=True, step_scale=step_scale)

    @classmethod
    def reuse(cls, step_scale: Any = 1.0) -> "LinearizationDirective":
        """A reuse-Jacobian, reuse-factor directive (chord Newton sweep)."""
        return cls(refresh_jacobian=False, refit_factor=False, step_scale=step_scale)


class LinearStats(NamedTuple):
    """Per-solve linear-work counters carried on a [NewtonResult][diff_soup.nonlinear.result.NewtonResult].

    Attributes:
        factorizations: Numeric factorisations performed this solve (int32).
        linear_solves: Linear-system solves performed this solve (int32).
        jac_refreshes: Jacobian re-evaluations performed this solve (int32).
        stale_retries: Outer stale-Jacobian retries taken this solve (int32).
    """

    factorizations: jax.Array
    linear_solves: jax.Array
    jac_refreshes: jax.Array
    stale_retries: jax.Array


class NewtonResult(eqx.Module):
    """The structured result of one Newton solve.

    Attributes:
        y: The converged augmented state ``[u; lam]`` (differential unknowns
            stacked with the global algebraic unknowns).
        correction: The accumulated Newton correction ``y - z0`` from the initial
            guess ``z0``.
        converged: Boolean scalar — the convergence test was met.
        iterations: The first-convergence iteration count (int32).
        residual_norm: The post-update max-abs residual ``|[R; g]|`` at the
            converged iterate.
        update_norm: The weighted-RMS norm of the total accepted ``correction``.
            Reported for diagnostics; the engines never consume it for control
            decisions.
        rate: The geometric-mean convergence-rate estimate.
        failure: The [FailureKind][diff_soup.nonlinear.result.FailureKind] value as a traced ``int32`` array
            (``int(FailureKind.NONE)`` on success).
        linear_stats: A [LinearStats][diff_soup.nonlinear.result.LinearStats] of per-solve linear-work counters,
            or ``None`` when the engine does not populate it.
    """

    y: jax.Array
    correction: jax.Array
    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    update_norm: jax.Array
    rate: jax.Array
    failure: jax.Array
    linear_stats: Optional[LinearStats]
