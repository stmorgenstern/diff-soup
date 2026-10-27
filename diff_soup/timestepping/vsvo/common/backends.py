"""Shared linear-backend resolution for the vsvo engines.

Backends are stateless dispatch objects — every mutable resource (native
solver handles, factor slots, symbolic analyses) lives in the linear layers
and the process-global registries. The resolver therefore returns
module-level singletons: a stable backend identity keeps an engine's static
configuration cache-stable, so two engines built from the same configuration
share one compiled program instead of recompiling per construction.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp

from diff_soup._precision import Precision
from diff_soup.linear import KLU, CuDSS, Dense, LinearSolver, SuperLU
from diff_soup.linear.superlu_backend import require_superlu
from diff_soup.nonlinear._kkt import LinearPlan

_SINGLETONS: dict[str, Any] = {}

# Corrector-tuned refinement settings the engines analyze with under
# ``solve_mode="fp32_ir"``: a Newton corrector only needs a modest linear
# residual (the outer iteration absorbs the rest), so a loose target with a
# small adaptive budget keeps the refinement at the 1-2 steps it is worth.
_FP32_IR_CORRECTOR_OPTIONS: dict[str, object] = {
    "ir_target_rel_residual": 0.05,
    "ir_max_steps": 5,
    # The mixed solver owns the fp64-residual refinement loop above the
    # factor.  A native cuDSS refinement step would nest a hidden sweep inside
    # every one of those solves, with especially severe cost for jacfwd's
    # multi-RHS batches, so the corrector pins it off whatever the default.
    "ir_nsteps": 0,
}

# A genuinely end-to-end fp32 solve already has two outer guards: the
# nonlinear corrector rejects a bad step, and the driver converts every
# non-success/non-finite trajectory into the optimization failure sentinel.
# Re-measuring every fresh sparse solve in fp64 defeats the point of this
# policy. Mixed fp64-state/fp32-factor policies do not receive these defaults
# and retain the checked linear gate.
_FP32_END_TO_END_CORRECTOR_OPTIONS: dict[str, object] = {
    "fp32_residual_gate": "outer",
    "fp32_scaling": "off",
    # This policy promises a genuinely direct fp32 solve, so it pins cuDSS's
    # native refinement off whatever the default: on wide sensitivity batches
    # a hidden refinement step can dominate the entire statistics kernel.
    "ir_nsteps": 0,
}


def resolve_backend(linear: Any) -> Any:
    """Resolve a CONCRETE linear-backend name to a process-global instance.

    ``"dense"`` returns the dense backend; ``"klu"``, ``"superlu"`` and
    ``"cudss"`` return their sparse-direct backends at their default settings.
    ``"superlu"`` and ``"cudss"`` raise when their native extension is
    unavailable — neither silently falls back to a dense ``O(n^2)`` solve.
    ``"auto"`` is NOT resolved here: the size- and platform-aware
    [BackendResolver][diff_soup.linear._resolver.BackendResolver] must pick a concrete
    name upstream, so a bare ``"auto"`` (or any unknown name) reaching this
    function is a bug and raises. The same name always returns the same instance.
    A configured [LinearSolver][diff_soup.linear.LinearSolver] instance (for
    example ``KLU(ordering="colamd")`` or ``Krylov(CuDSS())``) is returned as
    it is.

    Args:
        linear: A concrete backend name (``"cudss"``, ``"klu"``, ``"superlu"``,
            ``"dense"``), or a backend instance.

    Returns:
        The backend singleton for ``linear``, or the instance itself.

    Raises:
        RuntimeError: When ``"superlu"`` or ``"cudss"`` is requested but its
            native extension is not available on this host.
        ValueError: When ``linear`` is ``"auto"`` or an unrecognized name.
    """
    if isinstance(linear, LinearSolver):
        return linear
    name = str(linear).lower()
    inst = _SINGLETONS.get(name)
    if inst is not None:
        return inst
    if name == "klu":
        inst = KLU()
    elif name == "superlu":
        require_superlu("linear='superlu'")
        inst = SuperLU()
    elif name == "cudss":
        if CuDSS is None:
            raise RuntimeError(
                "the cuDSS backend was requested but its native extension is not "
                "available on this host; install the cuDSS build, or select 'klu' "
                "(CPU) or 'dense'."
            )
        inst = CuDSS()
    elif name == "dense":
        inst = Dense()
    else:
        raise ValueError(
            f"resolve_backend received the non-concrete name {linear!r}; the size- and "
            "platform-aware BackendResolver must choose a concrete backend ('dense', "
            "'klu', 'superlu', or 'cudss') before this call, so 'auto' must not reach here."
        )
    _SINGLETONS[name] = inst
    return inst


def sensitivity_precision(precision: Any, *, mode: str = "forward") -> Any:
    """The :class:`Precision` the engines analyze their sensitivity pattern with.

    The per-step root derivative factors the full-system matrix at the
    accepted root. A ``None`` direction-specific field inherits
    ``precision.solve_mode``; an explicit field overrides it. Forward and
    reverse choices remain distinct so either direction can opt back into a
    full-precision derivative factor under a reduced primal solve.

    Args:
        precision: The engine's :class:`~diff_soup._precision.Precision`, or
            ``None``.
        mode: The derivative direction whose policy is resolved:
            ``"forward"`` for JVP/Jacobian (tangent) solves or ``"reverse"``
            for transposed cotangent solves.

    Returns:
        The :class:`Precision` for the engine's sensitivity ``analyze`` call.
    """
    if precision is None:
        return Precision()
    if mode not in ("forward", "reverse"):
        raise ValueError(f"sensitivity mode must be 'forward' or 'reverse'; got {mode!r}.")
    field = (
        "forward_sensitivity_solve_mode" if mode == "forward" else "reverse_sensitivity_solve_mode"
    )
    target = getattr(precision, field, None)
    if target is None:
        target = precision.solve_mode
    if precision.solve_mode == target:
        return precision
    return eqx.tree_at(lambda p: p.solve_mode, precision, target)


def analyze_sensitivity(
    backend: Any,
    topology: Any,
    *,
    layout: Any,
    precision: Any,
    mode: str,
) -> Any:
    """The analysis plan of an accepted-root sensitivity system, with its complete policy.

    Keeping precision selection and analyze-option selection at one call site
    prevents a reduced sensitivity factor from silently falling back to a
    backend's default residual gate or native refinement settings. All stock
    VSVO engines use this helper for their exact-at-root derivative factors;
    the root's derivative rule analyses the returned
    [LinearPlan][diff_soup.nonlinear._kkt.LinearPlan] on the factor contract.

    Args:
        backend: The resolved linear backend (the sensitivity solves use it).
        topology: The full-system pattern.
        layout: The problem's [Layout][diff_soup.Layout].
        precision: The engine's [Precision][diff_soup._precision.Precision].
        mode: ``"forward"`` or ``"reverse"``, the derivative direction.

    Returns:
        The plan.
    """
    del backend
    selected = sensitivity_precision(precision, mode=mode)
    return LinearPlan(
        topology=topology,
        layout=layout,
        options=tuple(sorted((linear_analyze_options(selected) or {}).items())),
        precision=selected,
    )


def adaptive_rung_precisions(precision: Any) -> tuple:
    """The rung-ordered concrete precisions the adaptive mode switches over.

    The ladder is ``fp32_ir`` (rung 0 — the cheap attempt, corrector-tuned
    refinement over an fp32 factor) then ``fp64`` (rung 1 — the terminal
    precision). Every other field of ``precision`` is preserved on each rung.

    Args:
        precision: A [Precision][diff_soup._precision.Precision] with
            ``solve_mode="adaptive"``.

    Returns:
        The rung-ordered tuple of concrete [Precision][diff_soup._precision.Precision] records.
    """

    def _with_mode(mode: str):
        return eqx.tree_at(lambda p: p.solve_mode, precision, mode)

    return (_with_mode("fp32_ir"), _with_mode("fp64"))


def initialization_policy(problem: Any, precision: Any, *, mode: str = "forward") -> tuple:
    """The ``(layout, precision)`` an initialization factor is analyzed with.

    The consistent-initialization projection and the startup slope factor the
    same problem as the corrector, so they are placed by the lowered
    problem's ``Layout`` rather than the backend's default: a multi-GPU
    factorization request otherwise never reaches the projection every DAE
    with algebraic rows must run before its first step.

    The forward-direction analysis serves both the projection's Newton
    solves and the tangent solve of its implicit-function derivative at the
    converged root; the reverse-direction analysis serves the transposed
    factor a reverse sweep solves with. Each follows the accepted-root
    sensitivity policy of its direction
    ([sensitivity_precision][diff_soup.timestepping.vsvo.common.backends.sensitivity_precision]):
    a reduced mode is taken only where the caller declared one for that
    derivative direction, and the conservative full-precision factor is kept
    otherwise. Callers resolve both directions and pass the forward policy as
    ``precision=`` and the reverse policy as ``reverse_precision=``.

    Args:
        problem: The engine-facing problem view.
        precision: The engine's [Precision][diff_soup._precision.Precision],
            or ``None``.
        mode: ``"forward"`` (tangent solves, and the primal Newton solves) or
            ``"reverse"`` (transposed cotangent solves) — the derivative
            direction whose policy is resolved.

    Returns:
        ``(layout, precision)`` for
        [SolverView.consistent_initial_state][diff_soup.problem.implicit.solver_problem.SolverView.consistent_initial_state]
        and
        [SolverView.startup_slope][diff_soup.problem.implicit.solver_problem.SolverView.startup_slope];
        either entry is ``None`` when nothing was declared.
    """
    layout = getattr(problem, "layout", None)
    if precision is None:
        return layout, None
    selected = sensitivity_precision(precision, mode=mode)
    if selected.solve_mode == "adaptive":
        # Initialization has no accepted-step precision rung to inherit: the
        # adaptive ladder is a runtime policy owned by the timestep engine,
        # while projection runs before that carry exists. Keep its primal and
        # root factor concrete and conservative. A caller differentiating an
        # adaptive solve can select a concrete derivative mode explicitly.
        selected = eqx.tree_at(lambda item: item.solve_mode, selected, "fp64")
    return layout, selected


def linear_analyze_options(precision: Any, options: Optional[dict] = None) -> Optional[dict]:
    """Merge solve-policy defaults into a stock corrector's analyze options.

    ``fp32_ir`` receives corrector-tuned refinement settings and disables
    cuDSS's separate native refinement step so there is exactly one explicit,
    observable refinement loop. End-to-end fp32 (both ``compute_dtype`` and
    ``solve_mode`` are fp32) selects the direct sparse path whose linear
    accuracy is guarded by the outer nonlinear corrector and
    trajectory-failure sentinel; it therefore skips both native refinement
    and the otherwise-costly fp64 fresh-factor residual measurement. A mixed
    fp64-state/fp32-factor policy retains the checked linear gate. Caller
    options always win. The fp64 path returns ``options`` unchanged so its
    compiled program is byte-for-byte unaffected.

    Args:
        precision: The engine's [Precision][diff_soup._precision.Precision].
        options: Analyze options the caller already holds, or ``None``.

    Returns:
        The options dict to pass to ``backend.analyze``, or ``None``.
    """
    mode = getattr(precision, "solve_mode", "fp64")
    if mode == "fp32_ir":
        defaults = _FP32_IR_CORRECTOR_OPTIONS
    elif mode == "fp32":
        compute_dtype = jnp.dtype(getattr(precision, "compute_dtype", jnp.float64))
        defaults = (
            _FP32_END_TO_END_CORRECTOR_OPTIONS if compute_dtype == jnp.dtype(jnp.float32) else {}
        )
    else:
        defaults = {}
    if not defaults:
        return options
    merged: dict = dict(defaults)
    if options:
        merged.update(options)
    return merged


__all__ = [
    "resolve_backend",
    "linear_analyze_options",
    "analyze_sensitivity",
    "adaptive_rung_precisions",
    "sensitivity_precision",
]
