"""Public scalar solver classes: ``JNDF``, ``JBDF``, ``JIDA``.

Each class is an immutable ``eqx.Module`` bundling the integration
tolerances, the linear-backend choice, and an [ExecutionProfile][] — the
record that fixes which native engine kernel a solve runs and how it drives
that kernel's step-size control, nonlinear correction, and Jacobian reuse.
``profile.method_source`` is the sole identity axis distinguishing the three
classes: ``JNDF`` implements the Klopfenstein-Shampine NDF (numerical-
differentiation-formula) family, ``JBDF`` implements the classic backward-
differentiation-formula (BDF) family, and ``JIDA`` implements an implicit
index-1 differential-algebraic corrector.

Two presets assemble the common profiles:

* [stock][diff_soup.timestepping.vsvo.classes._VSVOSolver.stock] — the source-tracking profile: source-shaped step-size
  control and a Jacobian held across steps, the conservative default for
  production runs. Source-parity claims apply only to this profile.
* [flash][diff_soup.timestepping.vsvo.classes._VSVOSolver.flash] — a proportional-integral step-size controller with a fresh
  Jacobian every nonlinear iteration, for short, rapidly varying runs where
  Jacobian reuse hurts convergence. Makes no source-parity claim.

The native engine kernels are bound lazily inside ``solve``; every ``solve``
runs through the shared driver and returns the shared
[Solution][diff_soup.timestepping.vsvo.solution.Solution] record.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._precision import Precision, resolve_precision
from diff_soup.linear import LinearSolver
from diff_soup.linear._resolver import BackendResolver, array_platform, is_batched
from diff_soup.timestepping.tolerances import (
    ResolvedTolerances,
    ToleranceSpec,
    resolve_for_problem,
)
from diff_soup.timestepping.vsvo.engines import bdf_engine, ida_engine, ndf_engine
from diff_soup.utils.precision import warn_if_float32_default, warn_if_float32_on_cpu

from .common.contract import coerce_problem, pack_state
from .common.driver import DriverConfig, run_engine
from .solution import Solution

# The linear backend names a concrete sparse-direct or dense solver. The
# constructor only validates the name here; a concrete engine resolves it to
# a backend when it is built.
_LINEAR_NAMES = ("auto", "cudss", "klu", "superlu", "dense")


def _resolve_linear_name(
    configured: Any,
    problem: Any,
    y0: Any,
    theta: Any,
    t_span: Any,
    control_state0: Any,
    lambda0: Any,
    linear: Any,
) -> Any:
    """Resolve the effective linear backend for a solve: a concrete name, or an instance.

    Maps the requested backend (the ``solve`` override, else the solver's
    ``configured`` default) to a concrete name from the problem's state size and
    the initial state's device and every runtime input that may introduce an
    independent-problem batch: ``auto`` routes to a sparse backend on the real
    (large) problems and stays dense only for tiny blocks. Raises — never
    silently degrades to a dense ``O(n^2)`` solve — when a mandatory sparse
    backend is unavailable. The concrete name is what keys the engine builder's
    cache, so two problems of different sizes never share one backend. A
    configured [LinearSolver][diff_soup.linear.LinearSolver] instance (for
    example ``Krylov(CuDSS())``) is returned as it is.
    """
    requested = linear if linear is not None else configured
    if isinstance(requested, LinearSolver):
        return requested

    solver_problem = coerce_problem(problem)
    y0_arr = jnp.asarray(y0)
    return BackendResolver(backend=linear if linear is not None else configured).choose(
        block_size=int(solver_problem.n_u),
        platform=array_platform(y0_arr, theta, t_span, control_state0, lambda0),
        # A batch may enter through the span, initial state, parameters,
        # controls, or initial bordered globals. Backend selection must see all of
        # them: the iteration matrix can vary per lane even when y0 is shared.
        batched=is_batched(y0_arr, theta, t_span, control_state0, lambda0),
        needs_ad_rules=False,  # the engines differentiate via their own custom rules
    )


@dataclasses.dataclass(frozen=True)
class ExecutionProfile:
    """The step control, corrector and Jacobian policy a solver runs.

    `stock()` and `flash()` on each solver class build the two presets.

    Attributes:
        method_source: The engine, `"qndf"`, `"bdf"` or `"ida"`.
        controller_mode: `"source"` runs the reference method's step and order control, and
            `"flash"` resizes the step on every accepted step.
        nonlinear_mode: `"adaptive"` runs the adaptive corrector loop, and `"fixed"` a bounded
            number of iterations.
        jacobian_policy: `"source"` reuses the Jacobian as the reference method does, and
            `"fresh_each_iteration"` or `"fresh_each_step"` refreshes it.
        controller_law: The flash step-size law, `"pi"` (default) or `"h211b"`, read only with
            `controller_mode="flash"`.
    """

    method_source: Literal["qndf", "bdf", "ida"]
    controller_mode: Literal["source", "flash"]
    nonlinear_mode: Literal["adaptive", "fixed"]
    jacobian_policy: Literal["source", "fresh_each_iteration", "fresh_each_step"]
    controller_law: Literal["pi", "h211b"] = "pi"


def _stock_profile(method_source: str) -> ExecutionProfile:
    """The source-tracking profile for ``method_source``.

    Source-shaped step-size control and Jacobian reuse; source-parity claims
    apply only to this profile. ``nonlinear_mode="fixed"`` is this release's
    documented provisional default — a bounded-iteration corrector kept cheap
    by a small iteration cap — pending the endgame benchmark that decides the
    loop form for production.
    """
    return ExecutionProfile(method_source, "source", "fixed", "source")


def _flash_profile(method_source: str) -> ExecutionProfile:
    """The proportional-integral, fresh-Jacobian profile for ``method_source``.

    A proportional-integral step-size controller with an adaptive nonlinear
    corrector that refreshes its Jacobian every iteration. Makes no
    source-parity claim.
    """
    return ExecutionProfile(method_source, "flash", "adaptive", "fresh_each_iteration")


def _normalize_id_mask(value: Any) -> Any:
    """Normalize an IDA ``id`` differential-row mask to a static tuple of floats.

    ``None`` (derive the mask from the live mass diagonal) passes through; any
    iterable of per-row values becomes a hashable tuple of floats so the mask
    stays a clean static field and two masks differing only in container type
    (a list versus a tuple, say) compare and hash identically.
    """
    if value is None:
        return None
    return tuple(float(v) for v in value)


def _tolerance_wiring(tolerances: Any, problem: Any, solver: Any) -> tuple:
    """Resolve a solve's ``tolerances`` request into engine-ready values.

    Returns ``(record, rtol, atol, atol_rows, ic_tol, ic_rtol,
    newton_tol_factor)``. Without a request the solver's own scalar fields
    flow through unchanged (``record``, ``atol_rows``, and the
    initialization entries are ``None``). With one, the request supersedes
    the solver's scalar tolerances for this solve: ``atol_rows`` is the
    resolved per-row vector the error-weight law consumes, and the scalar
    ``atol`` entry is the tightest integration row. The scalar
    consistent-initialization gate ``ic_tol`` is set only when the request
    explicitly supplies an initialization tolerance (or is an already-resolved
    record); otherwise the engine applies the scalar-versus-vector gate policy
    to the original integration tolerance spelling.

    Args:
        tolerances: ``None``, a
            [ToleranceSpec][diff_soup.timestepping.tolerances.ToleranceSpec]
            (resolved here against the problem's layout), or an
            already-resolved
            [ResolvedTolerances][diff_soup.timestepping.tolerances.ResolvedTolerances]
            (validated against the problem's augmented-state size).
        problem: The problem the solve targets (any accepted form).
        solver: The solver whose scalar fields serve as the no-request
            values.

    Raises:
        ConfigurationError: For an unsupported ``tolerances`` type, or a
            pre-resolved record whose row count does not match the problem.
    """
    if tolerances is None:
        return None, solver.rtol, solver.atol, None, None, None, solver.newton_tol_factor

    coerced = coerce_problem(problem)
    if isinstance(tolerances, ToleranceSpec):
        record = resolve_for_problem(tolerances, coerced)
        explicit_ic_tol = tolerances.ic_atol is not None
    elif isinstance(tolerances, ResolvedTolerances):
        n = int(coerced.n_u) + int(coerced.n_lam)
        if int(tolerances.n_dofs) != n:
            raise ConfigurationError(
                f"the resolved tolerances cover {tolerances.n_dofs} rows but this "
                f"problem's augmented state has {n}. Resolve the specification "
                "against this problem (pass the ToleranceSpec itself, or "
                "re-resolve it)."
            )
        record = tolerances
        explicit_ic_tol = True
    else:
        raise ConfigurationError(
            "tolerances must be a ToleranceSpec or a ResolvedTolerances; got "
            f"{type(tolerances).__name__}."
        )
    atol_scalar = float(np.min(np.asarray(record.atol)))
    ic_tol = float(np.min(np.asarray(record.ic_atol))) if explicit_ic_tol else None
    return (
        record,
        record.rtol,
        atol_scalar,
        record.atol,
        ic_tol,
        record.ic_rtol,
        record.newton_tol_factor,
    )


def _reject_excluded_algebraic(record: Any) -> None:
    """Reject the algebraic-exclusion request on an engine that cannot honor it."""
    if record is not None and record.algebraic_error_control == "exclude":
        raise ConfigurationError(
            "this solver includes every row in its local error norm; "
            "algebraic_error_control='exclude' is available on JIDA. Use "
            "JIDA, or set algebraic_error_control='include'."
        )


class _VSVOSolver(eqx.Module):
    """The keyword settings of the solver constructors and of `stock()` and `flash()`.

    Attributes:
        rtol: The relative integration tolerance, default `1e-6`.
        atol: The absolute integration tolerance, default `1e-6`.
        newton_tol_factor: The scale of the Newton acceptance threshold, default `1.0`, while
            the step error test keeps the unscaled tolerances.
        ic_max_newton_iters: The Newton iteration cap of the consistent initialization, default
            `10`, on `JNDF` and `JBDF` only.
        linear: The linear backend, a name (`"auto"`, `"cudss"`, `"klu"`, `"superlu"`,
            `"dense"`) or a `ds.linear` object.
        precision: The `Precision`, used only when the problem declares none, so leave it unset.
        jacobian_values_dtype: Leave it unset, and set `Precision.jacobian_values_dtype` on the
            problem instead.
        profile: The `ExecutionProfile`, by default the class's `stock` profile.
    """

    rtol: float = 1e-6
    atol: float = 1e-6
    newton_tol_factor: float = 1.0
    ic_max_newton_iters: int = 10
    linear: Any = "auto"
    precision: Precision = eqx.field(static=True, default_factory=Precision)
    jacobian_values_dtype: Any = eqx.field(static=True, default=None)
    profile: ExecutionProfile = _stock_profile("qndf")

    def __post_init__(self) -> None:
        """Validate the linear-backend name and the corrector-band scale."""
        if not isinstance(self.linear, LinearSolver) and (
            not isinstance(self.linear, str) or self.linear.lower() not in _LINEAR_NAMES
        ):
            raise ConfigurationError(
                f"unknown linear backend {self.linear!r}; expected "
                "'auto', 'cudss', 'klu', 'superlu', 'dense', or a configured "
                "diff_soup.linear.LinearSolver instance such as KLU(ordering='colamd') or "
                "Krylov(CuDSS())."
            )
        if not float(self.newton_tol_factor) > 0.0:
            raise ConfigurationError(
                "newton_tol_factor must be a positive corrector-band scale; "
                f"got {self.newton_tol_factor!r}."
            )
        if int(self.ic_max_newton_iters) < 1:
            raise ConfigurationError(
                "ic_max_newton_iters must be a positive iteration cap for the "
                f"consistent-initialization Newton; got {self.ic_max_newton_iters!r}."
            )
        if self.precision.solve_mode == "adaptive" and self.jacobian_values_dtype is not None:
            raise ConfigurationError(
                "the adaptive solve mode manages linear-solve precision "
                "internally and always assembles Jacobian values in the state "
                "dtype; jacobian_values_dtype requires a concrete solve_mode "
                "('fp64', 'fp32', 'fp32_ir')."
            )
        if self.jacobian_values_dtype is not None:
            try:
                ok = jnp.issubdtype(jnp.dtype(self.jacobian_values_dtype), jnp.floating)
            except TypeError:
                ok = False
            if not ok:
                raise ConfigurationError(
                    "jacobian_values_dtype must be a floating dtype (for "
                    f"example jnp.float32) or None; got {self.jacobian_values_dtype!r}."
                )

    @classmethod
    def stock(cls, **kwargs: Any) -> "_VSVOSolver":
        """Build the solver with the stock profile, the reference method's step control and Jacobian reuse.

        Args:
            **kwargs: Constructor settings, which override the profile.

        Returns:
            The solver.
        """
        kwargs.setdefault("profile", _stock_profile(cls.method_source))
        return cls(**kwargs)

    @classmethod
    def flash(cls, **kwargs: Any) -> "_VSVOSolver":
        """Build the solver with the flash profile, proportional-integral step control and a fresh Jacobian.

        Args:
            **kwargs: Constructor settings, which override the profile.

        Returns:
            The solver.
        """
        kwargs.setdefault("profile", _flash_profile(cls.method_source))
        return cls(**kwargs)

    def _default_consistent_ic(self) -> bool:
        """The ``consistent_ic`` a solve uses when its own argument is ``None``."""
        return True

    def _build_engine(self, *, tol_record: Any, **options: Any) -> Any:
        """Build this solver's engine kernel; every concrete subclass implements it.

        ``options`` holds the resolved ``rtol``, ``atol``, ``linear``,
        ``newton_tol_factor``, ``precision``, ``ic_tol`` and ``ic_rtol``;
        ``tol_record`` is the resolved tolerance record or ``None``.
        """
        raise NotImplementedError("the engine is built by the concrete subclasses")

    def solve(
        self,
        problem: Any,
        t_span: Any,
        y0: Any,
        *,
        saveat: Any = None,
        linear: Any = None,
        hmin: float = 0.0,
        hmax: float = 0.0,
        dt0: float | Any = 0.0,
        max_order: int = 5,
        max_steps: int = 10_000,
        max_events: int = 16,
        event: Any = None,
        control_state0: Any = None,
        theta: Any = None,
        consistent_ic: Any = None,
        lambda0: Any = None,
        adjoint: Any = None,
        tolerances: Any = None,
        trace: bool = False,
        history: Any = None,
    ) -> Solution:
        """Integrate `problem` over `t_span` from `y0`.

        A repeat solve skips compilation for new values of `t_span`, `y0`, `theta`,
        `control_state0` and `dt0`. Keep the problem object stable across calls, since a new one
        compiles again.

        Args:
            problem: The built problem.
            t_span: The interval `(t0, t1)`.
            y0: The initial state.
            saveat: The output times, or `None` for the end of the interval.
            linear: A linear backend for this solve, overriding the solver's.
            hmin: The smallest step size, `0` to leave it to the engine.
            hmax: The largest step size, `0` to leave it to the engine.
            dt0: The first step size, `0` to estimate it.
            max_order: The highest integration order, at most `5`.
            max_steps: The step budget, which also sizes the output buffers.
            max_events: The capacity of the event log, past which `event_log.overflow` is set.
            event: An event or event set to monitor.
            control_state0: The initial control values, `{name: value}` for a Recipe with bound
                controls.
            theta: The parameters, or `None` for the problem's defaults.
            consistent_ic: `True` or `False` to decide the consistent initialization of this solve,
                `None` for the solver's default.
            lambda0: The initial values of the global unknowns, when `y0` holds only the field
                block.
            adjoint: The differentiation strategy, or `None` for the solver's.
            tolerances: A `ToleranceSpec`, overriding the solver's scalar tolerances.
            trace: `True` attaches a per-step diagnostic trace.
            history: `True` records `Solution.history`, and an earlier solve's `Solution.history`
                continues from its end.

        Returns:
            The `Solution`.

        Raises:
            ValueError: If the problem needs a capability the engine does not have.
            ConfigurationError: If the engine cannot honour a setting, such as
                `algebraic_error_control="exclude"` off `JIDA`.
        """
        warn_if_float32_default()
        precision = resolve_precision(
            coerce_problem(problem), self.precision, self.jacobian_values_dtype
        )
        warn_if_float32_on_cpu(precision, y0, theta, t_span)

        eff_adjoint = adjoint if adjoint is not None else self.adjoint
        tol_record, eff_rtol, eff_atol, atol_rows, ic_tol, ic_rtol, eff_ntf = _tolerance_wiring(
            tolerances, problem, self
        )
        engine = self._build_engine(
            tol_record=tol_record,
            rtol=eff_rtol,
            atol=eff_atol,
            linear=_resolve_linear_name(
                self.linear,
                problem,
                y0,
                theta,
                t_span,
                control_state0,
                lambda0,
                linear,
            ),
            newton_tol_factor=eff_ntf,
            precision=precision,
            ic_tol=ic_tol,
            ic_rtol=ic_rtol,
        )
        if lambda0 is not None:
            y0 = pack_state(y0, lambda0)
        cfg = DriverConfig(
            max_steps=max_steps,
            max_order=max_order,
            max_events=max_events,
            trace=trace,
            adjoint=eff_adjoint,
            hmin=hmin,
            hmax=hmax,
            dt0=dt0,
            atol=eff_atol if atol_rows is None else atol_rows,
            rtol=eff_rtol,
            event=event,
            control_state0=control_state0,
            consistent_ic=(
                self._default_consistent_ic() if consistent_ic is None else bool(consistent_ic)
            ),
            theta=theta,
            history=history,
        )
        t0, t1 = t_span
        sol = run_engine(engine, problem, t0, t1, y0, saveat, cfg)
        if tol_record is not None:
            sol = dataclasses.replace(sol, tolerances=tol_record)
        return sol


class JNDF(_VSVOSolver):
    """The numerical differentiation formula (NDF) solver, the method of MATLAB's `ode15s`.

    It solves ODEs and mass-form DAEs `M(u, t) du/dt = f`, including a state-dependent mass.

    Attributes:
        adjoint: The default differentiation strategy, where `None` is forward mode and
            `CheckpointAdjoint()` supports `jax.grad`.
    """

    method_source = "qndf"
    profile: ExecutionProfile = _stock_profile("qndf")
    adjoint: Any = eqx.field(static=True, default=None)

    def _build_engine(self, *, tol_record: Any, **options: Any) -> Any:
        """Build the NDF engine kernel from this solver's profile."""
        _reject_excluded_algebraic(tol_record)
        precision = options["precision"]
        return ndf_engine(
            self.profile,
            ic_max_newton_iters=self.ic_max_newton_iters,
            jacobian_values_dtype=precision.jacobian_values_dtype,
            **options,
        )


class JBDF(_VSVOSolver):
    """The backward differentiation formula (BDF) solver, with a fixed-leading-coefficient corrector.

    It solves ODEs and mass-form DAEs `M(t) du/dt = f` with a constant or time-dependent mass.

    Attributes:
        adjoint: The default differentiation strategy, where `None` is forward mode and
            `CheckpointAdjoint()` supports `jax.grad`.
    """

    method_source = "bdf"
    profile: ExecutionProfile = _stock_profile("bdf")
    adjoint: Any = eqx.field(static=True, default=None)

    def _build_engine(self, *, tol_record: Any, **options: Any) -> Any:
        """Build the BDF engine kernel from this solver's profile."""
        _reject_excluded_algebraic(tol_record)
        precision = options["precision"]
        return bdf_engine(
            self.profile,
            ic_max_newton_iters=self.ic_max_newton_iters,
            jacobian_values_dtype=precision.jacobian_values_dtype,
            **options,
        )


class JIDA(_VSVOSolver):
    """The implicit DAE solver of the SUNDIALS IDA family, on the residual form `F(u, du/dt, t) = 0`.

    It solves ODEs and fully implicit index-1 DAEs, and it is the recommended solver.

    Attributes:
        consistent_ic: `True` (default) projects the initial state before the first step, a
            no-op for a pure ODE.
        ic_mode: The initialization route, `"ya_ydp"` (default), `"ya_ydp_euler"`,
            `"y_fixed_ydot"` or `"assume_consistent"`.
        suppress_alg: `True` (default) keeps the algebraic rows out of the local error test.
        id: The differential-row mask of the error test, by default from the problem's mass
            structure.
        adjoint: The default differentiation strategy, where `None` is forward mode and
            `CheckpointAdjoint()` supports `jax.grad`.
    """

    method_source = "ida"
    profile: ExecutionProfile = _stock_profile("ida")
    consistent_ic: bool = eqx.field(static=True, default=True)
    ic_mode: str = eqx.field(static=True, default="ya_ydp")
    suppress_alg: bool = eqx.field(static=True, default=True)
    # The IDA ``id`` differential-row mask is gradient-dead configuration; it
    # is carried as a static tuple (its field converter normalizes it) so the
    # solver instance stays a clean static introspection handle (an
    # array-valued field would emit a stray-array warning wherever this
    # instance rides as static metadata).
    id: Any = eqx.field(static=True, default=None, converter=_normalize_id_mask)
    adjoint: Any = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        """Validate the initialization mode.

        The ``id`` differential-row mask is normalized to a static tuple by its
        field converter
        ([_normalize_id_mask][diff_soup.timestepping.vsvo.classes._normalize_id_mask]),
        so no post-init step is
        needed for it.
        """
        super().__post_init__()
        accepted_ic_modes = (
            "assume_consistent",
            "y_fixed_ydot",
            "ya_ydp",
            "ya_ydp_euler",
        )
        if self.ic_mode not in accepted_ic_modes:
            raise ConfigurationError(
                "JIDA.ic_mode must be one of {'assume_consistent', "
                "'y_fixed_ydot', 'ya_ydp', 'ya_ydp_euler'}; "
                f"got {self.ic_mode!r}."
            )

    def _default_consistent_ic(self) -> bool:
        """A solve without its own ``consistent_ic`` uses this solver's field."""
        return bool(self.consistent_ic)

    def _build_engine(self, *, tol_record: Any, **options: Any) -> Any:
        """Build the IDA engine kernel from this solver's profile.

        The resolved record's algebraic-error-control choice governs the
        error-norm masking for the solve; the solver's own ``suppress_alg``
        serves the plain-scalar spelling.
        """
        if int(self.ic_max_newton_iters) != 10:
            raise ConfigurationError(
                "ic_max_newton_iters is not wired to the implicit-DAE engine's "
                "initialization modes; JIDA runs its consistent initialization "
                "at the engine default. Use JNDF or JBDF for a configurable "
                "initialization iteration cap."
            )
        suppress = (
            self.suppress_alg
            if tol_record is None
            else tol_record.algebraic_error_control == "exclude"
        )
        precision = options["precision"]
        return ida_engine(
            self.profile,
            suppress_alg=suppress,
            id=self.id,
            ic_mode=self.ic_mode,
            jacobian_values_dtype=precision.jacobian_values_dtype,
            **options,
        )


__all__ = ["JNDF", "JBDF", "JIDA", "ExecutionProfile"]
