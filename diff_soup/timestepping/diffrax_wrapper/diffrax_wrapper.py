"""Recipe-integrated Diffrax backend for differential-only problems.

``DiffraxStepper`` exposes Diffrax's explicit non-stiff solver collection
through ``pdesolve`` while retaining diff_soup's shared ``SaveAt`` and
``Solution`` reading contracts. Eligible systems have identity mass on every
differential row; Dirichlet rows are handled by differentiating their boundary
values in time. Algebraic rows and general mass operators belong on JNDF,
JBDF, or JIDA.

The optional dependency stays lazy: importing or constructing this module does
not import Diffrax. It is required only when ``solve`` starts.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._adjoints import (
    AbstractAdjoint,
    BacksolveAdjoint,
    CheckpointAdjoint,
    ForwardAdjoint,
    ImplicitAdjoint,
)
from diff_soup._errors import ConfigurationError
from diff_soup._precision import cast_to_compute_dtype, resolve_precision
from diff_soup._solution import RESULT
from diff_soup.timestepping.event_records import TerminalRecord
from diff_soup.timestepping.saveat import SaveAt, _build_save_plan, output_emitter
from diff_soup.timestepping.vsvo.solution import (
    SavedOutputs,
    Solution,
    SolverStats,
    build_solution,
)
from diff_soup.utils.precision import warn_if_float32_on_cpu


def _require_diffrax() -> Any:
    """Import Diffrax or explain how to install the optional backend."""
    try:
        import diffrax  # noqa: PLC0415

        return diffrax
    except ImportError as exc:
        raise ImportError(
            "DiffraxStepper requires the optional Diffrax dependency; "
            "install it with `pip install diffrax`."
        ) from exc


def _map_adjoint(adjoint: Any, diffrax_mod: Any) -> Any:
    """Map the shared adjoint family onto Diffrax's adjoint implementations."""
    if adjoint is None or isinstance(adjoint, CheckpointAdjoint):
        checkpoints = None if adjoint is None else adjoint.checkpoints
        if checkpoints is None:
            return diffrax_mod.RecursiveCheckpointAdjoint()
        return diffrax_mod.RecursiveCheckpointAdjoint(checkpoints=checkpoints)
    if isinstance(adjoint, ForwardAdjoint):
        return diffrax_mod.ForwardMode()
    if isinstance(adjoint, BacksolveAdjoint):
        return diffrax_mod.BacksolveAdjoint()
    if isinstance(adjoint, ImplicitAdjoint):
        return diffrax_mod.ImplicitAdjoint()
    raise ConfigurationError(
        "DiffraxStepper received an unknown adjoint strategy "
        f"{type(adjoint).__name__}; use CheckpointAdjoint, ForwardAdjoint, "
        "BacksolveAdjoint, ImplicitAdjoint, or None."
    )


def _events(event: Any) -> tuple[Any, ...]:
    """Normalize a shared event declaration to declaration order."""
    if event is None:
        return ()
    return tuple(event.events) if hasattr(event, "events") else (event,)


def _event_root_finder(root_finder: Any, diffrax_mod: Any) -> Any:
    """Translate the shared bisection configuration to Optimistix."""
    if root_finder is None:
        return None
    import optimistix as optx  # noqa: PLC0415

    if isinstance(root_finder, optx.AbstractRootFinder):
        return root_finder
    if all(hasattr(root_finder, name) for name in ("rtol", "atol", "max_iters_cap")):
        return optx.Bisection(rtol=float(root_finder.rtol), atol=float(root_finder.atol))
    raise ConfigurationError(
        "DiffraxStepper cannot translate the event root finder "
        f"{type(root_finder).__name__}; use diff_soup.Bisection or an "
        "Optimistix AbstractRootFinder."
    )


def _map_events_to_diffrax(event: Any, diffrax_mod: Any, problem: Any) -> Any:
    """Map terminate-only shared events to a Diffrax event declaration."""
    declared = _events(event)
    if not declared:
        return None

    conditions = []
    root_finders = []
    for item in declared:
        if getattr(item, "affect", None) is not None:
            raise ConfigurationError(
                "DiffraxStepper is terminate-only and cannot apply an event affect; "
                "use JNDF, JBDF, or JIDA for mutate-and-continue events."
            )
        root_finder = getattr(item, "root_finder", None)
        direction = getattr(item, "direction", "any")
        if root_finder is not None and direction != "any":
            raise ConfigurationError(
                "Diffrax continuous events do not preserve an up/down crossing "
                f"direction (received direction={direction!r}); use direction='any' "
                "or a native JNDF, JBDF, or JIDA engine."
            )
        state_policy = getattr(item, "state_policy", None)
        if state_policy is not None and state_policy.mode == "projected":
            raise ConfigurationError(
                "DiffraxStepper cannot project event-interpolation states; use the "
                "default fast event policy or a native engine for projected events."
            )
        raw_condition = item.cond_fn

        def condition(t, y, args, _raw=raw_condition, **kwargs):
            local = y[: problem.n_u]
            bordered = y[problem.n_u :]
            return _raw(t, local, args, lambdas=bordered, control=None, **kwargs)

        conditions.append(condition)
        root_finders.append(_event_root_finder(root_finder, diffrax_mod))

    root_finder = next((finder for finder in root_finders if finder is not None), None)
    condition_tree: Any = conditions[0] if len(conditions) == 1 else tuple(conditions)
    return diffrax_mod.Event(cond_fn=condition_tree, root_finder=root_finder)


def _map_result(diffrax_result: Any, diffrax_mod: Any) -> RESULT:
    """Map Diffrax's traced result enumeration to the shared result enumeration."""
    native = diffrax_mod.RESULTS
    result = RESULT.UNKNOWN
    result = RESULT.where(diffrax_result == native.max_steps_reached, RESULT.MAX_STEPS, result)
    result = RESULT.where(diffrax_result == native.dt_min_reached, RESULT.STEP_COLLAPSE, result)
    result = RESULT.where(diffrax_result == native.event_occurred, RESULT.EVENT_STOP, result)
    return RESULT.where(diffrax_result == native.successful, RESULT.SUCCESS, result)


def _result_code(result: RESULT) -> jax.Array:
    """Return the shared integer code carried by aligned terminal metadata."""
    code = jnp.asarray(0, dtype=jnp.int32)
    # These are the engine-neutral termination codes consumed by
    # result_from_code and used in every TerminalRecord.
    for value, member in (
        (7, RESULT.ACCURACY_LIMIT),
        (6, RESULT.INCONSISTENT_IC),
        (5, RESULT.EVENT_STOP),
        (4, RESULT.MAX_STEPS),
        (3, RESULT.NONLINEAR_DIVERGED),
        (2, RESULT.STEP_COLLAPSE),
        (1, RESULT.SUCCESS),
    ):
        code = jnp.where(result == member, jnp.asarray(value, jnp.int32), code)
    return code


def _map_stats(diffrax_stats: dict, t_reached: Any, result: RESULT) -> SolverStats:
    """Map Diffrax's step counters to the shared solver-statistics record."""
    zero = jnp.asarray(0, dtype=jnp.int32)
    return SolverStats(
        accepts=jnp.asarray(diffrax_stats.get("num_accepted_steps", zero), jnp.int32),
        rejects=jnp.asarray(diffrax_stats.get("num_rejected_steps", zero), jnp.int32),
        trial_steps=jnp.asarray(diffrax_stats.get("num_steps", zero), jnp.int32),
        newton_iters=zero,
        jac_refreshes=zero,
        factorizations=zero,
        linear_solves=zero,
        rhs_evals=zero,
        residual_evals=zero,
        active_set_iters=zero,
        active_set_changes=zero,
        ic_alg_residual_norm=jnp.asarray(0.0),
        t_reached=jnp.asarray(t_reached),
        status=result,
    )


def _recipe_mass(problem: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return Recipe mass values and pinned local-row indices."""
    source = problem.source
    local = np.asarray(source.mass_diagonal(), dtype=np.float64)
    global_mass = np.zeros((problem.n_lam,), dtype=np.float64)
    for entry in source.global_layout():
        if entry["mass"] == "differential":
            lo = int(entry["offset"])
            global_mass[lo : lo + int(entry["size"])] = 1.0
    bc = source.bc_dof_indices()
    bc_rows = np.asarray((), dtype=np.int32) if bc is None else np.asarray(bc, np.int32)
    return np.concatenate((local, global_mass)), bc_rows


def _direct_mass(problem: Any) -> np.ndarray:
    """Materialize a declared-constant direct MassProblem mass operator."""
    mass = problem.problem.mass
    dependencies = mass.dependencies
    if any(
        bool(getattr(dependencies, name)) for name in ("time", "state", "parameters", "control")
    ):
        raise ConfigurationError(
            "DiffraxStepper requires a constant identity mass operator, but the "
            "direct MassProblem declares runtime-dependent mass coefficients; "
            "use JNDF, JBDF, or JIDA for non-constant mass."
        )
    zeros = jnp.zeros((problem.n,), dtype=jnp.float64)
    dense = mass.dense(jnp.asarray(0.0), zeros, None, None)
    return np.asarray(dense, dtype=np.float64)


def _validate_eligibility(problem: Any) -> np.ndarray:
    """Validate identity-mass differential rows and return Dirichlet rows."""
    if not all(hasattr(problem, name) for name in ("problem", "n", "n_u", "n_lam")):
        raise ConfigurationError(
            "DiffraxStepper consumes the normalized pdesolve problem surface; "
            "pass a MassProblem or authored Recipe through diff_soup.pdesolve."
        )
    mass_problem = problem.problem
    if not all(hasattr(mass_problem, name) for name in ("mass", "rhs", "structure")):
        raise ConfigurationError(
            "DiffraxStepper requires a mass-form differential problem with identity "
            "mass; use JNDF, JBDF, or JIDA for a fully implicit residual."
        )

    source = problem.source
    if source is not None and callable(getattr(source, "mass_diagonal", None)):
        mode = source.mass_mode()
        if mode != "constant":
            raise ConfigurationError(
                "DiffraxStepper requires mass_mode() == 'constant' and unit mass "
                f"on differential rows, but the problem reports {mode!r}; use "
                "JNDF, JBDF, or JIDA for non-constant mass."
            )
        mass, bc_rows = _recipe_mass(problem)
        matrix_is_identity = True
    else:
        matrix = _direct_mass(problem)
        mass = np.diag(matrix)
        bc_rows = np.asarray((), dtype=np.int32)
        matrix_is_identity = bool(np.array_equal(matrix, np.eye(problem.n)))

    pinned = np.zeros((problem.n,), dtype=bool)
    pinned[bc_rows] = True
    differential = np.asarray(problem.id_mask, dtype=bool)
    algebraic_local = int(np.sum(~differential[: problem.n_u] & ~pinned[: problem.n_u]))
    algebraic_global = int(np.sum(~differential[problem.n_u :]))
    differential_rows = differential & ~pinned
    nonunit = int(np.sum(mass[differential_rows] != 1.0))
    if algebraic_local or algebraic_global or nonunit or not matrix_is_identity:
        bound_note = (
            "; bound-control closing rows are algebraic globals"
            if problem.has_recipe_controls and algebraic_global
            else ""
        )
        coupled = 0 if matrix_is_identity else 1
        raise ConfigurationError(
            "DiffraxStepper eligibility requires unit identity mass on every "
            "differential row, with only Dirichlet pins outside that set; found "
            f"{algebraic_local} algebraic local row(s), {algebraic_global} "
            f"algebraic global row(s), {nonunit} non-unit differential row(s), "
            f"and {coupled} non-identity mass operator(s){bound_note}. Use "
            "JNDF, JBDF, or JIDA for algebraic or general-mass structure."
        )
    return bc_rows


def _boundary_residual(problem: Any, theta: Any, state: Any, time: Any) -> jax.Array:
    """Evaluate the authored boundary residual with the solve's dynamic parameters."""
    source = problem.source
    if source is None:
        return jnp.zeros((problem.n_u,), dtype=jnp.asarray(state).dtype)
    dynamic = getattr(theta, "inner", theta)
    _, static = eqx.partition(source, eqx.is_inexact_array)
    concrete = eqx.combine(dynamic, static) if dynamic is not None else source
    residual = concrete.bc_residual(state[: problem.n_u], time)
    if residual is None:
        return jnp.zeros((problem.n_u,), dtype=jnp.asarray(state).dtype)
    return jnp.asarray(residual)


def _terminal_event_metadata(
    event: Any, event_mask: Any, problem: Any, y0: Any, t0: Any, theta: Any
):
    """Derive declaration-order metadata for Diffrax's single terminal stop."""
    declared = _events(event)
    if not declared:
        missing = jnp.asarray(-1, jnp.int32)
        return missing, missing, missing
    mask = jnp.stack([jnp.asarray(value) for value in jax.tree_util.tree_leaves(event_mask)])
    fired = jnp.any(mask)
    index = jnp.where(fired, jnp.argmax(mask).astype(jnp.int32), -1)
    kinds = jnp.asarray(
        [2 if item.root_finder is not None else 1 for item in declared], dtype=jnp.int32
    )
    initial_values = []
    for item in declared:
        value = item.cond_fn(
            t0,
            y0[: problem.n_u],
            theta,
            lambdas=y0[problem.n_u :],
            control=None,
        )
        initial_values.append(jnp.asarray(value))
    initial = jnp.stack(initial_values)
    continuous_directions = jnp.where(initial <= 0, 1, -1).astype(jnp.int32)
    directions = jnp.where(kinds == 1, 0, continuous_directions)
    safe_index = jnp.maximum(index, 0)
    return (
        index,
        jnp.where(fired, kinds[safe_index], -1),
        jnp.where(fired, directions[safe_index], -1),
    )


class DiffraxStepper(eqx.Module):
    """An explicit Diffrax solver run on a problem with identity mass.

    Every differential row needs unit mass, and the only other rows allowed are Dirichlet pins. Events can only terminate the solve. It needs the optional `diffrax` package.

    Attributes:
        inner: Diffrax solver instance, `diffrax.Tsit5()` when `None`.
        rtol: Relative tolerance of the default `diffrax.PIDController`.
        atol: Absolute tolerance of the default `diffrax.PIDController`.
        dt0: Initial step size, chosen by the step size controller when `None`.
        max_steps: Step budget of a solve.
        adjoint: Differentiation strategy, mapped to the matching Diffrax adjoint, with `None` meaning `CheckpointAdjoint`.
    """

    inner: Optional[Any] = eqx.field(default=None)
    rtol: float = eqx.field(default=1e-6, static=True)
    atol: float = eqx.field(default=1e-6, static=True)
    dt0: Optional[float] = eqx.field(default=None, static=True)
    max_steps: int = eqx.field(default=10_000, static=True)
    adjoint: Optional[AbstractAdjoint] = eqx.field(default=None)

    def solve(
        self,
        problem: Any,
        t_span: tuple[float, float],
        y0: jax.Array,
        *,
        saveat: Optional[SaveAt] = None,
        adjoint: Any = None,
        theta: Any = None,
        lambda0: Any = None,
        trace: bool = False,
        event: Any = None,
        control_state0: Any = None,
        consistent_ic: bool = True,
        dt0: Optional[float] = None,
        max_steps: Optional[int] = None,
        tolerances: Any = None,
        linear: Any = None,
        **kwargs: Any,
    ) -> Solution:
        """Solve `problem` over `t_span` from `y0` with the wrapped Diffrax solver.

        Options that need `JNDF`, `JBDF`, or `JIDA` raise `ConfigurationError`, among them `SaveAt(dense=True)` and an integer `SaveAt(steps=...)`.

        Args:
            problem: Problem as normalized by `pdesolve`.
            t_span: Pair `(t0, t1)`.
            y0: Initial state.
            saveat: Output plan, a `SaveAt`.
            adjoint: Differentiation strategy, overriding the stepper's own.
            theta: Parameter pytree, the problem's defaults when `None`.
            lambda0: Not supported, and must stay `None`.
            trace: Not supported, and must stay `False`.
            event: Terminating `Event` or `EventSet`, whose continuous events need `direction="any"`.
            control_state0: Not supported, and must stay `None`.
            consistent_ic: If `True`, move Dirichlet-pinned rows of `y0` onto their boundary values first.
            dt0: Initial step size, overriding the stepper's.
            max_steps: Step budget, overriding the stepper's.
            tolerances: Not supported, since the stepper's `rtol` and `atol` apply.
            linear: Not supported, and must stay `None`.
            **kwargs: Passed to `diffrax.diffeqsolve`, for example `stepsize_controller=`.

        Returns:
            A `Solution`, whose `result` reports a failed solve instead of raising.
        """
        diffrax = _require_diffrax()
        bc_rows = _validate_eligibility(problem)
        if lambda0 is not None:
            raise ConfigurationError(
                "DiffraxStepper does not accept a separate lambda0 seed; include "
                "eligible differential globals in y0 or use a native DAE engine."
            )
        if trace:
            raise ConfigurationError(
                "DiffraxStepper does not emit the native per-step trace; pass "
                "trace=False or use JNDF, JBDF, or JIDA."
            )
        if control_state0 is not None:
            raise ConfigurationError(
                "DiffraxStepper is terminate-only and cannot carry control_state0; "
                "omit it or use JNDF, JBDF, or JIDA for protocol switching."
            )
        if tolerances is not None:
            raise ConfigurationError(
                "DiffraxStepper uses its scalar rtol/atol and cannot consume a "
                "row-resolved tolerances record; configure rtol/atol on the stepper."
            )
        if linear is not None:
            raise ConfigurationError(
                "DiffraxStepper is explicit and has no native linear-backend slot; "
                "omit linear or use an implicit native engine."
            )
        native_only = sorted(
            name for name in ("hmin", "hmax", "max_order", "max_events") if name in kwargs
        )
        if native_only:
            raise ConfigurationError(
                "DiffraxStepper cannot honor native-engine option(s) "
                f"{', '.join(native_only)}; remove them or select JNDF, JBDF, or JIDA."
            )
        if saveat is not None and not isinstance(saveat, SaveAt):
            raise ConfigurationError(
                "DiffraxStepper expects diff_soup.SaveAt so every backend shares "
                "one output contract; replace the Diffrax SaveAt declaration."
            )
        if saveat is not None and saveat.dense:
            raise ConfigurationError(
                "DiffraxStepper cannot expose Diffrax's dense object through the "
                "shared interpolation record; pass SaveAt(dense=False)."
            )
        if saveat is not None and saveat.output is not None and saveat.output.mode == "projected":
            raise ConfigurationError(
                "DiffraxStepper cannot apply projected output sampling; use the "
                "default fast output policy or a native engine."
            )
        raw_steps = False if saveat is None else saveat.steps
        if isinstance(raw_steps, (int, np.integer)) and not isinstance(raw_steps, bool):
            raise ConfigurationError(
                "DiffraxStepper supports accepted-step output only as "
                "SaveAt(steps=True); integer strides have no native Diffrax "
                "analogue. Use the boolean form or select JNDF, JBDF, or JIDA."
            )

        inner = self.inner if self.inner is not None else diffrax.Tsit5()
        if not isinstance(inner, diffrax.AbstractSolver):
            raise ConfigurationError(
                "DiffraxStepper.inner must be a diffrax.AbstractSolver instance; "
                f"got {type(inner).__name__}."
            )
        effective_adjoint = adjoint if adjoint is not None else self.adjoint
        mapped_adjoint = _map_adjoint(effective_adjoint, diffrax)
        theta = problem.default_theta() if theta is None else theta

        # The problem's precision owns the dtype of the solve: the state and
        # the floating parameter leaves are cast to its compute dtype.
        precision = resolve_precision(problem)
        warn_if_float32_on_cpu(precision, y0, theta, t_span)
        state0 = cast_to_compute_dtype(
            problem.to_internal_state(y0), precision, what="initial state"
        )
        theta = cast_to_compute_dtype(theta, precision, what="parameters")
        dtype = jnp.asarray(state0).dtype
        t0 = jnp.asarray(t_span[0], dtype=dtype)
        t1 = jnp.asarray(t_span[1], dtype=dtype)
        bc_index = jnp.asarray(bc_rows, dtype=jnp.int32)
        if bc_rows.size and consistent_ic:
            residual0 = _boundary_residual(problem, theta, state0, t0)
            state0 = state0.at[bc_index].add(-residual0[bc_index])

        def vector_field(time, state, args):
            field = problem.problem.rhs(time, state, args, None)
            if not bc_rows.size:
                return field

            def residual_at_time(candidate_time):
                return _boundary_residual(problem, args, state, candidate_time)

            _, time_tangent = jax.jvp(
                residual_at_time,
                (time,),
                (jnp.ones((), dtype=jnp.asarray(time).dtype),),
            )
            return field.at[bc_index].set(-time_tangent[bc_index])

        step_budget = self.max_steps if max_steps is None else max_steps
        plan = _build_save_plan(
            saveat,
            t0,
            t1,
            dtype,
            event=event,
            max_steps=step_budget,
            problem=problem,
        )
        output_plan = plan.outputs
        n_u = int(problem.n_u)
        if output_plan.requested and isinstance(mapped_adjoint, diffrax.BacksolveAdjoint):
            raise ConfigurationError(
                "SaveAt.outputs is not available with BacksolveAdjoint on DiffraxStepper "
                "(Diffrax's backsolve adjoint saves the state only); use "
                "CheckpointAdjoint or ForwardAdjoint, or pass outputs=None."
            )

        def emit_outputs(time, state, args):
            # The named outputs of one state; ``args`` is the parameter pytree.
            emit = output_emitter(output_plan, problem, args, dtype)
            lam = state[n_u:] if output_plan.needs_lam else None
            return emit(time, state[:n_u], lam)

        if plan.step_every:
            # Diffrax owns the accepted-step buffer and its exact event row;
            # the shared plan only supplies the public static capacity.
            sampling = dict(
                steps=True,
                t0=plan.save_initial,
                t1=bool(saveat is not None and saveat.t1),
            )
        else:
            query_host = np.asarray(plan.query_ts)
            query_has_t1 = bool(query_host.size and query_host[-1] == float(np.asarray(t1)))
            include_terminal = bool(event is not None or not query_has_t1)
            # A single SubSaveAt is required by Diffrax's BacksolveAdjoint.
            # Asking for t1 when it is not already a query gives us the exact
            # stopping point for Solution.terminal; that internal row is
            # discarded from the public grid unless an event uses the plan's
            # terminal reserve.
            sampling = dict(ts=plan.query_ts, t1=include_terminal)
        if output_plan.requested:
            # The named outputs at the sample times, and the exact stopping
            # state for Solution.terminal.
            diffrax_saveat = diffrax.SaveAt(
                subs=[
                    diffrax.SubSaveAt(**sampling, fn=emit_outputs),
                    diffrax.SubSaveAt(t1=True),
                ]
            )
        else:
            diffrax_saveat = diffrax.SaveAt(**sampling)
        diffrax_event = _map_events_to_diffrax(event, diffrax, problem)
        controller = kwargs.pop(
            "stepsize_controller", diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        )
        kwargs.setdefault("throw", False)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            inner,
            t0=t0,
            t1=t1,
            dt0=self.dt0 if dt0 is None else dt0,
            y0=state0,
            args=theta,
            stepsize_controller=controller,
            saveat=diffrax_saveat,
            max_steps=step_budget,
            adjoint=mapped_adjoint,
            event=diffrax_event,
            **kwargs,
        )

        result = _map_result(solution.result, diffrax)

        def finite_rows(rows):
            finite = jnp.ones((rows[0].shape[0],), dtype=bool)
            for row in rows:
                finite = jnp.logical_and(
                    finite, jnp.all(jnp.isfinite(row.reshape(row.shape[0], -1)), axis=1)
                )
            return finite

        if output_plan.requested:
            raw_ts = jnp.asarray(solution.ts[0])
            raw_rows = tuple(jnp.asarray(row) for row in solution.ys[0])
            raw_valid = jnp.logical_and(jnp.isfinite(raw_ts), finite_rows(raw_rows))
            terminal_t = jnp.asarray(solution.ts[1])[0]
            terminal_y = jnp.asarray(solution.ys[1])[0]
        else:
            raw_ts = jnp.asarray(solution.ts)
            raw_rows = (jnp.asarray(solution.ys),)
            raw_valid = jnp.logical_and(jnp.isfinite(raw_ts), finite_rows(raw_rows))
            raw_count = jnp.sum(raw_valid.astype(jnp.int32), dtype=jnp.int32)
            terminal_slot = jnp.maximum(raw_count - 1, 0)
            terminal_t = raw_ts[terminal_slot]
            terminal_y = raw_rows[0][terminal_slot]

        def masked(values, valid):
            shape = (values.shape[0],) + (1,) * (values.ndim - 1)
            return jnp.where(valid.reshape(shape), values, jnp.nan)

        output_ts = jnp.full((plan.capacity,), jnp.nan, dtype=raw_ts.dtype)
        output_rows = tuple(
            jnp.full((plan.capacity,) + row.shape[1:], jnp.nan, dtype=row.dtype) for row in raw_rows
        )
        if plan.step_every:
            raw_capacity = int(raw_ts.shape[0])
            output_ts = output_ts.at[:raw_capacity].set(jnp.where(raw_valid, raw_ts, jnp.nan))
            output_rows = tuple(
                out.at[:raw_capacity].set(masked(row, raw_valid))
                for out, row in zip(output_rows, raw_rows)
            )
        else:
            count = plan.requested_count
            requested_valid = raw_valid[:count]
            output_ts = output_ts.at[:count].set(
                jnp.where(requested_valid, raw_ts[:count], jnp.nan)
            )
            output_rows = tuple(
                out.at[:count].set(masked(row[:count], requested_valid))
                for out, row in zip(output_rows, raw_rows)
            )

        event_stop = result == RESULT.EVENT_STOP

        def write_event_output(payload):
            times, rows = payload
            valid = raw_valid[: plan.capacity]
            return (
                times.at[: plan.capacity].set(jnp.where(valid, raw_ts[: plan.capacity], jnp.nan)),
                tuple(
                    out.at[: plan.capacity].set(masked(row[: plan.capacity], valid))
                    for out, row in zip(rows, raw_rows)
                ),
            )

        if not plan.step_every:
            output_ts, output_rows = jax.lax.cond(
                event_stop,
                write_event_output,
                lambda payload: payload,
                (output_ts, output_rows),
            )
        valid_mask = jnp.logical_and(jnp.isfinite(output_ts), finite_rows(output_rows))
        valid_count = jnp.sum(valid_mask.astype(jnp.int32), dtype=jnp.int32)
        event_index, event_kind, event_direction = _terminal_event_metadata(
            event, solution.event_mask, problem, state0, t0, theta
        )
        last_slot = jnp.maximum(valid_count - 1, 0)
        missing = jnp.full((plan.capacity,), -1, dtype=jnp.int32)
        event_indices = jax.lax.cond(
            event_stop,
            lambda values: values.at[last_slot].set(event_index),
            lambda values: values,
            missing,
        )
        event_kinds = jax.lax.cond(
            event_stop,
            lambda values: values.at[last_slot].set(event_kind),
            lambda values: values,
            missing,
        )
        event_directions = jax.lax.cond(
            event_stop,
            lambda values: values.at[last_slot].set(event_direction),
            lambda values: values,
            missing,
        )
        result_code = _result_code(result)
        result_codes = jax.lax.cond(
            event_stop,
            lambda values: values.at[last_slot].set(result_code),
            lambda values: values,
            missing,
        )
        terminal = TerminalRecord.empty(problem.n_u, problem.n_lam, dtype=dtype).write(
            t=terminal_t,
            y=terminal_y[: problem.n_u],
            lam=terminal_y[problem.n_u :],
            result_code=result_code,
            control_epoch=0,
            event_index=jnp.where(event_stop, event_index, -1),
            event_kind=jnp.where(event_stop, event_kind, -1),
            event_direction=jnp.where(event_stop, event_direction, -1),
        )
        stats = _map_stats(solution.stats, terminal_t, result)
        if output_plan.requested:
            by_name = dict(zip(output_plan.names, output_rows))
            ys = by_name.get("state")
            lambdas = by_name.get("lambdas") if problem.n_lam else None
            saved_outputs = SavedOutputs(output_plan.names, output_rows)
        else:
            (output_ys,) = output_rows
            ys = output_ys[:, :n_u]
            lambdas = output_ys[:, n_u:] if problem.n_lam else None
            saved_outputs = None
        return build_solution(
            ts=output_ts,
            ys=ys,
            lambdas=lambdas,
            outputs=saved_outputs,
            valid_mask=valid_mask,
            valid_count=valid_count,
            control_epochs=jnp.where(valid_mask, 0, -1),
            event_indices=event_indices,
            event_kinds=event_kinds,
            event_directions=event_directions,
            result_codes=result_codes,
            stats=stats,
            result=result,
            info={"diffrax": dict(solution.stats)},
            event_mask=solution.event_mask,
            terminal=terminal,
            alg=self,
            prob=problem,
        )
