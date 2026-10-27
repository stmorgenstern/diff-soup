"""Output-sampling policy for the timestepping suite.

[SaveAt][diff_soup.timestepping.saveat.SaveAt] and [SubSaveAt][diff_soup.timestepping.saveat.SubSaveAt] declare *when* and *what* a solve
records into its [Solution][diff_soup.timestepping.vsvo.solution.Solution]. The API mirrors Diffrax's ``SaveAt`` so
Diffrax-style code transfers with minimal changes. This module is the single
definition consumed across the solver suite; ``ds.SaveAt`` /
``ds.SubSaveAt`` re-export these names.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError


def _hash_optional_array(arr) -> int:
    """Content-hash for a possibly-None jax/numpy array.

    Used by SaveAt / _SavePlan to provide a stable hash even when ``ts``
    is a concrete JAX array. Bytes-based when concrete; falls back to
    ``(shape, dtype, id)`` when the array is a JAX tracer (e.g. when
    SaveAt is constructed inside ``jax.jit`` / ``jax.grad``).
    """
    if arr is None:
        return 0
    if isinstance(arr, jax.core.Tracer):
        return hash((arr.shape, arr.dtype.str, id(arr)))
    return hash((arr.shape, arr.dtype.str, np.asarray(arr).tobytes()))


@dataclass(frozen=True)
class DenseOutputPolicy:
    """How the algebraic rows of interpolated samples are computed.

    Pass it as `SaveAt(output=...)` or `Event(state_policy=...)`.

    Args:
        mode: `"fast"` interpolates every row, and `"projected"` projects each sample's algebraic rows onto the constraints.
    """

    mode: str = "fast"

    def __post_init__(self) -> None:
        """Validate the sampling-mode name."""
        if self.mode not in ("fast", "projected"):
            raise ValueError(f"dense-output mode must be 'fast' or 'projected'; got {self.mode!r}.")

    @property
    def local_rows(self) -> str:
        """Consistency label for local rows: always ``"interpolated"``."""
        return "interpolated"

    @property
    def global_rows(self) -> str:
        """Consistency label for algebraic globals."""
        return "interpolated" if self.mode == "fast" else "projected"


@dataclass(frozen=True)
class SubSaveAt:
    """A Diffrax-style save request for one output, which no solver accepts.

    `SaveAt(subs=...)` raises `ConfigurationError`. Name several stored quantities with `SaveAt(outputs=...)` instead.
    """

    t0: bool = False
    t1: bool = False
    ts: Optional[jax.Array] = None
    steps: bool | int = False
    fn: Optional[Callable[..., Any]] = None

    def __hash__(self) -> int:
        """Hash over the static save flags + ``fn`` identity (cache key)."""
        return hash(
            (
                self.t0,
                self.t1,
                _hash_optional_array(self.ts),
                self.steps,
                id(self.fn) if self.fn is not None else 0,
            )
        )

    def __eq__(self, other) -> bool:
        """Equality by hash (static fields + ``fn`` identity)."""
        if not isinstance(other, SubSaveAt):
            return NotImplemented
        return hash(self) == hash(other)


@dataclass(frozen=True)
class SaveAt:
    """Determines which times and quantities a solve saves.

    Pass it as `saveat=` to `pdesolve`. With nothing requested the solve saves the final state only. Buffers keep a fixed size, so unfilled rows read `NaN`.

    Args:
        t0: If `True`, save at the start of the interval.
        t1: If `True`, save at the end of the interval.
        ts: Times inside the interval, sorted and deduplicated before the solve, and not combined with `steps`.
        steps: `True` saves every accepted step and an integer `k` every `k`-th, into a buffer sized from `max_steps`.
        dense: If `True`, record the interpolant for `Solution.evaluate`.
        fn: Not supported, and must stay `None`.
        subs: Not supported, and must stay `None`.
        solver_state: Not supported, and must stay `False`.
        controller_state: Not supported, and must stay `False`.
        made_jump: Not supported, and must stay `False`.
        output: A `DenseOutputPolicy` for the algebraic rows of interpolated samples.
        ordered: If `True`, use `ts` as given, strictly increasing and possibly traced, without `t0`, `t1`, or `steps`.
        outputs: Names to store in place of the state, each a declared observation, a declared field, `"state"`, or `"lambdas"`.
    """

    t0: bool = False
    t1: bool = False
    ts: Optional[jax.Array] = None
    steps: bool | int = False
    dense: bool = False
    fn: Optional[Callable[..., Any]] = None
    subs: Any = None
    solver_state: bool = False
    controller_state: bool = False
    made_jump: bool = False
    output: Optional[DenseOutputPolicy] = None
    ordered: bool = False
    outputs: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        """Normalize ``outputs`` to a tuple of distinct names."""
        if self.outputs is None:
            return
        names = self.outputs
        if isinstance(names, str) or not all(isinstance(name, str) for name in names):
            raise ConfigurationError(
                f"SaveAt.outputs takes a tuple of output names such as ('voltage',); got {names!r}."
            )
        names = tuple(names)
        if not names:
            raise ConfigurationError(
                "SaveAt.outputs names at least one output; pass outputs=None to store "
                "the full state."
            )
        if len(set(names)) != len(names):
            raise ConfigurationError(f"SaveAt.outputs names each output once; got {names!r}.")
        object.__setattr__(self, "outputs", names)

    def __hash__(self) -> int:
        """Hash over the static save flags + ``fn`` / ``subs`` identity (cache key)."""
        # ``subs`` is a pytree of SubSaveAt instances or None; we hash by id
        # since users typically pass a constant subs tree across calls.
        return hash(
            (
                self.t0,
                self.t1,
                _hash_optional_array(self.ts),
                self.steps,
                self.dense,
                id(self.fn) if self.fn is not None else 0,
                id(self.subs) if self.subs is not None else 0,
                self.solver_state,
                self.controller_state,
                self.made_jump,
                self.output,
                self.ordered,
                self.outputs,
            )
        )

    def __eq__(self, other) -> bool:
        """Equality by hash (static fields + ``fn`` / ``subs`` identity)."""
        if not isinstance(other, SaveAt):
            return NotImplemented
        return hash(self) == hash(other)


@dataclass(frozen=True)
class OutputSpec:
    """One stored output of a solve: what is stored at every save time.

    Attributes:
        name: The name the output was requested by.
        kind: ``"observable"`` (a declared observation), ``"field"`` (a
            declared field's slice of the internal state), ``"state"`` (the
            full internal state) or ``"lambdas"`` (the bordered globals).
        shape: The shape of one sample.
        needs_lam: Whether evaluating the output reads the bordered globals.
        start: For a field, the first index of its slice in the state.
        stop: For a field, one past the last index of its slice.
    """

    name: str
    kind: str
    shape: tuple[int, ...]
    needs_lam: bool = False
    start: int = 0
    stop: int = 0


@dataclass(frozen=True)
class OutputPlan:
    """The static output layout of a solve: one [OutputSpec][diff_soup.timestepping.saveat.OutputSpec] per stored output.

    Attributes:
        specs: The stored outputs, in request order.
        requested: ``True`` when the outputs were named with
            ``SaveAt(outputs=...)``; ``False`` for the default layout (the
            state and the bordered globals, reported as ``Solution.ys`` and
            ``Solution.lambdas``).
    """

    specs: tuple[OutputSpec, ...]
    requested: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        """The output names, in request order."""
        return tuple(spec.name for spec in self.specs)

    @property
    def needs_lam(self) -> bool:
        """Whether any output reads the bordered globals."""
        return any(spec.needs_lam for spec in self.specs)

    @property
    def observable_names(self) -> tuple[str, ...]:
        """The names of the requested observations, in request order."""
        return tuple(spec.name for spec in self.specs if spec.kind == "observable")

    @property
    def observables_need_lam(self) -> bool:
        """Whether any requested observation reads a bordered global."""
        return any(spec.needs_lam for spec in self.specs if spec.kind == "observable")


def default_output_plan(n_u: int, n_lam: int) -> OutputPlan:
    """The default layout: the full state and the bordered globals."""
    return OutputPlan(
        specs=(
            OutputSpec("state", "state", (int(n_u),)),
            OutputSpec("lambdas", "lambdas", (int(n_lam),), needs_lam=True),
        ),
        requested=False,
    )


def _declared_fields(source: Any) -> tuple[tuple[str, str, int, int], ...]:
    """``(component, field, start, stop)`` for every field the problem declares."""
    entries = getattr(source, "_layout_entries", ()) or ()
    return tuple(
        (str(owner), str(name), int(offset), int(offset) + int(size))
        for (owner, name), offset, size, _shape in entries
    )


def resolve_outputs(outputs: Optional[tuple[str, ...]], problem: Any) -> OutputPlan:
    """Resolve ``SaveAt.outputs`` against a problem into its static output layout.

    Each name resolves, in this order, to a declared observation, a declared
    field (``"c_e"``, or ``"component.c_e"`` to pick one of several
    components declaring ``c_e``), ``"state"`` or ``"lambdas"``.

    Args:
        outputs: The requested names, or ``None`` for the default layout.
        problem: The problem being solved: a lowered recipe problem, or a
            solver view of one (its ``source`` holds the declarations).

    Returns:
        The output layout.

    Raises:
        ConfigurationError: If a name is none of the above, or names a field
            that several components declare without the component.
    """
    n_u = int(problem.n_u)
    n_lam = int(getattr(problem, "n_lam", 0) or 0)
    if outputs is None:
        return default_output_plan(n_u, n_lam)
    source = getattr(problem, "source", None) or problem
    observables = {obs.name: obs for obs in (getattr(source, "observables", None) or ())}
    fields = _declared_fields(source)
    specs = []
    for name in outputs:
        if name in observables:
            record = observables[name]
            specs.append(
                OutputSpec(
                    name,
                    "observable",
                    tuple(int(d) for d in record.shape),
                    needs_lam=any(read[0] == "global" for read in record.reads),
                )
            )
            continue
        matches = [entry for entry in fields if entry[1] == name]
        if not matches and "." in name:
            component, _, field = name.partition(".")
            matches = [entry for entry in fields if entry[0] == component and entry[1] == field]
        if len(matches) > 1:
            spellings = sorted(f"{owner}.{field}" for owner, field, _a, _b in matches)
            raise ConfigurationError(
                f"SaveAt.outputs names the field {name!r}, which components "
                f"{sorted(owner for owner, *_rest in matches)!r} all declare; name one of "
                f"{spellings!r}."
            )
        if matches:
            _owner, _field, start, stop = matches[0]
            specs.append(OutputSpec(name, "field", (stop - start,), start=start, stop=stop))
            continue
        if name == "state":
            specs.append(OutputSpec(name, "state", (n_u,)))
            continue
        if name == "lambdas":
            specs.append(OutputSpec(name, "lambdas", (n_lam,), needs_lam=True))
            continue
        field_names = sorted({field for _owner, field, _a, _b in fields})
        raise ConfigurationError(
            f"SaveAt.outputs names {name!r}, which is not a declared observation or "
            f"field of this problem; name a declared observation {list(observables)!r}, "
            f"a declared field {field_names!r}, 'state' or 'lambdas'."
        )
    return OutputPlan(specs=tuple(specs), requested=True)


def output_emitter(plan: OutputPlan, problem: Any, theta: Any, dtype: Any) -> Callable:
    """``emit(t, u, lam) -> tuple`` evaluating the outputs of ``plan`` at one state.

    Observations evaluate through the recipe problem's ``observe`` with the
    solve's parameters ``theta`` (the explicit parameter channel the solve
    integrates against); a field is its slice of the internal state;
    ``"state"`` and ``"lambdas"`` pass the internal state and the bordered
    globals through. ``lam`` may be ``None`` when no output reads the globals.
    Each value is cast to ``dtype`` and shaped as its
    [OutputSpec][diff_soup.timestepping.saveat.OutputSpec] records.
    """
    names = plan.observable_names
    observer = None
    if names:
        source = getattr(problem, "source", None) or problem
        _dynamic, static = eqx.partition(source, eqx.is_inexact_array)
        observer = eqx.combine(getattr(theta, "inner", theta), static)
    pass_lam = plan.observables_need_lam

    def emit(t, u, lam):
        observed = observer.observe(u, t, lam if pass_lam else None, names=names) if names else {}
        rows = []
        for spec in plan.specs:
            if spec.kind == "observable":
                value = observed[spec.name]
            elif spec.kind == "field":
                value = u[spec.start : spec.stop]
            elif spec.kind == "state":
                value = u
            else:
                value = lam
            rows.append(jnp.asarray(value, dtype=dtype).reshape(spec.shape))
        return tuple(rows)

    return emit


@dataclass(frozen=True, eq=False)
class _SavePlan:
    """Canonical fixed-buffer layout for one explicit-output request.

    ``query_ts`` is the sorted, deduplicated grid the stepper evaluates. Its
    length is ``requested_count``. ``output_ts`` is the all-``NaN`` time
    buffer with structural ``capacity``; the driver fills it as requested
    samples become valid. Keeping query and output buffers separate lets an
    exact early terminal sample occupy the next prefix slot rather than a
    fixed tail position. The reserve is present whether or not an event fires,
    so compiled output shapes do not depend on runtime event satisfaction.

    This record is the sole owner of save-grid capacity. Engines may choose
    different write mechanisms, but must consume this layout rather than
    independently recounting ``SaveAt`` fields.

    Attributes:
        query_ts: Concrete canonical query grid, shape ``(N,)``.
        output_ts: Empty fixed-capacity time buffer, shape ``(capacity,)``,
            initialized entirely to ``NaN``.
        requested_count: Number ``N`` of canonical explicit samples.
        capacity: ``N + 1`` when a terminating event is declared, otherwise
            ``N``.
        has_terminal_reserve: Whether the final structural slot is reserved
            for an exact terminal event sample.
        terminal_slot: Reserve index ``N``, or ``-1`` when there is no reserve.
        step_every: Accepted-step stride, or ``0`` for an explicit query grid.
        save_initial: Whether an accepted-step plan starts with the state at
            ``t0``.
        outputs: The resolved [OutputPlan][diff_soup.timestepping.saveat.OutputPlan],
            or ``None`` when the plan was built without a problem.
    """

    query_ts: jax.Array
    output_ts: jax.Array
    requested_count: int
    capacity: int
    has_terminal_reserve: bool
    terminal_slot: int
    step_every: int = 0
    save_initial: bool = False
    outputs: Optional[OutputPlan] = None

    @classmethod
    def build(
        cls,
        saveat: Optional[SaveAt],
        t0: Any,
        t1: Any,
        dtype: Any,
        *,
        event: Any = None,
        has_terminating_event: Optional[bool] = None,
        max_steps: Optional[int] = None,
        problem: Any = None,
    ) -> "_SavePlan":
        """Build the deterministic query and output layout.

        Explicit ``ts`` and requested endpoints are canonicalized together:
        they are sorted and exact duplicates are removed. An omitted output
        request retains the solver-suite convention of saving ``t1``. Save
        times and interval endpoints must be concrete because they determine a
        compiled buffer shape.

        Args:
            saveat: Output request, or ``None`` for the terminal-only default.
            t0: Integration start time.
            t1: Integration end time.
            dtype: Floating dtype for the returned JAX arrays.
            event: Optional event or event set. A member whose ``affect`` is
                ``None`` is terminating and requests the structural reserve.
            has_terminating_event: Optional explicit reserve decision for an
                adapter that has already classified its event set.
            max_steps: Static attempt budget used to size accepted-step output.
                Required when ``SaveAt.steps`` is enabled.
            problem: The problem being solved, against which ``SaveAt.outputs``
                resolves (see
                [resolve_outputs][diff_soup.timestepping.saveat.resolve_outputs]).
                Without it the plan carries no output layout, and naming
                ``outputs`` is refused.

        Returns:
            The single canonical fixed-buffer save plan.

        Raises:
            ConfigurationError: If ``solver_state``, ``controller_state``, or
                ``made_jump`` carries anything but its default ``False``; if
                ``fn`` or ``subs`` is given; if ``outputs`` names an unknown
                output; or if accepted-step output is malformed, combined with
                explicit ``ts``, or requested by an adapter that supplied no
                step budget.
            ValueError: If a shape-defining time is traced, non-scalar interval
                endpoints are supplied, ``t1 < t0``, or requested times are
                non-finite or outside the integration interval.
        """
        reserved = (
            tuple(
                name
                for name in ("solver_state", "controller_state", "made_jump")
                if getattr(saveat, name, False) is not False
            )
            if saveat is not None
            else ()
        )
        if reserved:
            fields = ", ".join(f"SaveAt.{name}" for name in reserved)
            if len(reserved) == 1:
                raise ConfigurationError(
                    f"{fields} is reserved and no backend consumes it; leave it at "
                    "its default (False) to satisfy this check."
                )
            raise ConfigurationError(
                f"{fields} are reserved and no backend consumes them; leave them at "
                "their defaults (False) to satisfy this check."
            )
        if saveat is not None and saveat.fn is not None:
            raise ConfigurationError(
                "SaveAt.fn is not supported; name the quantities to store with "
                "SaveAt(outputs=(...)) instead: declared observations, declared fields, "
                "'state' or 'lambdas'."
            )
        if saveat is not None and saveat.subs is not None:
            raise ConfigurationError(
                "SaveAt.subs is not supported by the fixed-buffer solver adapters; "
                "omit subs and name several stored quantities with SaveAt(outputs=(...))."
            )
        requested_outputs = getattr(saveat, "outputs", None) if saveat is not None else None
        if problem is not None:
            output_plan = resolve_outputs(requested_outputs, problem)
        elif requested_outputs is not None:
            raise ConfigurationError(
                "SaveAt.outputs needs the problem to resolve its names against; this "
                "solver adapter does not support outputs. Pass outputs=None."
            )
        else:
            output_plan = None

        raw_steps = getattr(saveat, "steps", False) if saveat is not None else False
        if raw_steps is False:
            step_every = 0
        elif raw_steps is True:
            step_every = 1
        elif isinstance(raw_steps, (int, np.integer)) and not isinstance(raw_steps, bool):
            step_every = int(raw_steps)
            if step_every <= 0:
                raise ConfigurationError(
                    "SaveAt.steps must be True or a positive integer stride; "
                    f"got {raw_steps!r}. Leave it False to disable accepted-step output."
                )
        else:
            raise ConfigurationError(
                "SaveAt.steps must be False, True, or a positive integer stride; "
                f"got {raw_steps!r}."
            )
        raw_ts = getattr(saveat, "ts", None) if saveat is not None else None
        if step_every and raw_ts is not None:
            raise ConfigurationError(
                "SaveAt.steps cannot be combined with explicit SaveAt.ts in one "
                "output buffer; choose accepted-step output or an explicit time grid."
            )

        want_t0 = bool(getattr(saveat, "t0", False)) if saveat is not None else False
        want_t1 = bool(getattr(saveat, "t1", False)) if saveat is not None else False
        ordered = bool(getattr(saveat, "ordered", False)) if saveat is not None else False
        if ordered and (raw_ts is None or step_every or want_t0 or want_t1):
            raise ConfigurationError(
                "SaveAt(ordered=True) takes every save time from ts: pass ts, and leave "
                "t0, t1 and steps at their defaults (put an endpoint into ts instead)."
            )

        def terminal_reserve() -> bool:
            if has_terminating_event is not None:
                return bool(has_terminating_event)
            members = () if event is None else tuple(getattr(event, "events", (event,)))
            return any(getattr(member, "affect", None) is None for member in members)

        if step_every:
            if max_steps is None:
                raise ConfigurationError(
                    "SaveAt.steps needs a static step budget to size its output; "
                    "use JNDF, JBDF, or JIDA, or use boolean steps=True with "
                    "DiffraxStepper."
                )
            budget = int(max_steps)
            if budget <= 0:
                raise ConfigurationError(
                    f"accepted-step output needs max_steps to be positive; got {budget}."
                )
            reserve = terminal_reserve()
            save_initial = bool(getattr(saveat, "t0", False))
            step_capacity = (budget + step_every - 1) // step_every
            capacity = step_capacity + int(save_initial) + int(reserve)
            query_ts = jnp.empty((0,), dtype=dtype)
            return cls(
                query_ts=query_ts,
                output_ts=jnp.full((capacity,), jnp.nan, dtype=dtype),
                requested_count=0,
                capacity=capacity,
                has_terminal_reserve=reserve,
                terminal_slot=capacity - 1 if reserve else -1,
                step_every=step_every,
                save_initial=save_initial,
                outputs=output_plan,
            )

        if ordered:
            if jnp.ndim(raw_ts) != 1:
                raise ValueError("SaveAt.ts must be a one-dimensional array of times")
            if not isinstance(raw_ts, jax.core.Tracer):
                ordered_host = np.asarray(raw_ts, dtype=np.dtype(jnp.dtype(dtype)))
                if not np.all(np.isfinite(ordered_host)):
                    raise ValueError("SaveAt times must all be finite")
                if np.any(np.diff(ordered_host) <= 0):
                    raise ValueError(
                        "SaveAt(ordered=True) needs strictly increasing times in the dtype "
                        "of the solve; pass ordered=False to have them sorted and "
                        "deduplicated."
                    )
            query_ts = jnp.asarray(raw_ts, dtype=dtype)
            requested_count = int(query_ts.shape[0])
            reserve = terminal_reserve()
            capacity = requested_count + int(reserve)
            return cls(
                query_ts=query_ts,
                output_ts=jnp.full((capacity,), jnp.nan, dtype=query_ts.dtype),
                requested_count=requested_count,
                capacity=capacity,
                has_terminal_reserve=reserve,
                terminal_slot=requested_count if reserve else -1,
                outputs=output_plan,
            )

        endpoint_traced = any(isinstance(value, jax.core.Tracer) for value in (t0, t1))
        if isinstance(raw_ts, jax.core.Tracer):
            raise ValueError(
                "SaveAt.ts must be concrete because sorting and deduplication determine "
                "the fixed output shape; construct SaveAt outside the transformed solve, "
                "or pass ordered=True for times that are already strictly increasing."
            )
        if endpoint_traced:
            # A runtime span does not make the ordinary endpoint-only layout
            # dynamic: its flags determine the shape, while the endpoint values
            # remain traced data. Explicit times are likewise shape-stable when
            # no endpoint flag can collide with them. A flagged endpoint plus an
            # explicit grid needs concrete values so exact deduplication can
            # determine N before compilation.
            if raw_ts is not None and (want_t0 or want_t1):
                raise ValueError(
                    "traced integration endpoints cannot be combined with explicit "
                    "SaveAt.ts and t0/t1 flags because exact deduplication would make "
                    "the output shape data-dependent; make the span static or request "
                    "each endpoint in only one place."
                )
            if raw_ts is not None:
                requested_host = np.asarray(raw_ts)
                if requested_host.ndim != 1:
                    raise ValueError("SaveAt.ts must be a one-dimensional array of concrete times")
                if not np.all(np.isfinite(requested_host)):
                    raise ValueError("SaveAt times must all be finite")
                solve_dtype = np.dtype(jnp.dtype(dtype))
                requested_host = np.unique(np.asarray(requested_host, dtype=solve_dtype))
                query_ts = jnp.asarray(requested_host, dtype=dtype)
            else:
                pieces = []
                if want_t0:
                    pieces.append(jnp.asarray(t0, dtype=dtype).reshape(1))
                if want_t1 or not pieces:
                    pieces.append(jnp.asarray(t1, dtype=dtype).reshape(1))
                query_ts = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)
            requested_count = int(query_ts.shape[0])
            reserve = terminal_reserve()
            capacity = requested_count + int(reserve)
            return cls(
                query_ts=query_ts,
                output_ts=jnp.full((capacity,), jnp.nan, dtype=query_ts.dtype),
                requested_count=requested_count,
                capacity=capacity,
                has_terminal_reserve=reserve,
                terminal_slot=requested_count if reserve else -1,
                outputs=output_plan,
            )

        t0_host = np.asarray(t0)
        t1_host = np.asarray(t1)
        if t0_host.ndim != 0 or t1_host.ndim != 0:
            raise ValueError("the integration endpoints t0 and t1 must be scalar values")
        t0_value = float(t0_host)
        t1_value = float(t1_host)
        if not np.isfinite(t0_value) or not np.isfinite(t1_value):
            raise ValueError("the integration endpoints t0 and t1 must be finite")
        if t1_value < t0_value:
            raise ValueError("the integration endpoint t1 must be greater than or equal to t0")

        # Canonicalize in the solve dtype. Distinct host values can collapse
        # when a float32/bfloat16 solve receives a stronger-precision request;
        # deduplicating before that cast would overstate both N and capacity.
        solve_dtype = np.dtype(jnp.dtype(dtype))
        t0_solve = np.asarray(t0_value, dtype=solve_dtype).item()
        t1_solve = np.asarray(t1_value, dtype=solve_dtype).item()
        pieces: list[np.ndarray] = []
        if want_t0:
            pieces.append(np.asarray([t0_solve], dtype=solve_dtype))
        if raw_ts is not None:
            ts_host = np.asarray(raw_ts)
            if ts_host.ndim != 1:
                raise ValueError("SaveAt.ts must be a one-dimensional array of concrete times")
            pieces.append(np.asarray(ts_host, dtype=solve_dtype))
        if want_t1:
            pieces.append(np.asarray([t1_solve], dtype=solve_dtype))
        if not pieces or sum(piece.size for piece in pieces) == 0:
            pieces = [np.asarray([t1_solve], dtype=solve_dtype)]

        requested = np.unique(np.concatenate(pieces))
        if not np.all(np.isfinite(requested)):
            raise ValueError("SaveAt times must all be finite")
        if np.any(requested < t0_solve) or np.any(requested > t1_solve):
            raise ValueError(
                "SaveAt times must lie inside the closed integration interval [t0, t1]"
            )

        reserve = terminal_reserve()

        query_ts = jnp.asarray(requested, dtype=dtype)
        requested_count = int(requested.size)
        capacity = requested_count + int(reserve)
        output_ts = jnp.full((capacity,), jnp.nan, dtype=query_ts.dtype)
        return cls(
            query_ts=query_ts,
            output_ts=output_ts,
            requested_count=requested_count,
            capacity=capacity,
            has_terminal_reserve=reserve,
            terminal_slot=requested_count if reserve else -1,
            outputs=output_plan,
        )


def _build_save_plan(
    saveat: Optional[SaveAt],
    t0: Any,
    t1: Any,
    dtype: Any,
    *,
    event: Any = None,
    has_terminating_event: Optional[bool] = None,
    max_steps: Optional[int] = None,
    problem: Any = None,
) -> _SavePlan:
    """Return the shared fixed-buffer plan; see ``_SavePlan.build``."""
    return _SavePlan.build(
        saveat,
        t0,
        t1,
        dtype,
        event=event,
        has_terminating_event=has_terminating_event,
        max_steps=max_steps,
        problem=problem,
    )
