# LOC > 500 justification: the one-geometry leaf InferenceProblem and its
# composable nouns (objective, trajectory, initial-state, and parameter-group
# objects) share the duck-typed solver surface and the eager-build/dtype
# machinery; they belong in one module. The heavy decision-variable window
# mechanics live in _windows_dv.
"""inference_problem.py — the one-geometry inference problem and its composers.

An [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem] owns exactly one problem/geometry and one latent
trajectory.  It composes four interchangeable nouns:

* an **objective** — [LeastSquares][diff_soup.inference.fitting.inference_problem.LeastSquares] (weighted output-error fitting);
* a **trajectory** strategy — [SingleShoot][diff_soup.inference.fitting.inference_problem.SingleShoot] (one forward solve over the
  horizon) or [Multishoot][diff_soup.inference.fitting.inference_problem.Multishoot] (window partitioning);
* an **initial-state** policy — [FixedInitialState][diff_soup.inference.fitting.inference_problem.FixedInitialState] (the first
  differential state is fixed) or [LearnedInitialState][diff_soup.inference.fitting.inference_problem.LearnedInitialState] (it is optimized);
* **parameter groups** — [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups] naming the local and shared
  learnable selections.

An inference problem may carry several named [Series][diff_soup.inference.fitting.series.Series]
streams of the same trajectory.  In single-shoot mode one forward solve serves
the union of every stream's times; each stream's frames are sliced from that one
trajectory.

Both single-shoot and multishoot expose the duck-typed surface the solver
engines read — ``x0``, ``segment_fns``, ``n_segments``, ``regularization``,
``unflatten_fn``, ``prob_static`` — so ``optsolve`` can compose them with any backend.

θ-pipeline dtype law: ``x0`` carries the θ-leaves' dtype; residuals evaluate
in ``compute_dtype``; no hard fp64 casts in this module.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._precision import Precision, cast_to_compute_dtype, compute_dtype_of
from diff_soup._warnings import warn_user
from diff_soup.inference.fitting._forward_seam import (
    _FAILED_SENTINEL,
    GroupedThetaSelection,
    batched_transition_residual_factory,
    build_forward_seam,
    count_failed_frames,
    parameter_handle_leaf_ids,
    resolve_forward_adjoint,
    resolve_reverse_adjoint,
    selected_leaf_ids,
    union_residual_factory,
)
from diff_soup.inference.fitting._window_cache import (
    load_window_initialization,
    store_window_initialization,
    window_initialization_key,
)
from diff_soup.inference.fitting._windows_dv import (
    DecisionWindowBuild,
    WindowLayout,
    boundaries_from_windows,
    warn_if_over_ceiling,
)
from diff_soup.inference.fitting.bounds import Bounds
from diff_soup.inference.fitting.learnables import (
    LearnableSelector,
    has_declared_learnables,
    learnables,
)
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.series import FullState, Series, observes_full_state
from diff_soup.inference.fitting.windows import Continuity, RestartStep, StateBasis, Windows

# ---------------------------------------------------------------------------
# Objective objects
# ---------------------------------------------------------------------------


class Objective(eqx.Module):
    """Base class of the objective an [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem] takes through ``objective=``.

    An objective says how the misfit of the observations becomes the rows of
    the problem's residual.
    """


class LeastSquares(Objective):
    """The least-squares objective, half the squared norm of the residual."""


# ---------------------------------------------------------------------------
# Trajectory strategy objects
# ---------------------------------------------------------------------------


class Trajectory(eqx.Module):
    """Base class for trajectory strategies (how the latent trajectory is solved)."""


class SingleShoot(Trajectory):
    """Single shooting, one solve over the experiment's whole time span, the default trajectory."""


class Multishoot(Trajectory):
    """Multiple shooting, one solve per window of the time span, each from its own start state.

    Pass it as `trajectory=` of an `Experiment`, or of an `InferenceProblem` to apply it to
    every experiment.

    Args:
        windows: The [Windows][diff_soup.inference.Windows] that split the time span.
        continuity: The weight of the continuity term between fitted windows, `Continuity()` when `None`.
        states: The [StateBasis][diff_soup.inference.StateBasis] of the fitted window starts, or `None` to fit every differential entry.
        warm_start_steps: The number of optimizer steps on the parameters alone, with the window starts held, before the full fit.
        first_step: A `FirstStep` policy for the first time step of each window's solve, or `None` to leave it to the solver.
    """

    windows: Windows
    continuity: Continuity = eqx.field(default_factory=Continuity)
    states: StateBasis = eqx.field(default_factory=StateBasis.full)
    warm_start_steps: int = eqx.field(static=True, default=0)
    first_step: Optional[RestartStep] = None

    def __init__(
        self,
        windows: Windows,
        continuity: Optional[Continuity] = None,
        *,
        states: Optional[StateBasis] = None,
        warm_start_steps: int = 0,
        first_step: Optional[RestartStep] = None,
    ):
        """Store the window spec, continuity config, window-state coordinate, and step policies."""
        self.windows = windows
        self.continuity = continuity if continuity is not None else Continuity()
        if states is not None and not isinstance(states, StateBasis):
            raise TypeError(
                f"Multishoot 'states' must be a StateBasis; got {type(states).__name__}."
            )
        self.states = states if states is not None else StateBasis.full()
        if isinstance(warm_start_steps, bool) or int(warm_start_steps) < 0:
            raise ValueError(
                "Multishoot 'warm_start_steps' must be a nonnegative integer; "
                f"got {warm_start_steps!r}."
            )
        self.warm_start_steps = int(warm_start_steps)
        if first_step is not None and not isinstance(first_step, RestartStep):
            raise TypeError(
                "Multishoot 'first_step' must be a FirstStep policy (FirstStep.auto(), "
                f".fixed(dt), .fraction(f), .hinted(...)) or None; got {type(first_step).__name__}."
            )
        self.first_step = first_step


# ---------------------------------------------------------------------------
# Initial-state policy objects
# ---------------------------------------------------------------------------


def _declared_precision(problem: Any) -> Precision:
    """The precision a forward problem declares (the default when it declares none)."""
    declared = getattr(problem, "precision", None)
    if isinstance(declared, Precision):
        return declared
    from diff_soup.problem.implicit import get_or_build_solver_view

    try:
        view = get_or_build_solver_view(problem)
    except TypeError:
        return Precision()
    return getattr(view, "precision", None) or Precision()


def _lowered_schedule(schedule: Any, problem: Any) -> Any:
    """The segment arrays of ``schedule`` against ``problem``'s mode names, or ``None``.

    The arrays carry the targets and durations at the compute dtype of the
    problem's precision.
    """
    if schedule is None:
        return None
    from diff_soup._schedule import schedule_arrays

    return schedule_arrays(
        schedule,
        problem.operational_modes(),
        dtype=compute_dtype_of(_declared_precision(problem)),
    )


class InitialState(eqx.Module):
    """Base class for the trajectory initial-state policy."""


class FixedInitialState(InitialState):
    """A known initial state, the form a bare array passed as `initial_state=` takes.

    Attributes:
        y0: The state at the start of the experiment's time span.
        time: The start time, which the experiment sets to the start of its span.
    """

    y0: Any = None
    time: float = 0.0


class LearnedInitialState(InitialState):
    """An initial state fitted together with the parameters, starting from `guess`.

    Pass it as `initial_state=` of an `Experiment`. The fitted state follows the parameters
    in `result.params`, and `Windows.from_data` refuses it.

    Attributes:
        guess: The starting value of the full state, a 1-D array.
        time: The start time, which the experiment sets to the start of its span.
    """

    guess: Any
    time: float = 0.0


def _initial_state_array(initial_state: Any) -> Any:
    """Return the state array a declaration carries.

    A policy carries it as ``y0`` (fixed) or ``guess`` (learned); a bare array
    is the state itself. The array is what a prediction or a provenance digest
    reads, whichever policy holds it.
    """
    if isinstance(initial_state, LearnedInitialState):
        return initial_state.guess
    if isinstance(initial_state, FixedInitialState):
        return initial_state.y0
    return initial_state


def _initial_state_policy(initial_state: Any, *, time: float) -> InitialState:
    """Return the trajectory initial-state policy of a declaration, timed at ``time``.

    A bare array becomes ``FixedInitialState``; a policy keeps its kind and is
    re-timed, because the state a declaration carries is the state at the start
    of that declaration's own span.
    """
    if isinstance(initial_state, InitialState):
        return dataclasses.replace(initial_state, time=float(time))
    return FixedInitialState(initial_state, time=float(time))


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParameterGroups:
    """Named local and shared learnable-parameter selections. Frozen.

    Each value is a
    [LearnableSelector][diff_soup.inference.fitting.learnables.LearnableSelector]
    (from ``learnables(problem, names)``), a learnable name, or a sequence of
    names.  ``local`` groups are geometry-specific (e.g. a particle's ``k(x)``);
    ``shared`` groups name constitutive functions intended to be tied across
    geometries by a [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem]
    (e.g. ``j0(c; theta)``).  When a
    leaf is fit alone, shared groups are ordinary learnable groups.

    Attributes:
        local: Mapping from local group name to its learnable selector.
        shared: Mapping from shared group name to its learnable selector.
    """

    local: dict = dataclasses.field(default_factory=dict)
    shared: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Reject a name used as both a local and a shared group."""
        overlap = set(self.local) & set(self.shared)
        if overlap:
            raise ValueError(
                f"ParameterGroups: group name(s) {sorted(overlap)} appear in both "
                "local and shared; local and shared group names must be disjoint."
            )


def _group_value_names(value: Any) -> tuple:
    """Learnable names for a plain (non-selector) parameter-group value."""
    if isinstance(value, str):
        return (value,)
    return tuple(str(n) for n in value)


def _group_value_handles(value: Any) -> tuple:
    """Return Recipe handles from one group value, or ``()`` for name values."""
    if hasattr(value, "identity") and hasattr(value, "kind"):
        return (value,)
    if isinstance(value, (str, LearnableSelector)):
        return ()
    try:
        values = tuple(value)
    except TypeError:
        return ()
    if values and all(hasattr(item, "identity") and hasattr(item, "kind") for item in values):
        return values
    return ()


def _declared_learnable_names(problem: Any) -> frozenset:
    """Public names ``problem`` declares learnable, or an empty set.

    These are the names a declaration states — a Recipe parameter or function by the
    name it was declared under, a model-level declaration by the last segment of the
    path it states — not the leaf field names underneath them. A learnable function
    module named ``"reaction"`` is declared once under that name however many array
    fields it holds.
    """
    from diff_soup.inference.fitting.learnables import (
        _model_declared_paths,
        _traced_adapter_layer,
    )

    names = {path.rsplit(".", 1)[-1] for path in _model_declared_paths(problem)}
    located = _traced_adapter_layer(problem)
    if located is not None:
        _base, adapter = located
        names.update(getattr(adapter, "learnable_params", ()))
        names.update(getattr(adapter, "learnable_functions", ()))
    return frozenset(names)


def _theta_path_tokens(problem: Any) -> tuple:
    """Path tokens a plain group name can match on ``problem``'s theta pytree.

    Returns ``(tokens, labels)`` — every token on an inexact-array leaf's path (what a
    name is matched against) and the subset that names a leaf itself (what an error
    message advertises).
    """
    from diff_soup.inference.fitting._forward_seam import (
        _path_key_tokens,
        coerce_forward_problem,
    )

    theta0 = coerce_forward_problem(problem).default_theta()
    tokens: set = set()
    labels: set = set()
    for path, leaf in jax.tree_util.tree_leaves_with_path(theta0):
        if not eqx.is_inexact_array(leaf):
            continue
        path_tokens = _path_key_tokens(path)
        tokens.update(path_tokens)
        if path_tokens:
            labels.add(path_tokens[-1])
    return frozenset(tokens), frozenset(labels)


def _resolve_parameters(parameters: Optional[ParameterGroups], problem: Any) -> tuple:
    """Resolve [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups].

    Produces a θ-selection spec and group records. Returns ``(theta_selection, groups)``:

    * ``theta_selection`` is what the forward seam selects as learnable — ``None``
      (the problem's declared learnables) when no groups are declared, and otherwise a
      [GroupedThetaSelection]
      [diff_soup.inference.fitting._forward_seam.GroupedThetaSelection]. Its
      ``order`` puts the flat decision vector in declaration order: shared
      groups first, then local groups, each in the order declared -- the same
      order a joint problem gives its shared and per-experiment blocks. Declared
      names and explicit
      [LearnableSelector][diff_soup.inference.fitting.learnables.LearnableSelector]
      values are resolved to slot keys of the problem's path table, while legacy model names retain their
      path-token matching behavior. Thus, a declared learnable whose public name is
      not a θ-leaf path token still resolves.
    * ``groups`` is a tuple of ``(group_name, kind, leaf_ids, tokens)`` records in
      declaration order (locals first, then shared). ``leaf_ids`` are the slot
      keys a selector, a Recipe handle, or a declared-learnable name resolved
      to; ``tokens`` are the plain names left to path-token matching. A group can
      carry both when it mixes declared learnables with path tokens.
    """
    if parameters is None:
        return None, ()
    groups: list = []
    combined: list = []
    seen: set = set()
    name_sources: list = []
    mask_sources: list = []
    leaf_id_sources: list = []
    has_identity_selection = False
    uses_declared_learnables = has_declared_learnables(problem)
    declared_names: Optional[frozenset] = None
    for kind, mapping in (("local", parameters.local), ("shared", parameters.shared)):
        for gname, value in mapping.items():
            if isinstance(value, LearnableSelector):
                if value.names is None:
                    raise ValueError(
                        "ParameterGroups: a group selector must name specific learnables; "
                        "learnables(problem) with no names selects everything and cannot "
                        "label a group."
                    )
                has_identity_selection = True
                groups.append((gname, kind, selected_leaf_ids(problem, value.mask), frozenset()))
                mask_sources.append(value.mask)
            elif handles := _group_value_handles(value):
                has_identity_selection = True
                ids = frozenset().union(
                    *(parameter_handle_leaf_ids(handle, problem=problem) for handle in handles)
                )
                if not ids:
                    raise ValueError(
                        f"ParameterGroups: group {gname!r} selects no inexact theta leaves."
                    )
                groups.append((gname, kind, ids, frozenset()))
                leaf_id_sources.append(ids)
            else:
                names = _group_value_names(value)
                if uses_declared_learnables:
                    if declared_names is None:
                        declared_names = _declared_learnable_names(problem)
                    declared = tuple(n for n in names if n in declared_names)
                    tokens = tuple(n for n in names if n not in declared_names)
                    if tokens:
                        available, labels = _theta_path_tokens(problem)
                        unknown = sorted(n for n in tokens if n not in available)
                        if unknown:
                            raise ValueError(
                                f"ParameterGroups: group {gname!r} names {unknown}, which "
                                "match neither a declared learnable of this problem "
                                f"({sorted(declared_names)}) nor a theta leaf "
                                f"({sorted(labels)}). Name a declared learnable, or a "
                                "learnable leaf of the problem's theta pytree."
                            )
                else:
                    declared, tokens = (), tuple(names)
                group_ids: frozenset = frozenset()
                if declared:
                    selector = learnables(problem, declared)
                    if selector is None:  # pragma: no cover - learnables raises for missing names
                        raise ValueError(
                            f"ParameterGroups: group {gname!r} selects no learnable leaves."
                        )
                    has_identity_selection = True
                    group_ids = selected_leaf_ids(problem, selector.mask)
                    mask_sources.append(selector.mask)
                if tokens:
                    name_sources.append(tokens)
                groups.append((gname, kind, group_ids, frozenset(tokens)))
                for n in names:
                    if n not in seen:
                        seen.add(n)
                        combined.append(n)
    order = tuple(
        (leaf_ids, tokens)
        for kind in ("shared", "local")
        for _gname, group_kind, leaf_ids, tokens in groups
        if group_kind == kind
    )
    if has_identity_selection:
        theta_selection: Any = GroupedThetaSelection(
            name_sources=tuple(name_sources),
            mask_sources=tuple(mask_sources),
            leaf_id_sources=tuple(leaf_id_sources),
            order=order,
        )
    elif combined:
        theta_selection = GroupedThetaSelection(name_sources=(tuple(combined),), order=order)
    else:
        theta_selection = None
    return theta_selection, tuple(groups)


class ParameterValues(Mapping):
    """The parameter values a flat vector in a fit's order holds, read by parameter.

    Returned by `InferenceProblem.values(x)`. `values[rate]` is the value `fit[rate]`
    reads, and a parameter that varies by experiment gives a mapping by experiment.
    """

    def __init__(self, schema: Any, x: Any):
        """Read `x` through `schema`, a fit's parameter schema."""
        self._schema = schema
        self._x = x

    def __getitem__(self, parameter: Any) -> Any:
        """Return the value of `parameter`."""
        return self._schema.value(parameter, self._x)

    def __iter__(self):
        """Iterate over the fitted parameters in the fit's order."""
        return iter(self._schema.handles)

    def __len__(self) -> int:
        """Return the number of fitted parameters."""
        return len(self._schema.entries)

    def __repr__(self) -> str:
        """Name the fitted parameters."""
        return f"ParameterValues({[handle.name for handle in self._schema.handles]})"


def parameter_values(schema: Any, x: Any) -> ParameterValues:
    """Return the read-only view of `x` through `schema`, refusing a problem with none."""
    if schema is None:
        raise ValueError(
            "values: this inference problem fits no declared parameters, so a flat "
            "vector has no parameter values to read."
        )
    return ParameterValues(schema, x)


# Marks a ParameterSchema whose fingerprint labels one inference problem's own fitted
# coordinates. It carries no declaration content, so it is never comparable with a
# Study's content fingerprint.
_STRUCTURAL_SCHEMA_PREFIX = "structural:"


def _is_structural_schema(schema: Any) -> bool:
    """Whether ``schema`` labels one inference problem's own coordinates rather than a Study's.

    A structural schema is stamped on the result of fitting an inference problem directly.
    Its fingerprint records that inference problem's handle-to-column layout, so it states
    nothing about the declaration a Study fingerprint summarizes and the two are never
    equal by construction.

    Args:
        schema: A [ParameterSchema][diff_soup.problem.parameters.ParameterSchema], or
            ``None``.

    Returns:
        ``True`` when ``schema`` carries a structural fingerprint.
    """
    fingerprint = getattr(schema, "fingerprint", None)
    return isinstance(fingerprint, str) and fingerprint.startswith(_STRUCTURAL_SCHEMA_PREFIX)


def _recipe_parameter_schema(problem: Any, parameters: ParameterGroups, seam: Any):
    """Build handle lookup metadata for one structural Recipe inference problem.

    A handle earns an entry when the fit selected exactly the coordinates its
    declared value holds. A handle the selection covers only in part is left out,
    so the result stays indexable by every handle it can answer for. A handle the
    caller named in the declaration is an error when it resolves incompletely.
    """
    from diff_soup.problem._registry import parameter_handles
    from diff_soup.problem.parameters import (
        ParameterEntry,
        ParameterOccurrence,
        ParameterSchema,
        parameter_decoder,
    )

    explicit: list = []
    for mapping in (parameters.local, parameters.shared):
        for value in mapping.values():
            explicit.extend(_group_value_handles(value))

    named = bool(explicit)
    if named:
        handles = tuple(dict.fromkeys(explicit))
    else:
        namespaces = parameter_handles(problem)
        # Structurally identical Recipe builders share one lowered problem.
        # Retain every registered handle identity as an alias for the same
        # current structural coordinate, so a result remains indexable by the
        # handle held by whichever builder supplied it.
        handles = tuple(
            dict.fromkeys(handle for declarations in namespaces.values() for handle in declarations)
        )

    entries = []
    for handle in handles:
        if not bool(getattr(handle, "learnable", False)):
            continue
        leaf_ids = parameter_handle_leaf_ids(handle, problem=problem)
        columns = tuple(i for i, leaf_id in enumerate(seam.theta_leaf_ids) if leaf_id in leaf_ids)
        if not columns:
            continue
        size, decode = parameter_decoder(handle)
        if len(columns) != size:
            if not named:
                # A selection that covers a declared value only in part leaves this
                # handle without a complete coordinate block, so it gets no schema
                # entry rather than a partial one.
                continue
            raise ValueError(
                f"Recipe handle {handle.name!r} resolves to {len(columns)} fitted "
                f"coordinates, but its declared value has size {size}. Select every "
                "coordinate of the handle, or name a handle whose value the "
                "selection covers completely."
            )
        entries.append(
            ParameterEntry(
                handle=handle,
                occurrences=(ParameterOccurrence(owner=None, columns=columns),),
                size=size,
                decode=decode,
            )
        )
    if not entries:
        return None
    entries.sort(key=lambda entry: entry.occurrences[0].columns[0])
    fingerprint = _STRUCTURAL_SCHEMA_PREFIX + "|".join(
        f"{entry.handle.kind}:{entry.handle.name}:{entry.occurrences[0].columns}"
        for entry in entries
    )
    return ParameterSchema(entries, fingerprint=fingerprint)


def _group_columns(seam: Any, groups: tuple) -> tuple:
    """Map each parameter group to its flat-vector column indices.

    A group owns a column when that column's source θ-leaf slot key is one the group
    resolved (a selector, a Recipe handle, or a declared-learnable name — which
    resolves declared learnables whose public name is not a θ-leaf token), or when one
    of the group's remaining plain names appears in that column's path-token set (the
    same rule the name→leaf resolver uses).

    Returns ``(group_names, group_cols)`` in declaration order.
    """
    token_sets = seam.theta_token_sets
    leaf_ids = seam.theta_leaf_ids
    names: list = []
    cols: list = []
    owners: dict[int, str] = {}
    for gname, _kind, group_ids, group_tokens in groups:
        matched = tuple(
            i
            for i, (lid, ts) in enumerate(zip(leaf_ids, token_sets))
            if lid in group_ids or (group_tokens and ts & group_tokens)
        )
        for col in matched:
            previous = owners.get(col)
            if previous is not None:
                raise ValueError(
                    f"ParameterGroups: groups {previous!r} and {gname!r} overlap on "
                    f"learnable parameter column {col}. Each learnable scalar must belong "
                    "to exactly one parameter group."
                )
            owners[col] = gname
        names.append(gname)
        cols.append(matched)
    return tuple(names), tuple(cols)


def _parameter_groups_select_name(parameters: ParameterGroups, name: str) -> bool:
    """Whether an explicit nonempty parameter declaration selects ``name``.

    An empty [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups]
    states no selection at all, and the fit falls back to the problem's declared
    learnables, so callers should only use this helper after establishing that at
    least one group was declared.
    """
    for mapping in (parameters.local, parameters.shared):
        for value in mapping.values():
            if isinstance(value, LearnableSelector):
                if value.names is not None and name in value.names:
                    return True
            elif handles := _group_value_handles(value):
                if any(handle.name == name for handle in handles):
                    return True
            elif name in _group_value_names(value):
                return True
    return False


def build_stacked_residual_fns(segment_fns: tuple) -> tuple:
    """Build the jitted ``(r_fn, rj_fn)`` pair over a tuple of segment closures.

    ``r_fn(p)`` concatenates the per-segment residuals; ``rj_fn(p)`` returns
    ``(r, J)`` via forward-mode. Built ONCE per inference problem and cached, so a
    repeated ``solve()`` — every outer iteration of a penalty loop — reuses the
    compiled kernels instead of retracing the PDE residual. Regularizer rows are
    deliberately NOT part of these callables: they evaluate eagerly in the
    solvers, so penalty weights can change between solves without recompiling.
    """
    # Local import avoids a fitting-to-solver module cycle at import time.
    from diff_soup.inference.solvers.tangent import value_and_jacfwd

    @jax.jit
    def _r_fn(p):
        return jnp.concatenate([fn(p) for fn in segment_fns])

    @jax.jit
    def _rj_fn(p):
        return value_and_jacfwd(lambda p_: _r_fn(p_), p)

    return _r_fn, _rj_fn


def _stacked_residual_fn(segment_fns: tuple, batched_r: Any = None) -> Callable:
    """Build a jitted data-residual closure without differentiating it."""
    if batched_r is not None:
        return batched_r

    @jax.jit
    def residual(x):
        return jnp.concatenate([fn(x) for fn in segment_fns])

    return residual


def _loss_fn_from_residual(residual_fn: Callable) -> Callable:
    """Build the scalar data-misfit loss from a residual closure."""

    @jax.jit
    def loss(x):
        residual = residual_fn(x)
        return 0.5 * jnp.sum(residual * residual)

    return loss


def _loss_and_grad_from_residual(residual_fn: Callable, ad: str) -> Callable:
    """Build one scalar-loss value-and-gradient seam."""
    if ad == "forward":
        from diff_soup.inference.solvers.tangent import value_and_jacfwd

        @jax.jit
        def forward_loss_and_grad(x):
            residual, jacobian = value_and_jacfwd(residual_fn, x)
            loss = 0.5 * jnp.sum(residual * residual)
            grad = jacobian.T @ residual
            failed = jnp.any(jnp.abs(residual) >= _FAILED_SENTINEL)
            return loss, jnp.where(failed, jnp.zeros_like(grad), grad)

        return forward_loss_and_grad

    @jax.jit
    def reverse_loss_and_grad(x):
        def objective(point):
            residual = residual_fn(point)
            loss = 0.5 * jnp.sum(residual * residual)
            failed = jnp.any(jnp.abs(residual) >= _FAILED_SENTINEL)
            return loss, failed

        (loss, failed), grad = jax.value_and_grad(objective, has_aux=True)(x)
        # A zero cotangent entering a failed time loop can still encounter NaN
        # during replay. Select after the backward pass so it cannot escape.
        return loss, jnp.where(failed, jnp.zeros_like(grad), grad)

    return reverse_loss_and_grad


def _loss_grad_hessian_from_gradient(loss_and_grad_fn: Callable) -> Callable:
    """Build a forward derivative of a value-and-gradient seam."""

    @jax.jit
    def loss_grad_hessian(x):
        def gradient_with_aux(point):
            loss, grad = loss_and_grad_fn(point)
            return grad, (loss, grad)

        hessian, (loss, grad) = jax.jacfwd(gradient_with_aux, has_aux=True)(x)
        return loss, grad, hessian

    return _MemoryCheckedHessian(loss_grad_hessian, tile="full")


# The memory of an exact Hessian grows with the square of the parameter count:
# every tangent direction of the forward-over-reverse product carries the whole
# reverse tape. Measured on a decision-variable multishoot fit (CPU,
# 2026-10-28): 4.2 / 16.1 / 63.2 GB of compiled temporaries at 83 / 163 / 323
# parameters. Below this width the Hessian is used without measuring it first.
_HESSIAN_GUARD_MIN_WIDTH = 64
#: The share of the free device memory one Hessian evaluation may take.
_HESSIAN_MEMORY_SHARE = 0.9


class _HessianMemoryError(ConfigurationError):
    """The exact Hessian needs more memory than the device has free."""


def _cgroup_memory_headroom() -> Optional[int]:
    """The bytes this process may still allocate under its cgroup v2 limits, or ``None``."""
    try:
        with open("/proc/self/cgroup") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return None
    headroom = None
    path = os.path.normpath("/sys/fs/cgroup/" + relative.lstrip("/"))
    while path.startswith("/sys/fs/cgroup"):
        try:
            with open(os.path.join(path, "memory.max")) as handle:
                limit = handle.read().strip()
            with open(os.path.join(path, "memory.current")) as handle:
                current = int(handle.read().strip())
        except (OSError, ValueError):
            limit = "max"
            current = 0
        if limit != "max":
            room = max(int(limit) - current, 0)
            headroom = room if headroom is None else min(headroom, room)
        if path == "/sys/fs/cgroup":
            break
        path = os.path.dirname(path)
    return headroom


def _host_available_memory() -> Optional[int]:
    """The host memory available to this process in bytes, or ``None`` when unknown."""
    available = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        available = None
    if available is None:
        try:
            import psutil

            available = int(psutil.virtual_memory().available)
        except Exception:  # psutil is optional; without it the host is unmeasured
            available = None
    headroom = _cgroup_memory_headroom()
    if headroom is not None:
        available = headroom if available is None else min(available, headroom)
    return available


def _free_device_memory(device: Any) -> Optional[int]:
    """The bytes ``device`` can still allocate, or ``None`` when it cannot say."""
    if device.platform == "cpu":
        return _host_available_memory()
    try:
        stats = device.memory_stats() or {}
    except Exception:  # a backend without allocator statistics is unmeasured
        return None
    limit = stats.get("bytes_limit")
    if limit is None:
        return None
    return int(limit) - int(stats.get("bytes_in_use", 0))


def _check_hessian_memory(compiled: Any, width: int, tile: Any) -> None:
    """Raise `_HessianMemoryError` when a compiled Hessian needs more than the free memory.

    ``compiled`` is the ahead-of-time compiled evaluation of ``width``
    parameters in tiles of ``tile`` directions. A backend that reports no
    memory figures passes.
    """
    try:
        analysis = compiled.memory_analysis()
    except Exception:  # not every backend implements the analysis
        return
    if analysis is None:
        return
    need = int(getattr(analysis, "temp_size_in_bytes", 0) or 0) + int(
        getattr(analysis, "output_size_in_bytes", 0) or 0
    )
    device = jax.local_devices()[0]
    free = _free_device_memory(device)
    if free is None or need <= _HESSIAN_MEMORY_SHARE * free:
        return
    gib = 2.0**30
    how = "whole" if tile in (None, "full") else f"in tiles of {tile} directions"
    smaller = (
        "a smaller hessian_tile= (8 or 16), " if tile in (None, "full") or int(tile) > 8 else ""
    )
    raise _HessianMemoryError(
        f"The exact Hessian over {width} parameters, formed {how}, needs "
        f"{need / gib:.1f} GiB and the {device.platform} device has {free / gib:.1f} GiB "
        f"free. With NewtonTrustRegion pass {smaller}"
        'hessian="gauss-newton", or subproblem="cg" with '
        "x_scale=ds.inference.ConstantScale(...). For laplace and information, "
        'use the expected curvature (curvature="expected", kind="expected").'
    )


#: Set while a concrete check compiles: an enclosing check measures every
#: nested Hessian, so the nested ones skip their own compilation.
_ENCLOSING_CHECK = threading.local()


class _MemoryCheckedHessian:
    """A jitted Hessian evaluation whose memory is checked before its first run.

    Above `_HESSIAN_GUARD_MIN_WIDTH` parameters the evaluation is compiled
    ahead of time once per input shape and its memory compared with the free
    memory of the device. A concrete input is compiled exactly as the call
    compiles it, so the call reuses that compilation. A traced input (the
    Hessian inside another jitted function) is measured on its shape, which
    costs a second compilation, unless an enclosing check is compiling it.
    """

    def __init__(self, fn: Callable, *, tile: Any = "full"):
        self._fn = fn if hasattr(fn, "lower") else jax.jit(fn)
        self.tile = tile
        self._checked: set = set()

    def lower(self, *args, **kwargs):
        """Lower the wrapped jitted function, as `jax.jit(...).lower` does."""
        return self._fn.lower(*args, **kwargs)

    def check(self, x: Any) -> None:
        """Compile for ``x`` and raise `_HessianMemoryError` when it does not fit."""
        shape = tuple(np.shape(x))
        if len(shape) != 1 or shape[0] <= _HESSIAN_GUARD_MIN_WIDTH:
            return
        key = (shape, str(jnp.result_type(x)))
        traced = isinstance(x, jax.core.Tracer)
        if key in self._checked or (traced and getattr(_ENCLOSING_CHECK, "active", False)):
            return
        point = jax.ShapeDtypeStruct(shape, jnp.result_type(x)) if traced else x
        enclosing = getattr(_ENCLOSING_CHECK, "active", False)
        _ENCLOSING_CHECK.active = True
        try:
            compiled = self._fn.lower(point).compile()
        finally:
            _ENCLOSING_CHECK.active = enclosing
        _check_hessian_memory(compiled, shape[0], self.tile)
        self._checked.add(key)

    def __call__(self, x: Any):
        self.check(x)
        return self._fn(x)


def lower_residual_problem(
    problem: Any,
    optimizer: Any,
    *,
    evaluator: Any = None,
    lanes: Any = None,
    gauss_newton: Any = None,
    value: Any = None,
    regularization: Any = None,
    joint_view: Any = None,
) -> Any:
    """The [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem] of a residual problem.

    ``problem`` is an inference problem of residual rows: an
    [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem],
    a joint problem, or a view of one. The result's residual is its
    ``batched_r_fn`` (else ``stacked_r_fn``, else the concatenation of its
    ``segment_fns``) with the matching Jacobian; its value, gradient and exact
    Hessian are ``evaluator.loss_fn``, ``evaluator.loss_and_grad_fn(ad)`` and
    ``evaluator.loss_grad_hessian_fn(composition)`` when the evaluator has
    them, evaluated on first use, and are derived from the residual
    otherwise. ``ad`` and ``composition`` come from ``optimizer`` (its ``ad``
    and ``composition``; a solver that requires ``"hessian"`` and names no
    ``ad`` differentiates its gradient in the mode the composition's inner
    derivative uses). The bounds are resolved once.

    Args:
        problem: The residual problem; its ``x0``, ``regularization``,
            ``bounds``, ``unflatten_fn``, ``prob_static``, ``segment_fns``,
            ``n_segments``, ``n_residuals`` and ``trial_is_valid`` are read.
        optimizer: The solver the problem is lowered for.
        evaluator: The object whose seams evaluate the objective; ``None``
            is ``problem``.
        lanes: The lowered study that evaluates the objective on its lanes,
            or ``None``; with one the structure holds ``"batched_lanes"``.
        gauss_newton: ``x -> (cost, J^T r, J^T J, failed)`` evaluated without
            forming the Jacobian, or ``None``: the lanes' ``gauss_newton_fn``
            when given, the reduced statistics of a sharded problem, else
            ``J^T J`` of the residual's Jacobian.
        value: ``x -> `` the objective, NaN where an evaluation failed, to use
            in place of ``evaluator.loss_fn``: the cost reduced across the
            mesh for a sharded problem; ``None`` reads ``evaluator.loss_fn``,
            or a sharded problem's ``evaluate_cost_fn``.
        regularization: The regularization the solver adds; ``None`` is
            ``problem.regularization``.
        joint_view: ``() -> `` the joint problem with its level structure,
            for ``"batched_lanes"``.
    """
    from diff_soup.inference._abc import LeastSquaresProblem
    from diff_soup.inference.fitting.bounds import resolve_decision_bounds

    evaluator = problem if evaluator is None else evaluator
    requires = frozenset(getattr(optimizer, "requires", ()))
    asked = getattr(optimizer, "composition", "auto")
    resolve = getattr(evaluator, "resolved_hessian_composition", None)
    composition = None
    if "hessian" in requires and callable(resolve):
        composition = resolve(asked)
    ad = getattr(optimizer, "ad", None)
    if ad is None:
        ad = "forward" if composition not in (None, "forward-over-reverse") else "reverse"

    residual = getattr(problem, "batched_r_fn", None)
    residual_and_jacobian = getattr(problem, "batched_rj_fn", None)
    if residual is None or residual_and_jacobian is None:
        residual = getattr(problem, "stacked_r_fn", None)
        residual_and_jacobian = getattr(problem, "stacked_rj_fn", None)
    if residual is None or residual_and_jacobian is None:
        residual, residual_and_jacobian = build_stacked_residual_fns(tuple(problem.segment_fns))
    if getattr(evaluator, "loss_fn", None) is None:
        evaluator = _ResidualSeams(residual)

    if value is None and lanes is None and getattr(problem, "is_sharded", False):
        evaluate_cost = getattr(problem, "evaluate_cost_fn", None)
        if evaluate_cost is not None:
            value = _reduced_cost(evaluate_cost)
    if value is None:

        def value(x):
            return evaluator.loss_fn(x)

    def value_and_gradient(x):
        return evaluator.loss_and_grad_fn(ad)(x)

    def value_gradient_hessian(x):
        return evaluator.loss_grad_hessian_fn(asked if composition is None else composition)(x)

    hessian_vector = None
    if lanes is None:

        def hessian_vector(x, v):
            gradient = evaluator.loss_and_grad_fn(ad)
            return jax.jvp(lambda y: gradient(y)[1], (x,), (v,))[1]

    predicate = getattr(problem, "trial_is_valid", None)
    is_valid = None
    if callable(predicate):

        def is_valid(x):
            return problem.trial_is_valid(x)

    structure = {"value_gradient", "residual_jacobian", "gauss_newton"}
    if getattr(evaluator, "supported_compositions", frozenset()):
        structure.add("hessian")
    if lanes is not None:
        structure.add("batched_lanes")
    reduced = gauss_newton is not None
    if gauss_newton is None and lanes is None and getattr(problem, "is_sharded", False):
        evaluate = getattr(problem, "evaluate_lsq_fn", None)
        if evaluate is not None:
            gauss_newton, reduced = _reduced_statistics(evaluate), True
    regularization = (
        getattr(problem, "regularization", None) if regularization is None else regularization
    )
    n_residuals = getattr(problem, "n_residuals", None)
    segment_fns = tuple(getattr(problem, "segment_fns", ()) or ())
    return LeastSquaresProblem(
        x0=problem.x0,
        structure=frozenset(structure),
        _unflatten=getattr(problem, "unflatten_fn", None),
        _static=getattr(problem, "prob_static", None),
        bounds=resolve_decision_bounds(problem),
        _regularization=regularization,
        lanes=lanes,
        ad=ad,
        hessian_composition=composition,
        n_segments=int(getattr(problem, "n_segments", 1) or 1),
        reduced_gauss_newton=reduced,
        _value=value,
        _value_and_gradient=value_and_gradient,
        _value_gradient_hessian=value_gradient_hessian,
        _hessian_vector=hessian_vector,
        _gauss_newton=gauss_newton,
        _is_valid=is_valid,
        _joint_view=joint_view,
        _residual=residual,
        _residual_and_jacobian=residual_and_jacobian,
        segment_fns=segment_fns,
        n_residuals=None if n_residuals is None else int(n_residuals),
    )


def _reduced_cost(evaluate_cost: Callable) -> Callable:
    """``x -> `` the cost from a reduced ``evaluate_cost_fn``, NaN where a child failed."""

    def value(x):
        cost, _rows, failed = evaluate_cost(jnp.asarray(x))
        return jnp.where(failed == 0, cost, jnp.nan)

    return value


def _reduced_statistics(evaluate: Callable) -> Callable:
    """``x -> (cost, J^T r, J^T J, failed)`` from a reduced ``evaluate_lsq_fn``."""

    def gauss_newton(x):
        cost, gradient, matrix, _rows, failed = evaluate(jnp.asarray(x))
        return cost, gradient, matrix, failed

    return gauss_newton


class _ResidualSeams:
    """The scalar seams of a residual problem that defines none of its own."""

    def __init__(self, residual: Callable):
        self.loss_fn = _loss_fn_from_residual(residual)
        self._residual = residual
        self._gradients: dict = {}
        self.supported_compositions = frozenset({"forward-over-forward"})

    def loss_and_grad_fn(self, ad: str = "reverse") -> Callable:
        if ad not in self._gradients:
            self._gradients[ad] = _loss_and_grad_from_residual(self._residual, ad)
        return self._gradients[ad]

    def loss_grad_hessian_fn(self, composition: str = "auto") -> Callable:
        if "hessian" not in self._gradients:
            self._gradients["hessian"] = _loss_grad_hessian_from_gradient(
                self.loss_and_grad_fn("forward")
            )
        return self._gradients["hessian"]


def _resolve_spec_declared_bounds(inference_problem: Any, problem: Any):
    """Resolve bounds declared on the problem's spec params to flat ``(lb, ub)``.

    A declared bound binds a column when its parameter is a decision variable of
    this inference problem — learnable and selected into the flat vector. Columns are
    found by slot key through its stable Recipe handle (a declared
    parameter's public name need not be a θ-leaf path token). A bounded learnable intentionally
    omitted by a nonempty
    [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups]
    is fixed for this fit and
    needs no bound, so it is ignored silently. A declared non-learnable, or a
    learnable explicitly selected but absent from the decision vector, warns
    instead. Returns ``None`` when nothing binds or when an explicit
    ``InferenceProblem(bounds=...)`` overrides the declarations; the window-state
    tail always stays unbounded.
    """
    declared = getattr(problem, "_declared_param_bounds", ())
    if not declared:
        return None

    parameters = inference_problem.parameters
    explicit_selection = bool(parameters.local or parameters.shared)
    unenforceable = [
        getattr(handle, "name", str(handle)) for handle, _b, learnable in declared if not learnable
    ]
    entries = []
    if inference_problem.bounds is None:
        leaf_ids = inference_problem._seam.theta_leaf_ids
        for handle, (lo, hi), learnable in declared:
            name = getattr(handle, "name", str(handle))
            if not learnable:
                continue
            if explicit_selection and not _parameter_groups_select_name(parameters, name):
                # A nonempty ParameterGroups is an explicit decision set. Omitted
                # learnables are fixed at their declared values for this fit.
                continue
            ids = parameter_handle_leaf_ids(handle, problem=problem)
            cols = tuple(i for i, lid in enumerate(leaf_ids) if lid in ids)
            if not cols:
                unenforceable.append(name)
                continue
            entries.append((cols, lo, hi))

    if unenforceable:
        warn_user(
            f"bounds declared on parameter(s) {sorted(set(unenforceable))} cannot "
            "be enforced: they are not decision variables of this inference problem. "
            "A parameter bound binds only when the parameter is declared "
            "learnable=True and selected into the fit; otherwise drop its "
            "bounds or pass bounds explicitly to the InferenceProblem.",
        )

    if inference_problem.bounds is not None or not entries:
        return None

    n = int(np.asarray(inference_problem._x0).size)
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for cols, lo, hi in entries:
        idx = np.asarray(cols, dtype=int)
        lb[idx] = -np.inf if lo is None else float(lo)
        ub[idx] = np.inf if hi is None else float(hi)
    return jnp.asarray(lb), jnp.asarray(ub)


@dataclasses.dataclass(frozen=True)
class _ThetaOnlyRegularizer:
    """Apply a parameter regularizer only to the leading theta block."""

    base: Any
    n_theta: int

    def residual(self, x: Any):
        """Return the wrapped regularizer rows for ``x[:n_theta]``."""
        return self.base.residual(jnp.asarray(x)[: self.n_theta])

    def penalty(self, x: Any):
        """Return the wrapped regularizer penalty for ``x[:n_theta]``."""
        return self.base.penalty(jnp.asarray(x)[: self.n_theta])


def _regularizer_for_decision_vector(regularizer: Any, n_theta: int, n_decision: int) -> Any:
    """Restrict parameter regularization when a decision vector also carries states."""
    if n_theta == n_decision:
        return regularizer
    return _ThetaOnlyRegularizer(regularizer, int(n_theta))


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------


def _as_named_series_tuple(data: Any) -> tuple:
    """Coerce ``data`` into a tuple of [Series][diff_soup.inference.fitting.series.Series] with unique names."""
    series = (data,) if isinstance(data, Series) else tuple(data)
    if not series or not all(isinstance(s, Series) for s in series):
        raise TypeError("InferenceProblem 'data' must be a Series or a tuple of Series.")
    names = [s.name for s in series]
    if len(set(names)) != len(names):
        raise ValueError(
            f"InferenceProblem: Series names must be unique within an inference problem; got {names}."
        )
    return series


def _resolve_y0(data: tuple, y0: Any) -> Any:
    """Resolve the trajectory IC: explicit ``y0`` else full-state ``data[0]``.

    Raises:
        ValueError: If ``y0`` is ``None`` and the first stream is not full state.
    """
    if y0 is not None:
        return jnp.asarray(y0)
    s0 = data[0]
    if not observes_full_state(s0.observable):
        raise ValueError(
            "InferenceProblem: the initial state is required unless the first Series "
            "observes the full state (observable=FullState(), or a measurement "
            "declared with no readout). Pass initial_state=FixedInitialState(y0) "
            "or a full-state first Series."
        )
    return s0.values[0]


def _check_initial_state_time(initial_state: Any, data: tuple) -> None:
    """Reject a FixedInitialState time that contradicts the frame it labels.

    A [FixedInitialState][diff_soup.inference.fitting.inference_problem.FixedInitialState]
    with no ``y0`` takes its state from the first stream's first observed frame, so
    the only time that state can carry is that frame's own time.

    Raises:
        ValueError: If ``y0`` is ``None`` and ``time`` names a different instant.
    """
    if not isinstance(initial_state, FixedInitialState) or initial_state.y0 is not None:
        return
    first = float(np.asarray(data[0].ts).reshape((-1,))[0])
    stated = float(initial_state.time)
    if abs(stated - first) > 1e-12 * max(1.0, abs(first)):
        raise ValueError(
            f"FixedInitialState(time={stated}) with no y0 takes its state from the "
            f"first frame of Series {data[0].name!r}, which is observed at t={first}. "
            f"State time={first} to match the frame, pass FixedInitialState(y0, "
            f"time={stated}) to state a different instant with an explicit y0, or omit "
            "initial_state to take both the state and the time from the first frame."
        )


def _reached_tend(ts: Any, t_end: float) -> bool:
    """Host-side loud-status check: did the solve reach ``t_end``?"""
    return abs(float(ts[-1]) - t_end) <= 1e-9 * max(1.0, abs(t_end))


# ---------------------------------------------------------------------------
# The one-geometry leaf inference problem
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False)
class InferenceProblem:
    """The one-geometry leaf inference problem. Frozen, eager-build.

    Owns one problem/geometry and one latent trajectory, composes an objective,
    a trajectory strategy, an initial-state policy, and named parameter groups,
    and exposes the duck-typed surface the solver engines read.

    A plain frozen dataclass rather than an ``eqx.Module``: it holds non-pytree
    build products — the solver problem (which references JAX arrays via its
    cached ``theta0``), the injected solver's config, and the residual closures —
    that must be carried verbatim, not as pytree leaves.  The residual closures
    it builds are what flow through JAX.

    Construct as::

        InferenceProblem(
            problem,
            data=(Series("concentration", ts_c, c_obs, observable=c_op), ...),
            solver=solver,
            parameters=ParameterGroups(local={"k": learnables(problem, ["k"])}),
            objective=LeastSquares(),
            trajectory=SingleShoot(),
            initial_state=FixedInitialState(y0),
        )
    """

    problem: Any
    data: tuple
    solver: Any
    objective: Any
    trajectory: Any
    initial_state: Any
    parameters: Any
    parameter_schema: Any
    regularization: Any
    _seam: Any
    _segment_fns: tuple
    _residual_builder: Any
    _batched_r: Any
    _batched_rj: Any
    # Lazily-built jitted (r, rj) pair over ``segment_fns`` — one compile per
    # inference problem, reused by every subsequent solve.
    _stacked_r: Any
    _stacked_rj: Any
    _residual_seams: dict
    _loss_fn: Any
    _loss_grad_fns: dict
    _loss_grad_hessian_fns: dict
    _x0: np.ndarray
    _n_seg: int
    _baseline_reached_tend: Optional[bool]
    _baseline_probe: Any
    _residual_rows: int
    _parameters_argument: Any
    _reused_seam: Any
    _window_content_key: Any
    _window_record_pending: bool
    _dv_build: Any
    _stream_names: tuple
    _stream_slices: tuple
    _param_group_names: tuple
    _param_group_cols: tuple
    _union_ts: Any
    _solve_probe: Any
    state_prior_weight: float
    theta_only_steps: int
    project_windows: bool
    window_initialization: Any
    restart_step: Any
    trial_predicate: Any
    bounds: Optional[Bounds]
    # Flat (lb, ub) resolved from bounds declared on the problem's spec params;
    # ``None`` when nothing is declared or an explicit ``bounds=`` overrides.
    _declared_bounds: Any
    # The operating protocol the single shoot follows, and its segment arrays
    # against the problem's mode names; both ``None`` without a schedule.
    schedule: Any
    _schedule_arrays: Any

    def __init__(
        self,
        problem: Any,
        data: Union[Series, tuple],
        *,
        solver: Any,
        parameters: Optional[ParameterGroups] = None,
        objective: Optional[Objective] = None,
        trajectory: Optional[Trajectory] = None,
        initial_state: Any = None,
        regularizer: Any = None,
        state_prior_weight: float = 0.0,
        theta_only_steps: int = 0,
        project_windows: bool = True,
        window_initialization: Any = None,
        restart_step: Any = None,
        trial_predicate: Any = None,
        bounds: Optional[Bounds] = None,
        schedule: Any = None,
    ):
        """Eagerly build the residual closures for the chosen trajectory.

        ``schedule`` is an operating protocol (a ``ds.Schedule``) that the single
        shoot follows from the initial state: its segments are solved one after
        another in place of one solve. It needs a problem with an operational
        control table and a ``SingleShoot`` trajectory.
        """
        objective = objective if objective is not None else LeastSquares()
        trajectory = trajectory if trajectory is not None else SingleShoot()
        if not isinstance(objective, Objective):
            raise TypeError(
                f"InferenceProblem 'objective' must be an Objective (e.g. LeastSquares()); "
                f"got {type(objective).__name__}."
            )
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                f"InferenceProblem 'trajectory' must be a Trajectory (SingleShoot/Multishoot); "
                f"got {type(trajectory).__name__}."
            )
        if initial_state is not None and not isinstance(initial_state, InitialState):
            initial_state = FixedInitialState(jnp.asarray(initial_state))
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError(
                f"InferenceProblem 'bounds' must be a Bounds instance; got {type(bounds).__name__}."
            )
        if trial_predicate is not None and not callable(trial_predicate):
            raise TypeError("InferenceProblem 'trial_predicate' must be callable or None.")
        if schedule is not None and isinstance(trajectory, Multishoot):
            raise ValueError(
                "InferenceProblem 'schedule' is solved in a single shoot over the span, and "
                "a Multishoot trajectory splits the span into windows; pass "
                "trajectory=SingleShoot() with a schedule."
            )

        series = _as_named_series_tuple(data)
        # The problem's precision owns the inference arithmetic too: observed
        # data, likelihood and observable parameters, and the initial state
        # are cast to its compute dtype once, here, so residual rows, their
        # Jacobian, and the decision vector all run at that dtype.
        precision = _declared_precision(problem)
        series = tuple(cast_to_compute_dtype(s, precision, what="data") for s in series)
        initial_state = cast_to_compute_dtype(initial_state, precision, what="initial state policy")
        _check_initial_state_time(initial_state, series)
        t0 = (
            float(initial_state.time)
            if isinstance(initial_state, (FixedInitialState, LearnedInitialState))
            else None
        )
        theta_spec, groups = _resolve_parameters(parameters, problem)

        _set = object.__setattr__
        _set(self, "problem", problem)
        _set(self, "objective", objective)
        _set(self, "trajectory", trajectory)
        _set(self, "initial_state", initial_state)
        _set(self, "parameters", parameters if parameters is not None else ParameterGroups())
        _set(self, "_parameters_argument", parameters)
        _set(self, "_reused_seam", None)
        _set(self, "_window_content_key", None)
        _set(self, "_window_record_pending", False)
        _set(self, "regularization", regularizer if regularizer is not None else NoReg())
        _set(self, "state_prior_weight", float(state_prior_weight))
        _set(self, "theta_only_steps", int(theta_only_steps))
        _set(self, "project_windows", bool(project_windows))
        _set(self, "window_initialization", window_initialization)
        _set(self, "restart_step", restart_step)
        _set(self, "trial_predicate", trial_predicate)
        _set(self, "bounds", bounds)
        _set(self, "schedule", schedule)
        _set(self, "_schedule_arrays", _lowered_schedule(schedule, problem))

        if isinstance(trajectory, SingleShoot):
            self._build_single_shoot(series, solver, theta_spec, groups, t0)
        elif isinstance(trajectory, Multishoot):
            self._build_multishoot(series, solver, theta_spec, groups, trajectory, t0)
        else:  # pragma: no cover - guarded above
            raise TypeError(f"unsupported trajectory {type(trajectory).__name__}")
        _set(self, "_x0", np.asarray(self._x0).astype(compute_dtype_of(precision)))
        if jax.process_count() == 1:
            self.persist_window_initialization()

        # Bounds declared on the problem's spec params become default decision
        # bounds; an explicit ``bounds=`` above overrides them wholesale.
        _set(self, "_declared_bounds", _resolve_spec_declared_bounds(self, problem))
        _set(
            self,
            "parameter_schema",
            _recipe_parameter_schema(problem, self.parameters, self._seam),
        )
        _set(self, "_stacked_r", None)
        _set(self, "_stacked_rj", None)
        _set(self, "_batched_rj", None)
        _set(self, "_residual_seams", {})
        _set(self, "_loss_fn", None)
        _set(self, "_loss_grad_fns", {})
        _set(self, "_loss_grad_hessian_fns", {})

    def _seam_for(self, theta_spec: Any) -> Any:
        """The forward seam of this problem: the one handed over by ``with_data`` or a new one."""
        seam = self._reused_seam
        return build_forward_seam(self.problem, theta_spec) if seam is None else seam

    def with_data(self, data: Union[Series, tuple]) -> "InferenceProblem":
        """The same declaration over other observed values, reusing this problem's lowering.

        The new problem keeps the solver, parameters, objective, trajectory,
        initial state, regularizer, and bounds, and rebuilds only the residual
        closures over ``data``; the forward seam (the lowered problem, its
        learnable partition, and its solver view) is shared. Use it to replace
        placeholder values with loaded ones without lowering the experiment a
        second time.

        Args:
            data: Series with the same names, times, and shapes as the
                current data.

        Raises:
            ValueError: If ``data`` does not name the same streams with the
                same observed shapes as the current data.
        """
        precision = _declared_precision(self.problem)
        series = tuple(
            cast_to_compute_dtype(s, precision, what="data") for s in _as_named_series_tuple(data)
        )
        current = tuple(self.data)
        if tuple(item.name for item in series) != tuple(item.name for item in current) or any(
            tuple(np.shape(new.values)) != tuple(np.shape(old.values))
            for new, old in zip(series, current)
        ):
            raise ValueError(
                "with_data needs series of the same names and observed shapes as the "
                f"current data ({[(item.name, tuple(np.shape(item.values))) for item in current]}); "
                f"got {[(item.name, tuple(np.shape(item.values))) for item in series]}."
            )
        out = copy.copy(self)
        _set = object.__setattr__
        _set(out, "_reused_seam", self._seam)
        if self._dv_build is not None and self.window_initialization is None:
            _set(out, "window_initialization", self._dv_build.initialization())
        theta_spec, groups = _resolve_parameters(self._parameters_argument, self.problem)
        t0 = (
            float(self.initial_state.time)
            if isinstance(self.initial_state, (FixedInitialState, LearnedInitialState))
            else None
        )
        if isinstance(self.trajectory, SingleShoot):
            out._build_single_shoot(series, self.solver, theta_spec, groups, t0)
        else:
            out._build_multishoot(series, self.solver, theta_spec, groups, self.trajectory, t0)
        _set(out, "_x0", np.asarray(out._x0).astype(compute_dtype_of(precision)))
        _set(out, "_reused_seam", None)
        _set(out, "window_initialization", self.window_initialization)
        _set(out, "_window_content_key", self._window_content_key)
        _set(out, "_window_record_pending", self._window_record_pending)
        _set(out, "_stacked_r", None)
        _set(out, "_stacked_rj", None)
        _set(out, "_batched_rj", None)
        _set(out, "_residual_seams", {})
        _set(out, "_loss_fn", None)
        _set(out, "_loss_grad_fns", {})
        _set(out, "_loss_grad_hessian_fns", {})
        return out

    def with_trial_predicate(self, predicate: Any) -> "InferenceProblem":
        """Attach a cheap JAX-compatible validity check for optimizer trials."""
        if not callable(predicate):
            raise TypeError("trial predicate must be callable.")
        out = copy.copy(self)
        object.__setattr__(out, "trial_predicate", predicate)
        return out

    def trial_is_valid(self, x: Any):
        """Return a scalar boolean without evaluating the PDE residual."""
        if self.trial_predicate is None:
            return jnp.asarray(True)
        return jnp.asarray(self.trial_predicate(jnp.asarray(x)), dtype=bool)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_single_shoot(self, series, forward, theta_spec, groups, t0):
        """Build one union-solve residual over all streams' times."""
        from diff_soup import SaveAt, pdesolve

        init = self.initial_state
        fwd = forward
        seam = self._seam_for(theta_spec)
        n_theta = len(seam.x0)
        decision_to_theta = None
        decision_to_y0 = None
        if isinstance(init, LearnedInitialState):
            y0_arr = jnp.asarray(init.guess)
            dmask = np.asarray(seam.problem.id_mask, dtype=bool)
            if y0_arr.ndim != 1 or y0_arr.size != dmask.size:
                raise ValueError(
                    "LearnedInitialState guess must be a one-dimensional full-state "
                    f"vector of length {dmask.size}; got shape {y0_arr.shape}."
                )
            diff_indices = jnp.asarray(np.where(dmask)[0])
            x0 = np.concatenate((np.asarray(seam.x0), np.asarray(y0_arr)[dmask]))

            def decision_to_theta(x):
                return seam.theta_of(x[:n_theta])

            def decision_to_y0(x):
                return y0_arr.at[diff_indices].set(x[n_theta:])

        else:
            y0 = init.y0 if isinstance(init, FixedInitialState) else None
            y0_arr = _resolve_y0(series, y0)
            x0 = np.asarray(seam.x0)

        schedule = self._schedule_arrays
        residual, plans, union_ts, probe = union_residual_factory(
            seam,
            streams=series,
            y0=y0_arr,
            solver=fwd,
            t0=t0,
            decision_to_theta=decision_to_theta,
            decision_to_y0=decision_to_y0,
            schedule=schedule,
        )

        def residual_builder(selected_solver):
            selected, _plans, _times, _probe = union_residual_factory(
                seam,
                streams=series,
                y0=y0_arr,
                solver=selected_solver,
                t0=t0,
                decision_to_theta=decision_to_theta,
                decision_to_y0=decision_to_y0,
                schedule=schedule,
            )
            return (selected,), None

        # The host baseline at x0 (one forward solve outside the trace) runs on
        # the first request for ``baseline_reached_tend``, not at construction.
        saveat = SaveAt(ts=jnp.asarray(union_ts))
        t_span = (float(union_ts[0]), float(union_ts[-1]))

        def baseline_probe() -> bool:
            theta = (
                seam.theta_of(jnp.asarray(seam.x0))
                if decision_to_theta is None
                else decision_to_theta(jnp.asarray(x0))
            )
            adjoint = getattr(fwd, "adjoint", None)
            if schedule is None:
                sol = pdesolve(
                    seam.problem,
                    fwd,
                    t_span,
                    y0_arr,
                    saveat=saveat,
                    theta=theta,
                    adjoint=adjoint,
                )
            else:
                from diff_soup._schedule import solve_schedule

                sol, _ends = solve_schedule(
                    seam.problem,
                    fwd,
                    schedule,
                    y0_arr,
                    t0=t_span[0],
                    t_end=t_span[1],
                    times=jnp.asarray(union_ts),
                    theta=theta,
                    adjoint=adjoint,
                )
            return bool(_reached_tend(sol.ts, t_span[1]))

        group_names, group_cols = _group_columns(seam, groups)
        _set = object.__setattr__
        _set(self, "solver", fwd)
        _set(self, "data", series)
        _set(self, "_seam", seam)
        _set(self, "_segment_fns", (residual,))
        _set(self, "_residual_builder", residual_builder)
        _set(self, "_batched_r", None)
        _set(self, "_batched_rj", None)
        _set(self, "_x0", np.asarray(x0))
        _set(
            self,
            "regularization",
            _regularizer_for_decision_vector(self.regularization, n_theta, len(x0)),
        )
        _set(self, "_n_seg", 1)
        _set(self, "_baseline_reached_tend", None)
        _set(self, "_baseline_probe", baseline_probe)
        _set(self, "_residual_rows", max((p.row_slice[1] for p in plans), default=0))
        _set(self, "_dv_build", None)
        _set(self, "_stream_names", tuple(p.name for p in plans))
        _set(self, "_stream_slices", tuple(p.row_slice for p in plans))
        _set(self, "_param_group_names", group_names)
        _set(self, "_param_group_cols", group_cols)
        _set(self, "_union_ts", union_ts)
        _set(self, "_solve_probe", probe)

    def _build_multishoot(self, series, forward, theta_spec, groups, trajectory, t0):
        """Build window residuals (observed-IC or decision-variable mode)."""
        windows = trajectory.windows
        continuity = trajectory.continuity
        if windows.is_observed_ic:
            self._build_observed_ic(series, forward, theta_spec, groups, windows)
            return
        if len(series) != 1:
            raise NotImplementedError(
                "Multishoot accepts one Series in decision-variable window mode; "
                f"got {len(series)}. Pass one Series, or use SingleShoot for "
                "multiple observation streams."
            )
        if windows.is_decision_variable:
            self._build_decision_variable(
                series[0], forward, theta_spec, groups, windows, continuity, t0, trajectory.states
            )
        else:
            raise ValueError(f"Multishoot: unsupported Windows mode {windows.mode!r}.")

    @staticmethod
    def _resolve_observed_ic_streams(series, windows):
        """Split the declared streams into the window seed and the scored stream.

        One stream is the single-stream form: it both seeds the window starts and
        is scored, so it must be full-state.  Two streams decouple the roles — the
        named ``source`` seeds the windows and the other is scored through its own
        observable — which is what lets a partial or averaged readout be scored
        while the window starts still come from a known full state.

        A ``Windows.from_states`` declaration carries its seed states itself:
        they become an unscored full-state seed stream over the declared frame
        times, and the problem's one measured stream is scored.

        Returns:
            ``(seed, scored)``; ``scored`` is ``None`` for the single-stream form.
        """
        names = tuple(item.name for item in series)
        if windows.seed_states is not None:
            if len(series) != 1:
                raise ValueError(
                    "Multishoot(Windows.from_states(...)): the seed states score one "
                    f"measured stream; got {len(series)} streams {names}."
                )
            scored = series[0]
            seed_ts = np.asarray(windows.seed_times)
            scored_ts = np.asarray(scored.ts)
            if seed_ts.shape != scored_ts.shape or not np.allclose(seed_ts, scored_ts, atol=1e-12):
                raise ValueError(
                    "Multishoot(Windows.from_states(...)): the seed states' frame times "
                    "must equal the scored stream's frame times, because each window runs "
                    f"between consecutive frames and is scored at its end; got "
                    f"{seed_ts.shape[0]} seed times and {scored_ts.shape[0]} frame times "
                    f"of stream {scored.name!r}."
                )
            seed = Series(
                f"{scored.name}:window-seed",
                windows.seed_times,
                windows.seed_states,
                observable=FullState(),
            )
            return seed, scored
        if len(series) == 1:
            seed = series[0]
            if windows.source is not None and windows.source != seed.name:
                raise ValueError(
                    f"Multishoot(Windows.from_data(source={windows.source!r})): no stream "
                    f"named {windows.source!r}; the inference problem's stream is {seed.name!r}."
                )
            return seed, None
        if len(series) != 2:
            raise ValueError(
                "Multishoot(Windows.from_data()): observed-IC windows take the seed "
                f"stream and one scored stream; got {len(series)} streams {names}. Use "
                "SingleShoot for more streams."
            )
        if windows.source is None:
            raise ValueError(
                "Multishoot(Windows.from_data()): with two streams the seed stream must "
                f"be named, as Windows.from_data(source=...); the streams are {names}."
            )
        if windows.source not in names:
            raise ValueError(
                f"Multishoot(Windows.from_data(source={windows.source!r})): no stream "
                f"named {windows.source!r}; the streams are {names}."
            )
        seed_index = names.index(windows.source)
        seed = series[seed_index]
        scored = series[1 - seed_index]
        seed_ts = np.asarray(seed.ts)
        scored_ts = np.asarray(scored.ts)
        if seed_ts.shape != scored_ts.shape or not np.allclose(seed_ts, scored_ts, atol=1e-12):
            raise ValueError(
                f"Multishoot(Windows.from_data(source={seed.name!r})): the seed and scored "
                "streams must carry the same frame times, because each window runs "
                f"between consecutive seed frames and is scored at its end. Stream "
                f"{seed.name!r} has {seed_ts.shape[0]} frame times and {scored.name!r} "
                f"has {scored_ts.shape[0]}."
            )
        return seed, scored

    def _build_observed_ic(self, series, forward, theta_spec, groups, windows):
        """Build observed-IC windows: each window starts from a measured full state.

        The kept transitions are batched through a single ``vmap`` residual (one
        forward solve traced once, mapped over windows) so the jaxpr equation count
        is independent of the window count.  Per-window failure reporting and the
        per-window cost split read the same batched residual reshaped by window.
        """
        from diff_soup import SaveAt, pdesolve

        seed, scored = self._resolve_observed_ic_streams(series, windows)
        if not observes_full_state(seed.observable):
            raise ValueError(
                "Multishoot(Windows.from_data()): the seed stream must observe the "
                "full state (observable=FullState(), or a measurement declared with "
                "no readout); its values are used as each window's start. To score a "
                "partial or transformed observation, declare the measured full state "
                "as a second stream and name it with Windows.from_data(source=...), "
                "or use SingleShoot."
            )
        if isinstance(self.initial_state, LearnedInitialState):
            raise _learned_ic_not_wired_error("Multishoot(Windows.from_data())")

        fwd = forward
        seam = self._seam_for(theta_spec)

        batched_r, batched_rj, n_seg, baseline_windows = batched_transition_residual_factory(
            seam, series=seed, solver=fwd, scored=scored
        )

        def residual_builder(selected_solver):
            selected, _rj, _n_seg, _baseline = batched_transition_residual_factory(
                seam, series=seed, solver=selected_solver, scored=scored
            )
            return (selected,), selected

        # The host baseline (one solve per window at x0, outside the trace)
        # runs on the first request for ``baseline_reached_tend``.
        saveat_t1 = SaveAt(t1=True)
        adjoint = getattr(fwd, "adjoint", None)

        def baseline_probe() -> bool:
            theta0 = seam.theta_of(jnp.asarray(seam.x0))
            for ti, t1, y_start in baseline_windows:
                sol = pdesolve(
                    seam.problem,
                    fwd,
                    (ti, t1),
                    y_start,
                    saveat=saveat_t1,
                    theta=theta0,
                    adjoint=adjoint,
                )
                if not _reached_tend(sol.ts, t1):
                    return False
            return True

        group_names, group_cols = _group_columns(seam, groups)
        _set = object.__setattr__
        _set(self, "solver", fwd)
        _set(self, "data", tuple(series))
        _set(self, "_seam", seam)
        # The batched residual is the one solver-facing block; the LM engine drives
        # it through the batched path. Keep it as the single segment_fns entry so the
        # per-segment fallback (batching disabled) still produces the full residual.
        _set(self, "_segment_fns", (batched_r,))
        _set(self, "_residual_builder", residual_builder)
        _set(self, "_batched_r", batched_r)
        _set(self, "_batched_rj", batched_rj)
        _set(self, "_x0", np.asarray(seam.x0))
        _set(self, "_n_seg", n_seg)
        _set(self, "_baseline_reached_tend", None)
        _set(self, "_baseline_probe", baseline_probe)
        scored_values = np.asarray((seed if scored is None else scored).values)
        _set(self, "_residual_rows", int(n_seg) * int(np.prod(scored_values.shape[1:])))
        _set(self, "_dv_build", None)
        _set(self, "_stream_names", tuple(item.name for item in series))
        _set(self, "_stream_slices", ())
        _set(self, "_param_group_names", group_names)
        _set(self, "_param_group_cols", group_cols)
        _set(self, "_union_ts", None)
        _set(self, "_solve_probe", None)

    def _build_decision_variable(
        self, s, forward, theta_spec, groups, windows, continuity, t0, states
    ):
        """Eagerly build the decision-variable window stack.

        With
        [LearnedInitialState][diff_soup.inference.fitting.inference_problem.LearnedInitialState],
        the first window's differential start
        ``s_0`` joins the decision vector (``x = [theta | s_0 | s_1 | ... ]``); an
        observation at ``t_0`` then constrains it directly (the window-0 obs row uses
        the learned start without a time step), and the window-state rank diagnostic
        reports any initial-state directions the observable cannot see.
        """
        init = self.initial_state
        learn_initial = isinstance(init, LearnedInitialState)
        fwd = forward
        seam = self._seam_for(theta_spec)

        if learn_initial:
            y0_arr = jnp.asarray(init.guess)
        else:
            y0 = init.y0 if isinstance(init, FixedInitialState) else None
            y0_arr = _resolve_y0((s,), y0)
        t_ic = float(np.asarray(s.ts)[0]) if t0 is None else float(t0)

        dmask = np.asarray(seam.problem.id_mask, dtype=bool)
        has_algebraic = bool((~dmask).any())

        states = states if states is not None else StateBasis.full()
        states.check_state_size(int(dmask.sum()))
        boundaries = boundaries_from_windows(windows, np.asarray(s.ts), t_ic)
        layout = WindowLayout(
            n_theta=len(seam.x0),
            diff_mask=dmask,
            boundaries=boundaries,
            learn_initial=learn_initial,
            n_state=states.width(int(dmask.sum())),
        )
        warn_if_over_ceiling(layout.n_decision)

        atol = float(getattr(fwd, "atol", 1e-9))
        rtol = float(getattr(fwd, "rtol", 1e-6))
        # The manifold sweep is content-addressed: a stored record for this
        # structure, start, layout, basis, first-step policy, and solver is read
        # back instead of sweeping; a fresh sweep is stored by the process that
        # owns the experiment (persist_window_initialization).
        initialization = self.window_initialization
        content_key = None
        if initialization is None:
            content_key = window_initialization_key(
                seam=seam,
                y0=y0_arr,
                times=np.asarray(s.ts),
                boundaries=boundaries,
                states=states,
                restart_step=self.restart_step,
                solver=fwd,
                project_windows=self.project_windows,
            )
            initialization = load_window_initialization(content_key)
            if initialization is not None:
                object.__setattr__(self, "window_initialization", initialization)
        build = DecisionWindowBuild(
            seam=seam,
            solver=fwd,
            series=s,
            layout=layout,
            y0=y0_arr,
            continuity=continuity,
            state_prior_weight=self.state_prior_weight,
            theta_only_steps=self.theta_only_steps,
            project_windows=self.project_windows,
            has_algebraic=has_algebraic,
            atol=atol,
            rtol=rtol,
            failed_sentinel=_FAILED_SENTINEL,
            states=states,
            initialization=initialization,
            restart_step=self.restart_step,
        )
        _set = object.__setattr__
        _set(self, "_window_content_key", content_key)
        _set(self, "_window_record_pending", content_key is not None and initialization is None)

        def residual_builder(selected_solver):
            selected_build = copy.copy(build)
            selected_build.solver = selected_solver
            return (selected_build.residual,), None

        if self.window_initialization is not None:
            baseline_probe = None
            baseline_ok = bool(self.window_initialization.reached_tend)
        else:
            baseline_ok = None

            def baseline_probe() -> bool:
                x0_j = jnp.asarray(build.x0)
                sol = build._propagate_traced(
                    build._theta_pytree(x0_j),
                    build._window_start_state(x0_j, layout.n_windows - 1),
                    jnp.asarray(boundaries[-2], dtype=x0_j.dtype),
                    jnp.asarray(boundaries[-1], dtype=x0_j.dtype),
                    layout.n_windows - 1,
                )
                reached = bool(_reached_tend(sol.ts, float(boundaries[-1])))
                build._baseline_reached_tend = reached
                return reached

        group_names, group_cols = _group_columns(seam, groups)
        _set = object.__setattr__
        _set(self, "solver", fwd)
        _set(self, "data", (s,))
        _set(self, "_seam", seam)
        _set(self, "_segment_fns", (build.residual,))
        _set(self, "_residual_builder", residual_builder)
        _set(self, "_batched_r", None)
        _set(self, "_batched_rj", None)
        _set(self, "_x0", np.asarray(build.x0))
        _set(
            self,
            "regularization",
            _regularizer_for_decision_vector(
                self.regularization,
                layout.n_theta,
                layout.n_decision,
            ),
        )
        _set(self, "_n_seg", 1)
        _set(self, "_baseline_reached_tend", baseline_ok)
        _set(self, "_baseline_probe", baseline_probe)
        _set(self, "_residual_rows", int(build.rows_per_trajectory()))
        _set(self, "_dv_build", build)
        _set(self, "_stream_names", (s.name,))
        _set(self, "_stream_slices", ())
        _set(self, "_param_group_names", group_names)
        _set(self, "_param_group_cols", group_cols)
        _set(self, "_union_ts", None)
        # One solve per window per evaluation, all windows in one batch.
        _set(self, "_solve_probe", {"n": int(build._plan.n_segments)})

    # --- LM-engine duck-typed surface ---
    @property
    def x0(self) -> np.ndarray:
        """Initial flat decision vector."""
        return self._x0

    @property
    def n_params(self) -> int:
        """Number of scalar decision variables."""
        return len(self._x0)

    @property
    def n_segments(self) -> int:
        """Number of residual blocks (1 for single-shoot; windows for observed-IC)."""
        return self._n_seg

    @property
    def segment_fns(self) -> tuple:
        """Forward-resolved residual blocks used by Gauss-Newton solvers."""
        return self._residual_seam("forward")[0]

    def _lowered_problem(self) -> Any:
        """Return the lowered problem used by adjoint capability declarations."""
        source = getattr(self._seam.problem, "source", None)
        return self.problem if source is None else source

    def _resolved_solver(self, ad: str) -> Any:
        """Resolve one differentiation seam's solver without changing the problem."""
        lowered = self._lowered_problem()
        if ad == "forward":
            return resolve_forward_adjoint(self.solver, lowered)
        if ad == "reverse":
            return resolve_reverse_adjoint(self.solver, lowered)
        from diff_soup._errors import ConfigurationError

        raise ConfigurationError(
            f"optsolve differentiation mode must be 'forward' or 'reverse'; got {ad!r}."
        )

    def _residual_seam(self, ad: str) -> tuple:
        """Return cached ``(segments, batched, stacked, solver)`` for one AD mode."""
        cached = self._residual_seams.get(ad)
        if cached is not None:
            return cached
        selected = self._resolved_solver(ad)
        if selected is self.solver:
            segments, batched = self._segment_fns, self._batched_r
        else:
            segments, batched = self._residual_builder(selected)
        stacked = _stacked_residual_fn(segments, batched)
        cached = (segments, batched, stacked, selected)
        self._residual_seams[ad] = cached
        return cached

    def _view_for_ad(self, ad: str) -> "InferenceProblem":
        """Return a shallow child view carrying one resolved residual seam."""
        segments, batched, _stacked, solver = self._residual_seam(ad)
        out = copy.copy(self)
        object.__setattr__(out, "solver", solver)
        object.__setattr__(out, "_segment_fns", segments)
        object.__setattr__(out, "_batched_r", batched)
        if self._dv_build is not None:
            selected_build = copy.copy(self._dv_build)
            selected_build.solver = solver
            object.__setattr__(out, "_dv_build", selected_build)
        return out

    @property
    def stacked_r_fn(self) -> Callable:
        """The jitted concatenated-segment residual ``x -> r``, built once.

        The compiled kernel is cached on the inference problem, so repeated solves
        (e.g. the outer iterations of a penalty loop) reuse it instead of
        retracing the PDE residual. Regularizer rows stay outside — solvers
        evaluate them eagerly so penalty weights never trigger a recompile.
        """
        if self._stacked_r is None:
            _segments, _batched, r_fn, _solver = self._residual_seam("forward")

            from diff_soup.inference.solvers.tangent import value_and_jacfwd

            @jax.jit
            def rj_fn(x):
                return value_and_jacfwd(r_fn, x)

            _set = object.__setattr__
            _set(self, "_stacked_r", r_fn)
            _set(self, "_stacked_rj", rj_fn)
        return self._stacked_r

    @property
    def stacked_rj_fn(self) -> Callable:
        """The jitted ``x -> (r, J)`` companion of ``stacked_r_fn``, built once.

        See [stacked_r_fn][diff_soup.inference.fitting.inference_problem.InferenceProblem.stacked_r_fn].
        """
        if self._stacked_rj is None:
            self.stacked_r_fn  # noqa: B018 — builds and caches both callables
        return self._stacked_rj

    @property
    def batched_r_fn(self) -> Optional[Callable]:
        """The vmapped residual ``x -> r`` (observed-IC multishoot), or ``None``.

        Present (non-``None``) only for observed-transition multishoot, where every
        window runs through one batching construct.  The LM engine drives this O(1)
        path when both batched callables are non-``None``.
        """
        if self._batched_r is None:
            return None
        return self._residual_seam("forward")[1]

    @property
    def batched_rj_fn(self) -> Optional[Callable]:
        """The batched ``x -> (r, J)`` (observed-IC multishoot), or ``None``."""
        batched = self.batched_r_fn
        if batched is None:
            return None
        if self._batched_rj is None:
            from diff_soup.inference.solvers.tangent import value_and_jacfwd

            @jax.jit
            def batched_rj(x):
                return value_and_jacfwd(batched, x)

            object.__setattr__(self, "_batched_rj", batched_rj)
        return self._batched_rj

    @property
    def loss_fn(self) -> Callable:
        """The cached data-misfit loss ``x -> 0.5 * sum(residual(x)**2)``."""
        if self._loss_fn is None:
            residual = self._residual_seam("reverse")[2]
            object.__setattr__(self, "_loss_fn", _loss_fn_from_residual(residual))
        return self._loss_fn

    def loss_and_grad_fn(self, ad: str = "reverse") -> Callable:
        """Return a cached data-misfit value-and-gradient function.

        Args:
            ad: ``"forward"`` for tangent sensitivities or ``"reverse"`` for
                a checkpointed or backend-native reverse pass.
        """
        cached = self._loss_grad_fns.get(ad)
        if cached is not None:
            return cached
        if ad not in ("forward", "reverse"):
            self._resolved_solver(ad)
        supported = self.supported_compositions
        if ad not in supported:
            from diff_soup._errors import ConfigurationError

            raise ConfigurationError(
                f"optsolve differentiation mode {ad!r} is not supported by "
                f"{type(getattr(self.solver, 'adjoint', None)).__name__}; supported "
                f"modes are {sorted(supported)}."
            )
        residual = self._residual_seam(ad)[2]
        cached = _loss_and_grad_from_residual(residual, ad)
        self._loss_grad_fns[ad] = cached
        return cached

    @property
    def supported_compositions(self) -> frozenset[str]:
        """Differentiation modes declared by this problem's resolved adjoints."""
        lowered = self._lowered_problem()
        forward = self._resolved_solver("forward").adjoint
        reverse = self._resolved_solver("reverse").adjoint
        return frozenset(forward.compositions(lowered) | reverse.compositions(lowered))

    def resolved_hessian_composition(self, composition: str = "auto") -> str:
        """Resolve and validate a true-Hessian differentiation composition."""
        supported = self.supported_compositions
        if composition == "auto":
            if "forward-over-reverse" in supported:
                return "forward-over-reverse"
            composition = "forward-over-forward"
        if composition not in ("forward-over-reverse", "forward-over-forward"):
            from diff_soup._errors import ConfigurationError

            raise ConfigurationError(
                "loss_grad_hessian_fn composition must be 'auto', "
                "'forward-over-reverse', or 'forward-over-forward'; "
                f"got {composition!r}."
            )
        if composition not in supported:
            from diff_soup._errors import ConfigurationError

            ad = "reverse" if composition == "forward-over-reverse" else "forward"
            adjoint = getattr(self._resolved_solver(ad), "adjoint", None)
            raise ConfigurationError(
                f"Hessian composition {composition!r} is not supported by "
                f"{type(adjoint).__name__}; supported modes are {sorted(supported)}."
            )
        return composition

    def loss_grad_hessian_fn(self, composition: str = "auto") -> Callable:
        """Return cached data-misfit value, gradient, and true Hessian evaluation."""
        resolved = self.resolved_hessian_composition(composition)
        cached = self._loss_grad_hessian_fns.get(resolved)
        if cached is None:
            ad = "reverse" if resolved == "forward-over-reverse" else "forward"
            cached = _loss_grad_hessian_from_gradient(self.loss_and_grad_fn(ad))
            self._loss_grad_hessian_fns[resolved] = cached
        return cached

    @property
    def unflatten_fn(self) -> Callable:
        """Inverse ravel: flat x -> learnable-theta pytree (θ̃ slice in DV mode)."""
        n_theta = len(self._seam.x0)
        base = self._seam.unflatten
        return lambda x: base(x[:n_theta])

    @property
    def prob_static(self):
        """Frozen theta partition (for warm-start / inspection)."""
        return self._seam.theta_frozen

    @property
    def fits_window_states(self) -> bool:
        """Whether the decision vector holds window start states (decision-variable multishoot)."""
        return self._dv_build is not None

    @property
    def function_columns(self) -> tuple:
        """The columns of the theta block that hold learnable-function (network) weights."""
        from diff_soup.inference.fitting.learnables import learnable_function_leaf_ids

        owners = (self.problem, getattr(self._seam.problem, "source", None))
        ids = frozenset().union(
            *(learnable_function_leaf_ids(owner) for owner in owners if owner is not None)
        )
        if not ids:
            return ()
        return tuple(i for i, leaf_id in enumerate(self._seam.theta_leaf_ids) if leaf_id in ids)

    def values(self, x: Any) -> ParameterValues:
        """Return the parameter values a flat vector in this problem's fit order holds.

        Args:
            x: A vector in the fit's order, such as `fit.params`, `StepInfo.params` or a checkpoint's vector.

        Returns:
            A read-only mapping from each fitted parameter to its value.
        """
        return parameter_values(self.parameter_schema, x)

    @property
    def theta_names(self) -> tuple:
        """Stable flat order of learnable θ scalars.

        This is the scalar order used by ``x0`` and by
        [InequalityConstraints][diff_soup.inference.fitting.constrained.InequalityConstraints].
        It is suitable for naming constraint coordinates without relying on
        undocumented pytree leaf order.
        """
        return self._seam.theta_names

    def persist_window_initialization(self) -> Any:
        """Store this problem's window initialization under its content key.

        Called by the process that owns the experiment once its sweep has run;
        a problem built from a stored record, one given an explicit
        ``window_initialization``, or one without decision-variable windows
        stores nothing. Returns the record's path, or ``None``.
        """
        if not self._window_record_pending or self._dv_build is None:
            return None
        path = store_window_initialization(
            self._window_content_key, self._dv_build.initialization()
        )
        object.__setattr__(self, "_window_record_pending", False)
        return path

    @property
    def residual_rows(self) -> int:
        """Row count of the flat residual, known from the data layout without a trace."""
        return int(self._residual_rows)

    @property
    def residual_row_kinds(self) -> dict:
        """Rows of the flat residual by kind: ``observation``, ``continuity``, ``state_prior``.

        Decision-variable multiple shooting stacks the observation rows,
        one continuity row block per window boundary, and, with a state
        prior, one prior block per decision window; every other trajectory
        has observation rows only.
        """
        rows = int(self._residual_rows)
        build = self._dv_build
        if build is None:
            return {"observation": rows, "continuity": 0, "state_prior": 0}
        layout = build.layout
        width = int(layout.n_state)
        continuity = int(build._plan.cont_pool.shape[0]) * width
        prior = int(layout.n_decision_windows) * width if build.state_prior_weight > 0.0 else 0
        return {
            "observation": rows - continuity - prior,
            "continuity": continuity,
            "state_prior": prior,
        }

    @property
    def baseline_reached_tend(self) -> bool:
        """Whether the forward solve at ``x0`` reaches ``t_end``.

        The solve runs on the host, outside any trace, the first time this is
        read; construction does not pay for it.
        """
        if self._baseline_reached_tend is None:
            object.__setattr__(self, "_baseline_reached_tend", bool(self._baseline_probe()))
        return self._baseline_reached_tend

    @property
    def is_decision_variable(self) -> bool:
        """Whether this is the decision-variable window mode (vs single/observed-IC)."""
        return self._dv_build is not None

    # --- diagnostics-by-name surface ---
    @property
    def stream_names(self) -> tuple:
        """Names of the data streams, in declaration order."""
        return self._stream_names

    @property
    def stream_residual_slices(self) -> tuple:
        """Per-stream ``(start, stop)`` row ranges in the single-shoot residual.

        Empty for multishoot (where rows are organized by window, not stream).
        """
        return self._stream_slices

    @property
    def param_group_names(self) -> tuple:
        """Names of the parameter groups, in declaration order (locals then shared)."""
        return self._param_group_names

    @property
    def param_group_cols(self) -> tuple:
        """Per-group flat-vector column indices (parallel to ``param_group_names``)."""
        return self._param_group_cols

    @property
    def union_save_ts(self):
        """The union save grid of a single-shoot trajectory (or ``None``)."""
        return self._union_ts

    @property
    def trajectory_solves_per_eval(self) -> Optional[int]:
        """Forward solves performed per residual evaluation.

        A single shoot counts its traced solves: 1 once the residual has been
        traced. Fitted window starts solve each window once per evaluation, all in
        one batch, and count those solves. ``None`` for observed window starts.
        """
        return None if self._solve_probe is None else int(self._solve_probe["n"])

    def count_failed(self, x_flat: np.ndarray) -> tuple[int, tuple[int, ...]]:
        """Host re-eval: how many residual blocks hit the failure sentinel.

        Observed-transition multishoot reads the one batched residual and splits it
        by window (one block per window) so a failed window is reported by its index;
        otherwise each segment closure is re-evaluated independently.
        """
        if self._batched_r is not None and self._n_seg > 0:
            r = np.asarray(self._batched_r(jnp.asarray(x_flat)))
            if r.size % self._n_seg == 0:
                blocks = r.reshape((self._n_seg, -1))
                idx = [
                    i for i in range(self._n_seg) if np.any(np.abs(blocks[i]) >= _FAILED_SENTINEL)
                ]
                return len(idx), tuple(idx)
        n_failed = 0
        idx_list: list[int] = []
        for i, fn in enumerate(self._segment_fns):
            k, _ = count_failed_frames(fn, x_flat)
            if k:
                n_failed += 1
                idx_list.append(i)
        return n_failed, tuple(idx_list)

    # --- decision-variable diagnostics + warm-start hooks --------------

    def window_diagnostics(self, x_flat: np.ndarray):
        """Per-window obs-rank + continuity-defect RMS + unrepresented share + INFO notes.

        Returns ``(None, None, None, ())`` outside the decision-variable mode.
        """
        if self._dv_build is None:
            return None, None, None, ()
        return self._dv_build.window_diagnostics(x_flat)

    def window_states_of(self, x_flat: np.ndarray):
        """Final inferred differential states, or ``None`` when the trajectory infers none.

        Decision-variable multishoot returns its window states stacked
        ``(n_blocks, n_diff)``. A single-shoot trajectory that learns its initial
        state returns that one state as a ``(1, n_diff)`` block, so both routes
        report an inferred state the same way.
        """
        if self._dv_build is None:
            if not isinstance(self.initial_state, LearnedInitialState):
                return None
            n_theta = len(self._seam.x0)
            return np.asarray(x_flat)[n_theta:].reshape((1, -1))
        return self._dv_build.window_states_of(x_flat)

    @property
    def window_initialization_record(self):
        """Exact reusable window initialization, or ``None`` outside DV mode."""
        if self._dv_build is None:
            return None
        return self._dv_build.initialization()

    @property
    def residual_dependency_blocks(self) -> tuple:
        """Exact structural residual blocks, when the trajectory declares them.

        Decision-variable multiple shooting exposes one block per window. Other
        trajectory modes retain their existing whole-child residual and return
        an empty tuple so statistics assemblers can choose a safe fallback.
        """
        if self._dv_build is None:
            return ()
        return self._dv_build.residual_dependency_blocks()

    @property
    def residual_dependency_buckets(self) -> tuple:
        """Same-shape structural buckets, when the window plan supports them."""
        if self._dv_build is None:
            return ()
        return self._dv_build.residual_dependency_buckets()

    def theta_only_problem(self) -> "_FrozenStateMultishoot":
        """A reduced theta-only inference problem for the freeze-then-relax warm-start."""
        if self._dv_build is None:
            raise ValueError("theta_only_problem is decision-variable mode only.")
        return _FrozenStateMultishoot(self._dv_build, self.regularization)

    def with_warm_theta(self, x_full: np.ndarray) -> "InferenceProblem":
        """Return a copy whose θ̃ block is warm-started from ``x_full``.

        Only ``_x0`` changes; the eagerly-built closures ride along unchanged.
        """
        if self._dv_build is None:
            return self
        n_theta = self._dv_build.layout.n_theta
        new_x0 = np.asarray(self._x0).copy()
        new_x0[:n_theta] = np.asarray(x_full)[:n_theta]
        _set = object.__setattr__
        out = InferenceProblem.__new__(InferenceProblem)
        for f in dataclasses.fields(self):
            _set(out, f.name, getattr(self, f.name))
        _set(out, "_x0", new_x0)
        return out


def _learned_ic_not_wired_error(where: str):
    """Build the error for LearnedInitialState in the one trajectory that cannot carry it."""
    from diff_soup._errors import ConfigurationError

    return ConfigurationError(
        f"LearnedInitialState with {where}: every window already starts from a measured "
        "full state, so there is no initial differential state left to learn. Use "
        f"FixedInitialState with {where}, or learn the initial state with SingleShoot or "
        "a latent-window Multishoot (Windows.uniform(...) / Windows.at(...)), which place "
        "s_0 in the decision vector and constrain it through the trajectory observations."
    )


@dataclasses.dataclass(frozen=True, init=False)
class _FrozenStateMultishoot:
    """Theta-only reduced view of a decision-variable InferenceProblem.

    The window states are frozen at the manifold-init values; the decision vector
    is only θ̃.  Drives the LM engine through the same residual stack with the
    state blocks pinned.  Submodule-internal: the freeze-then-relax warm-start.
    """

    _build: Any
    regularization: Any
    _x0: np.ndarray

    def __init__(self, build: Any, regularization: Any):
        """Pin the window states at init; expose θ̃ as the only decision variable."""
        _set = object.__setattr__
        _set(self, "_build", build)
        _set(self, "regularization", regularization)
        _set(self, "_x0", np.asarray(build.x0)[: build.layout.n_theta])

    def _full_x(self, theta_flat: jnp.ndarray) -> jnp.ndarray:
        """Reassemble the full x with frozen (init) window states."""
        tail = jnp.asarray(self._build.x0)[self._build.layout.n_theta :]
        return jnp.concatenate([jnp.asarray(theta_flat), tail])

    @property
    def x0(self) -> np.ndarray:
        """Theta-only init vector."""
        return self._x0

    @property
    def n_segments(self) -> int:
        """Single residual block."""
        return 1

    @property
    def segment_fns(self) -> tuple:
        """The full residual stack, but over the theta-only reduced vector."""
        return (lambda th: self._build.residual(self._full_x(th)),)

    @property
    def unflatten_fn(self):
        """Inverse ravel of θ̃ (for reconstruct; the LM reads this)."""
        return self._build.seam.unflatten

    @property
    def prob_static(self):
        """Frozen theta partition."""
        return self._build.seam.theta_frozen
