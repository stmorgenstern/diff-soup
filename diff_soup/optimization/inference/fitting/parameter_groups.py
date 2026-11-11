"""The parameters an inference problem fits: their groups, names, columns, bounds and values."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PyTree

from diff_soup._warnings import warn_user
from diff_soup.optimization.inference.fitting._forward_seam import (
    ForwardSeam,
    GroupedThetaSelection,
    parameter_handle_leaf_ids,
    selected_leaf_ids,
)
from diff_soup.optimization.inference.fitting.learnables import (
    LearnableSelector,
    has_declared_learnables,
    learnables,
)
from diff_soup.problem.parameters import ParameterSchema

# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParameterGroups:
    """Named local and shared learnable-parameter selections. Frozen.

    Each value is a
    [LearnableSelector][diff_soup.optimization.inference.fitting.learnables.LearnableSelector]
    (from ``learnables(problem, names)``), a learnable name, or a sequence of
    names.  ``local`` groups are geometry-specific (e.g. a particle's ``k(x)``);
    ``shared`` groups name constitutive functions intended to be tied across
    geometries by a [JointInferenceProblem][diff_soup.optimization.inference.fitting.joint.JointInferenceProblem]
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


def _group_value_names(value: str | Iterable[str]) -> tuple:
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
    from diff_soup.optimization.inference.fitting.learnables import (
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
    from diff_soup.optimization.inference.fitting._forward_seam import (
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
    """Resolve [ParameterGroups][diff_soup.optimization.inference.fitting.parameter_groups.ParameterGroups].

    Produces a θ-selection spec and group records. Returns ``(theta_selection, groups)``:

    * ``theta_selection`` is what the forward seam selects as learnable — ``None``
      (the problem's declared learnables) when no groups are declared, and otherwise a
      [GroupedThetaSelection]
      [diff_soup.optimization.inference.fitting._forward_seam.GroupedThetaSelection]. Its
      ``order`` puts the flat decision vector in declaration order: shared
      groups first, then local groups, each in the order declared -- the same
      order a joint problem gives its shared and per-experiment blocks. Declared
      names and explicit
      [LearnableSelector][diff_soup.optimization.inference.fitting.learnables.LearnableSelector]
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
        theta_selection: Optional[GroupedThetaSelection] = GroupedThetaSelection(
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

    def __init__(self, schema: ParameterSchema, x: ArrayLike):
        """Read `x` through `schema`, a fit's parameter schema."""
        self._schema = schema
        self._x = x

    def __getitem__(self, parameter: Any) -> PyTree:
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


def parameter_values(schema: Optional[ParameterSchema], x: ArrayLike) -> ParameterValues:
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


def _is_structural_schema(schema: Optional[ParameterSchema]) -> bool:
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


def _recipe_parameter_schema(problem: Any, parameters: ParameterGroups, seam: ForwardSeam):
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
                occurrences=(
                    ParameterOccurrence(owner=None, columns=columns, declarations=(handle,)),
                ),
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


def _group_columns(seam: ForwardSeam, groups: tuple) -> tuple:
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

    An empty [ParameterGroups][diff_soup.optimization.inference.fitting.parameter_groups.ParameterGroups]
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


def _resolve_spec_declared_bounds(inference_problem: Any, problem: Any):
    """Resolve bounds declared on the problem's spec params to flat ``(lb, ub)``.

    A declared bound binds a column when its parameter is a decision variable of
    this inference problem — learnable and selected into the flat vector. Columns are
    found by slot key through its stable Recipe handle (a declared
    parameter's public name need not be a θ-leaf path token). A bounded learnable intentionally
    omitted by a nonempty
    [ParameterGroups][diff_soup.optimization.inference.fitting.parameter_groups.ParameterGroups]
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
