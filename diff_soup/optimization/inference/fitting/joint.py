# LOC > 500 justification: the joint problem ties named shared parameter
# groups across leaf problems and owns BOTH the geometry-axis batched
# residual (homogeneous children vmapped through one kernel) and the
# static-concatenation fallback (heterogeneous children), plus the joint
# decision-vector layout, construction-time validation, and per-child result
# reconstruction.  These share the joint-layout index machinery and belong
# together; the leaf InferenceProblem lives in inference_problem.py.
"""joint.py — fitting several leaf problems with tied shared parameters.

A [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem] composes several leaf
[InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem] children and ties
explicitly named **shared** parameter groups across them while each child keeps
its **local** parameter groups, geometry, solver, data, trajectory, and
trajectory states.

The joint decision vector is ``[shared | local_0, local_1, ... | states_0,
states_1, ...]``: each shared group is represented once; local groups and
trajectory-state blocks stay per child.  The seam threads each child's parameter
pytree explicitly into ``pdesolve``, so a shared scalar in the joint vector fans
out to every child and its gradient sums back automatically.

Two residual paths, picked at construction:

* **homogeneous geometry-axis batching** — when every child shares problem
  structure, solver configuration, parameter-group structure, trajectory
  strategy, and data-stream schema (only numeric leaves differ), the per-child
  residual runs through one ``jax.vmap`` kernel, so the traced program (and its
  jaxpr equation count) is independent of the child count;
* **static concatenation** — heterogeneous children (or latent-window children,
  whose learned initial state has no fixed decision-block offset) each evaluate
  their own residual closure and the rows are concatenated; each distinct
  structure compiles once.

Both expose the duck-typed surface the solver engines read, so
``optsolve(joint, optimizer=...)`` drives a joint fit exactly like a leaf fit.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.inference.engine._plan import n_padded, refuse_shared_storage
from diff_soup.inference.fitting._forward_seam import (
    _FAILED_SENTINEL,
    _solution_failed,
)
from diff_soup.inference.fitting._lane_map import folds_enclosing_batches
from diff_soup.inference.fitting._sharding import (
    device_count,
    identity_order,
    order_children,
    place_children,
    reduce_child_cost,
    reduce_child_lsq,
    run_child_kernel,
)
from diff_soup.inference.fitting.bounds import Bounds
from diff_soup.inference.fitting.inference_problem import (
    InferenceProblem,
    LearnedInitialState,
    _loss_and_grad_from_residual,
    _loss_fn_from_residual,
    _loss_grad_hessian_from_gradient,
    _stacked_residual_fn,
    build_stacked_residual_fns,
)
from diff_soup.inference.fitting.result import ChildFit

# ---------------------------------------------------------------------------
# Per-child column bookkeeping
# ---------------------------------------------------------------------------


def _child_group_cols(child: InferenceProblem, gname: str) -> Optional[tuple]:
    """Return child-θ column indices for parameter group ``gname`` (or ``None``)."""
    names = child.param_group_names
    if gname not in names:
        return None
    return tuple(int(c) for c in child.param_group_cols[names.index(gname)])


def _child_n_theta(child: InferenceProblem) -> int:
    """Number of learnable-θ scalars in this child's decision vector."""
    return len(child.theta_names)


def _child_local_cols(child: InferenceProblem, shared: tuple, owned: tuple = ()) -> tuple:
    """Child-θ columns not tied globally or by owner, in ascending order."""
    n_theta = _child_n_theta(child)
    tied_cols: set = set()
    for gname in shared + owned:
        cols = _child_group_cols(child, gname)
        if cols:
            tied_cols.update(cols)
    return tuple(c for c in range(n_theta) if c not in tied_cols)


def _child_shared_cols(child: InferenceProblem, shared: tuple) -> tuple:
    """Child-θ columns of the shared groups, in ``shared`` order then group order."""
    cols: list[int] = []
    for gname in shared:
        gcols = _child_group_cols(child, gname)
        if gcols:
            cols.extend(gcols)
    return tuple(cols)


def _child_owned_cols(child: InferenceProblem, owned: tuple) -> tuple:
    """Child-θ columns of owner-tied groups in declaration order."""
    cols: list[int] = []
    for gname in owned:
        gcols = _child_group_cols(child, gname)
        if gcols:
            cols.extend(gcols)
    return tuple(cols)


def _unique_in_order(values: tuple) -> tuple:
    """Return equality-unique static values while preserving first occurrence."""
    out: list = []
    for value in values:
        if not any(value == previous for previous in out):
            out.append(value)
    return tuple(out)


def _normalise_ownership(children: tuple, shared: tuple, ownership: Any) -> tuple:
    """Validate owner declarations and prepare one aligned entity partition.

    Returns ``(groups, owner_ids, child_owner_indices, owner_values, mapping)``.
    Every named owner group must exist on every child. Multiple groups may be tied
    together when they induce the same child partition; crossing partitions would
    couple nominally local Schur blocks and are rejected.
    """
    n_children = len(children)
    if ownership is None:
        return (), (), (), (), {}

    if callable(ownership):
        candidate_groups = tuple(g for g in children[0].param_group_names if g not in shared)
        raw: dict = {}
        for group in candidate_groups:
            labels = tuple(ownership(group, i) for i in range(n_children))
            if all(label is None for label in labels):
                continue
            if any(label is None for label in labels):
                raise ConfigurationError(
                    f"JointInferenceProblem ownership callable returned None for only part "
                    f"of group {group!r}; return an owner ID for every child or None "
                    "for every child in that group."
                )
            raw[group] = labels
    elif isinstance(ownership, dict):
        raw = dict(ownership)
    else:
        raise TypeError(
            "JointInferenceProblem 'ownership' must be a mapping, a callable "
            f"owner(group, child), or None; got {type(ownership).__name__}."
        )

    groups = tuple(str(group) for group in raw)
    if len(set(groups)) != len(groups):
        raise ValueError("JointInferenceProblem ownership group names must be unique.")
    if set(groups) & set(shared):
        overlap = sorted(set(groups) & set(shared))
        raise ConfigurationError(
            f"JointInferenceProblem groups {overlap} cannot be both globally shared and "
            "owner-tied; put each group in exactly one ownership level."
        )

    label_vectors: list[tuple] = []
    owner_mapping: dict = {}
    for group, source_group in zip(groups, raw):
        declared = raw[source_group]
        if isinstance(declared, dict):
            expected = set(range(n_children))
            keys = set(declared)
            if keys != expected:
                raise ConfigurationError(
                    f"JointInferenceProblem ownership[{group!r}] must name every child "
                    f"index {sorted(expected)} exactly once; got {sorted(keys)}."
                )
            labels = tuple(declared[i] for i in range(n_children))
        else:
            try:
                labels = tuple(declared)
            except TypeError as exc:
                raise TypeError(
                    f"JointInferenceProblem ownership[{group!r}] must be a child-index "
                    "mapping or a sequence of owner IDs."
                ) from exc
            if len(labels) != n_children:
                raise ConfigurationError(
                    f"JointInferenceProblem ownership[{group!r}] has {len(labels)} owner "
                    f"IDs for {n_children} children."
                )
        for label in labels:
            try:
                hash(label)
            except TypeError as exc:
                raise TypeError(
                    f"JointInferenceProblem owner IDs must be hashable; group {group!r} "
                    f"contains {label!r}."
                ) from exc

        _require_group_on_every_child(children, group)
        if label_vectors and labels != label_vectors[0]:
            raise ConfigurationError(
                "All JointInferenceProblem ownership groups must use the same child partition "
                "and owner IDs. Crossing or differently labeled partitions "
                "couple owner-local blocks and cannot use independent local Schur "
                "factorizations."
            )
        label_vectors.append(labels)

    if not groups:
        return (), (), (), (), {}

    reference_labels = label_vectors[0]
    owner_ids = _unique_in_order(reference_labels)
    rendered_owner_ids = tuple(str(owner) for owner in owner_ids)
    if len(set(rendered_owner_ids)) != len(rendered_owner_ids):
        raise ConfigurationError(
            "JointInferenceProblem owner IDs must have distinct text representations "
            "because those labels form public parameter-group names; got "
            f"{owner_ids!r}."
        )
    child_owner_indices = tuple(
        next(i for i, owner in enumerate(owner_ids) if label == owner) for label in reference_labels
    )
    owner_values: list[np.ndarray] = []
    for owner_index, owner_id in enumerate(owner_ids):
        members = tuple(i for i, oi in enumerate(child_owner_indices) if oi == owner_index)
        owner_values.append(_tied_values(children, groups, members, owner_id, "owner"))

    for group in groups:
        owner_mapping[group] = {
            child: owner_ids[child_owner_indices[child]] for child in range(n_children)
        }
    return groups, owner_ids, child_owner_indices, tuple(owner_values), owner_mapping


def _require_group_on_every_child(children: tuple, group: str) -> None:
    """Refuse a tied group that some child lacks or holds at another width."""
    reference_width = None
    for i, child in enumerate(children):
        cols = _child_group_cols(child, group)
        if not cols:
            raise ConfigurationError(
                f"JointInferenceProblem ownership group {group!r} is missing or has no "
                f"learnable columns on problems[{i}]. Owner-tied groups must "
                "exist on every child."
            )
        if reference_width is None:
            reference_width = len(cols)
        elif len(cols) != reference_width:
            raise ConfigurationError(
                f"JointInferenceProblem ownership group {group!r} has {len(cols)} "
                f"scalar(s) on problems[{i}] but {reference_width} on "
                "problems[0]."
            )


def _tied_values(children: tuple, groups: tuple, members: tuple, label: Any, noun: str):
    """The initial values of one tied block, which every child of it must agree on.

    ``members`` are the indices of the children that read the block; the value
    of each group comes from the first of them and is compared with the others.
    Returns the groups' values concatenated in ``groups`` order.
    """
    pieces: list[np.ndarray] = []
    for group in groups:
        cols0 = _child_group_cols(children[members[0]], group)
        reference = np.asarray(children[members[0]].x0)[list(cols0)]
        for child_index in members[1:]:
            cols = _child_group_cols(children[child_index], group)
            value = np.asarray(children[child_index].x0)[list(cols)]
            if not np.array_equal(value.astype(reference.dtype), reference):
                raise ConfigurationError(
                    f"JointInferenceProblem {noun} {label!r} has inconsistent initial "
                    f"values for group {group!r} across its children "
                    f"(problems[{members[0]}]={reference.tolist()} vs "
                    f"problems[{child_index}]={value.tolist()}). Give every "
                    f"child of one {noun} the same initial value."
                )
        pieces.append(reference)
    if not pieces:
        return np.zeros((0,), dtype=np.asarray(children[members[0]].x0).dtype)
    return np.concatenate(pieces)


def _normalise_levels(children: tuple, shared: tuple, levels: Any) -> tuple:
    """Validate nested levels of tied groups and prepare their member partitions.

    ``levels`` holds one ``(member labels, group names)`` pair per level,
    outermost first: the labels name each child's member of that level, and the
    groups take one value per member of it. Every member of an inner level sits
    under exactly one member of the level above, so the members form a tree.
    A level may tie no group; its members then hold empty blocks.

    Returns ``(level_groups, level_ids, child_levels, level_values, mapping)``:
    per level its group names and its member labels in first-child-occurrence
    order; per child its member index at every level; per level and member the
    initial values of the member's groups, concatenated; and
    ``{group: {child index: member label}}`` for every tied group.

    Raises:
        TypeError: If ``levels`` is not a sequence of such pairs, or a member
            label is not hashable.
        ConfigurationError: If a level labels a different number of children,
            two members of a level print alike, a group is tied twice or also
            shared, a group is missing on a child or differs in width, a member
            holds children under two members of the level above, or the
            children of one member disagree on an initial value.
    """
    n_children = len(children)
    try:
        entries = tuple(levels)
    except TypeError as exc:
        raise TypeError(
            "JointInferenceProblem levels= holds one (member labels, group names) pair per "
            f"level, outermost first; got {type(levels).__name__}."
        ) from exc
    label_rows: list[tuple] = []
    level_groups: list[tuple] = []
    for number, entry in enumerate(entries, start=1):
        try:
            labels, names = entry
            labels = tuple(labels)
            if isinstance(names, str):
                raise TypeError
            names = tuple(names)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "JointInferenceProblem levels= holds one (member labels, group names) pair "
                f"per level, the group names as a tuple of strings; level {number} is "
                f"{entry!r}."
            ) from exc
        if not all(isinstance(name, str) for name in names):
            raise TypeError(
                f"JointInferenceProblem levels= names the groups of level {number} by "
                f"strings; got {names!r}."
            )
        if len(labels) != n_children:
            raise ConfigurationError(
                f"JointInferenceProblem level {number} has {len(labels)} member labels for "
                f"{n_children} children."
            )
        for label in labels:
            try:
                hash(label)
            except TypeError as exc:
                raise TypeError(
                    f"JointInferenceProblem member labels must be hashable; level {number} "
                    f"contains {label!r}."
                ) from exc
        label_rows.append(labels)
        level_groups.append(names)

    tied = tuple(group for names in level_groups for group in names)
    repeated = sorted({group for group in tied if tied.count(group) > 1})
    if repeated:
        raise ConfigurationError(
            f"JointInferenceProblem levels= ties the groups {repeated} more than once; tie "
            "each group at exactly one level."
        )
    if set(tied) & set(shared):
        overlap = sorted(set(tied) & set(shared))
        raise ConfigurationError(
            f"JointInferenceProblem groups {overlap} cannot be both globally shared and "
            "owner-tied; put each group in exactly one ownership level."
        )
    for group in tied:
        _require_group_on_every_child(children, group)

    level_ids: list[tuple] = []
    member_rows: list[tuple] = []
    for number, labels in enumerate(label_rows, start=1):
        index_of: dict = {}
        for label in labels:
            if label not in index_of:
                index_of[label] = len(index_of)
        ids = tuple(index_of)
        rendered = tuple(str(member) for member in ids)
        if len(set(rendered)) != len(rendered):
            raise ConfigurationError(
                f"JointInferenceProblem member labels of level {number} must have distinct "
                "text representations because those labels form public parameter-group "
                f"names; got {ids!r}."
            )
        level_ids.append(ids)
        member_rows.append(tuple(index_of[label] for label in labels))

    for level in range(1, len(member_rows)):
        parent_of: dict = {}
        for child, (member, parent) in enumerate(zip(member_rows[level], member_rows[level - 1])):
            first = parent_of.setdefault(member, (parent, child))
            if first[0] != parent:
                raise ConfigurationError(
                    f"JointInferenceProblem level {level + 1} member "
                    f"{level_ids[level][member]!r} holds children under two members of "
                    f"level {level}: problems[{first[1]}] under "
                    f"{level_ids[level - 1][first[0]]!r} and problems[{child}] under "
                    f"{level_ids[level - 1][parent]!r}. Every member of a level has one "
                    "parent."
                )

    level_values: list[tuple] = []
    mapping: dict = {}
    for level, (groups, ids, row) in enumerate(zip(level_groups, level_ids, member_rows)):
        children_of: list[list[int]] = [[] for _ in ids]
        for child, member in enumerate(row):
            children_of[member].append(child)
        level_values.append(
            tuple(
                _tied_values(children, groups, tuple(indices), ids[member], "member")
                for member, indices in enumerate(children_of)
            )
        )
        for group in groups:
            mapping[group] = {child: ids[member] for child, member in enumerate(row)}
    child_levels = tuple(zip(*member_rows)) if member_rows else tuple(() for _ in children)
    return (
        tuple(level_groups),
        tuple(level_ids),
        tuple(tuple(int(m) for m in row) for row in child_levels),
        tuple(level_values),
        mapping,
    )


def _host_tree_equal(a: Any, b: Any) -> bool:
    """Host boolean equality for pytrees containing arrays and static leaves."""
    try:
        return bool(np.asarray(eqx.tree_equal(a, b)))
    except (TypeError, ValueError):
        try:
            return bool(a == b)
        except (TypeError, ValueError):
            return False


def _same_leaf_shapes(a: Any, b: Any) -> bool:
    """Whether two pytrees have equal structure and array leaf shape/dtype."""
    if jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b):
        return False
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return all(
        getattr(x, "shape", None) == getattr(y, "shape", None)
        and getattr(x, "dtype", None) == getattr(y, "dtype", None)
        for x, y in zip(leaves_a, leaves_b)
    )


def _stack_frozen_theta(children: tuple) -> Any:
    """Stack each child's frozen numeric theta leaves on a leading child axis."""
    frozen = tuple(child._seam.theta_frozen for child in children)

    def _stack(*xs):
        if all(x is None for x in xs):
            return None
        if any(x is None for x in xs):
            raise ValueError("homogeneous children have inconsistent frozen theta structure")
        return jnp.stack(tuple(jnp.asarray(x) for x in xs))

    return jax.tree_util.tree_map(_stack, *frozen, is_leaf=lambda x: x is None)


def _stacked_full_theta(
    children: tuple,
    seam: Any,
    theta_flat: jax.Array,
) -> Any:
    """Reconstruct a batched full-theta pytree with child-specific frozen leaves."""
    learn = jax.vmap(seam.unflatten)(theta_flat)
    return eqx.combine(learn, _stack_frozen_theta(children))


# ---------------------------------------------------------------------------
# Construction-time validation (shared declarations must agree across children)
# ---------------------------------------------------------------------------


def _validate_and_shared_init(children: tuple, shared: tuple, shared_init: dict) -> dict:
    """Validate shared declarations and resolve the joint shared-block values.

    Every shared name must be declared in every child's ``parameters.shared`` and
    resolve to the same scalar count.  Initial values must be exactly equal across
    children (after dtype conversion) unless an explicit joint value is supplied in
    ``shared_init``.

    Returns:
        ``{group_name: np.ndarray}`` — the joint initial value for each shared group.

    Raises:
        ConfigurationError: On a missing/mismatched/inconsistent shared declaration.
    """
    resolved: dict = {}
    for gname in shared:
        ref_cols = None
        per_child_vals: list[np.ndarray] = []
        for i, child in enumerate(children):
            if gname not in child.parameters.shared:
                raise ConfigurationError(
                    f"JointInferenceProblem: shared group {gname!r} is not declared in "
                    f"problems[{i}].parameters.shared. Every name in 'shared' must "
                    "be a shared parameter group on every child problem."
                )
            cols = _child_group_cols(child, gname)
            if not cols:
                raise ConfigurationError(
                    f"JointInferenceProblem: shared group {gname!r} on problems[{i}] "
                    "resolves to no learnable columns; check the group's selector."
                )
            if ref_cols is None:
                ref_cols = cols
            elif len(cols) != len(ref_cols):
                raise ConfigurationError(
                    f"JointInferenceProblem: shared group {gname!r} has {len(cols)} scalar(s) "
                    f"on problems[{i}] but {len(ref_cols)} on problems[0]; shared "
                    "group structure, shape, and dtype must match across children."
                )
            per_child_vals.append(np.asarray(child.x0)[list(cols)])

        if gname in shared_init:
            init = np.asarray(shared_init[gname])
            if init.reshape((-1,)).shape[0] != len(ref_cols):
                raise ConfigurationError(
                    f"JointInferenceProblem: shared_init[{gname!r}] has "
                    f"{init.reshape((-1,)).shape[0]} value(s) but the shared group has "
                    f"{len(ref_cols)} scalar(s)."
                )
            resolved[gname] = init.reshape((-1,)).astype(per_child_vals[0].dtype)
            continue

        ref = per_child_vals[0]
        for i, v in enumerate(per_child_vals[1:], start=1):
            if not np.array_equal(v.astype(ref.dtype), ref):
                raise ConfigurationError(
                    f"JointInferenceProblem: shared group {gname!r} has inconsistent initial "
                    f"values across children (problems[{i}]={v.tolist()} vs "
                    f"problems[0]={ref.tolist()}). Make the shared initial values equal, "
                    f"or pass an explicit joint value via shared_init={{{gname!r}: ...}}."
                )
        resolved[gname] = ref
    return resolved


# ---------------------------------------------------------------------------
# Joint decision-vector layout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JointLayout:
    """Host-static layout of the joint decision vector. Frozen.

    The joint vector is ``[global | level-1 member blocks | ... | level-D member
    blocks | child locals | states]``: each level of tied groups contributes one
    block per member, members in first-child-occurrence order, a block holding
    that level's groups in declared order. With one level the members are the
    owners and this is ``[global | owner_0 .. owner_K | child locals | states]``;
    with no ownership declaration, the tied section is empty and this is the
    original ``[shared | child locals | states]`` layout.

    Attributes:
        shared: The shared group names, in order.
        n_shared: Total shared scalar count (the leading block length).
        shared_slice: ``(start, stop)`` of the shared block.
        shared_cols: Per child, the child-θ columns receiving the shared block
            (position-aligned with the shared block).
        ownership_groups: Every tied group name, level by level.
        owner_ids: The members of the first level (the owners), in
            first-child-occurrence order.
        child_owner_indices: Per child, the integer index into ``owner_ids``.
        owner_cols: Per child, child-θ columns receiving its tied values, in the
            order of ``ownership_groups`` (aligned with ``child_tied_columns``).
        owner_slices: Per owner, the ``(start, stop)`` of its own block.
        owner_group_slices: Per owner and group, the group's joint-vector slice.
        level_ids: Per level, its member labels in first-child-occurrence order.
        level_groups: Per level, the names of the groups tied at it.
        child_levels: Per child, its member index at every level.
        level_slices: Per level and member, the ``(start, stop)`` of its block.
        level_group_slices: Per level, member and group, the group's slice.
        child_tied_columns: Per child, the joint columns of its members' blocks,
            level by level (aligned with ``owner_cols``).
        owner_columns: Per owner, the joint columns of its own block followed by
            the blocks of the members below it, level by level.
        local_cols: Per child, the child-θ columns receiving that child's local block.
        local_slices: Per child, the ``(start, stop)`` of its local block in the joint
            vector.
        state_slices: Per child, the ``(start, stop)`` of its trajectory-state block.
        n_theta_joint: Length of the joint θ-block (shared + all locals).
        n_joint: Total joint decision-vector length.
    """

    shared: tuple
    n_shared: int
    global_slice: tuple
    shared_slice: tuple
    shared_cols: tuple
    ownership_groups: tuple
    owner_ids: tuple
    child_owner_indices: tuple
    owner_cols: tuple
    owner_slices: tuple
    owner_group_slices: tuple
    level_ids: tuple
    level_groups: tuple
    child_levels: tuple
    level_slices: tuple
    level_group_slices: tuple
    child_tied_columns: tuple
    owner_columns: tuple
    local_cols: tuple
    local_slices: tuple
    state_slices: tuple
    n_theta_joint: int
    n_joint: int


@dataclasses.dataclass(frozen=True)
class _FrozenTailRegularizer:
    """View a full-joint regularizer over a frozen state tail."""

    inner: Any
    full_start: Any
    n_active: int

    def _full(self, x):
        """Extend the active decision vector with the frozen state tail."""
        return jnp.concatenate((jnp.asarray(x), jnp.asarray(self.full_start)[self.n_active :]))

    def residual(self, x):
        """Rows of the inner regularizer at the extended vector."""
        return self.inner.residual(self._full(x))

    def penalty(self, x):
        """Penalty of the inner regularizer at the extended vector."""
        return self.inner.penalty(self._full(x))

    def block_rows(self, x):
        """Block rows of the inner regularizer restricted to the active columns."""
        full = self._full(x)
        fallback_column = 0
        for columns, residual, jacobian in self.inner.block_rows(full):
            columns = tuple(int(column) for column in columns)
            keep = tuple(i for i, column in enumerate(columns) if column < self.n_active)
            if keep:
                yield (
                    tuple(columns[i] for i in keep),
                    residual,
                    jnp.asarray(jacobian)[:, jnp.asarray(keep, dtype=jnp.int32)],
                )
            elif self.n_active:
                # A frozen-only row contributes constant stage cost but no
                # gradient. Represent it with one exact-zero active column so
                # the structured regularizer coverage remains explicit.
                yield (
                    (fallback_column,),
                    residual,
                    jnp.zeros(
                        (jnp.asarray(residual).size, 1),
                        dtype=jnp.asarray(residual).dtype,
                    ),
                )


def _level_representatives(child_levels: tuple, level_ids: tuple) -> tuple:
    """Per level and member, the index of the first child under that member."""
    representatives = []
    for level, ids in enumerate(level_ids):
        first: list = [None] * len(ids)
        for child, row in enumerate(child_levels):
            if first[row[level]] is None:
                first[row[level]] = child
        representatives.append(tuple(first))
    return tuple(representatives)


def _build_layout(
    children: tuple,
    shared: tuple,
    ownership_groups: tuple = (),
    owner_ids: tuple = (),
    child_owner_indices: tuple = (),
    *,
    levels: Optional[tuple] = None,
) -> JointLayout:
    """Lay out the joint vector: global, tied blocks level by level, child locals, states.

    ``levels`` is ``(level_groups, level_ids, child_levels)`` as
    [_normalise_levels][diff_soup.inference.fitting.joint._normalise_levels]
    returns them; without it, ``ownership_groups``, ``owner_ids`` and
    ``child_owner_indices`` declare one level whose members are the owners.
    """
    if levels is not None:
        level_groups, level_ids, child_levels = levels
    elif ownership_groups:
        level_groups = (tuple(ownership_groups),)
        level_ids = (tuple(owner_ids),)
        child_levels = tuple((int(owner),) for owner in child_owner_indices)
    else:
        level_groups, level_ids = (), ()
        child_levels = tuple(() for _ in children)
    level_groups = tuple(tuple(groups) for groups in level_groups)
    level_ids = tuple(tuple(ids) for ids in level_ids)
    tied_groups = tuple(group for groups in level_groups for group in groups)

    shared_cols = tuple(_child_shared_cols(c, shared) for c in children)
    owner_cols = tuple(_child_owned_cols(c, tied_groups) for c in children)
    local_cols = tuple(_child_local_cols(c, shared, tied_groups) for c in children)
    n_shared = len(shared_cols[0]) if shared_cols else 0

    offset = n_shared
    representatives = _level_representatives(child_levels, level_ids)
    level_slices: list[tuple] = []
    level_group_slices: list[tuple] = []
    for groups, firsts in zip(level_groups, representatives):
        member_slices: list[tuple] = []
        member_group_slices: list[tuple] = []
        for child_index in firsts:
            member_start = offset
            group_slices: list[tuple] = []
            for group in groups:
                width = len(_child_group_cols(children[child_index], group) or ())
                group_slices.append((offset, offset + width))
                offset += width
            member_slices.append((member_start, offset))
            member_group_slices.append(tuple(group_slices))
        level_slices.append(tuple(member_slices))
        level_group_slices.append(tuple(member_group_slices))

    child_tied_columns = tuple(
        tuple(
            column
            for level, member in enumerate(row)
            for column in range(*level_slices[level][member])
        )
        for row in child_levels
    )
    owner_columns: tuple = ()
    if level_ids:
        # Each member below the first level lies under one owner (the member of
        # the first level above it); its block joins that owner's columns.
        below: list[list[list[int]]] = [[[] for _ in level_ids[1:]] for _ in level_ids[0]]
        for level in range(1, len(level_ids)):
            for member, child_index in enumerate(representatives[level]):
                below[child_levels[child_index][0]][level - 1].append(member)
        owner_columns = tuple(
            tuple(range(*level_slices[0][owner]))
            + tuple(
                column
                for level, members in enumerate(below[owner], start=1)
                for member in members
                for column in range(*level_slices[level][member])
            )
            for owner in range(len(level_ids[0]))
        )

    local_slices = []
    for cols in local_cols:
        local_slices.append((offset, offset + len(cols)))
        offset += len(cols)
    n_theta_joint = offset

    state_slices = []
    for child in children:
        n_state = len(np.asarray(child.x0)) - _child_n_theta(child)
        state_slices.append((offset, offset + n_state))
        offset += n_state

    return JointLayout(
        shared=tuple(shared),
        n_shared=n_shared,
        global_slice=(0, n_shared),
        shared_slice=(0, n_shared),
        shared_cols=shared_cols,
        ownership_groups=tied_groups,
        owner_ids=level_ids[0] if level_ids else (),
        child_owner_indices=tuple(row[0] for row in child_levels) if level_ids else (),
        owner_cols=owner_cols,
        owner_slices=level_slices[0] if level_ids else (),
        owner_group_slices=level_group_slices[0] if level_ids else (),
        level_ids=level_ids,
        level_groups=level_groups,
        child_levels=tuple(tuple(row) for row in child_levels),
        level_slices=tuple(level_slices),
        level_group_slices=tuple(level_group_slices),
        child_tied_columns=child_tied_columns,
        owner_columns=owner_columns,
        local_cols=local_cols,
        local_slices=tuple(local_slices),
        state_slices=tuple(state_slices),
        n_theta_joint=n_theta_joint,
        n_joint=offset,
    )


def _assemble_x0(
    children: tuple,
    layout: JointLayout,
    shared_vals: dict,
    level_vals: tuple = (),
) -> np.ndarray:
    """Build the initial vector from global, tied, child-local, and state blocks.

    ``level_vals`` holds, per level and member, the member's tied values.
    """
    x0 = np.zeros((layout.n_joint,), dtype=np.asarray(children[0].x0).dtype)
    start = 0
    for gname in layout.shared:
        v = np.asarray(shared_vals[gname]).reshape((-1,))
        x0[start : start + v.shape[0]] = v
        start += v.shape[0]
    for values, slices in zip(level_vals, layout.level_slices):
        for value, (lo, hi) in zip(values, slices):
            x0[lo:hi] = np.asarray(value).reshape((-1,))
    for child, cols, (lo, hi) in zip(children, layout.local_cols, layout.local_slices):
        x0[lo:hi] = np.asarray(child.x0)[list(cols)]
    for child, (lo, hi) in zip(children, layout.state_slices):
        n_theta = _child_n_theta(child)
        x0[lo:hi] = np.asarray(child.x0)[n_theta:]
    return x0


def _take_columns(x_joint: jax.Array, columns: tuple) -> jax.Array:
    """The entries of ``x_joint`` at host-static ``columns``: a slice when contiguous."""
    first = columns[0]
    if tuple(columns) == tuple(range(first, first + len(columns))):
        return x_joint[first : first + len(columns)]
    return x_joint[jnp.asarray(columns, dtype=jnp.int32)]


def _reconstruct_child_flat(x_joint: jax.Array, child_idx: int, layout: JointLayout):
    """Reassemble child ``child_idx``'s full flat decision vector from the joint vector.

    Concrete integer-index scatters (never boolean indexing): the global block
    fans into every child, each member block fans into every child under that
    member, and local/state blocks remain child-specific.
    """
    child_shared_cols = layout.shared_cols[child_idx]
    child_owner_cols = layout.owner_cols[child_idx]
    child_local_cols = layout.local_cols[child_idx]
    lo_l, hi_l = layout.local_slices[child_idx]
    lo_s, hi_s = layout.state_slices[child_idx]
    n_theta = len(child_shared_cols) + len(child_owner_cols) + len(child_local_cols)
    n_state = hi_s - lo_s
    x_child = jnp.zeros((n_theta + n_state,), dtype=x_joint.dtype)
    if child_shared_cols:
        x_child = x_child.at[jnp.asarray(child_shared_cols)].set(
            x_joint[layout.shared_slice[0] : layout.shared_slice[1]]
        )
    if child_owner_cols:
        tied = _take_columns(x_joint, layout.child_tied_columns[child_idx])
        x_child = x_child.at[jnp.asarray(child_owner_cols)].set(tied)
    if child_local_cols:
        x_child = x_child.at[jnp.asarray(child_local_cols)].set(x_joint[lo_l:hi_l])
    if n_state:
        x_child = x_child.at[n_theta:].set(x_joint[lo_s:hi_s])
    return x_child


# ---------------------------------------------------------------------------
# Homogeneity detection + per-child residual data
# ---------------------------------------------------------------------------


def _child_mode(child: InferenceProblem) -> str:
    """Classify a child's batched residual mode.

    ``"single"`` — one solve over the horizon; ``"observed_ic"`` — observed-start
    windows with no continuity rows; ``"window"`` — decision-variable windows,
    whose per-window start blocks join the decision vector; ``"latent"`` — a
    learned initial state, which is not a packed child batch because the
    trajectory's first state is itself inferred.
    """
    if isinstance(child.initial_state, LearnedInitialState):
        return "latent"
    if child.is_decision_variable:
        return "window"
    if child._batched_r is not None:
        return "observed_ic"
    return "single"


def _is_homogeneous(children: tuple) -> bool:
    """Whether every child shares structure for the geometry-axis vmap kernel.

    Homogeneous means: same trajectory/residual mode (single-shoot,
    observed-transition, or decision-variable windows — children that learn their
    initial state take the static path), identical problem and solver pytree
    structure, equal θ count, and an equal data schema (stream count, observation
    shapes, and time grids).
    Decision-variable window children additionally have to agree on their window
    structure.  Numeric leaves (parameter values, initial states, observations) may
    differ.
    """
    batchable = ("single", "observed_ic", "window")
    # The batched kernel solves one span without a protocol; a child that follows
    # a schedule is evaluated through its own residual.
    if any(_follows_schedule(c) for c in children):
        return False
    if len(children) < 2:
        return len(children) == 1 and _child_mode(children[0]) in batchable
    modes = {_child_mode(c) for c in children}
    if modes - set(batchable) or len(modes) != 1:
        return False
    ref = children[0]
    if _child_mode(ref) == "window" and not all(
        _same_window_structure(c, ref) for c in children[1:]
    ):
        return False
    ref_nstreams = len(ref.data)
    ref_ts = [np.asarray(s.ts) for s in ref.data]
    ref_vshape = [np.asarray(s.values).shape for s in ref.data]
    for c in children[1:]:
        # The representative solver problem may be reused only when its
        # compile-relevant signature is identical. Child-specific numeric
        # leaves travel through the explicitly batched full-theta pytree below.
        if c._seam.problem.structural_signature != ref._seam.problem.structural_signature:
            return False
        if not _host_tree_equal(c.solver, ref.solver):
            return False
        if _child_n_theta(c) != _child_n_theta(ref) or c.theta_names != ref.theta_names:
            return False
        if not _same_leaf_shapes(c._seam.theta_frozen, ref._seam.theta_frozen):
            return False
        if len(c.data) != ref_nstreams:
            return False
        for s, ref_s, t0, vs in zip(c.data, ref.data, ref_ts, ref_vshape):
            if s.name != ref_s.name:
                return False
            ts = np.asarray(s.ts)
            if ts.shape != t0.shape or not np.allclose(ts, t0):
                return False
            if np.asarray(s.values).shape != vs:
                return False
            if not _host_tree_equal(s.observable, ref_s.observable):
                return False
            mask = None if s.frame_mask is None else np.asarray(s.frame_mask)
            ref_mask = None if ref_s.frame_mask is None else np.asarray(ref_s.frame_mask)
            if (mask is None) != (ref_mask is None):
                return False
            if mask is not None and not np.array_equal(mask, ref_mask):
                return False
    return True


def _child_is_shardable(child: InferenceProblem) -> bool:
    """Whether a child can join a child-batch-sharded group.

    Fixed-initial-state single-shoot, observed-window, and decision-variable window
    children all pack into a batch: their per-child decision blocks have one fixed
    width, so the joint vector reconstructs them with a batched slice. A *latent*
    child (one that learns its initial state) does not, because the trajectory's
    first state is inferred rather than laid out at a fixed offset, so it takes the
    static-concatenation path instead. A child that follows a schedule does not
    either: the batched kernel solves one span without its protocol.
    """
    if _follows_schedule(child):
        return False
    return _child_mode(child) in ("single", "observed_ic", "window")


def _follows_schedule(child: InferenceProblem) -> bool:
    """Whether a child's single shoot follows an operating schedule."""
    return getattr(child, "_schedule_arrays", None) is not None


def _same_structure(a: InferenceProblem, b: InferenceProblem) -> bool:
    """Whether two children share the structure required for one batched kernel.

    The same predicate [_is_homogeneous][diff_soup.inference.fitting.joint._is_homogeneous]
    applies pairwise, used here to gather a
    heterogeneous joint into homogeneous structural groups (each group compiles and
    shards its own kernel).
    """
    if _child_mode(a) != _child_mode(b) or not _child_is_shardable(a):
        return False
    if a._seam.problem.structural_signature != b._seam.problem.structural_signature:
        return False
    if not _host_tree_equal(a.solver, b.solver):
        return False
    if _child_n_theta(a) != _child_n_theta(b) or a.theta_names != b.theta_names:
        return False
    if not _same_leaf_shapes(a._seam.theta_frozen, b._seam.theta_frozen):
        return False
    if len(a.data) != len(b.data):
        return False
    if _child_mode(a) == "window" and not _same_window_structure(a, b):
        return False
    for sa, sb in zip(a.data, b.data):
        if sa.name != sb.name:
            return False
        ta, tb = np.asarray(sa.ts), np.asarray(sb.ts)
        if ta.shape != tb.shape or not np.allclose(ta, tb):
            return False
        if np.asarray(sa.values).shape != np.asarray(sb.values).shape:
            return False
        if not _host_tree_equal(sa.observable, sb.observable):
            return False
        ma = None if sa.frame_mask is None else np.asarray(sa.frame_mask)
        mb = None if sb.frame_mask is None else np.asarray(sb.frame_mask)
        if (ma is None) != (mb is None):
            return False
        if ma is not None and not np.array_equal(ma, mb):
            return False
    return True


def _same_window_structure(a: InferenceProblem, b: InferenceProblem) -> bool:
    """Whether two decision-variable window children share one compiled kernel.

    Beyond the problem structure and data schema every child mode is checked for,
    a window kernel is fixed by its window boundaries, its segment plan, the width
    of its window-state coordinate, the continuity weight, and whether state-prior
    rows are stacked.  Two children agreeing on all of those produce
    element-for-element identical residual layouts and differ only in numeric data.
    """
    build_a, build_b = a._dv_build, b._dv_build
    if not np.array_equal(build_a.layout.boundaries, build_b.layout.boundaries):
        return False
    if build_a.layout.n_state != build_b.layout.n_state:
        return False
    if build_a.layout.n_decision_windows != build_b.layout.n_decision_windows:
        return False
    plan_a, plan_b = build_a._plan, build_b._plan
    if not np.array_equal(plan_a.seg_window, plan_b.seg_window):
        return False
    if not np.allclose(plan_a.seg_t1, plan_b.seg_t1):
        return False
    if not np.array_equal(plan_a.obs_pool, plan_b.obs_pool):
        return False
    if not np.array_equal(plan_a.cont_pool, plan_b.cont_pool):
        return False
    modes_a = build_a.states.modes
    modes_b = build_b.states.modes
    if (modes_a is None) != (modes_b is None):
        return False
    if modes_a is not None and not np.allclose(modes_a, modes_b):
        return False
    if not _host_tree_equal(build_a.continuity, build_b.continuity):
        return False
    if build_a.state_prior_weight != build_b.state_prior_weight:
        return False
    return bool(build_a.project == build_b.project)


def _structural_groups(children: tuple) -> tuple:
    """Partition children into homogeneous structural groups, in first-occurrence order.

    Returns a tuple of index tuples; every child in a group shares one compiled,
    independently sharded kernel.  A fully homogeneous joint yields a single group.
    """
    groups: list[list[int]] = []
    reps: list[InferenceProblem] = []
    for i, child in enumerate(children):
        for g, rep in zip(groups, reps):
            if _same_structure(child, rep):
                g.append(i)
                break
        else:
            groups.append([i])
            reps.append(child)
    return tuple(tuple(g) for g in groups)


# ---------------------------------------------------------------------------
# Homogeneous geometry-axis vmap kernel
# (single-shoot / observed-transition / decision-variable windows)
# ---------------------------------------------------------------------------


def _resolve_child_y0(child: InferenceProblem) -> np.ndarray:
    """The fixed trajectory initial differential state of a child (host array).

    Only children whose initial state is fixed reach the batched kernels; a child
    that learns its initial state is classified ``"latent"`` and reconstructs that
    state from its own slice of the joint decision vector.
    """
    from diff_soup.inference.fitting.inference_problem import FixedInitialState, _resolve_y0

    init = child.initial_state
    y0 = init.y0 if isinstance(init, FixedInitialState) else None
    return np.asarray(_resolve_y0(child.data, y0))


def _single_shoot_data(child: InferenceProblem):
    """Per-child initial state for the single-shoot kernel."""
    return _resolve_child_y0(child)


def _stream_weight(series: Any) -> jax.Array:
    """Return a weight array broadcast to one stream's observed-value shape."""
    if series.weight is None:
        return jnp.ones_like(series.values)
    return jnp.broadcast_to(jnp.asarray(series.weight), series.values.shape)


def _observed_ic_data(child: InferenceProblem):
    """Per-child stacked window starts, targets, and target weights."""
    seed, scored = child._resolve_observed_ic_streams(child.data, child.trajectory.windows)
    target = seed if scored is None else scored
    ts = np.asarray(seed.ts)
    ys = np.asarray(seed.values)
    target_values = np.asarray(target.values)
    n_frames = ts.shape[0]
    if target.frame_mask is None:
        keep_targets = np.ones(n_frames, dtype=bool)
    else:
        keep_targets = np.asarray(target.frame_mask).reshape((-1,)).astype(bool)
    kept = [i for i in range(n_frames - 1) if bool(keep_targets[i + 1])]
    y_starts = np.stack([ys[i] for i in kept])
    targets = np.stack([target_values[i + 1] for i in kept])
    if target.weight is None:
        weights = np.ones_like(targets)
    else:
        weight = np.asarray(target.weight)
        if weight.ndim >= 1 and weight.shape[0] == n_frames:
            weights = np.stack(
                [np.broadcast_to(weight[i + 1], targets[j].shape) for j, i in enumerate(kept)]
            )
        else:
            weights = np.broadcast_to(weight, targets.shape)
    return y_starts, targets, weights


def _child_column_table(layout: JointLayout, n_children: int) -> np.ndarray:
    """Joint columns each child's residual depends on, ``(n_children, n_narrow)``.

    A child sees the shared block, its members' blocks, its own local block, and its
    own trajectory-state block — every other joint column is structurally zero in its
    Jacobian. Declaring that lets the outer reduction differentiate each child
    against its own width instead of the whole joint vector.

    Returns an empty ``(n_children, 0)`` table when the widths are not uniform, which
    tells the caller to fall back to the dense path rather than guess.
    """
    rows: list[list[int]] = []
    for child_index in range(n_children):
        columns = list(range(layout.shared_slice[0], layout.shared_slice[1]))
        columns.extend(layout.child_tied_columns[child_index])
        columns.extend(range(*layout.local_slices[child_index]))
        columns.extend(range(*layout.state_slices[child_index]))
        rows.append(columns)
    widths = {len(row) for row in rows}
    if len(widths) != 1 or not rows:
        return np.zeros((n_children, 0), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def _build_homogeneous_residual(
    children: tuple, layout: JointLayout, mode: str, placement: Optional[Layout] = None
):
    """Build the geometry-axis residual + Jacobian for homogeneous children.

    Reconstruction stays vectorized (one batched scatter, no per-child Python loop),
    and the per-child residual recipe runs across the child axis, so the traced
    program is independent of the child count. With a single-device *placement*
    (or ``None``) the children run through one ``jax.vmap``; with a sharded
    *placement* (a data mesh axis) they fan out over the mesh — the
    per-child solves and their forward Jacobian are partitioned over the child
    axis. The single-device path is the numerical baseline both follow.

    Returns ``(batched_r, batched_rj, rows_per_child)``.
    """
    from diff_soup import SaveAt
    from diff_soup.pdesolve import pdesolve

    n_children = len(children)
    rep = children[0]
    seam = rep._seam
    forward = rep.solver
    adjoint = getattr(forward, "adjoint", None)

    # Reconstruct the (N, n_theta) learnable matrix with batched integer scatters.
    # Column positions may differ between children; the layout matrices preserve
    # each child's declared group ordering.
    # XLA's GPU dynamic-slice/scatter lowering requires one consistent index
    # width.  Keep all layout-derived indices explicitly 32-bit even when the
    # numerical model enables 64-bit floating point values.
    shared_cols = jnp.asarray(layout.shared_cols, dtype=jnp.int32) if layout.n_shared else None
    n_theta = _child_n_theta(rep)
    n_owner = len(layout.owner_cols[0])
    owner_cols = jnp.asarray(layout.owner_cols, dtype=jnp.int32) if n_owner else None
    # Per child, the joint columns of its members' blocks, level by level: one
    # integer table of shape (children, tied width), since a child's tied values
    # need not be one contiguous slice of the joint vector.
    tied_columns = (
        jnp.asarray(layout.child_tied_columns, dtype=jnp.int32).reshape(n_children, n_owner)
        if n_owner
        else None
    )
    n_local = len(layout.local_cols[0])
    local_cols = jnp.asarray(layout.local_cols, dtype=jnp.int32) if n_local else None
    child_rows = jnp.arange(n_children, dtype=jnp.int32)[:, None]

    def reconstruct_stacked(x_joint: jax.Array) -> jax.Array:
        stacked = jnp.zeros((n_children, n_theta), dtype=x_joint.dtype)
        if shared_cols is not None:
            shared_vals = x_joint[layout.shared_slice[0] : layout.shared_slice[1]]
            stacked = stacked.at[child_rows, shared_cols].set(
                jnp.broadcast_to(shared_vals, (n_children, shared_vals.shape[0]))
            )
        if owner_cols is not None:
            stacked = stacked.at[child_rows, owner_cols].set(x_joint[tied_columns])
        if local_cols is not None and n_local:
            lo = layout.local_slices[0][0]
            local_mat = jax.lax.dynamic_slice_in_dim(x_joint, lo, n_children * n_local).reshape(
                n_children, n_local
            )
            stacked = stacked.at[child_rows, local_cols].set(local_mat)
        return stacked

    active = jnp.ones((n_children,), dtype=jnp.bool_)

    # Reconstruct ONE child's learnable theta from the joint vector — the seam the
    # no-gather reduction differentiates per device.  Homogeneous children share the
    # shared/local column layout, so only each child's tied columns and local-block
    # start vary; they ride the child axis as ``tied_columns`` and ``local_starts``
    # so the reconstruction stays correct under sharding.
    rep_shared_cols = (
        jnp.asarray(layout.shared_cols[0], dtype=jnp.int32) if layout.n_shared else None
    )
    rep_owner_cols = jnp.asarray(layout.owner_cols[0], dtype=jnp.int32) if n_owner else None
    rep_local_cols = jnp.asarray(layout.local_cols[0], dtype=jnp.int32) if n_local else None
    shared_lo, shared_hi = layout.shared_slice
    frozen_stacked = _stack_frozen_theta(children)
    local_starts = jnp.asarray(
        [layout.local_slices[i][0] for i in range(n_children)], dtype=jnp.int32
    )

    def _child_theta_from_x(
        x_joint: jax.Array, tied: Optional[jax.Array], local_start: jax.Array
    ) -> jax.Array:
        theta = jnp.zeros((n_theta,), x_joint.dtype)
        if rep_shared_cols is not None:
            theta = theta.at[rep_shared_cols].set(x_joint[shared_lo:shared_hi])
        if rep_owner_cols is not None:
            theta = theta.at[rep_owner_cols].set(x_joint[tied])
        if rep_local_cols is not None:
            local_block = jax.lax.dynamic_slice(x_joint, (local_start,), (n_local,))
            theta = theta.at[rep_local_cols].set(local_block)
        return theta

    def _stack_children(kernel, inputs):
        """Run a per-child *kernel* over the child axis, masking padded children.

        ``inputs`` is a child-axis-leading pytree of the kernel's positional
        arguments. With a sharded *placement* the children fan out over the mesh;
        otherwise it is a plain ``vmap``. Returns the ``(N, rows)`` stack.
        """

        def masked(inp, act):
            r = kernel(*inp)
            if isinstance(r, tuple):
                r = r[0]
            return jnp.where(act, r, jnp.zeros_like(r))

        return run_child_kernel(
            masked, inputs, active, placement, fold=folds_enclosing_batches(forward)
        )

    if mode == "single":
        union_ts = rep._union_ts
        t_span = (float(union_ts[0]), float(union_ts[-1]))
        saveat = SaveAt(ts=jnp.asarray(union_ts))
        slices = rep.stream_residual_slices
        frame_indices = tuple(
            jnp.asarray(np.searchsorted(np.asarray(union_ts), np.asarray(s.ts))) for s in rep.data
        )
        frame_times = tuple(jnp.asarray(union_ts[index]) for index in frame_indices)
        keep_indices = tuple(
            None
            if s.frame_mask is None
            else jnp.asarray(np.where(np.asarray(s.frame_mask).reshape((-1,)))[0])
            for s in rep.data
        )
        observes = tuple(s.observe for s in rep.data)
        y0s = jnp.stack([jnp.asarray(_single_shoot_data(c)) for c in children])
        targets = tuple(
            jnp.stack([jnp.asarray(c.data[i].values) for c in children])
            for i in range(len(rep.data))
        )
        weights = tuple(
            jnp.stack([_stream_weight(c.data[i]) for c in children]) for i in range(len(rep.data))
        )
        rows_per_child = int(slices[-1][1]) if slices else 0

        def child_kernel(theta, y0, targets_c, weights_c):
            sol = pdesolve(
                seam.problem, forward, t_span, y0, saveat=saveat, theta=theta, adjoint=adjoint
            )
            rows = []
            for frame_idx, frame_ts, keep_idx, observe, target, weight in zip(
                frame_indices,
                frame_times,
                keep_indices,
                observes,
                targets_c,
                weights_c,
            ):
                # Parameter-dependent observables are part of the public
                # Measurement contract (wrt_theta=True). The leaf problem
                # passes both trial theta and frame time; the homogeneous joint
                # kernel must do the same or direct kinetic/readout derivatives
                # silently disappear in a multi-experiment fit.
                r = weight * (observe(sol.ys[frame_idx], frame_ts, theta) - target)
                if keep_idx is not None:
                    r = r[keep_idx]
                rows.append(jnp.reshape(r, (-1,)))
            r = jnp.concatenate(rows)
            failed = _solution_failed(sol)
            r = jnp.where(failed, jnp.full_like(r, _FAILED_SENTINEL), r)
            return r, _trial_steps(sol)

        def batched_r(x_joint):
            stacked = reconstruct_stacked(x_joint)
            theta = _stacked_full_theta(children, seam, stacked)
            rs = _stack_children(child_kernel, (theta, y0s, targets, weights))
            return jnp.reshape(rs, (-1,))

        if n_owner:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, y0_i, targets_i, weights_i, tc_i, ls_i = child_static
                flat = _child_theta_from_x(x_joint, tc_i, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                return child_kernel(theta, y0_i, targets_i, weights_i)

            lsq_inputs = (frozen_stacked, y0s, targets, weights, tied_columns, local_starts)
        else:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, y0_i, targets_i, weights_i, ls_i = child_static
                flat = _child_theta_from_x(x_joint, None, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                return child_kernel(theta, y0_i, targets_i, weights_i)

            lsq_inputs = (frozen_stacked, y0s, targets, weights, local_starts)

    else:  # observed_ic
        seed, scored = rep._resolve_observed_ic_streams(rep.data, rep.trajectory.windows)
        target_series = seed if scored is None else scored
        ts = np.asarray(seed.ts)
        if target_series.frame_mask is None:
            keep_targets = np.ones(ts.shape[0], dtype=bool)
        else:
            keep_targets = np.asarray(target_series.frame_mask).reshape((-1,)).astype(bool)
        kept = [i for i in range(ts.shape[0] - 1) if bool(keep_targets[i + 1])]
        t0s = jnp.asarray([float(ts[i]) for i in kept])
        t1s = jnp.asarray([float(ts[i + 1]) for i in kept])
        saveat = SaveAt(t1=True)
        child_data = tuple(_observed_ic_data(c) for c in children)
        y_starts = jnp.stack([jnp.asarray(data[0]) for data in child_data])
        obs_targets = jnp.stack([jnp.asarray(data[1]) for data in child_data])
        obs_weights = jnp.stack([jnp.asarray(data[2]) for data in child_data])
        rows_per_child = len(kept) * int(np.prod(np.asarray(target_series.values).shape[1:]))

        def window_residual(theta, y_starts_c, targets_c, weights_c):
            def one(seg):
                t0_i, t1_i, y_start_i, target_i, weight_i = seg
                sol = pdesolve(
                    seam.problem,
                    forward,
                    (t0_i, t1_i),
                    y_start_i,
                    saveat=saveat,
                    theta=theta,
                    adjoint=adjoint,
                )
                predicted = sol.ys[-1]
                if scored is not None:
                    predicted = target_series.observe(
                        predicted[None],
                        ts=jnp.reshape(t1_i, (1,)),
                        theta=theta,
                    )[0]
                r = weight_i * (predicted - target_i)
                r = jnp.reshape(r, (-1,))
                failed = _solution_failed(sol)
                return jnp.where(failed, jnp.full_like(r, _FAILED_SENTINEL), r), _trial_steps(sol)

            # The joint already maps this kernel over its child/owner axis.
            # A second vmap here would expose nested matrix-batch axes to sparse
            # direct solvers; native cuDSS factor banks intentionally own exactly
            # one such axis.  Scan the short window list instead.  Under the outer
            # child vmap JAX batches the scan body over children, so the solves
            # retain one efficient matrix-batch axis and one static loop axis.
            rs, steps = jax.lax.map(one, (t0s, t1s, y_starts_c, targets_c, weights_c))
            return jnp.reshape(rs, (-1,)), jnp.sum(steps)

        def batched_r(x_joint):
            stacked = reconstruct_stacked(x_joint)
            theta = _stacked_full_theta(children, seam, stacked)
            rs = _stack_children(window_residual, (theta, y_starts, obs_targets, obs_weights))
            return jnp.reshape(rs, (-1,))

        if n_owner:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, ys_i, tgt_i, w_i, tc_i, ls_i = child_static
                flat = _child_theta_from_x(x_joint, tc_i, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                return window_residual(theta, ys_i, tgt_i, w_i)

            lsq_inputs = (
                frozen_stacked,
                y_starts,
                obs_targets,
                obs_weights,
                tied_columns,
                local_starts,
            )
        else:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, ys_i, tgt_i, w_i, ls_i = child_static
                flat = _child_theta_from_x(x_joint, None, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                return window_residual(theta, ys_i, tgt_i, w_i)

            lsq_inputs = (frozen_stacked, y_starts, obs_targets, obs_weights, local_starts)

    if mode == "window":
        # The residual algebra lives once, on the representative's window build; the
        # joint supplies each child's initial state, anchors, targets, and weights and
        # maps that same kernel over the child axis. Sharing the implementation is what
        # keeps a sharded joint numerically identical to the leaf path.
        build = rep._dv_build
        state_lo = layout.state_slices[0][0]
        n_state_block = layout.state_slices[0][1] - state_lo
        state_starts = jnp.asarray(
            [layout.state_slices[i][0] for i in range(n_children)], dtype=jnp.int32
        )
        child_y0s = jnp.stack([jnp.asarray(c._dv_build.y0) for c in children])
        child_anchors = jnp.stack([jnp.asarray(c._dv_build.anchors()) for c in children])
        child_targets = jnp.stack([jnp.asarray(c.data[0].values) for c in children])
        has_weight = any(c.data[0].weight is not None for c in children)
        child_weights = (
            jnp.stack([_stream_weight(c.data[0]) for c in children])
            if has_weight
            else jnp.zeros((n_children,), dtype=child_targets.dtype)
        )
        state_shape = (build.layout.n_decision_windows, build.layout.n_state)
        rows_per_child = build.rows_per_trajectory()

        def window_state_residual(theta, state_flat, y0_c, anchors_c, targets_c, weights_c):
            return build.residual_kernel(
                theta,
                jnp.reshape(state_flat, state_shape),
                y0_c,
                anchors_c,
                targets_c,
                weights_c if has_weight else None,
                parallel_segments=False,
                with_steps=True,
            )

        def batched_r(x_joint):
            stacked = reconstruct_stacked(x_joint)
            theta = _stacked_full_theta(children, seam, stacked)
            states = jax.lax.dynamic_slice_in_dim(
                x_joint, state_lo, n_children * n_state_block
            ).reshape(n_children, n_state_block)
            rs = _stack_children(
                window_state_residual,
                (theta, states, child_y0s, child_anchors, child_targets, child_weights),
            )
            return jnp.reshape(rs, (-1,))

        if n_owner:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, y0_i, anch_i, tgt_i, w_i, tc_i, ls_i, ss_i = child_static
                flat = _child_theta_from_x(x_joint, tc_i, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                state_flat = jax.lax.dynamic_slice(x_joint, (ss_i,), (n_state_block,))
                return window_state_residual(theta, state_flat, y0_i, anch_i, tgt_i, w_i)

            lsq_inputs = (
                frozen_stacked,
                child_y0s,
                child_anchors,
                child_targets,
                child_weights,
                tied_columns,
                local_starts,
                state_starts,
            )
        else:

            def r_child_and_steps(x_joint, child_static):
                frozen_i, y0_i, anch_i, tgt_i, w_i, ls_i, ss_i = child_static
                flat = _child_theta_from_x(x_joint, None, ls_i)
                theta = eqx.combine(seam.unflatten(flat), frozen_i)
                state_flat = jax.lax.dynamic_slice(x_joint, (ss_i,), (n_state_block,))
                return window_state_residual(theta, state_flat, y0_i, anch_i, tgt_i, w_i)

            lsq_inputs = (
                frozen_stacked,
                child_y0s,
                child_anchors,
                child_targets,
                child_weights,
                local_starts,
                state_starts,
            )

    def batched_rj(x_joint):
        def f_aux(p):
            r = batched_r(p)
            return r, r

        jac, r = jax.jacfwd(f_aux, has_aux=True)(x_joint)
        return r, jac

    # Declared column sparsity: each child's residual depends only on the shared
    # block plus its own, so the reduction differentiates against that width rather
    # than the whole joint vector.
    child_columns = _child_column_table(layout, n_children)
    narrow_columns = child_columns if child_columns.shape[1] else None

    def r_child_from_x(x_joint, child_static):
        """One child's residual rows; the step count stays with the reductions."""
        out = r_child_and_steps(x_joint, child_static)
        return out[0] if isinstance(out, tuple) else out

    def evaluate_lsq_ordered(x_joint, order):
        """Reduced statistics plus per-child trial steps for one slot order."""
        return reduce_child_lsq(
            r_child_and_steps,
            x_joint,
            lsq_inputs,
            active,
            placement,
            columns=narrow_columns,
            order=order,
        )

    def evaluate_cost_ordered(x_joint, order):
        """Reduced ``(cost, n_active_rows, any_failed)`` for one slot order."""
        return reduce_child_cost(
            r_child_and_steps, x_joint, lsq_inputs, active, placement, order=order
        )

    balance = _StepBalance(n_children, placement, lsq_inputs, active)
    evaluate_lsq = balance.wrap_lsq(jax.jit(evaluate_lsq_ordered))
    evaluate_cost = balance.wrap_cost(jax.jit(evaluate_cost_ordered))
    evaluate_lsq.step_balance = balance

    return (
        jax.jit(batched_r),
        jax.jit(batched_rj),
        evaluate_lsq,
        evaluate_cost,
        rows_per_child,
        r_child_from_x,
        lsq_inputs,
    )


def _trial_steps(sol: Any) -> jax.Array:
    """The integrator's trial-step count of one solve, as an int32 scalar."""
    stats = getattr(sol, "stats", None)
    steps = getattr(stats, "trial_steps", None)
    if steps is None:
        return jnp.zeros((), dtype=jnp.int32)
    return jnp.asarray(steps, dtype=jnp.int32)


class _StepBalance:
    """Feed integrator step counts to the step planner between evaluations.

    A sharded homogeneous kernel evaluates its children in fixed slots, one
    contiguous block per experiment lane. After each evaluation this object
    reads every child's trial-step count and the measured evaluation time,
    asks the planner for a lane assignment whose predicted makespan saving
    exceeds the measured time of moving the rows, and uses that assignment as
    the slot order of the next evaluation. Slot shapes and the compiled kernel
    never change; the reorder is one collective gather inside the kernel.

    Balancing is active on a sharded placement whose lanes hold more than one
    child when ``Layout.balance_steps`` is ``True``; the step counts are read
    and reported either way. The first evaluation, which includes compilation,
    is never used as a timing sample.

    Each lane runs its children as one batch of fixed width, so a lane costs
    its largest step count times the full-width batch-step time and the
    makespan is the largest step count over all children. The planner models
    exactly that, which is why it predicts no saving from any reassignment
    until the integrator stops charging for children that have finished
    stepping; ``reorder`` applies an assignment directly.
    """

    def __init__(self, n_children: int, placement: Optional[Layout], inputs: Any, active: Any):
        self.n_children = int(n_children)
        self.placement = placement
        self.sharded = _sharding_requested(placement)
        self.order = identity_order(self.n_children, placement) if self.sharded else None
        self.n_lanes = device_count(placement) if self.sharded else 1
        self.capacity = int(np.ceil(self.n_children / self.n_lanes))
        self.enabled = (
            self.sharded and self.capacity > 1 and bool(getattr(placement, "balance_steps", False))
        )
        self.balancer = None
        self.transfer_seconds = None
        self.last_steps = None
        self.evaluations = 0
        self.reassignments = 0
        self._inputs = inputs
        self._active = active

    def _measure_transfer(self) -> float:
        """Time one collective reorder of the inputs onto the mesh."""
        rotated = jnp.asarray(np.roll(self.order, 1))

        def move(order):
            ordered, active = order_children(self._inputs, self._active, order)
            return place_children(ordered, active, self.placement)

        move = jax.jit(move)
        jax.block_until_ready(move(rotated))
        started = time.perf_counter()
        jax.block_until_ready(move(rotated))
        return float(time.perf_counter() - started)

    def feed(self, steps: np.ndarray, seconds: float) -> None:
        """Record one evaluation's step counts and wall time; maybe reassign lanes."""
        from diff_soup.inference.engine._balance import StepBalancer

        steps = np.maximum(np.asarray(steps, dtype=np.int64), 1)
        self.last_steps = steps
        self.evaluations += 1
        if not self.enabled or self.evaluations == 1:
            return
        if self.balancer is None:
            self.balancer = StepBalancer(
                self.n_children,
                self.n_lanes,
                max_steps=1 << 30,
                batch_step_seconds=(1.0,) * self.capacity,
                step_classes=1,
            )
            self.transfer_seconds = self._measure_transfer()
        per_step = float(seconds) / float(steps.max())
        self.balancer.batch_step_seconds = (per_step,) * self.capacity
        assignment = self.balancer.update(
            [int(value) for value in steps], transfer_seconds=self.transfer_seconds
        )
        if assignment.changed:
            self.order = assignment.row_order()
            self.reassignments += 1

    def reorder(self, order: Any) -> None:
        """Use *order* (one child index per slot, ``-1`` empty) from the next evaluation on."""
        order = np.asarray(order, dtype=np.int32)
        if self.order is None or order.shape != self.order.shape:
            raise ValueError("A slot order needs one entry per padded slot of the sharded kernel.")
        real = sorted(int(value) for value in order if value >= 0)
        if real != list(range(self.n_children)):
            raise ValueError("A slot order must place every child exactly once.")
        self.order = order
        self.reassignments += 1

    def wrap_lsq(self, inner: Callable) -> Callable:
        """Return ``x -> (cost, J^T r, J^T J, n_active_rows, any_failed)`` over *inner*."""
        from diff_soup._distributed import local_value

        def evaluate_lsq(x_joint):
            order = None if self.order is None else jnp.asarray(self.order)
            started = time.perf_counter()
            out = inner(x_joint, order)
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - started
            stats, steps = out[:5], out[5]
            if self.sharded:
                self.feed(local_value(steps), elapsed)
            else:
                self.last_steps = np.asarray(steps)
            return stats

        return evaluate_lsq

    def wrap_cost(self, inner: Callable) -> Callable:
        """Return ``x -> (cost, n_active_rows, any_failed)`` over *inner* at the current order."""

        def evaluate_cost(x_joint):
            order = None if self.order is None else jnp.asarray(self.order)
            return inner(x_joint, order)

        return evaluate_cost


class _GroupedStepBalance:
    """The step counts of a heterogeneous joint's structural groups, in child order.

    Each structural group carries its own planner; this view reads their last
    trial-step counts back into the joint's child order. Children of groups
    evaluated on other processes are ``nan``.
    """

    def __init__(self, groups: tuple, n_children: int):
        self._groups = groups
        self._n_children = int(n_children)

    @property
    def last_steps(self) -> Optional[np.ndarray]:
        """Per child, the integrator's trial steps at the last evaluation, or ``None`` before one."""
        steps = np.full(self._n_children, np.nan)
        seen = False
        for indices, balance in self._groups:
            last = getattr(balance, "last_steps", None)
            if last is None:
                continue
            seen = True
            last = np.asarray(last, dtype=float).reshape(-1)
            for position, child in enumerate(indices):
                if position < last.shape[0]:
                    steps[child] = last[position]
        return steps if seen else None

    @property
    def reassignments(self) -> int:
        """Reassignments applied across the groups' planners."""
        return int(sum(int(getattr(balance, "reassignments", 0)) for _i, balance in self._groups))


def _group_col_map(glayout: JointLayout, full_layout: JointLayout, group_indices: tuple):
    """Map a structural group's decision-vector columns onto the joint vector.

    The shared block is the same scalars in both layouts; each group child's local and
    state blocks move to that child's slice in the joint vector.  A group computes its
    statistics over its own columns; this index array scatters them back into the joint
    coordinate system so independent groups sum into one Gauss-Newton system.
    """
    # GPU scatter/gather lowering combines these columns with int32 layout
    # indices. Keep one index width throughout; an int64 group map makes XLA's
    # generated concatenate ill-typed (S32 versus S64) under shard_map.
    cm = np.zeros((glayout.n_joint,), dtype=np.int32)
    cm[0 : glayout.n_shared] = np.arange(0, glayout.n_shared)
    for owner_id, (glo, ghi) in zip(glayout.owner_ids, glayout.owner_slices):
        full_owner = full_layout.owner_ids.index(owner_id)
        flo, fhi = full_layout.owner_slices[full_owner]
        cm[glo:ghi] = np.arange(flo, fhi)
    for j, gi in enumerate(group_indices):
        glo, ghi = glayout.local_slices[j]
        flo, fhi = full_layout.local_slices[gi]
        cm[glo:ghi] = np.arange(flo, fhi)
        gslo, gshi = glayout.state_slices[j]
        fslo, fshi = full_layout.state_slices[gi]
        cm[gslo:gshi] = np.arange(fslo, fshi)
    return cm


def _group_submeshes(groups: tuple, children: tuple, mesh: Any) -> tuple[tuple, tuple]:
    """Assign each structural group a submesh of experiment lanes.

    With at most as many groups as lanes, every group gets a contiguous block of
    lanes sized by its data volume; with more groups than lanes, groups take
    successive waves of one-lane submeshes. Returns ``(submeshes, first_lanes)``.
    """
    from diff_soup._distributed import mesh_groups

    width = int(mesh.shape["data"])
    if len(groups) <= width:
        sizes = [1] * len(groups)
        work = [
            sum(int(np.prod(stream.values.shape)) for i in group for stream in children[i].data)
            for group in groups
        ]
        for _ in range(width - len(groups)):
            selected = max(range(len(groups)), key=lambda i: (work[i] / sizes[i], -i))
            sizes[selected] += 1
        submeshes = mesh_groups(mesh, sizes)
        lanes = tuple(np.cumsum([0, *sizes[:-1]]).tolist())
    else:
        # More structures than lanes need successive waves. Within each
        # wave every structure has a disjoint one-lane submesh.
        single = mesh_groups(mesh, (1,) * width)
        submeshes = tuple(single[i % width] for i in range(len(groups)))
        lanes = tuple(i % width for i in range(len(groups)))
    return submeshes, lanes


def plan_child_data_owners(
    children: tuple,
    shared: tuple,
    ownership: dict,
    placement: Optional[Layout],
    *,
    process_index: Optional[int] = None,
) -> tuple[bool, ...]:
    """Which children's observed data a process must hold under *placement*.

    Mirrors the joint's placement rules: each structural group fans its children
    over the experiment lanes of its submesh in contiguous padded blocks, and
    each owner tier's kernels run on the lane of their owner index. A process
    holds a child's data when it owns a device on any lane that evaluates that
    child. Without sharding, or with one process, every child is held.
    ``process_index`` defaults to this process.

    The children may be lowered over placeholder data: only shapes, times,
    masks, and structure decide the plan.
    """
    n_children = len(children)
    if not _sharding_requested(placement) or jax.process_count() == 1:
        return (True,) * n_children
    from diff_soup._distributed import mesh_groups

    process = jax.process_index() if process_index is None else int(process_index)
    mesh = placement.mesh
    lane_owned = tuple(
        any(device.process_index == process for device in lane.devices.flat)
        for lane in mesh_groups(mesh, (1,) * int(mesh.shape["data"]))
    )
    width = len(lane_owned)
    held = [False] * n_children

    def hold_block(indices: tuple, first_lane: int, n_lanes: int) -> None:
        per_lane = n_padded(len(indices), n_lanes) // n_lanes
        for position, child in enumerate(indices):
            if lane_owned[first_lane + position // per_lane]:
                held[child] = True

    if _is_homogeneous(children):
        hold_block(tuple(range(n_children)), 0, width)
    else:
        groups = _structural_groups(children)
        submeshes, lanes = _group_submeshes(groups, children, mesh)
        for group, first, submesh in zip(groups, lanes, submeshes):
            hold_block(group, first, int(submesh.shape["data"]))
    if ownership:
        _owned, owner_ids, child_owners, _values, _mapping = _normalise_ownership(
            children, shared, ownership
        )
        n_owners = len(owner_ids)
        per_lane = n_padded(n_owners, width) // width
        for child, owner in enumerate(child_owners):
            if owner is not None and lane_owned[int(owner) // per_lane]:
                held[child] = True
    return tuple(held)


def child_lanes(
    children: tuple, shared: tuple, ownership: Any, placement: Optional[Layout]
) -> tuple[tuple[int, int], ...]:
    """Per child, the experiment lane that evaluates it and the process owning that lane.

    An owned child runs where its owner's kernels run: owners fill the lanes in
    contiguous blocks of ``ceil(n_owners / lanes)`` in owner order. A child
    without an owner tier follows the child fan-out of its structural group.
    Without a placement mesh every child is on lane 0 of this process.
    """
    n_children = len(children)
    if not _sharding_requested(placement):
        return ((0, jax.process_index()),) * n_children
    from diff_soup._distributed import mesh_groups

    lanes = mesh_groups(placement.mesh, (1,) * int(placement.mesh.shape["data"]))
    width = len(lanes)
    lane_process = tuple(int(lane.devices.flat[0].process_index) for lane in lanes)
    assigned = [0] * n_children

    def assign_block(indices: tuple, first_lane: int, n_lanes: int) -> None:
        per_lane = n_padded(len(indices), n_lanes) // n_lanes
        for position, child in enumerate(indices):
            assigned[child] = first_lane + position // per_lane

    if _is_homogeneous(children):
        assign_block(tuple(range(n_children)), 0, width)
    else:
        groups = _structural_groups(children)
        submeshes, firsts = _group_submeshes(groups, children, placement.mesh)
        for group, first, submesh in zip(groups, firsts, submeshes):
            assign_block(group, first, int(submesh.shape["data"]))
    if ownership:
        _owned, owner_ids, child_owners, _values, _mapping = _normalise_ownership(
            children, shared, ownership
        )
        per_lane = n_padded(len(owner_ids), width) // width
        for child, owner in enumerate(child_owners):
            if owner is not None:
                assigned[child] = int(owner) // per_lane
    return tuple((lane, lane_process[lane]) for lane in assigned)


def _build_grouped_lsq(
    children: tuple,
    full_layout: JointLayout,
    shared: tuple,
    ownership: dict,
    placement: Optional[Layout],
):
    """Reduced statistics for a heterogeneous joint as a sum over homogeneous groups.

    Each structural group builds its own sharded ``evaluate_lsq`` over its own decision
    vector (the proven homogeneous path, unchanged); this wrapper gathers each group's
    columns from the joint vector, evaluates the group, and scatter-sums the group's
    ``cost`` / ``J^T r`` / ``J^T J`` into joint coordinates.  Groups fan out over the
    mesh independently and never assemble a global Jacobian.

    Returns ``(evaluate_lsq, evaluate_cost)``.
    """
    groups = _structural_groups(children)
    sharded = _sharding_requested(placement)
    if sharded:
        from diff_soup._distributed import mesh_groups, owns
        from diff_soup.inference.hierarchical._submeshes import on_submesh, sum_lanes

        submeshes, lanes = _group_submeshes(groups, children, placement.mesh)
        lane_meshes = mesh_groups(placement.mesh, (1,) * int(placement.mesh.shape["data"]))
    else:
        submeshes = (None,) * len(groups)
        lanes = (0,) * len(groups)
    built = []
    for gidx, submesh, lane in zip(groups, submeshes, lanes):
        if sharded and not owns(submesh):
            continue
        gchildren = tuple(children[i] for i in gidx)
        if ownership:
            group_ownership = {
                group: {j: mapping[i] for j, i in enumerate(gidx)}
                for group, mapping in ownership.items()
            }
            owned, owner_ids, child_owners, _values, _mapping = _normalise_ownership(
                gchildren, shared, group_ownership
            )
        else:
            owned, owner_ids, child_owners = (), (), ()
        glayout = _build_layout(gchildren, shared, owned, owner_ids, child_owners)
        mode = _child_mode(gchildren[0])
        group_placement = dataclasses.replace(placement, mesh=submesh) if sharded else placement
        _br, _brj, ev_lsq, ev_cost, _rows, _child_fn, _inputs = _build_homogeneous_residual(
            gchildren, glayout, mode, group_placement
        )
        built.append((ev_lsq, ev_cost, _group_col_map(glayout, full_layout, gidx), submesh, lane))

    n_joint = full_layout.n_joint
    step_balance = _GroupedStepBalance(
        tuple(
            (gidx, ev_lsq.step_balance)
            for gidx, (ev_lsq, *_rest) in zip(
                (g for g, submesh in zip(groups, submeshes) if not sharded or owns(submesh)), built
            )
        ),
        len(children),
    )

    def evaluate(x, *, cost_only):
        contributions = []
        for ev_lsq, ev_cost, columns, submesh, lane in built:
            point = on_submesh(x, submesh) if sharded else x
            cm = jnp.asarray(columns)
            if cost_only:
                value = ev_cost(point[cm])
            else:
                c, jtr_g, jtj_g, na_g, af_g = ev_lsq(point[cm])
                jtr = jnp.zeros((n_joint,), x.dtype).at[cm].add(jtr_g)
                jtj = jnp.zeros((n_joint, n_joint), x.dtype).at[cm[:, None], cm[None, :]].add(jtj_g)
                value = (c, jtr, jtj, na_g, af_g)
            if sharded:
                first_lane = lane_meshes[lane]
                if owns(first_lane):
                    value = jax.tree.map(lambda leaf: on_submesh(leaf, first_lane), value)
                    contributions.append((lane, value))
            else:
                contributions.append(value)
        if sharded:
            template = (jax.ShapeDtypeStruct((), x.dtype),)
            if not cost_only:
                template += (
                    jax.ShapeDtypeStruct((n_joint,), x.dtype),
                    jax.ShapeDtypeStruct((n_joint, n_joint), x.dtype),
                )
            template += (jax.ShapeDtypeStruct((), jnp.int32),) * 2
            return sum_lanes(placement.mesh, contributions, template=template)
        return jax.tree.map(lambda *items: sum(items[1:], items[0]), *contributions)

    def evaluate_lsq(x):
        return evaluate(x, cost_only=False)

    def evaluate_cost(x):
        return evaluate(x, cost_only=True)

    evaluate_lsq.step_balance = step_balance

    # Processes can own different structures. Keep dispatch outside a global
    # jit; each group's executable and the final sum are compiled separately.
    if sharded:
        return evaluate_lsq, evaluate_cost
    return jax.jit(evaluate_lsq), jax.jit(evaluate_cost)


# ---------------------------------------------------------------------------
# Static-concatenation residual (heterogeneous / latent-window children)
# ---------------------------------------------------------------------------


def _child_residual_fn(child: InferenceProblem, ad: Optional[str] = None) -> Callable:
    """A single ``x_child -> r`` closure for any child trajectory mode."""
    if ad is None:
        batched = child._batched_r
        seg_fns = child._segment_fns
    else:
        seg_fns, batched, _stacked, _solver = child._residual_seam(ad)
    if batched is not None:
        return batched
    if len(seg_fns) == 1:
        return seg_fns[0]
    return lambda x: jnp.concatenate([fn(x) for fn in seg_fns])


def _build_static_residual(
    children: tuple,
    layout: JointLayout,
    x0: np.ndarray,
    *,
    rows_per_child: Optional[tuple] = None,
    ad: Optional[str] = None,
):
    """Build the concatenated residual + Jacobian (each child evaluates its own closure).

    Returns ``(residual, rows_per_child)`` where ``rows_per_child`` is host-static.
    Row counts come from the caller when provided (validated against an abstract
    evaluation of the first child), otherwise from `jax.eval_shape` over
    each child closure -- a child residual is never executed to learn its length.
    """
    child_fns = [_child_residual_fn(c, ad) for c in children]
    x0_j = jnp.asarray(x0)

    def _abstract_rows(index: int) -> int:
        xc = _reconstruct_child_flat(x0_j, index, layout)
        return int(np.prod(jax.eval_shape(child_fns[index], xc).shape))

    if rows_per_child is None:
        # Each child knows its row count from its data layout; tracing every
        # child's solve here would cost a forward trace per experiment.
        rows_per_child = tuple(int(child.residual_rows) for child in children)
    else:
        rows_per_child = tuple(int(rows) for rows in rows_per_child)
        if len(rows_per_child) != len(children):
            raise ValueError(
                f"residual_rows carries {len(rows_per_child)} row counts for "
                f"{len(children)} children"
            )
        if any(rows < 0 for rows in rows_per_child):
            raise ValueError("residual_rows must be nonnegative")
        probe = _abstract_rows(0)
        if rows_per_child[0] != probe:
            raise ValueError(
                "residual_rows disagrees with the abstract child evaluation: "
                f"child 0 declares {rows_per_child[0]} rows but evaluates to {probe}"
            )

    def residual(x_joint):
        rows = []
        for i, fn in enumerate(child_fns):
            xc = _reconstruct_child_flat(x_joint, i, layout)
            rows.append(jnp.reshape(fn(xc), (-1,)))
        return jnp.concatenate(rows)

    return residual, tuple(rows_per_child)


# ---------------------------------------------------------------------------
# Joint regularizer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _JointRegularizer:
    """Concatenate each child's regularizer rows over its reconstructed columns. Frozen."""

    children: tuple
    layout: JointLayout

    def residual(self, x_joint: jax.Array) -> jax.Array:
        """The stacked child regularizer residual rows (length 0 when all are NoReg)."""
        rows = []
        for i, child in enumerate(self.children):
            xc = _reconstruct_child_flat(jnp.asarray(x_joint), i, self.layout)
            r = child.regularization.residual(xc)
            if r.shape[0] > 0:
                rows.append(jnp.reshape(r, (-1,)))
        if not rows:
            return jnp.zeros((0,), dtype=jnp.asarray(x_joint).dtype)
        return jnp.concatenate(rows)

    def penalty(self, x_joint: jax.Array) -> jax.Array:
        """``0.5·‖residual‖²`` of the stacked child regularizer rows."""
        r = self.residual(x_joint)
        return 0.5 * jnp.sum(r**2)


# ---------------------------------------------------------------------------
# Diagnostics naming (θ-block by named shared / local group)
# ---------------------------------------------------------------------------


def _joint_theta_names(children: tuple, layout: JointLayout) -> tuple:
    """Per-scalar names of the global, tied (level by level), then child-local theta blocks."""
    names: list[str] = []
    rep = children[0]
    for col in layout.shared_cols[0]:
        names.append(rep.theta_names[col])
    representatives = _level_representatives(layout.child_levels, layout.level_ids)
    for groups, ids, firsts in zip(layout.level_groups, layout.level_ids, representatives):
        for member_id, child_index in zip(ids, firsts):
            child = children[child_index]
            for group in groups:
                for col in _child_group_cols(child, group) or ():
                    names.append(f"{child.theta_names[col]}@{member_id}")
    for i, (child, cols) in enumerate(zip(children, layout.local_cols)):
        for col in cols:
            names.append(f"{child.theta_names[col]}@{i}")
    return tuple(names)


def _joint_group_cols(children: tuple, layout: JointLayout) -> tuple:
    """Named parameter-group to joint theta columns at every ownership level."""
    group_names: list[str] = []
    group_cols: list[tuple] = []
    # Shared groups: their columns are the leading shared block, by declaration order.
    rep = children[0]
    start = 0
    for gname in layout.shared:
        gcols = _child_group_cols(rep, gname)
        n = len(gcols) if gcols else 0
        group_names.append(gname)
        group_cols.append(tuple(range(start, start + n)))
        start += n
    # Tied groups: one named entry per level, member and group.
    for groups, ids, member_slices in zip(
        layout.level_groups, layout.level_ids, layout.level_group_slices
    ):
        for member_id, group_slices in zip(ids, member_slices):
            for group, (lo, hi) in zip(groups, group_slices):
                group_names.append(f"{group}@{member_id}")
                group_cols.append(tuple(range(lo, hi)))
    # Local groups: one entry per child, columns in that child's local block.
    for i, (child, (lo, _hi), cols) in enumerate(
        zip(children, layout.local_slices, layout.local_cols)
    ):
        col_to_joint = {c: lo + j for j, c in enumerate(cols)}
        for gname in child.parameters.local:
            if gname in layout.ownership_groups:
                continue
            gcols = _child_group_cols(child, gname)
            if not gcols:
                continue
            joint_cols = tuple(col_to_joint[c] for c in gcols if c in col_to_joint)
            if joint_cols:
                group_names.append(f"{gname}@{i}")
                group_cols.append(joint_cols)
    return tuple(group_names), tuple(group_cols)


# ---------------------------------------------------------------------------
# Device placement (child-batch sharding)
# ---------------------------------------------------------------------------


def _sharding_requested(layout: Optional[Layout]) -> bool:
    """Whether a placement asks for child-batch sharding (a data mesh axis)."""
    return layout is not None and layout.experiment_axis is not None and layout.mesh is not None


def _validate_sharding_mesh(mesh: Any) -> None:
    """Reject device meshes that child-batch sharding cannot place (eager, pre-lowering).

    Raises:
        ConfigurationError: If the mesh mixes device platforms.
    """
    devices = list(np.asarray(mesh.devices).reshape(-1))
    platforms = {getattr(d, "platform", None) for d in devices}
    platforms.discard(None)
    if len(platforms) > 1:
        raise ConfigurationError(
            "child-batch sharding requires a single-platform mesh; got devices on "
            f"mixed platforms {sorted(platforms)}. Build the mesh from devices on one "
            "platform (all GPU or all CPU)."
        )


def _validate_layout(layout: Optional[Layout], children: tuple) -> None:
    """Validate a placement against the joint's structure (eager, before lowering).

    ``None`` or a mesh without ``data`` keeps local execution.
    The ``data`` axis selects child sharding: the children fan out over the
    mesh, each device runs a local ``vmap`` of the same compiled child kernel.
    This requires a single-platform mesh and a solver that
    supports parallel batching (the JNDF/JBDF/JIDA JAX integrators).
    Structurally heterogeneous children shard as independent homogeneous
    groups; children that learn their initial state are not supported. A
    data axis of several devices takes backends whose factors are arrays
    (see [refuse_shared_storage][diff_soup.inference.engine._plan.refuse_shared_storage]).

    Args:
        layout: The requested placement, or ``None`` for single-device.
        children: The leaf problems, checked for a batchable solver.

    Raises:
        TypeError: If ``layout`` is neither a [Layout][diff_soup.Layout] nor
            ``None``.
        ConfigurationError: If the mesh mixes platforms, if any child
            learns its initial state or follows a schedule, or if this
            process holds several devices of the data axis and a child's
            solver keeps its factors in native storage.
    """
    if layout is None:
        return
    if not isinstance(layout, Layout):
        raise TypeError(
            "JointInferenceProblem 'layout' must be a diff_soup.Layout or None; "
            f"got {type(layout).__name__}."
        )
    if not _sharding_requested(layout):
        return

    _validate_sharding_mesh(layout.mesh)
    unshardable = tuple(i for i, c in enumerate(children) if not _child_is_shardable(c))
    if unshardable:
        scheduled = [i for i in unshardable if _follows_schedule(children[i])]
        learned = [i for i in unshardable if i not in scheduled]
        if scheduled:
            raise ConfigurationError(
                "child-batch sharding does not support children that follow a schedule "
                "(the batched kernel solves one span without the protocol) or that learn "
                "their initial state; problems"
                f"{scheduled} follow a schedule"
                + (f" and problems{learned} learn their initial state" if learned else "")
                + ". Omit the data axis to run them on one device, or leave the "
                "scheduled experiments out of the sharded study."
            )
        raise ConfigurationError(
            "child-batch sharding does not support children that learn their initial "
            "state (LearnedInitialState infers the trajectory's first state, so the "
            "child's decision blocks have no fixed layout offset); problems"
            f"{list(unshardable)} learn their initial state. Use FixedInitialState "
            "children, or omit the data axis to run on one device. Structurally "
            "heterogeneous fixed-initial-state children are supported and shard as "
            "independent homogeneous groups."
        )
    refuse_shared_storage(children, layout.mesh)


# ---------------------------------------------------------------------------
# The joint problem
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False)
class JointInferenceProblem:
    """Several leaf problems fit together with tied shared parameter groups.

    Frozen, eager-build.  Composes a tuple of leaf
    [InferenceProblem][diff_soup.inference.fitting.inference_problem.InferenceProblem] children and ties the
    named ``shared`` groups across them; each child keeps its local groups,
    geometry, solver, data, trajectory strategy, and trajectory states.

    Construct as::

        joint = JointInferenceProblem(
            problems=(particle_a, particle_b, particle_c),
            shared=("j0",),
        )
        result = optsolve(joint, optimizer=LevenbergMarquardt(...))

    The joint decision vector is ``[global | owner blocks | child locals | states]``.
    ``ownership={"field": {0: "particle-a", 1: "particle-a", ...}}`` ties the
    named group only among children with the same owner ID. With ``ownership=None``
    the owner section is empty and the original shared/local layout is unchanged.
    ``levels=`` ties groups at nested levels instead: one ``(member per child,
    group names)`` pair per level, outermost first, each member of an inner level
    under one member of the level above; the vector then holds the member blocks
    level by level, ``[global | level-1 blocks | level-2 blocks | ... | child
    locals | states]``, and the members of the first level are the owners. When
    every child is structurally homogeneous the per-child residual runs through one
    ``jax.vmap`` kernel (the traced program is independent of the child count);
    heterogeneous children concatenate their own residual closures.

    **Multi-GPU child-batch sharding.**  Pass a [Layout][diff_soup.Layout] with a
    mesh with a ``data`` axis to fan the children across a device mesh::

        from diff_soup import Layout
        from jax.sharding import AxisType, Mesh

        mesh = Mesh(jax.devices()[:2], ("data",), axis_types=(AxisType.Auto,))
        joint = JointInferenceProblem(children, shared=("j0",),
                                 layout=Layout(mesh=mesh))
        result = optsolve(joint, LevenbergMarquardt())   # or BFGS()

    Each device owns a slice of the children and runs the same compiled kernel on its
    slice; a child count that is not a multiple of the device count is padded with
    masked children that contribute nothing.  The optimizer reduces only the objective,
    gradient, and Gauss-Newton matrix across the mesh — the global residual and Jacobian
    are never assembled on one device, so per-device memory scales with the local child
    count.  The single-device path (``layout=None``, or a one-device mesh) is the
    numerical baseline and follows the same code.  Each child's own linear solve stays
    local on its device, with a backend whose factors are arrays (``Dense``,
    ``BlockTridiagonal``, ``MemberBlock``, or ``Woodbury`` / ``Krylov`` over one of
    these); a backend that keeps its factors in native storage (``KLU``, ``SuperLU``,
    ``CuDSS``, alone or inside ``Woodbury`` / ``Krylov``) is refused when this process
    holds several devices of the data axis.  Levenberg-Marquardt and BFGS / gradient
    descent are the supported sharded solvers.

    The ``model`` axis distributes each child's sparse factorization over the devices
    of one model group, with ``CuDSS``, the one backend that factors across devices
    (``Dense`` is refused there). CuDSS keeps its factors in native storage, so a mesh
    with both axes holds one model group per process: launch one process per entry of
    the data axis; a process holding several model groups is refused. Structurally
    heterogeneous children use submeshes.
    Children that learn their initial state raise a
    [ConfigurationError][diff_soup._errors.ConfigurationError]. Decision-variable
    window children with a fixed initial state *are* supported: their per-window
    decision blocks have one fixed width, so the joint reconstructs them with a
    batched slice and the windows of every child run in one mapped propagation.

    Attributes:
        problems: The leaf children, in input order.
        shared: The shared group names tied across children.
        ownership: Owner-tied group declarations, normalized to a mapping from
            group name to every child index and its owner ID (with ``levels=``,
            to every child index and its member of the group's level).
        regularization: A regularizer over the joint vector (concatenates the
            children's own regularizer rows).
        bounds: Optional named box bounds over the JOINT parameter groups
            (shared groups by their name, each child's local groups by the
            joint group naming). Solvers project the replicated joint vector
            in their outer loop, so bounds compose with child-batch sharding
            without touching the sharded kernel.
    """

    problems: tuple
    shared: tuple
    ownership: Any
    data_holders: tuple
    regularization: Any
    bounds: Optional[Bounds]
    _layout: Any
    _placement: Any
    _x0: np.ndarray
    _segment_fns: tuple
    _batched_r: Any
    _batched_rj: Any
    _stacked_r: Any
    _stacked_rj: Any
    _residual_builder: Any
    _residual_seams: dict
    _loss_fn: Any
    _loss_grad_fns: dict
    _loss_grad_hessian_fns: dict
    _evaluate_lsq: Any
    _evaluate_cost: Any
    _owner_residual_fns: Any
    _owner_reducer_fns: Any
    _owner_reducer_inputs: Any
    _owner_narrow_kernel_cache: Any
    _owner_point_plan_cache: Any
    _restricted_initialization: bool
    _trial_predicate: Any
    _n_seg: int
    _rows_per_child: tuple
    _homogeneous: bool
    _theta_names: tuple
    _param_group_names: tuple
    _param_group_cols: tuple
    _baseline_reached_tend: Optional[bool]
    _level_names: tuple

    def __init__(
        self,
        problems: tuple,
        *,
        shared: tuple = (),
        shared_init: Optional[dict] = None,
        ownership: Any = None,
        layout: Optional[Layout] = None,
        bounds: Optional[Bounds] = None,
        residual_rows: Optional[tuple] = None,
        data_holders: Optional[tuple] = None,
        levels: Any = None,
        level_names: Optional[tuple] = None,
    ):
        """Validate shared declarations, lay out the joint vector, build the residual.

        Args:
            problems: The leaf children, in input order.
            shared: Names of the parameter groups tied across children.
            shared_init: Optional initial value per shared group, overriding the
                children's own initial values when they disagree.
            ownership: Either ``{group_name: {child_index: owner_id}}``, a
                sequence-valued equivalent, a callable ``owner(group, child)``,
                or ``None``. Groups tied at this level must induce the same child
                partition so each owner remains an independent local block.
            layout: Optional device placement.  ``None`` (default) fits on one
                device.  A [Layout][diff_soup.Layout] with a ``mesh`` and
                ``data`` axis fans the children out across the mesh
                (child-batch sharding); without a ``data`` axis it records
                placement but keeps single-device execution.
            bounds: Optional [Bounds][diff_soup.inference.fitting.bounds.Bounds]
                keyed by the joint parameter-group names.
            residual_rows: Optional precomputed per-child residual row counts,
                in child order.  When provided (for example from a persisted
                structure record) the heterogeneous construction validates the
                first child against an abstract evaluation and skips deriving
                the remaining counts; when omitted, counts come from
                `jax.eval_shape` -- construction never executes a child
                residual either way.  Ignored for homogeneous joints, whose
                row counts come from the shared batched structure.
            data_holders: Optional tuple with one frozenset of process indices
                per child naming the processes that loaded that child's
                observed data (see ``plan_child_data_owners``); defaults to
                every process for every child.
            levels: Optional nested levels of tied groups, in place of
                ``ownership``: one ``(member labels, group names)`` pair per
                level, outermost first, with one member label per child. Each
                named group takes one value per member of its level, and every
                member of an inner level sits under one member of the level
                above. More than one level cannot be placed on a data mesh.
            level_names: Optional names of the levels, one per entry of
                ``levels``, used in messages.

        Raises:
            ConfigurationError: If both ``ownership`` and ``levels`` are given,
                if more than one level is given with a data mesh, if the
                declarations do not fit the children, or if this process
                holds several devices of the data axis and a child's solver
                keeps its factors in native storage.
        """
        placement = layout  # the device-placement Layout; `layout` below is the vector layout
        children = tuple(problems)
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError(
                f"JointInferenceProblem 'bounds' must be a Bounds instance; got {type(bounds).__name__}."
            )
        if len(children) < 1:
            raise ValueError("JointInferenceProblem requires at least one child problem.")
        if not all(isinstance(c, InferenceProblem) for c in children):
            raise TypeError(
                "JointInferenceProblem 'problems' must all be InferenceProblem objects."
            )
        shared = tuple(shared)
        shared_init = dict(shared_init) if shared_init is not None else {}
        if len(set(shared)) != len(shared):
            raise ValueError("JointInferenceProblem 'shared' group names must be unique.")

        if levels is not None and ownership is not None:
            raise ConfigurationError(
                "JointInferenceProblem takes the owner-tied groups either in ownership= or "
                "in levels=; pass one of them."
            )

        shared_vals = _validate_and_shared_init(children, shared, shared_init)
        if levels is None:
            owned, owner_ids, child_owners, owner_vals, ownership_map = _normalise_ownership(
                children, shared, ownership
            )
            layout = _build_layout(children, shared, owned, owner_ids, child_owners)
            level_vals = (owner_vals,) if owned else ()
        else:
            level_groups, level_ids, child_levels, level_vals, ownership_map = _normalise_levels(
                children, shared, levels
            )
            if (
                len(level_ids) > 1
                and isinstance(placement, Layout)
                and _sharding_requested(placement)
            ):
                raise ConfigurationError(
                    "JointInferenceProblem places children on a device mesh by the owners of "
                    f"one level, and levels= declares {len(level_ids)}; build it without a "
                    "data mesh."
                )
            layout = _build_layout(children, shared, levels=(level_groups, level_ids, child_levels))
        x0 = _assemble_x0(children, layout, shared_vals, level_vals)
        if level_names is None:
            level_names = tuple(f"level {n}" for n in range(1, len(layout.level_ids) + 1))
        else:
            named = () if isinstance(level_names, str) else tuple(level_names)
            if len(named) != len(layout.level_ids):
                raise ConfigurationError(
                    "JointInferenceProblem level_names= names each of the "
                    f"{len(layout.level_ids)} levels once; got {level_names!r}."
                )
            level_names = named

        homogeneous = _is_homogeneous(children)
        _validate_layout(placement, children)
        if homogeneous:
            mode = _child_mode(children[0])
            (
                batched_r,
                batched_rj,
                evaluate_lsq,
                evaluate_cost,
                rows,
                _child_fn,
                _child_inputs,
            ) = _build_homogeneous_residual(children, layout, mode, placement)
            segment_fns = (batched_r,)
            rows_per_child = tuple(rows for _ in children)
            n_seg = len(children)
        else:
            residual, rows_per_child = _build_static_residual(
                children, layout, x0, rows_per_child=residual_rows
            )
            batched_r, batched_rj = None, None
            # A heterogeneous joint placed on a mesh shards as a sum over its homogeneous
            # structural groups; off the sharded path it keeps the static residual.
            if _sharding_requested(placement):
                evaluate_lsq, evaluate_cost = _build_grouped_lsq(
                    children, layout, shared, ownership_map, placement
                )
            else:
                evaluate_lsq, evaluate_cost = None, None
            segment_fns = (residual,)
            n_seg = len(children)

        theta_names = _joint_theta_names(children, layout)
        group_names, group_cols = _joint_group_cols(children, layout)
        _set = object.__setattr__
        _set(self, "problems", children)
        _set(self, "shared", shared)
        _set(self, "ownership", ownership_map)
        everyone = frozenset(range(jax.process_count()))
        _set(
            self,
            "data_holders",
            (everyone,) * len(children) if data_holders is None else tuple(data_holders),
        )
        _set(self, "bounds", bounds)
        _set(self, "regularization", _JointRegularizer(children, layout))
        _set(self, "_layout", layout)
        _set(self, "_placement", placement)
        _set(self, "_x0", np.asarray(x0))
        _set(self, "_segment_fns", segment_fns)
        _set(self, "_batched_r", batched_r)
        _set(self, "_batched_rj", batched_rj)
        _set(self, "_stacked_r", None)
        _set(self, "_stacked_rj", None)
        _set(self, "_residual_builder", None)
        _set(self, "_residual_seams", {})
        _set(self, "_loss_fn", None)
        _set(self, "_loss_grad_fns", {})
        _set(self, "_loss_grad_hessian_fns", {})
        _set(self, "_evaluate_lsq", evaluate_lsq)
        _set(self, "_evaluate_cost", evaluate_cost)
        _set(self, "_owner_residual_fns", None)
        _set(self, "_owner_reducer_fns", None)
        _set(self, "_owner_reducer_inputs", None)
        # Created here so warm-started and regularized views (shallow copies)
        # share one cache of owner kernels and their executables.
        _set(self, "_owner_narrow_kernel_cache", {})
        _set(self, "_owner_point_plan_cache", None)
        _set(self, "_restricted_initialization", False)
        _set(self, "_trial_predicate", None)
        _set(self, "_n_seg", n_seg)
        _set(self, "_rows_per_child", tuple(rows_per_child))
        _set(self, "_homogeneous", homogeneous)
        _set(self, "_theta_names", theta_names)
        _set(self, "_param_group_names", group_names)
        _set(self, "_param_group_cols", group_cols)
        _set(self, "_baseline_reached_tend", None)
        _set(self, "_level_names", tuple(str(name) for name in level_names))

    # --- LM-engine duck-typed surface ---
    @property
    def x0(self) -> np.ndarray:
        """Joint initial vector ``[global | owners | child locals | states]``."""
        return self._x0

    def with_start(self, start: Any) -> "JointInferenceProblem":
        """Return an immutable warm-started view with unchanged residual kernels.

        Joint construction compiles the experiment structure and its residual
        kernels once.  Statistical front ends frequently need to reuse that
        structure with a fitted or user-supplied physical-coordinate start;
        rebuilding every child just to replace ``x0`` is both expensive and
        error-prone.  The decision-vector layout is therefore validated here
        while the compiled closures are shared by a shallow immutable copy.
        """
        start = np.asarray(start, dtype=np.asarray(self._x0).dtype)
        if start.shape != np.asarray(self._x0).shape:
            raise ValueError(
                f"JointInferenceProblem start has shape {start.shape}; expected "
                f"{np.asarray(self._x0).shape}."
            )
        if not np.all(np.isfinite(start)):
            raise ValueError("JointInferenceProblem start must contain only finite values.")
        warmed = copy.copy(self)
        object.__setattr__(warmed, "_x0", start.copy())
        return warmed

    def with_regularization(self, regularization: Any) -> "JointInferenceProblem":
        """Return a view carrying structured extra least-squares rows.

        The regularization supplies ``residual(x)`` plus ``block_rows(x)``.
        ``block_rows`` yields ``(columns, residual, jacobian)`` triples that
        declare which global or owner-local columns each extra row touches,
        with the Jacobian taken only with respect to ``columns``.

        This method deliberately refuses to replace existing child
        regularization. Composition requires an explicit ownership declaration
        for both sources and must not silently discard either one.
        """
        if not callable(getattr(regularization, "residual", None)) or not callable(
            getattr(regularization, "block_rows", None)
        ):
            raise TypeError(
                "JointInferenceProblem structured regularization requires residual(x) "
                "and block_rows(x) methods."
            )
        existing = self.regularization.residual(jnp.asarray(self._x0))
        if int(existing.shape[0]) != 0:
            raise ValueError(
                "JointInferenceProblem cannot replace existing child regularization with "
                "structured rows; compose both with explicit ownership first."
            )
        warmed = copy.copy(self)
        object.__setattr__(warmed, "regularization", regularization)
        return warmed

    def with_trial_predicate(self, predicate: Any) -> "JointInferenceProblem":
        """Return a view with a cheap JAX-compatible joint trial validator."""
        if not callable(predicate):
            raise TypeError("trial predicate must be callable.")
        out = copy.copy(self)
        object.__setattr__(out, "_trial_predicate", predicate)
        return out

    def trial_is_valid(self, x: Any):
        """Validate a joint point before any residual/PDE trial is launched."""
        point = jnp.asarray(x)
        valid = jnp.asarray(True)
        if self._trial_predicate is not None:
            valid = valid & jnp.asarray(self._trial_predicate(point), dtype=bool)
        for child_index, child in enumerate(self.problems):
            checker = getattr(child, "trial_is_valid", None)
            if checker is not None:
                valid = valid & jnp.asarray(
                    checker(_reconstruct_child_flat(point, child_index, self._layout)),
                    dtype=bool,
                )
        return valid

    @property
    def n_params(self) -> int:
        """Total scalar parameters in the joint vector."""
        return len(self._x0)

    @property
    def n_segments(self) -> int:
        """Residual-block count (one per child)."""
        return self._n_seg

    @property
    def residual_rows_per_child(self) -> tuple:
        """Host-static residual row count contributed by each child."""
        return self._rows_per_child

    @property
    def n_residuals(self) -> int:
        """Total host-static data-residual rows across all children."""
        return int(sum(self._rows_per_child))

    @property
    def segment_fns(self) -> tuple:
        """The forward-resolved residual closure used by Gauss-Newton solvers."""
        return self._residual_seam("forward")[0]

    def _residual_seam(self, ad: str) -> tuple:
        """Return cached residual and reduced evaluators for one AD mode."""
        cached = self._residual_seams.get(ad)
        if cached is not None:
            return cached
        if self._residual_builder is not None:
            residual = self._residual_builder(ad)
            stacked = _stacked_residual_fn((residual,))
            cached = ((residual,), None, None, stacked, None, None)
            self._residual_seams[ad] = cached
            return cached
        children = tuple(child._view_for_ad(ad) for child in self.problems)
        if self._homogeneous:
            mode = _child_mode(children[0])
            (
                batched,
                batched_rj,
                evaluate_lsq,
                evaluate_cost,
                _rows,
                _child_fn,
                _child_inputs,
            ) = _build_homogeneous_residual(children, self._layout, mode, self._placement)
            segments = (batched,)
        else:
            residual, _rows = _build_static_residual(
                children,
                self._layout,
                self._x0,
                rows_per_child=self._rows_per_child,
            )
            batched, batched_rj = None, None
            if _sharding_requested(self._placement):
                evaluate_lsq, evaluate_cost = _build_grouped_lsq(
                    children,
                    self._layout,
                    self.shared,
                    self.ownership,
                    self._placement,
                )
            else:
                evaluate_lsq, evaluate_cost = None, None
            segments = (residual,)
        stacked = _stacked_residual_fn(segments, batched)
        cached = (segments, batched, batched_rj, stacked, evaluate_lsq, evaluate_cost)
        self._residual_seams[ad] = cached
        return cached

    @property
    def stacked_r_fn(self) -> Callable:
        """The jitted concatenated-segment residual ``x -> r``, built once per joint.

        Heterogeneous joints solve through this pair; caching it means repeated
        solves reuse the compiled kernels instead of retracing every child.
        """
        if self._stacked_r is None:
            _segments, _batched, rj_fn, r_fn, _lsq, _cost = self._residual_seam("forward")
            if rj_fn is None:
                _r, rj_fn = build_stacked_residual_fns((_segments[0],))
            _set = object.__setattr__
            _set(self, "_stacked_r", r_fn)
            _set(self, "_stacked_rj", rj_fn)
        return self._stacked_r

    @property
    def stacked_rj_fn(self) -> Callable:
        """Jitted companion of [stacked_r_fn][diff_soup.inference.fitting.joint.JointInferenceProblem.stacked_r_fn].

        Returns the ``x -> (r, J)`` pair; built once.
        """
        if self._stacked_rj is None:
            self.stacked_r_fn  # noqa: B018 — builds and caches both callables
        return self._stacked_rj

    @property
    def batched_r_fn(self) -> Optional[Callable]:
        """The geometry-axis ``vmap`` residual (homogeneous children), or ``None``."""
        if self._batched_r is None:
            return None
        return self._residual_seam("forward")[1]

    @property
    def batched_rj_fn(self) -> Optional[Callable]:
        """The geometry-axis ``vmap`` ``x -> (r, J)`` (homogeneous children), or ``None``."""
        if self._batched_r is None:
            return None
        return self._residual_seam("forward")[2]

    @property
    def loss_fn(self) -> Callable:
        """The cached joint data-misfit loss."""
        if self._loss_fn is None:
            residual = self._residual_seam("reverse")[3]
            object.__setattr__(self, "_loss_fn", _loss_fn_from_residual(residual))
        return self._loss_fn

    def loss_and_grad_fn(self, ad: str = "reverse") -> Callable:
        """Return the cached joint data-misfit value and gradient for one AD mode.

        Raises:
            ConfigurationError: If ``ad`` is not ``"forward"`` or ``"reverse"``,
                or no child supports it.
        """
        if ad not in ("forward", "reverse"):
            raise ConfigurationError(
                f"optsolve differentiation mode must be 'forward' or 'reverse'; got {ad!r}."
            )
        cached = self._loss_grad_fns.get(ad)
        if cached is not None:
            return cached
        supported = self.supported_compositions
        if ad not in supported:
            raise ConfigurationError(
                f"JointInferenceProblem does not support differentiation mode {ad!r}; "
                f"all children must support it, and their common modes are {sorted(supported)}."
            )
        residual = self._residual_seam(ad)[3]
        cached = _loss_and_grad_from_residual(residual, ad)
        self._loss_grad_fns[ad] = cached
        return cached

    @property
    def supported_compositions(self) -> frozenset[str]:
        """Differentiation modes supported by every child problem."""
        common = set(self.problems[0].supported_compositions)
        for child in self.problems[1:]:
            common.intersection_update(child.supported_compositions)
        return frozenset(common)

    def resolved_hessian_composition(self, composition: str = "auto") -> str:
        """Resolve and validate a joint true-Hessian composition.

        ``"auto"`` resolves to ``"forward-over-reverse"`` when every child
        supports it, else to ``"forward-over-forward"``, on one device and on
        a device mesh alike.

        Raises:
            ConfigurationError: If the composition is unknown or not supported
                by every child.
        """
        supported = self.supported_compositions
        if composition == "auto":
            if "forward-over-reverse" in supported:
                return "forward-over-reverse"
            composition = "forward-over-forward"
        allowed = ("forward-over-reverse", "forward-over-forward")
        if composition not in allowed:
            raise ConfigurationError(
                "loss_grad_hessian_fn composition must be 'auto', "
                "'forward-over-reverse', or 'forward-over-forward'; "
                f"got {composition!r}."
            )
        if composition not in supported:
            raise ConfigurationError(
                f"Hessian composition {composition!r} is not supported by every "
                f"JointInferenceProblem child; common modes are {sorted(supported)}."
            )
        return composition

    def loss_grad_hessian_fn(self, composition: str = "auto") -> Callable:
        """Return cached joint data-misfit value, gradient, and true Hessian."""
        resolved = self.resolved_hessian_composition(composition)
        cached = self._loss_grad_hessian_fns.get(resolved)
        if cached is None:
            ad = "reverse" if resolved == "forward-over-reverse" else "forward"
            cached = _loss_grad_hessian_from_gradient(self.loss_and_grad_fn(ad))
            self._loss_grad_hessian_fns[resolved] = cached
        return cached

    @property
    def evaluate_lsq_fn(self) -> Optional[Callable]:
        """Reduced ``x -> (cost, J^T r, J^T J, n_active_rows, any_failed)`` (or ``None``).

        Available on the homogeneous path.  Forms the Gauss-Newton statistics by summing
        each child's local contributions across the device mesh, so a sharded joint is
        solved without assembling the global residual or Jacobian on one device.  On one
        device it reduces locally and is numerically identical to contracting the batched
        Jacobian.
        """
        return self._residual_seam("forward")[4]

    @property
    def step_balance(self) -> Any:
        """The step planner state behind the homogeneous reduction, or ``None``.

        Exposes the last trial-step counts per child (``last_steps``), the current
        slot order (``order``), and how many reassignments were applied
        (``reassignments``); see ``Layout.balance_steps``.
        """
        return getattr(self.evaluate_lsq_fn, "step_balance", None)

    @property
    def evaluate_cost_fn(self) -> Optional[Callable]:
        """Reduced ``x -> (cost, n_active_rows, any_failed)`` for trial steps (or ``None``).

        Available on the homogeneous path; reduces the scalar cost across the mesh
        without gathering the residual.
        """
        return self._residual_seam("forward")[5]

    @property
    def unflatten_fn(self):
        """No single fitted problem for a joint vector (per-child reconstruction)."""
        return None

    @property
    def prob_static(self):
        """No single static for a joint vector (per-child reconstruction)."""
        return None

    @property
    def theta_names(self) -> tuple:
        """Per-scalar names of the joint θ-block (shared then per-child locals)."""
        return self._theta_names

    @property
    def function_columns(self) -> tuple:
        """The columns of the joint theta block that hold learnable-function (network) weights."""
        layout = self._layout
        columns: set = set()
        for index, child in enumerate(self.problems):
            child_columns = getattr(child, "function_columns", ())
            if not child_columns:
                continue
            # Child theta column -> joint column, as _reconstruct_child_flat scatters them.
            to_joint = {
                col: layout.shared_slice[0] + k for k, col in enumerate(layout.shared_cols[index])
            }
            to_joint.update(zip(layout.owner_cols[index], layout.child_tied_columns[index]))
            start = layout.local_slices[index][0]
            to_joint.update((col, start + k) for k, col in enumerate(layout.local_cols[index]))
            columns.update(to_joint[col] for col in child_columns if col in to_joint)
        return tuple(sorted(columns))

    @property
    def is_homogeneous(self) -> bool:
        """Whether the geometry-axis vmap kernel is engaged (vs static concatenation)."""
        return self._homogeneous

    @property
    def is_decision_variable(self) -> bool:
        """Whether at least one child carries decision-window state blocks."""
        return any(child.is_decision_variable for child in self.problems)

    @property
    def theta_only_steps(self) -> int:
        """Requested joint freeze-then-relax budget (maximum child request)."""
        return max(
            (int(getattr(child, "theta_only_steps", 0) or 0) for child in self.problems),
            default=0,
        )

    def theta_only_problem(self) -> "JointInferenceProblem":
        """Return a joint view with every trajectory-state coordinate frozen.

        The view remains a
        [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem], so ownership-aware
        Block-LM still applies. Owner residuals receive only global/owner/local
        parameter coordinates and splice the fixed state tail back in before
        evaluating the original kernels.
        """
        if not self.is_decision_variable:
            raise ValueError("theta_only_problem needs decision-variable children.")
        n_active = int(self._layout.n_theta_joint)
        full_start = jnp.asarray(self._x0)
        # Build the original cached owner functions before copying.
        original_owner_fns = self.owner_residual_fns
        original_owner_indices = tuple(
            self._owner_parameter_indices_at(i) for i in range(len(self.owner_ids))
        )

        out = copy.copy(self)
        frozen_state_slices = tuple((n_active, n_active) for _ in self.problems)
        reduced_layout = dataclasses.replace(
            self._layout,
            state_slices=frozen_state_slices,
            n_joint=n_active,
        )
        reduced_owner_fns = []
        for owner_index, (owner_fn, old_indices) in enumerate(
            zip(original_owner_fns, original_owner_indices)
        ):
            new_indices = tuple(range(reduced_layout.n_shared)) + tuple(
                index
                for index in self._owner_parameter_indices_at(owner_index)[self._layout.n_shared :]
                if index < n_active
            )
            old_point = full_start[jnp.asarray(old_indices, dtype=jnp.int32)]
            old_position = {column: pos for pos, column in enumerate(old_indices)}
            target_positions = jnp.asarray(
                [old_position[column] for column in new_indices], dtype=jnp.int32
            )

            def reduced_owner(
                q,
                *,
                _owner_fn=owner_fn,
                _old_point=old_point,
                _positions=target_positions,
            ):
                return _owner_fn(_old_point.at[_positions].set(q))

            reduced_owner_fns.append(jax.jit(reduced_owner))

        object.__setattr__(out, "_layout", reduced_layout)
        object.__setattr__(out, "_x0", np.asarray(self._x0)[:n_active].copy())
        object.__setattr__(
            out,
            "regularization",
            _FrozenTailRegularizer(self.regularization, np.asarray(self._x0), n_active),
        )
        object.__setattr__(
            out,
            "_segment_fns",
            (lambda x: self.segment_fns[0](jnp.concatenate((x, full_start[n_active:]))),),
        )

        def reduced_residual(ad):
            full_residual = self._residual_seam(ad)[3]
            return lambda x: full_residual(jnp.concatenate((x, full_start[n_active:])))

        object.__setattr__(out, "_residual_builder", reduced_residual)
        object.__setattr__(out, "_batched_r", None)
        object.__setattr__(out, "_batched_rj", None)
        object.__setattr__(out, "_stacked_r", None)
        object.__setattr__(out, "_stacked_rj", None)
        object.__setattr__(out, "_residual_seams", {})
        object.__setattr__(out, "_loss_fn", None)
        object.__setattr__(out, "_loss_grad_fns", {})
        object.__setattr__(out, "_loss_grad_hessian_fns", {})
        object.__setattr__(out, "_evaluate_lsq", None)
        object.__setattr__(out, "_evaluate_cost", None)
        # The reduced wrappers are owner-specific compact closures. They retain
        # the full problem's owner-device placement while frozen state
        # coordinates disappear from every executable input.
        object.__setattr__(out, "_owner_residual_fns", tuple(reduced_owner_fns))
        object.__setattr__(out, "_owner_reducer_fns", tuple(None for _ in reduced_owner_fns))
        object.__setattr__(out, "_owner_reducer_inputs", tuple(None for _ in reduced_owner_fns))
        object.__setattr__(out, "_owner_narrow_kernel_cache", None)
        object.__setattr__(out, "_owner_point_plan_cache", None)
        object.__setattr__(out, "_restricted_initialization", True)
        if self._trial_predicate is not None:
            original_predicate = self._trial_predicate

            def reduced_predicate(x):
                return original_predicate(jnp.concatenate((jnp.asarray(x), full_start[n_active:])))

            object.__setattr__(out, "_trial_predicate", reduced_predicate)

        # Result reconstruction needs the frozen coordinates: splice the fixed
        # state tail back and delegate to the full problem, exactly as the
        # reduced residual and trial-predicate wrappers do. Without this, children
        # receive a stateless vector their window layouts reject.
        full_start_host = np.asarray(self._x0)

        def reduced_child_results(x_flat: np.ndarray) -> tuple:
            return self.child_results(
                np.concatenate((np.asarray(x_flat), full_start_host[n_active:]))
            )

        def reduced_shared_values(x_flat: np.ndarray) -> dict:
            return self.shared_values(
                np.concatenate((np.asarray(x_flat), full_start_host[n_active:]))
            )

        object.__setattr__(out, "child_results", reduced_child_results)
        object.__setattr__(out, "shared_values", reduced_shared_values)
        return out

    def with_warm_theta(self, x_full: np.ndarray) -> "JointInferenceProblem":
        """Warm only the joint parameter prefix, retaining state coordinates."""
        n_theta = int(self._layout.n_theta_joint)
        values = np.asarray(x_full)
        if values.size < n_theta:
            raise ValueError(
                f"joint theta warm start needs at least {n_theta} values; got {values.size}."
            )
        start = np.asarray(self._x0).copy()
        start[:n_theta] = values[:n_theta]
        return self.with_start(start)

    @property
    def baseline_reached_tend(self) -> bool:
        """Whether every child's forward solve at ``x0`` reaches ``t_end``.

        Each child's solve runs the first time this is read, not at
        construction.
        """
        if self._baseline_reached_tend is None:
            reached = all(bool(c.baseline_reached_tend) for c in self.problems)
            object.__setattr__(self, "_baseline_reached_tend", reached)
        return self._baseline_reached_tend

    # --- device placement ---
    @property
    def layout(self) -> Optional[Layout]:
        """The device-placement [Layout][diff_soup.Layout], or ``None`` (one device)."""
        return self._placement

    @property
    def ownership_layout(self) -> JointLayout:
        """Host-static slices for global, owner, child-local, and state blocks."""
        return self._layout

    @property
    def ownership_groups(self) -> tuple:
        """Names of parameter groups represented once per owner."""
        return self._layout.ownership_groups

    @property
    def owner_ids(self) -> tuple:
        """Stable owner labels in joint-vector order."""
        return self._layout.owner_ids

    @property
    def child_owner_indices(self) -> tuple:
        """Per child, the integer position of its owner in owner_ids.

        See [owner_ids][diff_soup.inference.fitting.joint.JointInferenceProblem.owner_ids].
        """
        return self._layout.child_owner_indices

    def child_flat(self, x_flat: Any, child_index: int) -> jax.Array:
        """Reconstruct one child's complete flat vector from a joint vector."""
        if not 0 <= int(child_index) < len(self.problems):
            raise IndexError(
                f"child_index must be in [0, {len(self.problems)}); got {child_index}."
            )
        return _reconstruct_child_flat(jnp.asarray(x_flat), int(child_index), self._layout)

    def owner_parameter_indices(self, owner: Any) -> tuple:
        """Indices of ``[global | owner | its child locals/states]``.

        ``owner`` is an ID from
        [owner_ids][diff_soup.inference.fitting.joint.JointInferenceProblem.owner_ids];
        integer IDs are treated as
        IDs, never ambiguously as positional indices. The returned order is the
        coordinate expected by
        [owner_residual_fns]
        [diff_soup.inference.fitting.joint.JointInferenceProblem.owner_residual_fns].
        """
        try:
            owner_index = self.owner_ids.index(owner)
        except ValueError as exc:
            raise KeyError(
                f"unknown owner ID {owner!r}; known owners are {self.owner_ids}."
            ) from exc
        return self._owner_parameter_indices_at(owner_index)

    def _owner_parameter_indices_at(self, owner_index: int) -> tuple:
        """Narrow parameter indices for one positional owner index.

        The owner's columns are its own block and the blocks of the members
        below it, level by level, then its children's locals and states.
        """
        indices: list[int] = list(range(self._layout.n_shared))
        indices.extend(self._layout.owner_columns[owner_index])
        for child_index, child_owner in enumerate(self.child_owner_indices):
            if child_owner != owner_index:
                continue
            lo, hi = self._layout.local_slices[child_index]
            indices.extend(range(lo, hi))
        for child_index, child_owner in enumerate(self.child_owner_indices):
            if child_owner != owner_index:
                continue
            lo, hi = self._layout.state_slices[child_index]
            indices.extend(range(lo, hi))
        return tuple(indices)

    def _require_one_level(self, route: str) -> None:
        """Refuse ``route``, which sums one owner tier, when groups are tied at nested levels.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        n_levels = len(self._layout.level_ids)
        if n_levels <= 1:
            return
        names = ", ".join(repr(name) for name in self._level_names)
        raise ConfigurationError(
            f"{route} sums each owner's experiments into one block over one intermediate "
            f"level, and this problem's hierarchy has {n_levels} ({names}); evaluate such a "
            "study through ds.lower(study), with LoweredStudy.evaluate and "
            "engine.reduce_levels."
        )

    @property
    def owner_residual_fns(self) -> tuple:
        """Cached residuals over each owner's narrow parameter coordinate.

        Function ``i`` accepts exactly the columns returned by
        ``_owner_parameter_indices_at(i)`` (or by
        ``owner_parameter_indices(owner_ids[i])``). Homogeneous children of one owner run
        through the same geometry-axis ``vmap`` kernel as the full joint;
        heterogeneous owner groups use static concatenation. No function sees
        coordinates belonging to another owner.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        self._require_one_level("JointInferenceProblem.owner_residual_fns")
        if self._owner_residual_fns is None:
            built = tuple(
                self._owner_residual_kernel_at(owner_index)
                for owner_index in range(len(self.owner_ids))
            )
            object.__setattr__(self, "_owner_residual_fns", tuple(item[0] for item in built))
            object.__setattr__(self, "_owner_reducer_fns", tuple(item[1] for item in built))
            object.__setattr__(self, "_owner_reducer_inputs", tuple(item[2] for item in built))
        return self._owner_residual_fns

    def owner_residual_fn_at(self, owner_index: int, ad: Optional[str] = None) -> Callable:
        """Return one cached narrow residual for the requested AD mode."""
        owner_index = self._checked_owner_index(owner_index)
        if ad is None and self._owner_residual_fns is not None:
            return self._owner_residual_fns[owner_index]
        return self._owner_residual_kernel_at(owner_index, ad=ad)[0]

    def _owner_residual_kernel_at(self, owner_index: int, ad: Optional[str] = None) -> tuple:
        self._require_one_level("JointInferenceProblem.owner_residual_fns")
        owner_index = self._checked_owner_index(owner_index)
        cache = self._owner_kernel_cache()
        key = ("residual_owner", ad, owner_index)
        built = cache.get(key)
        if built is not None:
            return built
        member_indices = tuple(
            child_index
            for child_index, child_owner in enumerate(self.child_owner_indices)
            if child_owner == owner_index
        )
        children = tuple(self.problems[index] for index in member_indices)
        owner_shared = self.shared + self.ownership_groups
        owner_layout = _build_layout(children, owner_shared)
        if _is_homogeneous(children):
            selected_children = (
                children if ad is None else tuple(child._view_for_ad(ad) for child in children)
            )
            (
                residual,
                _rj,
                _lsq,
                _cost,
                _rows,
                child_fn,
                child_inputs,
            ) = _build_homogeneous_residual(
                selected_children,
                owner_layout,
                _child_mode(selected_children[0]),
                None,
            )
        else:
            indices = self._owner_parameter_indices_at(owner_index)
            owner_x0 = np.asarray(self.x0)[list(indices)]
            residual, _rows = _build_static_residual(
                children,
                owner_layout,
                owner_x0,
                ad=ad,
            )
            residual = jax.jit(residual)
            child_fn = None
            child_inputs = None
        built = (residual, child_fn, child_inputs)
        cache[key] = built
        return built

    @property
    def owner_reducer_batches(self) -> tuple:
        """Per-owner child kernels and numeric child batches for block reducers."""
        self.owner_residual_fns  # build the three aligned caches
        return tuple(zip(self._owner_reducer_fns, self._owner_reducer_inputs))

    def _owner_kernel_cache(self) -> dict:
        cache = self._owner_narrow_kernel_cache
        if cache is None:
            cache = {}
            object.__setattr__(self, "_owner_narrow_kernel_cache", cache)
        return cache

    def _checked_owner_index(self, owner_index: int) -> int:
        owner_index = int(owner_index)
        if owner_index < 0 or owner_index >= len(self.owner_ids):
            raise IndexError(
                f"owner_index must be in [0, {len(self.owner_ids)}); got {owner_index}."
            )
        return owner_index

    @property
    def is_restricted_initialization(self) -> bool:
        """Whether this is the compact frozen-state initialization view."""
        return self._restricted_initialization

    @property
    def owner_point_plans(self) -> tuple:
        """Compact gather/scatter plans in deterministic owner order."""
        cache = self._owner_point_plan_cache
        if cache is None:
            from diff_soup.inference.hierarchical.owner_points import build_owner_point_plans

            cache = build_owner_point_plans(self)
            object.__setattr__(self, "_owner_point_plan_cache", cache)
        return cache

    @property
    def is_sharded(self) -> bool:
        """Whether the placement requests child-batch sharding (a data mesh axis)."""
        return _sharding_requested(self._placement)

    @property
    def placement(self) -> tuple[tuple[int, int], ...]:
        """Per child, ``(lane, process_index)``: the experiment lane that evaluates it.

        See [child_lanes][diff_soup.inference.fitting.joint.child_lanes] for the rule.
        """
        return child_lanes(self.problems, self.shared, self.ownership, self._placement)

    # --- diagnostics-by-name surface ---
    @property
    def stream_names(self) -> tuple:
        """Joint fits report cost per child block, not per named stream."""
        return ()

    @property
    def stream_residual_slices(self) -> tuple:
        """Rows are organized per child, not per stream."""
        return ()

    @property
    def param_group_names(self) -> tuple:
        """Named parameter groups over the joint θ-block (shared once, locals per child)."""
        return self._param_group_names

    @property
    def param_group_cols(self) -> tuple:
        """Per-group joint θ-block columns (parallel to ``param_group_names``)."""
        return self._param_group_cols

    @property
    def parameter_schema(self):
        """Recipe handles mapped to shared, owner-tied, or child-local values.

        Shared handles return one value. Owner-tied handles return a mapping
        keyed by the label of the member of their level (the owner label when
        one level is tied), and local handles use child indices. Handles
        from any compatible child's Recipe resolve by declared name and size.
        If any child lacks Recipe handle metadata, this property is None.
        """
        import hashlib

        from diff_soup.problem.parameters import (
            ParameterEntry,
            ParameterOccurrence,
            ParameterSchema,
        )

        schemas = tuple(getattr(child, "parameter_schema", None) for child in self.problems)
        if any(schema is None for schema in schemas):
            return None
        entries = {}
        occurrences = {}
        layout = self._layout
        for index, schema in enumerate(schemas):
            shared = dict(zip(layout.shared_cols[index], range(*layout.shared_slice)))
            local = dict(zip(layout.local_cols[index], range(*layout.local_slices[index])))
            owned = dict(zip(layout.owner_cols[index], layout.child_tied_columns[index]))
            # The level and member label of each tied child column, in the
            # order of ``owner_cols`` (level by level, groups in declared order).
            member_of = {}
            child = self.problems[index]
            columns_in_order = iter(layout.owner_cols[index])
            for level, member in enumerate(layout.child_levels[index]):
                for group in layout.level_groups[level]:
                    for _ in _child_group_cols(child, group) or ():
                        member_of[next(columns_in_order)] = (
                            level,
                            layout.level_ids[level][member],
                        )
            mapping = {**shared, **owned, **local}
            for entry in schema.entries:
                name = entry.handle.name
                previous = entries.setdefault(name, entry)
                if previous.size != entry.size:
                    raise ValueError(
                        f"joint handle {name!r} has incompatible decoded sizes "
                        f"{previous.size} and {entry.size}; child declarations must agree"
                    )
                for occurrence in entry.occurrences:
                    columns = occurrence.columns
                    if set(columns) <= shared.keys():
                        label = None
                    elif set(columns) <= owned.keys() and (
                        len({member_of[c] for c in columns}) == 1
                    ):
                        label = member_of[columns[0]][1]
                    elif set(columns) <= local.keys():
                        label = index
                    else:
                        raise ValueError(
                            f"joint handle {name!r} spans ownership levels; select its "
                            "complete value in one shared, owner, or local group"
                        )
                    value = ParameterOccurrence(
                        owner=label, columns=tuple(mapping[c] for c in columns)
                    )
                    values = occurrences.setdefault(name, [])
                    if value not in values:
                        values.append(value)
        record = (
            tuple(schema.fingerprint for schema in schemas),
            self.param_group_names,
            self.param_group_cols,
            tuple((name, tuple(values)) for name, values in occurrences.items()),
        )
        fingerprint = "structural:" + hashlib.sha256(repr(record).encode()).hexdigest()
        return ParameterSchema(
            [
                ParameterEntry(
                    handle=entry.handle,
                    occurrences=tuple(occurrences[name]),
                    size=entry.size,
                    decode=entry.decode,
                )
                for name, entry in entries.items()
            ],
            fingerprint=fingerprint,
        )

    def count_failed(self, x_flat: np.ndarray) -> tuple[int, tuple[int, ...]]:
        """How many children hit the failure sentinel (re-evaluated host-side)."""
        fn = self._batched_r if self._batched_r is not None else self._segment_fns[0]
        r = fn(jnp.asarray(x_flat))
        if isinstance(r, jax.Array) and not r.is_fully_addressable:
            from diff_soup._jax_compat import multihost_utils

            r = multihost_utils.process_allgather(r, tiled=True)
        r = np.asarray(r)
        idx: list[int] = []
        start = 0
        for i, n in enumerate(self._rows_per_child):
            block = r[start : start + n]
            if block.size and np.any(np.abs(block) >= _FAILED_SENTINEL):
                idx.append(i)
            start += n
        return len(idx), tuple(idx)

    # --- joint result reconstruction ---
    def child_results(self, x_flat: np.ndarray) -> tuple:
        """Reconstruct each child's fitted problem + local/state values from ``x_flat``.

        Returns a tuple of
        [ChildFit][diff_soup.inference.fitting.result.ChildFit], one per
        child in input order.
        """
        x = jnp.asarray(x_flat)
        out = []
        for i, child in enumerate(self.problems):
            xc = _reconstruct_child_flat(x, i, self._layout)
            fitted = None
            if child.unflatten_fn is not None and child.prob_static is not None:
                import equinox as eqx

                fitted = eqx.combine(child.unflatten_fn(xc), child.prob_static)
            local = {}
            for gname in child.parameters.local:
                cols = _child_group_cols(child, gname)
                if cols:
                    local[gname] = np.asarray(xc)[list(cols)]
            window_states = child.window_states_of(np.asarray(xc))
            out.append(ChildFit(opt_params=fitted, local=local, window_states=window_states))
        return tuple(out)

    def shared_values(self, x_flat: np.ndarray) -> dict:
        """Extract the fitted value of each shared group from the joint vector."""
        x = np.asarray(x_flat)
        vals = {}
        start = 0
        rep = self.problems[0]
        for gname in self.shared:
            cols = _child_group_cols(rep, gname)
            n = len(cols) if cols else 0
            vals[gname] = x[start : start + n]
            start += n
        return vals

    def owner_values(self, x_flat: np.ndarray) -> dict:
        """Extract each owner-tied group's fitted value, keyed by owner ID.

        Raises:
            ConfigurationError: If the joint ties groups at more than one level.
        """
        self._require_one_level("JointInferenceProblem.owner_values")
        x = np.asarray(x_flat)
        values: dict = {}
        for owner_index, owner_id in enumerate(self._layout.owner_ids):
            groups: dict = {}
            for group, (lo, hi) in zip(
                self._layout.ownership_groups,
                self._layout.owner_group_slices[owner_index],
            ):
                groups[group] = x[lo:hi]
            values[owner_id] = groups
        return values
