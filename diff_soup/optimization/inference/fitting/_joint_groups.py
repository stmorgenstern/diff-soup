"""The parameter groups of a joint problem: which children own, share or tie each group, and at which level."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.fitting.inference_problem import (
    InferenceProblem,
)

# ---------------------------------------------------------------------------
# Per-child column bookkeeping
# ---------------------------------------------------------------------------


def _child_group_cols(child: InferenceProblem, gname: str) -> Optional[tuple]:
    """Return child-θ column indices for parameter group ``gname`` (or ``None``)."""
    names = child.param_group_names
    if gname not in names:
        return None
    return tuple(int(c) for c in child.param_group_cols[names.index(gname)])


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
