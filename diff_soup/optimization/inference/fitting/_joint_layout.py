"""The joint layout: which columns of the joint decision vector each child reads."""

from __future__ import annotations

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.optimization.inference.fitting._joint_groups import _child_group_cols
from diff_soup.optimization.inference.fitting.inference_problem import (
    InferenceProblem,
)


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
    [_normalise_levels][diff_soup.optimization.inference.fitting._joint_groups._normalise_levels]
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
