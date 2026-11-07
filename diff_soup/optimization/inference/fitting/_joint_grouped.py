"""The grouped and static joint residuals, and the joint regularizers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup._layout import Layout
from diff_soup.optimization._derived import half_sum_of_squares
from diff_soup.optimization.inference.fitting._joint_groups import _normalise_ownership
from diff_soup.optimization.inference.fitting._joint_layout import (
    JointLayout,
    _build_layout,
    _reconstruct_child_flat,
)
from diff_soup.optimization.inference.fitting._joint_residual import _build_homogeneous_residual
from diff_soup.optimization.inference.fitting._joint_structure import (
    _child_mode,
    _group_submeshes,
    _sharding_requested,
    _structural_groups,
)
from diff_soup.optimization.inference.fitting.inference_problem import (
    InferenceProblem,
)


@dataclasses.dataclass(frozen=True)
class _FrozenTailRegularizer:
    """View a full-joint regularizer over a frozen state tail."""

    inner: Any
    full_start: ArrayLike
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
        from diff_soup.optimization.inference.hierarchical._submeshes import on_submesh, sum_lanes

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
        return half_sum_of_squares(self.residual(x_joint))
