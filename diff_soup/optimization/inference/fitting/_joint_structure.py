"""Which children of a joint problem share one structure, run as lanes, and shard."""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout
from diff_soup.optimization.inference.engine._plan import n_padded, refuse_shared_storage
from diff_soup.optimization.inference.fitting._joint_groups import _normalise_ownership
from diff_soup.optimization.inference.fitting._joint_layout import _child_n_theta
from diff_soup.optimization.inference.fitting.inference_problem import (
    InferenceProblem,
    LearnedInitialState,
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

    The same predicate [_is_homogeneous][diff_soup.optimization.inference.fitting._joint_structure._is_homogeneous]
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
    (see [refuse_shared_storage][diff_soup.optimization.inference.engine._plan.refuse_shared_storage]).

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
