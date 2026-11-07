"""The joint residual of children that share one structure, evaluated as lanes."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._layout import Layout
from diff_soup.optimization.inference.fitting._forward_seam import (
    _FAILED_SENTINEL,
    _solution_failed,
)
from diff_soup.optimization.inference.fitting._joint_layout import JointLayout, _child_n_theta
from diff_soup.optimization.inference.fitting._joint_structure import _sharding_requested
from diff_soup.optimization.inference.fitting._sharding import (
    device_count,
    identity_order,
    order_children,
    place_children,
    reduce_child_cost,
    reduce_child_lsq,
    run_child_kernel,
)
from diff_soup.optimization.inference.fitting.inference_problem import (
    InferenceProblem,
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
# Homogeneous geometry-axis vmap kernel
# (single-shoot / observed-transition / decision-variable windows)
# ---------------------------------------------------------------------------


def _resolve_child_y0(child: InferenceProblem) -> np.ndarray:
    """The fixed trajectory initial differential state of a child (host array).

    Only children whose initial state is fixed reach the batched kernels; a child
    that learns its initial state is classified ``"latent"`` and reconstructs that
    state from its own slice of the joint decision vector.
    """
    from diff_soup.optimization.inference.fitting.formulation import FixedInitialState
    from diff_soup.optimization.inference.fitting.inference_problem import _resolve_y0

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

        return run_child_kernel(masked, inputs, active, placement, fold=True)

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
        from diff_soup.optimization.inference.engine._balance import StepBalancer

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
