"""Child-batch sharding primitives for joint optimization.

A homogeneous joint evaluates the same compiled child kernel once per child.
These helpers pack the per-child inputs on an explicit leading child axis and
run the kernel either as a plain ``vmap`` (single device) or as a ``shard_map``
over a mesh's batch axis (many devices), with padded children masked so they
contribute exactly nothing.

The child axis is made explicit *before* the shard boundary: callers reconstruct
the per-child input arrays from the flat decision vector, then hand them here.
This is the seam an outer optimizer reduction plugs into; see the joint residual
builder for the kernel definitions.  Single-device meshes follow the same code
path and are the primary parity/debugging surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from diff_soup._layout import Layout
from diff_soup.inference.engine._plan import n_padded
from diff_soup.inference.engine._prepare import split_empty_leaves
from diff_soup.inference.fitting._lane_map import lane_map


def sharding_active(layout: Optional[Layout]) -> bool:
    """Whether *layout* requests child-batch sharding (a mesh and a batch axis)."""
    return layout is not None and layout.experiment_axis is not None and layout.mesh is not None


def device_count(layout: Layout) -> int:
    """Number of devices in the placement's mesh."""
    return int(layout.mesh.shape.get("data", 1))


def _pad_leading(x: jax.Array, n_pad: int) -> jax.Array:
    """Extend a child-axis leaf by *n_pad* rows that repeat the first child.

    Padded children reuse a real child's (valid) inputs so their forward solve
    cannot produce NaNs; their contribution is removed by the active mask.
    """
    if n_pad == 0:
        return x
    pad = jnp.broadcast_to(x[:1], (n_pad,) + x.shape[1:])
    return jnp.concatenate([x, pad], axis=0)


def pad_children(inputs: Any, active: jax.Array, n_pad: int) -> tuple[Any, jax.Array]:
    """Pad every child-axis leaf in *inputs*, and the active mask, by *n_pad* rows.

    Input leaves repeat the first child (valid data); the active mask pads with
    zeros so the new rows are inactive.

    Returns:
        ``(padded_inputs, padded_active)``.
    """
    if n_pad == 0:
        return inputs, active
    inputs = jax.tree_util.tree_map(lambda x: _pad_leading(x, n_pad), inputs)
    active = jnp.concatenate([active, jnp.zeros((n_pad,), active.dtype)])
    return inputs, active


def child_spec(layout: Layout) -> PartitionSpec:
    """The partition spec that shards a leading child axis over the mesh batch axis."""
    return PartitionSpec(layout.experiment_axis)


def place_children(inputs: Any, active: jax.Array, layout: Layout) -> tuple[Any, jax.Array]:
    """Place padded child-axis inputs on the mesh with a child-sharded ``NamedSharding``.

    The leaf count is assumed already padded to a multiple of the device count
    (so the child axis divides evenly).  Returns the same pytree/array with
    explicit sharding attached, ready to cross the shard boundary.
    """
    sharding = NamedSharding(layout.mesh, child_spec(layout))
    return jax.device_put(inputs, sharding), jax.device_put(active, sharding)


def _residual_and_steps(r_child_from_x: Callable, x: jax.Array, inp: Any) -> tuple:
    """Split a child residual function's output into rows and its trial-step count.

    A residual function returns either the rows alone or ``(rows, steps)``,
    where ``steps`` is the integrator's trial-step count for that child; a
    function without a count reports zero steps.
    """
    out = r_child_from_x(x, inp)
    if isinstance(out, tuple):
        rows, steps = out
        return rows, jnp.asarray(steps, dtype=jnp.int32)
    return out, jnp.zeros((), dtype=jnp.int32)


def identity_order(n_real: int, layout: Layout) -> np.ndarray:
    """The contiguous row order: real children in index order, then empty slots.

    An order has one entry per padded slot; ``-1`` marks a slot without a
    child. Slots are lane-major, so consecutive blocks of
    ``n_padded / n_lanes`` slots land on consecutive experiment lanes.
    """
    order = np.full((n_padded(n_real, device_count(layout)),), -1, dtype=np.int32)
    order[:n_real] = np.arange(n_real, dtype=np.int32)
    return order


def order_children(inputs: Any, active: jax.Array, order: jax.Array) -> tuple[Any, jax.Array]:
    """Arrange child-axis inputs by *order*, an int32 slot-to-child map with ``-1`` empty.

    Empty slots repeat child zero's inputs and are inactive. Under ``jit`` on
    a mesh this is one collective gather, so a changed order moves each child's
    rows to its new lane without changing any shape.
    """
    order = jnp.asarray(order, dtype=jnp.int32)
    index = jnp.where(order >= 0, order, 0)
    ordered = jax.tree_util.tree_map(lambda x: jnp.take(x, index, axis=0), inputs)
    return ordered, jnp.where(order >= 0, jnp.take(active, index, axis=0), False)


def reduce_child_lsq(
    r_child_from_x: Callable[[jax.Array, Any], jax.Array],
    x: jax.Array,
    inputs: Any,
    active: jax.Array,
    layout: Optional[Layout],
    columns: Optional[Any] = None,
    *,
    order: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reduce per-child least-squares statistics without forming a global Jacobian.

    Each device differentiates only its local children's residual rows and contracts
    that local Jacobian block into the normal-equation pieces on the spot; only the
    ``O(n_params^2)`` results cross the device boundary, reduced with a sum
    collective.  The full residual and the full Jacobian are never assembled on one
    device.

    **Declared column sparsity.**  A child's residual depends only on the shared
    parameters plus its own blocks — every other joint column is structurally zero
    in its Jacobian.  Pass ``columns`` to differentiate against just those, which
    turns the tangent count per child from the whole joint width into the child's
    own width.  Because ``jacfwd`` tangents share the primal matrix, they collapse
    into one factorization with that many right-hand sides, so the saving is close
    to linear in the ratio.  The narrow blocks are then scattered into the full
    statistics, giving results identical to the dense path up to float summation
    order.  With ``columns=None`` the reduction differentiates against the whole
    replicated vector, which is correct for any caller that has not declared its
    sparsity.

    Args:
        r_child_from_x: ``r_child_from_x(x, inputs_i) -> r_i`` returning one child's
            residual rows as a function of the (replicated) parameter vector and that
            child's input leaf.
        x: The replicated parameter vector of length ``n_params``.
        inputs: A pytree whose every leaf leads with the child axis (length ``N``).
        active: A length-``N`` mask; nonzero for a real child, zero for padding.
        layout: The placement.  ``None`` or an unsharded layout reduces locally; a
            sharded layout reduces per device and sums across the mesh.
        columns: Optional ``(N, n_narrow)`` integer array of the joint columns each
            child's residual depends on, of one fixed width across children.
        order: Optional slot-to-child map (see
            [identity_order][diff_soup.inference.fitting._sharding.identity_order])
            deciding which lane evaluates which child under a sharded layout;
            the contiguous order when omitted.

    Returns:
        ``(cost, jtr, jtj, n_active_rows, any_failed, steps)`` — a replicated scalar
        cost ``0.5||r||^2``, the gradient ``J^T r`` (length ``n_params``), the
        Gauss-Newton matrix ``J^T J`` (``n_params x n_params``), the count of active
        residual rows, a nonzero flag if any active child hit the failure sentinel,
        and the replicated ``(N,)`` int32 trial-step count per child (zero for a
        residual function that reports none).
    """
    from diff_soup.inference.fitting._forward_seam import _FAILED_SENTINEL

    n_params = int(jnp.asarray(x).shape[0])

    def dense_stats(x_local, inputs_d, active_d):
        def loc_r(xj):
            def one(inp, act):
                r, steps = _residual_and_steps(r_child_from_x, xj, inp)
                return jnp.where(act, r, jnp.zeros_like(r)), jnp.where(act, steps, 0)

            r, steps = jax.vmap(one)(inputs_d, active_d)
            r = jnp.reshape(r, (-1,))
            return r, (r, steps)

        # One fused pass: the residual rides the Jacobian's push-forward as its
        # auxiliary output instead of integrating the solves a second time.
        jac, (r, steps) = jax.jacfwd(loc_r, has_aux=True)(x_local)
        cost = 0.5 * jnp.sum(r**2)
        jtr = jac.T @ r
        jtj = jac.T @ jac
        rows_per_child = r.shape[0] // max(int(active_d.shape[0]), 1)
        n_active = jnp.sum(active_d.astype(jnp.int32)) * rows_per_child
        any_failed = jnp.any(jnp.abs(r) >= _FAILED_SENTINEL).astype(jnp.int32)
        return (cost, jtr, jtj, n_active, any_failed), steps

    def narrow_stats(x_local, inputs_d, active_d):
        """Differentiate each child against its own columns, then scatter the blocks."""
        columns_d, child_inputs = inputs_d

        def one(cols, inp, act):
            def child_r(q):
                xj = jnp.asarray(x_local).at[cols].set(q)
                r, steps = _residual_and_steps(r_child_from_x, xj, inp)
                return jnp.where(act, r, jnp.zeros_like(r)), jnp.where(act, steps, 0)

            def child_r_aux(q):
                r, steps = child_r(q)
                return r, (r, steps)

            q0 = jnp.asarray(x_local)[cols]
            jac, (r, steps) = jax.jacfwd(child_r_aux, has_aux=True)(q0)
            return r, jac.T @ r, jac.T @ jac, steps

        r, jtr_narrow, jtj_narrow, steps = jax.vmap(one)(columns_d, child_inputs, active_d)
        cost = 0.5 * jnp.sum(r**2)
        jtr = jnp.zeros((n_params,), dtype=jtr_narrow.dtype).at[columns_d].add(jtr_narrow)
        jtj = (
            jnp.zeros((n_params, n_params), dtype=jtj_narrow.dtype)
            .at[columns_d[:, :, None], columns_d[:, None, :]]
            .add(jtj_narrow)
        )
        n_active = jnp.sum(active_d.astype(jnp.int32)) * r.shape[-1]
        any_failed = jnp.any(jnp.abs(r) >= _FAILED_SENTINEL).astype(jnp.int32)
        return (cost, jtr, jtj, n_active, any_failed), steps

    if columns is None:
        device_stats = dense_stats
    else:
        device_stats = narrow_stats
        inputs = (jnp.asarray(columns, dtype=jnp.int32), inputs)

    if not sharding_active(layout):
        stats, steps = device_stats(x, inputs, active)
        return (*stats, steps)

    # The column table leads with the child axis like every other input, so it
    # rides the existing padding and placement rather than needing its own.
    n_real = int(active.shape[0])
    if order is None:
        order = identity_order(n_real, layout)
    order = jnp.asarray(order, dtype=jnp.int32)
    p_inputs, p_active = order_children(inputs, active, order)
    p_inputs, p_active = place_children(p_inputs, p_active, layout)
    x = jax.device_put(x, NamedSharding(layout.mesh, PartitionSpec()))

    spec = child_spec(layout)
    axis = layout.experiment_axis
    kept, restore = split_empty_leaves(p_inputs)

    def per_device(x_local, kept_d, active_d):
        inputs_d = restore(kept_d, active_d.shape[0])
        stats, steps = device_stats(x_local, inputs_d, active_d)
        return jax.lax.psum(stats, axis), steps

    reduced = shard_map(
        per_device,
        mesh=layout.mesh,
        in_specs=(PartitionSpec(), spec, spec),
        out_specs=((PartitionSpec(),) * 5, spec),
        check_vma=False,
    )
    stats, placed_steps = reduced(x, kept, p_active)
    steps = unorder_steps(placed_steps, order, n_real, layout)
    return (*stats, steps)


def unorder_steps(placed: jax.Array, order: jax.Array, n_real: int, layout: Layout) -> jax.Array:
    """Map per-slot step counts back to child order as one replicated array."""
    index = jnp.where(order >= 0, order, n_real)
    steps = jnp.zeros((n_real + 1,), dtype=jnp.int32).at[index].set(placed)[:n_real]
    return jax.lax.with_sharding_constraint(steps, NamedSharding(layout.mesh, PartitionSpec()))


def reduce_child_cost(
    r_child_from_x: Callable[[jax.Array, Any], jax.Array],
    x: jax.Array,
    inputs: Any,
    active: jax.Array,
    layout: Optional[Layout],
    *,
    order: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reduce the scalar least-squares cost across children without gathering the residual.

    The residual-only companion to
    [reduce_child_lsq][diff_soup.inference.fitting._sharding.reduce_child_lsq],
    used for trial-step
    acceptance where no Jacobian is needed: each device sums ``0.5||r_local||^2`` over
    its children and the scalar is summed across the mesh.  The full residual is never
    assembled on one device.

    Returns:
        ``(cost, n_active_rows, any_failed)`` — all replicated.
    """
    from diff_soup.inference.fitting._forward_seam import _FAILED_SENTINEL

    def device_cost(x_local, inputs_d, active_d):
        def one(inp, act):
            r, _steps = _residual_and_steps(r_child_from_x, x_local, inp)
            return jnp.where(act, r, jnp.zeros_like(r))

        r = jax.vmap(one)(inputs_d, active_d)
        cost = 0.5 * jnp.sum(r**2)
        rows_per_child = int(jnp.size(r)) // max(int(active_d.shape[0]), 1)
        n_active = jnp.sum(active_d.astype(jnp.int32)) * rows_per_child
        any_failed = jnp.any(jnp.abs(r) >= _FAILED_SENTINEL).astype(jnp.int32)
        return cost, n_active, any_failed

    if not sharding_active(layout):
        return device_cost(x, inputs, active)

    n_real = int(active.shape[0])
    if order is None:
        order = identity_order(n_real, layout)
    order = jnp.asarray(order, dtype=jnp.int32)
    p_inputs, p_active = order_children(inputs, active, order)
    p_inputs, p_active = place_children(p_inputs, p_active, layout)
    x = jax.device_put(x, NamedSharding(layout.mesh, PartitionSpec()))

    spec = child_spec(layout)
    axis = layout.experiment_axis
    kept, restore = split_empty_leaves(p_inputs)

    def per_device(x_local, kept_d, active_d):
        inputs_d = restore(kept_d, active_d.shape[0])
        return jax.lax.psum(device_cost(x_local, inputs_d, active_d), axis)

    reduced = shard_map(
        per_device,
        mesh=layout.mesh,
        in_specs=(PartitionSpec(), spec, spec),
        out_specs=(PartitionSpec(),) * 3,
        check_vma=False,
    )
    return reduced(x, kept, p_active)


def run_child_kernel(
    kernel: Callable[[Any, jax.Array], jax.Array],
    inputs: Any,
    active: jax.Array,
    layout: Optional[Layout],
    *,
    fold: bool = False,
) -> jax.Array:
    """Run ``kernel`` across the child axis and return the stacked real rows.

    Args:
        kernel: ``kernel(inputs_i, active_i) -> r_i`` returning one child's rows,
            already masked to zero when ``active_i`` is false.
        inputs: A pytree whose every leaf leads with the child axis (length N).
        active: A length-N mask; nonzero for a real child, zero for padding.
        layout: The placement.  ``None`` or an unsharded layout runs a plain
            ``vmap``; a sharded layout fans the children over the mesh and runs a
            local ``vmap`` on each device.
        fold: On the single-device path, fold any enclosing ``vmap`` into the
            child axis (see [lane_map][diff_soup.inference.fitting._lane_map.lane_map])
            so a linear backend with one factorization per batch lane still
            sees a single matrix-batch axis. Only valid when every traced value
            the kernel reads arrives through ``inputs``.

    Returns:
        The ``(N, ...)`` stack of per-child results with any device padding
        removed (``N`` is the real, unpadded child count).
    """
    n_real = int(active.shape[0])
    if not sharding_active(layout):
        return lane_map(kernel, (0, 0), inputs, active, fold=fold)

    pad = n_padded(n_real, device_count(layout)) - n_real
    p_inputs, p_active = pad_children(inputs, active, pad)
    p_inputs, p_active = place_children(p_inputs, p_active, layout)

    spec = child_spec(layout)
    kept, restore = split_empty_leaves(p_inputs)

    def per_device(kept_d, active_d):
        return jax.vmap(kernel)(restore(kept_d, active_d.shape[0]), active_d)

    sharded = shard_map(
        per_device,
        mesh=layout.mesh,
        in_specs=(spec, spec),
        out_specs=spec,
        check_vma=False,
    )
    rs = sharded(kept, p_active)
    return rs[:n_real]
