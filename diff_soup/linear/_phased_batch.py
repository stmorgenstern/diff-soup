"""Batching rules of the phased primitives: lanes run natively, in sequence, or are refused."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp

from diff_soup.linear._factor import FactorContractError
from diff_soup.linear._phased import (
    _VMAP_MODES,
    _backend_label,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)
from diff_soup.linear._phased_abstract import _lane_handle_shapes


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def _to_lanes(x: Any, dim: Optional[int], size: int) -> Any:
    """Move a batched operand's lane axis to the front, or broadcast an unbatched one."""
    if dim is None:
        return jnp.broadcast_to(x, (size,) + tuple(jnp.shape(x)))
    return jnp.moveaxis(x, dim, 0)


def _unbatched_order(order: Any, dim: Optional[int]) -> Any:
    """An order word without the batch axis: a batched one reduced to its maximum over the lanes.

    The maximum depends on every lane, so a bind that is not batched still
    follows the operations of every lane.
    """
    return order if dim is None else jnp.max(order, axis=dim)


_VALUES_BATCH = (
    "matrix values",
    "factor each matrix outside the vmap",
)
_PREDICATE_BATCH = (
    "refactor predicates (the values and handle are shared, so each lane would hold a "
    "different factor)",
    "reduce the predicate to one value outside the vmap, for example with jnp.any",
)


def _vmap_mode(analysis: Any, what: str, batch: tuple[str, str] = _VALUES_BATCH) -> str:
    """The backend's ``vmap_factors`` mode, refusing a backend that cannot batch.

    ``batch`` names what is batched and the remedy that avoids batching it.
    """
    mode = getattr(analysis.backend, "vmap_factors", "unsupported")
    label = _backend_label(analysis)
    if mode not in _VMAP_MODES:
        raise FactorContractError(
            f"The {label} backend declares vmap_factors={mode!r}; it must be one of {_VMAP_MODES}."
        )
    if mode == "unsupported":
        batched, remedy = batch
        raise FactorContractError(
            f"The {label} backend cannot run {what} over a batch of {batched} "
            f"(vmap_factors='unsupported'): {remedy}, or use a backend whose vmap_factors is "
            "'native' or 'sequential'."
        )
    return mode


def _over_lanes(
    analysis: Any,
    what: str,
    lane_ops: Sequence[Any],
    bind: Callable,
    batch: tuple[str, str] = _VALUES_BATCH,
) -> list:
    """Run ``bind`` over operands that all carry a leading lane axis.

    ``"sequential"`` maps the per-lane bind with ``lax.map``; ``"native"``
    binds once with the lane axis in place, folding it with a lane axis the
    operands already carry so the primitive never has more than one. The
    first operand decides whether the operands carry an inner lane axis.
    """
    if _vmap_mode(analysis, what, batch) == "sequential":
        return list(jax.lax.map(lambda ops: tuple(bind(*ops)), tuple(lane_ops)))
    first = lane_ops[0]
    if first.ndim == 3:
        size, inner = int(first.shape[0]), int(first.shape[1])
        folded = [op.reshape((size * inner,) + tuple(op.shape[2:])) for op in lane_ops]
        return [out.reshape((size, inner) + tuple(out.shape[1:])) for out in bind(*folded)]
    return list(bind(*lane_ops))


def _factor_batch(axis_data, args, dims, **params):
    """Batch a factorization over its values.

    A batched slot token is reduced to its maximum over the lanes. A native
    batch binds once with that slot token. A sequential batch of ``size``
    lanes gives lane ``k`` the slot token ``maximum * size + k``: the slot
    tokens are mixed-radix lane numbers, so under a nested ``vmap``, whose
    inner body is lowered once, lane ``(j, k)`` of an outer batch of ``J``
    lanes over an inner batch of ``K`` lanes factors under
    ``(t * K + k) * J + j``, distinct for every lane of the call site. Slot
    tokens are ``int32``: the numbering stays distinct while the base slot
    token plus one, times the product of the nested lane counts, is at most
    ``2**31`` (the public [factor][diff_soup.linear.factor] binds the base
    slot token ``0``). Beyond that the arithmetic wraps and lanes may share a
    slot, which stays correct (a solve rebuilds a slot that holds other
    values) but costs factorizations. The context id must be unbatched:
    every lane factors in the one context of the call. The order word
    batches with the values (broadcast when it is unbatched); a batched order
    word of unbatched values is reduced to its maximum.
    """
    values, token, context_id, order = args
    values_dim, token_dim, context_dim, order_dim = dims
    if context_dim is not None:
        raise FactorContractError(
            "phased_factor received a context id with a vmap axis, but a factor call belongs "
            "to one SolveContext: pass the context id as an unbatched int32 scalar, as "
            "diff_soup.linear.factor does, and factor each context's values in its own call."
        )
    if token_dim is not None:
        token = jnp.max(token, axis=token_dim)
    if values_dim is None:
        order = _unbatched_order(order, order_dim)
        outs = phased_factor_p.bind(values, token, context_id, order, **params)
        return outs, [None] * len(outs)
    analysis = params["analysis"]
    lanes = jnp.moveaxis(values, values_dim, 0)
    size = int(lanes.shape[0])
    orders = _to_lanes(order, order_dim, size)
    if _vmap_mode(analysis, "a factorization") == "sequential":
        tokens = token * size + jnp.arange(size, dtype=token.dtype)
        outs = list(
            jax.lax.map(
                lambda lane: tuple(
                    phased_factor_p.bind(lane[0], lane[1], context_id, lane[2], **params)
                ),
                (lanes, tokens, orders),
            )
        )
    else:
        outs = _over_lanes(
            analysis,
            "a factorization",
            [lanes, orders],
            lambda lane_values, lane_orders: phased_factor_p.bind(
                lane_values, token, context_id, lane_orders, **params
            ),
        )
    return outs, [0] * len(outs)


def _rhs_batch(x: Any, dim: Optional[int], size: int, lanes: int) -> Any:
    """Move a right-hand-side-like operand's batch axis behind its ``lanes`` value lane axes.

    An unbatched operand is broadcast to the batch there.
    """
    if dim is not None:
        return jnp.moveaxis(x, dim, lanes)
    shape = tuple(jnp.shape(x))
    return jnp.broadcast_to(jnp.expand_dims(x, lanes), shape[:lanes] + (size,) + shape[lanes:])


def _solve_batch(axis_data, args, dims, **params):
    """Fold a right-hand-side batch into the block; batch values by the backend's mode.

    An inexact solve's ``weights`` and ``x0`` have ``b``'s shape and batch,
    fold or broadcast exactly as ``b`` does. The status, the counts and the
    order word of a folded batch belong to the one factor, so they stay
    unbatched; a batched order word of an unbatched factor is reduced to its
    maximum.
    """
    n_handle = params["n_handle"]
    values, *operands = args
    handle = operands[:n_handle]
    order = operands[n_handle]
    rhs_like = operands[n_handle + 1 :]
    values_dim, handle_dims = dims[0], dims[1 : 1 + n_handle]
    order_dim = dims[1 + n_handle]
    rhs_dims = dims[2 + n_handle :]
    if values_dim is None and all(d is None for d in handle_dims):
        order = _unbatched_order(order, order_dim)
        if all(d is None for d in rhs_dims):
            outs = phased_solve_p.bind(values, *handle, order, *rhs_like, **params)
            return outs, [None] * len(outs)
        lanes = values.ndim - 1
        size = int(axis_data.size)
        moved = [_rhs_batch(op, d, size, lanes) for op, d in zip(rhs_like, rhs_dims)]
        shape = tuple(moved[0].shape)
        if len(shape) - lanes == 2:
            # (*lanes, B, n): the batch is the block of one solve.
            outs = phased_solve_p.bind(values, *handle, order, *moved, **params)
            return list(outs), [lanes] + [None] * (len(outs) - 1)
        # (*lanes, B, nrhs, n): fold the batch and the block into one block.
        folded = shape[:lanes] + (shape[lanes] * shape[lanes + 1], shape[-1])
        blocks = [op.reshape(folded) for op in moved]
        outs = list(phased_solve_p.bind(values, *handle, order, *blocks, **params))
        outs[0] = outs[0].reshape(shape)
        return outs, [lanes] + [None] * (len(outs) - 1)
    size = int(axis_data.size)
    lane_ops = [_to_lanes(values, values_dim, size)]
    lane_ops += [_to_lanes(leaf, d, size) for leaf, d in zip(handle, handle_dims)]
    lane_ops += [_to_lanes(order, order_dim, size)]
    lane_ops += [_to_lanes(op, d, size) for op, d in zip(rhs_like, rhs_dims)]
    outs = _over_lanes(
        params["analysis"],
        "a solve",
        lane_ops,
        lambda *ops: phased_solve_p.bind(*ops, **params),
    )
    return outs, [0] * len(outs)


def _refactor_batch(axis_data, args, dims, **params):
    """Batch a refactorization; the held values batch with the values, the predicate with its lanes.

    In the ``"sequential"`` mode the backend's optional ``label_empty_lanes``
    hook first gives each empty lane of the batch a marker of its own (the
    lanes are numbered as the slot tokens of a sequential factorization
    are), so a lane that asks is factored into storage of its own rather
    than into storage it would share with the batch's other empty lanes. A
    batched order word of an otherwise unbatched refactorization is reduced
    to its maximum.
    """
    *rest, order = args
    *rest_dims, order_dim = dims
    if all(d is None for d in rest_dims):
        order = _unbatched_order(order, order_dim)
        outs = phased_refactor_p.bind(*rest, order, **params)
        return outs, [None] * len(outs)
    analysis = params["analysis"]
    size = int(axis_data.size)
    lane_ops = [_to_lanes(arg, d, size) for arg, d in zip(args, dims)]
    values_dim, held_dim, _, *handle_dims = rest_dims
    predicate_only = values_dim is None and held_dim is None and all(d is None for d in handle_dims)
    batch = _PREDICATE_BATCH if predicate_only else _VALUES_BATCH
    label = getattr(analysis.backend, "label_empty_lanes", None)
    if label is not None and _vmap_mode(analysis, "a refactorization", batch) == "sequential":
        lanes = jnp.arange(size, dtype=jnp.int32)
        lane_ops[3:-1] = list(label(tuple(lane_ops[3:-1]), lanes, size))
    outs = _over_lanes(
        analysis,
        "a refactorization",
        lane_ops,
        lambda *ops: phased_refactor_p.bind(*ops, **params),
        batch,
    )
    return outs, [0] * len(outs)


_HANDLE_BATCH = (
    "factors",
    "invalidate each factor outside the vmap",
)


def _invalidate_batch(axis_data, args, dims, **params):
    """Batch an invalidation over its handle leaves; its order word batches with them.

    ``"native"`` binds once with the lane axis in front, a nested ``vmap``
    folded into that one axis as for a factorization; ``"sequential"`` maps
    the per-lane bind with ``lax.map``. The context id must be unbatched,
    as for a factorization. A batched order word of an unbatched handle is
    reduced to its maximum.
    """
    *handle, context_id, order = args
    *handle_dims, context_dim, order_dim = dims
    if context_dim is not None:
        raise FactorContractError(
            "phased_invalidate received a context id with a vmap axis, but a factor belongs to "
            "one SolveContext: pass the context id as an unbatched int32 scalar, as "
            "diff_soup.linear.invalidate does."
        )
    if all(d is None for d in handle_dims):
        order = _unbatched_order(order, order_dim)
        outs = phased_invalidate_p.bind(*handle, context_id, order, **params)
        return outs, [None] * len(outs)
    analysis = params["analysis"]
    size = int(axis_data.size)
    leaves = [_to_lanes(leaf, d, size) for leaf, d in zip(handle, handle_dims)]
    orders = _to_lanes(order, order_dim, size)
    if _vmap_mode(analysis, "an invalidation", _HANDLE_BATCH) == "sequential":

        def lane(operands: tuple) -> tuple:
            *lane_leaves, lane_order = operands
            return tuple(phased_invalidate_p.bind(*lane_leaves, context_id, lane_order, **params))

        outs = list(jax.lax.map(lane, (*leaves, orders)))
        return outs, [0] * len(outs)
    shapes = _lane_handle_shapes(analysis, params["orientation"], params["precision"])
    if leaves and leaves[0].ndim == len(shapes[0]) + 2:
        inner = int(leaves[0].shape[1])
        leaves = [leaf.reshape((size * inner,) + tuple(leaf.shape[2:])) for leaf in leaves]
        flat = orders.reshape((size * inner,) + tuple(orders.shape[2:]))
        (order_out,) = phased_invalidate_p.bind(*leaves, context_id, flat, **params)
        return [order_out.reshape((size, inner) + tuple(order_out.shape[1:]))], [0]
    outs = phased_invalidate_p.bind(*leaves, context_id, orders, **params)
    return list(outs), [0] * len(outs)
