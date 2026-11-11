"""Lowering helpers for backends: the FFI call, the jnp and switch lowerings, and the slots of a factor."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from diff_soup._jax_compat import (
    hlo,
    internal_mlir,
    ir,
    mlir,
)
from diff_soup.linear._factor import FactorContractError, SolveStatus
from diff_soup.linear._phased import ORDER_DTYPE
from diff_soup.linear._phased_abstract import _format_avals


# --------------------------------------------------------------------------- #
# Native custom calls ordered by the order word
# --------------------------------------------------------------------------- #
def cast_operand(ctx: Any, value: Any, aval: ShapedArray, dtype: Any) -> tuple[Any, ShapedArray]:
    """Convert an MLIR operand to ``dtype`` inside a lowering rule.

    Args:
        ctx: The lowering rule context.
        value: The MLIR value of the operand.
        aval: Its abstract value.
        dtype: The dtype the custom call expects.

    Returns:
        ``(value, aval)`` in ``dtype``; the operand itself when it already
        has that dtype.
    """
    dtype = np.dtype(dtype)
    if np.dtype(aval.dtype) == dtype:
        return value, aval
    target = ShapedArray(tuple(aval.shape), dtype)
    return internal_mlir.convert_hlo(ctx, value, aval, target), target


#: The attribute that marks the optimization barriers
#: [lower_ffi_call][diff_soup.linear._primitive.lower_ffi_call] puts around a
#: native call. A lowering that follows a value back to the call that produced
#: it may step through a barrier with this mark (an identity on every operand)
#: and through no other.
CALL_FENCE_ATTRIBUTE = "diff_soup.call_fence"


def _fence(values: Sequence[Any]) -> list:
    """``values`` passed through one optimization barrier marked with ``CALL_FENCE_ATTRIBUTE``.

    Result ``i`` of the barrier is operand ``i``; an empty list stays empty.
    """
    values = list(values)
    if not values:
        return values
    barrier = hlo.OptimizationBarrierOp(values)
    barrier.operation.attributes[CALL_FENCE_ATTRIBUTE] = ir.UnitAttr.get()
    return list(barrier.results)


def _value_aval(value: Any) -> ShapedArray:
    """The abstract value of an ``int32`` MLIR value (an order word), from its type."""
    shape = tuple(int(d) for d in ir.RankedTensorType(value.type).shape)
    return ShapedArray(shape, ORDER_DTYPE)


def lower_ffi_call(
    ctx: Any,
    target: str,
    order: Any,
    operands: Sequence[Any],
    operand_avals: Sequence[Any],
    result_avals: Sequence[Any],
    *,
    operand_output_aliases: Optional[dict[int, int]] = None,
    **attributes: Any,
) -> tuple[list, Any]:
    """Emit an XLA FFI custom call ordered by the factor's order word.

    The custom call ``target`` takes, in this order, a fresh XLA token, the
    buffer operands and the order word, and returns a token and the buffer
    results. The handler therefore declares an ``ffi::Token`` argument, its
    buffers and an ``int32`` buffer for the order word, which it never
    reads, then an ``ffi::Token`` result and its buffer results. The last
    buffer result, the call's ``int32`` status per lane, becomes the next
    order word: it has the order word's shape and dtype, and it exists only
    once the call has finished. A call without buffer results (an
    invalidation) returns the order word itself as its only buffer result,
    aliased to the order operand and left untouched by the handler. Either
    way the order word is a true data dependency of the call: the call
    cannot start before the operation that produced its order word, and
    every operation that takes the next order word waits for the call. The
    call is marked side-effecting, so XLA keeps it even when no result is
    used; whether it is in the program at all is decided by JAX, which
    removes an equation whose results reach no output.

    Taking the status as the next order word costs nothing; an order result
    aliased to an order word that is a program argument would make XLA copy
    the argument on every call.

    The buffer operands and the order word reach the call through one
    optimization barrier and the buffer results and the order result leave
    through another (both marked with ``CALL_FENCE_ATTRIBUTE``), so no slice
    of an operand and no update-slice of a result is fused into the call.
    Without them, XLA's GPU compiler fuses the call together with such a
    slice (a right-hand side ``B[k]``, a ``lax.scan`` slice, a solution
    stacked into a scan's outputs) into one fusion whose cost analysis
    cannot size the call's token, and the process aborts while compiling. A
    barrier is an identity: it moves no data and only keeps the slice
    outside the call.

    Args:
        ctx: The lowering rule context.
        target: The registered FFI target name.
        order: The MLIR value of the order word (``int32``, the lane shape
            of the call).
        operands: The MLIR values of the buffer operands, without the token
            and the order word.
        operand_avals: Their abstract values.
        result_avals: The abstract values of the buffer results, without the
            token; the last one is the status, ``int32`` of the order word's
            shape. Empty for a call that returns only the order word.
        operand_output_aliases: Buffer results that share their buffer with
            a buffer operand, as ``{operand position: result position}``,
            both counted among the buffers (the token and the order word are
            not counted). The handler then writes the result into the
            operand's buffer, and XLA copies the operand first only where it
            is still used after the call. An aliased pair must have equal
            shapes and dtypes.
        **attributes: Static attributes of the handler: numpy scalars,
            Python bools, or one-dimensional numpy integer arrays (an
            ``ffi::Span`` attribute of the handler; an empty array is an
            empty span).

    Returns:
        ``(results, order)``: the MLIR values of the buffer results and of
        the next order word (the status, or the order word a call without
        buffer results returns).

    Raises:
        FactorContractError: If an aliased pair differs in shape or dtype,
            or the last buffer result cannot stand for the order word.
    """
    order_aval = _value_aval(order)
    n_in, n_out = len(operands), len(result_avals)
    aliases: dict[int, int] = {}
    if n_out == 0:
        # Position 0 of both lists is the token; the order word follows the buffers.
        aliases[n_in + 1] = 1
        returned = [order_aval]
    else:
        status = result_avals[-1]
        if tuple(status.shape) != tuple(order_aval.shape) or np.dtype(status.dtype) != ORDER_DTYPE:
            raise FactorContractError(
                f"The custom call {target} returns "
                f"{_format_avals([(status.shape, status.dtype)])} last, but the next order word "
                f"is its last result and must be {_format_avals([(order_aval.shape, ORDER_DTYPE)])}"
                ", the shape and dtype of the order word it takes."
            )
        returned = list(result_avals)
    for source, result in (operand_output_aliases or {}).items():
        given, made = operand_avals[source], result_avals[result]
        if tuple(given.shape) != tuple(made.shape) or np.dtype(given.dtype) != np.dtype(made.dtype):
            raise FactorContractError(
                f"The custom call {target} aliases buffer operand {source} "
                f"({_format_avals([(given.shape, given.dtype)])}) to buffer result {result} "
                f"({_format_avals([(made.shape, made.dtype)])}); an aliased pair must have "
                "equal shapes and dtypes."
            )
        aliases[int(source) + 1] = int(result) + 1
    call_ctx = dataclasses.replace(
        ctx,
        avals_in=[jax.ffi.abstract_token, *operand_avals, order_aval],
        avals_out=[jax.ffi.abstract_token, *returned],
    )
    lowering: dict[str, Any] = {"has_side_effect": True}
    if aliases:
        lowering["operand_output_aliases"] = aliases
    rule = jax.ffi.ffi_lowering(target, **lowering)
    fenced = _fence([*operands, order])
    _, *results = rule(call_ctx, hlo.create_token(), *fenced, **attributes)
    results = _fence(results)
    if n_out == 0:
        return [], results[0]
    return results, results[-1]


def lower_jnp(
    ctx: Any,
    fun: Callable[..., Sequence[Any]],
    operands: Sequence[Any],
    operand_avals: Sequence[Any],
    result_avals: Sequence[Any],
) -> list:
    """Lower a ``jnp`` function inside a lowering rule.

    For the small array work a lowering hook does around its native calls
    (a select over values, a constant ``nan`` solution, a branch index).
    The function is traced on ``operand_avals`` and must return one array
    per entry of ``result_avals``, in those shapes and dtypes.

    Args:
        ctx: The lowering rule context.
        fun: The function, called with one array per operand.
        operands: The MLIR values of the operands.
        operand_avals: Their abstract values.
        result_avals: The abstract values of the results.

    Returns:
        The MLIR values of the results.
    """
    inner = dataclasses.replace(
        ctx,
        avals_in=list(operand_avals),
        avals_out=list(result_avals),
        tokens_in=mlir.TokenSet(),
        tokens_out=None,
    )
    return list(
        mlir.lower_fun(lambda *ops: list(fun(*ops)), multiple_results=True)(inner, *operands)
    )


def lower_switch(
    ctx: Any,
    index: Any,
    order: Any,
    branches: Sequence[Callable[[Any], tuple[Sequence[Any], Any]]],
    result_avals: Sequence[Any],
) -> tuple[list, Any]:
    """Emit a ``stablehlo.case`` whose branches may make native calls ordered by the order word.

    For a lowering hook that decides at run time which native call to make,
    for example none for a lane that holds no factorization. Branch ``i``
    runs when ``index`` holds ``i`` (the last branch for an index out of
    range). Each branch is called with the order word and returns ``(results,
    order)``: a branch that emits
    [lower_ffi_call][diff_soup.linear._primitive.lower_ffi_call] returns the
    call's order word, and a branch that makes no call returns the order word
    it was given. The order word leaving the case is the one of the branch
    that ran, so the operations that take it wait for that branch's call.

    Args:
        ctx: The lowering rule context.
        index: The MLIR value of an ``int32`` scalar that selects the branch.
        order: The MLIR value of the order word.
        branches: One callable per branch, taking the order word and
            returning the MLIR values of the results and of the order word.
        result_avals: The abstract values of the results every branch returns.

    Returns:
        ``(results, order)``: the MLIR values of the results and of the
        order word.
    """
    types = [
        order.type,
        *(internal_mlir.aval_to_ir_type(ctx.module_context, aval) for aval in result_avals),
    ]
    case = hlo.CaseOp(types, index=index, num_branches=len(branches))
    for region, branch in zip(case.regions, branches):
        block = region.blocks.append()
        with ir.InsertionPoint(block):
            outs, order_out = branch(order)
            hlo.return_([order_out, *outs])
    order_out, *results = case.results
    return list(results), order_out


# --------------------------------------------------------------------------- #
# Backends whose handle names a slot of native storage
# --------------------------------------------------------------------------- #
#: The slot an empty handle names: no factorization is held.
EMPTY_SLOT = -1

_INT32_MAX = int(np.iinfo(np.int32).max)
_SCALAR_INT32 = ShapedArray((), np.dtype(np.int32))


def empty_slot_handle(context_id: int, leaves: int = 2) -> tuple:
    """The empty handle of a backend whose handle is ``(context id, slot, ...)``.

    Args:
        context_id: The owning context's ``context_id`` (``-1`` without one).
        leaves: The number of ``int32`` scalar leaves; every leaf after the
            context id holds ``-1`` (the slot, and a generation if any).

    Returns:
        ``leaves`` concrete ``int32`` scalars: the context id, then ``-1``.
    """
    return (jnp.asarray(context_id, jnp.int32),) + tuple(
        jnp.asarray(EMPTY_SLOT, jnp.int32) for _ in range(leaves - 1)
    )


def label_empty_slots(handle: tuple, lanes: Any, size: int, *, slot_leaf: int = 1) -> tuple:
    """Give every empty lane of a sequential batch a slot marker of its own.

    An empty lane holds a negative slot ``-1 - m``; lane ``k`` of a batch of
    ``size`` lanes gets ``-1 - (m * size + k)``, the mixed-radix lane number
    a sequential factorization gives its slot tokens, so the lanes of nested
    batches stay distinct. A number past the ``int32`` range falls back to
    ``k`` alone (lanes of an outer batch may then share storage, which stays
    correct and costs factorizations). Lanes that hold a factor keep their
    slot.

    Args:
        handle: The batch's handle leaves, each with a leading lane axis.
        lanes: The ``int32`` lane numbers ``0, ..., size - 1``.
        size: The number of lanes.
        slot_leaf: The position of the slot among the leaves.

    Returns:
        The handle leaves with the empty lanes' slots relabelled.
    """
    slot = handle[slot_leaf]
    number = -1 - slot
    limit = (_INT32_MAX - (size - 1)) // max(size, 1)
    labelled = jnp.where(number <= limit, number * size + lanes, lanes)
    leaves = list(handle)
    leaves[slot_leaf] = jnp.where(slot < 0, -1 - labelled, slot).astype(slot.dtype)
    return tuple(leaves)


def lower_slot_refactor(
    ctx: Any,
    values: Any,
    held_values: Any,
    predicate: Any,
    held_context: Any,
    held_slot: Any,
    order: Any,
    *,
    refactor_target: str,
    factor_target: str,
    site: Any,
    analysis_id: Any,
    values_dtype: Any,
    storage: Any = 0,
) -> list:
    """Lower a refactor for a backend whose handle is ``(context id, slot)``.

    Emits the kept values (``values`` where the predicate holds, else
    ``held_values``) and a switch over three cases of the held slot:

    - a slot that holds a factor: the backend's refactor call, with
      operands ``(kept values, predicate, context id, slot)`` and
      attributes ``site``, ``analysis_id`` and ``storage``;
    - an empty slot (negative) whose lane asks: the backend's factor call,
      with operands ``(kept values, slot, context id)`` and the same
      attributes, so the lane is factored into the call site's slot keyed by
      the empty marker (one slot per empty lane of a sequential batch, see
      [label_empty_slots][diff_soup.linear._primitive.label_empty_slots]),
      or into the shared storage's slot for that marker when ``storage`` is
      not ``0``; a later refactor at this site refactors that slot in place;
    - an empty slot whose lane does not ask: no call; the handle stays empty
      (slot ``-1``) with ``SolveStatus.EMPTY``.

    Every call takes the order word last and returns it
    ([lower_ffi_call][diff_soup.linear._primitive.lower_ffi_call]).

    Args:
        ctx: The lowering rule context of the refactor equation.
        values: The MLIR value of the new values.
        held_values: The MLIR value of the held values.
        predicate: The MLIR value of the predicate.
        held_context: The MLIR value of the handle's context id.
        held_slot: The MLIR value of the handle's slot.
        order: The MLIR value of the order word.
        refactor_target: The FFI target of a refactor.
        factor_target: The FFI target of a factorization.
        site: The native site key of this call ([lowered_site_key][]).
        analysis_id: The native analysis id the calls check and create the
            site on.
        values_dtype: The dtype the native calls take the values in.
        storage: The shared-storage key ([storage_key][]), ``0`` for none.

    Returns:
        The MLIR values of ``(kept_values, context id, slot, status, order)``.
    """
    values_aval, held_aval, predicate_aval, context_aval, slot_aval, _ = ctx.avals_in
    kept_aval, *handle_avals, status_aval, _ = ctx.avals_out
    results = [*handle_avals, status_aval]
    (kept,) = lower_jnp(
        ctx,
        lambda v, h, p: [jnp.where(p, v, h)],
        [values, held_values, predicate],
        [values_aval, held_aval, predicate_aval],
        [kept_aval],
    )
    # The calls take the values each lane keeps: a shared storage compares them with the values
    # it holds when the predicate is false; any other call factors only where it holds.
    cast, cast_aval = cast_operand(ctx, kept, kept_aval, values_dtype)
    (index,) = lower_jnp(
        ctx,
        lambda s, p: [jnp.where(s < 0, jnp.where(p, 1, 2), 0).astype(jnp.int32)],
        [held_slot, predicate],
        [slot_aval, predicate_aval],
        [_SCALAR_INT32],
    )
    common = dict(site=np.int64(site), analysis_id=np.int64(analysis_id), storage=np.int64(storage))

    def held(word: Any) -> tuple[list, Any]:
        return lower_ffi_call(
            ctx,
            refactor_target,
            word,
            [cast, predicate, held_context, held_slot],
            [cast_aval, predicate_aval, context_aval, slot_aval],
            results,
            **common,
        )

    def fill(word: Any) -> tuple[list, Any]:
        return lower_ffi_call(
            ctx,
            factor_target,
            word,
            [cast, held_slot, held_context],
            [cast_aval, slot_aval, context_aval],
            results,
            **common,
        )

    def stay(word: Any) -> tuple[list, Any]:
        outs = lower_jnp(
            ctx,
            lambda c: [
                c,
                jnp.full((), EMPTY_SLOT, jnp.int32),
                jnp.full(status_aval.shape, int(SolveStatus.EMPTY), jnp.int32),
            ],
            [held_context],
            [context_aval],
            results,
        )
        return outs, word

    outs, order_out = lower_switch(ctx, index, order, [held, fill, stay], results)
    return [kept, *outs, order_out]


def lower_slot_apply(
    ctx: Any,
    values: Any,
    held_context: Any,
    slot: Any,
    order: Any,
    b: Any,
    *,
    target: str,
    values_dtype: Any,
    **attributes: Any,
) -> list:
    """Lower a solve for a backend whose handle is ``(context id, slot)``.

    A slot that holds a factor solves with the backend's call, operands
    ``(values, context id, slot, b)``, the order word and the given
    attributes; an empty slot (negative) makes no call and returns a ``nan``
    solution with ``SolveStatus.EMPTY`` and the order word it was given.

    Args:
        ctx: The lowering rule context of the solve equation.
        values: The MLIR value of the values.
        held_context: The MLIR value of the handle's context id.
        slot: The MLIR value of the handle's slot.
        order: The MLIR value of the order word.
        b: The MLIR value of the right-hand side.
        target: The FFI target of a solve.
        values_dtype: The dtype the native call takes the values in.
        **attributes: The handler attributes of the solve call.

    Returns:
        The MLIR values of ``(x, status, order)``.
    """
    values_aval, context_aval, slot_aval, _, b_aval = ctx.avals_in
    x_aval, status_aval, _ = ctx.avals_out
    results = [x_aval, status_aval]
    cast, cast_aval = cast_operand(ctx, values, values_aval, values_dtype)
    (index,) = lower_jnp(
        ctx,
        lambda s: [jnp.where(s < 0, 1, 0).astype(jnp.int32)],
        [slot],
        [slot_aval],
        [_SCALAR_INT32],
    )

    def held(word: Any) -> tuple[list, Any]:
        return lower_ffi_call(
            ctx,
            target,
            word,
            [cast, held_context, slot, b],
            [cast_aval, context_aval, slot_aval, b_aval],
            results,
            **attributes,
        )

    def empty(word: Any) -> tuple[list, Any]:
        outs = lower_jnp(
            ctx,
            lambda: [
                jnp.full(x_aval.shape, jnp.nan, x_aval.dtype),
                jnp.full(status_aval.shape, int(SolveStatus.EMPTY), jnp.int32),
            ],
            [],
            [],
            results,
        )
        return outs, word

    outs, order_out = lower_switch(ctx, index, order, [held, empty], results)
    return [*outs, order_out]


#: The handler name suffix of each dtype a host slot backend factors (KLU, SuperLU).
SLOT_DTYPES: dict[np.dtype, str] = {np.dtype(np.float64): "f64", np.dtype(np.complex128): "c128"}


def slot_dtype(dtype: Any, backend: str, what: str = "values") -> np.dtype:
    """The dtype a host slot backend's native calls take for operands of ``dtype``.

    Args:
        dtype: The dtype of the operand.
        backend: The backend's name, for the error message.
        what: What the operand is, for the error message.

    Returns:
        ``complex128`` for a complex dtype, ``float64`` for a real floating one.

    Raises:
        FactorContractError: If ``dtype`` is neither real nor complex floating point.
    """
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return np.dtype(np.complex128)
    if jnp.issubdtype(dtype, jnp.floating):
        return np.dtype(np.float64)
    raise FactorContractError(
        f"{backend} factors real and complex floating-point matrices; got {what} of dtype "
        f"{jnp.dtype(dtype)}."
    )


def slot_targets(prefix: str, dtype: np.dtype) -> tuple[str, str, str]:
    """The ``(factor, refactor, apply)`` FFI targets of a host slot backend for ``dtype``.

    Args:
        prefix: The targets' common prefix, for example ``"diffsoup_klu"``.
        dtype: A key of [SLOT_DTYPES][diff_soup.linear._primitive.SLOT_DTYPES].

    Returns:
        The three target names.
    """
    suffix = SLOT_DTYPES[np.dtype(dtype)]
    return (f"{prefix}_factor_{suffix}", f"{prefix}_refactor_{suffix}", f"{prefix}_apply_{suffix}")


def lower_split_apply(
    ctx: Any,
    b: Any,
    apply: Callable[[Any, Any], list],
    *,
    axis: int,
    real_dtype: Any,
) -> list:
    """Lower a solve of a real factor with a complex right-hand side as one real solve.

    The real and imaginary parts of ``b`` are stacked along its right-hand
    side axis into one real block of twice the columns, ``apply`` solves the
    block with the factor once, and the two halves are joined into the
    complex solution. A failed factor gives ``nan`` in both parts.

    Args:
        ctx: The lowering rule context of the solve; ``b`` is its last operand.
        b: The MLIR value of the complex right-hand side.
        apply: Lowers the real solve: called with the rule context of the
            real block and the block's MLIR value, it returns the MLIR values
            of ``(x, status, order)``.
        axis: The right-hand side axis of ``b``: ``0`` for ``(n,)`` or
            ``(nrhs, n)``, ``1`` for a batched ``(lanes, n)`` or
            ``(lanes, nrhs, n)``.
        real_dtype: The dtype of the real block.

    Returns:
        The MLIR values of ``(x, status, order)``.
    """
    b_aval = ctx.avals_in[-1]
    x_aval, status_aval, order_aval = ctx.avals_out
    shape = tuple(b_aval.shape)
    columns_shape = shape if len(shape) == axis + 2 else (*shape[:axis], 1, shape[-1])
    columns = columns_shape[axis]
    block = ShapedArray((*shape[:axis], 2 * columns, shape[-1]), np.dtype(real_dtype))

    def stack(v: Any) -> list:
        v = v.reshape(columns_shape)
        return [jnp.concatenate([jnp.real(v), jnp.imag(v)], axis=axis).astype(block.dtype)]

    def join(v: Any) -> list:
        real, imag = jnp.split(v, 2, axis=axis)
        return [jax.lax.complex(real, imag).reshape(x_aval.shape).astype(x_aval.dtype)]

    (stacked,) = lower_jnp(ctx, stack, [b], [b_aval], [block])
    real_ctx = dataclasses.replace(
        ctx,
        avals_in=[*ctx.avals_in[:-1], block],
        avals_out=[block, status_aval, order_aval],
    )
    x_block, status, order = apply(real_ctx, stacked)
    (x,) = lower_jnp(ctx, join, [x_block], [block], [x_aval])
    return [x, status, order]
