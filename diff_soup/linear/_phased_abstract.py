"""Abstract evaluation of the phased primitives: their operand checks and their output avals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from diff_soup.linear._factor import FactorContractError, rung_dtype, solve_dtype
from diff_soup.linear._phased import _ORIENTATIONS, ORDER_DTYPE, _backend_label


# --------------------------------------------------------------------------- #
# Abstract evaluation
# --------------------------------------------------------------------------- #
def _format_avals(avals: Sequence[tuple[tuple[int, ...], np.dtype]]) -> str:
    """Render ``(shape, dtype)`` pairs as ``float64[5, 5], int32[5]``."""
    return ", ".join(f"{np.dtype(dtype).name}{list(shape)}" for shape, dtype in avals) or "none"


def _check_static(name: str, analysis: Any, orientation: Any) -> None:
    """Refuse an analysis without a backend and topology, or an unknown orientation."""
    if getattr(analysis, "backend", None) is None or getattr(analysis, "topology", None) is None:
        raise FactorContractError(
            f"{name} needs an analysis that carries a backend and a topology, as "
            f"diff_soup.linear.analyze returns; got {analysis!r}."
        )
    if orientation not in _ORIENTATIONS:
        raise FactorContractError(f"{name} needs orientation 'A' or 'AT'; got {orientation!r}.")


def _lanes_of(name: str, analysis: Any, values: Any) -> tuple[int, ...]:
    """The lane shape of ``values`` after checking it against the pattern."""
    nnz = int(analysis.nnz)
    if values.ndim not in (1, 2) or values.shape[-1] != nnz:
        raise FactorContractError(
            f"{name} needs values of shape ({nnz},), one per pattern entry, with at most one "
            f"leading lane axis; got shape {list(values.shape)}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise FactorContractError(
            f"{name} needs floating-point values; got dtype {np.dtype(values.dtype).name}."
        )
    return tuple(values.shape[:-1])


def _handle_avals(
    analysis: Any, values: Any, *, orientation: str, precision: str, tol: Any
) -> tuple[ShapedArray, ...]:
    """The abstract handle leaves for ``values``, lane axes prepended."""
    lanes = tuple(values.shape[:-1])
    lane = ShapedArray(tuple(values.shape[-1:]), values.dtype)
    avals = analysis.backend.factor_avals(
        analysis, lane, orientation=orientation, precision=precision, tol=tol
    )
    return tuple(ShapedArray(lanes + tuple(a.shape), a.dtype) for a in avals)


def _check_handle(name: str, handle: Sequence[Any], expected: Sequence[ShapedArray]) -> None:
    """Refuse handle leaves whose shapes or dtypes differ from the factor's."""
    got = tuple((tuple(h.shape), np.dtype(h.dtype)) for h in handle)
    want = tuple((tuple(a.shape), np.dtype(a.dtype)) for a in expected)
    if got != want:
        raise FactorContractError(
            f"{name} received handle leaves {_format_avals(got)}, but a factor of this analysis "
            f"and orientation has {_format_avals(want)}: pass the handle its factorization "
            "returned, with the lane axes of the values."
        )


def check_empty_handle(
    analysis: Any,
    values_aval: ShapedArray,
    handle: Sequence[Any],
    *,
    orientation: str,
    precision: str,
    tol: Any,
) -> tuple:
    """Check a backend's empty handle against the abstract handle of a factor.

    An empty factor must be interchangeable with a built one in a loop carry,
    so every leaf the backend's ``empty_handle`` hook returns must have the
    exact shape and dtype ``factor_avals`` gives for unbatched values.

    Args:
        analysis: The analysis.
        values_aval: The abstract ``(nnz,)`` values of the empty factor.
        handle: The leaves the hook returned.
        orientation: ``"A"`` or ``"AT"``.
        precision: The precision rung.
        tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

    Returns:
        The leaves as a tuple of JAX arrays.

    Raises:
        FactorContractError: If the leaves differ from the factor's in
            number, shape or dtype.
    """
    leaves = tuple(jnp.asarray(leaf) for leaf in handle)
    expected = analysis.backend.factor_avals(
        analysis, values_aval, orientation=orientation, precision=precision, tol=tol
    )
    got = tuple((tuple(leaf.shape), np.dtype(leaf.dtype)) for leaf in leaves)
    want = tuple((tuple(a.shape), np.dtype(a.dtype)) for a in expected)
    if got != want:
        raise FactorContractError(
            f"The {_backend_label(analysis)} backend's empty_handle returned leaves "
            f"{_format_avals(got)}, but a factor of this analysis has {_format_avals(want)}: "
            "an empty handle must have exactly the leaves factor_avals describes."
        )
    return leaves


def _solve_flags(params: dict) -> tuple[bool, bool]:
    """``(has_weights, has_x0)`` of a solve's parameters (absent on an exact backend's bind)."""
    return bool(params.get("has_weights", False)), bool(params.get("has_x0", False))


def _split_solve_operands(
    name: str,
    operands: Sequence[Any],
    n_handle: int,
    has_weights: bool = False,
    has_x0: bool = False,
) -> tuple:
    """Split the operands after ``values`` into ``(handle, order, b, weights, x0)``.

    ``weights`` and ``x0`` are ``None`` unless their flag is set.
    """
    extra = int(bool(has_weights)) + int(bool(has_x0))
    if len(operands) != n_handle + 2 + extra:
        raise FactorContractError(
            f"{name} needs the {n_handle} handle leaves, the order word, one right-hand side "
            f"and {extra} optional operands (weights, x0) after the values; got "
            f"{len(operands)} operands."
        )
    rest = list(operands[n_handle + 2 :])
    weights = rest.pop(0) if has_weights else None
    x0 = rest.pop(0) if has_x0 else None
    return tuple(operands[:n_handle]), operands[n_handle], operands[n_handle + 1], weights, x0


def _is_exact(analysis: Any) -> bool:
    """Whether the analysis's backend declares exact solves (``is_exact``)."""
    return bool(getattr(analysis.backend, "is_exact", True))


def _check_storage(name: str, storage: Any) -> None:
    """Refuse a shared-storage parameter that is neither ``None`` nor a string."""
    if storage is not None and not isinstance(storage, str):
        raise FactorContractError(
            f"{name} needs the storage parameter as None or the storage name string that "
            f"empty_factor(..., storage=...) gave the factor; got {storage!r}."
        )


def _check_order(name: str, order: Any, lanes: tuple[int, ...]) -> None:
    """Refuse an order word that is not ``int32`` with the lane shape of the values."""
    if tuple(order.shape) != tuple(lanes) or np.dtype(order.dtype) != ORDER_DTYPE:
        raise FactorContractError(
            f"{name} needs the factor's order word as int32 of shape {list(lanes)}, the lane "
            f"shape of the values (Factor.order); got {np.dtype(order.dtype).name}"
            f"{list(order.shape)}."
        )


def _order_aval(lanes: tuple[int, ...]) -> ShapedArray:
    """The abstract order word of a call with ``lanes`` lane axes."""
    return ShapedArray(tuple(lanes), ORDER_DTYPE)


def _factor_abstract_eval(
    values, slot_token, context_id, order, *, analysis, orientation, precision, tol
):
    """Handle leaves from the backend, a status per lane and the order word."""
    _check_static("phased_factor", analysis, orientation)
    lanes = _lanes_of("phased_factor", analysis, values)
    if slot_token.shape != () or np.dtype(slot_token.dtype) != np.dtype(np.int32):
        raise FactorContractError(
            "phased_factor needs the slot token as an int32 scalar, the type the native "
            f"handlers read; got {np.dtype(slot_token.dtype).name}{list(slot_token.shape)}."
        )
    if context_id.shape != () or np.dtype(context_id.dtype) != np.dtype(np.int32):
        raise FactorContractError(
            "phased_factor needs the context id as an int32 scalar (the SolveContext's "
            f"context_id, or -1 without a context); got {np.dtype(context_id.dtype).name}"
            f"{list(context_id.shape)}."
        )
    _check_order("phased_factor", order, lanes)
    handle = _handle_avals(analysis, values, orientation=orientation, precision=precision, tol=tol)
    return (*handle, ShapedArray(lanes, jnp.int32), _order_aval(lanes))


def _solve_abstract_eval(
    values,
    *operands,
    n_handle,
    analysis,
    orientation,
    transpose,
    precision,
    tol,
    storage=None,
    has_weights=False,
    has_x0=False,
):
    """A solution shaped like ``b`` in the rung's result dtype, a status per lane, the order word.

    An inexact backend's solve also returns its ``int32`` iteration and
    application counts per lane, before the order word.
    """
    del transpose
    _check_static("phased_solve", analysis, orientation)
    _check_storage("phased_solve", storage)
    exact = _is_exact(analysis)
    if exact and (has_weights or has_x0):
        raise FactorContractError(
            f"phased_solve on the exact {_backend_label(analysis)} backend takes no weights or "
            "initial guess (has_weights and has_x0 must be False): they belong to an inexact "
            "backend's stopping rule, and diff_soup.linear.solve drops them for an exact factor."
        )
    handle, order, b, weights, x0 = _split_solve_operands(
        "phased_solve", operands, n_handle, has_weights, has_x0
    )
    lanes = _lanes_of("phased_solve", analysis, values)
    _check_handle(
        "phased_solve",
        handle,
        _handle_avals(analysis, values, orientation=orientation, precision=precision, tol=tol),
    )
    _check_order("phased_solve", order, lanes)
    n = int(analysis.n)
    block = b.ndim - len(lanes)
    if tuple(b.shape[: len(lanes)]) != lanes or block not in (1, 2) or b.shape[-1] != n:
        raise FactorContractError(
            f"phased_solve needs a right-hand side of shape ({n},) or (nrhs, {n}) behind the "
            f"values' lane axes {list(lanes)}; got shape {list(b.shape)}."
        )
    if not jnp.issubdtype(b.dtype, jnp.inexact):
        raise FactorContractError(
            "phased_solve needs a floating-point right-hand side; got dtype "
            f"{np.dtype(b.dtype).name}."
        )
    x_dtype = solve_dtype(precision, values.dtype, b.dtype)
    if np.dtype(b.dtype) != x_dtype:
        raise FactorContractError(
            f"phased_solve needs a right-hand side in the result dtype {x_dtype.name}; got "
            f"{np.dtype(b.dtype).name}. The result has the dtype of the precision rung "
            f"{precision!r}, complex when the values or the right-hand side are complex, and "
            "the right-hand side must be cast to it before the bind."
        )
    real = rung_dtype(precision)
    for name, extra, dtype in (("weights", weights, real), ("x0", x0, x_dtype)):
        if extra is None:
            continue
        if tuple(extra.shape) != tuple(b.shape) or np.dtype(extra.dtype) != dtype:
            raise FactorContractError(
                f"phased_solve needs {name} of the right-hand side's shape {list(b.shape)} in "
                f"{dtype.name}; got {np.dtype(extra.dtype).name}{list(extra.shape)}."
            )
    outs = [ShapedArray(tuple(b.shape), x_dtype), ShapedArray(lanes, jnp.int32)]
    if not exact:
        outs += [ShapedArray(lanes, jnp.int32), ShapedArray(lanes, jnp.int32)]
    return (*outs, _order_aval(lanes))


def _refactor_abstract_eval(
    values,
    held_values,
    predicate,
    *operands,
    n_handle,
    analysis,
    orientation,
    precision,
    tol,
    adopt,
    storage=None,
):
    """The kept values shaped like ``values``, the handle leaves, a status per lane, the order word."""
    _check_static("phased_refactor", analysis, orientation)
    _check_storage("phased_refactor", storage)
    if adopt is not True and adopt is not False:
        raise FactorContractError(
            f"phased_refactor needs a Python bool for adopt; got {adopt!r}. Pass True to refactor "
            "in place in the storage the held factor names, False for storage of this call site."
        )
    lanes = _lanes_of("phased_refactor", analysis, values)
    if tuple(held_values.shape) != tuple(values.shape) or np.dtype(held_values.dtype) != np.dtype(
        values.dtype
    ):
        raise FactorContractError(
            "phased_refactor needs held values of the shape and dtype of the new values, "
            f"{np.dtype(values.dtype).name}{list(values.shape)} (the values the held factor "
            f"was built from); got {np.dtype(held_values.dtype).name}"
            f"{list(held_values.shape)}."
        )
    if len(operands) != n_handle + 1:
        raise FactorContractError(
            f"phased_refactor needs {n_handle} handle leaves and the order word after the "
            f"predicate; got {len(operands)} operands."
        )
    *handle, order = operands
    if np.dtype(predicate.dtype) != np.dtype(np.bool_) or tuple(predicate.shape) != lanes:
        raise FactorContractError(
            f"phased_refactor needs a boolean predicate of shape {list(lanes)}, one per lane; "
            f"got {np.dtype(predicate.dtype).name}{list(predicate.shape)}."
        )
    expected = _handle_avals(
        analysis, values, orientation=orientation, precision=precision, tol=tol
    )
    _check_handle("phased_refactor", handle, expected)
    _check_order("phased_refactor", order, lanes)
    kept = ShapedArray(tuple(values.shape), values.dtype)
    return (kept, *expected, ShapedArray(lanes, jnp.int32), _order_aval(lanes))


def _lane_handle_shapes(analysis: Any, orientation: str, precision: str) -> tuple:
    """The shapes of one lane's handle leaves, from the backend's ``factor_avals``.

    The values are taken real at the rung's width; the shapes of a handle do
    not depend on whether its factor's values are real or complex.
    """
    lane = ShapedArray((int(analysis.nnz),), rung_dtype(precision))
    avals = analysis.backend.factor_avals(
        analysis, lane, orientation=orientation, precision=precision, tol=None
    )
    return tuple(tuple(a.shape) for a in avals)


def _invalidate_lanes(
    analysis: Any, handle: Sequence[Any], orientation: str, precision: str
) -> tuple[int, ...]:
    """The lane shape of an invalidation's handle leaves, after checking them against a factor's."""
    shapes = _lane_handle_shapes(analysis, orientation, precision)
    got = tuple((tuple(h.shape), np.dtype(h.dtype)) for h in handle)
    lanes = None
    if len(handle) == len(shapes):
        lanes = tuple(handle[0].shape[: len(handle[0].shape) - len(shapes[0])]) if handle else ()
        for leaf, shape in zip(handle, shapes):
            if len(lanes) > 1 or tuple(leaf.shape) != lanes + shape:
                lanes = None
                break
    if lanes is None:
        want = ", ".join(f"{list(shape)}" for shape in shapes) or "none"
        raise FactorContractError(
            f"phased_invalidate received handle leaves {_format_avals(got)}, but a factor of "
            f"this analysis and orientation has leaves of shapes {want}, with at most one "
            "leading lane axis: pass the handle its factorization returned."
        )
    return lanes


def _invalidate_abstract_eval(*operands, n_handle, analysis, orientation, precision, storage=None):
    """The order word: the invalidation's only result."""
    _check_static("phased_invalidate", analysis, orientation)
    _check_storage("phased_invalidate", storage)
    if len(operands) != n_handle + 2:
        raise FactorContractError(
            f"phased_invalidate needs the {n_handle} handle leaves, the context id and the "
            f"order word; got {len(operands)} operands."
        )
    *handle, context_id, order = operands
    if context_id.shape != () or np.dtype(context_id.dtype) != np.dtype(np.int32):
        raise FactorContractError(
            "phased_invalidate needs the context id as an int32 scalar (the SolveContext's "
            f"context_id, or -1 when the handle names it); got "
            f"{np.dtype(context_id.dtype).name}{list(context_id.shape)}."
        )
    lanes = _invalidate_lanes(analysis, handle, orientation, precision)
    _check_order("phased_invalidate", order, lanes)
    return (_order_aval(lanes),)
