"""The phased primitives of the linear-solver contract: factor, solve, refactor, invalidate.

Every backend runs its numeric work through three JAX primitives, and a
fourth marks native factor storage stale, so their differentiation,
batching and ordering rules are written once for all backends and a backend
supplies only shapes and lowerings.

``phased_factor_p.bind(values, slot_token, context_id, order, *, analysis, orientation, precision, tol)``
    Returns ``(*handle, status, order)``: the backend's factor handle, a
    tuple of arrays (the LU and its pivots for a dense backend, the context
    id and a slot for a backend that holds factors natively), an ``int32``
    status per lane and the factor's order word. ``slot_token`` is an
    ``int32`` scalar that tells the lanes of one call site apart (a backend
    with native storage keeps one slot per call site, context and slot
    token). ``context_id`` is the ``int32`` scalar ``context_id`` of the
    owning [SolveContext][diff_soup.linear.SolveContext], ``-1`` without
    one; it is an operand, never a parameter, so one compiled program serves
    every context, and it cannot carry a ``vmap`` axis.
``phased_solve_p.bind(values, *handle, order, b, [weights], [x0], *, n_handle, analysis, orientation, transpose, precision, tol, storage=None, has_weights=False, has_x0=False)``
    Returns ``(x, status, order)``. ``b`` is ``(n,)`` or ``(nrhs, n)`` and ``x`` has
    the shape of ``b``. The result has the precision rung's dtype
    (``float64`` for ``"fp64"``, ``float32`` for ``"fp32"``), or its complex
    counterpart of the same width when the values or ``b`` are complex.
    ``b`` must already be in that dtype: the caller casts the right-hand
    side on entry, as [solve][diff_soup.linear.solve] does, and a bind with
    any other dtype is refused. On a backend that declares
    ``is_exact = False`` the solve stops at the tolerance ``tol`` and
    returns ``(x, status, iterations, applications, order)``, the two counts
    ``int32`` per lane; it may take the optional operands ``weights`` (the
    error weights of the tolerance's weighted norm, real, of ``b``'s shape)
    and then ``x0`` (the initial guess, ``b``'s shape and dtype), each
    present when its static flag ``has_weights`` / ``has_x0`` is true. An
    exact backend's solve takes neither: its binds, abstract evaluation and
    lowering carry no flag and no extra operand.
``phased_refactor_p.bind(values, held_values, predicate, *handle, order, *, n_handle, analysis, orientation, precision, tol, adopt, storage=None)``
    Returns ``(kept_values, *handle, status, order)``. In every lane whose boolean
    ``predicate`` holds, the lane is factored from ``values`` and
    ``kept_values`` is ``values``; in every other lane no factorization
    runs, ``kept_values`` is ``held_values`` (the values the held factor was
    built from), the handle is the input handle and the status is the held
    factor's. ``held_values`` has the shape and dtype of ``values``, so a
    refactor carries both value operands and the backend selects between
    them: no select over the values runs in front of the bind. A handle
    that holds no factorization (see below) reports
    ``SolveStatus.EMPTY`` in a lane that does not ask; a lane that asks is
    factored into storage of the refactor's call site. A backend with native
    storage finds the owning context in the handle. With ``adopt`` true a
    lane that asks is refactored in place in the storage its handle names,
    whichever call site built that storage, instead of storage of this
    call site; the older factor that named it is then stale, and its next
    solve rebuilds that storage from its own values. A backend whose
    factors are arrays, or that always refactors in place, ignores the flag.
``phased_invalidate_p.bind(*handle, context_id, order, *, n_handle, analysis, orientation, precision, storage=None)``
    Returns ``(order,)``. After it, the native storage the handle names no
    longer stands for any factor: the next solve that uses it refactors it
    from the values that solve carries. ``context_id`` is the ``int32``
    scalar ``context_id`` of the owning context, ``-1`` when the handle names
    it (as every native handle does); like the factor's, it cannot carry a
    ``vmap`` axis. A backend without a native invalidation (factors held as
    arrays, or storage that needs no marking) lowers it to the identity on
    the order word.

``order`` is the factor's order word ([Factor.order][diff_soup.linear.Factor]):
an ``int32`` array with the lane shape of the values (a scalar for an
unbatched factor) that every primitive takes as an operand and returns as a
result. Its value is never read; it makes every operation on a factor a data
dependency of the next one, which is how operations on one factor are
ordered (see the ordering rule below).

``storage`` is the shared-storage name of a factor that
[empty_factor][diff_soup.linear.empty_factor] started with ``storage=...``
(``None`` for any other factor; a string otherwise). A backend that keeps its
factors natively keeps such a factor in storage named by its context, the
device that runs the call, analysis, precision, lane count and that name,
shared by every call site and program on that device, and trusts it: a
refactor makes the storage hold the factor's matrix (comparing the values it
keeps with the values the storage holds) and a solve in the same execution
of the program uses the storage as it stands.
The tangent rule and the transpose rule bind their solves without ``storage``,
so derivative solves never trust; one that factors the storage again from its own values
leaves the storage untrusted until the next refactor of it, so a later solve of the same
execution does not take that matrix on trust. A lowering hook receives ``storage`` only for such a
factor; a backend whose factors are arrays ignores it.

``values`` holds one entry per position of ``analysis.topology`` (duplicate
positions add), with at most one leading lane axis; the handle leaves, the
predicate, ``b``, the status and the order word carry the same lane axes in
front. The
assembled operator ``A_bc`` is ``A`` with every entry of a Dirichlet row
(``topology.bc_indices``) replaced by zero and one on that row's diagonal. A
factor of orientation ``"A"`` represents ``M = A_bc`` and one of orientation
``"AT"`` represents ``M = A_bc^T``; a solve with ``transpose=False`` solves
``M x = b`` and one with ``transpose=True`` solves ``M^T x = b`` from the same
handle. Every parameter is hashable, ``analysis`` by content.

An empty handle holds no factorization. A backend builds one with its
``empty_handle`` hook (the leaves of ``factor_avals``, holding the backend's
empty marker: slot ``-1`` for a backend with native storage, pivots ``-1``
for the dense LU). A solve on an empty lane returns ``nan`` with
``SolveStatus.EMPTY`` and calls no library, an invalidation of it does
nothing, and a refactor fills it as stated above.

Rules, registered once:

- Differentiation. The tangent of a solve is a solve on the same handle,
  ``dx = solve(db - dM x)``, where ``dM x`` is ``pattern_matvec`` over the
  value tangent on the static topology; Dirichlet rows contribute no tangent.
  The solve is linear in ``b`` and transposes into the solve with
  ``transpose`` flipped, so reverse mode never factors again and the rules
  compose to any order. An inexact solve is differentiated the same way, as
  the exact inverse of ``M``: its ``weights`` and ``x0`` contribute no
  tangent, the tangent solve binds the same ``weights`` and no initial
  guess, and the flipped solve of the transpose rule keeps ``weights``,
  drops ``x0`` and gives neither a cotangent. A factorization has zero tangents, and a nonzero
  tangent that reaches its values is refused at trace time. A refactor's
  ``kept_values`` has the tangent ``select(predicate, dvalues,
  dheld_values)`` and its handle and status have none; the select is linear
  in both tangents, so reverse mode and higher orders need no rule of the
  refactor's own. Tangents on the handle leaves of a solve are ignored:
  derivatives in the matrix flow only through the ``values`` operand of the
  solve, never through the handle. A factor is differentiated through the
  solves that use it, so bind the factorization on
  ``jax.lax.stop_gradient(values)`` and pass the differentiated values to
  the solve; a refactor passes its values through to the solves itself.
  The order word has a ``float0`` tangent and the tangent of an order
  operand is ignored. The tangent solve and the flipped solve of the
  transpose rule take the order word the primal solve took (in reverse mode
  it is a residual of the forward pass, or a zero word when it came out of a
  solve on a linear right-hand side); their own order results are
  dropped. Neither trusts shared storage (both bind ``storage=None``): no
  later operation waits for them, so they apply the rule of a factor of its
  own storage, which stays exact whatever runs on the storage in between.
- Batching. A ``vmap`` over right-hand sides alone folds into the block
  dimension: one solve with ``nrhs`` equal to lanes times columns, whatever
  the backend. A ``vmap`` over values, handle or predicate follows the
  backend's ``vmap_factors`` flag: ``"native"`` binds the primitive with a
  leading lane axis (a nested ``vmap`` folds into that one axis, so a backend
  never sees more than one), ``"sequential"`` maps the unbatched bind with
  ``lax.map`` (lane ``k`` of a batch of ``size`` lanes gets the slot token
  ``t * size + k``, ``t`` the largest slot token of the batch, so every lane
  of one call site, nested batches included, factors under a slot token of
  its own; a refactor's empty lanes are numbered the same way through the
  backend's optional ``label_empty_lanes`` hook), and ``"unsupported"``
  raises [FactorContractError][diff_soup.linear.FactorContractError] at
  trace time. A refactor's ``held_values`` batch with its ``values``. A
  solve's ``weights`` and ``x0`` batch with ``b`` (an unbatched one is
  broadcast to the batch). The order word batches like the handle: an
  unbatched order operand of a batched bind is broadcast to the lanes, the
  sequential route maps it lane by lane, and a batched order operand of an
  otherwise unbatched bind is reduced to its maximum over the lanes (it
  stays a dependency of every lane), so a ``vmap`` over ``b`` alone leaves
  the order word unbatched.
- Ordering. No primitive carries a JAX effect: a program of these
  primitives has neither ordered nor unordered effects and is dispatched
  on JAX's fastest path. Operations are ordered by data instead. The handle
  is the data dependency of every solve on its factor, and the order word
  is the data dependency of every operation on the operation before it on
  the same factor: each primitive takes the factor's order word and returns
  the next one, which [solve][diff_soup.linear.solve],
  [refactor][diff_soup.linear.refactor] and
  [invalidate][diff_soup.linear.invalidate] put in the factor they return.
  Threading the factor (``x, f = solve(f, b)``; ``f = refactor(f, ...)``;
  ``f = invalidate(f)``) therefore orders every operation on it, in the
  JAX program and in the compiled one, where a native call takes the order
  word as an operand and returns it as a result. Operations on one storage
  through different factor objects (a shared storage named by several
  factors, or an empty factor of a filled storage) are ordered only where
  their order words are linked with [after][diff_soup.linear.after]. An
  operation whose results reach no output of the program is dead code like
  any other equation: JAX removes it before compiling, so an invalidation
  or a refactorization must return its factor into something the program
  returns or passes on. Rematerialisation (``jax.checkpoint``) saves the
  outputs of every factorization, refactorization and invalidation, and of
  every solve on a backend whose ``factor_residency`` is not ``"arrays"``,
  instead of recomputing them (a partial-evaluation rule of each primitive
  keeps such an equation on the known side). An invalidation has no
  tangent: its JVP binds it on the primal handle and returns the order word
  with a zero tangent. Under ``vmap`` it follows the backend's
  ``vmap_factors`` flag like a factorization: one bind over the lane axis,
  or ``lax.map`` over the lanes.
- Lowering. A backend that defines ``lower_factor``, ``lower_apply``,
  ``lower_refactor`` or ``lower_invalidate`` gets that MLIR lowering rule,
  called as ``hook(ctx, *operands, **params)`` with every operand of the
  primitive, the order word included, and returning the MLIR values of
  every result, the order word last. A native custom call is emitted by
  [lower_ffi_call][diff_soup.linear._primitive.lower_ffi_call]: the call
  takes the order word as its last buffer operand, which the handler never
  reads, and its status result becomes the next order word (an
  invalidation, which has no other result, returns the order word aliased
  to the operand), so the call cannot start before the operation that
  produced the order word and the next operation cannot start before the
  call has finished.
  The call also takes a fresh XLA token and is marked side-effecting, and
  its buffer operands and results are fenced with optimization barriers so
  that XLA never fuses a slice into the call. Eager evaluation compiles a
  one-equation program and runs it; an eager factorization runs under a
  slot token no other eager factorization of the process has used, so each
  eager factor keeps storage of its own. Otherwise the backend's JAX-level
  ``factor_fun``, ``apply_fun`` and ``refactor_fun`` are lowered with
  ``mlir.lower_fun`` and evaluated directly outside ``jit``; each receives
  the order word as the keyword ``order`` and returns it as its last output
  (a backend whose JAX-level work binds another backend's primitives
  threads it through them, and every other returns it unchanged). Their
  outputs are checked against the abstract shapes and cast to the abstract
  dtypes; the status is ``int32``. An invalidation without a
  ``lower_invalidate`` hook lowers to the identity on the order word.

The backend is reached as ``analysis.backend``, which provides
``factor_avals(analysis, values_aval, *, orientation, precision, tol)`` (the
abstract handle leaves of one lane), ``empty_handle`` (the concrete leaves of
an empty handle), the three JAX-level functions (each accepting one optional
leading lane axis), the ``vmap_factors`` flag and the optional lowering
hooks. The analysis provides ``topology``, ``n``, ``nnz`` and
``backend_name``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import itertools
from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from diff_soup._jax_compat import (
    Primitive,
    Var,
    ad,
    batching,
    hlo,
    internal_mlir,
    internal_partial_eval,
    ir,
    mlir,
    partial_eval,
)
from diff_soup.linear._factor import FactorContractError, SolveStatus, rung_dtype, solve_dtype

__all__ = [
    "CALL_FENCE_ATTRIBUTE",
    "EMPTY_SLOT",
    "ORDER_DTYPE",
    "cast_operand",
    "check_empty_handle",
    "empty_slot_handle",
    "label_empty_slots",
    "lower_ffi_call",
    "lower_jnp",
    "lower_slot_apply",
    "lower_slot_refactor",
    "lower_switch",
    "lowered_site_key",
    "pattern_matvec",
    "phased_factor_p",
    "phased_invalidate_p",
    "phased_refactor_p",
    "phased_solve_p",
    "storage_key",
]

#: The dtype of a factor's order word.
ORDER_DTYPE = np.dtype(np.int32)

_ORIENTATIONS: tuple[str, ...] = ("A", "AT")
_VMAP_MODES: tuple[str, ...] = ("native", "sequential", "unsupported")
_HOOKS: dict[str, str] = {
    "factor": "lower_factor",
    "apply": "lower_apply",
    "refactor": "lower_refactor",
    "invalidate": "lower_invalidate",
}
_FUNS: dict[str, str] = {
    "factor": "factor_fun",
    "apply": "apply_fun",
    "refactor": "refactor_fun",
    "invalidate": "invalidation",
}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _backend_label(analysis: Any) -> str:
    """The name a backend is reported under in error messages."""
    name = getattr(analysis, "backend_name", None)
    if name:
        return str(name)
    return type(getattr(analysis, "backend", analysis)).__name__


phased_factor_p = Primitive("phased_factor")
"""Factor the assembled operator: ``(values, slot_token, context_id, order) -> (*handle, status, order)``."""

phased_solve_p = Primitive("phased_solve")
"""Solve with a factor: ``(values, *handle, order, b[, weights][, x0]) -> (x, status[, iterations, applications], order)``, linear in ``b``."""

phased_refactor_p = Primitive("phased_refactor")
"""Refactor where a predicate holds: ``(values, held_values, predicate, *handle, order) -> (kept_values, *handle, status, order)``."""

phased_invalidate_p = Primitive("phased_invalidate")
"""Mark the native storage a handle names stale: ``(*handle, context_id, order) -> (order,)``."""

for _primitive in (phased_factor_p, phased_solve_p, phased_refactor_p, phased_invalidate_p):
    _primitive.multiple_results = True


# --------------------------------------------------------------------------- #
# The pattern matvec
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def _pattern_indices(topology: Any) -> tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Host index arrays of a topology's entries outside its Dirichlet rows.

    Returns ``(nnz, rows, cols, keep)``: the pattern length, the rows and
    columns of the kept entries, and the kept entry positions (``None`` when
    the topology has no Dirichlet rows and every entry is kept). The arrays
    are read-only host numpy, never ``jnp``, so the cache holds no tracer.
    """
    rows = np.asarray(topology.rows, dtype=np.int32)
    cols = np.asarray(topology.cols, dtype=np.int32)
    nnz = int(rows.shape[0])
    bc = topology.bc_indices
    keep = None
    if bc is not None and int(np.asarray(bc).size) > 0:
        keep = np.flatnonzero(~np.isin(rows, np.asarray(bc))).astype(np.int32)
        rows, cols = rows[keep], cols[keep]
        keep.flags.writeable = False
    rows = np.array(rows, copy=True)
    cols = np.array(cols, copy=True)
    rows.flags.writeable = False
    cols.flags.writeable = False
    return nnz, rows, cols, keep


def pattern_matvec(analysis: Any, dvalues: Any, x: Any, *, transpose: bool) -> jax.Array:
    """Apply the operator whose pattern entries are ``dvalues`` to ``x``.

    Computes ``D x`` (or ``D^T x`` with ``transpose=True``), where ``D`` has
    the entries ``dvalues`` on ``analysis.topology`` with every entry of a
    Dirichlet row dropped, which is the derivative of the assembled operator
    ``A_bc`` along a value tangent ``dvalues``. With ``dvalues`` set to the
    values themselves it is ``A_bc x`` without the Dirichlet rows' unit
    diagonal. The index arrays are host numpy taken from the topology, so the
    function is linear in ``dvalues`` and in ``x`` and differentiates and
    batches like any ``jnp`` code.

    A backend whose values describe more than the pattern's entries (a
    low-rank term ``U W^T`` whose entries of ``U`` follow the base values)
    defines a method ``pattern_matvec(analysis, dvalues, x, *, transpose)``
    with this contract, linear in ``dvalues`` and ``x``; it is used in place
    of the pattern's gather and scatter, so derivatives and residuals cover
    the whole operator.

    Args:
        analysis: An analysis carrying the ``topology``.
        dvalues: Entry values of shape ``(*lanes, nnz)``.
        x: Vector ``(*lanes, n)`` or block ``(*lanes, nrhs, n)``.
        transpose: Apply ``D^T`` instead of ``D``.

    Returns:
        An array of the shape of ``x``, in the promoted dtype of ``dvalues``
        and ``x``.

    Raises:
        FactorContractError: If the shapes do not match the topology.
    """
    backend_matvec = getattr(getattr(analysis, "backend", None), "pattern_matvec", None)
    if backend_matvec is not None:
        return backend_matvec(analysis, dvalues, x, transpose=transpose)
    topology = analysis.topology
    n = int(topology.n_dofs)
    nnz, rows, cols, keep = _pattern_indices(topology)
    dvalues = jnp.asarray(dvalues)
    x = jnp.asarray(x)
    if dvalues.ndim < 1 or dvalues.shape[-1] != nnz:
        raise FactorContractError(
            f"pattern_matvec needs values of shape (*lanes, {nnz}), one per pattern entry; "
            f"got shape {list(dvalues.shape)}."
        )
    lanes = tuple(dvalues.shape[:-1])
    block = x.ndim - len(lanes)
    if tuple(x.shape[: len(lanes)]) != lanes or block not in (1, 2) or x.shape[-1] != n:
        raise FactorContractError(
            f"pattern_matvec needs x of shape (*lanes, {n}) or (*lanes, nrhs, {n}) with the "
            f"values' lanes {list(lanes)}; got shape {list(x.shape)}."
        )
    coef = dvalues if keep is None else dvalues[..., keep]
    src, dst = (rows, cols) if transpose else (cols, rows)
    if block == 2:
        coef = coef[..., None, :]
    contributions = coef * x[..., src]
    out = jnp.zeros(contributions.shape[:-1] + (n,), contributions.dtype)
    return out.at[..., dst].add(contributions)


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


def storage_key(storage: Optional[str]) -> np.int64:
    """The 64-bit key a native handler receives for a shared-storage name; ``0`` for none.

    A stable hash of the name (the same in every process), so a compiled
    program that names the storage has the same text whenever it is lowered.

    Args:
        storage: The storage name, or ``None``.

    Returns:
        A positive ``int64`` for a name, ``0`` for ``None``.
    """
    if storage is None:
        return np.int64(0)
    digest = hashlib.blake2b(storage.encode("utf-8"), digest_size=8).digest()
    return np.int64((int.from_bytes(digest, "little") & ((1 << 62) - 1)) | 1)


#: The entry of a module's lowering cache that counts the contract calls lowered into it.
_SITE_ORDINALS = object()


def lowered_site_key(ctx: Any, kind: str, *parts: Any) -> np.int64:
    """A deterministic key of one lowered native factor or refactor call, for its ``site``.

    The key is a stable 64-bit hash of the module's symbol name, the call's
    name stack, the ordinal of this call among the calls this function has
    keyed in the module so far, ``kind`` and ``parts`` (the analysis id,
    orientation, precision, lane count and whatever else tells the call
    apart). Two lowerings of one program therefore give the same keys and
    the same program text (so JAX's persistent compile cache recognises the
    program), and the native handlers create a site on the first call that
    carries an unknown key; the calls of one program, and of distinct
    programs, get distinct keys. The key has bit 62 set, so it never equals
    an id the extensions number their sites with by themselves.

    Args:
        ctx: The lowering rule context.
        kind: What the call does (``"factor"``, ``"refactor"``).
        *parts: Further values that identify the call, turned into text.

    Returns:
        The key, a positive ``int64``.
    """
    module = ctx.module_context
    counters = module.cached_primitive_lowerings
    ordinal = counters.get(_SITE_ORDINALS, 0)
    counters[_SITE_ORDINALS] = ordinal + 1
    try:
        name = ir.StringAttr(module.module.operation.attributes["sym_name"]).value
    except Exception:  # noqa: BLE001 - a module without a name keys on the rest
        name = ""
    text = "|".join(str(part) for part in (name, ctx.name_stack, ordinal, kind, *parts))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.int64((int.from_bytes(digest, "little") & ((1 << 62) - 1)) | (1 << 62))


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


# --------------------------------------------------------------------------- #
# Differentiation
# --------------------------------------------------------------------------- #
def _is_zero(tangent: Any) -> bool:
    """Whether a tangent is symbolically zero (or of the zero-size ``float0`` type)."""
    return type(tangent) is ad.Zero or getattr(tangent, "dtype", None) == jax.dtypes.float0


def _zero_tangents(outs: Sequence[Any]) -> list:
    """Symbolic zero tangents for primal outputs."""
    return [ad.Zero(jax.typeof(out).to_tangent_aval()) for out in outs]


def _refuse_value_tangent(name: str, analysis: Any) -> None:
    """Raise for a value tangent of a factorization that is not a symbolic zero."""
    raise FactorContractError(
        f"The tangent of the matrix values of {name} on the {_backend_label(analysis)} "
        "backend must be a symbolic zero; it is a traced or instantiated array. A factor is "
        "differentiated through the solves that use it, never through the factorization: "
        "bind the factorization on jax.lax.stop_gradient(values) and pass the "
        "differentiated values to the solve."
    )


def _factor_jvp(primals, tangents, **params):
    """Zero tangents; a nonzero value tangent is refused."""
    if not _is_zero(tangents[0]):
        _refuse_value_tangent("phased_factor", params["analysis"])
    outs = phased_factor_p.bind(*primals, **params)
    return outs, _zero_tangents(outs)


def _refactor_jvp(primals, tangents, **params):
    """``select(predicate, dvalues, dheld_values)`` for the kept values; zero elsewhere.

    The primal outputs come from binding the primals, so the factorization
    runs once. The kept values are ``values`` where the predicate holds and
    ``held_values`` elsewhere, so their tangent selects between the two value
    tangents; both symbolically zero gives a zero tangent. The handle, the
    status and the order word have none. The select is linear in both
    tangents, so reverse mode transposes it without a rule of the refactor's
    own and higher orders compose.
    """
    values, _, predicate = primals[:3]
    dvalues, dheld = tangents[0], tangents[1]
    outs = phased_refactor_p.bind(*primals, **params)
    out_tangents = _zero_tangents(outs)
    if not getattr(params["analysis"].backend, "refactor_keeps_values", True):
        # A backend whose factor preconditions a current operator keeps the new
        # values in every lane, so they carry the tangent of the new values.
        if not _is_zero(dvalues):
            out_tangents[0] = ad.instantiate_zeros(dvalues)
        return outs, out_tangents
    if _is_zero(dvalues) and _is_zero(dheld):
        return outs, out_tangents
    dvalues, dheld = ad.instantiate_zeros(dvalues), ad.instantiate_zeros(dheld)
    mask = jnp.broadcast_to(jnp.expand_dims(predicate, -1), jnp.shape(values))
    out_tangents[0] = jax.lax.select(mask, dvalues, dheld)
    return outs, out_tangents


def _invalidate_jvp(primals, tangents, **params):
    """The invalidation on the primal handle; its order word has a zero tangent."""
    del tangents
    outs = phased_invalidate_p.bind(*primals, **params)
    return outs, _zero_tangents(outs)


def _derivative_params(params: dict) -> dict:
    """The parameters of a derivative solve: no initial guess, and no shared-storage trust.

    A derivative solve's order result is dropped (a tangent is not part of
    the primal order chain), so no later operation waits for it; it applies
    the rule of a factor of its own storage, which stays exact whatever runs
    on the storage before it.
    """
    return (
        dict(params, has_x0=False, storage=None)
        if "has_x0" in params
        else dict(params, storage=None)
    )


def _solve_jvp(primals, tangents, **params):
    """``dx = solve(db - dM x)`` on the same handle; the status, the counts and the order word have no tangent.

    The tangents of an inexact solve's ``weights`` and ``x0`` contribute
    nothing: the solve is differentiated as the exact inverse of ``M``. The
    tangent solve takes the order word the primal solve took, binds the same
    ``weights``, no initial guess and no shared storage, and its own order
    result is dropped.
    """
    n_handle = params["n_handle"]
    has_weights, has_x0 = _solve_flags(params)
    values, *operands = primals
    handle, order, b, weights, _ = _split_solve_operands(
        "phased_solve", operands, n_handle, has_weights, has_x0
    )
    dvalues, db = tangents[0], tangents[n_handle + 2]
    outs = list(phased_solve_p.bind(*primals, **params))
    x = outs[0]
    rest_dots = _zero_tangents(outs[1:])
    values_zero, b_zero = _is_zero(dvalues), _is_zero(db)
    if values_zero and b_zero:
        (x_dot,) = _zero_tangents([x])
        return outs, [x_dot, *rest_dots]
    rhs = None if b_zero else db
    if not values_zero:
        effective = (params["orientation"] == "AT") != bool(params["transpose"])
        dm_x = pattern_matvec(params["analysis"], dvalues, x, transpose=effective)
        # x has the result dtype, which is complex whenever the values or b
        # are, so this cast never drops an imaginary part; it brings dM x to
        # the rung's width when the values are wider than the rung (float64
        # values on an fp32 factor), which is the dtype db and the bind need.
        dm_x = dm_x.astype(x.dtype)
        rhs = -dm_x if rhs is None else rhs - dm_x
    extras = (weights,) if has_weights else ()
    x_dot = phased_solve_p.bind(values, *handle, order, rhs, *extras, **_derivative_params(params))[
        0
    ]
    return outs, [x_dot, *rest_dots]


def _solve_transpose(cts, values, *operands, **params):
    """The cotangent of ``b``: a solve with ``transpose`` flipped on the cotangent of ``x``.

    The flipped solve takes the order word of the forward solve (a residual
    in reverse mode) and its own order result is dropped. When that order
    word is itself an output of a linear operation (the forward solve
    followed another solve on a linear right-hand side, as the correction
    solves of a refined tangent do), it has no value in the backward pass:
    JAX counts every output of an equation with a linear input as linear.
    The flipped solve then takes a zero order word. Its value is never read
    and a derivative solve never trusts shared storage, so it stays exact; it
    is ordered after the operations that produced its handle and its
    cotangent. An inexact solve's flipped bind keeps ``weights`` and drops
    ``x0``; neither receives a cotangent.
    """
    has_weights, has_x0 = _solve_flags(params)
    handle, order, b, weights, x0 = _split_solve_operands(
        "phased_solve", operands, params["n_handle"], has_weights, has_x0
    )
    if (
        ad.is_undefined_primal(values)
        or any(ad.is_undefined_primal(leaf) for leaf in handle)
        or not ad.is_undefined_primal(b)
        or any(extra is not None and ad.is_undefined_primal(extra) for extra in (weights, x0))
    ):
        raise FactorContractError(
            "phased_solve is linear in its right-hand side only: it transposes with respect "
            "to b, with the values, the handle, the weights and the initial guess known."
        )
    if ad.is_undefined_primal(order):
        order = jnp.zeros(tuple(order.aval.shape), ORDER_DTYPE)
    x_bar = cts[0]
    if type(x_bar) is ad.Zero:
        b_bar = ad.Zero(b.aval.to_tangent_aval())
    else:
        # b is bound in the result dtype, which x and its cotangent share, so
        # the flipped solve returns the cotangent of b in b's dtype. A caller
        # that cast b on entry gets the conversion back through that cast.
        # The flipped solve never trusts shared storage: reverse mode solves
        # with the factors of every step after later refactors of the storage,
        # so it applies the rule of a factor of its own storage.
        flipped = dict(_derivative_params(params), transpose=not params["transpose"])
        extras = (weights,) if has_weights else ()
        b_bar = phased_solve_p.bind(values, *handle, order, x_bar, *extras, **flipped)[0]
    return [None] * (1 + len(handle)) + [None, b_bar] + [None] * (int(has_weights) + int(has_x0))


# --------------------------------------------------------------------------- #
# Rematerialisation
# --------------------------------------------------------------------------- #
def _saves_outputs(eqn: Any) -> bool:
    """Whether rematerialisation keeps ``eqn`` known and saves its outputs.

    Every factorization, refactorization and invalidation, and every solve on
    a backend whose ``factor_residency`` is not ``"arrays"``: recomputing
    one would factor again, or solve with native storage that later
    operations have since rewritten.
    """
    if eqn.primitive is not phased_solve_p:
        return True
    backend = eqn.params["analysis"].backend
    return getattr(backend, "factor_residency", "arrays") != "arrays"


def _remat_rule(saveable, unks_in, inst_in, eqn):
    """The partial-evaluation rule of ``jax.checkpoint`` for the phased primitives.

    An equation with an unknown operand is staged with its known operands
    as residuals, as any equation is. An equation whose operands are all
    known runs on the known side; when
    [_saves_outputs][diff_soup.linear._primitive._saves_outputs] holds its
    outputs are saved (never recomputed), and otherwise (a solve on
    array-held factors) the rematerialisation policy decides, as for any
    equation.
    """
    n_out = len(eqn.outvars)
    residuals = [var for var, inst in zip(eqn.invars, inst_in) if isinstance(var, Var) and not inst]
    if any(unks_in):
        return None, eqn, [True] * n_out, [True] * n_out, residuals
    if not _saves_outputs(eqn):
        policy = internal_partial_eval.ensure_enum(
            saveable(eqn.primitive, *[var.aval for var in eqn.invars], **eqn.params)
        )
        if isinstance(policy, internal_partial_eval.RecomputeType):
            return eqn, eqn, [False] * n_out, [True] * n_out, residuals
    return eqn, None, [False] * n_out, [False] * n_out, []


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


# --------------------------------------------------------------------------- #
# Evaluation and lowering
# --------------------------------------------------------------------------- #
def _lowering_hook(kind: str, params: dict) -> Optional[Callable]:
    """The backend's MLIR lowering hook for ``kind``, or ``None``."""
    return getattr(params["analysis"].backend, _HOOKS[kind], None)


def _conform(kind: str, analysis: Any, outs: Any, avals_out: Sequence[Any]) -> list:
    """Check a JAX-level backend function's outputs against the abstract outputs."""
    outs = list(outs)
    label, fun = _backend_label(analysis), _FUNS[kind]
    if len(outs) != len(avals_out):
        raise FactorContractError(
            f"The {label} backend's {fun} returned {len(outs)} outputs; the contract requires "
            f"{len(avals_out)}, the order word last."
        )
    conformed = []
    for position, (out, aval) in enumerate(zip(outs, avals_out)):
        out = jnp.asarray(out)
        if tuple(out.shape) != tuple(aval.shape):
            raise FactorContractError(
                f"The {label} backend's {fun} returned output {position} with shape "
                f"{list(out.shape)}; the contract requires {list(aval.shape)}."
            )
        conformed.append(out if out.dtype == aval.dtype else out.astype(aval.dtype))
    return conformed


def _jax_level(kind: str, avals_out: Sequence[Any]) -> Callable:
    """The backend's JAX-level function for ``kind``, called with the primitive's operands.

    Each function receives the order word as the keyword ``order`` and
    returns it as its last output. An invalidation has no JAX-level form:
    it returns the order word unchanged.
    """

    def run(*operands: Any, **params: Any) -> list:
        analysis = params["analysis"]
        if kind == "invalidate":
            # Without a lowering hook there is no native storage to mark.
            return _conform(kind, analysis, (operands[-1],), avals_out)
        backend = analysis.backend
        common = dict(
            analysis=analysis,
            orientation=params["orientation"],
            precision=params["precision"],
            tol=params["tol"],
        )
        if kind == "factor":
            values, token, context_id, order = operands
            outs = backend.factor_fun(values, token, context_id, order=order, **common)
        elif kind == "apply":
            values, *rest = operands
            handle, order, b, weights, x0 = _split_solve_operands(
                "phased_solve", rest, params["n_handle"], *_solve_flags(params)
            )
            if not _is_exact(analysis):
                # An inexact backend also receives the caller's weights and initial guess.
                common.update(weights=weights, x0=x0)
            outs = backend.apply_fun(
                values, *handle, order=order, b=b, transpose=params["transpose"], **common
            )
        else:
            values, held_values, predicate, *handle, order = operands
            outs = backend.refactor_fun(
                values,
                held_values,
                predicate,
                *handle,
                order=order,
                adopt=params["adopt"],
                **common,
            )
        return _conform(kind, analysis, outs, avals_out)

    return run


@functools.lru_cache(maxsize=64)
def _compiled(primitive: Primitive, frozen_params: tuple) -> Callable:
    """A compiled program that binds ``primitive`` once with fixed parameters."""
    params = dict(frozen_params)

    def run(*operands: Any) -> Any:
        return primitive.bind(*operands, **params)

    return jax.jit(run)


#: Slot tokens of eager factorizations through a lowering hook: each eager
#: call takes the next one, so no two eager factorizations of the process
#: share a (context, call site, slot token) key and each keeps storage of its
#: own.
_EAGER_TOKENS = itertools.count(1)


def _eager_token() -> jax.Array:
    """A slot token no earlier eager factorization of this process has used (wrapping at ``2**31``)."""
    return jnp.asarray(next(_EAGER_TOKENS) % (2**31), jnp.int32)


def _impl(kind: str, primitive: Primitive) -> Callable:
    """Eager evaluation: the JAX-level function, or a compiled bind for a lowering hook.

    A backend with a lowering hook runs a compiled one-equation program,
    cached per parameter set, so an eager call takes the same native route
    as a call inside ``jit``. Every eager factorization through it gets a
    fresh slot token, so eager factors never share a slot of that program's
    call site. A backend without one receives every operand as a JAX
    array: an operand bound as a NumPy scalar (the context id and slot
    token [factor][diff_soup.linear.factor] binds, so that they lower as
    program constants) is converted first.
    """

    def impl(*operands: Any, **params: Any) -> Any:
        if _lowering_hook(kind, params) is not None:
            if kind == "factor":
                values, _, context_id, order = operands
                operands = (values, _eager_token(), context_id, order)
            with jax.disable_jit(False):
                return _compiled(primitive, tuple(sorted(params.items())))(*operands)
        operands = tuple(
            jnp.asarray(op) if isinstance(op, (np.ndarray, np.generic)) else op for op in operands
        )
        avals_out = primitive.abstract_eval(*(jax.typeof(op) for op in operands), **params)[0]
        return _jax_level(kind, avals_out)(*operands, **params)

    return impl


def _check_platform(ctx: Any, analysis: Any) -> None:
    """Refuse a lowering for a platform outside the backend's ``platforms``.

    Raises:
        FactorContractError: Naming the backend, the platform and device the
            program is compiled for, and the platforms the backend runs on.
    """
    allowed = tuple(getattr(analysis.backend, "platforms", ()) or ())
    if not allowed:
        return
    lowered_for = tuple(getattr(ctx.module_context, "platforms", ()) or ())
    outside = [platform for platform in lowered_for if platform not in allowed]
    if not outside:
        return
    try:
        device = str(jax.devices(outside[0])[0])
    except RuntimeError:
        device = "unavailable"
    raise FactorContractError(
        f"The {_backend_label(analysis)} backend runs on the {' or '.join(allowed)} platform "
        f"only, but this program is compiled for the {' and '.join(outside)} platform "
        f"(device {device}). Compile it for {allowed[0]}: give it {allowed[0]} arrays or run it "
        f"under jax.default_device(jax.devices({allowed[0]!r})[0]), or analyze with a backend "
        f"that runs on {outside[0]}."
    )


def _lowering(kind: str) -> Callable:
    """The lowering rule for ``kind``: the backend's hook, else ``mlir.lower_fun``.

    The rule runs once per equation (the primitives are registered
    uncacheable). A hook receives every operand,
    the order word included, and the ``storage`` parameter only when it
    names shared storage, and returns the MLIR values of every result, the
    order word last. A program lowered for a platform the backend does not
    declare in ``platforms`` is refused here, before any custom call is
    emitted.
    """

    def rule(ctx: Any, *args: Any, **params: Any) -> Any:
        _check_platform(ctx, params["analysis"])
        hook = _lowering_hook(kind, params)
        if hook is not None:
            if params.get("storage", "") is None:
                # A hook receives the storage name only for a factor that has one.
                params = {key: value for key, value in params.items() if key != "storage"}
            return hook(ctx, *args, **params)
        lowered = mlir.lower_fun(_jax_level(kind, ctx.avals_out), multiple_results=True)
        return lowered(ctx, *args, **params)

    return rule


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


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
phased_factor_p.def_abstract_eval(_factor_abstract_eval)
phased_solve_p.def_abstract_eval(_solve_abstract_eval)
phased_refactor_p.def_abstract_eval(_refactor_abstract_eval)
phased_invalidate_p.def_abstract_eval(_invalidate_abstract_eval)

for _primitive, _kind in (
    (phased_factor_p, "factor"),
    (phased_solve_p, "apply"),
    (phased_refactor_p, "refactor"),
    (phased_invalidate_p, "invalidate"),
):
    _primitive.def_impl(_impl(_kind, _primitive))
    # Uncacheable: each equation runs the rule, which sees the operands'
    # producing operations (the cuDSS lowerings read constant context ids and
    # slot tokens from them).
    mlir.register_lowering(_primitive, _lowering(_kind), cacheable=False)
    partial_eval.partial_eval_jaxpr_custom_rules[_primitive] = _remat_rule

ad.primitive_jvps[phased_factor_p] = _factor_jvp
ad.primitive_jvps[phased_solve_p] = _solve_jvp
ad.primitive_jvps[phased_refactor_p] = _refactor_jvp
ad.primitive_jvps[phased_invalidate_p] = _invalidate_jvp
ad.primitive_transposes[phased_solve_p] = _solve_transpose

batching.fancy_primitive_batchers[phased_factor_p] = _factor_batch
batching.fancy_primitive_batchers[phased_solve_p] = _solve_batch
batching.fancy_primitive_batchers[phased_refactor_p] = _refactor_batch
batching.fancy_primitive_batchers[phased_invalidate_p] = _invalidate_batch
