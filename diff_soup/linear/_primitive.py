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

from diff_soup._jax_compat import (
    ad,
    batching,
    mlir,
    partial_eval,
)
from diff_soup.linear._ffi_lowering import (
    CALL_FENCE_ATTRIBUTE,
    EMPTY_SLOT,
    SLOT_DTYPES,
    cast_operand,
    empty_slot_handle,
    label_empty_slots,
    lower_ffi_call,
    lower_jnp,
    lower_slot_apply,
    lower_slot_refactor,
    lower_split_apply,
    lower_switch,
    slot_dtype,
    slot_targets,
)
from diff_soup.linear._phased import (
    ORDER_DTYPE,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)
from diff_soup.linear._phased_abstract import (
    _factor_abstract_eval,
    _invalidate_abstract_eval,
    _refactor_abstract_eval,
    _solve_abstract_eval,
    check_empty_handle,
)
from diff_soup.linear._phased_ad import (
    _factor_jvp,
    _invalidate_jvp,
    _refactor_jvp,
    _remat_rule,
    _solve_jvp,
    _solve_transpose,
    pattern_matvec,
)
from diff_soup.linear._phased_batch import (
    _factor_batch,
    _invalidate_batch,
    _refactor_batch,
    _solve_batch,
)
from diff_soup.linear._phased_impl import _impl, _lowering, lowered_site_key, storage_key

__all__ = [
    "CALL_FENCE_ATTRIBUTE",
    "EMPTY_SLOT",
    "ORDER_DTYPE",
    "SLOT_DTYPES",
    "cast_operand",
    "check_empty_handle",
    "empty_slot_handle",
    "label_empty_slots",
    "lower_ffi_call",
    "lower_jnp",
    "lower_slot_apply",
    "lower_slot_refactor",
    "lower_split_apply",
    "lower_switch",
    "lowered_site_key",
    "pattern_matvec",
    "phased_factor_p",
    "phased_invalidate_p",
    "phased_refactor_p",
    "phased_solve_p",
    "slot_dtype",
    "slot_targets",
    "storage_key",
]


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
