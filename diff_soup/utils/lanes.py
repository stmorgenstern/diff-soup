"""Branch selection that stays a branch under ``vmap``.

A batch of independent solves (the windows of a multiple-shooting residual,
the experiments of a joint fit, a ``vmap`` over parameter points) runs as one
``jax.vmap`` over the solves' lanes. Under ``vmap`` a ``lax.cond`` whose
predicate differs between lanes becomes a ``select``: both branches run on
every lane at every call. For the refresh branches of an implicit step
(Jacobian assembly, refactorization, the retry after a stale-Jacobian failure,
the frozen iterations of a bounded Newton loop) that turns a rare event into
per-step work, and a batch of solves pays several times the cost of the same
solves run one by one.

[lane_cond][diff_soup.utils.lanes.lane_cond] keeps the branch. Unbatched it
compiles to one ``lax.cond`` and runs one branch. Batched, it runs
``true_fun`` only when at least one lane takes it and ``false_fun`` only when
at least one lane does not, and each lane keeps its own branch's result. Each
branch is traced once and the compiled program holds one copy of it.

A batched loop carries its finished lanes along until the last lane finishes.
[live_lane_cond][diff_soup.utils.lanes.live_lane_cond] takes each lane's loop
condition and lets only the lanes still running decide which branches run.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from diff_soup._jax_compat import Primitive, ad, batching, internal_batching, mlir
from diff_soup._jax_compat import internal_api_util as api_util
from diff_soup._jax_compat import internal_control_flow as control_flow
from diff_soup._jax_compat import internal_core as core
from diff_soup._jax_compat import internal_flattree as flattree
from diff_soup._jax_compat import internal_partial_eval as pe


def _reduction(name: str, reduce: Callable) -> Primitive:
    """A boolean identity whose ``vmap`` reduces the predicate over the batch."""
    primitive = Primitive(name)
    primitive.def_impl(lambda pred: pred)
    primitive.def_abstract_eval(lambda aval: aval)
    mlir.register_lowering(primitive, mlir.lower_fun(lambda pred: pred, multiple_results=False))

    def batch(args, dims):
        (pred,), (dim,) = args, dims
        return primitive.bind(reduce(pred, axis=dim)), None

    def jvp(primals, tangents):
        out = primitive.bind(*primals)
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())

    batching.primitive_batchers[primitive] = batch
    ad.primitive_jvps[primitive] = jvp
    return primitive


#: ``pred`` unbatched; under ``vmap``, whether any (all) lanes hold it.
_any_lane_p = _reduction("diffsoup_any_lane", jnp.any)
_all_lanes_p = _reduction("diffsoup_all_lanes", jnp.all)


def _live_reduction(name: str, reduce: Callable, count: Callable, lanes: Primitive) -> Primitive:
    """A boolean identity on ``pred`` whose ``vmap`` reduces it over the live lanes only.

    Unbatched the primitive returns ``pred`` and ``live`` is unused. Batched,
    ``count(pred, live)`` replaces the predicate of a lane that is not live by
    the reduction's neutral value, and the result enters ``lanes``, which
    reduces it again at every outer ``vmap`` level.
    """
    primitive = Primitive(name)
    primitive.def_impl(lambda pred, live: pred)
    primitive.def_abstract_eval(lambda pred, live: pred)
    mlir.register_lowering(
        primitive, mlir.lower_fun(lambda pred, live: pred, multiple_results=False)
    )

    def batch(args, dims):
        # At least one operand carries the batch axis; an unbatched one broadcasts against it.
        pred, live = (a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims))
        return lanes.bind(reduce(count(pred, live), axis=0)), None

    def jvp(primals, tangents):
        out = primitive.bind(*primals)
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())

    batching.primitive_batchers[primitive] = batch
    ad.primitive_jvps[primitive] = jvp
    return primitive


def _counted_true(pred, live):
    """A live lane that takes the true branch."""
    return jnp.logical_and(pred, live)


def _not_counted_false(pred, live):
    """A lane that does not stop the true branch from being the only one run."""
    return jnp.logical_or(pred, jnp.logical_not(live))


#: ``pred`` unbatched; under ``vmap``, whether any live lane holds it.
_any_live_lane_p = _live_reduction("diffsoup_any_live_lane", jnp.any, _counted_true, _any_lane_p)
#: ``pred`` unbatched; under ``vmap``, whether every live lane holds it.
_all_live_lanes_p = _live_reduction(
    "diffsoup_all_live_lanes", jnp.all, _not_counted_false, _all_lanes_p
)


def any_lane(pred: Any) -> jax.Array:
    """``pred`` unbatched; under ``vmap``, whether any lane holds it.

    The result is the same on every lane, at every level of nesting, so a
    ``lax.cond`` on it stays a conditional under ``vmap``.

    Args:
        pred: Scalar boolean predicate (per lane under ``vmap``).
    """
    return _any_lane_p.bind(jnp.asarray(pred, dtype=jnp.bool_))


def _lane_where_batch(args, dims):
    """Select per lane on an axis that batches ``pred``; keep the primitive for an outer one."""
    pred, taken, other = args
    pred_dim, taken_dim, other_dim = dims
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)

    def lanes(x, d):
        return jnp.broadcast_to(x, (size, *x.shape)) if d is None else jnp.moveaxis(x, d, 0)

    taken, other = lanes(taken, taken_dim), lanes(other, other_dim)
    if pred_dim is None:
        return _lane_where_p.bind(pred, taken, other), 0
    pred = jnp.moveaxis(pred, pred_dim, 0)
    pred = jnp.expand_dims(pred, tuple(range(pred.ndim, taken.ndim)))
    return jnp.where(pred, taken, other), 0


def _lane_where_jvp(primals, tangents):
    """The tangent of a lane select: the same select of the two tangents."""
    pred, taken, other = primals
    out = _lane_where_p.bind(pred, taken, other)
    _, dtaken, dother = tangents
    if type(dtaken) is ad.Zero and type(dother) is ad.Zero:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    return out, _lane_where_p.bind(pred, ad.instantiate_zeros(dtaken), ad.instantiate_zeros(dother))


def _lane_where_transpose(cotangent, pred, taken, other):
    """The transpose of a lane select: each lane's cotangent goes to the operand it took."""
    assert not ad.is_undefined_primal(pred)
    if type(cotangent) is ad.Zero:
        return (
            None,
            *(ad.Zero(a.aval) if ad.is_undefined_primal(a) else None for a in (taken, other)),
        )
    zeros = jnp.zeros_like(cotangent)
    return (
        None,
        _lane_where_p.bind(pred, cotangent, zeros) if ad.is_undefined_primal(taken) else None,
        _lane_where_p.bind(pred, zeros, cotangent) if ad.is_undefined_primal(other) else None,
    )


#: ``taken`` unbatched; under ``vmap``, ``where(pred, taken, other)`` per lane.
_lane_where_p = Primitive("diffsoup_lane_where")
_lane_where_p.def_impl(lambda pred, taken, other: taken)
_lane_where_p.def_abstract_eval(lambda pred, taken, other: taken)
mlir.register_lowering(
    _lane_where_p, mlir.lower_fun(lambda pred, taken, other: taken, multiple_results=False)
)
batching.primitive_batchers[_lane_where_p] = _lane_where_batch
ad.primitive_jvps[_lane_where_p] = _lane_where_jvp
ad.primitive_transposes[_lane_where_p] = _lane_where_transpose


def lane_where(pred: Any, taken: Any, other: Any) -> Any:
    """``where(pred, taken, other)`` whose unbatched form is ``taken`` and reads nothing else.

    For a selection whose predicate can differ from true only on a lane of a batch: a lane
    a batched loop keeps alive after it finished, or a lane that took a branch only because
    another lane needed it. Unbatched, the caller guarantees ``pred`` holds, so ``other`` is
    dead after this call and XLA can update it in place. Under ``vmap`` with a per-lane
    ``pred`` each lane keeps its own side, for every level of nesting.

    Args:
        pred: Scalar boolean predicate (per lane under ``vmap``).
        taken: The array of the lanes where ``pred`` holds, and the result unbatched.
        other: The array of the other lanes.
    """
    pred = jnp.asarray(pred, dtype=jnp.bool_)
    return _lane_where_p.bind(pred, taken, other)


# The lane conditional. ``_lane_cond_p`` takes ``(pred, live, *operands)`` and the two
# branches as jaxprs ``(false, true)`` over the operands, as ``lax.cond`` stores them.
# Unbatched it lowers to one ``lax.cond``. Its batching rule and its partial evaluation
# (the linearization of reverse mode, and ``jax.checkpoint``) both write it as
# ``_two_conditionals``: plain conditionals on the any-lane and all-lanes reductions,
# which ``lax.cond``'s own rules then batch, differentiate and transpose.


def _where(pred, taken, kept):
    """``where(pred, taken, kept)``, ``pred`` per leading-axis lane."""
    ndim = max(jnp.ndim(taken), jnp.ndim(kept))
    return jnp.where(jnp.reshape(pred, pred.shape + (1,) * (ndim - pred.ndim)), taken, kept)


def _computed(jaxpr):
    """The outputs a branch computes, and for each output the operand it forwards, if any."""
    forwards = pe._jaxpr_forwarding(jaxpr)
    computed = pe.prune_closed_jaxpr_outputs(jaxpr, [f is None for f in forwards])
    return core.jaxpr_as_fun(computed), computed.out_avals, forwards


def _two_conditionals(pred, any_true, all_true, branches, false_is_free, operands):
    """Run each branch where some lane takes it, with one copy of each branch.

    ``any_true()`` and ``all_true()`` bind the lane reductions of ``pred``, each
    only where a conditional reads it.

    The false branch runs first, unless every lane takes the true branch; the
    true branch runs second, where some lane takes it, together with the
    select of each lane's result. An output a branch forwards from its
    operands bypasses that branch's conditional. A false branch declared free
    runs inside the true branch's conditional instead.
    """
    false_jaxpr, true_jaxpr = branches
    if false_is_free:
        true_fun, false_fun = (core.jaxpr_as_fun(j) for j in (true_jaxpr, false_jaxpr))

        def either(*ops):
            return [_where(pred, t, f) for t, f in zip(true_fun(*ops), false_fun(*ops))]

        return jax.lax.cond(any_true(), either, false_fun, *operands)

    false_fun, false_avals, false_forwards = _computed(false_jaxpr)
    true_fun, _, true_forwards = _computed(true_jaxpr)

    def skipped(*_):
        return [jnp.zeros(a.shape, a.dtype) for a in false_avals]

    computed = iter(jax.lax.cond(all_true(), skipped, false_fun, *operands))
    kept = [next(computed) if f is None else operands[f] for f in false_forwards]

    def mixed(kept, *ops):
        return [_where(pred, t, k) for t, k in zip(true_fun(*ops), kept)]

    selected = iter(
        jax.lax.cond(
            any_true(),
            mixed,
            lambda kept, *_: kept,
            [k for k, f in zip(kept, true_forwards) if f is None],
            *operands,
        )
    )
    return [
        next(selected) if f is None else _where(pred, operands[f], k)
        for k, f in zip(kept, true_forwards)
    ]


def _expanded(pred, live, *operands, branches, false_is_free):
    """``_lane_cond_p`` written as plain conditionals on the lane reductions."""
    return _two_conditionals(
        pred,
        lambda: _any_live_lane_p.bind(pred, live),
        lambda: _all_live_lanes_p.bind(pred, live),
        branches,
        false_is_free,
        operands,
    )


def _lane_cond_impl(pred, live, *operands, branches, false_is_free):
    """One ``lax.cond``: unbatched, ``live`` is not read and one branch runs."""
    false_jaxpr, true_jaxpr = branches
    return jax.lax.cond(
        pred, core.jaxpr_as_fun(true_jaxpr), core.jaxpr_as_fun(false_jaxpr), *operands
    )


def _lane_cond_abstract_eval(pred, live, *operands, branches, false_is_free):
    """The true branch's outputs, which the false branch's match."""
    return branches[1].out_avals


def _lane_cond_batch(axis_data, args, dims, *, branches, false_is_free):
    """Batch the lane conditional: one conditional for a shared predicate, else two."""
    pred, live, *operands = args
    pred_dim, live_dim, *op_dims = dims
    ops_batched = [d is not None for d in op_dims]
    operands = [x if d is None else jnp.moveaxis(x, d, 0) for x, d in zip(operands, op_dims)]
    if pred_dim is None:
        # Every lane takes the same branch; a lane group counts when any of its lanes is live.
        if live_dim is not None:
            live = jnp.any(live, axis=live_dim)
        out_batched = [
            any(batched)
            for batched in zip(
                *(
                    internal_batching.batch_jaxpr(j, axis_data, ops_batched, False)[1]
                    for j in branches
                )
            )
        ]
        batched = tuple(
            internal_batching.batch_jaxpr(j, axis_data, ops_batched, out_batched)[0]
            for j in branches
        )
        out = _lane_cond_p.bind(
            pred, live, *operands, branches=batched, false_is_free=false_is_free
        )
        return out, [0 if b else None for b in out_batched]
    pred = jnp.moveaxis(pred, pred_dim, 0)
    live = (
        jnp.broadcast_to(live, pred.shape) if live_dim is None else jnp.moveaxis(live, live_dim, 0)
    )
    batched = tuple(
        internal_batching.batch_jaxpr(j, axis_data, ops_batched, True)[0] for j in branches
    )
    out = _two_conditionals(
        pred,
        lambda: _any_lane_p.bind(jnp.any(_counted_true(pred, live))),
        lambda: _all_lanes_p.bind(jnp.all(_not_counted_false(pred, live))),
        batched,
        false_is_free,
        operands,
    )
    return out, [0] * len(out)


def _lane_cond_jvp(primals, tangents, *, branches, false_is_free):
    """The lane conditional of the branches' JVPs, as ``lax.cond`` differentiates."""
    pred, live, *operands = primals
    _, _, *op_dots = tangents
    nonzero = [type(t) is not ad.Zero for t in op_dots]
    out_nonzero = [
        any(nz) for nz in zip(*(ad.jvp_jaxpr(j, nonzero, instantiate=False)[1] for j in branches))
    ]
    jvp_branches = tuple(ad.jvp_jaxpr(j, nonzero, instantiate=out_nonzero)[0] for j in branches)
    dots = [t for t, nz in zip(op_dots, nonzero) if nz]
    out = _lane_cond_p.bind(
        pred, live, *operands, *dots, branches=jvp_branches, false_is_free=false_is_free
    )
    out_primals, out_dots = out[: len(out_nonzero)], iter(out[len(out_nonzero) :])
    out_tangents = [
        next(out_dots) if nz else ad.Zero(jax.typeof(p).to_tangent_aval())
        for p, nz in zip(out_primals, out_nonzero)
    ]
    return out_primals, out_tangents


def _lane_cond_partial_eval(trace, *tracers, branches, false_is_free):
    """Partially evaluate the plain conditionals the lane conditional expands to."""
    with core.set_current_trace(trace):
        return _expanded(*tracers, branches=branches, false_is_free=false_is_free)


def _lane_cond_partial_eval_custom(saveable, unks_in, inst_in, eqn):
    """``jax.checkpoint``'s partial evaluation, on the plain conditionals as a closed call."""
    expanded = jax.make_jaxpr(lambda *args: _expanded(*args, **eqn.params))(
        *(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in eqn.invars)
    )
    call = eqn.replace(primitive=core.closed_call_p, params={"call_jaxpr": expanded})
    rule = pe.partial_eval_jaxpr_custom_rules[core.closed_call_p]
    return rule(saveable, unks_in, inst_in, call)


def _lane_cond_dce(used_outputs, eqn):
    """Drop the unused outputs from both branches, and the operands neither branch reads."""
    if not any(used_outputs):
        return [False] * len(eqn.invars), None
    branches = eqn.params["branches"]
    used_ops = [False] * (len(eqn.invars) - 2)
    for jaxpr in branches:
        used_ops = list(map(operator.or_, used_ops, pe.dce_jaxpr(jaxpr, used_outputs, False)[1]))
    kept = tuple(pe.dce_jaxpr(jaxpr, used_outputs, instantiate=used_ops)[0] for jaxpr in branches)
    used_inputs = [True, True, *used_ops]
    new_eqn = pe.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive,
        dict(eqn.params, branches=kept),
        eqn.effects,
        eqn.source_info,
        eqn.ctx,
    )
    return used_inputs, new_eqn


_lane_cond_p = Primitive("diffsoup_lane_cond")
_lane_cond_p.multiple_results = True
_lane_cond_p.def_impl(_lane_cond_impl)
_lane_cond_p.def_abstract_eval(_lane_cond_abstract_eval)
mlir.register_lowering(_lane_cond_p, mlir.lower_fun(_lane_cond_impl, multiple_results=True))
batching.fancy_primitive_batchers[_lane_cond_p] = _lane_cond_batch
ad.primitive_jvps[_lane_cond_p] = _lane_cond_jvp
pe.custom_partial_eval_rules[_lane_cond_p] = _lane_cond_partial_eval
pe.partial_eval_jaxpr_custom_rules[_lane_cond_p] = _lane_cond_partial_eval_custom
pe.dce_rules[_lane_cond_p] = _lane_cond_dce


def _branches(true_fun, false_fun, operands):
    """Trace both branches once, as ``lax.cond`` does, with their closures as operands."""
    args = flattree.flatten((operands, {}))
    avals = args.map(core.typeof)
    traced = []
    for name, fun in (("true_fun", true_fun), ("false_fun", false_fun)):
        debug = api_util.debug_info(f"lane_cond {name}", fun, operands, {})
        jaxpr, out_avals = pe.trace_to_jaxpr(fun, avals, debug)
        traced.append((*pe.separate_consts(jaxpr), out_avals))
    (true_jaxpr, true_consts, out_avals), (false_jaxpr, false_consts, false_avals) = traced
    if out_avals.tree != false_avals.tree or not all(map(core.typematch, out_avals, false_avals)):
        raise TypeError(
            "lane_cond branch outputs must have the same pytree structure and types: "
            f"true_fun returns {out_avals.unflatten()}, false_fun returns "
            f"{false_avals.unflatten()}."
        )
    (true_jaxpr, false_jaxpr), consts = control_flow._merge_common_consts(
        (true_jaxpr, false_jaxpr), (true_consts, false_consts)
    )
    return (false_jaxpr, true_jaxpr), [*consts, *args], out_avals


def _lane_cond(pred, true_fun, false_fun, operands, live, false_is_free):
    """Bind the lane conditional; a branch with effects takes the plain conditionals."""
    pred = jnp.asarray(pred, dtype=jnp.bool_)
    live = jnp.asarray(True if live is None else live, dtype=jnp.bool_)
    branches, args, out_avals = _branches(true_fun, false_fun, operands)
    params = dict(branches=branches, false_is_free=false_is_free)
    if any(j.effects for j in branches):
        out = _expanded(pred, live, *args, **params)
    else:
        out = _lane_cond_p.bind(pred, live, *args, **params)
    return out_avals.update(out).unflatten()


def lane_cond(
    pred: Any,
    true_fun: Callable,
    false_fun: Callable,
    *operands: Any,
    false_is_free: bool = False,
) -> Any:
    """``lax.cond`` whose branches run under ``vmap`` only when some lane takes them.

    Unbatched this is ``jax.lax.cond(pred, true_fun, false_fun, *operands)``,
    compiled to one conditional. Under ``vmap`` with a per-lane ``pred``, only
    ``true_fun`` runs when every lane takes it, only ``false_fun`` when none
    does, and otherwise both run for the whole batch and each lane keeps its
    own branch's result. Each lane therefore sees the result it would see
    alone, provided a branch has no effect beyond its return value on the
    lanes that discard it. Each branch is traced once, and the compiled
    program holds one copy of each, batched or not.

    Args:
        pred: Scalar boolean predicate (per lane under ``vmap``).
        true_fun: The branch taken where ``pred`` holds.
        false_fun: The branch taken elsewhere.
        *operands: Arguments passed to the branch.
        false_is_free: Declares that ``false_fun`` computes nothing (it returns
            its operands or values computed outside the branch). Under
            ``vmap`` ``false_fun`` then also runs wherever ``true_fun`` runs,
            which spares a second conditional.
    """
    return _lane_cond(pred, true_fun, false_fun, operands, None, false_is_free)


def live_lane_cond(
    pred: Any,
    true_fun: Callable,
    false_fun: Callable,
    *operands: Any,
    live: Any = None,
    false_is_free: bool = False,
) -> Any:
    """[lane_cond][diff_soup.utils.lanes.lane_cond] in which only live lanes count.

    A batched loop runs its body on every lane until the last lane finishes and
    then discards the body's result on the lanes whose loop condition is false.
    Passing that condition as ``live`` keeps those lanes from running a branch
    for the whole batch: under ``vmap``, ``true_fun`` runs only when a live lane
    takes it and ``false_fun`` only when a live lane does not. A live lane keeps
    the result of its own branch, exactly as under ``lane_cond``; a lane that
    is not live receives an unspecified value, so the caller must discard it.
    Unbatched, ``live`` is not read and a branch runs exactly where
    ``lane_cond`` runs it.

    Args:
        pred: Scalar boolean predicate (per lane under ``vmap``).
        true_fun: The branch taken where ``pred`` holds.
        false_fun: The branch taken elsewhere.
        *operands: Arguments passed to the branch.
        live: Scalar boolean, whether this lane's result is kept (per lane
            under ``vmap``). ``None`` counts every lane, as ``lane_cond`` does.
        false_is_free: As in ``lane_cond``.
    """
    return _lane_cond(pred, true_fun, false_fun, operands, live, false_is_free)


__all__ = ["any_lane", "lane_cond", "lane_where", "live_lane_cond"]
