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
takes the branch ``lax.cond`` takes and runs no other (the compiled program
carries the batched form's outer conditional, with the same executed work).
Batched, it runs ``true_fun`` only when at least one lane
takes it, and ``false_fun`` only when at least one lane does not; each lane
keeps its own branch's result.

A batched loop carries its finished lanes along until the last lane finishes.
[live_lane_cond][diff_soup.utils.lanes.live_lane_cond] takes each lane's loop
condition and lets only the lanes still running decide which branches run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from diff_soup._jax_compat import Primitive, ad, batching, mlir


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


#: ``pred`` unbatched; under ``vmap``, whether any live lane holds it.
_any_live_lane_p = _live_reduction("diffsoup_any_live_lane", jnp.any, jnp.logical_and, _any_lane_p)
#: ``pred`` unbatched; under ``vmap``, whether every live lane holds it.
_all_live_lanes_p = _live_reduction(
    "diffsoup_all_live_lanes",
    jnp.all,
    lambda pred, live: jnp.logical_or(pred, jnp.logical_not(live)),
    _all_lanes_p,
)


def _dispatch(pred, true_fun, false_fun, operands, false_is_free, any_lane, all_lanes):
    """Run the branches some lane takes; ``any_lane()``/``all_lanes()`` reduce ``pred``."""

    def either(*ops):
        taken = true_fun(*ops)
        kept = false_fun(*ops)
        return jax.tree_util.tree_map(lambda t, f: jnp.where(pred, t, f), taken, kept)

    if false_is_free:
        return jax.lax.cond(any_lane(), either, false_fun, *operands)

    def mixed_or_false(*ops):
        return jax.lax.cond(any_lane(), either, false_fun, *ops)

    return jax.lax.cond(all_lanes(), true_fun, mixed_or_false, *operands)


def lane_cond(
    pred: Any,
    true_fun: Callable,
    false_fun: Callable,
    *operands: Any,
    false_is_free: bool = False,
) -> Any:
    """``lax.cond`` whose branches run under ``vmap`` only when some lane takes them.

    Unbatched this is ``jax.lax.cond(pred, true_fun, false_fun, *operands)``.
    Under ``vmap`` with a per-lane ``pred``, only ``true_fun`` runs when every
    lane takes it, only ``false_fun`` when none does, and otherwise both run
    for the whole batch and each lane keeps its own branch's result. Each lane
    therefore sees the result it would see alone, provided a branch has no
    effect beyond its return value on the lanes that discard it.

    Args:
        pred: Scalar boolean predicate (per lane under ``vmap``).
        true_fun: The branch taken where ``pred`` holds.
        false_fun: The branch taken elsewhere.
        *operands: Arguments passed to the branch.
        false_is_free: Declares that ``false_fun`` computes nothing (it returns
            its operands or values computed outside the branch). ``false_fun``
            then also runs wherever ``true_fun`` runs, and ``true_fun`` is
            traced once instead of twice.
    """
    pred = jnp.asarray(pred, dtype=jnp.bool_)
    lanes = (lambda: _any_lane_p.bind(pred), lambda: _all_lanes_p.bind(pred))
    return _dispatch(pred, true_fun, false_fun, operands, false_is_free, *lanes)


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
    is not live may receive the other branch's result, so the caller must
    discard it. Unbatched, ``live`` is not read and a branch runs exactly where
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
    if live is None:
        return lane_cond(pred, true_fun, false_fun, *operands, false_is_free=false_is_free)
    pred = jnp.asarray(pred, dtype=jnp.bool_)
    live = jnp.asarray(live, dtype=jnp.bool_)
    lanes = (lambda: _any_live_lane_p.bind(pred, live), lambda: _all_live_lanes_p.bind(pred, live))
    return _dispatch(pred, true_fun, false_fun, operands, false_is_free, *lanes)


__all__ = ["lane_cond", "live_lane_cond"]
