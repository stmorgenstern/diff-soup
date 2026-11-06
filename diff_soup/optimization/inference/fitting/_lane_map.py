"""A solve-lane ``vmap`` that absorbs enclosing ``vmap`` batches into its own axis.

Multiple-shooting and multi-experiment residuals map one solve over their
windows or children with a ``jax.vmap``. A linear backend that holds one
factorization per batch lane -- the native cuDSS factor bank -- serves exactly
one such matrix-batch axis, so an outer ``vmap`` over parameter points (MCMC
chains, importance samples, a population of trial points) reached it as a
second, nested matrix axis and was refused.

[lane_map][diff_soup.inference.fitting._lane_map.lane_map] computes exactly
``jax.vmap(kernel, in_axes)(*args)``, but when an enclosing ``vmap`` batches its
arguments it folds that batch into the lane axis: ``N`` points over ``L``
lanes run as one ``N * L``-lane map, so the solves still meet the backend with
a single matrix-batch axis. Any depth of enclosing ``vmap`` folds the same way.

Forward-mode differentiation composes with the fold. The directional
derivative is its own operation, which evaluates the lanes' primal and tangent
together (the cost of ``jax.jvp`` of the plain map) and folds an enclosing
batch of the primals into the lanes, while a batch of the tangents alone (the
direction axis of ``jax.jacfwd``) stays a direction axis, never a matrix axis.
Under ``jax.linearize`` it splits: the primal map runs on the known inputs and
the staged tangent map recomputes the primal it differentiates, as a staged
``while_loop`` does.

Reverse-mode differentiation is not defined for the fold, so it is used only on
forward-only solve seams, which reverse mode cannot differentiate in any case;
[folds_enclosing_batches][diff_soup.inference.fitting._lane_map.folds_enclosing_batches]
states that condition for a solver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._jax_compat import Primitive, ad, batching, mlir
from diff_soup._jax_compat import partial_eval as pe


def folds_enclosing_batches(solver: Any) -> bool:
    """Whether a solve seam run by ``solver`` may fold enclosing batches into its lanes.

    The fold carries forward-mode derivatives only, so it is used where the
    solver's adjoint is forward-only: ``None`` (the solver default, a forward
    solve) or a non-composable forward strategy. A strategy that also supports
    reverse mode keeps the plain ``vmap``.
    """
    adjoint = getattr(solver, "adjoint", None)
    if adjoint is None:
        return True
    return bool(getattr(adjoint, "is_forward_only", False)) and not bool(
        getattr(adjoint, "composable", False)
    )


def lane_map(
    kernel: Callable,
    in_axes: Sequence[Optional[int]],
    *args: Any,
    fold: bool = True,
) -> Any:
    """Map ``kernel`` over a lane axis, folding enclosing ``vmap`` batches into it.

    Args:
        kernel: The per-lane function. Every traced value it reads must arrive
            through ``args``; closing over a traced value would hide it from
            the fold.
        in_axes: One entry per positional argument: ``0`` for an argument whose
            array leaves lead with the lane axis, ``None`` for one shared by
            every lane.
        *args: The kernel's arguments (pytrees). Non-array leaves are static.
        fold: ``False`` runs a plain ``jax.vmap``.

    Returns:
        ``jax.vmap(kernel, in_axes=in_axes)(*args)``.
    """
    in_axes = tuple(in_axes)
    if len(in_axes) != len(args):
        raise ValueError(
            f"lane_map needs one in_axes entry per argument; got {len(in_axes)} "
            f"entries for {len(args)} arguments."
        )
    if not fold:
        return jax.vmap(kernel, in_axes=in_axes)(*args)
    if all(axis is None for axis in in_axes):
        raise ValueError("lane_map needs at least one argument mapped over the lane axis.")

    dynamic, static = eqx.partition(args, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(dynamic)
    axes: list[Optional[int]] = []
    for arg_dynamic, axis in zip(dynamic, in_axes):
        axes.extend([axis] * len(jax.tree_util.tree_leaves(arg_dynamic)))
    lane_sizes = {int(leaf.shape[0]) for leaf, axis in zip(leaves, axes) if axis == 0}
    if len(lane_sizes) != 1:
        raise ValueError(
            f"lane_map lane-mapped arguments disagree on the lane count: {sorted(lane_sizes)}."
        )
    out_tree: list = []

    def flat_kernel(*flat_leaves):
        rebuilt = jax.tree_util.tree_unflatten(treedef, list(flat_leaves))
        out_leaves, tree = jax.tree_util.tree_flatten(kernel(*eqx.combine(rebuilt, static)))
        if not out_tree:
            out_tree.append(tree)
        return out_leaves

    outs = _lane_map_p.bind(*leaves, kernel=_Kernel(flat_kernel), axes=tuple(axes))
    return jax.tree_util.tree_unflatten(out_tree[0], outs)


class _Kernel:
    """A lane kernel carried as a primitive parameter (hashed by identity)."""

    __slots__ = ("fn", "__weakref__")

    def __init__(self, fn: Callable):
        """Hold ``fn``, the kernel on flat leaves."""
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


# ---------------------------------------------------------------------------
# Plain evaluations: the semantics every rule reproduces
# ---------------------------------------------------------------------------


def _is_differentiable(value: Any) -> bool:
    """Whether ``value`` has an inexact dtype, and so carries a tangent."""
    return jnp.issubdtype(jnp.result_type(value), jnp.inexact)


def _map_plain(*leaves, kernel, axes):
    """The lane map itself: ``jax.vmap(kernel, in_axes=axes)`` on flat leaves."""
    return list(jax.vmap(kernel, in_axes=axes)(*leaves))


def _jvp_plain(*leaves, kernel, axes, differentiable):
    """Primal outputs, then one tangent per inexact output with a leading direction axis."""
    n = len(axes)
    primals, directions = leaves[:n], leaves[n:]
    index = [i for i, d in enumerate(differentiable) if d]

    def lanes(*diff_primals):
        full = list(primals)
        for i, value in zip(index, diff_primals):
            full[i] = value
        return _map_plain(*full, kernel=kernel, axes=axes)

    diff_primals = tuple(primals[i] for i in index)

    def one_direction(*direction):
        out, d_out = jax.jvp(lanes, diff_primals, direction)
        return out, [t for o, t in zip(out, d_out) if _is_differentiable(o)]

    out, d_out = jax.vmap(one_direction, out_axes=(None, 0))(*directions)
    return list(out) + list(d_out)


def _abstract(fn: Callable) -> Callable:
    """Abstract evaluation of a plain evaluation: its output shapes and its effects.

    The plain evaluation is traced on shapes. The effects of the traced
    kernel become the equation's effects, as the effects of a ``lax.scan``
    body become the scan's: a kernel with an ordered effect (a host callback
    that keeps its order) makes the enclosing program thread that effect's
    token into the lane map's lowering. The linear-solver operations a
    kernel runs carry no effect: they are ordered by the data they pass on
    inside the kernel.
    """

    def rule(*avals, **params):
        shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
        traced = jax.make_jaxpr(lambda *a: fn(*a, **params))(*shapes)
        avals_out = [jax.core.ShapedArray(o.shape, o.dtype) for o in traced.out_avals]
        return avals_out, traced.effects

    return rule


def _n_primal_outputs(res: Sequence[Any]) -> int:
    """Number of primal outputs in a directional-derivative result.

    The result holds ``n`` primal outputs followed by one tangent per inexact
    primal output; ``n`` plus the inexact count among the first ``n`` grows
    strictly with ``n``, so exactly one ``n`` accounts for every value.
    """
    total = len(res)
    for n in range(total, 0, -1):
        if n + sum(_is_differentiable(o) for o in res[:n]) == total:
            return n
    raise ValueError("lane_map derivative returned an inconsistent output count.")


def _zero_tangent(value: Any) -> Any:
    """A symbolic zero tangent for ``value``."""
    return ad.Zero(jax.typeof(value).to_tangent_aval())


# ---------------------------------------------------------------------------
# Fold arithmetic
# ---------------------------------------------------------------------------


def _lane_count(leaves, axes, dims) -> int:
    """The lane count, read from a lane-mapped leaf (behind its batch axis if batched)."""
    for leaf, axis, dim in zip(leaves, axes, dims):
        if axis == 0:
            return int(leaf.shape[1] if dim is not None else leaf.shape[0])
    raise ValueError("lane_map lost its lane axis.")


def _fold_leaf(leaf, axis, batched: bool, n: int, lanes: int, lead: int = 0):
    """Return ``leaf`` with an ``(n, lanes)`` pair merged into one lane axis at ``lead``.

    A batched leaf carries the enclosing batch at axis 0; ``lead`` counts the
    leading axes (a tangent's direction axis) that stay in front.
    """
    if batched:
        leaf = jnp.moveaxis(leaf, 0, lead)
    else:
        index = (slice(None),) * lead + (None,)
        leaf = jnp.broadcast_to(leaf[index], leaf.shape[:lead] + (n,) + leaf.shape[lead:])
    if axis is None:
        shape = leaf.shape
        index = (slice(None),) * (lead + 1) + (None,)
        leaf = jnp.broadcast_to(leaf[index], shape[: lead + 1] + (lanes,) + shape[lead + 1 :])
    shape = leaf.shape
    return leaf.reshape(shape[:lead] + (n * lanes,) + shape[lead + 2 :])


def _unfold_leaf(leaf, n: int, lanes: int, lead: int = 0):
    """Split the merged lane axis at ``lead`` into ``(n, lanes)`` and move ``n`` first."""
    shape = leaf.shape
    leaf = leaf.reshape(shape[:lead] + (n, lanes) + shape[lead + 1 :])
    return jnp.moveaxis(leaf, lead, 0)


def _front(args, dims) -> list:
    """Move each batched argument's batch axis to the front."""
    return [a if d is None else jnp.moveaxis(a, d, 0) for a, d in zip(args, dims)]


def _batch_size(args, dims) -> int:
    """The enclosing batch size, read from the first batched argument."""
    return int(next(a.shape[0] for a, d in zip(args, dims) if d is not None))


# ---------------------------------------------------------------------------
# The lane-map primitive
# ---------------------------------------------------------------------------

_lane_map_p = Primitive("diffsoup_lane_map")
_lane_map_p.multiple_results = True
_lane_map_p.def_impl(_map_plain)
_lane_map_p.def_effectful_abstract_eval(_abstract(_map_plain))
# ``mlir.lower_fun`` lowers the traced kernel on the tokens of the ordered
# effects the abstract evaluation declares, if any (``ctx.tokens_in``), and
# makes the tokens the kernel leaves the map's output tokens
# (``ctx.set_tokens_out``); a kernel without effects has none to thread.
mlir.register_lowering(_lane_map_p, mlir.lower_fun(_map_plain, multiple_results=True))


def _lane_map_batching(args, dims, *, kernel, axes):
    """Fold the enclosing batch into the lane axis and map once."""
    args = _front(args, dims)
    n = _batch_size(args, dims)
    lanes = _lane_count(args, axes, dims)
    new_args, new_axes = [], []
    for leaf, axis, dim in zip(args, axes, dims):
        if dim is not None or axis == 0:
            new_args.append(_fold_leaf(leaf, axis, dim is not None, n, lanes))
            new_axes.append(0)
        else:
            new_args.append(leaf)
            new_axes.append(None)
    outs = _lane_map_p.bind(*new_args, kernel=kernel, axes=tuple(new_axes))
    return [_unfold_leaf(o, n, lanes) for o in outs], [0] * len(outs)


def _lane_map_jvp(primals, tangents, *, kernel, axes):
    """Forward rule: one fused primal-and-tangent evaluation of the lanes.

    Arguments with a symbolically zero tangent (data, masks, anything the
    differentiated input does not reach) are held constant, as ``jax.jvp`` of
    the plain map would.
    """
    differentiable = tuple(
        bool(_is_differentiable(p)) and type(t) is not ad.Zero for p, t in zip(primals, tangents)
    )
    if not any(differentiable):
        outs = _lane_map_p.bind(*primals, kernel=kernel, axes=axes)
        return outs, [_zero_tangent(o) for o in outs]
    directions = [t[None] for t, d in zip(tangents, differentiable) if d]
    res = _lane_jvp_p.bind(
        *primals, *directions, kernel=kernel, axes=axes, differentiable=differentiable
    )
    n_out = _n_primal_outputs(res)
    outs, d_outs = res[:n_out], iter(res[n_out:])
    tangents_out = [next(d_outs)[0] if _is_differentiable(o) else _zero_tangent(o) for o in outs]
    return outs, tangents_out


ad.primitive_jvps[_lane_map_p] = _lane_map_jvp
batching.primitive_batchers[_lane_map_p] = _lane_map_batching


# ---------------------------------------------------------------------------
# The directional-derivative primitive
# ---------------------------------------------------------------------------

_lane_jvp_p = Primitive("diffsoup_lane_map_jvp")
_lane_jvp_p.multiple_results = True
_lane_jvp_p.def_impl(_jvp_plain)
_lane_jvp_p.def_effectful_abstract_eval(_abstract(_jvp_plain))
mlir.register_lowering(_lane_jvp_p, mlir.lower_fun(_jvp_plain, multiple_results=True))


def _lane_jvp_batching(args, dims, *, kernel, axes, differentiable):
    """Fold a primal batch into the lanes, or a tangent-only batch into the directions."""
    n_p = len(axes)
    args = _front(args, dims)
    primals, directions = args[:n_p], args[n_p:]
    p_dims, d_dims = dims[:n_p], dims[n_p:]
    n = _batch_size(args, dims)
    params = dict(kernel=kernel, differentiable=differentiable)
    if any(d is not None for d in p_dims):
        lanes = _lane_count(primals, axes, p_dims)
        index = [i for i, d in enumerate(differentiable) if d]
        direction_of = {
            i: (leaf, dim is not None) for i, leaf, dim in zip(index, directions, d_dims)
        }
        new_primals, new_dirs, new_axes = [], [], []
        for i, (leaf, axis, dim) in enumerate(zip(primals, axes, p_dims)):
            direction = direction_of.get(i)
            fold_it = dim is not None or axis == 0 or (direction is not None and direction[1])
            new_primals.append(
                _fold_leaf(leaf, axis, dim is not None, n, lanes) if fold_it else leaf
            )
            new_axes.append(0 if fold_it else None)
            if direction is not None:
                d_leaf, d_batched = direction
                new_dirs.append(
                    _fold_leaf(d_leaf, axis, d_batched, n, lanes, lead=1) if fold_it else d_leaf
                )
        res = _lane_jvp_p.bind(*new_primals, *new_dirs, axes=tuple(new_axes), **params)
        n_out = _n_primal_outputs(res)
        outs = [_unfold_leaf(o, n, lanes) for o in res[:n_out]]
        d_outs = [_unfold_leaf(t, n, lanes, lead=1) for t in res[n_out:]]
        return outs + d_outs, [0] * len(res)
    # Only the directions are batched (``jax.jacfwd`` over this map): fold the
    # batch into the direction axis and keep the primal unbatched.
    n_dirs = next(
        int(leaf.shape[1] if dim is not None else leaf.shape[0])
        for leaf, dim in zip(directions, d_dims)
    )
    new_dirs = []
    for leaf, dim in zip(directions, d_dims):
        if dim is None:
            leaf = jnp.broadcast_to(leaf[None], (n,) + leaf.shape)
        new_dirs.append(leaf.reshape((n * n_dirs,) + leaf.shape[2:]))
    res = _lane_jvp_p.bind(*primals, *new_dirs, axes=axes, **params)
    n_out = _n_primal_outputs(res)
    d_outs = [t.reshape((n, n_dirs) + t.shape[1:]) for t in res[n_out:]]
    return list(res[:n_out]) + d_outs, [None] * n_out + [0] * len(d_outs)


def _lane_jvp_partial_eval(trace, *tracers, kernel, axes, differentiable):
    """Known primals, staged directions: run the primal map now, stage the tangents.

    ``jax.linearize`` needs the primal outputs known while the tangent map is
    recorded for later application; the staged map recomputes the primal it
    differentiates.
    """
    params = dict(kernel=kernel, axes=axes, differentiable=differentiable)
    n_p = len(axes)
    known = [t.pval.get_known() for t in tracers]
    if all(k is not None for k in known) or any(k is None for k in known[:n_p]):
        return trace.default_process_primitive(_lane_jvp_p, tracers, params)
    outs = _lane_map_p.bind(*known[:n_p], kernel=kernel, axes=axes)
    staged = trace.default_process_primitive(_lane_jvp_p, tracers, params)
    return [trace.new_const(o) for o in outs] + list(staged[len(outs) :])


def _lane_jvp_jvp(primals, tangents, *, kernel, axes, differentiable):
    """Second-order forward rule: differentiate the fused evaluation directly."""
    tangents = [ad.instantiate_zeros(t) for t in tangents]

    def fused(*leaves):
        return _jvp_plain(*leaves, kernel=kernel, axes=axes, differentiable=differentiable)

    out, d_out = jax.jvp(fused, tuple(primals), tuple(tangents))
    return list(out), list(d_out)


ad.primitive_jvps[_lane_jvp_p] = _lane_jvp_jvp
batching.primitive_batchers[_lane_jvp_p] = _lane_jvp_batching
pe.custom_partial_eval_rules[_lane_jvp_p] = _lane_jvp_partial_eval


__all__ = ["folds_enclosing_batches", "lane_map"]
