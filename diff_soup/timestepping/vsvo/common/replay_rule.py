"""The derivative rule of the checkpointed replay loop, for every composition of two passes.

Equinox's checkpointed loop is a `custom_vjp`: its forward pass stores loop states and its
backward pass replays from them. JAX can neither push a tangent through a `custom_vjp` nor
transpose the replay, which is itself a loop, so a gradient of a gradient through it fails.
[replayed_while_loop][] wraps it in a `custom_jvp` whose tangent is the product with the loop's
Jacobian `J`, bound as a primitive with its own transpose:

- `J v` evaluates as a forward pass with tangents through the loop, and transposes to
- `J^T c`, the checkpointed replay from the stored states, whose derivative is
- `H dz + J^T dc`, with `H` the Hessian of `c . final` in the loop's inputs, which evaluates as
  a forward pass with tangents through the replay (forward over reverse) and transposes in `dz`
  to itself, since `H` is symmetric;
- `d(J v)[dz]`, the derivative of the first product, evaluates as a forward pass with two
  tangents and transposes to `H v`.

A gradient runs the checkpointed forward pass and one replay, as equinox's rule alone does, to
the bit. Reverse over reverse transposes the third product, so a Hessian-vector product by
`jax.grad` of `jax.grad` costs what forward over reverse does. Forward over reverse runs the
forward pass, the first product and the third as separate passes, so it costs about twice what
it does under equinox's rule (measured on batjax's small cell), which is why
`CheckpointAdjoint(reverse_over_reverse=True)` selects this rule and the default does not. A
third derivative raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal._loop.checkpointed import _checkpointed_while_loop
from equinox.internal._primitive import _vprim_p
from jax import lax
from jax.interpreters import ad, batching
from jaxtyping import PyTree

__all__ = ["replayed_while_loop"]

_THIRD = (
    "A third derivative through a checkpointed solve is not supported: the solve's replay "
    "rule gives first and second derivatives only."
)


class _Loop(eqx.Module):
    """The loop's inputs, `(init, body)`, and its fixed configuration.

    Attributes:
        vjp_arg: `(init, body)`: the first carry and the converted body with its constants.
        cond_fun: The converted condition, gradient-dead.
        checkpoints: The number of stored loop states.
        buffers: The selector of the carry's write-only buffers.
        max_steps: The bound on the number of steps.
    """

    vjp_arg: PyTree
    cond_fun: Any
    checkpoints: int = eqx.field(static=True)
    buffers: Callable = eqx.field(static=True)
    max_steps: int | None = eqx.field(static=True)

    def take(self, mask: tuple[bool, ...]) -> list:
        """The input leaves where `mask` holds."""
        return [leaf for leaf, m in zip(jtu.tree_leaves(self.vjp_arg), mask) if m]

    def put(self, mask: tuple[bool, ...], values: list) -> _Loop:
        """The loop with the input leaves where `mask` holds replaced by `values`."""
        leaves, treedef = jtu.tree_flatten(self.vjp_arg)
        given = iter(values)
        leaves = [next(given) if m else leaf for leaf, m in zip(leaves, mask)]
        return eqx.tree_at(lambda loop: loop.vjp_arg, self, jtu.tree_unflatten(treedef, leaves))

    def checkpointed(self) -> PyTree:
        """The final carry by equinox's checkpointed loop (plain when not differentiated)."""
        return _checkpointed_while_loop(
            self.vjp_arg, self.cond_fun, self.checkpoints, self.buffers, self.max_steps
        )

    def plain(self) -> PyTree:
        """The final carry by a `lax.while_loop`, which JAX pushes tangents through."""
        init, body = self.vjp_arg
        return lax.while_loop(lambda carry: self.cond_fun(carry), lambda carry: body(carry), init)


def _split(final: PyTree) -> tuple[PyTree, PyTree]:
    """The inexact leaves of `final`, as a pytree, and the rest of it."""
    return eqx.partition(final, eqx.is_inexact_array)


def _pullback(loop: _Loop, mask: tuple[bool, ...]):
    """`outputs, pullback, rest` of the checkpointed loop in the leaves where `mask` holds."""

    def run(values):
        return _split(loop.put(mask, values).checkpointed())

    return jax.vjp(run, loop.take(mask), has_aux=True)


def _zeros(values: PyTree, like: PyTree) -> PyTree:
    """`values` with a zero array shaped as `like`'s leaf in place of each `None`."""
    return jtu.tree_map(
        lambda v, x: jnp.zeros(x.shape, x.dtype) if v is None and x is not None else v,
        values,
        like,
        is_leaf=lambda v: v is None,
    )


def _structs(tree: PyTree) -> PyTree:
    """The shapes and dtypes of `tree`'s leaves."""
    return jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)


def _avals(tree: PyTree) -> PyTree:
    """The abstract values of `tree`'s leaves, as a primitive's abstract evaluation returns them."""
    return jtu.tree_map(lambda x: jax.core.ShapedArray(x.shape, x.dtype), tree)


def _given(loop: _Loop, t_vjp_arg: PyTree) -> tuple[tuple[bool, ...], list]:
    """Where the loop's input leaves have a tangent, and those tangents, in leaf order."""
    leaves, treedef = jtu.tree_flatten(loop.vjp_arg)
    flat = [None] * len(leaves) if t_vjp_arg is None else treedef.flatten_up_to(t_vjp_arg)
    return tuple(t is not None for t in flat), [t for t in flat if t is not None]


def _none(tree: PyTree) -> PyTree:
    """`tree` with `None` in every leaf: the cotangent of an input a transpose holds fixed."""
    return jtu.tree_map(lambda _: None, tree, is_leaf=_is_undefined)


def _is_undefined(x: Any) -> bool:
    """Whether `x` is an input a transpose rule is asked the cotangent of."""
    return isinstance(x, ad.UndefinedPrimal)


def _undefined(tree: PyTree) -> bool:
    """Whether a transpose rule is asked for the cotangent of `tree`."""
    return any(map(_is_undefined, jtu.tree_leaves(tree, is_leaf=_is_undefined)))


def _asked(cotangent: PyTree, operand: PyTree) -> PyTree:
    """`cotangent` where `operand`'s leaf is asked for (undefined), `None` where it is defined.

    Under `vmap` a zero tangent arrives materialized, so a linear operand can be defined in part.
    """
    return jtu.tree_map(
        lambda x, c: c if _is_undefined(x) else None, operand, cotangent, is_leaf=_is_undefined
    )


def _check_dead(tree: PyTree) -> None:
    """Raise where a residual that the rule holds gradient-dead carries a tangent."""
    if any(t is not None for t in jtu.tree_leaves(tree, is_leaf=lambda x: x is None)):
        raise NotImplementedError(_THIRD)


# `J v`: a forward pass with tangents through the plain loop.


def _tangent_impl(loop, mask, pullback, structs, v):
    """`J v`."""
    del pullback, structs

    def outputs(values):
        return _split(loop.put(mask, values).plain())[0]

    return jax.jvp(outputs, (loop.take(mask),), (v,))[1]


def _tangent_abstract(loop, mask, pullback, structs, v):
    """The outputs' abstract values."""
    del loop, mask, pullback, v
    return _avals(structs)


def _tangent_jvp(primals, tangents):
    """`J dv`, plus `d(J v)[dz]` where the loop's inputs have a tangent `dz`."""
    loop, mask, pullback, structs, v = primals
    t_loop, _, t_pullback, _, t_v = tangents
    _check_dead(t_pullback)
    out = _bind(_tangent_p, loop, mask, pullback, structs, v)
    tangent = _bind(_tangent_p, loop, mask, pullback, structs, _zeros(t_v, v))
    q, dz = _given(loop, None if t_loop is None else t_loop.vjp_arg)
    if any(q):
        curvature = _bind(_curvature_p, loop, mask, q, structs, v, dz)
        tangent = jtu.tree_map(jnp.add, tangent, curvature)
    return out, tangent


def _tangent_transpose(inputs, cts):
    """`J^T c`."""
    loop, mask, pullback, structs, _ = inputs
    loop, mask, pullback, structs, v = inputs
    c = _bind(_cotangent_p, loop, mask, pullback, structs, _zeros(cts, structs))
    return (*_none((loop, mask, pullback, structs)), _asked(c, v))


# `J^T c`: the checkpointed replay, from the stored states when they belong to `mask`.


def _cotangent_impl(loop, mask, pullback, structs, c):
    """`J^T c`, by the replay from `pullback`'s stored states, or a fresh one."""
    del structs
    if pullback is None:
        _, pullback, _ = _pullback(loop, mask)
    return pullback(c)[0]


def _cotangent_abstract(loop, mask, pullback, structs, c):
    """The abstract values of the inputs where `mask` holds."""
    del pullback, structs, c
    return _avals(loop.take(mask))


def _cotangent_jvp(primals, tangents):
    """`J^T dc`, plus `H dz` where the loop's inputs have a tangent `dz`."""
    loop, mask, pullback, structs, c = primals
    t_loop, _, t_pullback, _, t_c = tangents
    _check_dead(t_pullback)
    out = _bind(_cotangent_p, loop, mask, pullback, structs, c)
    q, dz = _given(loop, None if t_loop is None else t_loop.vjp_arg)
    dc = _zeros(t_c, structs)
    if not any(q):
        return out, _bind(_cotangent_p, loop, mask, pullback, structs, dc)
    return out, _bind(_second_p, loop, mask, q, structs, c, dz, dc)


def _cotangent_transpose(inputs, cts):
    """`J w`."""
    loop, mask, pullback, structs, _ = inputs
    loop, mask, pullback, structs, c = inputs
    v = _bind(_tangent_p, loop, mask, pullback, structs, _zeros(cts, loop.take(mask)))
    return (*_none((loop, mask, pullback, structs)), _asked(v, c))


# `H dz + J^T dc`: a forward pass with tangents through the checkpointed loop and its replay.


def _second_impl(loop, mask, q, structs, c, dz, dc):
    """`H dz + J^T dc`."""
    del structs

    def replay(values, k):
        return _pullback(loop.put(q, values), mask)[1](k)[0]

    return jax.jvp(replay, (loop.take(q), c), (dz, dc))[1]


def _second_abstract(loop, mask, q, structs, c, dz, dc):
    """The abstract values of the inputs where `mask` holds."""
    del q, structs, c, dz, dc
    return _avals(loop.take(mask))


def _second_jvp(primals, tangents):
    """A third derivative."""
    del primals, tangents
    raise NotImplementedError(_THIRD)


def _second_transpose(inputs, cts):
    """`(H w, J w)`, by symmetry of `H`."""
    loop, mask, q, structs, c, dz, dc = inputs
    w = _zeros(cts, loop.take(mask))
    h_w, j_w = _bind(_adjoint_p, loop, mask, q, structs, c, w)
    return (*_none((loop, mask, q, structs, c)), _asked(h_w, dz), _asked(j_w, dc))


# `(H^T w, J w)`, the transpose of the above: a forward pass with tangents through the loop and
# its replay in the leaves where `q` holds.


def _adjoint_impl(loop, mask, q, structs, c, w):
    """`(H^T w, J w)`, as one forward pass with tangents."""
    del structs

    def outputs_and_replay(values):
        outputs, pullback, _ = _pullback(loop.put(mask, values), q)
        return outputs, pullback(c)[0]

    _, (j_w, h_w) = jax.jvp(outputs_and_replay, (loop.take(mask),), (w,))
    return h_w, j_w


def _adjoint_abstract(loop, mask, q, structs, c, w):
    """The abstract values of the inputs where `q` holds, and of the outputs."""
    del mask, c, w
    return _avals(loop.take(q)), _avals(structs)


def _adjoint_jvp(primals, tangents):
    """A third derivative."""
    del primals, tangents
    raise NotImplementedError(_THIRD)


def _adjoint_transpose(inputs, cts):
    """`H dz + J^T dc`."""
    loop, mask, q, structs, c, _ = inputs
    dz, dc = cts
    w = _bind(_second_p, loop, mask, q, structs, c, _zeros(dz, loop.take(q)), _zeros(dc, structs))
    return (*_none((loop, mask, q, structs, c)), _asked(w, inputs[-1]))


# `d(J v)[dz]`: a forward pass with two tangents through the plain loop.


def _curvature_impl(loop, mask, q, structs, v, dz):
    """The second derivative of the outputs along `v` (in `mask`) and `dz` (in `q`)."""
    del structs

    def tangent(values):
        moved = loop.put(q, values)

        def outputs(inner):
            return _split(moved.put(mask, inner).plain())[0]

        return jax.jvp(outputs, (moved.take(mask),), (v,))[1]

    return jax.jvp(tangent, (loop.take(q),), (dz,))[1]


def _curvature_abstract(loop, mask, q, structs, v, dz):
    """The outputs' abstract values."""
    del loop, mask, q, v, dz
    return _avals(structs)


def _curvature_jvp(primals, tangents):
    """A third derivative."""
    del primals, tangents
    raise NotImplementedError(_THIRD)


def _curvature_transpose(inputs, cts):
    """`H dz` with `H` the Hessian of `c . final`, `c` the cotangent: the transpose in `dz`."""
    loop, mask, q, structs, v, dz = inputs
    if _undefined(v):
        raise NotImplementedError(_THIRD)
    h_v, _ = _bind(_adjoint_p, loop, mask, q, structs, _zeros(cts, structs), v)
    return (*_none((loop, mask, q, structs, v)), _asked(h_v, dz))


def _bind(prim, *args):
    """Bind one of this module's primitives on its pytree arguments."""
    return eqxi.filter_primitive_bind(prim, *args)


def _lift(operands) -> list:
    """`operands` with each constant that JAX forwarded as a Python literal an array again.

    Equinox's wrappers count only arrays among a primitive's operands, and JAX can hand a rule a
    residual it forwarded as a literal.
    """
    return [x if _is_undefined(x) or eqx.is_array(x) else jnp.asarray(x) for x in operands]


def _lifted_jvp(jvp: Callable) -> Callable:
    """`jvp` on lifted primals."""

    def rule(primals, tangents, **params):
        return jvp(_lift(primals), tangents, **params)

    return rule


def _lifted_transpose(transpose: Callable) -> Callable:
    """`transpose` on lifted operands."""

    def rule(cts, *operands, **params):
        return transpose(cts, *_lift(operands), **params)

    return rule


def _primitive(name, impl, abstract, jvp, transpose):
    """A primitive batched by `vmap` of its rules, on pytree arguments, every operand batched.

    Equinox's batching transposes a lane of the rule at a time and cannot return an unbatched
    cotangent for an unbatched operand of batched lanes (a parameter shared by the lanes), so
    every operand is broadcast to the batch first: the broadcast's own transpose sums the lanes.
    """
    prim = eqxi.create_vprim(
        name,
        eqxi.filter_primitive_def(impl),
        eqxi.filter_primitive_def(abstract),
        _lifted_jvp(eqxi.filter_primitive_jvp(jvp)),
        _lifted_transpose(eqxi.filter_primitive_transpose(materialise_zeros=False)(transpose)),
    )

    def batch(axis_size, axis_name, trace_type, operands, axes, **params):
        """Bind `prim` once on operands that all carry the batch on their first axis."""
        del trace_type
        if all(axis is None for axis in axes):
            out = prim.bind(*operands, **params)
            return out, [None] * len(out)
        operands = [
            jnp.broadcast_to(x, (axis_size, *jnp.shape(x)))
            if axis is None
            else jnp.moveaxis(x, axis, 0)
            for x, axis in zip(operands, axes)
        ]
        out = _vprim_p.bind(
            *operands,
            prim=prim,
            __axis_size=axis_size,
            __axis_name=axis_name,
            __batch_axes=(0,) * len(operands),
            params=tuple(params.items()),
        )
        return out, [0] * len(out)

    batching.axis_primitive_batchers[prim] = batch
    return prim


_tangent_p = _primitive(
    "replay_tangent", _tangent_impl, _tangent_abstract, _tangent_jvp, _tangent_transpose
)
_cotangent_p = _primitive(
    "replay_cotangent", _cotangent_impl, _cotangent_abstract, _cotangent_jvp, _cotangent_transpose
)
_second_p = _primitive(
    "replay_second", _second_impl, _second_abstract, _second_jvp, _second_transpose
)
_adjoint_p = _primitive(
    "replay_adjoint", _adjoint_impl, _adjoint_abstract, _adjoint_jvp, _adjoint_transpose
)
_curvature_p = _primitive(
    "replay_curvature", _curvature_impl, _curvature_abstract, _curvature_jvp, _curvature_transpose
)


@eqx.filter_custom_jvp
def _attach(outputs, loop, pullback, *, mask, structs):
    """`outputs`, whose tangent in the loop's inputs is `J dz`."""
    del loop, pullback, mask, structs
    return outputs


@_attach.def_jvp
def _attach_jvp(primals, tangents, *, mask, structs):
    """`J dz`, from the stored states when `dz` is in the leaves they belong to."""
    outputs, loop, pullback = primals
    t_outputs, t_loop, t_pullback = tangents
    _check_dead((t_outputs, t_pullback))
    q, dz = _given(loop, None if t_loop is None else t_loop.vjp_arg)
    held = pullback if q == mask else None
    return outputs, _bind(_tangent_p, loop, q, held, structs, dz)


@eqx.filter_custom_jvp
def _replayed(vjp_arg, cond_fun, *, checkpoints, buffers, max_steps):
    """Equinox's checkpointed loop, with this module's rule."""
    return _Loop(vjp_arg, cond_fun, checkpoints, buffers, max_steps).checkpointed()


@_replayed.def_jvp
def _replayed_jvp(primals, tangents, *, checkpoints, buffers, max_steps):
    """The checkpointed forward pass, its outputs' tangent `J dz`, and the rest unperturbed."""
    vjp_arg, cond_fun = primals
    t_vjp_arg, _ = tangents
    loop = _Loop(vjp_arg, cond_fun, checkpoints, buffers, max_steps)
    mask, dz = _given(loop, t_vjp_arg)
    dead = jtu.tree_map(lambda x: lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, loop)
    outputs, pullback, rest = _pullback(dead, mask)
    structs = _structs(outputs)
    outputs = _attach(outputs, loop, pullback, mask=mask, structs=structs)
    t_outputs = _bind(_tangent_p, loop, mask, pullback, structs, dz)
    return eqx.combine(outputs, rest), t_outputs


def replayed_while_loop(
    vjp_arg: PyTree, cond_fun: Any, checkpoints: int, buffers: Callable, max_steps: int | None
) -> PyTree:
    """Equinox's checkpointed loop on `vjp_arg = (init, body)`, differentiable twice in any order.

    Args:
        vjp_arg: `(init, body)`, the first carry and the converted body.
        cond_fun: The converted condition, gradient-dead.
        checkpoints: The number of stored loop states.
        buffers: The selector of the carry's write-only buffers.
        max_steps: The bound on the number of steps.

    Returns:
        The final carry.
    """
    return _replayed(
        vjp_arg, cond_fun, checkpoints=checkpoints, buffers=buffers, max_steps=max_steps
    )
