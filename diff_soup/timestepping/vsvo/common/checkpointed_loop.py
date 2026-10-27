"""The checkpointed replay loop, entered with JAX's own carry-type rule.

The driver's reverse-mode strategy runs equinox's checkpointed while loop
(``equinox.internal.while_loop(kind="checkpointed")``). That loop traces its
condition and body functions to jaxprs once, at entry, and on every later call
compares the carry it receives with the carry it was traced at: shape, dtype,
and the sharding JAX assigns to each leaf's type. It compares the shardings
literally.

JAX does not. A value's type carries the device mesh it lives on, and a fresh
constant -- the loop's own step counter, or a ``jnp.array(False)`` flag in the
solver carry -- is typed on the *empty* mesh, which JAX's type equivalence
treats as unspecified. Inside a ``shard_map`` body (a joint fitted across a
device mesh, or a sensitivity reduced across devices) every value derived from
the body's inputs is typed on that mesh, and the first arithmetic step on a
constant adopts it. So at entry the carry mixes empty-mesh and mesh-typed
leaves; in the backward pass, which replays from saved loop states, every leaf
is mesh-typed. JAX accepts the two as the same type; equinox's literal
comparison rejects the replay, so reverse mode through a sharded solve failed
at the first backward step.

This module runs the same loop, with the same checkpointing, replay, and
custom reverse rule, but converts the loop functions itself and checks each
call with JAX's rule: shapes and dtypes must match, and shardings are compared
only when both are specified on a mesh with an explicit axis. Nothing in
equinox is modified.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal._loop.checkpointed import (
    _checkpointed_while_loop,
    _stop_gradient,
    _stop_gradient_on_unperturbed,
)
from equinox.internal._loop.common import common_rewrite
from jax.sharding import AxisType

__all__ = ["checkpointed_while_loop"]


def _has_explicit_axis(sharding: Any) -> bool:
    """Whether a type-level sharding lives on a mesh with an explicit axis."""
    mesh = getattr(sharding, "mesh", None)
    axis_types = getattr(mesh, "axis_types", ()) or ()
    return any(axis_type == AxisType.Explicit for axis_type in axis_types)


def _same_type(called: jax.ShapeDtypeStruct, stored: jax.ShapeDtypeStruct) -> bool:
    """JAX's carry-type equivalence: shape and dtype, and sharding only where it is typed.

    JAX types a value on the empty mesh when its sharding is unspecified and
    reports that as ``None``; such a leaf matches any sharding. Two specified
    shardings are compared only when one of them lives on a mesh with an
    explicit axis, which is the rule ``jax.lax.while_loop`` applies to its own
    carry.
    """
    if called.shape != stored.shape or called.dtype != stored.dtype:
        return False
    if called.sharding is None or stored.sharding is None:
        return True
    if _has_explicit_axis(called.sharding) or _has_explicit_axis(stored.sharding):
        return called.sharding == stored.sharding
    return True


def _carry_structs(dynamic: Any) -> tuple[list[jax.ShapeDtypeStruct], Any]:
    """Flatten the array leaves of a carry to their shape, dtype, and type-level sharding."""
    structs = jax.eval_shape(lambda: dynamic)
    structs = jtu.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), structs
    )
    return jtu.tree_flatten(structs)


class _ConvertedLoopFunction(eqx.Module):
    """A loop function traced to a jaxpr, with its closed-over arrays as pytree leaves.

    The jaxpr and the captured constants are leaves rather than static fields so
    that the checkpointed loop's reverse rule can track perturbations on the
    captured arrays and JAX's jaxpr walks can see the nested program.
    """

    jaxpr: Any
    consts: Any
    in_structs: tuple[list[jax.ShapeDtypeStruct], Any] = eqx.field(static=True)
    in_static: tuple[list[Any], Any] = eqx.field(static=True)
    out_structs: tuple[list[jax.ShapeDtypeStruct], Any] = eqx.field(static=True)
    out_static: tuple[list[Any], Any] = eqx.field(static=True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the traced program on a carry of the same type as the traced one."""
        dynamic, static = eqx.partition((args, kwargs), eqx.is_array)
        called, called_def = _carry_structs(dynamic)
        stored, stored_def = self.in_structs
        if called_def != stored_def or not all(map(_same_type, called, stored)):
            raise ValueError(
                "The checkpointed loop was called with a carry whose type differs from "
                "the carry it was traced at; every leaf must keep its shape and dtype."
            )
        static_leaves, static_def = jtu.tree_flatten(static)
        if (static_leaves, static_def) != (list(self.in_static[0]), self.in_static[1]):
            raise ValueError(
                "The checkpointed loop was called with different non-array arguments "
                "from the ones it was traced with."
            )
        out_flat = jax.core.eval_jaxpr(self.jaxpr, self.consts, *jtu.tree_leaves(dynamic))
        out = jtu.tree_unflatten(self.out_structs[1], out_flat)
        return eqx.combine(out, jtu.tree_unflatten(self.out_static[1], self.out_static[0]))


def _convert(fn: Callable, *args: Any) -> _ConvertedLoopFunction:
    """Trace ``fn`` at ``args`` and capture its closed-over arrays as leaves."""
    dynamic, static = eqx.partition(
        (args, {}), lambda leaf: eqx.is_array(leaf) or isinstance(leaf, jax.ShapeDtypeStruct)
    )
    closed_jaxpr, out_dynamic, out_static = eqx.filter_make_jaxpr(fn)(*args)
    out_leaves, out_def = jtu.tree_flatten(out_dynamic)
    return _ConvertedLoopFunction(
        jaxpr=closed_jaxpr.jaxpr,
        consts=closed_jaxpr.consts,
        in_structs=_carry_structs(dynamic),
        in_static=jtu.tree_flatten(static),
        out_structs=(out_leaves, out_def),
        out_static=jtu.tree_flatten(out_static),
    )


def checkpointed_while_loop(
    cond_fun: Callable[[Any], Any],
    body_fun: Callable[[Any], Any],
    init_val: Any,
    *,
    max_steps: int | None = None,
    buffers: Callable[[Any], Any | Sequence[Any]] | None = None,
    checkpoints: int | None = None,
) -> Any:
    """Run equinox's checkpointed while loop on a carry that may be typed on a mesh.

    Same semantics and arguments as ``equinox.internal.while_loop(...,
    kind="checkpointed")``: a bounded while loop whose reverse rule replays
    segments from online checkpoints, so ``jax.grad`` works through the whole
    trajectory in bounded memory. ``max_steps`` bounds the iteration count;
    ``checkpoints`` sets the number of saved loop states (default about the
    square root of ``max_steps``); ``buffers`` selects carry leaves written
    once per step and never read back, which ride outside the checkpointed
    carry.

    The loop functions are traced once at ``init_val`` and every later call is
    checked against that carry with JAX's type rule, so the loop runs unchanged
    inside a ``shard_map`` body.

    Args:
        cond_fun: ``carry -> bool``; the loop runs while it returns ``True``.
        body_fun: ``carry -> carry`` with the same pytree structure and leaf types.
        init_val: The initial carry.
        max_steps: Upper bound on the number of iterations; required unless
            ``checkpoints`` is given.
        buffers: Optional selector of write-only buffer leaves of the carry.
        checkpoints: Number of loop states saved for the reverse replay.

    Returns:
        The final carry.
    """
    if checkpoints is None:
        if max_steps is None:
            raise ValueError("The checkpointed loop needs `max_steps` or `checkpoints`.")
        if max_steps == 1:
            checkpoints = 1
        else:
            checkpoints = math.floor(-1.5 + math.sqrt(2 * max_steps + 0.25)) + 1
    if checkpoints < 1:
        raise ValueError("The checkpointed loop needs at least one checkpoint.")
    init_val = jtu.tree_map(jnp.array, init_val)
    if max_steps == 0:
        return init_val
    if max_steps is not None:
        checkpoints = min(checkpoints, max_steps)
    cond_fun_, body_fun_, init_val_, buffers_ = common_rewrite(
        cond_fun, body_fun, init_val, max_steps, buffers, makes_false_steps=False
    )
    cond_fun_ = jtu.tree_map(_stop_gradient, _convert(cond_fun_, init_val_))
    body_fun_ = _convert(body_fun_, init_val_)
    final_val_ = _checkpointed_while_loop(
        (init_val_, body_fun_), cond_fun_, checkpoints, buffers_, max_steps
    )
    _, _, _, final_val = _stop_gradient_on_unperturbed(init_val_, final_val_, body_fun_)
    return final_val
