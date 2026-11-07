"""Inspection of traced and compiled programs for the structural performance contracts."""

from __future__ import annotations

from typing import Any

import jax


def _lane_kernel(equation) -> Any:
    """The per-lane jaxpr of a lane map, whose kernel is a Python callable, or ``None``.

    A folded lane map carries its kernel as a callable and its lane axes as ``axes``;
    the kernel is traced on one lane's shapes.
    """
    kernel, axes = equation.params.get("kernel"), equation.params.get("axes")
    if not callable(kernel) or not isinstance(axes, tuple):
        return None
    lane = [
        jax.ShapeDtypeStruct(v.aval.shape[1:] if axis == 0 else v.aval.shape, v.aval.dtype)
        for v, axis in zip(equation.invars, axes)
    ]
    return jax.make_jaxpr(kernel)(*lane)


def _replay_body(equation) -> Any:
    """The jaxpr of a checkpointed replay rule's primitive, traced from its implementation, or ``None``.

    The replay rule's primitives (``replay_*``) carry their bodies as Python functions; the
    forward pass with tangents or the backward loop they run counts as nested.
    """
    if not equation.primitive.name.startswith("replay_"):
        return None
    operands = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in equation.invars]
    return jax.make_jaxpr(lambda *xs: equation.primitive.impl(*xs, **equation.params))(*operands)


def equations(closed_jaxpr) -> list[Any]:
    """Flatten the equations of a closed jaxpr and of every jaxpr nested in it.

    A lane map's kernel is a callable, not a jaxpr; its per-lane jaxpr counts as nested, as does
    the body of a checkpointed replay rule's primitive.
    """
    found: list[Any] = []

    def visit(value):
        inner = getattr(value, "jaxpr", None)
        if inner is not None and inner is not value:
            visit(inner)
            return
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                found.append(equation)
                for kernel in (_lane_kernel(equation), _replay_body(equation)):
                    if kernel is not None:
                        visit(kernel)
                for param in equation.params.values():
                    if hasattr(param, "jaxpr") or hasattr(param, "eqns"):
                        visit(param)
                    elif isinstance(param, (tuple, list)):
                        for item in param:
                            if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                                visit(item)

    visit(closed_jaxpr)
    return found


def primitive_names(fn, *args, **kwargs) -> list[str]:
    """The primitive names of every equation ``fn`` traces to on ``args``, nested ones included."""
    return [eqn.primitive.name for eqn in equations(jax.make_jaxpr(fn)(*args, **kwargs))]


def stablehlo_text(fn, *args, **kwargs) -> str:
    """The StableHLO text of ``fn`` lowered on ``args``."""
    return jax.jit(fn).lower(*args, **kwargs).as_text()
