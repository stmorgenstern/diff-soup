"""Inspection of traced and compiled programs for the structural performance contracts."""

from __future__ import annotations

from typing import Any

import jax


def equations(closed_jaxpr) -> list[Any]:
    """Flatten the equations of a closed jaxpr and of every jaxpr nested in it."""
    found: list[Any] = []

    def visit(value):
        inner = getattr(value, "jaxpr", None)
        if inner is not None and inner is not value:
            visit(inner)
            return
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                found.append(equation)
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
