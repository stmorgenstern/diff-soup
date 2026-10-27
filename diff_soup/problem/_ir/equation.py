"""Equation IR nodes — single-write [Equation][diff_soup.problem._ir.equation.Equation] and multi-write [MultiEquation][diff_soup.problem._ir.equation.MultiEquation].

[Equation][diff_soup.problem._ir.equation.Equation] writes residual/RHS contributions for a single variable;
[MultiEquation][diff_soup.problem._ir.equation.MultiEquation] writes several variables from one closure, avoiding
redundant kernel invocations for tightly coupled fields.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import equinox as eqx

from .variable import GlobalVariable, Variable


class Equation(eqx.Module):
    """A residual/RHS contribution writing one variable.

    ``rhs_fn`` is intentionally NOT static.  When the user supplies an
    ``eqx.Module`` (e.g. a kinetics model with parametrised internal arrays),
    placing it in a non-static slot lets ``eqx.partition`` recurse into the
    module and surface its array leaves on the AD boundary.  Plain Python
    functions are still placed on the static side under any filter (functions
    are not arrays), so the static-closure path is unchanged.

    ``inputs`` accepts both ``Variable`` and ``GlobalVariable``.  Globals are
    threaded into the residual ``var_view`` from the constraint ``lam`` slot at
    residual time — they do not contribute to the sparse Jacobian topology
    since their effect on residual rows flows through the dense Schur ``B`` block
    via ``jax.jacrev``.
    """

    var: Variable = eqx.field(static=True)
    rhs_fn: Callable
    inputs: tuple[Union[Variable, GlobalVariable], ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        var: Variable,
        rhs_fn: Callable,
        inputs: tuple,
        name: str | None = None,
    ):
        if not all(isinstance(v, (Variable, GlobalVariable)) for v in inputs):
            raise TypeError("inputs must be Variable or GlobalVariable instances")
        self.var = var
        self.rhs_fn = rhs_fn
        self.inputs = tuple(inputs)
        self.name = name or var.name


class MultiEquation(eqx.Module):
    """Multi-write equation: one closure produces residuals for several variables.

    The closure ``rhs_fn(view, ctx, t)`` returns a dict ``{var_name: residual}``
    keyed on each output variable's name.  At residual time the IR runs the
    closure once and distributes each entry into the assembled residual at the
    matching variable's offset.  This avoids the 4× JAX-trace redundancy that
    would arise from registering four single-write equations all calling the
    same per-cell element kernel — the kernel runs once per residual call and
    XLA never sees the redundant copies.

    The intended use case is FEM element kernels that compute residuals for
    several coupled DOFs simultaneously (Allen-Cahn / mechanics / eigenstrain
    coupling) and similar bundled physics where running the kernel multiple
    times would explode trace memory.

    ``vars`` is the ordered tuple of output variables.  Their names must match
    the keys returned by ``rhs_fn``.  ``inputs`` follows the same convention as
    [Equation][diff_soup.problem._ir.equation.Equation].
    """

    vars: tuple[Variable, ...] = eqx.field(static=True)
    rhs_fn: Callable
    inputs: tuple[Union[Variable, GlobalVariable], ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vars: tuple,
        rhs_fn: Callable,
        inputs: tuple,
        name: str | None = None,
    ):
        if not vars:
            raise ValueError("MultiEquation requires at least one output variable")
        if not all(isinstance(v, Variable) for v in vars):
            raise TypeError("vars must be Variable instances (not GlobalVariable)")
        if not all(isinstance(v, (Variable, GlobalVariable)) for v in inputs):
            raise TypeError("inputs must be Variable or GlobalVariable instances")
        self.vars = tuple(vars)
        self.rhs_fn = rhs_fn
        self.inputs = tuple(inputs)
        self.name = name or "+".join(v.name for v in vars)
