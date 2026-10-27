"""Typed variable handles for the EquationSet IR.

Frozen ``eqx.Module`` nodes with host-side metadata in ``eqx.field(static=True)``
and structural equality keying the content hash.  Contains no lowering logic.

[Variable][diff_soup.problem._ir.variable.Variable] represents a state
variable block owned by a
[Component][diff_soup.problem._ir.Component].
[GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable] represents a
scalar or vector unknown in the bordered partition; its declared mass selects
either an ordinary differential-global equation or an algebraic
[Constraint][diff_soup.problem._ir.constraint.Constraint] that closes it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Variable(eqx.Module):
    """A typed handle for a state variable block."""

    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    mass: str = eqx.field(static=True, default="differential")

    def __init__(self, name: str, shape: tuple[int, ...], mass: str = "differential"):
        if mass not in {"differential", "algebraic"}:
            raise ValueError("mass must be 'differential' or 'algebraic'")
        self.name = str(name)
        self.shape = tuple(int(x) for x in shape)
        self.mass = mass

    @property
    def size(self) -> int:
        out = 1
        for n in self.shape:
            out *= int(n)
        return int(out)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.name == other.name and self.shape == other.shape and self.mass == other.mass

    def __lt__(self, other) -> bool:
        # Orderable by structural identity so a ``{Variable: value}`` dict can be
        # a JAX pytree node (``jax.vmap`` sorts dict keys when flattening — the
        # family gather path keys per-member gathered values by their macro
        # source Variable, and multiple gathers onto one family need a total
        # order to flatten).  Falls back to NotImplemented for non-Variables.
        if not isinstance(other, Variable):
            return NotImplemented
        return (self.name, self.shape, self.mass) < (other.name, other.shape, other.mass)

    def __hash__(self) -> int:
        return hash((self.name, self.shape, self.mass))


class GlobalVariable(eqx.Module):
    """A scalar or vector global unknown in the bordered partition.

    ``init`` may be a scalar or a one-dimensional array; ``size`` is its
    flat length, and the row that writes it carries that many rows — for an
    algebraic global, the
    [Constraint][diff_soup.problem._ir.constraint.Constraint] that closes it
    (``Constraint.dim == closes.size``).
    Globals are threaded into ``var_view`` from the ``lam`` slot at
    residual time so equation closures can read globals via ``view[g]``
    and the Schur ``B = df/dlam`` block becomes non-zero.

    ``mass`` is the global's structural role: ``"algebraic"`` globals are
    closed by a constraint or control row and carry zero mass;
    ``"differential"`` globals are written by exactly one ordinary equation
    and carry unit mass — they are ordinary differential unknowns that
    happen to live in the bordered partition, with derivatives, error
    control, and interpolation of their own.
    """

    name: str = eqx.field(static=True)
    init: jax.Array
    mass: str = eqx.field(static=True, default="algebraic")

    def __init__(self, name: str, init, mass: str = "algebraic"):
        if mass not in {"differential", "algebraic"}:
            raise ValueError("mass must be 'differential' or 'algebraic'")
        self.name = str(name)
        self.init = jnp.asarray(init)
        self.mass = mass

    @property
    def size(self) -> int:
        return int(self.init.size)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GlobalVariable):
            return NotImplemented
        # Equality by name + init shape + mass role; init values are not
        # hashable (jax.Array), so declaration identity is the structural key.
        return (
            self.name == other.name
            and self.init.shape == other.init.shape
            and self.mass == other.mass
        )

    def __hash__(self) -> int:
        return hash(("GlobalVariable", self.name, tuple(self.init.shape), self.mass))
