"""IR nodes for differential-global equations and algebraic global constraints.

The marker [Constraint][diff_soup.problem._ir.constraint.Constraint] base carries no fields — each concrete subclass
redefines the full field list explicitly.  This sidesteps the dataclass
"non-default after default" inheritance error and keeps Equinox's ``__init__``
synthesis honest.  Lowering dispatches by ``isinstance`` on the marker base to
emit the Schur block.

The closed discriminated union: ``DifferentialGlobalEquation`` — the one
differential row — plus the algebraic constraints ``GalvanostaticConstraint``,
``DomainIntegralConstraint`` (alias ``IntegralConstraint``),
``ReduceResidualConstraint``, ``BoundaryIntegralConstraint``,
``OperationalControlConstraint``, and ``FluxTotalConstraint``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx

from diff_soup.domains.boundary_set import BoundarySet

from .variable import GlobalVariable, Variable


class Constraint(eqx.Module):
    """Base class of the records that close a global unknown with algebraic rows.

    Each record names the global it `closes`, and its `dim` property is that
    global's size. `spec.constraint` builds them from a recipe.
    """

    @property
    def dim(self) -> int:
        return int(self.closes.size)


class DifferentialGlobalEquation(Constraint):
    """The single ordinary equation writing a differential global.

    ``closes`` names the differential
    [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable] this equation
    writes; ``rhs_fn`` returns its time derivative (the same right-hand-side
    convention as field equations). The lowered row supplies the RHS at the
    global's lambda slot, and the global's unit mass turns the residual into
    ``d(global)/dt - rhs``. Unlike the algebraic constraint nodes, this row is
    differential: it participates in error control, interpolation, and
    consistent initialization as an ordinary unknown.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    rhs_fn: Callable
    reads: tuple[str, ...] = eqx.field(static=True, default=())


class GalvanostaticConstraint(Constraint):
    """Constraint record that drives a computed scalar to a target.

    The row is `current_fn(q, ctx, t) - target_fn(t) / integrand_scale`.

    Attributes:
        name: Label of the constraint.
        closes: Global unknown the row determines.
        current_fn: Function `(q, ctx, t)` returning the constrained scalar.
        target_fn: Target as a function of dimensional time.
        inputs: Declared inputs of the closure, left empty by the Recipe API.
        integrand_scale: Divisor of the target, reconciling a dimensional target with a non-dimensional quantity.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    current_fn: Callable
    target_fn: Callable
    inputs: tuple[Any, ...] = eqx.field(static=True, default=())
    integrand_scale: float = eqx.field(static=True, default=1.0)


class DomainIntegralConstraint(Constraint):
    """Constraint record that drives the domain integral of an integrand to a target.

    The integrand returns one value per node, weighted by the node's volume, or
    a scalar, weighted by the domain's total volume. Any other length raises
    `ValueError`.

    Attributes:
        name: Label of the constraint.
        closes: Global unknown the row determines.
        field: Field whose domain supplies the volume measure.
        integrand_fn: Function `(q, ctx, t)` returning the integrand.
        target_fn: Target as a function of dimensional time.
        inputs: Declared inputs of the closure, left empty by the Recipe API.
        integrand_scale: Divisor of the target, reconciling a dimensional target with a non-dimensional quantity.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    field: Variable
    integrand_fn: Callable
    target_fn: Callable
    inputs: tuple[Any, ...] = eqx.field(static=True, default=())
    integrand_scale: float = eqx.field(static=True, default=1.0)


# ``IntegralConstraint`` is a shorter alias of ``DomainIntegralConstraint``:
# the same class under both names, so ``isinstance(x, IntegralConstraint)``
# selects every volumetric integral constraint.
IntegralConstraint = DomainIntegralConstraint


class ReduceResidualConstraint(Constraint):
    """Constraint record that drives a reduction of a field's residual rows to a target.

    The row is `reduce_fn(f) - target_fn(t) / integrand_scale`, where `f` holds
    the field's assembled residual rows.

    Attributes:
        name: Label of the constraint.
        closes: Global unknown the row determines.
        field: Field whose residual rows are reduced.
        reduce_fn: Function of those rows returning the constrained scalar.
        target_fn: Target as a function of dimensional time.
        inputs: Declared inputs of the closure, left empty by the Recipe API.
        integrand_scale: Divisor of the target, reconciling a dimensional target with a non-dimensional quantity.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    field: Variable
    reduce_fn: Callable
    target_fn: Callable
    inputs: tuple[Any, ...] = eqx.field(static=True, default=())
    integrand_scale: float = eqx.field(static=True, default=1.0)


class BoundaryIntegralConstraint(Constraint):
    """Constraint record that drives the boundary integral of an integrand to a target.

    The integrand returns a scalar, one value per boundary node, per facet, or
    per mesh node, each weighted by the matching boundary measure. Any other
    length raises `ValueError`.

    Attributes:
        name: Label of the constraint.
        closes: Global unknown the row determines.
        field: Field whose domain holds the boundary.
        integrand_fn: Function `(q, ctx, t)` returning the integrand.
        target_fn: Target as a function of dimensional time.
        tag: Name of a single boundary tag, used when `selector` is `None`.
        selector: [`BoundarySet`][diff_soup.problem.BoundarySet] choosing the boundary.
        inputs: Declared inputs of the closure, left empty by the Recipe API.
        integrand_scale: Divisor of the target, reconciling a dimensional target with a non-dimensional quantity.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    field: Variable
    integrand_fn: Callable
    target_fn: Callable
    tag: Optional[str] = eqx.field(static=True, default=None)
    selector: Optional[BoundarySet] = eqx.field(static=True, default=None)
    inputs: tuple[Any, ...] = eqx.field(static=True, default=())
    integrand_scale: float = eqx.field(static=True, default=1.0)


class OperationalControlConstraint(Constraint):
    """Constraint record that closes a current unknown with one row of an operating table.

    The active row is `current - target`, `voltage - target`, or
    `power - target`, chosen at run time by the mode and target of an operating
    [`ControlState`][diff_soup.ControlState].

    Attributes:
        name: Label of the table.
        closes: Global current unknown the row determines.
        current_fn: Function `(q, ctx, t)` returning the current, or `None` when `field` supplies it.
        voltage_fn: Function `(q, ctx, t)` returning the voltage.
        modes: Rows of the table, a subset of `"current"`, `"voltage"`, `"power"`, indexed by the mode.
        seed_voltage_threshold: Voltage magnitude above which a power mode without a current seed starts from `target / V`.
        field: Field whose summed residual rows give the current, or `None`.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    current_fn: Optional[Callable]
    voltage_fn: Callable
    modes: tuple[str, ...] = eqx.field(static=True)
    seed_voltage_threshold: float = eqx.field(static=True, default=1e-6)
    field: Optional[Variable] = None


class FluxTotalConstraint(Constraint):
    """Pin the total of one or more boundary fluxes to a prescribed value.

    Each name in `from_bc` refers to a flux boundary condition authored
    with ``spec.bc.flux(..., name=...)``, in any of its forms.  The closure
    re-evaluates each named flux's own density on its own facet table,
    integrates ``Σ(q · face measure)`` over that boundary, and sums the
    integrals across the listed boundary conditions.  The residual is
    ``total − target_SI / integrand_scale`` and pins the global named by
    `closes`.

    The integration reuses the same flux density and the same facet table that
    drive the natural boundary condition, so the integral equals the total
    boundary flux sink by construction.
    """

    name: str = eqx.field(static=True)
    closes: GlobalVariable
    target_fn: Callable
    from_bc: tuple[str, ...] = eqx.field(static=True, default=())
    integrand_scale: float = eqx.field(static=True, default=1.0)
