"""Control IR nodes — time-dependent drivers of a prescribed or derived quantity.

By default ``fn`` is called at residual-evaluation time.  It is authored in
dimensional units: called with dimensional time and returning a dimensional
value, and the lowering applies the recipe's
[Scales][diff_soup.problem._ir.scales.Scales] on both sides.

Two kinds exist, split by where the controlled value comes from:

* [DirectControl][diff_soup.problem._ir.control.DirectControl] prescribes the
  value outright.  Nothing has to be solved for to realize it, so it declares
  no [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable] and no
  constraint.
* [DerivedControl][diff_soup.problem._ir.control.DerivedControl] drives a
  quantity the solve assembles.  ``binds_to`` names the free GlobalVariable
  that carries it; the constraint closing that global has its closing row
  replaced by ``global - fn(t) = 0``.

Both kinds are named protocols a solve may override through one payload
channel: an explicit ``control_state0`` mapping is resolved once before the
solve runs, replacing ``fn`` with a frozen value for the rest of the solve.

The marker [Control][diff_soup.problem._ir.control.Control] base carries no
fields; concrete subclasses list their own.  Lowering dispatches by
``isinstance``.

``binds_to`` stores the *name* (``str``) of the GlobalVariable the control
drives, NOT the live ``GlobalVariable`` object.  The live object carries a
``jax.Array`` ``init`` field; storing it directly in the static
``EquationSet.controls`` slot would trigger Equinox's "A JAX array is being set
as static!" warning.  Resolution to the live ``GlobalVariable`` happens at
lower-time in the provider.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx


class Control(eqx.Module):
    """Marker base; lowering dispatches by ``isinstance``."""


class DirectControl(Control):
    """A protocol fed straight in: no global variable, no constraint row."""

    name: str = eqx.field(static=True)
    fn: Callable


class DerivedControl(Control):
    """A protocol on an assembled quantity, carried by the global ``binds_to`` names."""

    name: str = eqx.field(static=True)
    fn: Callable
    binds_to: str  # name of the GlobalVariable the protocol drives
