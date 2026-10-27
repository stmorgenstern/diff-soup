"""Compile a recipe's control declarations into their operational form.

A [Control][diff_soup.problem._ir.control.Control] is a named time-dependent
protocol.  This module turns the declarations into a compiled artifact at
lowering time:

* a [DerivedControl][diff_soup.problem._ir.control.DerivedControl] becomes a
  [CompiledControlBinding][diff_soup._lowering.control_bindings.CompiledControlBinding] naming the
  driven [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable]'s
  algebraic slice ``(lam_offset, dim)`` in the concatenated bordered-global
  vector (global declaration order — the same layout
  ``_build_global_layout_entries`` records);
* a [DirectControl][diff_soup.problem._ir.control.DirectControl] becomes a
  [CompiledDirectControl][diff_soup._lowering.control_bindings.CompiledDirectControl].
  It drives no global and writes no row: its value is prescribed outright, so
  the compiled entry carries only the name and the clock its ``fn`` reads;
* the whole set is a [ControlBindingSet][diff_soup._lowering.control_bindings.ControlBindingSet], whose
  [bound_values][diff_soup._lowering.control_bindings.ControlBindingSet.bound_values] evaluates every
  control at an internal time ``t`` — from the authored callables, or from an
  explicit ``{control_name: value}`` payload that overrides them.  Both kinds
  share that payload channel, so an override mapping names every declared
  control.

A driven global's value comes from its binding alone: the residual
replaces the closing row of the constraint that closes it with
``bound_value - lambda``, so the constraint's own integrand never contributes
and is not evaluated.

A control is authored in dimensional units, exactly like a Dirichlet ``value_fn``: its
``fn`` is called with dimensional time and returns a dimensional value, and the compiled entry
performs the conversions the recipe's
[Scales][diff_soup.problem._ir.scales.Scales]
declare.  ``Scales`` fixes ``t_SI = time_ref * t_internal`` and, for a driven
global, ``value_SI = state_offset + state_ref * value_internal``; so a control
evaluated at internal time ``t`` is called at ``time_ref * t``, and a bound
control's dimensional return value reaches the driven global's slice as
``(value_SI - state_offset) / state_ref``.  An explicit
``{control_name: value}`` payload names the same dimensional quantity the callable
would have returned.  Under the default identity scales both conversions are
the identity and the authored value is written through unchanged.

[bound_values][diff_soup._lowering.control_bindings.ControlBindingSet.bound_values]
returns the dimensional values — the quantity a model or event callback reads —
and
[internal_values][diff_soup._lowering.control_bindings.ControlBindingSet.internal_values]
returns them in the driven global's internal units, which is what the
closing row is written from.

Control callables are read off the runtime ``controls`` tuple — not captured
at compile time — so learnable array leaves inside a callable control reach
AD, matching the residual kernels' hybrid-cache contract.

This module lives under ``diff_soup/_lowering`` and may import
``diff_soup.problem._ir`` IR nodes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from diff_soup.problem._ir.equation_set import EquationSet

from .layout import _build_global_layout_entries


class CompiledControlBinding(NamedTuple):
    """One derived-quantity control bound to one algebraic global's slice.

    Attributes:
        control_name: The driving control's name (its callable is looked up by
            this name on the runtime ``controls`` tuple at evaluation time).
        global_name: The driven
            [GlobalVariable][diff_soup.problem._ir.variable.GlobalVariable]'s name.
        lam_offset: Flat offset of the global's slice in the concatenated
            bordered-global vector.
        dim: Size of that slice.
        time_ref: The recipe's dimensional time scale, so ``fn`` is called at
            ``time_ref * t_internal``.
        value_scale: The driven global's dimensional reference value
            (``Scales.state_ref[("", global_name)]``).
        value_offset: The driven global's dimensional offset
            (``Scales.state_offset[("", global_name)]``).
    """

    control_name: str
    global_name: str
    lam_offset: int
    dim: int
    time_ref: float = 1.0
    value_scale: float = 1.0
    value_offset: float = 0.0

    def si_time(self, t):
        """Dimensional time for an internal time ``t`` (identity when ``time_ref`` is 1)."""
        if self.time_ref == 1.0:
            return t
        return self.time_ref * t

    def to_internal(self, value_si):
        """The driven global's internal units for a dimensional control value."""
        if self.value_scale == 1.0 and self.value_offset == 0.0:
            return jnp.asarray(value_si)
        return (jnp.asarray(value_si) - self.value_offset) / self.value_scale


class CompiledDirectControl(NamedTuple):
    """One direct control: a prescribed value, driving no global and writing no row.

    Attributes:
        control_name: The control's name (its callable is looked up by this
            name on the runtime ``controls`` tuple at evaluation time, and it
            is the key an override payload must carry).
        time_ref: The recipe's dimensional time scale, so ``fn`` is called at
            ``time_ref * t_internal``.
        size: Flat size of the authored value, learned by evaluating the
            authored ``fn`` at dimensional time zero when bindings compile; an override
            payload must match it (or be a scalar).  ``None`` when the
            authored value could not be evaluated at compile time, in which
            case the payload passes unchecked.
    """

    control_name: str
    time_ref: float = 1.0
    size: object = None

    def si_time(self, t):
        """Dimensional time for an internal time ``t`` (identity when ``time_ref`` is 1)."""
        if self.time_ref == 1.0:
            return t
        return self.time_ref * t


class ControlBindingSet(NamedTuple):
    """The compiled control artifact for one lowered problem.

    Attributes:
        bindings: One [CompiledControlBinding][diff_soup._lowering.control_bindings.CompiledControlBinding] per
            derived-quantity control.
        lam_dim: Total bordered-global-vector size (for shape checks by consumers).
        direct: One [CompiledDirectControl][diff_soup._lowering.control_bindings.CompiledDirectControl] per
            direct control.  These drive no bordered-global slice; they share
            the override payload and nothing else.
    """

    bindings: tuple
    lam_dim: int
    direct: tuple = ()

    @property
    def control_names(self) -> tuple[str, ...]:
        """Every declared control name — the exact key set an override payload carries."""
        return tuple(b.control_name for b in self.bindings) + tuple(
            d.control_name for d in self.direct
        )

    @property
    def is_empty(self) -> bool:
        """Whether the recipe declared no controls at all."""
        return not self.bindings and not self.direct

    @staticmethod
    def _unwrap_state(payload):
        """Read a ``ControlState`` payload without importing the timestep layer."""
        values = getattr(payload, "values", None)
        if hasattr(payload, "epoch") and not callable(values):
            return values
        return payload

    def normalize_explicit(self, payload, *, dtype=None) -> dict:
        """Validate and shape one explicit ``{control_name: value}`` dimensional payload."""
        payload = self._unwrap_state(payload)
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Recipe control payload must be a mapping from each declared control "
                f"name to its value; expected {list(self.control_names)!r}."
            )
        required = set(self.control_names)
        supplied = set(payload)
        missing = sorted(required - supplied)
        unknown = sorted(supplied - required)
        if missing:
            raise ValueError(f"Recipe control payload is missing control(s) {missing!r}.")
        if unknown:
            raise ValueError(f"Recipe control payload has unknown control(s) {unknown!r}.")
        out = {}
        for binding in self.bindings:
            value = jnp.asarray(payload[binding.control_name]).reshape(-1)
            if int(value.size) not in (1, binding.dim):
                raise ValueError(
                    f"Recipe control {binding.control_name!r} has shape "
                    f"{jnp.asarray(payload[binding.control_name]).shape}, but global "
                    f"{binding.global_name!r} requires a scalar or {binding.dim} values."
                )
            if dtype is not None:
                value = value.astype(dtype)
            out[binding.control_name] = jnp.broadcast_to(value, (binding.dim,))
        for entry in self.direct:
            # A direct control drives no algebraic-global slice; its width is fixed
            # by the authored value the compile step evaluated.
            value = jnp.asarray(payload[entry.control_name]).reshape(-1)
            if entry.size is not None and int(value.size) not in (1, int(entry.size)):
                raise ValueError(
                    f"Recipe control {entry.control_name!r} has shape "
                    f"{jnp.asarray(payload[entry.control_name]).shape}, but its "
                    f"authored value carries {int(entry.size)} value(s); supply "
                    f"a scalar or exactly {int(entry.size)} value(s)."
                )
            out[entry.control_name] = value if dtype is None else value.astype(dtype)
        return out

    def bound_values(self, controls, t, payload=None, *, dtype=None) -> dict:
        """Resolve dimensional control values from authored callables or an explicit mapping.

        Covers every declared control, direct and derived alike.  ``t`` is
        internal time; each authored callable is handed the dimensional time it
        corresponds to.  The returned values are dimensional — the quantity a model or
        an event callback reads.  Use
        [internal_values][diff_soup._lowering.control_bindings.ControlBindingSet.internal_values]
        for a driven algebraic-global's own units.
        """
        payload = self._unwrap_state(payload)
        if payload is None:
            by_name = self._fn_by_name(controls)
            out = {
                b.control_name: self._bound_value(by_name[b.control_name], t, b, dtype=dtype)
                for b in self.bindings
            }
            for entry in self.direct:
                val = jnp.asarray(by_name[entry.control_name](entry.si_time(t))).reshape(-1)
                out[entry.control_name] = val if dtype is None else val.astype(dtype)
            return out
        return self.normalize_explicit(payload, dtype=dtype)

    def internal_values(self, controls, t, payload=None, *, dtype=None) -> dict:
        """Derived-control values in the driven algebraic-globals' internal units.

        The dimensional values from
        [bound_values][diff_soup._lowering.control_bindings.ControlBindingSet.bound_values]
        rescaled by each driven global's ``(state_ref, state_offset)``; the
        closing row is written from these.  Direct controls drive no algebraic-global
        and so do not appear.
        """
        values = self.bound_values(controls, t, payload, dtype=dtype)
        out = {}
        for b in self.bindings:
            val = b.to_internal(values[b.control_name])
            out[b.control_name] = val if dtype is None else val.astype(dtype)
        return out

    def _fn_by_name(self, controls) -> dict:
        by_name = {}
        for ctl in controls:
            by_name[ctl.name] = ctl.fn
        missing = [name for name in self.control_names if name not in by_name]
        if missing:
            raise ValueError(
                f"control binding set: the runtime controls tuple has no entry "
                f"named {missing!r}; it must carry every declared control "
                f"(available: {sorted(by_name)!r})."
            )
        return by_name

    def _bound_value(self, fn, t, b: CompiledControlBinding, dtype=None) -> jax.Array:
        val = jnp.asarray(fn(b.si_time(t))).reshape(-1)
        if int(val.size) not in (1, b.dim):
            raise ValueError(
                f"control {b.control_name!r} bound to global {b.global_name!r} "
                f"returned {int(val.size)} value(s), but the driven algebraic-global "
                f"slice has size {b.dim}; the control fn must return a scalar "
                f"or exactly {b.dim} value(s)."
            )
        if dtype is not None:
            val = val.astype(dtype)
        return jnp.broadcast_to(val, (b.dim,))


def _authored_value_size(fn) -> object:
    """Flat size of a direct control's authored value, from ``fn`` at dimensional zero.

    The compile step evaluates the authored callable once at dimensional time ``0.0``
    to fix the width an override payload must match.  A callable that cannot
    be evaluated there (a protocol undefined at zero, say) leaves the width
    unchecked rather than failing the build.
    """
    try:
        return int(jnp.asarray(fn(jnp.zeros(()))).size)
    except Exception:
        return None


def compile_control_bindings(eqset: EquationSet) -> ControlBindingSet:
    """Compile ``eqset``'s control declarations into a [ControlBindingSet][diff_soup._lowering.control_bindings.ControlBindingSet].

    Resolves each derived-quantity control's target global against the
    bordered-global layout (one slice per global, declaration order), and
    records the ``eqset.scales`` factors that carry the control between dimensional and
    internal units: the time scale its ``fn`` is called on, and the driven
    global's dimensional reference and offset.  Direct controls drive no global, so
    they compile to a name and that same time scale.

    Raises:
        ValueError: when a control binds to a global no constraint closes (a
            derived-quantity control drives the algebraic-global slice of the
            constraint its global closes — declare a constraint with
            ``closes=<global>``, or declare the protocol with
            ``spec.control.direct`` if nothing has to be solved for to realize
            it), or when two controls bind the same global (each driven
            algebraic-global slice takes exactly one driver).
    """
    lam_entries = _build_global_layout_entries(eqset)
    slice_by_global = {name: (int(off), int(size)) for name, off, size in lam_entries}
    lam_dim = int(sum(size for _name, _off, size in lam_entries))
    scales = eqset.scales
    time_ref = float(scales.time_ref)

    bindings = []
    direct = []
    driver_by_global: dict = {}
    for ctl in eqset.controls:
        bound = getattr(ctl, "binds_to", None)
        if bound is None:
            direct.append(
                CompiledDirectControl(
                    control_name=str(ctl.name),
                    time_ref=time_ref,
                    size=_authored_value_size(ctl.fn),
                )
            )
            continue
        entry = slice_by_global.get(bound)
        if entry is None:
            closed = sorted(slice_by_global)
            raise ValueError(
                f"control {ctl.name!r} binds to global {bound!r}, but no "
                f"constraint closes that global. A derived-quantity control "
                f"drives the algebraic-global slice of the constraint its "
                f"global closes — declare a constraint with closes={bound!r}, "
                f"or declare {ctl.name!r} with spec.control.direct if its "
                f"value is prescribed outright. Globals closed by "
                f"constraints: {closed!r}."
            )
        other = driver_by_global.get(bound)
        if other is not None:
            raise ValueError(
                f"controls {other!r} and {ctl.name!r} both bind to global "
                f"{bound!r}; each driven algebraic-global slice takes exactly one "
                f"control — remove one of the bindings."
            )
        driver_by_global[bound] = ctl.name
        off, size = entry
        bindings.append(
            CompiledControlBinding(
                control_name=str(ctl.name),
                global_name=str(bound),
                lam_offset=off,
                dim=size,
                time_ref=time_ref,
                value_scale=float(scales.state_ref.get(("", bound), 1.0)),
                value_offset=float(scales.state_offset.get(("", bound), 0.0)),
            )
        )
    return ControlBindingSet(bindings=tuple(bindings), lam_dim=lam_dim, direct=tuple(direct))
