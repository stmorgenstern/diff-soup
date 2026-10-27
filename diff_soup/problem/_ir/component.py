"""Component IR node.

A frozen ``eqx.Module`` collecting a named domain's variables, equations, and
boundary conditions.  Validated at construction: every variable must have
exactly one writing equation; every BC must reference a declared variable.
[structural_key][] produces a hashable fingerprint used to group compatible
components for ``vmap`` in lowering.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Optional

import equinox as eqx
import numpy as np

from diff_soup._errors import ConfigurationError

from .bc import BoundaryCondition, DirichletBC, FluxBC
from .equation import Equation, MultiEquation
from .variable import Variable


def validate_operator_factory(factory) -> None:
    """Reject operator factories whose structural identity is not stable.

    An operator factory selects the spatial operator bank a component or family
    is lowered against.  It is stored as opaque structural configuration and
    participates in content identity by its qualified callable/class name, so it
    must have a *stable* name across reconstruction.  The accepted forms are:

    * ``None`` — the built-in bank is selected;
    * a top-level operator-bank class (a ``type`` whose qualified name has no
      ``<locals>`` enclosing scope);
    * a top-level named callable ``domain -> bank`` (a function whose qualified
      name names neither a lambda nor a local enclosing scope).

    Lambdas, closures, nested classes, and [partial][] are
    rejected: their identity is not stable enough for the content-hash and
    structural-key contract.  This validates only structural identity — whether
    the constructed bank is a valid operator type is checked later, by the
    lowering resolver.
    """
    if factory is None:
        return
    if isinstance(factory, functools.partial):
        raise ConfigurationError(
            "operators= must be an operator-bank class or a top-level named "
            "callable; lambdas, closures, and functools.partial are rejected "
            "because their identity is not stable"
        )
    qualname = getattr(factory, "__qualname__", None)
    if inspect.isclass(factory):
        if qualname is not None and "<locals>" not in qualname:
            return
    elif inspect.isfunction(factory):
        if qualname is not None and "<lambda>" not in qualname and "<locals>" not in qualname:
            return
    raise ConfigurationError(
        "operators= must be an operator-bank class or a top-level named "
        "callable; lambdas, closures, and functools.partial are rejected "
        "because their identity is not stable"
    )


class Component(eqx.Module):
    """A named domain-local collection of variables, equations, and BCs."""

    name: str = eqx.field(static=True)
    domain: object
    variables: tuple[Variable, ...] = eqx.field(static=True)
    # NON-static (pytree), like ``domain``: each Equation's own static-vs-leaf
    # split is internal (its ``var`` / ``inputs`` / ``name`` spine is static;
    # ``rhs_fn`` is a runtime field), so a closure's parameter arrays surface
    # as Component leaves — differentiable and rewritable — while the equation
    # spine stays in the Equation treedef. Content identity comes from the
    # explicit ``content_hash`` / ``structural_key`` walks, not from field
    # staticness.
    equations: tuple[Equation, ...] = ()
    bcs: tuple[BoundaryCondition, ...] = eqx.field(static=True, default=())
    # Operator-bank factory selecting the spatial operators this component is
    # lowered against (an operator-bank class or a top-level named callable
    # ``domain -> bank``).  Stored opaquely as static structural configuration;
    # ``None`` selects the built-in bank.  The lowering resolver constructs and
    # type-checks the bank from this component's current ``domain``.
    operator_factory: object = eqx.field(static=True, default=None)
    # Name of the declared scalar parameter that scales this component's grid
    # (``spec.geometry``), or ``None``.  When set, ``domain`` is the unit grid
    # and every kernel reads the parameter's current value and evaluates on
    # ``domain.scaled(value)``.
    geometry_scale: Optional[str] = eqx.field(static=True, default=None)
    # Runtime (non-static) tuple of flux callables for component FluxBCs,
    # parallel to Family.member_flux_callables.  ``bcs`` carries array-stripped
    # versions (for static fingerprinting); ``flux_callables`` holds the originals
    # so AD / fit reach learnable array fields inside them.  Populated in
    # __init__ from the FluxBC objects; empty tuple = backward-compatible default.
    flux_callables: tuple = ()
    # Runtime originals of Dirichlet value functions. Array-bearing modules
    # cannot live in the static BC spine without being frozen by Equinox.
    dirichlet_value_fns: tuple = ()

    def __init__(
        self,
        *,
        name: str,
        domain,
        variables: tuple[Variable, ...],
        equations: tuple,
        bcs: tuple[BoundaryCondition, ...] = (),
        operator_factory: object = None,
        geometry_scale: Optional[str] = None,
    ):
        validate_operator_factory(operator_factory)
        names = [v.name for v in variables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variable name in Component")
        variable_set = set(variables)
        written: set[Variable] = set()
        for eq in equations:
            eq_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
            for ev in eq_vars:
                if ev not in variable_set:
                    raise ValueError("equation writes to variable not in component")
                written.add(ev)
        for v in variables:
            if v not in written:
                raise ValueError(f"no equation writes to variable {v.name!r}")
        for bc in bcs:
            if bc.var not in variable_set:
                raise ValueError("boundary condition references variable not in component")

        self.name = str(name)
        self.domain = domain
        self.operator_factory = operator_factory
        self.geometry_scale = None if geometry_scale is None else str(geometry_scale)
        self.variables = tuple(variables)
        self.equations = tuple(equations)

        # Strip JAX arrays from any eqx.Module flux_fn callables in
        # the BC tuple before storing in the static ``bcs`` slot.  This avoids the
        # equinox "A JAX array is being set as static!" warning without call-site
        # suppressions.  content_hash._eqx_module_fingerprint already ignores array
        # values (it partitions by eqx.is_array and hashes only the static spine),
        # so the stripped version produces an identical fingerprint.  The ORIGINAL
        # (array-bearing) callables are preserved in the non-static ``flux_callables``
        # field so AD / fit reach learnable array leaves.
        def _strip_bc(bc):
            if isinstance(bc, FluxBC) and isinstance(bc.flux_fn, eqx.Module):
                stripped_fn = eqx.filter(bc.flux_fn, lambda x: not eqx.is_array(x))
                return FluxBC(
                    var=bc.var,
                    where=bc.where,
                    flux_fn=stripped_fn,
                    bc_name=bc.bc_name,
                    cross_facet=bool(getattr(bc, "cross_facet", False)),
                )
            if isinstance(bc, DirichletBC) and isinstance(bc.value_fn, eqx.Module):
                stripped_fn = eqx.filter(bc.value_fn, lambda x: not eqx.is_array(x))
                return DirichletBC(
                    var=bc.var,
                    where=bc.where,
                    value_fn=stripped_fn,
                    component=bc.component,
                )
            return bc

        self.bcs = tuple(_strip_bc(bc) for bc in bcs)
        # Keep original callables (with arrays) in a non-static tuple for AD.
        self.flux_callables = tuple(bc.flux_fn for bc in bcs if isinstance(bc, FluxBC))
        self.dirichlet_value_fns = tuple(bc.value_fn for bc in bcs if isinstance(bc, DirichletBC))

    def runtime_bcs(self) -> tuple[BoundaryCondition, ...]:
        """Return this component's BCs with their array-bearing callables restored.

        ``bcs`` stores Dirichlet and flux conditions whose callable had its JAX
        arrays stripped, so the tuple can be a static field without freezing
        traced values; the originals live in ``dirichlet_value_fns`` and
        ``flux_callables``.  Anything that rebuilds a Component from an existing
        one must substitute into *these* conditions, otherwise the rebuild
        re-records the array-free callables and the boundary loses its values.
        """
        dirichlet_fns = tuple(self.dirichlet_value_fns)
        flux_fns = tuple(self.flux_callables)
        restored: list[BoundaryCondition] = []
        dirichlet_index = 0
        flux_index = 0
        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                if dirichlet_index < len(dirichlet_fns):
                    bc = DirichletBC(
                        var=bc.var,
                        where=bc.where,
                        value_fn=dirichlet_fns[dirichlet_index],
                        component=bc.component,
                    )
                dirichlet_index += 1
            elif isinstance(bc, FluxBC):
                if flux_index < len(flux_fns):
                    bc = FluxBC(
                        var=bc.var,
                        where=bc.where,
                        flux_fn=flux_fns[flux_index],
                        bc_name=bc.bc_name,
                        cross_facet=bool(getattr(bc, "cross_facet", False)),
                    )
                flux_index += 1
            restored.append(bc)
        return tuple(restored)

    def structural_key(self) -> tuple:
        """Hashable structural fingerprint of this Component.

        Folds the domain, the equations, and every static field.  Equations
        and BCs route through [_eq_id][] / [_bc_id][] so
        closure-bearing nodes produce stable fingerprints across
        re-construction (plain lambdas hash by id, which would break key
        stability); ``equations`` is folded explicitly because it is a runtime
        pytree field (its closures carry parameter arrays), not a static one.
        Domain is keyed via [_domain_id][].

        Components with identical structural keys can be group-vmap'd in
        ``lower``.
        """
        from .content_hash import _bc_id, _domain_id, _eq_id

        parts: list = [type(self).__qualname__, _domain_id(self.domain)]
        parts.append(("equations", tuple(_eq_id(e) for e in self.equations)))
        for f in dataclasses.fields(self):
            if not f.metadata.get("static", False):
                continue
            value = getattr(self, f.name)
            if f.name == "bcs":
                parts.append((f.name, tuple(_bc_id(b) for b in value)))
            else:
                parts.append((f.name, _static_field_value_id(value)))
        return tuple(parts)


def _static_field_value_id(value):
    """Fallback fingerprint for static field values without a dedicated path.

    Used for ``name`` (str), ``variables`` (tuple of Variable, hash-stable by
    struct), the ``operator_factory`` (a class or top-level callable, keyed by
    qualified name so it is stable across reconstruction), and any future static
    field added to Component.  Closures, ndarrays, and tuples are normalised to
    hashable surrogates so the enclosing [structural_key][] tuple
    stays hashable.
    """
    if isinstance(value, type):
        return ("class", value.__module__, value.__qualname__)
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", repr(value)),
        )
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, tuple):
        return tuple(_static_field_value_id(v) for v in value)
    return value
