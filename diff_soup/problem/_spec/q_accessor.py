# The ``q`` accessor for Recipe API equation closures.
"""The ``q`` accessor — sugar over ``view[Variable]`` lookups inside ``@spec.equation`` closures.

Bindings:

* ``q.<name>`` / ``q["<name>"]`` -> ``view[Variable]``
* ``q.ops`` -> operator bank for the closure's component
* ``q.p.<name>`` -> resolved parameter value
* ``q.fn.<name>`` -> declared callable (its per-node application when the
  declaration carries ``pointwise=True``)
* ``q.t`` -> current time scalar
* ``q.x`` -> ``domain.node_coordinates()``
* ``q.grad(field)`` -> raw quad-point gradient via ``q.ops.quad_gradient``
* ``q.flux(expr_at_quad)`` -> assembled IBP flux residual via ``q.ops.integrate_shape_gradient``
* ``q.source(expr_at_quad)`` -> assembled source residual via ``q.ops.integrate_shape``

The accessor is a plain Python object — it never crosses a JIT trace
boundary on its own.  Its ``view`` and ``t`` carry traced JAX arrays;
its ``var_map``, ``ops``, and ``params`` are static host-side state that
is materialised fresh per residual call.

On a grid domain (FVM/FDM), ``q.flux``, ``q.source``, and ``q.grad`` surface
the capability error from ``q.ops``; those hooks raise and direct users to
``q.ops.gradient``, ``q.ops.div_flux``, and ``q.ops.laplacian`` instead.

The accessor answers ``ops``, ``p``, ``fn``, ``t``, ``x``, and ``control``
from its own machinery, ahead of the field table.  Those six names are therefore
reserved: ``spec.field`` and ``spec.global_var`` refuse them, because a field
or global carrying one could never be read back as ``q.<name>``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

# Each reserved attribute the accessor answers itself, mapped to the private
# slot holding its value.
_RESERVED_Q_SLOTS = {
    "ops": "_ops",
    "p": "_p",
    "fn": "_fn",
    "t": "_t",
    "x": "_coords",
    "control": "_control",
}

#: The attribute names ``q`` resolves before consulting its field table.  A
#: field or global with one of these names is unreachable through the
#: documented ``q.<name>`` spelling, so the declaration verbs reject them.
RESERVED_Q_NAMES = frozenset(_RESERVED_Q_SLOTS)


class _ParamsNS:
    """Dotted access namespace for parameters."""

    __slots__ = ("_params",)

    def __init__(self, params: dict[str, Any]) -> None:
        self._params = params

    def __getattr__(self, name: str) -> Any:
        params = object.__getattribute__(self, "_params")
        if name in params:
            return params[name]
        raise AttributeError(f"q.p has no parameter {name!r}")


class _ControlNS:
    """Dotted access namespace for resolved control values.

    ``q.control.<name>`` is the dimensional value of the declared control ``<name>`` at
    the evaluation's own time — the authored ``fn(t_SI)`` by default, or the
    override payload riding the evaluation.  One source of truth: the value an
    equation reads here is the value the events and the solve-time override
    channel see.
    """

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        if not values:
            raise AttributeError(
                f"q.control.{name}: no control values reach this closure. "
                "Controls reach equation, constraint and boundary-flux "
                "closures; declare the control with spec.control.direct(...) "
                "and read it from one of those."
            )
        raise AttributeError(
            f"q.control has no control {name!r}. Declared controls: "
            f"{sorted(values)!r}. Declare it with spec.control.direct(...) "
            "or spec.control.derived(...)."
        )


class _FunctionNS:
    """Dotted access namespace for named functions.

    A name declared ``spec.function(name, fn, pointwise=True)`` states that
    ``fn`` maps one sample, so this namespace hands back ``jax.vmap(fn)`` — the
    per-node application — instead of the raw callable.  Every other name
    returns exactly what was declared.
    """

    __slots__ = ("_functions", "_pointwise")

    def __init__(self, functions: dict[str, Any], pointwise: "Any" = ()) -> None:
        self._functions = functions
        self._pointwise = frozenset(pointwise or ())

    def __getattr__(self, name: str) -> Any:
        functions = object.__getattribute__(self, "_functions")
        if name in functions:
            fn = functions[name]
            pointwise = object.__getattribute__(self, "_pointwise")
            return jax.vmap(fn) if name in pointwise else fn
        raise AttributeError(f"q.fn has no function {name!r}")


class _QView:
    """Read-only accessor passed to ``@spec.equation`` closures."""

    __slots__ = (
        "_view",
        "_var_map",
        "_ops",
        "_p",
        "_fn",
        "_t",
        "_coords",
        "_quad_fields",
        "_control",
    )

    def __init__(
        self,
        *,
        # ``view`` is any owner-keyed accessor supporting ``view[key]``: the
        # component path passes a plain ``{Variable: array}`` dict, while the
        # family member residual lowering passes a per-member ``MergedView`` —
        # both are duck-typed at ``_resolve_field``, so the annotation must not
        # pin ``dict`` (the beartype import hook enforces it).
        view: Any,
        var_map: dict[str, Any],
        ops: Any,
        params: Optional[dict[str, Any]],
        t: Any,
        coords: Any,
        functions: Optional[dict[str, Any]] = None,
        pointwise_functions: "Any" = (),
        quad_fields: "Any" = (),
        controls: Optional[dict[str, Any]] = None,
    ) -> None:
        self._view = view
        self._var_map = var_map
        self._ops = ops
        self._p = _ParamsNS(params or {})
        self._fn = _FunctionNS(functions or {}, pointwise_functions)
        self._t = t
        self._coords = coords
        self._control = _ControlNS(controls or {})
        # ``quad_fields`` arrives as any iterable of names (a sorted tuple from
        # the build adapters so the closure default stays content-hashable);
        # normalise to a frozenset for O(1) membership at field resolution.
        self._quad_fields = frozenset(quad_fields or ())

    def _resolve_field(self, name: str) -> Any:
        """Look up field ``name`` and wrap quad-opted FEM fields in a ``Field``.

        On an FEM domain, a field declared ``spec.field(quad=True)`` (or by the
        FEM default ``quad=None`` → ``True``) returns a
        [Field][diff_soup.domains.operators._field.Field] so the closure authors
        the strong form against ``q.<name>`` directly — same closure style on FEM
        as on FVM/FDM.  On grids and for non-opted fields, returns the raw view
        array unchanged.

        The Field wrapping is performed by ``ops.wrap_quad_field(val)`` — a method
        call on the already-provided Operators instance; no new domain-layer import.
        """
        view = object.__getattribute__(self, "_view")
        var_map = object.__getattribute__(self, "_var_map")
        val = view[var_map[name]]
        quad_fields = object.__getattribute__(self, "_quad_fields")
        if name not in quad_fields:
            return val
        ops = object.__getattribute__(self, "_ops")
        # ``ops.wrap_quad_field`` wraps on FEM, passes through on grids.
        # The Field class lives in L1 (domains); we reach it via a method call
        # on the L1 Operators object — no new L1 import in this L2 module.
        return ops.wrap_quad_field(val)

    def __getitem__(self, name: str) -> Any:
        var_map = object.__getattribute__(self, "_var_map")
        if name in var_map:
            return self._resolve_field(name)
        raise KeyError(f"q has no field {name!r}")

    def __getattr__(self, name: str) -> Any:
        # The reserved names are answered from the accessor's own slots, ahead
        # of ``var_map``; ``RESERVED_Q_NAMES`` is the set the declaration verbs
        # reject, so both sides stay in step.
        slot = _RESERVED_Q_SLOTS.get(name)
        if slot is not None:
            return object.__getattribute__(self, slot)
        var_map = object.__getattribute__(self, "_var_map")
        if name in var_map:
            return self._resolve_field(name)
        raise AttributeError(f"q has no field {name!r}")

    def dt(self, field: str):
        """The time derivative of ``field`` as a derivative-expression leaf.

        An equation closure that returns a derivative expression states a
        residual (``expression == 0``); a closure that returns a plain array
        keeps the right-hand-side reading ``d(field)/dt = value``. The leaf
        composes through the derivative algebra — scaling by derivative-free
        values, declared linear operators, and gather/reduce/scatter maps.
        """
        from diff_soup.problem._spec.derivative_expr import TimeDerivativeRef

        var_map = object.__getattribute__(self, "_var_map")
        if field not in var_map:
            raise AttributeError(f"q.dt has no field {field!r}")
        var = var_map[field]
        return TimeDerivativeRef(field_name=field, var=var, size=int(var.size))

    def grad(self, field: Any) -> Any:
        r"""Raw quad-point gradient of ``field``.

        Routes to ``q.ops.quad_gradient(q.ops.unwrap_field(field))`` for both
        scalar and vector fields.  On FEM this returns quadrature-point gradient
        values: scalar input gives ``(n_cells, n_q, dim)``; vector input gives
        ``(n_cells, n_q, n_components, dim)`` with ``H[..., i, j] = du_i/dx_j``.

        ``q.ops.grad`` uses the same index order for a vector field but a
        different normalization: it projects the weak gradient to nodes and
        divides by the lumped mass, giving ``(n_nodes, n_components, dim)``.
        Use this one for quadrature-point composition (``q.flux``,
        ``q.source``) and that one for a pointwise nodal Jacobian.

        On a grid (FVM/FDM) the capability error from ``q.ops.quad_gradient``
        is raised, directing the closure author to ``q.ops.gradient``,
        ``q.ops.div_flux``, or ``q.ops.laplacian`` instead.

        Accepts a [Field][diff_soup.domains.operators._field.Field] (its
        nodal DOFs are unwrapped via ``q.ops.unwrap_field``) or a plain nodal
        array.
        """
        ops = object.__getattribute__(self, "_ops")
        return ops.quad_gradient(ops.unwrap_field(field))

    def flux(self, expr_at_quad: Any) -> Any:
        r"""Assemble the IBP flux weak form :math:`\int \text{expr}\cdot\nabla N_i`.

        Routes to ``q.ops.integrate_shape_gradient(expr_at_quad)`` and returns an
        already-integrated residual term.  On FEM, ``expr_at_quad`` must be a
        quadrature-point flux array of shape ``(n_cells, n_q, dim)`` (scalar field
        flux) or ``(n_cells, n_q, n_voigt)`` (Voigt stress tensor).

        On a grid (FVM/FDM) the capability error from
        ``q.ops.integrate_shape_gradient`` is raised, directing the closure author
        to ``q.ops.div_flux`` or ``q.ops.laplacian`` instead.
        """
        ops = object.__getattribute__(self, "_ops")
        return ops.integrate_shape_gradient(expr_at_quad)

    def source(self, expr_at_quad: Any) -> Any:
        r"""Assemble the source weak form :math:`\int \text{expr}\cdot N_i`.

        Routes to ``q.ops.integrate_shape(expr_at_quad)`` and returns an
        already-integrated residual term.  On FEM, ``expr_at_quad`` must be a
        quadrature-point scalar array of shape ``(n_cells, n_q)``.

        On a grid (FVM/FDM) the capability error from ``q.ops.integrate_shape``
        is raised, directing the closure author to ``q.ops.laplacian`` or a
        pointwise residual expression instead.
        """
        ops = object.__getattribute__(self, "_ops")
        return ops.integrate_shape(expr_at_quad)


class _BoundaryNodal:
    """Full nodal values, recording the wider dependency of an interior read."""

    def __init__(self, q):
        self._q = q

    def __getattr__(self, name):
        q = self._q
        if name not in q._backing:
            raise KeyError(name)
        q.accessed.add(name)
        q.nodal_accessed.add(name)
        return q._backing[name]


class _BoundaryGeom:
    """Outward unit normals and positions of the selected facets."""

    def __init__(self, normal, x):
        self.normal = normal
        self.x = x


class _BoundaryQ(_QView):
    """Boundary accessor with facet means and access to the owning nodal fields."""

    def __init__(
        self,
        backing,
        global_names,
        face_indices,
        normals,
        centroids,
        *,
        ops=None,
        params=None,
        functions=None,
        pointwise_functions=(),
        controls=None,
        t=0.0,
        member=None,
    ):
        domain = getattr(ops, "domain", None)
        coords_fn = getattr(domain, "node_coordinates", None)
        super().__init__(
            view=backing,
            var_map={name: name for name in backing},
            ops=ops,
            params=params,
            functions=functions,
            pointwise_functions=pointwise_functions,
            controls=controls,
            t=t,
            coords=coords_fn() if callable(coords_fn) else None,
        )
        self._backing = backing
        self._globals = frozenset(global_names)
        self._fi = jnp.asarray(face_indices, dtype=jnp.int32)
        self.accessed = set()
        self.nodal_accessed = set()
        self.geom = _BoundaryGeom(normals, centroids)
        self.nodal = _BoundaryNodal(self)
        self.m = member

    def _resolve_field(self, name):
        self.accessed.add(name)
        value = self._backing[name]
        return value if name in self._globals else jnp.mean(value[self._fi], axis=1)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("__"):
                raise
            raise KeyError(name) from None
