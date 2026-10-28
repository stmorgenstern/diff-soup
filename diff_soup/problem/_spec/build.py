# LOC > 500: single cohesive spec->IR walk — emitters (BC/constraint/obs/
# control) + the Scales/residual_scale resolution share the var_map/params
# closure plumbing; splitting would fragment that shared state.
"""Walk a [ProblemSpec][] and emit a frozen [EquationSet][].

Steps:

1. Variable per FieldSpec; GlobalVariable per GlobalVarSpec.
2. Adapter wrapping each EquationSpec.fn so it accepts the
   ``rhs_fn(view, ctx, t)`` signature the solver-problem lowering pipeline
   expects.  The adapter materialises a [_QView][] populated with
   the IR ``view`` dict, the ``ParamSpec`` values, ``ops`` from the
   lowering ``ctx``, and ``t`` / coords from the domain.
3. DirichletBC / FluxBC IR nodes per BCSpec.
4. Constraint / Observation / Control IR nodes per spec lists (typed
   handle resolution).
5. Component(s) — one per registered domain.
6. EquationSet packaging the lot.

The static validation pass runs first, before any of the above: it reads the
declarations alone, so a recipe whose parts do not line up is refused before a
single closure is traced.

The family / population and cross-component coupling emit arms
(``_emit_coupling``, ``_resolve_coupling_endpoint``, ``_phantom_self_rhs``,
read-through alias threading, and per-family scale propagation) are fully
wired. The build walk emits ``couplings`` / ``families`` onto the EquationSet
(consumed by the [diff_soup._lowering][] provider). Specs with no coupling
or family emit only the core same-domain IR (components, globals, BCs, the four
typed constraint kinds, observations, controls, and Scales / residual_scale
resolution).

Equation and observation closures are wrapped by ``_make_q_adapter``. The
``quad_fields`` set (the field names declared ``quad=True``) rides the equation
adapters, never stored directly on a frozen IR node.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._warnings import warn_user
from diff_soup.problem._ir import (
    BoundarySpec,
    Component,
    DirichletBC,
    Equation,
    EquationSet,
    FluxBC,
    GlobalVariable,
    MultiEquation,
    PeriodicBC,
    RobinBC,
    SymmetryBC,
    TractionBC,
    Variable,
    _BoundaryFn,
)
from diff_soup.problem._ir.bc import _constant_fn, _time_flux_fn
from diff_soup.problem._ir.constraint import (
    BoundaryIntegralConstraint,
    DifferentialGlobalEquation,
    DomainIntegralConstraint,
    FluxTotalConstraint,
    GalvanostaticConstraint,
    OperationalControlConstraint,
    ReduceResidualConstraint,
)
from diff_soup.problem._ir.content_hash import _closure_fingerprint
from diff_soup.problem._ir.control import DerivedControl, DirectControl
from diff_soup.problem._ir.coupling import Coupling
from diff_soup.problem._ir.observation import (
    BoundaryMeanObservation,
    FieldObservation,
    ScalarObservation,
    SpatialMeanObservation,
)
from diff_soup.problem._ir.scales import Scales

from .q_accessor import _QView
from .specs import (
    BCSpec,
    ConstraintSpec,
    ControlSpec,
    ObservationSpec,
    OperationalControlSpec,
)


class _NamedValues(eqx.Module):
    """Small pytree namespace with stable names and dynamic values.

    The values are keyed by declaration name, so a declared parameter or
    function keeps its name on the key path of every array leaf it owns —
    that is what lets ``theta=("D",)`` name a declared parameter and what
    labels a fitted coordinate.

    ``pointwise_names`` marks the subset whose value maps ONE sample; the
    ``q`` accessor batches those over the nodes when the closure reads them.
    It is empty for the parameter namespace and for every function declared
    without ``spec.function(..., pointwise=True)``.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    values: dict[str, Any]
    pointwise_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, values: dict[str, Any], pointwise_names: Iterable[str] = ()):
        names = tuple(values.keys())
        self.names = names
        self.values = {name: values[name] for name in names}
        self.pointwise_names = tuple(pointwise_names)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def __getattr__(self, name: str) -> Any:
        try:
            names = object.__getattribute__(self, "names")
        except AttributeError:
            raise AttributeError(name) from None
        if name in names:
            return self[name]
        raise AttributeError(name)

    def as_dict(self) -> dict[str, Any]:
        return {name: self.values[name] for name in self.names}


def _make_q_view(view, ctx, t, var_map, params, functions, quad_fields=()):
    controls = getattr(ctx, "controls", None)
    domain = getattr(ctx.ops, "domain", None)
    coords_fn = getattr(domain, "node_coordinates", None) if domain is not None else None
    # Grids store their coordinates in double precision; a single-precision
    # problem hands its equations coordinates in its own (time) dtype.
    time_dtype = getattr(t, "dtype", None)
    narrow = time_dtype is not None and jnp.promote_types(jnp.float64, time_dtype) != time_dtype
    in_field_dtype = getattr(domain, "in_field_dtype", None)
    if narrow and in_field_dtype is not None:
        coords_fn = in_field_dtype(jnp.zeros((), time_dtype)).node_coordinates
    coords = coords_fn() if callable(coords_fn) else None
    if narrow and coords is not None and jnp.promote_types(coords.dtype, time_dtype) != time_dtype:
        coords = jnp.asarray(coords).astype(time_dtype)
    params_dict = params.as_dict() if hasattr(params, "as_dict") else params
    functions_dict = functions.as_dict() if hasattr(functions, "as_dict") else functions
    # The pointwise marks ride the functions namespace itself, so every closure
    # that reads ``q.fn`` — equation, constraint integrand, observation — sees
    # the same per-node application.
    pointwise_functions = getattr(functions, "pointwise_names", ())
    return _QView(
        view=view,
        var_map=var_map,
        ops=ctx.ops,
        params=params_dict,
        functions=functions_dict,
        pointwise_functions=pointwise_functions,
        t=t,
        coords=coords,
        quad_fields=quad_fields,
        controls=controls,
    )


class _TracedQAdapter(eqx.Module):
    """Wrap a user closure ``f(q)`` so its ``eqx.Module`` leaves survive partition.

    When the user's closure is an ``eqx.Module``, wrapping it in a plain
    Python function would bury its parameter arrays in a closure cell —
    those leaves would never reach
    ``eqx.partition(problem, eqx.is_inexact_array)[0]`` and ``jax.grad``
    would silently return zero.  Routing through this adapter (itself an
    ``eqx.Module``) keeps the ``inner`` field non-static so its leaves
    surface on the AD boundary.
    """

    inner: object
    var_map: object
    params: object
    functions: object
    learnable_params: tuple[str, ...] = eqx.field(static=True, default=())
    learnable_functions: tuple[str, ...] = eqx.field(static=True, default=())
    # Sorted tuple of quad-opted field names (a tuple, not a frozenset, so the
    # content-hash static-leaf walk fingerprints it deterministically).
    quad_fields: tuple[str, ...] = eqx.field(static=True, default=())

    def __call__(self, view, ctx, t):
        q = _make_q_view(view, ctx, t, self.var_map, self.params, self.functions, self.quad_fields)
        return self.inner(q)


def _make_q_adapter(
    spec_fn: Callable,
    var_map: dict[str, Any],
    params: Any,
    functions: Any,
    *,
    learnable_params: Iterable[str] = (),
    learnable_functions: Iterable[str] = (),
    quad_fields: Iterable[str] = (),
) -> Callable:
    """Wrap a spec-level closure ``f(q)`` for IR-level callsite ``f(view, ctx, t)``.

    ``ctx.ops`` carries an [Operators][] whose ``.domain`` exposes
    ``node_coordinates()`` for ``q.x``.  ``eqx.Module`` callables route
    through [_TracedQAdapter][] to keep their leaves visible under
    AD partition. Declared parameters and declared functions also use a
    traced adapter, so their values are leaves of the lowered problem: fixed
    per-experiment inputs can be replaced without rebuilding the residual, and
    no declared value is baked into a compiled program.
    A plain callable with no learnable declarations still obeys the strict
    static-capture contract, even when that adapter carries fixed parameters.

    ``quad_fields`` is the set of field names declared ``spec.field(quad=True)``;
    it rides into the per-call [_QView][].  Stored as a SORTED TUPLE (the
    closure default / static eqx field must be content-hashable — a frozenset is
    not).  Defaults to empty (no field wrapped — today's behaviour).
    """
    learnable_params = tuple(learnable_params)
    learnable_functions = tuple(learnable_functions)
    quad_fields = tuple(sorted(quad_fields))
    needs_traced_adapter = (
        isinstance(spec_fn, eqx.Module)
        or bool(learnable_params)
        or bool(learnable_functions)
        or bool(getattr(params, "names", ()))
        or bool(getattr(functions, "names", ()))
    )
    if needs_traced_adapter:
        if not isinstance(spec_fn, eqx.Module) and not (learnable_params or learnable_functions):
            _closure_fingerprint(spec_fn)
        return _TracedQAdapter(
            inner=spec_fn,
            var_map=var_map,
            params=params,
            functions=functions,
            learnable_params=learnable_params,
            learnable_functions=learnable_functions,
            quad_fields=quad_fields,
        )

    def adapter(
        view,
        ctx,
        t,
        _spec_fn=spec_fn,
        _var_map=var_map,
        _params=params,
        _functions=functions,
        _quad_fields=quad_fields,
    ):
        q = _make_q_view(view, ctx, t, _var_map, _params, _functions, _quad_fields)
        return _spec_fn(q)

    return adapter


def _make_multi_extractor(multi_rhs: Callable, key: str) -> Callable:
    """Return ``rhs(view, ctx, t)[key]`` — one output of a multi-write equation.

    Lets a reduce_residual constraint read a single field's residual without
    duplicating the multi-write closure.

    The returned function captures ``multi_rhs`` by reference; when the
    reduce_residual constraint integrand_fn is invoked inside the same
    JIT trace as the equation's rhs_fn, XLA's HLO-level CSE folds the
    redundant per-cell kernel computation.
    """

    def extractor(view, ctx, t):
        out = multi_rhs(view, ctx, t)
        return jnp.asarray(out[key]).reshape(-1)

    return extractor


def _phantom_self_rhs(var: Variable) -> Callable:
    """Self-determining ``rhs_fn`` ``(view, ctx, t)`` for a smart-couple phantom.

    Returns ``-view[var]`` so the assembled algebraic row is
    ``r = -phantom + injected`` (the scatter-ADD contributes ``+injected``),
    which solves to ``phantom == injected`` with a well-posed
    ``(phantom, phantom)`` diagonal (``dr/dphantom == -1``).

    Carrying the self term makes every phantom row non-singular, whether or not
    a source actually writes it — a plain ``r = injected`` base would leave
    ``0 = 0`` (rank-deficient) at any target node with no incoming contribution.

    Still live for the smart-couple SCATTER auto-phantom path (a
    ``couple(proj.surface(), onto=<domain>, reduce=)`` WITHOUT ``as_=``); the
    NO-DOF read-through path (``couple(..., as_=<alias>)``) needs no phantom row.
    """

    def adapter(view, ctx, t, _var=var):  # noqa: ARG001
        return -view[_var]

    return adapter


def _emit_coupling(cs, var_map: dict[str, Any]) -> Coupling:
    """Resolve CouplingSpec endpoints to IR Variable handles.

    Endpoints are either a [FieldSpec][] (resolved through ``var_map`` to
    the component Variable) or — for a smart family scatter / a read-through
    alias — an already-built [Variable][] (passed through by identity so
    the lowering routes it via the family scatter / read-through branch).
    ``read_through`` and ``target_owner`` ride the spec onto the IR
    [Coupling][].
    """
    src_var = _resolve_coupling_endpoint(cs.source, var_map)
    tgt_var = _resolve_coupling_endpoint(cs.target, var_map)
    return Coupling(
        name=cs.name,
        source=src_var,
        target=tgt_var,
        op=cs.via,
        read_through=bool(getattr(cs, "read_through", False)),
        target_owner=getattr(cs, "target_owner", None),
    )


def _resolve_coupling_endpoint(endpoint, var_map: dict[str, Any]):
    """Resolve a coupling endpoint to an IR ``Variable``.

    An endpoint is either an already-built [Variable][] (passed through by
    identity — a family scatter passes the member surface ``Variable``, and a
    family GATHER passes the ``onto`` family's member ``Variable`` so
    ``_resolve_family_couplings`` routes it on the family branch) or a component
    [FieldSpec][] (looked up by name in ``var_map``).
    """
    if isinstance(endpoint, Variable):
        return endpoint
    return var_map[endpoint.name]


def _make_flux_fn(value: Any, value_fn: Optional[Callable]) -> Callable:
    """Build a ``(view, ctx, t) -> flux`` callable for a FluxBC.

    The IR ``FluxBC.flux_fn`` signature is ``(view, ctx, t)`` to support
    state-dependent fluxes (Butler-Volmer, convective cooling, etc.), while the
    public flux datum is either a constant ``value=`` or a ``value_fn=`` of time
    alone.  Both are wrapped into that internal form here: the constant in a
    closure that ignores its arguments and returns it, the time-dependent datum
    in the ``(view, ctx, t) -> value_fn(t)`` adapter.
    """
    if value_fn is not None and value is not None:
        raise ValueError("BC flux: provide only one of value or value_fn")
    if value_fn is not None:
        return _time_flux_fn(value_fn)
    arr = jnp.asarray(0.0 if value is None else value)

    def fn(view, ctx, t, _value=arr):  # noqa: ARG001
        return _value

    return fn


def _resolve_where(where: dict[str, Any], domain=None) -> BoundarySpec:
    if where.get("face") is not None:
        from diff_soup._lowering.bcs import _grid_face_cells

        axis, side = where["face"]
        if getattr(domain, "discretization_type", None) not in ("fvm", "fdm"):
            raise ValueError(
                f"axis={axis}, side={side!r} names a face of a structured grid, and this "
                f"field lives on a {type(domain).__name__}. Select the boundary with on=, "
                "tag= or node_indices= instead."
            )
        return BoundarySpec(node_indices=_grid_face_cells(domain, axis, side), face=(axis, side))
    if "selector" in where and where["selector"] is not None:
        return BoundarySpec(selector=where["selector"])
    if "tag" in where and where["tag"] is not None:
        return BoundarySpec(tag=where["tag"])
    if "node_indices" in where and where["node_indices"] is not None:
        ni = np.asarray(where["node_indices"])
        if ni.dtype == bool:
            # A boolean mask selects the True positions; pre-casting to
            # int32 here would coerce it to the literal indices {0, 1}
            # before BoundarySpec's bool guard could fire.
            ni = np.nonzero(ni.reshape(-1))[0]
        return BoundarySpec(node_indices=np.asarray(ni, dtype=np.int32))
    raise ValueError("BCSpec.where missing a selector, tag, and node_indices")


def _inflow_flux_fn(velocity, value, value_fn):
    """The flux density ``velocity * c_in`` an inflow face carries into the domain."""
    speed = jnp.asarray(velocity, dtype=float)
    inlet = _constant_fn(value) if value_fn is None else value_fn

    def flux_fn(view, ctx, t, _speed=speed, _inlet=inlet):  # noqa: ARG001
        return _speed * jnp.asarray(_inlet(t))

    return flux_fn


def _emit_bc(bc: BCSpec, var_map: dict[str, Any], domain=None):
    var = var_map[bc.var]
    where = _resolve_where(bc.where, domain)
    if bc.kind == "inflow":
        # Danckwerts ``v c - D dc/dn = v c_in``: the face carries the total flux
        # ``v c_in`` into the domain, while div_advect carries no flux in through
        # it and div_flux keeps its natural zero flux there.
        return FluxBC(var=var, where=where, flux_fn=_inflow_flux_fn(bc.velocity, bc.value, bc.value_fn))
    if bc.kind == "dirichlet":
        return DirichletBC(
            var=var,
            where=where,
            value=None if bc.value_fn is not None or bc.fn is not None else bc.value,
            value_fn=_BoundaryFn(fn=bc.fn, bc_name=bc.name) if bc.fn is not None else bc.value_fn,
            component=bc.component,
        )
    if bc.kind == "flux":
        if bc.fn is not None:
            return FluxBC(
                var=var,
                where=where,
                flux_fn=_BoundaryFn(fn=bc.fn, bc_name=bc.name),
                bc_name=bc.name,
                cross_facet=bool(getattr(bc, "cross_facet", False)),
            )
        # ``name=`` is a property of the boundary condition, not of the syntax
        # used to state its density, so the constant and time-dependent forms
        # carry it exactly as the field-dependent one does.
        return FluxBC(
            var=var,
            where=where,
            flux_fn=_make_flux_fn(bc.value, bc.value_fn),
            bc_name=bc.name,
        )
    if bc.kind == "robin":
        return RobinBC(
            var=var,
            where=where,
            alpha=bc.alpha,
            beta=bc.beta,
            value=None if bc.value_fn is not None or bc.fn is not None else bc.value,
            value_fn=_BoundaryFn(fn=bc.fn, bc_name=bc.name) if bc.fn is not None else bc.value_fn,
        )
    if bc.kind == "traction":
        return TractionBC(
            var=var,
            where=where,
            value=None if bc.value_fn is not None or bc.fn is not None else bc.value,
            value_fn=_BoundaryFn(fn=bc.fn, bc_name=bc.name) if bc.fn is not None else bc.value_fn,
        )
    if bc.kind == "symmetry":
        return SymmetryBC(var=var, where=where)
    raise ValueError(f"unknown BC kind {bc.kind!r}")


def _grid_dim(domain) -> int:
    """Number of spatial axes of a grid domain (1 for ``Grid1D``, ``len(axes)`` for ``GridND``)."""
    axes = getattr(domain, "axes", None)
    if axes is not None:
        return len(axes)
    return 1


def _check_closes_pairing(spec) -> None:
    """Check that ``closes=`` pairs constraints and algebraic globals one-to-one.

    A global contributes one unknown slice — a scalar unless it is declared
    with a wider shape — and a constraint contributes the rows that close it;
    ``closes=`` is what pairs the two: the multiplier vector holds one slice
    per constraint, and the global named by that constraint's ``closes=`` is
    the unknown living in it.  The declared system
    is square only when that pairing is a bijection, so a global no constraint
    closes, a global two constraints close, and a ``closes=`` naming a field
    are all refused here — at the declaration, before any equation is traced.

    A global declared ``mass="differential"`` is written by its own ordinary
    equation instead of a constraint row, so it is not part of this pairing;
    the equation that writes it is checked with the other global closures.

    Args:
        spec: The ``ProblemSpec`` being walked.

    Raises:
        ValueError: If a constraint's ``closes=`` does not name a declared
            global, if two constraints close the same global, or if a declared
            algebraic global is closed by no constraint.
    """
    declared = sorted(spec._globals)
    closed_by: dict[str, str] = {}
    for c in spec._constraints:
        target = c.closes
        if target is None:
            raise ValueError(
                f"constraint {c.name!r} names no global to close; every "
                "constraint determines exactly one global, passed as "
                f"closes=<name>. Declared globals: {declared!r}."
            )
        if target not in spec._globals:
            if target in spec._fields:
                raise ValueError(
                    f"constraint {c.name!r} closes {target!r}, which is a field, "
                    "not a global; a field already carries one row per node, "
                    "while closes= names the single scalar unknown the "
                    "constraint determines, declared with spec.global_var. "
                    f"Declared globals: {declared!r}."
                )
            raise ValueError(
                f"constraint {c.name!r} closes {target!r}, which is not a "
                "declared global; closes= names a global declared with "
                f"spec.global_var. Declared globals: {declared!r}."
            )
        first = closed_by.get(target)
        if first is not None:
            raise ValueError(
                f"constraints {first!r} and {c.name!r} both close global "
                f"{target!r}; a global carries one multiplier and is determined "
                "by exactly one constraint row — give one of the two its own "
                "spec.global_var to close, or drop it."
            )
        closed_by[target] = c.name
    for name, gspec in spec._globals.items():
        if name in closed_by:
            continue
        if getattr(gspec, "mass", "algebraic") == "differential":
            # Written by its own ordinary equation, not by a constraint row.
            continue
        raise ValueError(
            f"global {name!r} is closed by no constraint; an algebraic global "
            "is determined by the one constraint row that closes it — declare "
            f"a constraint with closes={name!r} (spec.constraint.galvanostatic, "
            ".integral, .boundary_integral, .reduce_residual, .flux_total), name "
            f"it as the current unknown of spec.control.operational(...), "
            f'declare it mass="differential" and write it with an equation, or '
            f"drop the global. spec.control.derived(..., binds_to={name!r}) "
            "drives the multiplier of that constraint; it does not close the "
            "global."
        )


def _emit_operational_control(
    c: OperationalControlSpec,
    var_map: dict[str, Any],
    params: Any,
    functions: Any,
    learnable_params: tuple[str, ...],
    learnable_functions: tuple[str, ...],
    observations,
    field_to_rhs: Optional[dict[str, Callable]] = None,
):
    """Emit the operational-control table node from its authored record.

    The table stores the terminal observables' q-adapted closures; at
    residual time they evaluate against the constraint merged view, which
    binds fields and globals by name.  A table whose current is the sum of a
    field's assembled residual rows stores that field instead of a current
    closure.
    """
    if c.closes not in var_map:
        raise ValueError(f"operational control {c.name!r} closes unknown global {c.closes!r}")
    if c.field is not None and (field_to_rhs is None or c.field not in field_to_rhs):
        raise ValueError(
            f"operational control {c.name!r} takes its current from the residual "
            f"rows of field {c.field!r}, and no equation writes that field; name a "
            "field an equation writes in current_from_residual="
        )
    obs_by_name = {o.name: o for o in observations}
    adapted = {"current": None}
    for role, obs_name in (("current", c.current), ("voltage", c.voltage)):
        if obs_name is None:
            continue
        adapted[role] = _make_q_adapter(
            obs_by_name[obs_name].fn,
            var_map,
            params,
            functions,
            learnable_params=learnable_params,
            learnable_functions=learnable_functions,
        )
    return OperationalControlConstraint(
        name=c.name,
        closes=var_map[c.closes],
        current_fn=adapted["current"],
        voltage_fn=adapted["voltage"],
        modes=tuple(c.modes),
        seed_voltage_threshold=float(c.seed_voltage_threshold),
        field=None if c.field is None else var_map[c.field],
    )


def _emit_constraint(
    c: ConstraintSpec,
    var_map: dict[str, Any],
    params: Any,
    functions: Any,
    learnable_params: tuple[str, ...],
    learnable_functions: tuple[str, ...],
    field_to_rhs: Optional[dict[str, Callable]] = None,
    integrand_scale_by_name: Optional[dict[str, float]] = None,
    field_to_domain: Optional[dict[str, Any]] = None,
):
    if c.closes is None or c.closes not in var_map:
        raise ValueError(
            f"constraint {c.name!r} closes {c.closes!r}, which is not a declared "
            "global; closes= names a global declared with spec.global_var"
        )
    closes = var_map[c.closes]
    if not isinstance(closes, GlobalVariable):
        raise ValueError(
            f"constraint {c.name!r} closes {c.closes!r}, which is a field, not a "
            "global; closes= names the single scalar unknown the constraint "
            "determines, declared with spec.global_var"
        )
    # Resolve the constraint node's integrand_scale.  The resolved integrand
    # reference (inline ``ConstraintSpec`` or central ``spec.scales(integrand=...)``)
    # sets BOTH the node's ``integrand_scale`` AND ``Scales.integrand_ref[name]``
    # to the same value here, which is the only place the two are tied
    # together -- there is no separate check downstream that reconciles them.
    # Default 1.0 when unset.
    iscale = (integrand_scale_by_name or {}).get(c.name)
    if iscale is None:
        iscale = 1.0
    iscale = float(iscale)
    if c.kind == "galvanostatic":
        return GalvanostaticConstraint(
            name=c.name,
            closes=closes,
            current_fn=_make_q_adapter(
                c.current_fn,
                var_map,
                params,
                functions,
                learnable_params=learnable_params,
                learnable_functions=learnable_functions,
            ),
            target_fn=c.target_fn,
            integrand_scale=iscale,
        )
    if c.kind == "integral":
        return DomainIntegralConstraint(
            name=c.name,
            closes=closes,
            field=var_map[c.field],
            integrand_fn=_make_q_adapter(
                c.integrand_fn,
                var_map,
                params,
                functions,
                learnable_params=learnable_params,
                learnable_functions=learnable_functions,
            ),
            target_fn=c.target_fn,
            integrand_scale=iscale,
        )
    if c.kind == "reduce_residual":
        if field_to_rhs is None or c.field not in field_to_rhs:
            raise ValueError(
                f"reduce_residual constraint {c.name!r} references field "
                f"{c.field!r} but no equation writes that field"
            )
        return ReduceResidualConstraint(
            name=c.name,
            closes=closes,
            field=var_map[c.field],
            reduce_fn=c.reduce,
            target_fn=c.target_fn,
            integrand_scale=iscale,
        )
    if c.kind == "boundary_integral":
        # ``field`` may be None (galvanostatic(boundary=) router): the integrand
        # reads fields via ``q`` directly and the boundary measure is resolved
        # from the field's owning domain (or the sole component) at lower time.
        # Bind a representative Variable so the IR node stays well-formed.
        if c.field is not None:
            field_var = var_map[c.field]
        else:
            field_var = _representative_variable(var_map)
        return BoundaryIntegralConstraint(
            name=c.name,
            closes=closes,
            field=field_var,
            tag=c.tag,
            selector=c.selector,
            integrand_fn=_make_q_adapter(
                c.integrand_fn,
                var_map,
                params,
                functions,
                learnable_params=learnable_params,
                learnable_functions=learnable_functions,
            ),
            target_fn=c.target_fn,
            integrand_scale=iscale,
        )
    if c.kind == "flux_total":
        return FluxTotalConstraint(
            name=c.name,
            closes=closes,
            from_bc=tuple(c.from_bc or ()),
            target_fn=c.target_fn,
            integrand_scale=iscale,
        )
    raise ValueError(f"unknown constraint kind {c.kind!r}")


def _representative_variable(var_map: dict[str, Any]):
    """Return any spatial [Variable][] from ``var_map``.

    For a boundary constraint that names no specific field (the integrand reads
    via ``q``).
    """
    for v in var_map.values():
        if isinstance(v, Variable):
            return v
    raise ValueError(
        "galvanostatic(boundary=...): no spatial field declared to anchor the boundary constraint"
    )


class _ObserveTraceCtx:
    """The evaluation context an observation sees while it is traced at build."""

    __slots__ = ("controls", "ops")

    def __init__(self, ops: Any) -> None:
        self.ops = ops
        self.controls = None


class _RecordingView:
    """Zero state keyed by a var map's handles; records each handle read.

    ``entries`` maps ``id(handle)`` to ``(read_record, zero_value)``; the
    handles are the [Variable][] / [GlobalVariable][] objects of the var map
    the observation's ``q`` resolves through.
    """

    def __init__(self, entries: dict[int, tuple[tuple[str, ...], Any]], reads: set) -> None:
        self._entries = entries
        self._reads = reads

    def __getitem__(self, key):
        entry = self._entries.get(id(key))
        if entry is None:
            raise KeyError(key)
        record, value = entry
        self._reads.add(record)
        return value

    def __contains__(self, key) -> bool:
        return id(key) in self._entries


class _RecordingDict(dict):
    """A name-keyed mapping that records every name looked up by subscript."""

    def __init__(self, values: dict[str, Any], kind: str, reads: set) -> None:
        super().__init__(values)
        self._kind = kind
        self._reads = reads

    def __getitem__(self, name):
        value = super().__getitem__(name)
        self._reads.add((self._kind, name))
        return value


class _RecordingNames:
    """A parameter or function namespace whose ``as_dict`` records lookups."""

    def __init__(self, source: Any, kind: str, reads: set) -> None:
        values = source.as_dict() if hasattr(source, "as_dict") else dict(source or {})
        self._dict = _RecordingDict(values, kind, reads)
        self.pointwise_names = tuple(getattr(source, "pointwise_names", ()))

    def as_dict(self) -> dict[str, Any]:
        return self._dict


def _aliases_by_component(spec) -> dict[str, dict[str, Any]]:
    """The read-through aliases of each component, ``{component: {alias: Variable}}``."""
    out: dict[str, dict[str, Any]] = {}
    for (comp_name, alias_name), alias_var in getattr(spec, "_read_aliases", {}).items():
        out.setdefault(comp_name, {})[alias_name] = alias_var
    return out


def _component_var_map(
    var_map: dict[str, Any], aliases_by_comp: dict[str, dict[str, Any]], dname: str
) -> dict[str, Any]:
    """``var_map`` overlaid with the read-through aliases of component ``dname``."""
    comp_aliases = aliases_by_comp.get(dname)
    if not comp_aliases:
        return var_map
    merged = dict(var_map)
    merged.update(comp_aliases)
    return merged


def _quote_names(names: Iterable[str]) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b' and 'c'``."""
    quoted = [repr(n) for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


class _ObservationTracer:
    """Trace observations at build to find their owner, read set and output.

    One instance serves one build: it holds the per-component var maps and
    caches each component's operator bank, so every observation is traced
    against the same zero state.
    """

    def __init__(self, spec, var_map: dict[str, Any], params: Any, functions: Any) -> None:
        self.spec = spec
        self.var_map = var_map
        self.params = params
        self.functions = functions
        self.aliases_by_comp = _aliases_by_component(spec)
        with_fields = set(spec._field_domains.values())
        self.components = tuple(d for d in spec._domains if d in with_fields)
        self.state_dtype = jnp.zeros(()).dtype
        self._banks: dict[str, Any] = {}

    def bank(self, dname: str):
        if dname not in self._banks:
            from diff_soup._lowering.operator_resolve import resolve_operator

            factory = getattr(self.spec, "_operator_factories", {}).get(dname)
            self._banks[dname] = resolve_operator(self.spec._domains[dname], factory)
        return self._banks[dname]

    def comp_var_map(self, dname: str) -> dict[str, Any]:
        return _component_var_map(self.var_map, self.aliases_by_comp, dname)

    def _entries(self, dname: str, comp_var_map: dict[str, Any]) -> dict[int, tuple]:
        comp_aliases = self.aliases_by_comp.get(dname, {})
        field_domains = self.spec._field_domains
        entries: dict[int, tuple] = {}
        for name, handle in comp_var_map.items():
            if comp_aliases.get(name) is handle:
                record = ("alias", dname, name)
                shape = tuple(handle.shape)
            elif isinstance(handle, GlobalVariable):
                record = ("global", name)
                size = int(np.size(handle.init))
                shape = () if size == 1 else (size,)
            elif name in field_domains:
                record = ("field", field_domains[name], name)
                shape = tuple(handle.shape)
            else:
                continue
            entries[id(handle)] = (record, jnp.zeros(shape, self.state_dtype))
        return entries

    def trace(self, o: ObservationSpec, dname: Optional[str]):
        """Trace ``o`` with component ``dname``'s var map and bank.

        Returns ``(reads, out, error)``: the read records, the traced output
        (``None`` on failure) and the exception the trace raised, if any.
        """
        comp_var_map = self.var_map if dname is None else self.comp_var_map(dname)
        entries = self._entries(dname or "", comp_var_map)
        reads: set = set()
        view = _RecordingView(entries, reads)
        params = _RecordingNames(self.params, "param", reads)
        functions = _RecordingNames(self.functions, "function", reads)
        state_dtype = self.state_dtype
        try:
            ops = None if dname is None else self.bank(dname)
        except Exception as exc:  # noqa: BLE001 - reported by the caller as the trace failure.
            return reads, None, exc

        def run():
            t = jnp.zeros((), state_dtype)
            q = _make_q_view(view, _ObserveTraceCtx(ops), t, comp_var_map, params, functions)
            return o.fn(q)

        try:
            out = jax.eval_shape(run)
        except Exception as exc:  # noqa: BLE001 - reported by the caller as an authoring error.
            return reads, None, exc
        return reads, out, None

    def _field_owners(self, reads: set) -> list[str]:
        return sorted({r[1] for r in reads if r[0] in ("field", "alias") and r[1]})

    def resolve(self, o: ObservationSpec) -> tuple[str, tuple, tuple[int, ...], str]:
        """Return ``(owner, inputs, out_shape, out_dtype)`` for ``o``."""
        declared = tuple(self.spec._domains)
        if o.domain is not None:
            if o.domain not in self.spec._domains:
                raise ConfigurationError(
                    f"observation {o.name!r} names domain {o.domain!r}, which is not "
                    f"declared; domain= names one of the declared domains {list(declared)!r}"
                )
            if o.domain not in self.components:
                raise ConfigurationError(
                    f"observation {o.name!r} names domain {o.domain!r}, which carries "
                    "no fields; domain= names a domain that carries fields: "
                    f"{list(self.components)!r}"
                )
            owner = o.domain
            reads, out, error = self.trace(o, owner)
        else:
            first = self.components[0] if self.components else None
            reads, out, error = self.trace(o, first)
            owners = self._field_owners(reads)
            if len(owners) >= 2:
                raise ConfigurationError(
                    f"observation {o.name!r} reads fields of the components "
                    f"{_quote_names(owners)}; declare the domain it evaluates on "
                    f"with spec.observe.{o.kind}(..., domain=<name>)"
                )
            owner = owners[0] if owners else first
            if owner != first:
                reads, out, error = self.trace(o, owner)
        if error is not None:
            raise ConfigurationError(
                f"observation {o.name!r} could not be evaluated at build: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise ConfigurationError(
                f"observation {o.name!r} could not be evaluated at build: "
                f"TypeError: it returns {type(out).__name__}; an observation "
                "returns a single array"
            )
        aliases = sorted(r[2] for r in reads if r[0] == "alias")
        if aliases:
            raise ConfigurationError(
                f"observation {o.name!r} reads the read-through alias "
                f"{_quote_names(aliases)} of component {owner!r}; the values an "
                "observation reads are the declared fields and globals, so observe "
                "the alias's source field instead"
            )
        inputs = tuple(sorted(reads))
        return owner or "", inputs, tuple(int(n) for n in out.shape), str(out.dtype)


class _MeanObservationFn(eqx.Module):
    """Average an observation's nodal values with the weights of its domain.

    Without ``nodes`` the weights are the lumped mass of the evaluating bank
    (cell volumes on a grid, ``int N_i`` on a mesh); with ``nodes`` they are
    the boundary measure resolved at build, one weight per selected node. The
    weights are normalized, so a uniform geometry scale cancels.
    """

    inner: object
    nodes: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)
    weights: Optional[tuple[float, ...]] = eqx.field(static=True, default=None)

    def __call__(self, view, ctx, t):
        from diff_soup.domains.operators.dispatch import _WeakTerm

        values = jnp.asarray(self.inner(view, ctx, t))
        if self.nodes is None:
            weights = ctx.ops.lumped_mass()
            if isinstance(weights, _WeakTerm):
                weights = weights.value
        else:
            values = values[np.asarray(self.nodes)]
            weights = np.asarray(self.weights)
        weights = jnp.asarray(weights).astype(values.dtype)
        return jnp.sum(weights * values) / jnp.sum(weights)


def _mean_observation(o: ObservationSpec, fn, tracer: _ObservationTracer, owner: str, out_shape):
    """Wrap a ``spatial_mean`` or ``boundary_mean`` closure in its weighted mean."""
    domain = tracer.spec._domains.get(owner) if owner else None
    if domain is None:
        raise ConfigurationError(
            f"observation {o.name!r}: spec.observe.{o.kind} averages over the nodes of "
            "a domain that carries fields, and the recipe declares none"
        )
    n_nodes = int(domain.num_nodes)
    if tuple(out_shape) != (n_nodes,):
        raise ConfigurationError(
            f"observation {o.name!r}: spec.observe.{o.kind} averages one value per node, "
            f"so fn must return an array of length {n_nodes} on domain {owner!r}; it "
            f"returns shape {tuple(out_shape)}. Return the nodal values, or declare a "
            "precomputed value with spec.observe.scalar"
        )
    if o.kind == "spatial_mean":
        return _MeanObservationFn(inner=fn)
    from diff_soup._lowering.bcs import _boundary_integral_measure
    from diff_soup.domains.boundary_set import tags

    tag = (o.where or {}).get("tag", "")
    try:
        measure = _boundary_integral_measure(
            domain, tags(tag), constraint_desc=f"observation {o.name!r}"
        )
    except (ValueError, TypeError, KeyError) as error:
        raise ConfigurationError(
            f"observation {o.name!r}: the boundary tag {tag!r} does not resolve on "
            f"domain {owner!r}: {error}"
        ) from error
    nodes = tuple(int(i) for i in np.asarray(measure.tag_nodes))
    if not nodes:
        raise ConfigurationError(
            f"observation {o.name!r}: the boundary tag {tag!r} selects no node on domain {owner!r}"
        )
    weights = tuple(float(w) for w in np.asarray(measure.nodal_measure, dtype=np.float64))
    return _MeanObservationFn(inner=fn, nodes=nodes, weights=weights)


def _emit_observation(
    o: ObservationSpec,
    tracer: _ObservationTracer,
    params: Any,
    functions: Any,
    learnable_params: tuple[str, ...],
    learnable_functions: tuple[str, ...],
):
    """Lower one observation, recording its owner, read set and output.

    The observation is traced once under ``jax.eval_shape`` against zero state
    to find the component it evaluates on, the names it reads and the shape
    and dtype of its value; its closure is then wrapped with that component's
    var map.  A read-through alias is refused at build (the observe view holds
    the state and the bordered globals, not the coupled values), and members
    of a family population are not part of the var map an observation reads,
    so neither appears in the read set. A ``spatial_mean`` or
    ``boundary_mean`` closure is wrapped in its weighted mean and records a
    scalar output.

    Raises:
        ConfigurationError: If ``domain=`` names no declared domain with
            fields, the observation reads the fields of two or more components
            without ``domain=``, the trace fails or returns something other
            than a single array, a mean's closure returns other than one value
            per node, or a ``boundary_mean`` tag does not resolve.
    """
    owner, inputs, out_shape, out_dtype = tracer.resolve(o)
    comp_var_map = tracer.comp_var_map(owner) if owner else tracer.var_map
    fn = _make_q_adapter(
        o.fn,
        comp_var_map,
        params,
        functions,
        learnable_params=learnable_params,
        learnable_functions=learnable_functions,
    )
    if o.kind in ("spatial_mean", "boundary_mean"):
        fn = _mean_observation(o, fn, tracer, owner, out_shape)
        out_shape = ()
    record = {"owner": owner, "inputs": inputs, "out_shape": out_shape, "out_dtype": out_dtype}
    if o.kind == "scalar":
        return ScalarObservation(name=o.name, fn=fn, **record)
    if o.kind == "field":
        return FieldObservation(name=o.name, fn=fn, **record)
    if o.kind == "boundary_mean":
        tag = (o.where or {}).get("tag", "")
        return BoundaryMeanObservation(name=o.name, fn=fn, tag=tag, **record)
    if o.kind == "spatial_mean":
        return SpatialMeanObservation(name=o.name, fn=fn, **record)
    raise ValueError(f"unknown observation kind {o.kind!r}")


def _emit_control(ctl: ControlSpec, var_map: dict[str, Any]):
    # ``binds_to`` stores the global NAME (str), not the live GlobalVariable —
    # the GlobalVariable carries a jax.Array ``init`` and placing it behind the
    # static EquationSet.controls would trigger a JAX static-array warning.
    if ctl.kind == "direct":
        return DirectControl(name=ctl.name, fn=ctl.fn)
    if ctl.kind == "derived":
        return DerivedControl(name=ctl.name, fn=ctl.fn, binds_to=ctl.binds_to)
    raise ValueError(f"unknown control kind {ctl.kind!r}")


def _make_var_map(spec) -> dict[str, Any]:
    var_map: dict[str, Any] = {}
    for name, fspec in spec._fields.items():
        domain = spec._domains[spec._field_domains[name]]
        if fspec.shape is not None:
            shape = tuple(fspec.shape)
        else:
            shape = (int(getattr(domain, "num_nodes")),)
        var_map[name] = Variable(name=name, shape=shape, mass=fspec.mass)
    for name, gspec in spec._globals.items():
        var_map[name] = GlobalVariable(name=name, init=jnp.asarray(gspec.initial), mass=gspec.mass)
    return var_map


def eval_field_init(init: Callable, coords: Any, size: int, field_name: str):
    """Evaluate a ``field(init=...)`` closure over node coordinates → a y0 block.

    ``init`` takes one positional coordinate array per spatial dim
    (``*xs``; the convention ``init=lambda *xs: ...``).  For spatial fields the
    IC is materialised against the domain ``node_coordinates()`` rather than
    carried on the IR [Variable][] (which has no ``init`` slot — the IC
    stays decoupled).

    Args:
        init: The field's ``init`` callable, ``init(*xs)``.
        coords: ``(num_nodes, dim)`` node coordinates (``None`` if the domain
            exposes no ``node_coordinates()``).
        size: The field's expected flat DOF count.
        field_name: For the precise mismatch error.

    Returns:
        A flat ``(size,)`` y0 block for this field.

    Raises:
        ValueError: If the domain has no node coordinates, or the materialised
            block does not flatten to ``size``.
    """
    if coords is None:
        raise ValueError(
            f"field {field_name!r}: init= was given but the field's domain "
            "exposes no node_coordinates(); pass an explicit value to "
            "assemble_state instead"
        )
    dim = int(coords.shape[1])
    xs = [coords[:, d] for d in range(dim)]
    arr = jnp.asarray(init(*xs))
    flat = jnp.reshape(arr, (-1,))
    if flat.size != size:
        raise ValueError(
            f"field {field_name!r}: init= materialised to size {flat.size}, "
            f"but the field's flat size is {size}"
        )
    return flat


def _params_dict(spec) -> dict[str, Any]:
    return {
        name: jnp.asarray(pspec.value, dtype=float)
        if isinstance(pspec.value, (int, float))
        else pspec.value
        for name, pspec in spec._params.items()
    }


def _functions_dict(spec) -> dict[str, Any]:
    return {name: fspec.fn for name, fspec in spec._functions.items()}


def _learnable_param_names(spec) -> tuple[str, ...]:
    return tuple(name for name, pspec in spec._params.items() if pspec.learnable)


def _learnable_function_names(spec) -> tuple[str, ...]:
    return tuple(name for name, fspec in spec._functions.items() if fspec.learnable)


def _pointwise_function_names(spec) -> tuple[str, ...]:
    """Names declared ``spec.function(..., pointwise=True)`` — mapped per node."""
    return tuple(
        name for name, fspec in spec._functions.items() if getattr(fspec, "pointwise", False)
    )


def _probe_coefficient_state_dependence(rhs: Callable, view, ctx, reference) -> Optional[tuple]:
    """Which fields a form's derivative coefficients read, by value probing.

    Re-evaluates the equation closure at a concrete synthetic state, then once
    more per field with only that field perturbed, and compares the normalized
    coefficient values. Runs host-side, once, at structure-capture time.

    Args:
        rhs: The equation's ``(view, ctx, t)`` adapter (the unwrapped one).
        view: The owner-keyed view of the capturing call — used only for its
            variable keys and array shapes/dtypes.
        ctx: The capturing call's lowering context (static geometry).
        reference: The
            [DerivativeIR][diff_soup.problem._ir.derivative_ir.DerivativeIR]
            captured from the real call —
            probe results must match its per-field patterns.

    Returns:
        A sorted tuple of field names whose perturbation moves any coefficient
        value; ``()`` for state-invariant coefficients; ``None`` when the
        probe cannot decide (the lowering then declares conservative
        dependence on the whole state).
    """
    import jax

    from diff_soup.problem._ir.derivative_ir import normalize_expression
    from diff_soup.problem._spec.derivative_expr import DerivativeExpr

    if not hasattr(view, "items"):
        return None
    try:
        # The capturing call may run inside an active trace (the build's
        # sparsity probes); the probe's inputs are concrete, so escape to
        # eager evaluation for the whole comparison.
        with jax.ensure_compile_time_eval():
            base_view = {}
            for var, arr in view.items():
                shape = tuple(getattr(arr, "shape", ()))
                dtype = getattr(arr, "dtype", None)
                if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
                    return None
                size = int(np.prod(shape)) if shape else 1
                vals = jnp.linspace(0.75, 1.25, size).astype(dtype).reshape(shape)
                base_view[var] = vals
            t_probe = jnp.asarray(0.37)

            def coefficient_values(probe_view):
                expr = rhs(probe_view, ctx, t_probe)
                if not isinstance(expr, DerivativeExpr):
                    return None
                ir = normalize_expression(expr, out_size=int(expr.size))
                if not hasattr(ir, "terms"):
                    return None
                return {
                    term.in_field: (term.pattern, np.asarray(term.values())) for term in ir.terms
                }

            base = coefficient_values(base_view)
            if base is None or set(base) != {t.in_field for t in reference.terms}:
                return None
            deps = []
            for var in base_view:
                probe = dict(base_view)
                probe[var] = base_view[var] * 1.003 + 0.0017
                got = coefficient_values(probe)
                if got is None or set(got) != set(base):
                    return None
                moved = False
                for field, (pattern, values) in got.items():
                    ref_pattern, ref_values = base[field]
                    if not (
                        np.array_equal(pattern[0], ref_pattern[0])
                        and np.array_equal(pattern[1], ref_pattern[1])
                    ):
                        return None
                    if values.shape != ref_values.shape or not np.array_equal(values, ref_values):
                        moved = True
                if moved:
                    deps.append(str(getattr(var, "name", var)))
        return tuple(sorted(deps))
    except Exception:
        return None


class _FormCell:
    """Identity-hashed holder for one equation's captured derivative structure.

    Lives on the static side of the adapter module, so it must be hashable;
    the default object identity hash is correct here — the cell is created
    per adapter and filled in place by the build-time capture pass.
    """

    __slots__ = ("data",)

    def __init__(self):
        self.data: dict = {}


def _weight_by(action, weight):
    """Multiply a derivative action by one weight per node, shared by a vector field's components."""
    action = jnp.asarray(action)
    weight = jnp.asarray(weight)
    if action.size != weight.size:
        return (action.reshape(weight.size, -1) * weight[:, None]).reshape(action.shape)
    return action * weight.reshape(action.shape)


def _lumped_weight(ops):
    """The lumped mass of a mesh bank, one weight per node."""
    from diff_soup.domains.operators.dispatch import _WeakTerm

    domain = getattr(ops, "domain", None)
    if domain is not None and hasattr(domain, "lumped_mass"):
        weight = domain.lumped_mass()
    else:
        weight = ops.lumped_mass()
    if isinstance(weight, _WeakTerm):
        weight = weight.value
    return jnp.asarray(weight)


def _lumped_derivative_action(ctx, action):
    """Weight a derivative action by the lumped mass on a mesh.

    On a mesh the rows of a residual are integrated against the test
    functions, and the explicit form weights ``d(field)/dt`` by the lumped
    mass (the problem's mass diagonal). The derivative action gets the same
    weight and is marked as already integrated. On a grid it is unchanged.
    """
    ops = getattr(ctx, "ops", None)
    if ops is None or getattr(ops, "discretization_type", None) != "fem":
        return action
    return ops.as_weak(_weight_by(action, _lumped_weight(ops)))


def _has_cell_scale(expr) -> bool:
    """Whether a per-cell coefficient scales any derivative in ``expr``."""
    from diff_soup.problem._spec.derivative_expr import (
        DerivativeScale,
        DerivativeSum,
        NonlinearDerivativeExpr,
    )

    if isinstance(expr, DerivativeScale) and expr.per_cell:
        return True
    if isinstance(expr, DerivativeSum):
        return any(_has_cell_scale(term) for term in expr.terms)
    if isinstance(expr, NonlinearDerivativeExpr):
        return any(_has_cell_scale(operand) for operand in expr.operands)
    inner = getattr(expr, "expr", None)
    return inner is not None and _has_cell_scale(inner)


def _split_cell_scale(term):
    """Split ``term`` into the expression without its per-cell scales and their product.

    A per-cell coefficient must scale a field's derivative directly, possibly
    through other scales; anywhere else it is refused.
    """
    from diff_soup.problem._spec.derivative_expr import DerivativeScale

    if isinstance(term, DerivativeScale):
        inner, cell = _split_cell_scale(term.expr)
        if term.per_cell:
            return inner, term.coeff if cell is None else cell * jnp.asarray(term.coeff)
        return (term if inner is term.expr else DerivativeScale(inner, term.coeff)), cell
    if _has_cell_scale(term):
        raise ValueError(
            "a per-cell coefficient must scale q.dt(field) directly, as in "
            "rho_c * q.dt(field), not a gathered, reduced, scattered, applied or "
            "nonlinear derivative expression"
        )
    return term, None


def _cell_terms(expr, ctx):
    """The terms of ``expr`` as ``(term without per-cell scales, coefficient per cell)``.

    Each per-cell coefficient is read against the mesh of the evaluating bank,
    which refuses another length than the number of cells; on a grid any
    length other than the field's is refused.
    """
    from diff_soup.domains.operators.dispatch import _read_fem_operand
    from diff_soup.problem._spec.derivative_expr import DerivativeSum, scale_size_error

    terms = expr.terms if isinstance(expr, DerivativeSum) else (expr,)
    ops = getattr(ctx, "ops", None)
    on_mesh = ops is not None and getattr(ops, "discretization_type", None) == "fem"
    out = []
    for term in terms:
        inner, cell = _split_cell_scale(term)
        if cell is not None:
            if not on_mesh:
                raise scale_size_error(int(np.shape(cell)[0]), int(inner.size))
            length = int(np.shape(cell)[0])
            cell, layout = _read_fem_operand(ops.domain, cell, "a per-cell coefficient of q.dt")
            if layout != "quad":
                raise scale_size_error(length, int(inner.size))
        out.append((inner, cell))
    return out


def _check_cell_scales(expr, ctx) -> None:
    """Refuse a per-cell coefficient of ``expr`` that its domain cannot read."""
    if _has_cell_scale(expr):
        _cell_terms(expr, ctx)


def _cell_weighted_parts(expr, slopes: dict, ctx, include: bool):
    """``(derivative action, residual part)`` of an expression with per-cell coefficients.

    A term scaled per cell by ``c`` is weighted by ``int c N_i``, the lumped
    mass with the coefficient inside the integral; the other terms take the
    lumped mass.
    """
    from diff_soup.domains.operators.dispatch import dispatch_integrate_scalar_at_quad
    from diff_soup.problem._ir.derivative_ir import evaluate_parts, sum_residual_terms
    from diff_soup.problem._spec.derivative_expr import DerivativeSum

    ops = ctx.ops
    size = int(expr.size)
    action = None
    for inner, cell in _cell_terms(expr, ctx):
        value, _ = evaluate_parts(inner, slopes, size, include_residual=False)
        if cell is None:
            weight = _lumped_weight(ops)
        else:
            weight = dispatch_integrate_scalar_at_quad(ops.domain, cell)
        weighted = _weight_by(value, weight)
        action = weighted if action is None else action + weighted
    residual = None
    if include and isinstance(expr, DerivativeSum):
        residual = sum_residual_terms(expr.residual_terms, size)
    return ops.as_weak(action), residual


def _slope_evaluation(expr, slopes: dict, ctx):
    """Evaluate a derivative expression inside a slope context.

    Returns the derivative action alone, or the full residual rows when the
    context includes the residual part. On a mesh the derivative action is
    weighted by the lumped mass (see ``_lumped_derivative_action``), and a term
    with a per-cell coefficient by the mass integral of that coefficient.
    """
    from diff_soup.problem._ir.derivative_ir import evaluate_parts
    from diff_soup.problem._spec.derivative_expr import slope_context_includes_residual

    include = slope_context_includes_residual()
    if _has_cell_scale(expr):
        action, residual = _cell_weighted_parts(expr, slopes, ctx, include)
    else:
        action, residual = evaluate_parts(expr, slopes, int(expr.size), include_residual=include)
        action = _lumped_derivative_action(ctx, action)
    if residual is None:
        return action
    return action + residual


class _ResidualFormAdapter(eqx.Module):
    """Adapt a residual-form derivative-expression return onto the rhs seam.

    A closure returning a derivative expression states the residual
    ``expression == 0``. Outside a slope context the adapter emits the
    NEGATED derivative-free part, so the ordinary rhs channel keeps the
    reading ``residual = derivative_part - rhs``. Inside a slope context
    (installed by the solver-problem lowering) the adapter evaluates the
    expression against the provided slopes — the full residual rows by
    default, or the derivative action alone when the context excludes the
    residual part.

    During the build-time capture pass (and only then, see
    [capture_forms][diff_soup.problem._ir.derivative_ir.capture_forms]) the
    adapter records the expression's static structure
    (``_derivative_form``): the written field, each contributing field's
    exact coefficient pattern, and which fields the coefficients read
    (``state_deps``, probed by value; ``None`` when undecidable).
    Expressions that are nonlinear in the state derivative are recorded as
    such; they lower through the general implicit form instead.

    An ``eqx.Module`` (not a plain closure) for the same reason as
    [_TracedQAdapter][diff_soup.problem._spec.build._TracedQAdapter]: the
    wrapped rhs sits in a non-static field, so
    a learnable-parameter module inside it keeps its array leaves visible to
    ``eqx.partition`` and gradient transformations.
    """

    inner: object
    equation_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    cell: _FormCell = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    @property
    def _derivative_form(self) -> dict:
        """The captured structure cell (empty until the build-time capture pass)."""
        return self.cell.data

    def __call__(self, view, ctx, t):
        from diff_soup.problem._ir.derivative_ir import (
            ImplicitResidualIR,
            evaluate_expression,
            normalize_expression,
            record_form,
            record_plain,
            zero_slopes_for,
        )
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr, current_slopes

        result = self.inner(view, ctx, t)
        form_cell = self.cell.data
        if not isinstance(result, DerivativeExpr):
            record_plain(form_cell)
            return result
        slopes = current_slopes()
        if slopes is not None:
            return _slope_evaluation(result, slopes, ctx)
        _check_cell_scales(result, ctx)
        ir = normalize_expression(result, out_size=int(result.size))
        record_form(
            form_cell,
            self.field_name,
            ir,
            state_deps=lambda: _probe_coefficient_state_dependence(self.inner, view, ctx, ir),
        )
        if isinstance(ir, ImplicitResidualIR):
            # No slope context: contribute the residual at zero slopes (the
            # derivative-free evaluation the rhs channel expects).
            return evaluate_expression(result, zero_slopes_for(result), int(result.size))
        return -ir.residual_part()


def _wrap_derivative_return(rhs: Callable, *, equation_name: str, field_name: str) -> Callable:
    """Wrap an equation rhs in the residual-form adapter (see the class)."""
    return _ResidualFormAdapter(
        inner=rhs,
        equation_name=equation_name,
        field_name=field_name,
        cell=_FormCell(),
    )


class _BundledReturnGuard(eqx.Module):
    """Reject derivative expressions inside a bundled multi-output return.

    An ``eqx.Module`` so the wrapped multi-output rhs keeps any learnable
    module leaves visible to ``eqx.partition`` (same rationale as
    [_ResidualFormAdapter][diff_soup.problem._spec.build._ResidualFormAdapter]).
    """

    inner: object
    equation_name: str = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    def __call__(self, view, ctx, t):
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr

        result = self.inner(view, ctx, t)
        if isinstance(result, dict) and any(isinstance(v, DerivativeExpr) for v in result.values()):
            raise NotImplementedError(
                "derivative expressions are not supported inside a bundled "
                f"multi-output equation ({self.equation_name!r}); author "
                "residual-form rows as single-output equations."
            )
        return result


def _reject_derivative_returns_in_bundle(rhs: Callable, equation_name: str) -> Callable:
    """Wrap a bundled multi-output rhs in the derivative-return guard."""
    return _BundledReturnGuard(inner=rhs, equation_name=equation_name)


class _NeumannCondition(eqx.Module):
    """One ``spec.bc.neumann`` declaration, resolved to a grid face.

    ``positions`` are the selected cells' positions in the face (the grid shape
    with ``axis`` removed, row-major), or ``None`` for the whole face.
    ``value_fn`` takes dimensional time and returns a scalar or one value per
    selected cell.
    """

    field: str = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    side: str = eqx.field(static=True)
    face_shape: tuple[int, ...] = eqx.field(static=True)
    positions: Optional[tuple[int, ...]] = eqx.field(static=True)
    time_ref: float = eqx.field(static=True)
    value_fn: Callable

    def evaluate(self, t):
        """The operator-level condition at internal time ``t``."""
        from diff_soup.domains.operators.flux_bc import _NormalDerivativeBC

        value = jnp.asarray(self.value_fn(t * self.time_ref))
        size = int(np.prod(self.face_shape))
        if self.positions is None:
            value = value if value.ndim == 0 else value.reshape(self.face_shape)
            return _NormalDerivativeBC(axis=self.axis, side=self.side, value=value)
        where = jnp.asarray(self.positions)
        face = jnp.zeros((size,), dtype=value.dtype).at[where].set(value)
        mask = jnp.zeros((size,), dtype=value.dtype).at[where].set(1.0)
        return _NormalDerivativeBC(
            axis=self.axis,
            side=self.side,
            value=face.reshape(self.face_shape),
            mask=mask.reshape(self.face_shape),
        )


class _MeshNeumannCondition(eqx.Module):
    """One ``spec.bc.neumann`` declaration on a tagged mesh, resolved to its facets.

    ``facets`` are the selected rows of ``mesh.boundary_facets()``; ``value_fn``
    takes dimensional time and returns a scalar or one value per selected facet.
    """

    field: str = eqx.field(static=True)
    facets: tuple[int, ...] = eqx.field(static=True)
    time_ref: float = eqx.field(static=True)
    value_fn: Callable

    def evaluate(self, t):
        """The operator-level condition at internal time ``t``."""
        from diff_soup.domains.operators.flux_bc import _MeshNormalDerivative

        value = jnp.asarray(self.value_fn(t * self.time_ref))
        return _MeshNormalDerivative(facets=self.facets, value=value)


class _OpsContext:
    """The residual context with ``ops`` replaced; every other attribute forwards."""

    __slots__ = ("_ctx", "ops")

    def __init__(self, ctx, ops) -> None:
        self._ctx = ctx
        self.ops = ops

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ctx"), name)


class _NeumannRhs(eqx.Module):
    """Carry a component's ``spec.bc.neumann`` conditions into its equations' operators.

    Each evaluation hands the wrapped rhs a ``q.ops`` whose ``laplacian`` and
    ``div_flux``, applied to a conditioned field's own values, impose its outward
    normal derivative: through a ghost value on a grid, as the natural boundary
    term ``int D s N_i dS`` on a mesh. The operator supplies its own
    diffusivity, so the condition holds for any ``D``, ``D(u)`` included.
    """

    inner: object
    fields: dict
    conditions: tuple

    def __call__(self, view, ctx, t):
        from diff_soup.domains.operators.base import _FieldConditionView

        targets = []
        for name, variable in self.fields.items():
            conditions = tuple(c.evaluate(t) for c in self.conditions if c.field == name)
            targets.append((view[variable], conditions))
        return self.inner(view, _OpsContext(ctx, _FieldConditionView(ctx.ops, targets)), t)


def _neumann_conditions(spec, var_map) -> dict[str, tuple]:
    """Resolve every ``spec.bc.neumann`` to its grid face or mesh facets, grouped by domain name."""
    from diff_soup._lowering.bcs import (
        _grid_face_cells,
        _grid_face_of,
        _grid_shape,
        _resolve_facet_mask,
        _resolve_selector,
    )

    time_ref = 1.0 if spec._time_scale is None else float(spec._time_scale)
    out: dict[str, list] = {}
    for bc in spec._bcs:
        if bc.kind != "neumann" or bc.var not in spec._fields:
            continue
        dname = spec._field_domains[bc.var]
        domain = spec._domains[dname]
        desc = f"spec.bc.neumann({bc.var!r})"
        value_fn = bc.value_fn if bc.value_fn is not None else _constant_fn(bc.value)
        if hasattr(domain, "boundary_facets") and bc.where.get("selector") is not None:
            mask = _resolve_facet_mask(domain, bc.where["selector"], bc_desc=desc)
            out.setdefault(dname, []).append(
                _MeshNeumannCondition(
                    field=bc.var,
                    facets=tuple(int(i) for i in np.flatnonzero(mask)),
                    time_ref=time_ref,
                    value_fn=value_fn,
                )
            )
            continue
        if getattr(domain, "discretization_type", None) not in ("fvm", "fdm") or not hasattr(
            domain, "cell_volumes"
        ):
            raise ValueError(
                f"{desc}: the normal derivative is imposed on structured grids and tagged "
                f"meshes, and {bc.var!r} lives on a {type(domain).__name__}. State the flux "
                "density D du/dn with spec.bc.flux."
            )
        if bc.where.get("face") is not None:
            face = tuple(bc.where["face"])
            cells = _grid_face_cells(domain, *face)
        else:
            cells = _resolve_selector(domain, bc.where["selector"], want="nodes")
            face = _grid_face_of(domain, cells, bc_desc=desc)
        axis, side = face
        face = _grid_face_of(domain, cells, face=face, bc_desc=desc)
        shape = _grid_shape(domain)
        face_shape = tuple(n for k, n in enumerate(shape) if k != axis)
        multi = np.unravel_index(np.asarray(cells, dtype=np.int64), shape)
        others = tuple(m for k, m in enumerate(multi) if k != axis)
        positions = np.ravel_multi_index(others, face_shape) if face_shape else np.zeros(1, int)
        whole = len(cells) == int(np.prod(face_shape))
        out.setdefault(dname, []).append(
            _NeumannCondition(
                field=bc.var,
                axis=int(axis),
                side=side,
                face_shape=face_shape,
                positions=None if whole else tuple(int(p) for p in positions),
                time_ref=time_ref,
                value_fn=value_fn,
            )
        )
    return {dname: tuple(conditions) for dname, conditions in out.items()}


def _build_components(
    spec,
    var_map: dict[str, Any],
    params: Any,
    functions: Any,
    learnable_params: tuple[str, ...],
    learnable_functions: tuple[str, ...],
) -> tuple[tuple[Component, ...], dict[str, Callable]]:
    """Build IR components and a ``{field_name -> rhs adapter}`` map.

    The map is consumed by
    [_emit_constraint][diff_soup.problem._spec.build._emit_constraint] so a
    ``reduce_residual``
    constraint can re-use the field-equation's rhs closure as its integrand
    without round-tripping through the user.
    """
    by_domain: dict[str, dict[str, list[Any]]] = {}
    field_to_rhs: dict[str, Callable] = {}
    # Names of fields opted into the FEM strong-form quad-native ``Field`` path.
    # Captured by the equation adapters so ``q.<name>`` decays to a ``Field``
    # at residual time on FEM domains.  Empty for specs with no ``quad=True``
    # field — the q.<name> path returns a plain array (the default behaviour).
    quad_fields = frozenset(fname for fname, f in spec._fields.items() if getattr(f, "quad", False))
    for dname in spec._domains:
        by_domain[dname] = {"vars": [], "eqs": [], "bcs": []}

    for fname, dname in spec._field_domains.items():
        by_domain[dname]["vars"].append(var_map[fname])

    # Per-component read-through alias map: ``couple(..., as_=<alias>)``
    # registers a NO-DOF alias keyed ``(consuming_component, alias_name)``.
    # The consuming closure reads it by name (``q.<alias>``) and the alias
    # Variable must appear in that equation's ``inputs`` so lowering resolves
    # the owner and the cross-block sparsity permits the read.  Aliases are
    # component-local, so build a fresh per-component var_map that overlays
    # this component's aliases (a component cannot see a sibling's same-named
    # alias).
    aliases_by_comp = _aliases_by_component(spec)

    def _comp_var_map(dname: str) -> dict[str, Any]:
        return _component_var_map(var_map, aliases_by_comp, dname)

    neumann = _neumann_conditions(spec, var_map)

    def _with_neumann(rhs, dname: str):
        conditions = neumann.get(dname)
        if not conditions:
            return rhs
        fields = {c.field: var_map[c.field] for c in conditions}
        return _NeumannRhs(inner=rhs, fields=fields, conditions=conditions)

    for eqspec in spec._equations:
        if eqspec.writes in spec._globals:
            # Differential-global equations lower to a bordered closing row,
            # not a component field equation (see _build_equation_set).
            continue
        if eqspec.writes not in spec._fields:
            raise ValueError(
                f"equation {eqspec.name!r} writes unknown name {eqspec.writes!r} "
                f"(known fields={sorted(spec._fields)}, globals={sorted(spec._globals)})"
            )
        dname = spec._field_domains[eqspec.writes]
        var = var_map[eqspec.writes]
        comp_aliases = aliases_by_comp.get(dname, {})
        comp_var_map = _comp_var_map(dname)
        # Translate `reads` strings into Variable handles, skipping globals
        # (lower() does not thread globals through `inputs`).  A read that names
        # a read-through alias of THIS component resolves to the alias Variable
        # so the consuming equation declares it in ``inputs``.
        inputs: list[Variable] = []
        for r in eqspec.reads:
            if r in comp_aliases:
                inputs.append(comp_aliases[r])
            elif r in spec._fields:
                inputs.append(var_map[r])
        rhs = _make_q_adapter(
            eqspec.fn,
            comp_var_map,
            params,
            functions,
            learnable_params=learnable_params,
            learnable_functions=learnable_functions,
            quad_fields=quad_fields,
        )
        rhs = _with_neumann(rhs, dname)
        rhs = _wrap_derivative_return(rhs, equation_name=eqspec.name, field_name=eqspec.writes)
        eq = Equation(var=var, rhs_fn=rhs, inputs=tuple(inputs), name=eqspec.name)
        by_domain[dname]["eqs"].append(eq)
        field_to_rhs[eqspec.writes] = rhs

    for meqspec in getattr(spec, "_multi_equations", []):
        # Validate every output is a registered field, all on the same domain.
        for w in meqspec.writes:
            if w not in spec._fields:
                raise ValueError(f"multi-equation {meqspec.name!r} writes unknown field {w!r}")
        dnames = {spec._field_domains[w] for w in meqspec.writes}
        if len(dnames) != 1:
            raise ValueError(
                f"multi-equation {meqspec.name!r} writes to fields on different "
                f"domains {sorted(dnames)!r}; multi-write requires a single domain"
            )
        dname = dnames.pop()
        # A bundled multi-output closure sees the same component-local alias
        # overlay as a single-output closure: alias reads resolve through the
        # component's var map and declare the alias Variable in ``inputs``.
        comp_aliases = aliases_by_comp.get(dname, {})
        comp_var_map = _comp_var_map(dname)
        out_vars = tuple(var_map[w] for w in meqspec.writes)
        inputs = []
        _input_names: set[str] = set()
        for r in meqspec.reads:
            if r in comp_aliases and r not in _input_names:
                inputs.append(comp_aliases[r])
                _input_names.add(r)
            elif r in spec._fields and r not in _input_names:
                inputs.append(var_map[r])
                _input_names.add(r)
        # A multi-write closure is a single bundled kernel: every output is
        # materialised together, so any output residual may depend on any
        # co-written sibling.  Union the co-written outputs into ``inputs`` so
        # the full writes-by-writes Jacobian block is declared even when the
        # user's ``reads`` omits a sibling.
        for w in meqspec.writes:
            if w not in _input_names:
                inputs.append(var_map[w])
                _input_names.add(w)
        rhs = _make_q_adapter(
            meqspec.fn,
            comp_var_map,
            params,
            functions,
            learnable_params=learnable_params,
            learnable_functions=learnable_functions,
            quad_fields=quad_fields,
        )
        rhs = _with_neumann(rhs, dname)
        rhs = _reject_derivative_returns_in_bundle(rhs, meqspec.name)
        eq = MultiEquation(vars=out_vars, rhs_fn=rhs, inputs=tuple(inputs), name=meqspec.name)
        by_domain[dname]["eqs"].append(eq)
        # Per-output extractor closures so reduce_residual constraints can query
        # a single field's residual without re-implementing the multi-closure.
        for w in meqspec.writes:
            field_to_rhs[w] = _make_multi_extractor(rhs, w)

    # Smart-couple domain-target auto-phantoms carry no user equation: emit a
    # zero-rhs algebraic Equation for each so the Component validator is
    # satisfied and the coupling SET/scatter-ADD lands on a base of 0.
    # Skip any phantom a user equation already writes (a user may legitimately
    # bind a closing equation to the deterministic name).
    written_fields = {eqspec.writes for eqspec in spec._equations}
    for meqspec in getattr(spec, "_multi_equations", []):
        written_fields.update(meqspec.writes)
    for phantom_name in getattr(spec, "_phantom_fields", {}):
        if phantom_name in written_fields:
            continue
        dname = spec._field_domains[phantom_name]
        ph_var = var_map[phantom_name]
        by_domain[dname]["eqs"].append(
            Equation(
                var=ph_var,
                rhs_fn=_phantom_self_rhs(ph_var),
                # The phantom carries its own ``(phantom, phantom)`` diagonal so
                # the algebraic row ``phantom = injected`` is well-posed even at
                # a target node that NO source writes (e.g. a cell node with no
                # co-located scatter member).  ``_phantom_self_rhs`` returns
                # ``-view[ph_var]`` (a real self-dependency, so the diagonal is
                # in the sparsity pattern); the coupling SET (gather) /
                # scatter-ADD then lands as ``r = -phantom + injected``.  Without
                # the self term an unvisited scatter-target row collapses to
                # ``0 = 0`` (singular), tripping a step-size collapse at ``t=0``.
                inputs=(ph_var,),
                name=f"phantom_{phantom_name}",
            )
        )

    for bcspec in spec._bcs:
        if bcspec.var not in spec._fields:
            raise ValueError(f"BC kind={bcspec.kind!r} references unknown field {bcspec.var!r}")
        dname = spec._field_domains[bcspec.var]
        domain = spec._domains[dname]
        if bcspec.kind == "periodic":
            axes = bcspec.axes if bcspec.axes is not None else tuple(range(_grid_dim(domain)))
            by_domain[dname]["bcs"].append(PeriodicBC(var=var_map[bcspec.var], axes=axes))
        elif bcspec.kind == "neumann":
            continue  # carried by the equations' operators (see _neumann_conditions)
        else:
            by_domain[dname]["bcs"].append(_emit_bc(bcspec, var_map, domain))

    components: list[Component] = []
    for dname, payload in by_domain.items():
        if not payload["vars"]:
            continue
        components.append(
            Component(
                name=dname,
                domain=spec._domains[dname],
                variables=tuple(payload["vars"]),
                equations=tuple(payload["eqs"]),
                bcs=tuple(payload["bcs"]),
                # Operator-bank factory selected via ``domain(..., operators=)``;
                # ``None`` (or an absent entry) selects the built-in bank.  The
                # lowering resolver constructs the bank from this component's
                # current domain.
                operator_factory=getattr(spec, "_operator_factories", {}).get(dname),
                # Scale parameter bound by ``spec.geometry``; ``None`` leaves the
                # declared grid as it is.
                geometry_scale=getattr(spec, "_geometry", {}).get(dname),
            )
        )
    return tuple(components), field_to_rhs


def _resolve_scales(
    spec,
    families: tuple[Any, ...] = (),
) -> tuple[Scales, dict[tuple[str, str], float], dict[str, float]]:
    """Run the host-time Scales / residual_scale resolution.

    Returns ``(scales, residual_scale, integrand_scale_by_name)``:

    * ``scales`` — a [Scales][diff_soup.problem._ir.scales.Scales] carrying
      ``state_ref`` / ``state_offset``
      / ``time_ref`` / ``integrand_ref`` keyed by IR keys
      (``(component, field)`` for fields, ``("", name)`` for globals,
      ``(family, member_var)`` for family members, constraint names for
      integrands).
    * ``residual_scale`` — the per-residual-row scale mapping
      ``(component, field) -> float``.
    * ``integrand_scale_by_name`` — ``constraint_name -> float`` so the
      constraint nodes are born with the resolved ``integrand_scale``
      (same value populates both this and ``integrand_ref``).

    Emits the conflict and un-scaled UserWarnings at the user's own call.

    ``families`` is the registration-order tuple of
    [Family][diff_soup.problem._ir.family.Family] IR nodes;
    step 6 propagates phantom scale/offset and step 7 applies per-family
    member scales.
    """
    fd = spec._field_domains

    # 1. inline accumulators
    state_ref: dict[tuple[str, str], float] = {}
    state_offset: dict[tuple[str, str], float] = {}
    integrand_ref: dict[str, float] = {}
    rscale: dict[tuple[str, str], float] = {}

    # 2. per-FieldSpec inline state / offset / residual_scale
    for fname, f in spec._fields.items():
        key = (fd[fname], fname)
        if f.scale is not None:
            state_ref[key] = float(f.scale)
        if f.offset is not None:
            state_offset[key] = float(f.offset)
        if f.residual_scale is not None:
            # Same guard spec.equation_scale() applies at declaration: the scale
            # divides the whole equation row, residual and differential mass
            # diagonal alike, so a zero divides both by zero.  An infinite mass
            # diagonal survives the all-zero degeneracy check downstream, so an
            # unguarded zero here surfaces as wrong numbers, not as an error.
            if float(f.residual_scale) == 0.0:
                raise ValueError(
                    f"spec.field({fname!r}, residual_scale=0.0): the residual scale "
                    "divides that field's equation row, so it must be non-zero. Pass "
                    "a non-zero scale, or leave residual_scale unset to leave the row "
                    "unscaled."
                )
            rscale[key] = float(f.residual_scale)

    # 2b. central equation_scale() overrides inline residual_scale
    for fname, v in spec._equation_scales.items():
        if fname in fd:
            rscale[(fd[fname], fname)] = float(v)

    # 3. per-GlobalVarSpec inline scale (global sentinel "")
    for gname, g in spec._globals.items():
        if g.scale is not None:
            state_ref[("", gname)] = float(g.scale)

    # 4. single time scale (default 1.0)
    time_ref = 1.0 if spec._time_scale is None else float(spec._time_scale)

    # 5. constraint integrand scales (from inline ConstraintSpec.integrand_scale)
    integrand_scale_by_name: dict[str, float] = {}
    for c in spec._constraints:
        if getattr(c, "integrand_scale", None) is not None:
            integrand_scale_by_name[c.name] = float(c.integrand_scale)

    # 6. Scatter-target fields: copy the source field's state reference and
    #    offset onto each target field that ``spec.couple`` created for a
    #    scatter coupling, so both scale alike.  The target's residual_scale
    #    stays 1.0 (not copied).  No-op when no scatter coupling exists.
    phantom_sources = getattr(spec, "_phantom_sources", {})
    for phantom_name, source_name in phantom_sources.items():
        if phantom_name not in fd or source_name not in fd:
            continue
        src_key = (fd[source_name], source_name)
        ph_key = (fd[phantom_name], phantom_name)
        if src_key in state_ref:
            state_ref[ph_key] = state_ref[src_key]
        if src_key in state_offset:
            state_offset[ph_key] = state_offset[src_key]

    # 7. PER-FAMILY MEMBER SCALES — family owner namespace key
    #    ``(family_name, member_var)``.  A family's ``scale`` /
    #    ``residual_scale`` are ``Mapping[member_var_name -> float]`` (not a
    #    single float): look up the per-member value and emit a key ONLY for
    #    member vars that carry a scale; skip members absent from the mapping.
    for fam in families:
        fam_scale = getattr(fam, "scale", None) or {}
        fam_rscale = getattr(fam, "residual_scale", None) or {}
        for mv in getattr(fam, "member_variables", ()):
            s = fam_scale.get(mv.name)
            if s is not None:
                state_ref[(fam.name, mv.name)] = float(s)
            rs = fam_rscale.get(mv.name)
            if rs is not None:
                rscale[(fam.name, mv.name)] = float(rs)

    # 8. central scales() — bare names resolved to IR keys.
    central_state: dict[tuple[str, str], float] = {}
    central_offset: dict[tuple[str, str], float] = {}
    fam_member_owner = {
        mv.name: fam.name for fam in families for mv in getattr(fam, "member_variables", ())
    }

    def _resolve_state_key(bare):
        if isinstance(bare, tuple):
            return bare
        owners = []
        if bare in fd:
            owners.append(fd[bare])
        if bare in spec._globals:
            owners.append("")
        if bare in fam_member_owner:
            owners.append(fam_member_owner[bare])
        if len(owners) > 1:
            raise ValueError(
                f"spec.scales: field {bare!r} is owned by {owners}; use an "
                "explicit (component, field) key to disambiguate"
            )
        if not owners:
            # Unknown name: leave a sentinel key so _validate_scales_keys
            # raises the precise unknown-key error at EquationSet.__init__.
            return ("", bare)
        return (owners[0], bare)

    for bare, v in spec._central_state.items():
        central_state[_resolve_state_key(bare)] = float(v)
    for bare, v in spec._central_state_offset.items():
        central_offset[_resolve_state_key(bare)] = float(v)

    # 9. MERGE central -> inline, warning on conflict (central wins).
    def _merge(inline, central, label_fn):
        for key, cval in central.items():
            if key in inline and inline[key] != cval:
                ival = inline[key]
                warn_user(
                    label_fn(key, cval, ival),
                    UserWarning,
                )
            inline[key] = cval

    def _conflict_label(key, c, i):
        return (
            f"spec.scales: central override of {key} = {c!r} differs from "
            f"inline scale {i!r}; central value wins"
        )

    _merge(state_ref, central_state, _conflict_label)
    _merge(state_offset, central_offset, _conflict_label)
    # central time
    if spec._central_time is not None:
        if spec._time_scale is not None and float(spec._time_scale) != float(spec._central_time):
            warn_user(
                f"spec.scales: central override of 'time' = "
                f"{float(spec._central_time)!r} differs from inline scale "
                f"{float(spec._time_scale)!r}; central value wins",
                UserWarning,
            )
        time_ref = float(spec._central_time)
    # central integrand: sets BOTH integrand_ref and the constraint node.
    central_integrand = {k: float(v) for k, v in spec._central_integrand.items()}
    _merge(integrand_scale_by_name, central_integrand, _conflict_label)
    # integrand_ref mirrors integrand_scale_by_name (step 5 + step 8).
    integrand_ref = dict(integrand_scale_by_name)

    # 10. UN-SCALED WARNING — one aggregated warning listing differential /
    #     algebraic fields whose scale was None.  Skip globals/params;
    #     field(scale=1.0) does NOT warn (explicit 1.0 != omitted None).
    unscaled = [
        fname
        for fname, f in spec._fields.items()
        if f.scale is None and (fd[fname], fname) not in state_ref
    ]
    if unscaled:
        names = ", ".join(repr(n) for n in unscaled)
        warn_user(
            f"spec.scales: fields {names} have no scale= and none was set "
            "centrally; defaulting to 1.0 (dimensional). Set scale= per field "
            "or via spec.scales(state=...) to non-dimensionalize.",
            UserWarning,
        )

    scales = Scales(
        state_ref=state_ref,
        state_offset=state_offset,
        time_ref=time_ref,
        integrand_ref=integrand_ref,
    )
    return scales, rscale, integrand_scale_by_name


def _build_equation_set(spec) -> EquationSet:
    from .validate import SpecValidationError, _validate_static

    # The static pass reads the declarations only — no mesh traversal, no
    # tracing — so it runs on every build.  It repeats the checks the
    # declaration verbs already make (a spec assembled by any other route is
    # held to them too) and adds the ones that only become visible once the
    # whole recipe is in hand: an equation reading a name nobody declared, two
    # equations answering to one name, a boundary condition off the end of its
    # domain, a global whose declared mass does not match the row that writes
    # it.
    structural_errors = _validate_static(spec)
    if structural_errors:
        raise SpecValidationError(spec.name, structural_errors)

    # Before any equation closure is traced: an equation reading a global that
    # no constraint closes would otherwise fail as a lookup miss inside a traced
    # residual, naming the symptom instead of the missing declaration.
    _check_closes_pairing(spec)

    var_map = _make_var_map(spec)
    params = _NamedValues(_params_dict(spec))
    functions = _NamedValues(_functions_dict(spec), _pointwise_function_names(spec))
    learnable_params = _learnable_param_names(spec)
    learnable_functions = _learnable_function_names(spec)

    # Frozenset of field names that decay to a quad-native Field on FEM
    # (spec.field(quad=True) or the FEM default quad=None->True).  Stored on the
    # EquationSet and threaded to LoweredProblem.
    quad_fields = frozenset(fname for fname, f in spec._fields.items() if getattr(f, "quad", False))

    components, field_to_rhs = _build_components(
        spec,
        var_map,
        params,
        functions,
        learnable_params,
        learnable_functions,
    )

    # Trigger every registered handle's lazy re-materialise here — the build walk
    # is the final reader of ``spec._families``.  A family registered directly
    # into ``_families`` (e.g. IR-level test fixtures) carries no handle and is
    # consumed as-is.  ``spec._families`` is a name-keyed dict (registration
    # order preserved); the IR carries the Family values in that order.
    for _name, _handle in getattr(spec, "_family_handles", {}).items():
        _handle._ensure_synced()
    _fam_registry = getattr(spec, "_families", {})
    families = (
        tuple(_fam_registry.values()) if isinstance(_fam_registry, dict) else tuple(_fam_registry)
    )

    # Scales / residual_scale resolution (host-time, before any IR node
    # construction so the single merged Scales rides EquationSet directly).
    # Steps 7 (phantom propagation) and 8 (per-family member scales) consume
    # the resolved ``families`` tuple.
    scales, residual_scale, integrand_scale_by_name = _resolve_scales(spec, families)

    globals_ = tuple(var_map[name] for name in spec._globals)
    field_to_domain = {fname: spec._domains[dname] for fname, dname in spec._field_domains.items()}
    constraints = tuple(
        _emit_operational_control(
            c,
            var_map,
            params,
            functions,
            learnable_params,
            learnable_functions,
            spec._observations,
            field_to_rhs,
        )
        if isinstance(c, OperationalControlSpec)
        else _emit_constraint(
            c,
            var_map,
            params,
            functions,
            learnable_params,
            learnable_functions,
            field_to_rhs,
            integrand_scale_by_name,
            field_to_domain,
        )
        for c in spec._constraints
    )
    global_equation_rows = tuple(
        DifferentialGlobalEquation(
            name=eqspec.name,
            closes=var_map[eqspec.writes],
            rhs_fn=_make_q_adapter(
                eqspec.fn,
                var_map,
                params,
                functions,
                learnable_params=learnable_params,
                learnable_functions=learnable_functions,
            ),
            reads=tuple(eqspec.reads),
        )
        for eqspec in spec._equations
        if eqspec.writes in spec._globals
    )
    if global_equation_rows or constraints:
        # One closing row per global, in globals declaration order — the lambda
        # layout every downstream walk (layout entries, initial lambda, residual
        # row assembly) reads off the constraints tuple.
        closing_by_global = {node.closes.name: node for node in constraints + global_equation_rows}
        constraints = tuple(closing_by_global[name] for name in spec._globals)
    tracer = _ObservationTracer(spec, var_map, params, functions)
    observations = tuple(
        _emit_observation(
            o,
            tracer,
            params,
            functions,
            learnable_params,
            learnable_functions,
        )
        for o in spec._observations
    )
    controls = tuple(_emit_control(ctl, var_map) for ctl in spec._controls)

    couplings = tuple(_emit_coupling(cs, var_map) for cs in spec._couplings)

    eqset = EquationSet(
        components=components,
        globals=globals_,
        constraints=constraints,
        observations=observations,
        controls=controls,
        couplings=couplings,
        families=families,
        scales=scales,
        residual_scale=residual_scale,
        quad_fields=quad_fields,
    )

    return eqset
