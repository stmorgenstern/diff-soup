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
(consumed by the [diff_soup.problem._lowering][] provider). Specs with no coupling
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
import jax.numpy as jnp

from diff_soup._warnings import warn_user
from diff_soup.problem._ir import (
    EquationSet,
    GlobalVariable,
    Variable,
)
from diff_soup.problem._ir.constraint import (
    BoundaryIntegralConstraint,
    DifferentialGlobalEquation,
    DomainIntegralConstraint,
    FluxTotalConstraint,
    GalvanostaticConstraint,
    OperationalControlConstraint,
    ReduceResidualConstraint,
)
from diff_soup.problem._ir.control import DerivedControl, DirectControl
from diff_soup.problem._ir.coupling import Coupling
from diff_soup.problem._ir.scales import Scales
from diff_soup.problem._spec.build_components import _build_components
from diff_soup.problem._spec.observation_trace import _emit_observation, _ObservationTracer
from diff_soup.problem._spec.q_adapter import _make_q_adapter

from .specs import (
    ConstraintSpec,
    ControlSpec,
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
                "by exactly one constraint row; give one of the two its own "
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
            "is determined by the one constraint row that closes it; declare "
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
