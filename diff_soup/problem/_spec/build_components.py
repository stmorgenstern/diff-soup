"""The components of an equation set: their right-hand sides, boundary conditions and Neumann conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup.problem._ir import (
    BoundarySpec,
    Component,
    DirichletBC,
    Equation,
    FluxBC,
    MultiEquation,
    PeriodicBC,
    RobinBC,
    SymmetryBC,
    TractionBC,
    Variable,
    _BoundaryFn,
)
from diff_soup.problem._ir.bc import _constant_fn, _time_flux_fn
from diff_soup.problem._spec.observation_trace import _aliases_by_component, _component_var_map
from diff_soup.problem._spec.q_adapter import _make_q_adapter
from diff_soup.problem._spec.residual_forms import (
    _reject_derivative_returns_in_bundle,
    _wrap_derivative_return,
)

from .specs import (
    BCSpec,
)


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
        from diff_soup.problem._lowering.bcs import _grid_face_cells

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
        return FluxBC(
            var=var, where=where, flux_fn=_inflow_flux_fn(bc.velocity, bc.value, bc.value_fn)
        )
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
    from diff_soup.problem._lowering.bcs import (
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
