"""A spec's observations traced into the variables they read, and emitted."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.problem._ir import (
    GlobalVariable,
)
from diff_soup.problem._ir.observation import (
    BoundaryMeanObservation,
    FieldObservation,
    ScalarObservation,
    SpatialMeanObservation,
)
from diff_soup.problem._spec.q_adapter import _make_q_adapter, _make_q_view

from .specs import (
    ObservationSpec,
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
            from diff_soup.problem._lowering.operator_resolve import resolve_operator

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
    from diff_soup.domains.boundary_set import tags
    from diff_soup.problem._lowering.bcs import _boundary_integral_measure

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
