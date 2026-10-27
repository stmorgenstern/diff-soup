"""Batched family residual builder.

A [Family][diff_soup.problem._ir.family.Family] is a population of ``count``
structurally-identical members (e.g. MPM particles).  Assembling its residual
member-by-member in a Python loop re-traces the per-member kernel ``count``
times.  [_build_family_residual][diff_soup._lowering.family_residual._build_family_residual] instead traces the per-member kernel
**exactly once** and batches it with **one** ``jax.vmap``'d primitive:

1. ``u_mem = u[fam_base : fam_base+state_size].reshape(count, member_state_size)``
2. ``params_b = {k: v for k, v in fam.member_params.items()}`` — each leading
   axis ``count`` (array leaves, never closure-captured constants; capturing
   per-member constants as closures silently reintroduces N traces).
3. ONE member kernel ``single(u_row, m_row, t)``: unpacks ``u_row`` into a
   per-member [MergedView][diff_soup.problem._ir._merged_view.MergedView] keyed ``fam:{name}.{var}``; builds a
   ``_MemberView`` exposing ``m.<param>`` as a SCALAR tracer (+ ``m.member_index``);
   calls each member equation ``rhs_fn(member_view, ctx_member, m, t)`` (folding
   the member reaction flux / Neumann at r=R inside the same kernel); reads any
   galvanostatic global (``i_app``) through the merged-view global slot.
4. ``f_mem = jax.vmap(single, in_axes=(0, 0, None))(u_mem, params_b, t)`` — one
   trace, one vmap'd primitive.
5. ``f = f.at[fam_base : fam_base+state_size].add(f_mem.reshape(-1))`` — one scatter.
6. Divide member rows by ``residual_scale`` (scalar) if set.

``count=1`` keeps the length-1 leading axis (no special branch), so the emitted
XLA is shape-identical to ``count>1``.

The member kernel uses only the tagging ops (weak-form provenance) and
``finalize_residual``; no plain ``Operators`` handle is allocated.

This module lives under ``diff_soup/_lowering`` and imports both
``diff_soup.domains.operators`` and ``diff_soup.problem._ir`` IR nodes.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import LoweringError
from diff_soup.domains.operators.dispatch import finalize_residual
from diff_soup.problem._ir._merged_view import MergedView
from diff_soup.problem._ir.bc import FluxBC, _BoundaryFn
from diff_soup.problem._ir.equation import MultiEquation
from diff_soup.problem._ir.family import Family
from diff_soup.problem._ir.var_ref import VarRef
from diff_soup.problem._ir.variable import GlobalVariable
from diff_soup.problem._spec.derivative_expr import (
    current_slopes,
    slope_context,
    slope_context_includes_residual,
)

from .bcs import (
    _FaceView,
    _grid_face_factor,
    _grid_face_geometry,
    _resolve_grid_local_indices,
)
from .operator_resolve import resolve_operator

# Reserved ``params_b`` key carrying the per-member index.  Bundled into the
# batched param pytree so the SAME vmap axis exposes ``m.member_index`` without
# a second vmap'd primitive.  Underscore-prefixed so it cannot collide with a
# user ``member_params`` name.
_MEMBER_INDEX_KEY = "__member_index__"


class _MemberView:
    """Per-member parameter accessor handed to a member equation's ``rhs_fn``.

    Exposes each ``member_params`` entry as ``m.<param>`` — a SCALAR tracer
    under ``jax.vmap`` (the member's slice of the leading ``count`` axis) — plus
    ``m.member_index``, the member's position in the population.
    """

    __slots__ = ("_row", "member_index")

    def __init__(self, row: Mapping[str, jax.Array]):
        # ``row`` is the per-member dict produced by vmapping ``params_b``;
        # values are scalar tracers.  ``member_index`` is pulled out of the
        # reserved key so it is a first-class attribute, not a param lookup.
        # Plain slot assignment: this class defines only ``__getattr__`` (which
        # fires on MISSING attrs), so a normal set of a real slot is not
        # intercepted.
        self._row = row
        self.member_index = row.get(_MEMBER_INDEX_KEY, jnp.asarray(0))

    def __getattr__(self, name: str):
        # ``__getattr__`` only fires for names not found normally, so it never
        # shadows ``member_index`` / ``_row`` (both real slots).
        try:
            return object.__getattribute__(self, "_row")[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"member view has no parameter {name!r}; declared "
                f"member_params are {sorted(self._row)!r}"
            ) from exc

    def __getitem__(self, name: str):
        return self._row[name]


class _GraphNamespace:
    """Read-only dotted namespace used by graph endpoint and edge views."""

    __slots__ = ("_values", "_label")

    def __init__(self, values, label: str):
        self._values = values
        self._label = label

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        label = object.__getattribute__(self, "_label")
        raise AttributeError(
            f"{label} has no value {name!r}; available names are {sorted(values)!r}."
        )

    def __getitem__(self, name: str):
        return object.__getattribute__(self, "_values")[name]


class _GraphEndpointView:
    """One edge endpoint's declared member fields and parameter row."""

    __slots__ = ("_fields", "_all_fields", "_declaration", "p", "g", "member_index")

    def __init__(
        self,
        *,
        fields,
        all_fields,
        declaration: str,
        params,
        globals_,
        member_index,
    ):
        self._fields = fields
        self._all_fields = all_fields
        self._declaration = declaration
        self.p = _GraphNamespace(params, "family member parameters")
        self.g = _GraphNamespace(globals_, "family graph globals")
        self.member_index = member_index

    def __getattr__(self, name: str):
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        # Member parameters are available both as ``endpoint.p.rate`` and as
        # ``endpoint.rate`` when they do not collide with a field name.
        params = object.__getattribute__(self, "p")._values
        if name in params:
            return params[name]
        all_fields = object.__getattribute__(self, "_all_fields")
        if name in all_fields:
            declaration = object.__getattribute__(self, "_declaration")
            raise AttributeError(
                f"family graph endpoint field {name!r} is not declared; add "
                f"{name!r} to {declaration}."
            )
        raise AttributeError(
            f"family graph endpoint has no field or member parameter {name!r}; "
            f"member fields are {sorted(all_fields)!r}."
        )

    def __getitem__(self, name: str):
        return getattr(self, name)


class _GraphEdgeView:
    """Per-edge data passed to weight and exchange callbacks."""

    __slots__ = (
        "weight",
        "index",
        "source_index",
        "target_index",
        "t",
        "g",
        "graph",
        "directed",
    )

    def __init__(
        self,
        *,
        weight,
        index,
        source_index,
        target_index,
        t,
        globals_,
        graph: str,
        directed: bool,
    ):
        self.weight = weight
        self.index = index
        self.source_index = source_index
        self.target_index = target_index
        self.t = t
        self.g = _GraphNamespace(globals_, "family graph globals")
        self.graph = graph
        self.directed = directed


class _FamilyCtx:
    """Residual ctx for member equations — carries the member-domain ops."""

    def __init__(self, ops):
        self.ops = ops


def _check_member_multi_return(fam: Family, eq: MultiEquation, r) -> None:
    """Validate a multi-output member closure's return (dict keyed by outputs).

    A multi-output member equation must return a dict with one residual entry
    per written member field, keyed by the field names in ``writes``.  Runs at
    trace time (host-side structure only).

    Raises:
        TypeError: when the closure returns anything but a dict.
        ValueError: when a written field's entry is missing or the dict carries
            an extra key no output claims.
    """
    expected = {v.name for v in eq.vars}
    if not isinstance(r, dict):
        raise TypeError(
            f"family {fam.name!r} multi-output member equation {eq.name!r} must "
            f"return a dict keyed by its written fields {sorted(expected)!r}; "
            f"got {type(r).__name__}."
        )
    missing = sorted(expected - set(r))
    extra = sorted(set(r) - expected)
    if missing or extra:
        raise ValueError(
            f"family {fam.name!r} multi-output member equation {eq.name!r} "
            f"returned keys {sorted(r)!r}, but writes= declares "
            f"{sorted(expected)!r}"
            + (f"; missing {missing!r}" if missing else "")
            + (f"; unexpected {extra!r}" if extra else "")
            + "."
        )


def _validate_member_params(fam: Family) -> dict[str, jax.Array]:
    """Coerce + validate ``member_params`` into traced leaves (count-axis guard).

    Every value must be ``jnp.asarray``-able with leading axis ``== count`` (or a
    shared scalar broadcast to ``(count,)``).  A Python list of per-member
    closures / a per-member-captured constant is REJECTED with an error naming
    the param and pointing at the array-leaf requirement.

    Raises:
        TypeError: when a member-param value is not a ``jnp.asarray``-able leaf.
        ValueError: when a leaf's leading axis is neither ``count`` nor scalar.
    """
    count = int(fam.count)
    out: dict[str, jax.Array] = {}
    for name, value in fam.member_params.items():
        try:
            arr = jnp.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Family {fam.name!r} member_params[{name!r}] is not an array "
                f"leaf (got {type(value).__name__}); each member param must be a "
                f"jnp.asarray-able array with leading axis == count (={count}) "
                f"or a shared scalar — a per-member Python list/closure "
                f"silently causes one re-trace per member."
            ) from exc
        if arr.ndim == 0:
            arr = jnp.broadcast_to(arr, (count,))
        elif arr.shape[0] != count:
            raise ValueError(
                f"Family {fam.name!r} member_params[{name!r}] has leading axis "
                f"{arr.shape[0]} but count={count}; each member param array's "
                f"leading axis must equal count (or be a shared scalar)."
            )
        out[name] = arr
    return out


class _FamilyResidualBlock:
    """Callable family-residual block + introspection hooks for the smoke test.

    Calling the block scatters the batched member residual into ``f`` at
    ``[fam_base : fam_base + state_size]`` (one ``.at[].add``).  The
    ``member_residual`` method returns the pre-scatter ``(count, member_state)``
    batched residual, and ``trace_count`` reports how many times the per-member
    kernel ``single`` was traced (should be exactly 1 under jax.vmap).

    ``time_ref`` is how many dimensional seconds one internal time unit stands for.
    Member boundary fluxes are boundary callbacks, so they are authored on the
    experiment's own clock: the block converts the internal solve time to dimensional
    once before dispatching them.  The unscaled default (``1.0``) leaves the
    time untouched.
    """

    def __init__(self, fam: Family, fam_base: int, total_size: int, *, time_ref: float = 1.0):
        self._fam = fam
        self._fam_base = int(fam_base)
        self._total_size = int(total_size)
        self._count = int(fam.count)
        self._member_state = int(fam.member_state_size)
        self._time_ref = float(time_ref)
        self.trace_count = 0

        # Member-field local offsets within one member's flat slab.
        self._field_offsets: list[tuple[object, int, tuple[int, ...]]] = []
        loc = 0
        for v in fam.member_variables:
            self._field_offsets.append((v, loc, v.shape))
            loc += int(v.size)

        # Member-domain operator surfaces (the residual view gives weak-form FEM
        # outputs the ``integrated`` provenance; ``finalize_residual``
        # integrates leftover nodal terms once).  Resolved host-side once from the
        # family's operator factory (the built-in bank when none is given); the
        # member kernel only needs this residual view and ``finalize_residual``.
        self._tag_ops = resolve_operator(fam.domain, fam.operator_factory).for_residual()
        self._domain = fam.domain

        # Reserved family VarRefs (family=True) so closures indexing by VarRef
        # resolve, distinct from a structurally-identical component variable.
        self._refs = {v: VarRef(fam.name, v.name, family=True) for v in fam.member_variables}
        # Member fields by name, for the ``q`` accessor a member-authored
        # boundary flux reads (``q.c_s`` resolves through this map).
        self._member_var_map = {v.name: v for v in fam.member_variables}
        self._graph_field_info = {
            v.name: (loc, int(v.size), tuple(v.shape)) for v, loc, _shape in self._field_offsets
        }
        self._graph_by_name = {graph.name: graph for graph in getattr(fam, "graphs", ())}
        for graph in self._graph_by_name.values():
            if int(graph.member_count) != self._count:
                raise ValueError(
                    f"family graph {graph.name!r} declares member_count="
                    f"{int(graph.member_count)}, but family {fam.name!r} has "
                    f"count={self._count}; rebuild the graph for this family."
                )
        for exchange in getattr(fam, "graph_exchanges", ()):
            if exchange.graph not in self._graph_by_name:
                raise ValueError(
                    f"family {fam.name!r} graph exchange {exchange.name!r} names "
                    f"unknown graph {exchange.graph!r}; declare the graph on "
                    "the same family."
                )

        # Globals referenced by member equations — resolved by name at trace
        # time from the runtime ``member_globals`` dict.
        self._global_names: tuple[str, ...] = tuple(
            dict.fromkeys(
                inp.name
                for eq in fam.member_equations
                for inp in eq.inputs
                if isinstance(inp, GlobalVariable)
            )
        )
        self._global_by_name: dict[str, GlobalVariable] = {
            inp.name: inp
            for eq in fam.member_equations
            for inp in eq.inputs
            if isinstance(inp, GlobalVariable)
        }

        # Residual-row scale (scalar shared across the member axis).
        res_scale = fam.residual_scale or {}
        self._res_scale = {k: float(v) for k, v in dict(res_scale).items()}

        # Pre-resolve member FluxBC GEOMETRY to (var, local_node_indices,
        # face_factors, var_loc, flux_form, face_geometry) 6-tuples — geometry
        # is host-side static.  ``flux_form`` records which view the density was
        # authored against, so the member kernel calls each callable with the
        # signature its authoring surface documents:
        #
        # * ``"face"`` — ``parts.bc.flux(fn=...)``, called ``(face, t)`` with the
        #   boundary-restricted face view built from ``face_geometry``.
        # * ``"nodal"`` — a constant or time-only density, which arrives as the
        #   internal ``(view, ctx, t)`` adapter.
        #
        # The flux CALLABLE is intentionally NOT captured here; it is threaded at
        # runtime off the runtime family's ``member_flux_callables`` (a
        # non-static leaf so AD and vmap reach any learnable array fields inside
        # the callable).  Every member boundary condition either resolves here or
        # raises: the member kernel assembles boundary fluxes only, and a
        # condition it cannot assemble must stop the build rather than vanish
        # from the residual.  A selector that names a boundary tag raises too —
        # the family member domain is a grid with no facet table, so only
        # raw-node selectors resolve.
        self._member_flux_bcs: list[tuple[object, np.ndarray, np.ndarray, int, str, object]] = []
        domain = fam.domain
        member_var_names = {v.name for v, _loc, _shape in self._field_offsets}
        for bc in fam.member_bcs:
            if not isinstance(bc, FluxBC):
                raise LoweringError(
                    f"Family {fam.name!r} declares a {type(bc).__name__} on member "
                    f"field {bc.var.name!r}, and a member kernel assembles boundary "
                    "fluxes only. State the condition as parts.bc.flux(...), or put "
                    "the field on a component, where every boundary condition applies."
                )
            references_tag = bc.where.selector.reduce(
                on_tags=lambda names: True,
                on_nodes=lambda idx: False,
                on_union=lambda a, b: a or b,
                on_difference=lambda a, b: a or b,
            )
            if references_tag:
                named = bc.where.tag if bc.where.tag is not None else repr(bc.where.selector)
                raise LoweringError(
                    f"Family {fam.name!r}: the member boundary condition on "
                    f"{bc.var.name!r} selects {named}, a boundary tag, and a member "
                    "domain is a grid with no tagged facets. Name the boundary cells "
                    "directly (node_indices=[...] or on=from_nodes(...))."
                )
            # Delegate index resolution + face-factor to the shared helpers
            # in diff_soup/_lowering/bcs.py so geometry behaviour stays
            # identical to the component FluxBC path.
            local = _resolve_grid_local_indices(domain, bc.where)
            factor = _grid_face_factor(domain, local)
            # Find local member-slab offset for the BC's variable.
            var_loc = None
            for v, loc, _ in self._field_offsets:
                if v is bc.var or v.name == bc.var.name:
                    var_loc = loc
                    break
            if var_loc is None:
                raise LoweringError(
                    f"Family {fam.name!r}: the member boundary condition names field "
                    f"{bc.var.name!r}, which the member layout does not carry "
                    f"(it holds {sorted(member_var_names)}). Declare the field with "
                    "parts.field_decl(...), or name one that is declared."
                )
            face_geom = None
            if isinstance(bc.flux_fn, _BoundaryFn):
                flux_form = "face"
                face_geom = _grid_face_geometry(
                    domain,
                    local,
                    bc_desc=f"Family {fam.name!r} member flux on {bc.var.name!r}",
                )
            else:
                flux_form = "nodal"
            self._member_flux_bcs.append((bc.var, local, factor, var_loc, flux_form, face_geom))

        # ------------------------------------------------------------------
        # Build-time derivative-form capture.  A member equation authored with
        # ``q.dt`` records its member-local coefficient structure the first
        # time it evaluates; run one eager per-member evaluation NOW so the
        # structure exists before any engine reads ``derivative_forms()`` —
        # deferring capture to the first solve trace would let the diagonal
        # mass surface (which cannot represent scaled or cross-field
        # derivative terms) lower a form-recording family silently wrong.
        # A capture failure is stamped into each still-empty cell so the
        # ``derivative_forms()`` read raises with the stored reason instead of
        # silently dropping the structure.
        try:
            self._capture_member_forms()
        except Exception as exc:  # noqa: BLE001 — stamped, re-raised at the read.
            note = f"{type(exc).__name__}: {exc}"
            for _eq, cell in self._member_form_cells():
                if cell is not None and not cell.get("captured") and not cell.get("captured_plain"):
                    cell.setdefault("capture_error", note)
        self._has_forms = any(
            cell is not None and cell.get("captured") for _eq, cell in self._member_form_cells()
        )
        if self._has_forms and len(fam.member_variables) > 1:
            raise LoweringError(
                f"family {fam.name!r} declares {len(fam.member_variables)} "
                "member fields and a member equation records time-derivative "
                "structure; derivative-form member equations require a family "
                "with exactly ONE member field, so the recorded structure "
                "occupies one contiguous state block. Move the other member "
                "fields to a separate family or component."
            )
        if self._has_forms:
            # A field written by BOTH a residual-form equation and a plain
            # additive equation would mix the residual-channel and rhs-channel
            # sign conventions on the same rows under slope evaluation.
            form_fields = {
                cell["out_field"]
                for _eq, cell in self._member_form_cells()
                if cell is not None and cell.get("captured")
            }
            for eq in fam.member_equations:
                cell = getattr(eq.rhs_fn, "_derivative_form", None)
                if cell is not None and cell.get("captured"):
                    continue
                outs = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
                shared = sorted(v.name for v in outs if v.name in form_fields)
                if shared:
                    raise LoweringError(
                        f"family {fam.name!r}: member field {shared[0]!r} is "
                        f"written by the residual-form equation and also by "
                        f"{getattr(eq, 'name', '?')!r}; a field stated as a "
                        "derivative-expression residual must be written by "
                        "that one equation — fold the extra terms into it."
                    )

    # ------------------------------------------------------------------
    def _member_form_cells(self):
        """Yield ``(equation, form_cell_or_None)`` for every member equation."""
        for eq in self._fam.member_equations:
            fn = eq.rhs_fn
            cell = getattr(fn, "_derivative_form", None)
            yield eq, cell

    def _synthetic_member_inputs(self):
        """Concrete per-member kernel inputs for the build-time capture pass.

        The state is a generic positive profile (mirroring the stencil probe's
        convention), member params take member 0's slice, globals take their
        declared initials, and each gathered macro read is a unit scalar.
        """
        u_row = jnp.linspace(0.75, 1.25, max(int(self._member_state), 1))
        m_row = {k: jnp.asarray(v)[0] for k, v in dict(self._fam.member_params).items()}
        m_row[_MEMBER_INDEX_KEY] = jnp.asarray(0)
        member_globals = {
            name: jnp.asarray(self._global_by_name[name].init) for name in self._global_names
        }
        member_ids = {id(v) for v in self._fam.member_variables}
        gather_row = {}
        for eq in self._fam.member_equations:
            for inp in eq.inputs:
                if isinstance(inp, GlobalVariable) or id(inp) in member_ids:
                    continue
                gather_row.setdefault(inp, jnp.asarray(1.0))
        return u_row, m_row, member_globals, gather_row

    def _capture_member_forms(self) -> None:
        """Eagerly evaluate each member equation once so form cells capture.

        Runs host-side at block construction (concrete inputs, one member),
        per equation, so a closure that cannot evaluate at the synthetic state
        stamps only ITS cell with the failure.  Idempotent — captured cells
        make the evaluation a plain pass-through.  Afterwards, probes each
        captured linear form's coefficient state dependence by re-evaluating
        at per-field-perturbed member states.
        """
        if not self._fam.member_equations:
            return
        u_row, m_row, member_globals, gather_row = self._synthetic_member_inputs()
        view = self._build_member_view(u_row, gather_row, member_globals)
        m = _MemberView(m_row)
        ctx = _FamilyCtx(self._tag_ops)
        for eq, cell in self._member_form_cells():
            if cell is None or cell.get("captured") or cell.get("captured_plain"):
                continue
            try:
                with jax.ensure_compile_time_eval():
                    eq.rhs_fn(view, ctx, m, jnp.asarray(0.0))
            except Exception as exc:  # noqa: BLE001 — stamped, re-raised at the read.
                cell.setdefault("capture_error", f"{type(exc).__name__}: {exc}")
        self._probe_member_form_state_deps(u_row, m_row, member_globals, gather_row)

    def _probe_member_form_state_deps(self, u_row, m_row, member_globals, gather_row):
        """Which member fields each captured form's coefficients read.

        Mirrors the component-path value probe: re-normalize each captured
        linear form's expression at the base member state and once per member
        field with only that field's band perturbed; a coefficient-value move
        marks dependence.  A member equation that also reads gathered macro
        values keeps the conservative ``None`` (whole-state) dependence — the
        probe holds gathers fixed, so their influence cannot be ruled out.
        Any probe failure falls back to ``None`` (conservative), never raises.
        """
        from diff_soup.problem._ir.derivative_ir import normalize_expression
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr

        m = _MemberView(m_row)
        ctx = _FamilyCtx(self._tag_ops)

        def coefficient_values(inner, row):
            view = self._build_member_view(row, gather_row, member_globals)
            expr = inner(view, ctx, m, jnp.asarray(0.37))
            if not isinstance(expr, DerivativeExpr):
                return None
            ir = normalize_expression(expr, out_size=int(expr.size))
            if not hasattr(ir, "terms"):
                return None
            return {term.in_field: (term.pattern, np.asarray(term.values())) for term in ir.terms}

        member_ids = {id(v) for v in self._fam.member_variables}
        for eq, cell in self._member_form_cells():
            if cell is None or not cell.get("captured") or cell.get("nonlinear"):
                continue
            if cell.get("state_deps") is not None:
                continue  # already resolved.
            if any(
                not isinstance(inp, GlobalVariable) and id(inp) not in member_ids
                for inp in eq.inputs
            ):
                continue  # gathered macro reads: keep conservative dependence.
            inner = getattr(eq.rhs_fn, "_raw_rhs", None)
            if inner is None:
                continue
            try:
                with jax.ensure_compile_time_eval():
                    base = coefficient_values(inner, u_row)
                    if base is None:
                        continue
                    deps = []
                    for v, loc, _shape in self._field_offsets:
                        pert = u_row.at[loc : loc + int(v.size)].set(
                            u_row[loc : loc + int(v.size)] * 1.003 + 0.0017
                        )
                        got = coefficient_values(inner, pert)
                        if got is None or set(got) != set(base):
                            deps = None
                            break
                        moved = False
                        for field, (pattern, values) in got.items():
                            ref_pattern, ref_values = base[field]
                            if not (
                                np.array_equal(pattern[0], ref_pattern[0])
                                and np.array_equal(pattern[1], ref_pattern[1])
                            ):
                                deps = None
                                moved = None
                                break
                            if values.shape != ref_values.shape or not np.array_equal(
                                values, ref_values
                            ):
                                moved = True
                        if moved is None:
                            deps = None
                            break
                        if moved:
                            deps.append(str(v.name))
                    if deps is not None:
                        cell["state_deps"] = tuple(sorted(deps))
            except Exception:  # noqa: BLE001 — probe stays conservative.
                continue

    # ------------------------------------------------------------------
    def _build_member_view(self, u_row, gather_row, member_globals) -> MergedView:
        """Per-member [MergedView][diff_soup.problem._ir._merged_view.MergedView].

        Covers one member's flat state slice. Registers each member field's slab band under its family VarRef, each
        gathered macro value under an owner-aware gather VarRef, and every
        referenced global in the merged-view global slot.
        """
        by_ref: dict = {}
        var_id_to_ref: dict[int, VarRef] = {}
        ref_to_var: dict = {}
        for v, loc, shape in self._field_offsets:
            arr = u_row[loc : loc + int(v.size)].reshape(shape)
            ref = self._refs[v]
            by_ref[ref] = arr
            var_id_to_ref[id(v)] = ref
            ref_to_var.setdefault(ref, v)

        # Gather-coupling sources read per-member (gather row).  Each
        # ``gather_row[macro_var]`` is this member's vmapped slice; register it
        # under the macro Variable's owner-aware VarRef so the member closure's
        # ``view[macro_var]`` resolves by id-exact match.
        for macro_var, gathered in gather_row.items():
            g_ref = VarRef(f"__gather__:{self._fam.name}", macro_var.name)
            by_ref[g_ref] = gathered
            var_id_to_ref[id(macro_var)] = g_ref
            ref_to_var.setdefault(g_ref, macro_var)

        # Galvanostatic globals threaded through the merged-view global slot.
        globals_by_name: dict[str, jax.Array] = {}
        global_id_to_name: dict[int, str] = {}
        name_to_global: dict[str, GlobalVariable] = {}
        for name in self._global_names:
            if name in member_globals:
                globals_by_name[name] = member_globals[name]
                g = self._global_by_name[name]
                global_id_to_name[id(g)] = name
                name_to_global[name] = g

        return MergedView(
            by_ref=by_ref,
            var_id_to_ref=var_id_to_ref,
            ref_to_var=ref_to_var,
            globals_by_name=globals_by_name,
            global_id_to_name=global_id_to_name,
            name_to_global=name_to_global,
        )

    def _member_face_view(
        self, u_row, gather_row, member_globals, face_geom, *, context=None, member=None, t=0.0
    ) -> _FaceView:
        """The ``face`` a member boundary flux reads, for one member.

        Member fields go in as their full member-local arrays, so the view
        restricts each to the boundary cells the geometry names. A global and a
        gathered macro value are already this member's own value, with no node
        index to restrict, so they pass through untouched — the same treatment
        the component face view gives a global.
        """
        backing: dict = {}
        for v, loc, shape in self._field_offsets:
            backing[v.name] = u_row[loc : loc + int(v.size)].reshape(shape)
        unrestricted: set = set()
        for name in member_globals:
            if name in member_globals and name not in backing:
                backing[name] = member_globals[name]
                unrestricted.add(name)
        for macro_var, gathered in gather_row.items():
            if macro_var.name not in backing:
                backing[macro_var.name] = gathered
                unrestricted.add(macro_var.name)
        fi, normals, centroids = face_geom
        return _FaceView(
            backing,
            unrestricted,
            jnp.asarray(fi, dtype=jnp.int32),
            jnp.asarray(normals),
            jnp.asarray(centroids),
            ops=self._tag_ops,
            member=member,
            t=t,
            **(context or {}),
        )

    def _single(
        self,
        u_row,
        m_row,
        t,
        member_globals,
        gather_row,
        flux_callables,
        t_flux,
        slope_row=None,
        slope_include=True,
        boundary_context=None,
    ):
        """Per-member kernel: ONE trace, returns the flat member residual.

        Closes over no per-member constants — every per-member quantity arrives
        via ``u_row`` / ``m_row`` / ``gather_row`` / ``slope_row`` (vmap'd axes).

        ``gather_row`` is the per-member dict ``{macro_var: gathered_value}``
        produced by a gather coupling (``c_e -> family``): under the family vmap
        each gathered array is mapped on axis 0, so the value here is this
        member's own ``NodalExtractOp`` slice (e.g. ``c_e[at_indices[m]]``).  Its
        macro Variable is registered in the member MergedView so the member
        equation reads ``view[c_e]`` as if local.

        ``flux_callables``: tuple of flux callables read from the
        runtime family's ``member_flux_callables``.  Threaded here (not captured
        at build time) so learnable eqx.Module callables expose their array leaves
        to AD.  Length must match ``_member_flux_bcs``.

        ``t_flux`` is ``t`` on the experiment's own clock — the dimensional time every
        boundary callback reads.  It arrives already converted (once, outside
        this kernel) and is the only time the flux callables ever see; ``t``
        itself stays the internal solve time the member equations integrate.

        ``slope_row``: per-member field slopes (``{field: (size,)}``) when an
        enclosing slope context is active — a member-local slope context is
        installed around the member equations so residual-form adapters
        evaluate against THIS member's slopes.  ``slope_include`` mirrors the
        enclosing context's residual flag: ``False`` yields the derivative
        action alone, so the derivative-free member surface flux is excluded;
        ``True`` yields full residual rows, where the surface flux enters with
        the residual sign (``residual = derivative + free_terms - flux``).
        """
        fam = self._fam
        member_view = self._build_member_view(u_row, gather_row, member_globals)
        m = _MemberView(m_row)
        ctx = _FamilyCtx(self._tag_ops)

        def _eval_member_equations() -> list:
            parts: list = []
            for eq in fam.member_equations:
                r = eq.rhs_fn(member_view, ctx, m, t)
                if isinstance(eq, MultiEquation):
                    _check_member_multi_return(fam, eq, r)
                    for v in eq.vars:
                        arr = jnp.asarray(finalize_residual(self._domain, r[v.name])).reshape(-1)
                        parts.append((v, arr))
                else:
                    arr = jnp.asarray(finalize_residual(self._domain, r)).reshape(-1)
                    parts.append((eq.var, arr))
            return parts

        in_slope_mode = bool(slope_row)
        if in_slope_mode:
            with slope_context(dict(slope_row), include_residual=slope_include):
                f_parts = _eval_member_equations()
        else:
            f_parts = _eval_member_equations()

        # Assemble member residual at member-local field offsets (matches the
        # member-major layout slab).
        f_member = jnp.zeros((self._member_state,), dtype=u_row.dtype)
        loc_of = {v: loc for v, loc, _ in self._field_offsets}
        for v, arr in f_parts:
            loc = loc_of[v]
            size = int(v.size)
            if arr.size != size:
                raise ValueError(
                    f"family {fam.name!r} member equation residual for "
                    f"variable {v.name!r} has size {int(arr.size)} but the "
                    f"member field has {size} DOFs."
                )
            scale = self._res_scale.get(v.name)
            if scale is not None:
                arr = arr / scale
            f_member = f_member.at[loc : loc + size].add(arr.astype(u_row.dtype))

        # Apply member FluxBCs (Neumann BCs at member boundaries, e.g. the
        # lithium surface flux j_li at r=R on each particle).  The geometry
        # (node indices + face_area/cell_volume factors) was pre-computed at
        # build time; flux callables come from the runtime ``flux_callables``
        # argument so learnable callables' array leaves are traced.
        #
        # Slope modes: the surface flux is derivative-free, so the
        # derivative-action mode excludes it entirely, and the full-residual
        # mode SUBTRACTS it — the rhs-channel reading ``residual =
        # derivative_part - (rhs + flux)`` and the residual-channel reading
        # must place the flux with the same sign.
        if in_slope_mode and not slope_include:
            return f_member
        flux_sign = -1.0 if in_slope_mode else 1.0
        # Boundary callbacks see the member\'s facet state, parameter row, and
        # dimensional clock. Constant/time-only values use the nodal adapter.
        for k, (_var, local, factor, var_loc, flux_form, face_geom) in enumerate(
            self._member_flux_bcs
        ):
            flux_fn = flux_callables[k]
            if flux_form == "nodal":
                flux_val = jnp.asarray(flux_fn(member_view, ctx, t_flux))
            else:
                face = self._member_face_view(
                    u_row,
                    gather_row,
                    member_globals,
                    face_geom,
                    context=boundary_context,
                    member=m,
                    t=t_flux,
                )
                try:
                    flux_val = jnp.asarray(flux_fn(face, t_flux))
                except KeyError as e:
                    bad = e.args[0] if e.args else "?"
                    raise LoweringError(
                        f"Family {self._fam.name!r}: the member flux on {_var.name!r} read "
                        f"q.{bad}, which is not a member field, a global, or a "
                        "value gathered onto the family. Declare it with "
                        "parts.field_decl(...) / spec.global_var(...), or gather it "
                        "with spec.couple(..., onto=<family>)."
                    ) from None
            ni = jnp.asarray(local, dtype=jnp.int32)
            ff = jnp.asarray(factor, dtype=u_row.dtype)
            fv = jnp.broadcast_to(flux_val, ni.shape)
            f_member = f_member.at[var_loc + ni].add((flux_sign * fv * ff).astype(u_row.dtype))

        return f_member

    # ------------------------------------------------------------------
    def member_residual(
        self, u, t, member_globals, runtime_fam=None, member_gathers=None, boundary_context=None
    ):
        """Return the batched ``(count, member_state)`` member residual.

        Slice + reshape to member-major, build the batched param pytree (each
        leaf leading-axis ``count``), then vmap ``single`` ONCE — exactly one
        trace, exactly one vmap'd primitive.

        ``runtime_fam`` supplies the CURRENT call's ``member_params`` (traced
        leaves threaded from the runtime ``families`` arg).  When omitted (the
        smoke-test / standalone path) the builder-time family is used.  Reading
        params off the runtime family — not the captured ``self._fam`` — is what
        keeps them traced leaves for AD instead of closure-cell constants.

        ``member_gathers`` is a ``{macro_var: batched_value}`` dict from gather
        couplings (``c_e -> family``); each value has leading axis ``count`` and
        is mapped on axis 0 by the SAME vmap (no extra primitive), so the member
        kernel reads its own per-member gathered slice via ``view[macro_var]``.
        """
        fam = runtime_fam if runtime_fam is not None else self._fam
        base = self._fam_base
        state_size = self._count * self._member_state
        u_mem = u[base : base + state_size].reshape(self._count, self._member_state)

        params = _validate_member_params(fam)
        params_b = dict(params)
        # member_index rides the SAME vmap axis (one primitive, no extra trace).
        params_b[_MEMBER_INDEX_KEY] = jnp.arange(self._count) + int(
            getattr(fam, "member_index_offset", 0)
        )

        gathers_b = dict(member_gathers) if member_gathers else {}

        # Read flux callables off the runtime family's non-static
        # ``member_flux_callables`` field so learnable eqx.Module callables expose
        # their array leaves through AD (not captured as closure-cell constants).
        flux_callables = tuple(getattr(fam, "member_flux_callables", ()))

        # Member boundary fluxes are boundary callbacks, and every boundary
        # callback is authored on the experiment's own clock.  Convert the
        # internal solve time to dimensional HERE — once per residual evaluation, outside
        # the member vmap — so all three flux forms read the same instant and no
        # form converts a second time.  ``time_ref`` stays a weakly typed scalar
        # so the multiply cannot promote the kernel's dtype, and an unscaled
        # model (``time_ref == 1.0``) keeps the time untouched.
        if self._time_ref == 1.0:
            t_flux = t
        else:
            t_flux = jnp.asarray(self._time_ref) * jnp.asarray(t)

        # An enclosing slope context (the derivative-form lowering evaluating
        # the residual against state slopes) carries each field's slope over
        # its FLAT layout block.  When this family records derivative forms,
        # slice its member field's slopes to per-member rows and map them on
        # the SAME vmap axis, so the member kernel installs a member-local
        # slope context.  A form-recording family declares exactly one member
        # field (enforced at build), so the flat block reshapes member-major.
        slopes_b: dict = {}
        slope_include = True
        if self._has_forms:
            outer_slopes = current_slopes()
            if outer_slopes is not None:
                slope_include = slope_context_includes_residual()
                for v in self._fam.member_variables:
                    s = outer_slopes.get(v.name)
                    if s is not None:
                        slopes_b[v.name] = jnp.asarray(s).reshape(self._count, int(v.size))

        def single(u_row, m_row, gather_row, slope_row, t_, t_flux_):
            self.trace_count += 1
            return self._single(
                u_row,
                m_row,
                t_,
                member_globals,
                gather_row,
                flux_callables,
                t_flux_,
                slope_row=slope_row,
                slope_include=slope_include,
                boundary_context=boundary_context,
            )

        f_mem = jax.vmap(single, in_axes=(0, 0, 0, 0, None, None))(
            u_mem, params_b, gathers_b, slopes_b, t, t_flux
        )
        graph_term = self._graph_residual(
            u_mem,
            t,
            member_globals,
            fam,
            params,
        )
        outer_slopes = current_slopes()
        if outer_slopes is None:
            f_mem = f_mem + graph_term
        elif slope_context_includes_residual():
            # Graph exchange is derivative-free and therefore enters the full
            # residual with the opposite sign from the ordinary RHS channel.
            f_mem = f_mem - graph_term
        return f_mem

    def _graph_residual(self, u_mem, t, member_globals, fam, params):
        """Evaluate every declared edge once and scatter ``-source/+target``."""
        runtime_graphs = {graph.name: graph for graph in getattr(fam, "graphs", ())}
        runtime_exchanges = tuple(getattr(fam, "graph_exchanges", ()))
        if not runtime_exchanges:
            return jnp.zeros_like(u_mem)
        all_fields = frozenset(self._graph_field_info)
        result = jnp.zeros_like(u_mem)

        def endpoint(row, index, allowed, declaration):
            fields = {
                name: row[loc : loc + size].reshape(shape)
                for name in allowed
                for loc, size, shape in (self._graph_field_info[name],)
            }
            param_row = {name: value[index] for name, value in params.items()}
            return _GraphEndpointView(
                fields=fields,
                all_fields=all_fields,
                declaration=declaration,
                params=param_row,
                globals_=member_globals,
                member_index=index + int(getattr(fam, "member_index_offset", 0)),
            )

        for exchange in runtime_exchanges:
            graph = runtime_graphs.get(exchange.graph)
            if graph is None:
                raise ValueError(
                    f"family {self._fam.name!r} graph exchange {exchange.name!r} "
                    f"names missing runtime graph {exchange.graph!r}; preserve "
                    "the authored graph structure when replacing parameters."
                )
            edge_count = int(graph.edge_count)
            if edge_count == 0:
                continue
            sources = jnp.asarray(graph.edge_sources, dtype=jnp.int32)
            targets = jnp.asarray(graph.edge_targets, dtype=jnp.int32)
            numeric_weights = None
            if not callable(graph.weights):
                numeric_weights = jnp.asarray(graph.weights, dtype=u_mem.dtype)
                if numeric_weights.ndim == 0:
                    numeric_weights = jnp.broadcast_to(numeric_weights, (edge_count,))

            def one_edge(edge_index, source_index, target_index):
                source_weight = endpoint(
                    u_mem[source_index],
                    source_index,
                    exchange.weight_reads,
                    "weight_reads",
                )
                target_weight = endpoint(
                    u_mem[target_index],
                    target_index,
                    exchange.weight_reads,
                    "weight_reads",
                )
                if callable(graph.weights):
                    unweighted = _GraphEdgeView(
                        weight=jnp.asarray(1.0, dtype=u_mem.dtype),
                        index=edge_index,
                        source_index=source_index,
                        target_index=target_index,
                        t=t,
                        globals_=member_globals,
                        graph=graph.name,
                        directed=graph.directed,
                    )
                    weight = jnp.asarray(graph.weights(source_weight, target_weight, unweighted))
                    if int(weight.size) == 1:
                        weight = weight.reshape(())
                    else:
                        raise ValueError(
                            f"family graph {graph.name!r} weight rule must return "
                            f"one scalar per edge; received shape {weight.shape}."
                        )
                else:
                    weight = numeric_weights[edge_index]

                source = endpoint(u_mem[source_index], source_index, exchange.reads, "reads")
                target = endpoint(u_mem[target_index], target_index, exchange.reads, "reads")
                edge = _GraphEdgeView(
                    weight=weight.astype(u_mem.dtype),
                    index=edge_index,
                    source_index=source_index,
                    target_index=target_index,
                    t=t,
                    globals_=member_globals,
                    graph=graph.name,
                    directed=graph.directed,
                )
                flow = jnp.asarray(exchange.fn(source, target, edge)).reshape(-1)
                _loc, write_size, _shape = self._graph_field_info[exchange.writes]
                if int(flow.size) != write_size:
                    raise ValueError(
                        f"family graph exchange {exchange.name!r} returned "
                        f"{int(flow.size)} values, but member field "
                        f"{exchange.writes!r} has {write_size} DOFs; return one "
                        "value per written field DOF."
                    )
                scale = self._res_scale.get(exchange.writes)
                if scale is not None:
                    flow = flow / scale
                return flow.astype(u_mem.dtype)

            try:
                flow = jax.vmap(one_edge)(jnp.arange(edge_count, dtype=jnp.int32), sources, targets)
            except AttributeError as exc:
                raise ValueError(str(exc)) from exc
            write_loc, write_size, _shape = self._graph_field_info[exchange.writes]
            columns = write_loc + jnp.arange(write_size, dtype=jnp.int32)
            result = result.at[targets[:, None], columns[None, :]].add(flow)
            result = result.at[sources[:, None], columns[None, :]].add(-flow)
        return result

    def __call__(
        self, f, u, t, member_globals, runtime_fam=None, member_gathers=None, boundary_context=None
    ):
        """Scatter-add the batched family residual into ``f`` (one ``.at[].add``)."""
        f_mem = self.member_residual(
            u,
            t,
            member_globals,
            runtime_fam=runtime_fam,
            member_gathers=member_gathers,
            boundary_context=boundary_context,
        )
        base = self._fam_base
        state_size = self._count * self._member_state
        return f.at[base : base + state_size].add(f_mem.reshape(-1))


def _build_family_residual(
    fam: Family, fam_base: int, *, total_size: int, time_ref: float = 1.0
) -> _FamilyResidualBlock:
    """Build the batched residual block for one [Family][diff_soup.problem._ir.family.Family].

    Args:
        fam: The family population.
        fam_base: Flat offset of the family's member-major slab in the assembled
            state vector (after all components / prior families).
        total_size: Total assembled state size (for shape bookkeeping).
        time_ref: dimensional seconds one internal time unit stands for. The block
            converts the internal solve time to dimensional once before it calls a member
            boundary flux, the clock every boundary callback is authored on;
            ``1.0`` (the unscaled default) leaves the time untouched.

    Returns:
        A
        [_FamilyResidualBlock][diff_soup._lowering.family_residual._FamilyResidualBlock]
        callable as ``block(f, u, t,
        member_globals)`` that scatter-adds the family residual into ``f`` with
        one trace + one vmap'd primitive.  ``member_globals`` is a
        ``{global_name: value}`` dict supplying galvanostatic globals read by
        member equations.
    """
    return _FamilyResidualBlock(fam, fam_base, total_size, time_ref=time_ref)
