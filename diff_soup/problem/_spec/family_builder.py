# LOC > 500: one cohesive family facade — the _FamilyHandle mutable record, its
# member field/equation/bc/couple builders, the _MemberView + (q, m)->IR adapter,
# and the param-coercion guard taxonomy all share the handle's template state
# and must re-materialise the Family IR node together; splitting fragments it.
"""``spec.family(...)`` facade — the user-facing family/population builder.

A [Family][diff_soup.problem._ir.family.Family] is a population of ``count``
structurally-identical members (e.g. an MPM electrode's many particles) sharing
ONE member domain and per-member variable / equation / BC / coupling templates.
This module is the declarative front-end that compiles down to that IR node +
``diff_soup._lowering.family_residual`` lowering — it does NOT re-implement the
vmapped residual / sparsity / Schur machinery.

The user writes::

    parts = spec.family("p", count=3, domain=unit_ball, params={"R": [...]})
    parts.field_decl("c_s", mass="differential")

    @parts.equation(writes="c_s", reads=("c_s",))
    def fick(q, m):                       # 2-arg: (q-accessor, member-view)
        return m.D_s * q.ops.div_flux(jnp.ones_like(q.c_s), q.c_s)

    parts.bc.flux("c_s", node_indices=[...], fn=lambda face, t: ...)
    parts.bc.flux(                                   # read member parameters
        "c_s", node_indices=[...], fn=lambda q, t: q.m.k * (1.0 - q.c_s)
    )
    spec.couple(c_e, onto="p", at=[...])             # gather (macro -> family)
    parts.couple(parts.field("c_s").surface(), onto="cell", at=[...])  # scatter

Per-member ``params`` ride the Family IR node as the only non-static (pytree)
leaves so they remain differentiable / vmappable; the construction guards
(closure-cell trap, leading-dim mismatch, underscore-prefixed names) reject
silent-footgun inputs at ``family()`` build time.

The ``_FamilyHandle`` uses a dirty flag ([_mark_dirty][diff_soup.problem._spec.family_builder._FamilyHandle._mark_dirty]) and a lazy
[_ensure_synced][diff_soup.problem._spec.family_builder._FamilyHandle._ensure_synced] so the frozen Family IR node is re-materialised ONCE
at the next read boundary (smart-couple registry resolvers and the build walk),
rather than on every field/equation/bc mutation.  The registry always reflects
the latest declarations the moment anything reads it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup._warnings import warn_user
from diff_soup.domains.boundary_set import BoundarySet, from_nodes, tags
from diff_soup.problem._ir.equation import Equation, MultiEquation
from diff_soup.problem._ir.family import Family
from diff_soup.problem._ir.family_graph import FamilyGraph, FamilyGraphExchange
from diff_soup.problem._ir.variable import GlobalVariable, Variable

from .q_accessor import RESERVED_Q_NAMES, _QView
from .specs import BCSpec

# A ``count`` above this prints the soft over-population warning (no hard
# ceiling — large populations are legal, just flagged).
_SOFT_COUNT_CEILING = 256

# Reserved member-view attribute exposing the vmap row index.
_MEMBER_INDEX = "member_index"


@dataclass(frozen=True)
class FamilyFieldSummary:
    """One member field, in member terms.

    Attributes:
        name: The member field's name.
        mass: ``"differential"`` or ``"algebraic"``.
        size: The field's per-member size (the node count on the member
            domain for a profile field; ``1`` for a per-member scalar).
    """

    name: str
    mass: str
    size: int


@dataclass(frozen=True)
class FamilyBucketSummary:
    """One static size bucket of a family population.

    Attributes:
        index: Bucket ordinal (0 for a uniform family's only bucket).
        group: The lowered population-group name — the family name for a
            uniform family, ``"<family>[<bucket>]"`` for a ragged one.  This
            is the owner key the bucket's fields carry in the state layout
            (tolerance keys, ``family_view``, ``assemble_state`` keys).
        count: Number of members in the bucket.
        first_member: Global index of the bucket's first member.
        fields: Per-member field summaries for this bucket's member size.
    """

    index: int
    group: str
    count: int
    first_member: int
    fields: tuple[FamilyFieldSummary, ...]


@dataclass(frozen=True)
class FamilySummary:
    """A family population described in field/member terms (no flat offsets).

    Attributes:
        name: The authored family name.
        count: Total member count across all buckets.
        params: The declared per-member parameter names.
        buckets: The static size buckets, in member order.
    """

    name: str
    count: int
    params: tuple[str, ...]
    buckets: tuple[FamilyBucketSummary, ...]


@dataclass(frozen=True)
class FamilyMemberSummary:
    """One member of a family population, in field/member terms.

    Attributes:
        index: The member's global index in the authored population (the value
            ``m.member_index`` carries in member closures).
        bucket: The member's static size bucket ordinal.
        group: The member's lowered population-group name.
        fields: The member's field summaries (sized for its bucket).
        params: This member's parameter values, keyed by param name.
    """

    index: int
    bucket: int
    group: str
    fields: tuple[FamilyFieldSummary, ...]
    params: dict[str, float]


class _MemberView:
    """Frozen read-only member-parameter namespace handed to family closures.

    ``m`` (the 2nd positional arg of a family closure ``def fick(q, m)``) is a
    read-only namespace whose attributes are EXACTLY the keys of ``params={}``
    plus the reserved [member_index][diff_soup.problem._spec.family_builder._MemberView.member_index] (an int32 scalar tracer equal to
    the vmap row index).  Under the family vmap each ``(count,)`` param is
    mapped on axis 0, so ``m.R`` is a scalar tracer inside the closure.

    Reading ``m.<name>`` for a name not in ``params={}`` raises
    `AttributeError` listing the declared params at first trace.
    """

    __slots__ = ("_row",)

    def __init__(self, row: dict[str, Any]) -> None:
        # Plain slot assignment — ``__slots__`` already closes the namespace, so
        # assigning any non-``_row`` attribute raises ``AttributeError``
        # (the view stays effectively read-only without a custom
        # ``__setattr__``).  Reading an undeclared param routes through
        # ``__getattr__`` below.
        self._row = dict(row)

    def __getattr__(self, name: str) -> Any:
        # Only fires for attrs NOT found normally — ``_row`` is a real slot, so
        # it never reaches here once ``__init__`` has set it.
        row = object.__getattribute__(self, "_row")
        if name == _MEMBER_INDEX:
            return row.get(_MEMBER_INDEX, jnp.asarray(0))
        if name in row:
            return row[name]
        declared = sorted(k for k in row if k != _MEMBER_INDEX)
        raise AttributeError(
            f"family member view has no param {name!r}; declared params: {declared}"
        )

    def __getitem__(self, name: str) -> Any:
        return object.__getattribute__(self, "_row")[name]


class _FamilyMemberRHS:
    """Adapter wrapping a user ``(q, m)`` closure for the IR rhs callsite.

    The IR callsite is ``rhs_fn(view, ctx, m, t)``.  The member-residual
    lowering (``family_residual``) passes a
    [MergedView][diff_soup.problem._ir._merged_view.MergedView] (``view``), a
    member-domain ctx (``ctx.ops``), the runtime member-view ``m`` (built from
    the vmapped param row), and ``t``.  These are re-wrapped in a
    [_QView][diff_soup.problem._spec.q_accessor._QView] so the user closure reads
    ``q.c_s`` / ``q.ops`` exactly as in a plain
    ``@spec.equation`` body, then calls ``inner(q, m)``.

    A plain Python closure is captured by reference; an ``eqx.Module`` closure
    is held on the non-static [inner][diff_soup.problem._spec.family_builder._FamilyMemberAdapter.inner] slot of [_FamilyMemberAdapter][diff_soup.problem._spec.family_builder._FamilyMemberAdapter]
    so its array leaves reach AD.
    """

    def __init__(self, inner: Callable, var_map: dict[str, Any]):
        self._inner = inner
        self._var_map = var_map

    @property
    def _raw_rhs(self):
        """The wrapped user closure — the content-identity of this equation.

        Exposed so the content-hash fingerprint folds the user closure's own
        body (bytecode / captured cells) instead of collapsing every member
        equation to this wrapper class's name.
        """
        return self._inner

    def __call__(self, view, ctx, m, t):
        q = _QView(
            view=view,
            var_map=self._var_map,
            ops=ctx.ops,
            params={},
            functions={},
            t=t,
            coords=None,
            controls=getattr(ctx, "controls", None),
        )
        return self._inner(q, m)


class _FamilyMemberAdapter(eqx.Module):
    """``eqx.Module`` variant of [_FamilyMemberRHS][diff_soup.problem._spec.family_builder._FamilyMemberRHS].

    Keeps the user closure on a non-static slot so an ``eqx.Module`` member
    closure's parameter pytree reaches ``eqx.partition`` and ``jax.grad``.
    """

    inner: object
    var_map: object = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped user closure — the content-identity of this equation."""
        return self.inner

    def __call__(self, view, ctx, m, t):
        q = _QView(
            view=view,
            var_map=self.var_map,
            ops=ctx.ops,
            params={},
            functions={},
            t=t,
            coords=None,
            controls=getattr(ctx, "controls", None),
        )
        return self.inner(q, m)


def _make_member_rhs(inner: Callable, var_map: dict[str, Any]) -> Callable:
    if isinstance(inner, eqx.Module):
        return _FamilyMemberAdapter(inner=inner, var_map=var_map)
    return _FamilyMemberRHS(inner, var_map)


class _MemberResidualFormAdapter(eqx.Module):
    """Adapt a member equation's derivative-expression return onto the rhs seam.

    The per-member counterpart of the component residual-form adapter: a member
    closure returning a derivative expression (built from ``q.dt``) states the
    residual ``expression == 0`` for that member's rows.  Outside a slope
    context the adapter emits the NEGATED derivative-free part so the ordinary
    rhs channel keeps the reading ``residual = derivative_part - rhs``; inside
    a slope context (the member kernel installs a member-local one) it
    evaluates the expression against the provided per-member slopes.

    The structure cell, filled by the family block's build-time capture pass
    only, records the expression's MEMBER-LOCAL pattern
    (rows/cols within one member's field block); the lowered problem expands it
    across the population when surfacing ``derivative_forms()``.  A member
    derivative expression may only reference this family's member fields —
    taking ``q.dt`` of a gathered macro field has no per-member coefficient
    block and is rejected at capture with a directing error.

    An ``eqx.Module`` so a learnable module inside the wrapped closure keeps
    its array leaves visible to ``eqx.partition`` and gradient transformations.
    """

    inner: object
    equation_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    member_fields: tuple = eqx.field(static=True)
    cell: object = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    @property
    def _derivative_form(self) -> dict:
        """The captured member-local structure cell (empty until the capture pass)."""
        return self.cell.data

    def _check_member_only(self, referenced) -> None:
        """Reject a derivative reference to anything but this family's fields."""
        foreign = sorted(set(referenced) - set(self.member_fields))
        if foreign:
            raise ValueError(
                f"member equation {self.equation_name!r} takes the time "
                f"derivative of {foreign[0]!r}, which is not a member field of "
                f"this family (member fields: {sorted(self.member_fields)!r}); "
                "a member derivative expression may only reference the "
                "family's own member fields."
            )

    def __call__(self, view, ctx, m, t):
        from diff_soup.problem._ir.derivative_ir import (
            ImplicitResidualIR,
            evaluate_expression,
            normalize_expression,
            record_form,
            record_plain,
            zero_slopes_for,
        )
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr, current_slopes

        result = self.inner(view, ctx, m, t)
        form_cell = self.cell.data
        if not isinstance(result, DerivativeExpr):
            record_plain(form_cell)
            return result
        from .build import _check_cell_scales, _slope_evaluation

        slopes = current_slopes()
        if slopes is not None:
            return _slope_evaluation(result, slopes, ctx)
        _check_cell_scales(result, ctx)
        ir = normalize_expression(result, out_size=int(result.size))
        # Coefficient state dependence is probed by the family residual block
        # after its capture pass; conservative until then.
        record_form(
            form_cell,
            self.field_name,
            ir,
            state_deps=lambda: form_cell.get("state_deps"),
            check_fields=self._check_member_only,
        )
        if isinstance(ir, ImplicitResidualIR):
            return evaluate_expression(result, zero_slopes_for(result), int(result.size))
        return -ir.residual_part()


def _wrap_member_derivative_return(
    rhs: Callable, *, equation_name: str, field_name: str, member_fields: tuple
) -> Callable:
    """Wrap a single-output member rhs in the residual-form adapter."""
    from .build import _FormCell

    return _MemberResidualFormAdapter(
        inner=rhs,
        equation_name=equation_name,
        field_name=field_name,
        member_fields=tuple(member_fields),
        cell=_FormCell(),
    )


class _MemberBundleGuard(eqx.Module):
    """Reject derivative expressions inside a bundled multi-output member return.

    A multi-output member equation returns a plain dict of residual arrays;
    time-derivative structure (``q.dt``) must be authored in single-output
    member equations so its coefficient pattern can be recorded per output
    row block.  An ``eqx.Module`` so a learnable module inside the wrapped
    closure keeps its array leaves visible to ``eqx.partition``.
    """

    inner: object
    equation_name: str = eqx.field(static=True)

    @property
    def _raw_rhs(self):
        """The wrapped rhs adapter — the content-identity of this equation."""
        return self.inner

    def __call__(self, view, ctx, m, t):
        from diff_soup.problem._spec.derivative_expr import DerivativeExpr

        result = self.inner(view, ctx, m, t)
        if isinstance(result, dict) and any(isinstance(v, DerivativeExpr) for v in result.values()):
            raise NotImplementedError(
                "derivative expressions are not supported inside a bundled "
                f"multi-output member equation ({self.equation_name!r}); author "
                "residual-form rows as single-output member equations."
            )
        return result


def _validate_closure_is_two_arg(fn: Callable, name: str) -> None:
    """Raise ``TypeError`` if a family equation closure is not 2-arg ``(q, m)``.

    Probes the callable's required positional arity host-side; a ``*args``
    closure is allowed (it can absorb both).
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return  # builtins / C callables — let the trace decide.
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        return
    required_positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    if len(required_positional) != 2:
        raise TypeError(
            f"family equation {name!r}: closure must be 2-arg (q, m) — the "
            "q-accessor and the per-member view; got "
            f"{len(required_positional)} required positional arg(s)."
        )


def _validate_graph_callable_is_three_arg(fn: Callable, *, where: str) -> None:
    """Require the public ``(source, target, edge)`` graph signature."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        return
    required_positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    if len(required_positional) != 3:
        raise TypeError(
            f"{where} must accept exactly three required positional arguments "
            "(source, target, edge); received "
            f"{len(required_positional)}."
        )


class _FamilyEquationDecorator:
    """``parts.equation(writes, reads, name=None)`` — registers a member equation.

    Distinct from the one-arg component
    [_EquationDecorator][diff_soup.problem._spec.decorators._EquationDecorator]: the
    registered closure takes ``(q, m)`` (2-arg), where ``m`` is the per-member
    view.  The decorator lowers the closure to an IR
    [Equation][diff_soup.problem._ir.equation.Equation] whose ``rhs_fn`` has the
    ``(view, ctx, m, t)`` signature ``family_residual`` expects.

    ``writes`` accepts one member-field name (single-output) or a tuple/list of
    several (multi-output).  A multi-output closure returns a dict keyed by each
    written field's name; the member kernel runs the closure ONCE per member and
    distributes each entry to its field's rows, so tightly coupled member fields
    share one bundled kernel instead of re-tracing it per output.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: "_FamilyHandle") -> None:
        self._handle = handle

    def __call__(
        self,
        *,
        writes: Any,
        reads: tuple[str, ...] = (),
        name: Optional[str] = None,
    ):
        if not isinstance(writes, str):
            writes = tuple(writes)
            if len(writes) == 0:
                raise ValueError("parts.equation: writes= must name at least one member field.")
            if not all(isinstance(w, str) for w in writes):
                raise TypeError(
                    "parts.equation: writes= must be a member-field name or a "
                    "tuple/list of member-field names."
                )
            if len(set(writes)) != len(writes):
                raise ValueError(
                    f"parts.equation: writes= lists a duplicate field name "
                    f"({writes!r}); each output field may appear once."
                )
            if len(writes) == 1:
                writes = writes[0]

        def decorator(fn: Callable) -> Callable:
            if name is not None:
                eq_name = name
            elif isinstance(writes, str):
                eq_name = getattr(fn, "__name__", writes)
            else:
                eq_name = getattr(fn, "__name__", "+".join(writes))
            _validate_closure_is_two_arg(fn, eq_name)
            self._handle._register_equation(writes, tuple(reads), fn, eq_name)
            return fn

        return decorator


class _FamilyBCNamespace:
    """Boundary fluxes evaluated with each member's state and parameters."""

    __slots__ = ("_handle",)

    def __init__(self, handle: "_FamilyHandle") -> None:
        self._handle = handle

    def flux(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        fn: Optional[Callable] = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
    ) -> None:
        """Member boundary flux ``fn(q, t)``, one density per boundary cell.

        Boundary location is selected by exactly one of:

        * ``on=`` — a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet]
          (used as-is), a tag-name string, or a node-index array-like.
        * ``tag=`` — a tagged-mesh facet-tag string (the single-form spelling of
          ``on=<str>``).
        * ``node_indices=`` — an explicit integer node-index array-like (the
          single-form spelling of ``on=<array>``).

        ``fn`` receives the member's boundary ``q`` view and the time, and
        returns the flux density there, counted positive into the member — the
        same convention and the same closure signature as
        [flux][diff_soup.problem._spec.bcs_namespace.flux] on a component.

        ``t`` arrives in **dimensional seconds**, the experiment's own clock, like every
        other boundary callback. ``spec.time_scale(tau)`` integrates the model on
        internal time ``t_int = t_SI / tau``, and the flux still receives
        ``t_SI``: the same closure returns the same density at the same physical
        instant whether or not the model is nondimensionalized.

        ``q.<field>`` is a member field at the selected boundary cell,
        ``q.<name>`` also reaches the globals and any macro value gathered
        onto the family, and ``q.geom.normal`` / ``q.geom.x`` give the
        outward unit normal and the cell centre. The member-residual lowering
        folds the flux into the same per-member vmapped kernel, so every member
        evaluates its own boundary state. ``q.m.<name>`` reads this member's
        entry of ``params=``; ``q.nodal.<field>`` reads its full nodal array,
        and ``q.ops`` acts on that array. ``value=`` and ``value_fn=lambda t:``
        supply constant and time-only densities. Exactly one of ``value=``,
        ``value_fn=``, and ``fn=`` is required.
        """
        if sum([on is not None, tag is not None, node_indices is not None]) != 1:
            raise ValueError("parts.bc.flux: provide exactly one of on= / tag= / node_indices=")
        from .bcs_namespace import _check_value

        _check_value("flux", value, value_fn, fn)
        # Fold tag= / node_indices= into on= then promote to a BoundarySet,
        # mirroring the spec.bc.* _where() logic.
        if tag is not None:
            on = tag
        elif node_indices is not None:
            on = node_indices
        if isinstance(on, BoundarySet):
            selector = on
        elif isinstance(on, str):
            selector = tags(on)
        else:
            selector = from_nodes(on)
        where: dict[str, Any] = {"selector": selector}
        self._handle._member_bcs.append(
            BCSpec(
                kind="flux",
                var=var,
                where=where,
                fn=fn,
                value=value,
                value_fn=value_fn,
            )
        )
        # Defer re-materialisation — just mark dirty.  The freshly appended
        # member BC surfaces on the node the moment the next reader (couple
        # registry / build walk) calls _ensure_synced().
        self._handle._mark_dirty()


class _FamilyHandle:
    """A declared family, returned by `spec.family(...)`."""

    def __init__(
        self,
        spec,
        *,
        name: str,
        count: int,
        domain: Any,
        params: dict[str, Any],
        scale: Optional[dict[str, float]],
        residual_scale: Optional[dict[str, float]],
        operators: Any = None,
        buckets: Optional[tuple] = None,
        stencil_radius: Optional[int] = None,
    ) -> None:
        self._spec = spec
        self.name = str(name)
        self.count = int(count)
        # Declared per-member Jacobian reach; ``None`` keeps the member
        # domain's own radius-1 pattern.  Validated by ``Family.__init__``.
        self._stencil_radius = stencil_radius
        # Static size buckets: ``((domain, count), ...)``.  A uniform family is
        # the one-bucket case; a ragged family (per-member domains of differing
        # size) lowers ONE population group per bucket, each with its own
        # uniform member kernel.  ``self.domain`` stays the first bucket's
        # domain as the representative for uniform-only paths.
        self._buckets: tuple = tuple(buckets) if buckets is not None else ((domain, int(count)),)
        self.domain = self._buckets[0][0]
        self._params = dict(params)
        self._scale = dict(scale) if scale else None
        self._residual_scale = dict(residual_scale) if residual_scale else None
        # Opaque operator-bank factory threaded onto the Family IR node; ``None``
        # selects the built-in bank.  Validated by ``Family.__init__``.
        self._operator_factory = operators
        # Per-bucket member Variable lists (created ONCE at field_decl so
        # coupling resolution by identity stays stable across re-syncs).  The
        # uniform path reads bucket 0.
        self._bucket_vars: list[list[Variable]] = [[] for _ in self._buckets]
        self._member_eq_specs: list[tuple[Any, tuple[str, ...], Callable, str]] = []
        self._member_bcs: list[BCSpec] = []
        self._graphs: dict[str, FamilyGraph] = {}
        self._graph_exchanges: list[FamilyGraphExchange] = []
        self._bc_ns: Optional[_FamilyBCNamespace] = None
        self._dirty = True
        # Register the handle so the smart-couple resolvers and build walk can
        # trigger the lazy re-materialise at their read boundary.
        spec._family_handles[self.name] = self
        # Materialise the (empty-template) Family immediately so the smart-couple
        # registry has a concrete node before any field/equation is declared
        # (and so ``spec._families[name]`` is always a Family, never a handle).
        self._sync()

    # ------------------------------------------------------------- bucket info
    @property
    def is_ragged(self) -> bool:
        """Whether the family's member domains form more than one size bucket."""
        return len(self._buckets) > 1

    @property
    def _member_vars(self) -> list:
        """Bucket 0's member Variables (the uniform-path template list)."""
        return self._bucket_vars[0]

    def _bucket_names(self) -> tuple[str, ...]:
        """Lowered population-group names: the family name, or one per bucket.

        A ragged family's buckets appear in the state layout (and so in
        tolerance keys, ``family_view``, and ``assemble_state`` keys) as
        ``"<family>[<bucket>]"``.
        """
        if not self.is_ragged:
            return (self.name,)
        return tuple(f"{self.name}[{k}]" for k in range(len(self._buckets)))

    # -------------------------------------------------- lazy sync / dirty flag
    def _mark_dirty(self) -> None:
        """Flag a pending re-materialise without doing the O(N) rebuild now."""
        self._dirty = True

    def _ensure_synced(self) -> Any:
        """Re-materialise the Family node(s) if a mutation has marked them dirty.

        Called by the smart-couple registry resolvers and the build walk at
        their read boundaries; the nodes in ``spec._families`` always reflect
        the latest declarations by the time anything reads them.  Idempotent
        when the handle is already clean.  Returns the single node for a
        uniform family, the tuple of bucket nodes for a ragged one.
        """
        if self._dirty:
            self._sync()
        nodes = tuple(self._spec._families[n] for n in self._bucket_names())
        return nodes[0] if not self.is_ragged else nodes

    # ------------------------------------------------------------------ decls
    def field_decl(
        self,
        name: str,
        *,
        mass: str,
        shape: Optional[tuple[int, ...]] = None,
    ) -> Any:
        """Declare a member field, a profile on the member domain or a per-member scalar.

        A profile has one value per member-domain node. `shape=(1,)` declares a
        per-member scalar, which must be `mass="algebraic"` and is closed by its own
        member equation.

        Args:
            name: Member field name, unique in the family.
            mass: `"differential"` or `"algebraic"`.
            shape: `(1,)` for a per-member scalar, or `None` for a profile, refused for a ragged family.

        Returns:
            The member field's variable, or one per size bucket for a ragged family.
        """
        if shape is not None and self.is_ragged:
            raise ValueError(
                f"family {self.name!r} field {name!r}: shape= cannot be given "
                "for a ragged family — each size bucket's field takes its own "
                "member domain's node count."
            )
        if any(v.name == name for v in self._member_vars):
            raise ValueError(f"family {self.name!r}: duplicate member field {name!r}")
        if name in RESERVED_Q_NAMES:
            reserved = ", ".join(sorted(RESERVED_Q_NAMES))
            raise ValueError(
                f"family {self.name!r} field {name!r}: {name!r} is reserved by the "
                f"equation query object, which answers q.{name} out of its own "
                f"machinery, so a member field named {name!r} could never be read "
                f"back as q.{name}. The reserved names are {reserved}. Give the "
                "member field a name outside that set."
            )
        made: list[Variable] = []
        for k, (dom, _cnt) in enumerate(self._buckets):
            n_nodes = int(getattr(dom, "num_nodes"))
            if shape is None:
                shape_k = (n_nodes,)
            else:
                shape_k = tuple(int(s) for s in shape)
            member_size = 1
            for s in shape_k:
                member_size *= s
            if shape_k == (1,) and n_nodes != 1:
                if mass != "algebraic":
                    raise ValueError(
                        f"family {self.name!r} field {name!r}: a per-member scalar "
                        f"(shape (1,)) must be mass='algebraic' (0 = f, closed by "
                        f"its own member equation); received mass={mass!r}. A "
                        "differential per-member scalar has no member mass row — "
                        "declare it algebraic, or make it a profile field on the "
                        "member domain."
                    )
            elif member_size != n_nodes:
                raise ValueError(
                    f"family {self.name!r} field {name!r}: a multi-dim member field "
                    f"(shape {shape_k}) is not supported — a family member field is "
                    "a single 1-D radial profile or a per-member algebraic scalar "
                    "(shape=(1,), mass='algebraic'). Reformulate the physics on the "
                    "shared unit ball with the size entering only as the scalar "
                    "D_s/R^2."
                )
            var = Variable(name=name, shape=shape_k, mass=mass)
            self._bucket_vars[k].append(var)
            made.append(var)
        self._mark_dirty()
        return made[0] if not self.is_ragged else tuple(made)

    @property
    def equation(self) -> _FamilyEquationDecorator:
        """Decorator that registers a member equation, as `@members.equation(writes=..., reads=..., name=None)`.

        The closure takes `(q, m)`, with `m.<name>` the member's entry of a
        per-member parameter and `m.member_index` its index. `writes` names one
        member field or a tuple of them.
        """
        return _FamilyEquationDecorator(self)

    @property
    def bc(self) -> _FamilyBCNamespace:
        """Member boundary conditions, `members.bc.flux(var, on=..., value= | value_fn= | fn=...)`.

        The flux takes the same arguments as `spec.bc.flux` without `name` and
        `cross_facet`. Inside `fn`, `q.m.<name>` reads the member's entry of a
        per-member parameter.
        """
        if self._bc_ns is None:
            self._bc_ns = _FamilyBCNamespace(self)
        return self._bc_ns

    def _register_equation(
        self, writes: str, reads: tuple[str, ...], fn: Callable, name: str
    ) -> None:
        self._member_eq_specs.append((writes, reads, fn, name))
        self._mark_dirty()

    # ------------------------------------------------------- adjacency graphs
    def graph(
        self,
        name: str,
        *,
        edges: Any,
        directed: bool,
        weights: Any = 1.0,
    ) -> FamilyGraph:
        """Declare a graph of edges between members, for pairwise exchanges.

        A ragged family cannot hold a graph.

        Args:
            name: Graph name, unique in the family.
            edges: Integer array of shape `(2, E)`, source member indices in row 0 and targets in row 1.
            directed: Whether each edge has a direction.
            weights: A float scalar, a float array of length `E`, or a function `(source, target, edge)` returning each edge's weight.

        Returns:
            The [`FamilyGraph`][diff_soup.problem.FamilyGraph].
        """
        if self.is_ragged:
            raise ValueError(
                f"family {self.name!r}: a graph cannot span a ragged population "
                "because its size buckets have separate state slabs; declare "
                "one uniform family per graph."
            )
        if name in self._graphs:
            raise ValueError(
                f"family {self.name!r}: duplicate graph name {name!r}; choose a unique graph name."
            )
        if callable(weights):
            _validate_graph_callable_is_three_arg(
                weights, where=f"family graph {name!r} weight rule"
            )
        graph = FamilyGraph(
            name=name,
            member_count=self.count,
            edges=edges,
            directed=directed,
            weights=weights,
        )
        self._graphs[name] = graph
        self._mark_dirty()
        return graph

    def graph_exchange(
        self,
        *,
        graph: str,
        writes: str,
        reads: tuple[str, ...],
        weight_reads: tuple[str, ...] = (),
        name: Optional[str] = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator that declares a pairwise exchange along the edges of a graph.

        The decorated `fn(source, target, edge)` returns the flow into the target,
        with `source.<field>` and `target.<field>` the fields in `reads` and
        `edge.weight` the edge's weight.

        Args:
            graph: Name of a graph declared with `graph`.
            writes: Member field whose rows receive the exchange.
            reads: Member fields the exchange reads at both endpoints.
            weight_reads: Member fields a weight function reads.
            name: Exchange name, or `None` for the function's name.

        Returns:
            The decorator, which returns the function unchanged.
        """
        if graph not in self._graphs:
            raise ValueError(
                f"family {self.name!r}: graph_exchange names unknown graph "
                f"{graph!r}; declare it first with family.graph(...)."
            )
        field_names = {v.name for v in self._member_vars}
        if writes not in field_names:
            raise ValueError(
                f"family {self.name!r}: graph exchange writes unknown member "
                f"field {writes!r}; declare it with field_decl(...)."
            )

        def _field_tuple(value: Any, *, label: str) -> tuple[str, ...]:
            if isinstance(value, str):
                raise TypeError(
                    f"family {self.name!r}: {label} must be a sequence of field "
                    f"names, not the string {value!r}."
                )
            normalized = tuple(value)
            if any(not isinstance(item, str) for item in normalized):
                raise TypeError(f"family {self.name!r}: every {label} entry must be a field name.")
            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    f"family {self.name!r}: {label} contains duplicate field names; "
                    "declare each dependency once."
                )
            unknown = sorted(set(normalized) - field_names)
            if unknown:
                raise ValueError(
                    f"family {self.name!r}: {label} names unknown member field "
                    f"{unknown[0]!r}; declare the field or remove the dependency."
                )
            return normalized

        reads_tuple = _field_tuple(reads, label="reads")
        weight_reads_tuple = _field_tuple(weight_reads, label="weight_reads")

        def decorate(fn: Callable) -> Callable:
            if not callable(fn):
                raise TypeError("graph_exchange must decorate a callable.")
            exchange_name = name or getattr(fn, "__name__", type(fn).__name__)
            _validate_graph_callable_is_three_arg(
                fn, where=f"family graph exchange {exchange_name!r}"
            )
            if any(item.name == exchange_name for item in self._graph_exchanges):
                raise ValueError(
                    f"family {self.name!r}: duplicate graph exchange name "
                    f"{exchange_name!r}; give each exchange a unique name."
                )
            self._graph_exchanges.append(
                FamilyGraphExchange(
                    name=exchange_name,
                    graph=graph,
                    writes=writes,
                    reads=reads_tuple,
                    weight_reads=weight_reads_tuple,
                    fn=fn,
                )
            )
            self._mark_dirty()
            return fn

        return decorate

    # -------------------------------------------------------------- summaries
    def _bucket_field_summaries(self, k: int) -> tuple[FamilyFieldSummary, ...]:
        """Per-member field summaries for bucket ``k`` (name / mass / size)."""
        return tuple(
            FamilyFieldSummary(name=v.name, mass=v.mass, size=int(v.size))
            for v in self._bucket_vars[k]
        )

    def describe(self) -> FamilySummary:
        """Return a summary of the family's declaration.

        Returns:
            A record of the family's `name`, `count`, parameter names `params`, and per size bucket its `count`, `first_member` and `fields`.
        """
        buckets = []
        start = 0
        for k, (node_name, (_dom, cnt)) in enumerate(zip(self._bucket_names(), self._buckets)):
            buckets.append(
                FamilyBucketSummary(
                    index=k,
                    group=node_name,
                    count=int(cnt),
                    first_member=start,
                    fields=self._bucket_field_summaries(k),
                )
            )
            start += int(cnt)
        return FamilySummary(
            name=self.name,
            count=self.count,
            params=tuple(sorted(self._params)),
            buckets=tuple(buckets),
        )

    def members(self) -> tuple[FamilyMemberSummary, ...]:
        """Return a summary of every member of the family.

        Returns:
            One record per member with its `index`, size `bucket`, `group`, member `fields`, and `params` values as floats.
        """
        out = []
        start = 0
        for k, (node_name, (_dom, cnt)) in enumerate(zip(self._bucket_names(), self._buckets)):
            fields = self._bucket_field_summaries(k)
            for i in range(int(cnt)):
                idx = start + i
                out.append(
                    FamilyMemberSummary(
                        index=idx,
                        bucket=k,
                        group=node_name,
                        fields=fields,
                        params={p: float(arr[idx]) for p, arr in self._params.items()},
                    )
                )
            start += int(cnt)
        return tuple(out)

    # ----------------------------------------------------------- couple sugar
    def field(self, name: str):
        """Return a member field as the source of a coupling out of the family.

        Call `.surface()` on the result for each member's value at its outer
        surface, the projection `spec.couple` takes. A ragged family raises
        `NotImplementedError`.

        Args:
            name: Member field declared with `field_decl`.

        Returns:
            The member-field projection.
        """
        from .problem_spec import _FieldProjection

        if self.is_ragged:
            raise NotImplementedError(
                f"family {self.name!r}: coupling a ragged family's field is "
                "not supported — the buckets carry different member sizes. "
                "Author one uniform family per size and couple each."
            )
        if not any(v.name == name for v in self._member_vars):
            raise ValueError(
                f"family {self.name!r} has no member field {name!r}; declare it "
                "with parts.field_decl(...) first"
            )
        return _FieldProjection(family_name=self.name, field_name=name)

    def couple(self, source: Any, **kwargs) -> Any:
        """Couple a member field out of the family, the same as `spec.couple`.

        Args:
            source: A field returned by `spec.field`, or a member projection such as `field(name).surface()`.
            **kwargs: Keywords of `spec.couple`, such as `onto`, `at` and `reduce`.

        Returns:
            The recipe builder, for chaining.
        """
        return self._spec.couple(source, **kwargs)

    # --------------------------------------------------------------- IR build
    def _build_member_equations(self, member_vars: tuple) -> tuple[Any, ...]:
        var_by_name = {v.name: v for v in member_vars}
        # Macro field handles (for cross-component gather reads) + globals are
        # resolvable by name so the member ``q`` accessor / IR inputs work.
        macro_vars = self._spec_macro_var_map()
        eqs: list[Any] = []
        for writes, reads, fn, eq_name in self._member_eq_specs:
            write_names = (writes,) if isinstance(writes, str) else tuple(writes)
            for w in write_names:
                if w not in var_by_name:
                    raise ValueError(
                        f"family {self.name!r} equation {eq_name!r} writes unknown "
                        f"member field {w!r}"
                    )
            # var_map handed to the member q-accessor: member fields by name,
            # plus any read that resolves to a macro field / global.
            var_map: dict[str, Any] = dict(var_by_name)
            inputs: list[Any] = []
            for r in reads:
                if r in var_by_name:
                    if var_by_name[r] not in inputs:
                        inputs.append(var_by_name[r])
                elif r in macro_vars:
                    var_map[r] = macro_vars[r]
                    if macro_vars[r] not in inputs:
                        inputs.append(macro_vars[r])
            if isinstance(writes, str):
                rhs = _wrap_member_derivative_return(
                    _make_member_rhs(fn, var_map),
                    equation_name=eq_name,
                    field_name=writes,
                    member_fields=tuple(var_by_name),
                )
                eqs.append(
                    Equation(
                        var=var_by_name[writes],
                        rhs_fn=rhs,
                        inputs=tuple(inputs),
                        name=eq_name,
                    )
                )
                continue
            # Multi-output member equation: one bundled closure writing several
            # member fields.  Any output residual may depend on any co-written
            # sibling (the bundle is one kernel), so union the written fields
            # into ``inputs`` even when the author's ``reads`` omits a sibling —
            # this declares the full writes-by-writes Jacobian block.
            for w in write_names:
                if var_by_name[w] not in inputs:
                    inputs.append(var_by_name[w])
            rhs = _MemberBundleGuard(inner=_make_member_rhs(fn, var_map), equation_name=eq_name)
            eqs.append(
                MultiEquation(
                    vars=tuple(var_by_name[w] for w in write_names),
                    rhs_fn=rhs,
                    inputs=tuple(inputs),
                    name=eq_name,
                )
            )
        return tuple(eqs)

    def _spec_macro_var_map(self) -> dict[str, Any]:
        """Resolve macro field + global names to IR handles for member reads."""
        out: dict[str, Any] = {}
        for fname, fspec in self._spec._fields.items():
            dom = self._spec._domains[self._spec._field_domains[fname]]
            shape = (
                tuple(fspec.shape) if fspec.shape is not None else (int(getattr(dom, "num_nodes")),)
            )
            out[fname] = Variable(name=fname, shape=shape, mass=fspec.mass)
        for gname, gspec in self._spec._globals.items():
            out[gname] = GlobalVariable(
                name=gname, init=jnp.asarray(gspec.initial), mass=gspec.mass
            )
        return out

    def _build_member_bcs(self, member_vars: tuple) -> tuple[Any, ...]:
        # Emit IR boundary-condition nodes. Re-uses the same ``build._emit_bc``
        # machinery that a component reaction-flux lowers through, resolving the
        # member-var name against the member Variable so the IR BC's ``.var``
        # is the member Variable.  The flux closure is then folded into the
        # per-member vmap kernel by the residual lowering.
        from .build import _emit_bc

        var_map = {v.name: v for v in member_vars}
        return tuple(_emit_bc(bc, var_map) for bc in self._member_bcs)

    def _sync(self) -> None:
        """(Re)materialise the frozen Family IR node(s) into the spec registry.

        A uniform family materialises one node under the family name; a ragged
        family materialises one node PER static size bucket, named
        ``"<family>[<bucket>]"``, each carrying that bucket's member domain and
        Variables, its slice of the per-member params (members are ordered
        bucket-major, matching the authored ``domain=`` sequence), and its
        running ``member_index_offset`` so ``m.member_index`` stays global.
        """
        # Family extracts flux callables into the non-static
        # ``member_flux_callables`` field, so Equinox does not see JAX arrays
        # behind the static ``member_bcs`` tuple.
        start = 0
        for k, (node_name, (dom, cnt)) in enumerate(zip(self._bucket_names(), self._buckets)):
            member_vars = tuple(self._bucket_vars[k])
            member_eqs = self._build_member_equations(member_vars) if member_vars else ()
            member_bcs = self._build_member_bcs(member_vars)
            params_k = (
                self._params
                if not self.is_ragged
                else {p: arr[start : start + cnt] for p, arr in self._params.items()}
            )
            fam = Family(
                name=node_name,
                count=cnt,
                domain=dom,
                member_variables=member_vars,
                member_equations=member_eqs,
                member_bcs=member_bcs,
                scale=self._scale,
                residual_scale=self._residual_scale,
                operator_factory=self._operator_factory,
                member_index_offset=start,
                member_params=params_k,
                graphs=tuple(self._graphs.values()),
                graph_exchanges=tuple(self._graph_exchanges),
                stencil_radius=self._stencil_radius,
            )
            self._spec._families[node_name] = fam
            start += cnt
        self._dirty = False


def _coerce_member_params(name: str, count: int, params: dict[str, Any]) -> dict[str, Any]:
    """Validate + coerce ``params`` to ``{name: (count,) array}``.

    Applies, in order: underscore-prefix reject, closure-cell-trap reject,
    leading-dim/scalar coercion, and a cross-param length-consistency check.
    """
    # Underscore-prefix reject runs first (it is a name guard, independent of
    # value shape).
    for pname in params:
        if str(pname).startswith("_"):
            raise ValueError(f"family {name!r}: param {pname!r} may not start with underscore")
    # Cross-param length consistency BEFORE per-param leading-dim coercion: if
    # two non-scalar params disagree on their raw per-member length (mixed
    # lengths), that is reported distinctly from a single-param leading-dim
    # mismatch.
    raw_lengths: dict[str, int] = {}
    for pname, value in params.items():
        ln = _raw_member_length(value)
        if ln is not None:
            raw_lengths[pname] = ln
    if len(set(raw_lengths.values())) > 1:
        a, b = _first_conflicting_pair(raw_lengths)
        raise ValueError(
            f"family {name!r}: params {a!r} and {b!r} have inconsistent "
            f"lengths {raw_lengths[a]} vs {raw_lengths[b]}; all per-member "
            f"arrays must be shape ({count},)"
        )
    out: dict[str, Any] = {}
    for pname, value in params.items():
        out[pname] = _coerce_one_param(name, count, pname, value)
    return out


def _raw_member_length(value: Any) -> Optional[int]:
    """Per-member (leading-axis) length of a param value, else ``None``.

    Returns ``None`` for a scalar / non-shaped value.  Used for the cross-param
    consistency pre-check.
    """
    if isinstance(value, (list, tuple)):
        if any(callable(v) for v in value):
            return None  # closure-cell trap handled downstream
        return len(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return None
    return int(arr.shape[0])


def _first_conflicting_pair(lengths: dict[str, int]) -> tuple[str, str]:
    items = list(lengths.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1] != items[j][1]:
                return items[i][0], items[j][0]
    return items[0][0], items[1][0]  # pragma: no cover - guarded by caller


def _coerce_one_param(name: str, count: int, pname: str, value: Any):
    """Coerce one param value to a ``(count,)``-leading array, or raise on invalid input."""
    # Reject a Python list/tuple of non-numeric objects (closure-cell trap):
    # callables / dicts / nested objects that ``jnp.asarray`` would refuse or
    # silently object-ify.
    if isinstance(value, (list, tuple)):
        if any(callable(v) for v in value):
            raise ValueError(
                f"family {name!r}: param {pname!r} is a Python list of "
                "per-member closures/objects (the closure-cell trap that "
                "silently reintroduces N traces); each member param must be a "
                f"single array leaf of shape ({count},) or a shared scalar."
            )
    elif callable(value):
        raise ValueError(
            f"family {name!r}: param {pname!r} is a callable (the closure-cell "
            "trap that silently reintroduces N traces); each member param must "
            f"be a single array leaf of shape ({count},) or a shared scalar."
        )
    try:
        arr = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"family {name!r}: param {pname!r} is not an array leaf (got "
            f"{type(value).__name__}); each member param must be a single array "
            f"leaf of shape ({count},) or a shared scalar."
        ) from exc
    if not jnp.issubdtype(arr.dtype, jnp.number):
        raise ValueError(
            f"family {name!r}: param {pname!r} is not numeric (dtype "
            f"{arr.dtype}); each member param must be a numeric array leaf of "
            f"shape ({count},) or a shared scalar."
        )
    if arr.ndim == 0:
        arr = jnp.broadcast_to(arr, (count,))
    elif arr.shape[0] != count:
        raise ValueError(
            f"family {name!r}: param {pname!r} must be a scalar or shape "
            f"({count},), got {tuple(arr.shape)}"
        )
    return arr


def _bucketize_domains(name: str, count: int, domains: Any) -> tuple:
    """Group a per-member domain sequence into contiguous identity runs.

    Each run of consecutive identical domain OBJECTS forms one static size
    bucket (one uniform member kernel).  A domain object reappearing after a
    different one would demand a member permutation the layout never performs,
    so it is rejected with an ordering instruction.
    """
    doms = list(domains)
    if len(doms) != int(count):
        raise ValueError(
            f"family {name!r}: domain= lists {len(doms)} member domains but "
            f"count={count}; give one domain per member (repeat the object for "
            "members sharing a geometry)."
        )
    buckets: list[list] = []
    seen: set[int] = set()
    for d in doms:
        if buckets and buckets[-1][0] is d:
            buckets[-1][1] += 1
            continue
        if id(d) in seen:
            raise ValueError(
                f"family {name!r}: domain= lists the same member domain in "
                "non-adjacent runs; order the members so all members sharing "
                "a domain are adjacent (one static size bucket per run)."
            )
        seen.add(id(d))
        buckets.append([d, 1])
    return tuple((d, c) for d, c in buckets)


def make_family(
    spec,
    name: str,
    *,
    count: int,
    domain: Any,
    params: Optional[dict[str, Any]] = None,
    scale: Optional[dict[str, float]] = None,
    residual_scale: Optional[dict[str, float]] = None,
    operators: Any = None,
    stencil_radius: Optional[int] = None,
) -> _FamilyHandle:
    """Build a [_FamilyHandle][diff_soup.problem._spec.family_builder._FamilyHandle] and register its [Family][diff_soup.problem._ir.family.Family] IR node(s).

    Implements the ``spec.family(...)`` facade including construction guards:

    * ``count <= 0`` -> ``ValueError``;
    * ``count > 256`` -> soft ``UserWarning`` (no hard ceiling);
    * name collision with a domain, field, or another family -> ``ValueError``;
    * ``[`` / ``]`` in the name -> ``ValueError`` (reserved for the
      ``"<family>[<bucket>]"`` group names of a ragged population);
    * the ``params`` taxonomy (mixed lengths, leading-dim mismatch, closure-cell
      trap, underscore prefix) via
      [_coerce_member_params][diff_soup.problem._spec.family_builder._coerce_member_params].

    ``domain`` is either ONE member domain shared by every member (uniform
    family) or a sequence of ``count`` per-member domains.  A sequence declares
    a RAGGED population: consecutive members sharing a domain object form one
    static size bucket, and each bucket lowers to its own population group
    (``"<family>[<bucket>]"``) with one uniform member kernel — members are
    never padded to a common size.  Members must be ordered bucket-major (all
    members of a bucket adjacent); ``params`` stay one ``(count,)`` array per
    name over the whole authored population.

    ``operators`` selects the spatial operator bank each member is lowered
    against (an operator-bank class or a top-level named callable
    ``domain -> bank``); it is threaded onto the
    [Family][diff_soup.problem._ir.family.Family] IR node's
    ``operator_factory``.  ``None`` selects the built-in bank.

    ``stencil_radius`` declares how far (in cells per axis of a structured
    member grid) a member kernel's Jacobian reaches; ``None`` declares the
    member domain's own radius-1 pattern.  It is threaded onto the IR node's
    ``stencil_radius`` and validated there.
    """
    if count <= 0:
        raise ValueError(f"family {name!r}: count must be a positive int, got {count}")
    if "[" in name or "]" in name:
        raise ValueError(
            f"family {name!r}: '[' and ']' are reserved for the "
            "'<family>[<bucket>]' group names of a ragged population; choose a "
            "name without brackets."
        )
    if name in spec._domains:
        raise ValueError(f"family {name!r}: name collides with an existing domain")
    if name in spec._fields:
        raise ValueError(f"family {name!r}: name collides with an existing field")
    if name in spec._families or name in spec._family_handles:
        raise ValueError(f"duplicate family name {name!r}")
    buckets = None
    if isinstance(domain, (list, tuple)):
        buckets = _bucketize_domains(name, int(count), domain)
        domain = buckets[0][0]
    if count > _SOFT_COUNT_CEILING:
        warn_user(
            f"family {name!r}: count={count} exceeds the soft ceiling of "
            f"{_SOFT_COUNT_CEILING}; large populations are legal but assemble a "
            "wide vmap batch — confirm this is intended.",
            UserWarning,
        )
    coerced = _coerce_member_params(name, int(count), dict(params or {}))
    return _FamilyHandle(
        spec,
        name=name,
        count=int(count),
        domain=domain,
        params=coerced,
        scale=scale,
        residual_scale=residual_scale,
        operators=operators,
        buckets=buckets,
        stencil_radius=stencil_radius,
    )
