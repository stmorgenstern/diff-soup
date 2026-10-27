# LOC > 500: one cohesive host-side builder — every declarative verb (domain/
# field/global/equation/bc/scales) shares the single ProblemSpec mutable-record
# state; splitting would fragment that state
# and the build-time walk that reads it.
"""ProblemSpec — mutable host-side builder; emits a frozen EquationSet on ``.to_equation_set()``.

The builder is the user-authoring entry point for the Recipe API.  It collects
declarative records ([FieldSpec][], [EquationSpec][], [BCSpec][],
...) and walks them in
[diff_soup.problem._spec.build._build_equation_set][] to produce a frozen
[diff_soup.problem._ir.EquationSet][].  [build][] then lowers that into
a solve-ready ``LoweredProblem``; the intermediate ``EquationSet`` stays
reachable via [to_equation_set][].

The cross-component coupling verbs and the family/population builder are fully
wired.  ``couple`` is the smart auto-phantom verb (GATHER / SCATTER direction
inferred from arg-0's kind), the explicit ``source=/target=/via=`` escape
hatch, and the NO-DOF read-through form ``couple(..., as_=)``.  ``family`` registers a [Family][] population via the
[diff_soup.problem._spec.family_builder][].  The lowering arms (residual /
sparsity / Schur) live in [diff_soup._lowering][].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional
from uuid import uuid4

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

from .specs import (
    BCSpec,
    ConstraintSpec,
    ControlSpec,
    CouplingSpec,
    EquationSpec,
    FieldSpec,
    FunctionSpec,
    GlobalVarSpec,
    MultiEquationSpec,
    ObservationSpec,
    ParamSpec,
)


class CouplingInferenceError(Exception):
    """Raised when ``spec.couple(...)`` cannot infer the transfer direction.

    The smart ``couple`` verb infers GATHER vs SCATTER and
    the matching [TransferOp][diff_soup.problem._ir.coupling.TransferOp] from the
    *kind* of arg-0 and ``onto=``.  When arg-0 is structurally ambiguous (e.g.
    a bare, un-projected family-field handle that could either be read from or
    written to), no single op is implied; this error lists the candidate ops
    and instructs the user to disambiguate with the explicit
    ``couple(source=, target=, via=)`` escape hatch.
    """


class _FieldProjection:
    """Host-side handle for a (projected) family-field coupling source.

    Returned by ``parts.field(name)`` and the projection methods
    [surface][_FieldProjection.surface] / [boundary][_FieldProjection.boundary] / [at][_FieldProjection.at].  Carries the owner
    family name, the member field name, the projection kind, and any
    projection keyword arguments.  A bare handle
    (``projection_kind is None``) is *not* a valid smart-couple source — it is
    structurally ambiguous and raises [CouplingInferenceError][]; a
    ``.surface()`` projection lowers to a many:1
    [FamilyScatterOp][diff_soup.problem._ir.coupling.FamilyScatterOp].
    """

    __slots__ = ("family_name", "field_name", "projection_kind", "proj_kwargs")

    def __init__(
        self,
        *,
        family_name: str,
        field_name: str,
        projection_kind: Optional[str] = None,
        **proj_kwargs: Any,
    ) -> None:
        self.family_name = str(family_name)
        self.field_name = str(field_name)
        self.projection_kind = projection_kind
        self.proj_kwargs = dict(proj_kwargs)

    def surface(self) -> "_FieldProjection":
        """The radial r=R surface trace (one surface DOF per member)."""
        return _FieldProjection(
            family_name=self.family_name,
            field_name=self.field_name,
            projection_kind="surface",
        )

    def boundary(self, tag: str) -> "_FieldProjection":
        """The boundary trace at ``tag``."""
        return _FieldProjection(
            family_name=self.family_name,
            field_name=self.field_name,
            projection_kind="boundary",
            tag=tag,
        )

    def at(self, idx: Any) -> "_FieldProjection":
        """A specific member-local DOF index projection."""
        return _FieldProjection(
            family_name=self.family_name,
            field_name=self.field_name,
            projection_kind="at",
            idx=idx,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        kind = self.projection_kind or "<none>"
        return f"_FieldProjection({self.family_name!r}.{self.field_name!r}, projection={kind})"


# A smart-couple arg-0 (``_CoupleSource``) is either a macro ``FieldSpec`` or a
# family-field ``_FieldProjection`` handle.
_COUPLE_SOURCE_TYPES = (FieldSpec, _FieldProjection)


class ProblemSpec:
    """Mutable builder that collects a problem's domains, unknowns, equations and conditions, then builds it.

    Open one with [`problem`][diff_soup.problem.problem]. Fields, globals,
    parameters and functions share one pool of names. `spec.bc`,
    `spec.constraint`, `spec.observe` and `spec.control` hold the remaining
    declaration methods.
    """

    def __init__(self, name: str) -> None:
        self.name = str(name)
        # Recipe-returned parameter/function records are public inference handles.
        # The visible Recipe name is not an identity: two independent builders
        # may deliberately reuse it. Keep one opaque scope for this builder so
        # its handles remain distinct for the builder's lifetime.
        self._parameter_namespace = f"{self.name}:{uuid4().hex}"
        self._domains: dict[str, Any] = {}
        # Per-domain operator-bank factory selected via ``domain(..., operators=)``.
        # Keyed by domain name; ``None`` (or an absent entry) selects the built-in
        # bank.  Stored as opaque structural configuration and threaded onto the
        # IR ``Component`` so the lowering resolver builds the bank from the
        # component's current domain.
        self._operator_factories: dict[str, Any] = {}
        # Grids scaled by a declared parameter (``geometry``): domain name ->
        # parameter name.  Threaded onto the IR ``Component`` so every kernel
        # evaluates on the grid scaled by the parameter's current value.
        self._geometry: dict[str, str] = {}
        self._fields: dict[str, FieldSpec] = {}
        self._field_domains: dict[str, str] = {}
        self._globals: dict[str, GlobalVarSpec] = {}
        self._params: dict[str, ParamSpec] = {}
        self._functions: dict[str, FunctionSpec] = {}
        self._equations: list[EquationSpec] = []
        self._multi_equations: list[MultiEquationSpec] = []
        self._bcs: list[BCSpec] = []
        self._constraints: list[ConstraintSpec] = []
        self._observations: list[ObservationSpec] = []
        self._controls: list[ControlSpec] = []
        self._couplings: list[CouplingSpec] = []
        # Family populations, keyed by name (``spec.family(...)`` registers via
        # the family_builder; smart ``couple`` consumes the registry to resolve
        # scatter member sizes / surface DOFs).
        self._families: dict[str, Any] = {}
        # Live ``_FamilyHandle`` per family name, so the read boundaries (couple
        # resolvers and build walk) can trigger the handle's lazy re-materialise
        # before consuming ``spec._families[name]``.  A family registered
        # directly into ``_families`` (e.g. IR-level test fixtures) carries no
        # handle here and is read as-is.
        self._family_handles: dict[str, Any] = {}
        # Smart-couple auto-phantoms whose ``onto=`` target is a FAMILY (gather
        # onto a population).  Keyed by phantom field name ->
        # ``(family_name, FieldSpec)``; routed to the family member Variable so
        # the gather ``Coupling`` endpoint resolves.  A domain-target phantom is
        # NOT recorded here — it lands in ``_fields``.
        self._family_phantoms: dict[str, Any] = {}
        # Domain-target smart-couple auto-phantoms (scatter ONTO a registered
        # domain).  These are algebraic component [FieldSpec][] records in
        # ``_fields`` that carry no user equation — the coupling injects their
        # value.  The build walk emits a zero-rhs algebraic Equation for each so
        # the coupling SET (gather) / scatter-ADD lands on a base of 0.
        # ``_phantom_sources`` maps the phantom field name -> its macro source
        # field name for phantom scale/offset propagation.
        self._phantom_fields: dict[str, Optional[str]] = {}
        self._phantom_sources: dict[str, str] = {}
        # Read-through aliases: ``couple(..., as_=<alias>)`` makes the source
        # value readable by name inside the consuming component's closure with no
        # state DOF / row / IC.  Keyed ``(consuming_component, alias_name) ->
        # alias Variable``; the alias is never added to ``_fields`` or any
        # ``Component.variables``.  A read-through alias is component-local —
        # the same alias name may be reused in a different consuming component.
        self._read_aliases: dict[tuple[str, str], Any] = {}
        # Non-dimensionalization bookkeeping (host-time).
        self._time_scale: Optional[float] = None
        self._equation_scales: dict[str, float] = {}
        # central scales() overrides (bare-name keyed; resolved at build time)
        self._central_state: dict[str, float] = {}
        self._central_state_offset: dict[str, float] = {}
        self._central_time: Optional[float] = None
        self._central_integrand: dict[str, float] = {}
        self._bc_ns: Any = None
        self._constraint_ns: Any = None
        self._observe_ns: Any = None
        self._control_ns: Any = None

    # ------------------------------------------------------------------ domains
    def domain(
        self,
        name: str,
        domain: Any,
        *,
        operators: Any = None,
    ) -> "ProblemSpec":
        """Register a domain under a name.

        Args:
            name: Domain name, unique in the recipe.
            domain: A grid or mesh from `diff_soup.domains`.
            operators: Operator-bank class or top-level named function of the domain, or `None` for the built-in operators.

        Returns:
            The builder, for chaining.
        """
        from diff_soup.problem._ir.component import validate_operator_factory

        if name in self._domains:
            raise ValueError(f"duplicate domain name {name!r}")
        # Validate at store time so a bad factory raises a clear error here at
        # the authoring callsite rather than later during lowering.
        validate_operator_factory(operators)
        for declaration in self._functions.values():
            validate_domain = getattr(declaration.fn, "_validate_domain", None)
            if callable(validate_domain):
                validate_domain(domain, name=name)
        self._domains[name] = domain
        self._operator_factories[name] = operators
        return self

    # --------------------------------------------------------------- declarations
    def _claim_name(self, name: str, kind: str, component: Optional[str] = None) -> None:
        """Reserve ``name`` for a new declaration of ``kind``, or refuse it.

        Fields, globals, params, and functions draw from a single name pool: an
        equation's ``reads=`` list holds bare strings with no namespace on them,
        so one identifier has to name exactly one declared object.  Fields and
        globals are read as ``q.<name>`` on top of that, so they may not take
        one of the names the query object answers itself.

        A read-through alias (``couple(..., as_=)``) holds its name inside the
        component it is read in: a field may not take it in that component,
        and a global, param or function, visible to every component, may not
        take it at all.

        Args:
            name: The identifier the declaration wants to claim.
            kind: ``"field"``, ``"global"``, ``"param"`` or ``"function"`` —
                the kind being declared, used both for the message and to decide
                whether the reserved-name rule applies.
            component: The domain a field is declared on; ``None`` for the
                other kinds.

        Raises:
            ValueError: If another declaration already holds ``name``, or if a
                field or global asks for a name reserved by the query object.
            ConfigurationError: If a read-through alias visible where the
                declaration is visible already holds ``name``.
        """
        from .q_accessor import RESERVED_Q_NAMES

        for held_kind, registry in (
            ("field", self._fields),
            ("global", self._globals),
            ("param", self._params),
            ("function", self._functions),
        ):
            if name in registry:
                raise ValueError(
                    f"{kind} {name!r}: the name {name!r} is already declared as a "
                    f"{held_kind}. Fields, globals, params, and functions share one "
                    f"name pool, because an equation's reads=({name!r},) carries no "
                    f"namespace to tell them apart. Give the {kind} a name no other "
                    "declaration uses."
                )
        for alias_comp, alias in self._read_aliases:
            if alias == name and (kind != "field" or alias_comp == component):
                raise ConfigurationError(
                    f"{kind} {name!r}: the name {name!r} is already declared as a "
                    f"read-through alias (couple(..., as_={name!r})) in component "
                    f"{alias_comp!r}, where q.{name} reads the coupled value. Give "
                    f"the {kind} a name no declaration of that component holds."
                )
        if kind in ("field", "global") and name in RESERVED_Q_NAMES:
            reserved = ", ".join(sorted(RESERVED_Q_NAMES))
            raise ValueError(
                f"{kind} {name!r}: {name!r} is reserved by the equation query object, "
                f"which answers q.{name} out of its own machinery, so a {kind} named "
                f"{name!r} could never be read back as q.{name}. The reserved names "
                f"are {reserved}. Give the {kind} a name outside that set."
            )

    def field(
        self,
        name: str,
        *,
        mass: str,
        shape: Optional[tuple[int, ...]] = None,
        components: Optional[int] = None,
        bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
        domain: Optional[str] = None,
        scale: Optional[float] = None,
        offset: Optional[float] = None,
        residual_scale: Optional[float] = None,
        init: Optional[Callable[..., Any]] = None,
        quad: Optional[bool] = None,
    ) -> FieldSpec:
        """Declare a field, an unknown with values on a domain.

        The dimensional value is `offset + scale * internal`. Omitting `scale`
        defaults it to 1.0 with a warning at build time.

        Args:
            name: Field name, unique among fields, globals, parameters and functions.
            mass: `"differential"` for a field with a time derivative, `"algebraic"` for one without.
            shape: Array shape of the field, or `None` for one value per node.
            components: Number of values per node of a vector field, exclusive with `shape`.
            bounds: Reserved, and any value other than `None` raises `NotImplementedError`.
            domain: Name of the field's domain, required when more than one domain is registered.
            scale: State scale, nonzero.
            offset: State offset.
            residual_scale: Divisor of the field's residual rows, nonzero.
            init: Default initial condition, a callable taking one coordinate array per spatial dimension.
            quad: On a finite-element domain, whether `q.<name>` holds quadrature-point values rather than the nodal array, `None` meaning true there.

        Returns:
            The declared field, which `spec.couple` takes as a source.
        """
        if init is not None and not callable(init):
            raise TypeError(
                f"field({name!r}, init=...): init must be a callable "
                "init(*xs) taking one coordinate array per spatial dim, "
                f"got {type(init).__name__}"
            )
        if scale is not None and float(scale) == 0.0:
            raise ValueError(
                f"field({name!r}, scale=0.0): state scale must be non-zero "
                "(it divides the dimensional state; a zero scale is a singular "
                "non-dimensionalization)"
            )
        if bounds is not None:
            raise NotImplementedError(
                f"field({name!r}, bounds=...): field bounds are not yet wired "
                "into lowering and would be silently ignored. The timestepper "
                "uses Newton as its corrector; pluggable and box-constrained "
                "correctors are not available in this version."
            )
        if domain is None:
            if len(self._domains) != 1:
                raise ValueError("field(): provide domain= when multiple domains are registered")
            domain = next(iter(self._domains))
        if domain not in self._domains:
            raise ValueError(f"unknown domain {domain!r}")
        self._claim_name(name, "field", component=domain)
        # FEM fields are quad-native by default.  ``None`` (the default) resolves
        # to ``True`` on an FEM domain and ``False`` on a grid (FVM/FDM domains
        # have no quadrature, so the flag is inert there).  An explicit
        # ``quad=True/False`` is honoured verbatim.
        if quad is None:
            is_fem = getattr(self._domains[domain], "discretization_type", None) == "fem"
            quad = is_fem
        if components is not None:
            if shape is not None:
                raise ValueError(f"field({name!r}): pass exactly one of components= or shape=")
            num_nodes = int(self._domains[domain].num_nodes)
            shape = (num_nodes, int(components))
        spec = FieldSpec(
            name=name,
            mass=mass,
            shape=shape,
            bounds=bounds,
            scale=None if scale is None else float(scale),
            offset=None if offset is None else float(offset),
            residual_scale=(None if residual_scale is None else float(residual_scale)),
            init=init,
            quad=bool(quad),
        )
        self._fields[name] = spec
        self._field_domains[name] = domain
        return spec

    def global_var(
        self,
        name: str,
        *,
        initial: Any = 0.0,
        mass: str = "algebraic",
        scale: Optional[float] = None,
    ) -> GlobalVarSpec:
        """Declare a global, a scalar or one-dimensional unknown with no domain.

        An algebraic global is closed by exactly one constraint or driven by a
        control. A differential global is written by exactly one equation.

        Args:
            name: Global name, unique among fields, globals, parameters and functions.
            initial: Initial value, a scalar or one-dimensional array, which seeds the solve.
            mass: `"algebraic"` or `"differential"`.
            scale: State scale, nonzero, or `None` for 1.

        Returns:
            The declared global.
        """
        self._claim_name(name, "global")
        if scale is not None and float(scale) == 0.0:
            raise ValueError(f"global_var({name!r}, scale=0.0): state scale must be non-zero")
        initial_array = jnp.asarray(initial)
        if initial_array.ndim > 1:
            raise ValueError(
                f"global_var({name!r}): initial must be a scalar or one-dimensional "
                f"value, got shape {initial_array.shape}"
            )
        spec = GlobalVarSpec(
            name=name,
            initial=initial,
            mass=mass,
            scale=None if scale is None else float(scale),
        )
        self._globals[name] = spec
        return spec

    def param(
        self,
        name: str,
        *,
        value: Any,
        learnable: bool = False,
        bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
        prior: Any = None,
    ) -> ParamSpec:
        """Declare a parameter, read in closures as `q.p.<name>`.

        Args:
            name: Parameter name, unique among fields, globals, parameters and functions.
            value: Nominal value, a scalar or an array, never a callable.
            learnable: Whether a fit may select the parameter, which then needs a floating-point value.
            bounds: Bounds `(lo, hi)` for a fit, with `None` for an open side.
            prior: Prior for MAP estimation.

        Returns:
            The declared parameter, by which a fit selects it.
        """
        self._claim_name(name, "param")
        if callable(value) or isinstance(value, eqx.Module):
            raise TypeError(
                f"param({name!r}, value=...): value must hold an array value "
                "(a scalar, list, NumPy array, or JAX array), not a callable "
                f"or eqx.Module (got {type(value).__name__}). A quantity that "
                "depends on time or state is declared with spec.function("
                f"{name!r}, fn, ...) and called explicitly in the equation as "
                f"q.fn.{name}(q.t)."
            )
        if bounds is not None:
            try:
                lo, hi = bounds
                lo = None if lo is None else float(lo)
                hi = None if hi is None else float(hi)
            except (TypeError, ValueError):
                raise TypeError(
                    f"param({name!r}, bounds=...): bounds must be a (lo, hi) "
                    "pair of numbers, with None for an unbounded side (got "
                    f"{bounds!r})."
                ) from None
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(
                    f"param({name!r}, bounds=({lo}, {hi})): the lower bound "
                    "must be strictly below the upper bound."
                )
            bounds = (lo, hi)
        if learnable:
            # A fresh array per declaration: two learnable parameters written
            # from one array are still two unknowns, and every selector that
            # keys on leaf identity must be able to tell them apart.
            value = jnp.array(value)
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                raise TypeError(
                    f"learnable param {name!r} must be an inexact scalar/array, "
                    f"got dtype {value.dtype}"
                )
        spec = ParamSpec(
            name=name,
            value=value,
            learnable=learnable,
            bounds=bounds,
            prior=prior,
            namespace=self._parameter_namespace,
        )
        self._params[name] = spec
        return spec

    def geometry(
        self,
        name: str,
        *,
        domain: Optional[str] = None,
        value: float = 1.0,
        learnable: bool = False,
        bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> ParamSpec:
        """Declare a parameter that sets the size of a one-dimensional grid.

        The grid is declared with outer coordinate 1, and its coordinates are
        multiplied by the parameter at every evaluation. Closures read the size as
        `q.p.<name>`, and a learnable size needs a finite positive lower bound.

        Args:
            name: Parameter name.
            domain: Name of the grid, required when more than one domain is registered.
            value: Size, finite and positive.
            learnable: Whether a fit may select the size.
            bounds: Bounds `(lo, hi)` for a fit.

        Returns:
            The declared parameter, by which a fit selects it.
        """
        call = f"spec.geometry({name!r})"
        if domain is None:
            if len(self._domains) != 1:
                declared = sorted(self._domains)
                raise ValueError(
                    f"{call}: domain= names the grid the scale applies to, and may be "
                    f"omitted only when exactly one domain is declared; {len(declared)} "
                    f"are declared ({declared}). Declare the grid with spec.domain(...) "
                    "and pass its name as domain=."
                )
            domain = next(iter(self._domains))
        if domain not in self._domains:
            raise ValueError(
                f"{call}: no domain named {domain!r} is declared (declared: "
                f"{sorted(self._domains)}). Declare the grid with spec.domain(...) "
                "before scaling it."
            )
        grid = self._domains[domain]
        # A Grid1D is recognised by its capabilities: the problem layer does not
        # import the domains layer.
        if (
            not callable(getattr(grid, "scaled", None))
            or getattr(grid, "coord_system", None) is None
        ):
            raise ValueError(
                f"{call}: domain {domain!r} is a {type(grid).__name__}, and a scale "
                "applies only to a one-dimensional grid, a Grid1D (cartesian_grid_1d, "
                "cylindrical_grid_1d or spherical_grid_1d). Build any other domain at "
                "its size, without spec.geometry."
            )
        factory = self._operator_factories.get(domain)
        if factory is not None:
            raise ValueError(
                f"{call}: domain {domain!r} was declared with operators="
                f"{getattr(factory, '__qualname__', factory)!r}, and a scaled grid is "
                "evaluated with the built-in operators only. Declare the domain "
                "without operators=, or build the grid at its size without spec.geometry."
            )
        held = self._geometry.get(domain)
        if held is not None:
            raise ValueError(
                f"{call}: domain {domain!r} is already scaled by {held!r}, and a grid "
                "takes one scale. Read that scale as q.p." + held + " instead."
            )
        outer = float(grid.bounds[0][1])
        if abs(outer - 1.0) > 1e-12:
            raise ValueError(
                f"{call}: domain {domain!r} ends at {outer!r}, but a scaled grid is "
                f"declared as a unit grid, with outer coordinate 1, and {name!r} sets "
                f"its size. Build the grid with outer coordinate 1 and give the size "
                f"as the value of {name!r}."
            )
        try:
            size = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"spec.geometry({name!r}, value={value!r}): the size must be a finite "
                "positive number."
            ) from None
        if not (np.isfinite(size) and size > 0.0):
            raise ValueError(
                f"spec.geometry({name!r}, value={value!r}): {name!r} multiplies the "
                "grid's coordinates, so its value must be a finite positive number."
            )
        if learnable:
            lower = None
            try:
                lower = None if bounds is None or bounds[0] is None else float(bounds[0])
            except (TypeError, ValueError, IndexError):
                lower = None
            if lower is None or not (np.isfinite(lower) and lower > 0.0):
                raise ValueError(
                    f"spec.geometry({name!r}, learnable=True, bounds={bounds!r}): a fitted "
                    "size needs bounds=(lo, hi) with a finite positive lower bound lo, so "
                    f"that no iterate makes {name!r} zero or negative."
                )
        declaration = self.param(name, value=size, learnable=learnable, bounds=bounds)
        self._geometry[domain] = name
        return declaration

    def function(
        self,
        name: str,
        fn: Any,
        *,
        learnable: bool = False,
        prior: Any = None,
        pointwise: bool = False,
    ) -> FunctionSpec:
        """Declare a named function, called in closures as `q.fn.<name>(...)`.

        Args:
            name: Function name, unique among fields, globals, parameters and functions.
            fn: The callable, such as a plain function or an `eqx.Module`.
            learnable: Whether a fit may select the function's array leaves.
            prior: Prior for MAP estimation.
            pointwise: Whether to apply `fn` at every node, for a callable that does not broadcast.

        Returns:
            The declared function, by which a fit selects it.
        """
        self._claim_name(name, "function")
        if not callable(fn):
            raise TypeError(f"function({name!r}): expected a callable, got {type(fn).__name__}")
        validate_domain = getattr(fn, "_validate_domain", None)
        if callable(validate_domain):
            for domain_name, domain in self._domains.items():
                validate_domain(domain, name=domain_name)
        spec = FunctionSpec(
            name=name,
            fn=fn,
            learnable=learnable,
            prior=prior,
            namespace=self._parameter_namespace,
            pointwise=bool(pointwise),
        )
        self._functions[name] = spec
        return spec

    # -------------------------------------------------------------------- family
    def family(
        self,
        name: str,
        *,
        count: int,
        domain: Any,
        params: Optional[dict[str, Any]] = None,
        scale: Optional[dict[str, float]] = None,
        residual_scale: Optional[dict[str, float]] = None,
        operators: Any = None,
        stencil_radius: Optional[int] = None,
    ):
        """Declare a population of `count` identical members that differ only in parameter values.

        The returned family declares the member template. Member equations take
        `(q, m)`, with `m.<name>` the member's entry of a parameter in `params`.

        Args:
            name: Family name, distinct from every domain and field.
            count: Number of members, positive.
            domain: Member domain, or a sequence of `count` domains for a ragged family.
            params: Per-member parameter arrays of shape `(count,)`, or shared scalars.
            scale: State scale of each member field, keyed by field name.
            residual_scale: Residual divisor of each member field, keyed by field name.
            operators: Operator-bank class or top-level named function of the domain, or `None` for the built-in operators.
            stencil_radius: Jacobian reach of a member kernel in cells per axis, or `None` for one cell.

        Returns:
            The family, whose methods declare the member fields, equations and boundary conditions.
        """
        from .family_builder import make_family

        return make_family(
            self,
            name,
            count=count,
            domain=domain,
            params=params,
            scale=scale,
            residual_scale=residual_scale,
            operators=operators,
            stencil_radius=stencil_radius,
        )

    # ------------------------------------------------------------------ equation
    @property
    def equation(self):
        """Decorator that registers a closure as the equation of one field.

        Use it as `@spec.equation(writes=..., reads=..., name=None)`, where `writes`
        names the field, `reads` lists every field, global and coupled alias the
        closure reads, and `name` defaults to the function's name. The closure
        returns `d(field)/dt`, or an expression containing `q.dt(field)` that equals
        zero.
        """
        from .decorators import _EquationDecorator

        return _EquationDecorator(self)

    @property
    def equations(self):
        """Decorator that registers one closure as the equations of several fields.

        Use it as `@spec.equations(writes=(...), reads=..., name=None)`, with at
        least two fields in `writes`. The closure returns a dict keyed by field name.
        """
        from .decorators import _MultiEquationDecorator

        return _MultiEquationDecorator(self)

    # --------------------------------------------------------------- namespaces
    @property
    def bc(self):
        """Lazy [_BCNamespace][diff_soup.problem._spec.bcs_namespace._BCNamespace]; use as ``spec.bc.dirichlet(...)``."""
        if self._bc_ns is None:
            from .bcs_namespace import _BCNamespace

            self._bc_ns = _BCNamespace(self)
        return self._bc_ns

    @property
    def constraint(self):
        """Lazy [_ConstraintNamespace][diff_soup.problem._spec.constraints_namespace._ConstraintNamespace]."""
        if self._constraint_ns is None:
            from .constraints_namespace import _ConstraintNamespace

            self._constraint_ns = _ConstraintNamespace(self)
        return self._constraint_ns

    @property
    def observe(self):
        """Lazy [_ObserveNamespace][diff_soup.problem._spec.observe_namespace._ObserveNamespace]."""
        if self._observe_ns is None:
            from .observe_namespace import _ObserveNamespace

            self._observe_ns = _ObserveNamespace(self)
        return self._observe_ns

    @property
    def control(self):
        """Lazy [_ControlNamespace][diff_soup.problem._spec.control_namespace._ControlNamespace]."""
        if self._control_ns is None:
            from .control_namespace import _ControlNamespace

            self._control_ns = _ControlNamespace(self)
        return self._control_ns

    # ----------------------------------------------------------------- coupling
    def couple(
        self,
        source: Any = None,
        *,
        target: Optional[FieldSpec] = None,
        via: Any = None,
        onto: Optional[str] = None,
        at: Any = None,
        reduce: Optional[str] = None,
        as_: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "ProblemSpec":
        """Couple a field's values into another component or family.

        A field coupled onto a family gives member `m` the value `source[at[m]]`. A
        member projection coupled onto a domain places one value per member at the
        nodes `at`. With `as_`, the domain `onto` reads the coupled value as
        `q.<as_>`.

        Args:
            source: A field returned by `spec.field`, or a member projection such as `family.field("c").surface()`.
            target: Receiving field returned by `spec.field`, in the explicit form without `onto`.
            via: A [`TransferOp`][diff_soup.problem.TransferOp] to apply instead of the default index transfer.
            onto: Name of the receiving domain or family.
            at: Node indices, or a boolean mask, of the nodes the values come from or go to.
            reduce: How members landing on one node merge, `"sum"` (default), `"mean"`, or `"none"` to refuse a collision.
            as_: Name under which the domain `onto` reads the value as `q.<as_>`.
            name: Coupling label, or `None` for `"couple_<n>"`.

        Returns:
            The builder, for chaining.
        """
        if onto is None:
            # Explicit keyword form (escape hatch).  ``reduce=`` is only
            # meaningful for the smart scatter path.
            return self._couple_explicit(source=source, target=target, via=via, name=name)
        if as_ is not None:
            return self._couple_read_through(
                source,
                onto=onto,
                at=at,
                reduce=reduce,
                as_=as_,
                via=via,
                name=name,
            )
        return self._couple_smart(source, onto=onto, at=at, reduce=reduce, name=name)

    def _couple_explicit(
        self,
        *,
        source: FieldSpec,
        target: FieldSpec,
        via: Any,
        name: Optional[str],
    ) -> "ProblemSpec":
        """Validate and record an explicit (escape-hatch) coupling specification."""
        if not isinstance(source, FieldSpec):
            raise TypeError(
                f"spec.couple(source=): expected FieldSpec, got {type(source).__name__}"
            )
        if not isinstance(target, FieldSpec):
            raise TypeError(
                f"spec.couple(target=): expected FieldSpec, got {type(target).__name__}"
            )
        return self._record_coupling(source, target, via, name)

    def _record_coupling(
        self, source: Any, target: Any, via: Any, name: Optional[str]
    ) -> "ProblemSpec":
        """Append a validated [CouplingSpec][] to the spec's coupling registry."""
        if name is None:
            name = f"couple_{len(self._couplings)}"
        if any(c.name == name for c in self._couplings):
            raise ValueError(f"duplicate coupling name {name!r}")
        self._couplings.append(CouplingSpec(source=source, target=target, via=via, name=name))
        return self

    def _couple_smart(
        self,
        source: Any,
        *,
        onto: str,
        at: Any,
        reduce: Optional[str],
        name: Optional[str],
    ) -> "ProblemSpec":
        """Desugar the smart ``couple(field, onto=, at=, reduce=)`` form.

        Infers direction + op from arg-0's kind (FieldSpec = GATHER,
        _FieldProjection = SCATTER), then forwards to the explicit
        ``source=/target=/via=`` recorder.
        """
        if not isinstance(source, _COUPLE_SOURCE_TYPES):
            raise TypeError(
                "spec.couple(source): expected a FieldSpec or a family-field "
                f"projection, got {type(source).__name__}"
            )
        if isinstance(source, FieldSpec):
            return self._couple_gather(source, onto=onto, at=at, reduce=reduce, name=name)
        return self._couple_scatter(source, onto=onto, at=at, reduce=reduce, name=name)

    # ----------------------------------------------- read-through couplings
    def _register_read_alias(self, consuming_comp: str, alias: str, size: int):
        """Build + register a NO-DOF read-through alias Variable.

        The alias is keyed ``(consuming_comp, alias)`` — component-local, so the
        same alias name may be reused in a different consuming component, and a
        field of another component may share it.  A field of the consuming
        component, a global, a param, a function, or a name ``q`` answers
        itself may not, and raises a
        [ConfigurationError][diff_soup._errors.ConfigurationError].  It is
        NEVER added to ``_fields`` / a ``Component.variable`` (layout allocates
        no DOF).  Returns the alias
        [Variable][diff_soup.problem._ir.variable.Variable].
        """
        from diff_soup.problem._ir.variable import Variable

        from .q_accessor import RESERVED_Q_NAMES

        held = None
        if alias in self._fields and self._field_domains.get(alias) == consuming_comp:
            held = "field"
        for kind, registry in (
            ("global", self._globals),
            ("param", self._params),
            ("function", self._functions),
        ):
            if alias in registry:
                held = kind
        if alias in RESERVED_Q_NAMES:
            raise ConfigurationError(
                f"spec.couple(as_={alias!r}, onto={consuming_comp!r}): {alias!r} is "
                f"reserved by the equation query object, which answers q.{alias} "
                "itself. as_= must be a name outside "
                f"{', '.join(sorted(RESERVED_Q_NAMES))}."
            )
        if held is not None:
            raise ConfigurationError(
                f"spec.couple(as_={alias!r}, onto={consuming_comp!r}): the alias name "
                f"{alias!r} is already declared as a {held} in component "
                f"{consuming_comp!r}, so q.{alias} could not tell the two apart. as_= "
                "must be a name no declaration of that component holds."
            )
        key = (consuming_comp, alias)
        existing = self._read_aliases.get(key)
        if existing is not None:
            if int(existing.size) != int(size):
                raise ValueError(
                    f"spec.couple(as_={alias!r}, onto={consuming_comp!r}): the "
                    f"alias is already registered with size {int(existing.size)} "
                    f"but this couple delivers size {int(size)}; a read-through "
                    "alias is single-valued per consuming component."
                )
            return existing
        var = Variable(name=alias, shape=(int(size),), mass="algebraic")
        self._read_aliases[key] = var
        return var

    def _couple_read_through(
        self,
        source: Any,
        *,
        onto: str,
        at: Any,
        reduce: Optional[str],
        as_: str,
        via: Any,
        name: Optional[str],
    ) -> "ProblemSpec":
        """Record a NO-DOF read-through coupling.

        ``onto`` must be a registered domain (the consuming component): a
        read-through alias is injected into a component closure's MergedView,
        never into a family member kernel.  Source kinds and alias shape:

        * a macro [FieldSpec][] (component->component).  Default op is a
          compact [NodalExtractOp][diff_soup.problem._ir.coupling.NodalExtractOp]
          (``source[at]`` -> alias of size ``len(at)``).  Pass ``via=`` a
          [NodalInjectOp][diff_soup.problem._ir.coupling.NodalInjectOp] to instead
          deliver a full ``onto.num_nodes`` array (``source`` placed at the op's
          ``target_indices``, 0 elsewhere).
        * a [_FieldProjection][] ``.surface()`` (family->component).  A
          [FamilyScatterOp][diff_soup.problem._ir.coupling.FamilyScatterOp] places each
          member's surface DOF at ``at`` in a full ``onto.num_nodes`` array.
        """
        from diff_soup.problem._ir.coupling import (
            FamilyScatterOp,
            NodalExtractOp,
            Sum,
            TransferOp,
        )

        if onto not in self._domains:
            raise ValueError(
                f"spec.couple(onto={onto!r}, as_={as_!r}): a read-through alias "
                "is injected into a component closure, so onto= must be a "
                "registered DOMAIN (not a family)."
            )
        tgt_domain = self._domains[onto]
        n_target = int(getattr(tgt_domain, "num_nodes"))
        if isinstance(source, FieldSpec):
            if reduce is not None:
                raise ValueError(
                    "spec.couple: reduce= is only valid for a family-source "
                    "(surface) read-through coupling"
                )
            src_var = self._fields[source.name]
            if via is not None:
                # Explicit transfer op. Any TransferOp works here — the op
                # itself declares its alias length, application, and sparsity;
                # no specific built-in subclass is required.
                if not isinstance(via, TransferOp):
                    raise TypeError(
                        "spec.couple(as_=, via=): a component-source "
                        "read-through op must be a TransferOp declaring "
                        "alias_size, apply, and sparsity; got "
                        f"{type(via).__name__}"
                    )
                src_domain = self._domains[self._field_domains[source.name]]
                n_source = int(getattr(src_domain, "num_nodes"))
                alias_size = int(via.alias_size(n_source=n_source, n_target=n_target))
                alias_var = self._register_read_alias(onto, as_, alias_size)
                return self._record_read_through(src_var, alias_var, via, name, onto=onto)
            # Default COMPACT gather: each target node reads source[at].
            src_domain = self._domains[self._field_domains[source.name]]
            src_idx = self._check_at_range(at, src_domain, source.name)
            at_indices = tuple(int(i) for i in src_idx.tolist())
            op = NodalExtractOp(source_indices=at_indices)
            alias_var = self._register_read_alias(onto, as_, len(at_indices))
            return self._record_read_through(src_var, alias_var, op, name, onto=onto)
        if not isinstance(source, _FieldProjection):
            raise TypeError(
                "spec.couple(source): a read-through coupling expects a "
                f"FieldSpec or a family-field projection, got "
                f"{type(source).__name__}"
            )
        # family->component (surface) read-through -> FULL onto-size alias.
        if source.projection_kind is None:
            raise CouplingInferenceError(
                f"spec.couple(as_={as_!r}): the bare family-field handle "
                f"{source.family_name!r}.{source.field_name!r} is ambiguous; "
                "project it (e.g. .surface()) before a read-through couple."
            )
        if source.projection_kind != "surface":
            raise NotImplementedError(
                f"spec.couple: family-field projection "
                f"{source.projection_kind!r} is not wired for read-through; "
                "only .surface() is supported."
            )
        idx = self._check_at_range(at, tgt_domain, onto)
        at_indices = tuple(int(i) for i in idx.tolist())
        if reduce is None:
            reduce = "sum"
        if reduce not in ("sum", "mean", "none"):
            raise ValueError(f"spec.couple(reduce={reduce!r}): expected 'sum', 'mean' or 'none'")
        member_var, n_member_dof, member_surface_dof = self._resolve_family_member(source)
        # ``reduce='none'`` is the typed ``Sum()`` + ``injective=True``.  After
        # the host-side collision guard, build the op with that pair.
        injective = reduce == "none"
        op = FamilyScatterOp(
            at_indices=at_indices,
            member_surface_dof=member_surface_dof,
            n_member_dof=n_member_dof,
            reduce=Sum() if injective else reduce,
            injective=injective,
        )
        if injective:
            self._guard_scatter_collision(at_indices)
        alias_var = self._register_read_alias(onto, as_, n_target)
        return self._record_read_through(member_var, alias_var, op, name, onto=onto)

    def _record_read_through(
        self, source: Any, alias_var: Any, op: Any, name: Optional[str], *, onto: str
    ) -> "ProblemSpec":
        """Append a ``read_through=True`` [CouplingSpec][] (alias target).

        The ``target`` is the alias Variable (BY IDENTITY) so the build walk
        resolves it without a ``var_map`` lookup.  ``read_through=True`` rides
        the spec so ``build._emit_coupling`` stamps the IR ``Coupling``.  ``onto``
        (the consuming component) is carried as ``target_owner`` so an alias read
        only by a boundary-condition callback still resolves its owner at lowering.
        """
        if name is None:
            name = f"couple_{len(self._couplings)}"
        if any(c.name == name for c in self._couplings):
            raise ValueError(f"duplicate coupling name {name!r}")
        self._couplings.append(
            CouplingSpec(
                source=source,
                target=alias_var,
                via=op,
                name=name,
                read_through=True,
                target_owner=onto,
            )
        )
        return self

    def _couple_gather(
        self,
        source: FieldSpec,
        *,
        onto: str,
        at: Any,
        reduce: Optional[str],
        name: Optional[str],
    ) -> "ProblemSpec":
        from diff_soup.problem._ir.coupling import NodalExtractOp

        if reduce is not None:
            raise ValueError(
                "spec.couple: reduce= is only valid for scatter (family-source) couplings"
            )
        self._reject_ragged_family_couple(onto)
        if onto not in self._domains and onto not in self._families:
            raise ValueError(f"spec.couple(onto={onto!r}): unknown domain or family")
        # GATHER reads the SOURCE field per member: each member reads
        # ``source[at_m]``.  Range-check ``at`` against the source field's
        # domain node count.
        src_domain = self._domains[self._field_domains[source.name]]
        idx = self._check_at_range(at, src_domain, source.name)
        op = NodalExtractOp(source_indices=tuple(idx.tolist()))
        if onto in self._families:
            # GATHER onto a FAMILY: the per-member ``NodalExtractOp`` makes
            # ``source`` readable inside every member's kernel (``view[source]``).
            # The family-residual lowering keys the gathered value under the macro
            # source Variable, so the coupling target must resolve — by identity —
            # to a member Variable of the ``onto`` family (routing it through the
            # family gather branch in ``_resolve_family_couplings``).  A
            # deterministic ``{source}_at_{onto}`` phantom is still registered
            # (family-owned) for uniform naming.
            target = self._family_gather_target(source, onto)
            return self._record_coupling(source, target, op, name)
        # GATHER onto a DOMAIN (component->component): use the NO-DOF
        # read-through form instead of materialising an auto-phantom recipient
        # DOF.
        raise ValueError(
            f"spec.couple({source.name!r}, onto={onto!r}): gathering a component "
            "field onto another DOMAIN uses the NO-DOF read-through form — "
            f"add as_=<alias> (e.g. spec.couple({source.name!r}, onto={onto!r}, "
            f"at=..., as_={source.name!r})) and read it as q.<alias> in the "
            f"{onto!r} closures."
        )

    def _couple_scatter(
        self,
        proj: "_FieldProjection",
        *,
        onto: str,
        at: Any,
        reduce: Optional[str],
        name: Optional[str],
    ) -> "ProblemSpec":
        from diff_soup.problem._ir.coupling import FamilyScatterOp, Sum

        if proj.projection_kind is None:
            # A bare (un-projected) family handle is structurally ambiguous —
            # it could gather (read) or scatter (write); no single op is implied.
            raise CouplingInferenceError(
                f"spec.couple: cannot infer a transfer direction for the bare "
                f"family-field handle {proj.family_name!r}.{proj.field_name!r}; "
                "it is ambiguous between a SCATTER (FamilyScatterOp) and a "
                "GATHER (NodalExtractOp). Project it (e.g. .surface()) or use "
                "the explicit couple(source=, target=, via=) escape hatch."
            )
        if proj.projection_kind != "surface":
            raise NotImplementedError(
                f"spec.couple: family-field projection "
                f"{proj.projection_kind!r} is not yet supported for scatter; "
                "only.surface() is wired."
            )
        if reduce is None:
            reduce = "sum"
        if reduce not in ("sum", "mean", "none"):
            raise ValueError(f"spec.couple(reduce={reduce!r}): expected 'sum', 'mean' or 'none'")
        # SCATTER injects into the target domain: range-check ``at``
        # against the target domain node count.
        if onto not in self._domains:
            raise ValueError(f"spec.couple(onto={onto!r}): unknown domain or family")
        tgt_domain = self._domains[onto]
        idx = self._check_at_range(at, tgt_domain, onto)
        at_indices = tuple(idx.tolist())
        member_var, n_member_dof, member_surface_dof = self._resolve_family_member(proj)
        # ``reduce='none'`` is the typed ``Sum()`` + ``injective=True`` (an
        # injectivity assertion, not a sum-over-contributions divisor).
        injective = reduce == "none"
        op = FamilyScatterOp(
            at_indices=at_indices,
            member_surface_dof=member_surface_dof,
            n_member_dof=n_member_dof,
            reduce=Sum() if injective else reduce,
            injective=injective,
        )
        if injective:
            self._guard_scatter_collision(at_indices)
        phantom = self._auto_phantom_named(f"{proj.family_name}_{proj.field_name}_at_{onto}", onto)
        return self._record_coupling(member_var, phantom, op, name)

    # --------------------------------------------------- smart-couple helpers
    @staticmethod
    def _check_at_range(at: Any, domain: Any, owner: str) -> "np.ndarray":
        """Range-check ``at`` against ``domain.num_nodes`` (mirror BC check).

        Returns the flattened int64 index array on success; raises
        ``ValueError`` naming ``owner`` and the valid range on an
        out-of-range index.
        """
        arr = np.asarray(at)
        if arr.dtype == bool:
            arr = np.nonzero(arr.reshape(-1))[0]
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        n = getattr(domain, "num_nodes", None)
        if n is not None and arr.size:
            if (arr < 0).any() or (arr >= n).any():
                raise ValueError(
                    f"spec.couple: at= index out of range for {owner!r} "
                    f"(min={int(arr.min())}, max={int(arr.max())}, "
                    f"num_nodes={n})"
                )
        return arr

    def _reject_ragged_family_couple(self, name: str) -> None:
        """Raise when a coupling endpoint names a ragged family.

        A ragged family's buckets carry different member sizes, so no single
        per-member transfer op applies across the population.

        Raises:
            NotImplementedError: when ``name`` is a ragged family handle.
        """
        handle = self._family_handles.get(name)
        if handle is not None and getattr(handle, "is_ragged", False):
            raise NotImplementedError(
                f"spec.couple: family {name!r} is ragged (static size "
                "buckets); coupling a ragged family is not supported — author "
                "one uniform family per size and couple each."
            )

    def _resolve_family_member(self, proj: "_FieldProjection"):
        """Resolve a ``_FieldProjection`` to its member Variable + sizes.

        Looks the family up by name in ``self._families`` (registered by
        ``spec.family(...)``), finds the member Variable by name, and returns
        ``(member_variable, member_state_size, surface_dof)``.  The surface DOF
        is the projected field's own last radial node, counted from the start
        of that field inside a member (``member_variable.size - 1``): the
        lowering reads each member's trace at ``offset_of((family, field)) +
        member * member_state_size + surface_dof``, and that block offset
        already includes the field's position in the member layout.  A family
        with a single member field therefore keeps its previous index
        (``member_state_size - 1``); with several member fields each field
        projects its own surface node rather than the member's last DOF.
        """
        self._reject_ragged_family_couple(proj.family_name)
        fam = self._families.get(proj.family_name)
        if fam is None:
            raise ValueError(
                f"spec.couple: unknown family {proj.family_name!r} "
                "(declare it with spec.family(...) first)"
            )
        # Ensure the registry node reflects the latest member declarations
        # before reading its member_variables / member_state_size.
        fam = self._ensure_family_synced(proj.family_name, fam)
        member_var = None
        for v in fam.member_variables:
            if getattr(v, "name", None) == proj.field_name:
                member_var = v
                break
        if member_var is None:
            raise ValueError(
                f"spec.couple: family {proj.family_name!r} has no member field {proj.field_name!r}"
            )
        n_member_dof = int(fam.member_state_size)
        member_surface_dof = int(member_var.size) - 1
        return member_var, n_member_dof, member_surface_dof

    def _ensure_family_synced(self, name: str, fam: Any) -> Any:
        """Re-materialise a lazily-dirty Family node before reading it.

        The family builder defers re-materialisation to a dirty flag; the smart-
        couple resolvers are a read boundary, so trigger the lazy sync here.  A
        family registered directly (e.g. IR-level test fixtures) carries no
        handle and is read as-is.
        """
        handle = self._family_handles.get(name)
        if handle is not None:
            return handle._ensure_synced()
        return fam

    def _family_gather_target(self, source: FieldSpec, onto: str):
        """Resolve the gather-onto-family coupling target.

        Registers the deterministic ``{source}_at_{onto}`` phantom as a
        family-owned record (``_family_phantoms``) for uniform auto-phantom
        naming, then returns the ``onto`` family's first member
        [Variable][] — by identity — as the coupling target.  Routing
        through that member Variable lands the coupling on the family GATHER
        branch in ``_resolve_family_couplings``, which keys the per-member
        gathered value under the macro source (not the phantom).
        """
        self._auto_phantom(source, onto)  # registers the family-owned phantom
        fam = self._families.get(onto)
        if fam is None:
            raise ValueError(f"spec.couple(onto={onto!r}): unknown domain or family")
        fam = self._ensure_family_synced(onto, fam)
        members = getattr(fam, "member_variables", ())
        if not members:
            raise ValueError(
                f"spec.couple(onto={onto!r}): family {onto!r} has no member "
                "fields yet — declare at least one with parts.field_decl(...) "
                "before gathering onto it."
            )
        return members[0]

    @staticmethod
    def _guard_scatter_collision(at_indices: tuple[int, ...]) -> None:
        """Raise when multiple members map to the same target node under ``reduce='none'``."""
        counts: dict[int, int] = {}
        for node in at_indices:
            counts[int(node)] = counts.get(int(node), 0) + 1
        for node, k in counts.items():
            if k > 1:
                raise ValueError(
                    f"spec.couple(reduce='none'): target node {node} receives "
                    f"{k} contributions; use reduce='sum' or 'mean'"
                )

    def _auto_phantom(self, source: FieldSpec, onto: str) -> FieldSpec:
        """Register the deterministic synthetic phantom target.

        Name is ``f"{source}_at_{onto}"``; registered as an algebraic
        [FieldSpec][] on the ``onto`` domain inheriting the source's
        non-dim ``scale``/``offset``.
        """
        return self._auto_phantom_named(f"{source.name}_at_{onto}", onto, source=source)

    def _auto_phantom_named(
        self, phantom_name: str, onto: str, source: Optional[FieldSpec] = None
    ) -> FieldSpec:
        """Register or look up a named auto-phantom FieldSpec (domain or family target).

        Idempotent — returns the existing phantom if the name is already
        registered.  Domain targets land in ``_fields``; family targets land in
        ``_family_phantoms``.
        """
        # ``onto`` may be a family name (gather onto a family); the phantom lives
        # on the domain when ``onto`` is a domain, and on the family's member
        # layout when ``onto`` is a family.  For a domain target the phantom is a
        # real FieldSpec in the component registry; for a family target it is a
        # family-owned member field tracked in ``_family_phantoms``.
        is_family = onto not in self._domains and onto in self._families
        existing = self._fields.get(phantom_name)
        if existing is not None:
            return existing
        if is_family:
            existing_fam = self._family_phantoms.get(phantom_name)
            if existing_fam is not None:
                return existing_fam[1]
        spec = FieldSpec(
            name=phantom_name,
            mass="algebraic",
            shape=None,
            bounds=None,
            scale=None if source is None else source.scale,
            offset=None if source is None else source.offset,
            residual_scale=None,
        )
        if onto in self._domains:
            # Domain target: the phantom is an ordinary algebraic component field.
            # Record it as a phantom (no user equation) so the build walk
            # synthesises its zero-rhs algebraic closure, and register the macro
            # source for phantom scale/offset propagation.
            self._fields[phantom_name] = spec
            self._field_domains[phantom_name] = onto
            self._phantom_fields[phantom_name] = source.name if source is not None else None
            if source is not None:
                self._phantom_sources[phantom_name] = source.name
        elif is_family:
            # Family target: record the phantom as a family-owned member field so
            # the auto-phantom name is on record without polluting the component
            # field registry (the gather coupling resolves its target to the
            # family member Variable directly).
            self._family_phantoms[phantom_name] = (onto, spec)
        return spec

    # ----------------------------------------------------- non-dimensionalization
    def time_scale(self, value: float) -> "ProblemSpec":
        """Set the time scale, so the model integrates on internal time `t / value`.

        Equation closures read internal time, and authored callbacks such as
        controls and constraint targets read dimensional time. A second call with a
        different value raises `ValueError`.

        Args:
            value: Time scale, nonzero.

        Returns:
            The builder, for chaining.
        """
        v = float(value)
        if v == 0.0:
            raise ValueError("spec.time_scale(0.0): time scale must be non-zero")
        if self._time_scale is not None and self._time_scale != v:
            raise ValueError(
                f"spec.time_scale: already set to {self._time_scale!r}; cannot reset to {v!r}"
            )
        self._time_scale = v
        return self

    def scales(
        self,
        *,
        state: Optional[dict[str, float]] = None,
        state_offset: Optional[dict[str, float]] = None,
        time: Optional[float] = None,
        integrand: Optional[dict[str, float]] = None,
    ) -> "ProblemSpec":
        """Set scales centrally by name, overriding the inline declarations.

        A central value that differs from an inline one wins, with a warning at
        build time.

        Args:
            state: State scale per field, global or member field name, or per `(component, field)` key.
            state_offset: State offset, keyed like `state`.
            time: Time scale.
            integrand: Target divisor per constraint name.

        Returns:
            The builder, for chaining.
        """
        if state:
            self._central_state.update({k: float(v) for k, v in state.items()})
        if state_offset:
            self._central_state_offset.update({k: float(v) for k, v in state_offset.items()})
        if time is not None:
            self._central_time = float(time)
        if integrand:
            self._central_integrand.update({k: float(v) for k, v in integrand.items()})
        return self

    def equation_scale(self, field: str, value: float) -> "ProblemSpec":
        """Set the divisor of a field's residual rows, overriding `field(residual_scale=...)`.

        Args:
            field: Field name.
            value: Divisor, nonzero.

        Returns:
            The builder, for chaining.
        """
        v = float(value)
        if v == 0.0:
            raise ValueError(
                f"spec.equation_scale({field!r}, 0.0): residual scale must be non-zero"
            )
        self._equation_scales[field] = v
        return self

    # --------------------------------------------------------------- build/validate
    def to_equation_set(self):
        """Validate the recipe and freeze it into an [`EquationSet`][diff_soup.problem.EquationSet].

        Returns:
            The equation set, ready for [`lower`][diff_soup.problem.lower].

        Raises:
            ValueError: If the recipe fails static validation, listing every problem found.
        """
        from .build import _build_equation_set

        return _build_equation_set(self)

    def build(self, *, layout=None, precision=None):
        """Validate the recipe and lower it into a solve-ready problem.

        Recipes equal in structure and values return the same problem.

        Args:
            layout: Device placement, a [`Layout`][diff_soup.Layout], or `None` for the default.
            precision: [`Precision`][diff_soup.Precision], or `None` for the default.

        Returns:
            The [`LoweredProblem`][diff_soup.problem.LoweredProblem].

        Raises:
            ValueError: If the recipe fails static validation, listing every problem found.
        """
        from diff_soup.problem._ir.lower import lower_declared

        declared_bounds = tuple(
            (ps, ps.bounds, ps.learnable) for ps in self._params.values() if ps.bounds is not None
        )
        handles = tuple(self._params.values()) + tuple(self._functions.values())
        # Bounds are values, not structure: they take part in the returned
        # problem's identity. The geometry names are the parameters that scale a
        # grid; an experiment checks the values it gives them.
        problem = lower_declared(
            self.to_equation_set(),
            layout=layout,
            precision=precision,
            bounds=declared_bounds,
            declarations=handles,
            geometry_scales=tuple(self._geometry.values()),
        )
        # Handle provenance is cache metadata rather than a field on this frozen
        # numerical model. It therefore cannot affect the pytree or the
        # structural signature. Recipes equal in structure and values share the
        # returned problem, so every compatible public namespace is retained.
        from diff_soup.problem._registry import register_parameter_handles

        register_parameter_handles(problem, self._parameter_namespace, handles)
        return problem

    def assemble_state(self, values, *, default=0.0):
        """Assemble the flat initial state from the declared initial conditions and overrides.

        An entry in `values` wins over the field's `init`, and `default` fills a
        field with neither. Globals are not included.

        Args:
            values: Mapping from field name, or `"<family>.<field>"`, to a scalar, an array, or a callable of the node coordinates.
            default: Value of every field with no entry and no `init`.

        Returns:
            The state vector, component fields first, then family members.
        """
        from .build import eval_field_init

        eqset = self.to_equation_set()
        pieces = []
        for comp in eqset.components:
            domain = comp.domain
            coords_fn = getattr(domain, "node_coordinates", None)
            coords = coords_fn() if callable(coords_fn) else None
            for var in comp.variables:
                fspec = self._fields.get(var.name)
                field_init = getattr(fspec, "init", None) if fspec else None
                if var.name in values:
                    # An explicit value always overrides the field-level init.
                    v = values[var.name]
                    if callable(v):
                        arr = jnp.asarray(v(coords))
                    elif jnp.ndim(v) == 0:
                        arr = jnp.broadcast_to(jnp.asarray(v), var.shape)
                    else:
                        arr = jnp.asarray(v)
                    flat = jnp.reshape(arr, (-1,))
                elif field_init is not None:
                    # The declared ``field(init=...)`` populates this y0
                    # block, winning over the scalar ``default``.  It takes one
                    # coordinate array per spatial dim (``*xs``).
                    flat = eval_field_init(field_init, coords, var.size, var.name)
                else:
                    arr = jnp.broadcast_to(jnp.asarray(default), var.shape)
                    flat = jnp.reshape(arr, (-1,))
                if flat.size != var.size:
                    raise ValueError(
                        f"field {var.name!r}: value materialised to size "
                        f"{flat.size}, but the field's flat size is {var.size} "
                        f"(shape {var.shape})"
                    )
                pieces.append(flat)
        # Family populations follow ALL components, in ``families`` order.
        # A Family carries its own member ``domain`` +
        # ``member_params``, so its ``(coords, m)`` closures are evaluable here.
        # Family-field IC keys are ``"<fam.name>.<field>"``.
        families = getattr(eqset, "families", ())
        if families:
            from diff_soup._lowering.family_assemble import assemble_family_pieces

            pieces.extend(assemble_family_pieces(families, values, default))
        return jnp.concatenate(pieces) if pieces else jnp.zeros(0)

    def validate(self, *, level: str = "static"):
        """Run the static validation pass, which `build` also runs.

        Args:
            level: Validation level, only `"static"`.

        Raises:
            ValueError: If the recipe fails validation, listing every problem found.
        """
        from .validate import _validate_spec

        return _validate_spec(self, level=level)


def problem(name: str) -> ProblemSpec:
    """Open a [`ProblemSpec`][diff_soup.problem.ProblemSpec] builder for a new problem.

    Args:
        name: Problem name.

    Returns:
        The empty builder.
    """
    return ProblemSpec(name=name)
